# screens/inference_screen.py - Detection view screen
import flet as ft
import cv2
import base64
import threading


class InferenceScreen:
    def __init__(self, app):
        self.app = app
        self.image_display = ft.Image(width=350, height=350, fit=ft.ImageFit.CONTAIN)
        self.stat_texts = {}
        self.confidence_slider = None
        self.iou_slider = None
        self.detect_button = None
        self.view_results_button = None
        self.is_processing = False

    def build(self) -> ft.View:
        self.stat_texts = {
            "total": ft.Text("Total: 0"),
            "avg_conf": ft.Text("Avg Confidence: 0.00"),
            "pet": ft.Text("PET: 0"), "hdpe": ft.Text("HDPE: 0"), "pvc": ft.Text("PVC: 0"),
            "ldpe": ft.Text("LDPE: 0"), "pp": ft.Text("PP: 0"), "ps": ft.Text("PS: 0"),
        }

        self.confidence_slider = ft.Slider(min=0.01, max=1.0, value=self.app.settings.get("conf", 0.25), label="Conf: {value}", on_change=self.on_conf_change)
        self.iou_slider = ft.Slider(min=0.1, max=1.0, value=self.app.settings.get("iou", 0.45), label="IoU: {value}", on_change=self.on_iou_change)
        self.detect_button = ft.ElevatedButton("Detect", icon=ft.Icons.PLAY_ARROW, on_click=lambda _: self.run_detection(), bgcolor=ft.Colors.PRIMARY, color=ft.Colors.WHITE)
        self.view_results_button = ft.ElevatedButton("View Results", icon=ft.Icons.VISIBILITY, on_click=lambda _: self.app.go("/result"), visible=False)

        if self.app.current_file:
            self.load_image(self.app.current_file)

        return ft.View(
            "/inference",
            [
                ft.AppBar(title=ft.Text("Inference"), leading=ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=lambda _: self.app.go("/gallery")), bgcolor=ft.Colors.SURFACE),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Container(content=self.image_display, border=ft.Border.all(1, ft.Colors.OUTLINE), border_radius=10),
                            ft.Container(
                                content=ft.Column([ft.Text("Confidence"), self.confidence_slider, ft.Text("IoU"), self.iou_slider], spacing=5),
                                padding=10,
                            ),
                            ft.Row([self.detect_button, self.view_results_button], spacing=10, alignment=ft.MainAxisAlignment.CENTER),
                            ft.Card(content=ft.Container(content=ft.Column([ft.Text("Results", size=16, weight=ft.FontWeight.BOLD), self.stat_texts["total"], self.stat_texts["avg_conf"], ft.Divider(), self.stat_texts["pet"], self.stat_texts["hdpe"], self.stat_texts["pvc"], self.stat_texts["ldpe"], self.stat_texts["pp"], self.stat_texts["ps"]], spacing=5), padding=10)),
                        ],
                        spacing=10,
                    ),
                    padding=10,
                    expand=True,
                ),
                self.app.nav_bar.build() if hasattr(self.app, 'nav_bar') else ft.Container(),
            ],
        )

    def load_image(self, file_path):
        try:
            if self.app.file_handler.is_video(file_path):
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    _, buf = cv2.imencode('.jpg', frame)
                    self.image_display.src_base64 = base64.b64encode(buf).decode('utf-8')
            else:
                frame = cv2.imread(file_path)
                if frame is not None:
                    _, buf = cv2.imencode('.jpg', frame)
                    self.image_display.src_base64 = base64.b64encode(buf).decode('utf-8')
        except Exception as e:
            print(f"[Inference] Error: {e}")

    def on_conf_change(self, e):
        from core.settings_manager import save_settings
        self.app.settings["conf"] = float(e.control.value)
        save_settings(self.app.settings)

    def on_iou_change(self, e):
        from core.settings_manager import save_settings
        self.app.settings["iou"] = float(e.control.value)
        save_settings(self.app.settings)

    def run_detection(self):
        if self.app.current_file is None:
            self.app.show_snackbar("No file selected")
            return
        if self.app.current_engine is None:
            self.app.show_snackbar("No model loaded")
            return
        if self.is_processing:
            return
        self.is_processing = True
        self.detect_button.disabled = True
        self.app.page.update()
        threading.Thread(target=self._detection_thread, daemon=True).start()

    def _detection_thread(self):
        try:
            conf = self.app.settings.get("conf", 0.25)
            iou = self.app.settings.get("iou", 0.45)
            if self.app.file_handler.is_video(self.app.current_file):
                cap = cv2.VideoCapture(self.app.current_file)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise RuntimeError("Failed to read video")
            else:
                frame = cv2.imread(self.app.current_file)
                if frame is None:
                    raise RuntimeError("Failed to read image")
            results = self.app.current_engine.detect(frame, conf_thresh=conf, iou_thresh=iou)
            from core.analytics import compute_stats
            from core.vision import draw_boxes
            stats = compute_stats(results)
            annotated = draw_boxes(frame, results)
            self.app.current_results = results
            self._update_ui(stats, annotated)
        except Exception as e:
            print(f"[Inference] Error: {e}")
            self.app.show_snackbar(f"Detection failed: {e}")
        finally:
            self.is_processing = False
            self.detect_button.disabled = False
            self.app.page.update()

    def _update_ui(self, stats, annotated):
        try:
            _, buf = cv2.imencode('.jpg', annotated)
            self.image_display.src_base64 = base64.b64encode(buf).decode('utf-8')
            total = stats.get("total", 0)
            self.stat_texts["total"].value = f"Total: {total}"
            self.stat_texts["avg_conf"].value = f"Avg Confidence: {stats.get('avg_conf', 0):.2f}"
            for cls in ["pet", "hdpe", "pvc", "ldpe", "pp", "ps"]:
                count = stats.get("per_class", {}).get(cls.upper(), {}).get("count", 0)
                self.stat_texts[cls].value = f"{cls.upper()}: {count}"
            self.view_results_button.visible = True
            self.app.page.update()
            self.app.show_snackbar(f"Found {total} particles")
        except Exception as e:
            print(f"[Inference] UI error: {e}")
