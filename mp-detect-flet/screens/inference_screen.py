# screens/inference_screen.py
import flet as ft
import cv2
import base64
import threading
from components.layout import ResponsiveLayout, LeftSidebar, RightInspector


class InferenceScreen:
    def __init__(self, app):
        self.app = app
        self.image_display = ft.Image(src="", width=400, height=400, fit="contain")
        self.is_processing = False
        self.detect_button = None
        self.view_results_button = None
        self.hud_fps = ft.Text("FPS: --", color=ft.Colors.CYAN, size=12)
        self.hud_latency = ft.Text("Latency: --ms", color=ft.Colors.CYAN, size=12)
        self.hud_backend = ft.Text("", color=ft.Colors.GREEN, size=12)
        self.left_sidebar = LeftSidebar(app)
        self.right_inspector = RightInspector(app)
        self._has_image = False
        self._cap = None
        self._inf_active = False
        self._inf_thread_busy = False

    def build(self) -> ft.View:
        self.detect_button = ft.ElevatedButton("Detect", icon=ft.Icons.PLAY_ARROW, on_click=lambda _: self.toggle_detection(), bgcolor=ft.Colors.PRIMARY, color=ft.Colors.WHITE, width=120)
        self.view_results_button = ft.ElevatedButton("View Results", icon=ft.Icons.VISIBILITY, on_click=lambda _: self.app.go("/result"), visible=False)

        if self.app.current_file:
            self._load_image(self.app.current_file)

        if self.app.current_engine:
            fmt = "TFLite" if hasattr(self.app.current_engine, "interpreter") else "ONNX"
            self.hud_backend.value = f"[{fmt}]"

        center_content = ft.Column(
            [
                ft.Container(
                    content=ft.Stack(
                        [
                            self.image_display,
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Icon(ft.Icons.MICROSCOPE, size=48, color=ft.Colors.with_opacity(0.3, ft.Colors.ON_SURFACE)),
                                        ft.Text("No Micrograph Loaded", size=16, color=ft.Colors.with_opacity(0.5, ft.Colors.ON_SURFACE)),
                                        ft.Text("Click 'Open File' or press Ctrl+O", size=12, color=ft.Colors.with_opacity(0.3, ft.Colors.ON_SURFACE)),
                                    ],
                                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                    spacing=8,
                                ),
                                alignment=ft.Alignment.CENTER,
                                visible=not self._has_image,
                            ),
                        ],
                        expand=True,
                    ),
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.2, ft.Colors.OUTLINE)),
                    border_radius=8,
                    expand=True,
                ),
                ft.Container(
                    content=ft.Row([self.hud_fps, self.hud_latency, self.hud_backend], spacing=16),
                    padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                ),
                ft.Container(
                    content=ft.Row(
                        [
                            self.detect_button,
                            ft.ElevatedButton("Snapshot", icon=ft.Icons.CAMERA, on_click=lambda _: self.app.show_snackbar("Snapshot saved")),
                            ft.ElevatedButton("Record", icon=ft.Icons.FIBER_MANUAL_RECORD, on_click=lambda _: self.app.show_snackbar("Recording toggled")),
                            self.view_results_button,
                        ],
                        spacing=8,
                    ),
                    padding=8,
                    bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
                    border_radius=6,
                ),
            ],
            spacing=0,
            expand=True,
        )

        layout = ResponsiveLayout(self.app)
        content = layout.build(
            left_sidebar=self.left_sidebar.build(),
            center_content=center_content,
            right_inspector=self.right_inspector.build(),
        )

        return ft.View(
            "/inference",
            [
                ft.AppBar(title=ft.Text("Inference"), leading=ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=lambda _: self.app.go("/gallery")), bgcolor=ft.Colors.SURFACE),
                ft.Container(content=content, expand=True),
                self.app.nav_bar.build() if hasattr(self.app, 'nav_bar') else ft.Container(),
            ],
        )

    def _load_image(self, file_path):
        try:
            if self.app.file_handler.is_video(file_path):
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    _, buf = cv2.imencode('.jpg', frame)
                    b64 = base64.b64encode(buf).decode()
                    self.image_display.src = f"data:image/jpeg;base64,{b64}"
                    self._has_image = True
            else:
                frame = cv2.imread(file_path)
                if frame is not None:
                    _, buf = cv2.imencode('.jpg', frame)
                    b64 = base64.b64encode(buf).decode()
                    self.image_display.src = f"data:image/jpeg;base64,{b64}"
                    self._has_image = True
        except Exception as e:
            print(f"[Inference] Error: {e}")

    def pick_file(self):
        file_picker = ft.FilePicker()
        self.app.page.overlay.append(file_picker)
        self.app.page.update()
        result = file_picker.pick_files(
            dialog_title="Select Micrograph",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "mp4", "avi", "mov", "mkv"],
        )
        self.app.page.overlay.remove(file_picker)
        self.app.page.update()
        if result and len(result) > 0:
            self.app.current_file = result[0].path
            self._load_image(result[0].path)
            self.app.show_snackbar(f"Loaded: {result[0].path.split('/')[-1]}")

    def toggle_detection(self):
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
            _, buf = cv2.imencode('.jpg', annotated)
            b64 = base64.b64encode(buf).decode()
            self.image_display.src = f"data:image/jpeg;base64,{b64}"
            self.right_inspector.update_stats(stats)
            self.view_results_button.visible = True
            self.detect_button.disabled = False
            self.is_processing = False
            self.app.page.update()
            self.app.show_snackbar(f"Found {stats.get('total', 0)} particles")
        except Exception as e:
            print(f"[Inference] Error: {e}")
            self.detect_button.disabled = False
            self.is_processing = False
            self.app.page.update()
            self.app.show_snackbar(f"Detection failed: {e}")
