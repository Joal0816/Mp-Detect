# screens/result_screen.py - Results view screen
import flet as ft
import cv2
import base64
import os
import threading


class ResultScreen:
    def __init__(self, app):
        self.app = app
        self.video_display = ft.Image(width=350, height=300, fit=ft.ImageFit.CONTAIN)
        self.stat_texts = {}

    def build(self) -> ft.View:
        self.stat_texts = {
            "total": ft.Text("Total: 0"), "avg_conf": ft.Text("Avg Confidence: 0.00"),
            "pet": ft.Text("PET: 0"), "hdpe": ft.Text("HDPE: 0"), "pvc": ft.Text("PVC: 0"),
            "ldpe": ft.Text("LDPE: 0"), "pp": ft.Text("PP: 0"), "ps": ft.Text("PS: 0"),
        }
        self.load_results()
        return ft.View(
            "/result",
            [
                ft.AppBar(title=ft.Text("Results"), leading=ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=lambda _: self.app.go("/inference")), bgcolor=ft.Colors.SURFACE),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Container(content=self.video_display, border=ft.Border.all(1, ft.Colors.OUTLINE), border_radius=10),
                            ft.Card(content=ft.Container(content=ft.Column([ft.Text("Summary", size=16, weight=ft.FontWeight.BOLD), self.stat_texts["total"], self.stat_texts["avg_conf"], ft.Divider(), self.stat_texts["pet"], self.stat_texts["hdpe"], self.stat_texts["pvc"], self.stat_texts["ldpe"], self.stat_texts["pp"], self.stat_texts["ps"]], spacing=5), padding=10)),
                            ft.Container(
                                content=ft.Column([
                                    ft.Text("Export", size=14, weight=ft.FontWeight.BOLD),
                                    ft.Row([ft.ElevatedButton("CSV", icon=ft.Icons.FILE_DOWNLOAD, on_click=lambda _: self.export_csv()), ft.ElevatedButton("Report", icon=ft.Icons.DESCRIPTION, on_click=lambda _: self.export_json())], spacing=10),
                                    ft.Row([ft.ElevatedButton("Image", icon=ft.Icons.IMAGE, on_click=lambda _: self.export_image()), ft.ElevatedButton("Video", icon=ft.Icons.VIDEO_FILE, on_click=lambda _: self.export_video())], spacing=10),
                                ], spacing=10),
                                padding=10,
                            ),
                        ],
                        spacing=10,
                    ),
                    padding=10,
                    expand=True,
                ),
                self.app.nav_bar.build() if hasattr(self.app, 'nav_bar') else ft.Container(),
            ],
        )

    def load_results(self):
        from core.vision import draw_boxes
        if self.app.current_file is None:
            return
        try:
            if self.app.file_handler.is_video(self.app.current_file):
                cap = cv2.VideoCapture(self.app.current_file)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    annotated = draw_boxes(frame, self.app.current_results)
                    _, buf = cv2.imencode('.jpg', annotated)
                    self.video_display.src_base64 = base64.b64encode(buf).decode('utf-8')
            else:
                frame = cv2.imread(self.app.current_file)
                if frame is not None:
                    annotated = draw_boxes(frame, self.app.current_results)
                    _, buf = cv2.imencode('.jpg', annotated)
                    self.video_display.src_base64 = base64.b64encode(buf).decode('utf-8')
        except Exception as e:
            print(f"[Result] Error: {e}")
        self.update_stats()

    def update_stats(self):
        results = self.app.current_results
        if not results:
            return
        total = len(results)
        self.stat_texts["total"].value = f"Total: {total}"
        self.stat_texts["avg_conf"].value = f"Avg Confidence: {sum(c for _, _, c in results) / total:.2f}" if total else "Avg Confidence: 0.00"
        per_class = {"PET": 0, "HDPE": 0, "PVC": 0, "LDPE": 0, "PP": 0, "PS": 0}
        for _, label, _ in results:
            if label in per_class:
                per_class[label] += 1
        for cls in ["pet", "hdpe", "pvc", "ldpe", "pp", "ps"]:
            self.stat_texts[cls].value = f"{cls.upper()}: {per_class[cls.upper()]}"
        self.app.page.update()

    def export_csv(self):
        try:
            from core.export import export_csv
            path = self.app.file_handler.get_export_csv_path()
            export_csv(self.app.current_results, path, self.app.settings)
            self.app.show_snackbar(f"CSV saved: {os.path.basename(path)}")
        except Exception as e:
            self.app.show_snackbar(f"Export failed: {e}")

    def export_json(self):
        try:
            from core.export import export_json_report
            path = self.app.file_handler.get_export_report_path()
            export_json_report(self.app.current_results, path, self.app.settings, self.app.current_file, "Flet Backend")
            self.app.show_snackbar(f"Report saved: {os.path.basename(path)}")
        except Exception as e:
            self.app.show_snackbar(f"Export failed: {e}")

    def export_image(self):
        try:
            from core.export import export_annotated_image
            if self.app.current_file is None:
                return
            frame = cv2.imread(self.app.current_file) if not self.app.file_handler.is_video(self.app.current_file) else None
            if frame is None and self.app.file_handler.is_video(self.app.current_file):
                cap = cv2.VideoCapture(self.app.current_file)
                ret, frame = cap.read()
                cap.release()
            if frame is None:
                return
            path = self.app.file_handler.get_annotated_image_path()
            export_annotated_image(frame, self.app.current_results, path)
            self.app.show_snackbar(f"Image saved: {os.path.basename(path)}")
        except Exception as e:
            self.app.show_snackbar(f"Export failed: {e}")

    def export_video(self):
        try:
            from core.export import export_annotated_video
            if self.app.current_file is None or not self.app.file_handler.is_video(self.app.current_file):
                return
            path = self.app.file_handler.get_annotated_video_path()
            threading.Thread(target=lambda: export_annotated_video(self.app.current_file, self.app.current_results, path, self.app.settings), daemon=True).start()
            self.app.show_snackbar("Exporting video...")
        except Exception as e:
            self.app.show_snackbar(f"Export failed: {e}")
