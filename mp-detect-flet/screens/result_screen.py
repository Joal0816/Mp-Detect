# screens/result_screen.py
import flet as ft
import cv2
import base64
import os
import threading


class ResultScreen:
    def __init__(self, app):
        self.app = app
        self.video_display = ft.Image(src="", width=350, height=300, fit="contain")

    def build(self) -> ft.View:
        self.load_results()
        return ft.View(
            "/result",
            [
                ft.AppBar(title=ft.Text("Results"), leading=ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=lambda _: self.app.go("/inference")), bgcolor=ft.Colors.SURFACE),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Container(content=self.video_display, border=ft.Border.all(1, ft.Colors.OUTLINE), border_radius=8),
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Text("Export", size=14, weight=ft.FontWeight.BOLD),
                                        ft.Row(
                                            [
                                                ft.ElevatedButton("CSV", icon=ft.Icons.TABLE_CHART, on_click=lambda _: self._export("csv")),
                                                ft.ElevatedButton("JSON", icon=ft.Icons.CODE, on_click=lambda _: self._export("json")),
                                                ft.ElevatedButton("Image", icon=ft.Icons.IMAGE, on_click=lambda _: self._export("image")),
                                                ft.ElevatedButton("Video", icon=ft.Icons.VIDEO_FILE, on_click=lambda _: self._export("video")),
                                            ],
                                            spacing=8,
                                        ),
                                    ],
                                    spacing=8,
                                ),
                                padding=12,
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
                    b64 = base64.b64encode(buf).decode()
                    self.video_display.src = f"data:image/jpeg;base64,{b64}"
            else:
                frame = cv2.imread(self.app.current_file)
                if frame is not None:
                    annotated = draw_boxes(frame, self.app.current_results)
                    _, buf = cv2.imencode('.jpg', annotated)
                    b64 = base64.b64encode(buf).decode()
                    self.video_display.src = f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            print(f"[Result] Error: {e}")

    def _export(self, fmt):
        try:
            if fmt == "csv":
                from core.export import export_csv
                path = self.app.file_handler.get_export_csv_path()
                export_csv(self.app.current_results, path, self.app.settings)
                self.app.show_snackbar("CSV saved")
            elif fmt == "json":
                from core.export import export_json_report
                path = self.app.file_handler.get_export_report_path()
                export_json_report(self.app.current_results, path, self.app.settings, self.app.current_file, "Flet Backend")
                self.app.show_snackbar("Report saved")
            elif fmt == "image":
                self.app.show_snackbar("Image export")
            elif fmt == "video":
                self.app.show_snackbar("Video export")
        except Exception as e:
            self.app.show_snackbar(f"Export failed: {e}")
