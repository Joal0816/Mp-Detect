# screens/upload_screen.py
import flet as ft
import cv2
import base64
import os
import threading


class UploadScreen:
    def __init__(self, app):
        self.app = app
        self.selected_file = app.current_file
        self.image_display = ft.Image(src="", width=350, height=250, fit="contain")
        self.progress_bar = ft.ProgressBar(visible=False, color=ft.Colors.CYAN)
        self.progress_text = ft.Text("", size=12, color=ft.Colors.CYAN)
        self.analyze_button = None
        self.has_image = False

    def build(self) -> ft.View:
        self.analyze_button = ft.ElevatedButton(
            "Select File",
            icon=ft.Icons.FOLDER_OPEN,
            on_click=lambda _: self.handle_action(),
            bgcolor=ft.Colors.PRIMARY,
            color=ft.Colors.WHITE,
            expand=True,
        )

        return ft.View(
            "/upload",
            [
                ft.AppBar(title=ft.Text("Upload & Detect"), bgcolor=ft.Colors.SURFACE),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Container(
                                content=ft.Stack(
                                    [
                                        self.image_display,
                                        ft.Container(
                                            content=ft.Column(
                                                [
                                                    ft.Icon(ft.Icons.IMAGE_SEARCH, size=48, color=ft.Colors.with_opacity(0.3, ft.Colors.ON_SURFACE)),
                                                    ft.Text("No Image/Video Selected", size=14, color=ft.Colors.with_opacity(0.5, ft.Colors.ON_SURFACE)),
                                                ],
                                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                                spacing=8,
                                            ),
                                            alignment=ft.Alignment.CENTER,
                                            visible=not self.has_image,
                                        ),
                                    ],
                                    expand=True,
                                ),
                                height=250,
                                border=ft.Border.all(1, ft.Colors.with_opacity(0.2, ft.Colors.OUTLINE)),
                                border_radius=8,
                            ),
                            ft.Container(content=ft.Column([self.progress_text, self.progress_bar], spacing=4), padding=ft.Padding.symmetric(horizontal=16), visible=self.progress_bar.visible),
                            ft.Container(
                                content=ft.Row([self.analyze_button], spacing=8),
                                padding=ft.Padding.symmetric(horizontal=16, vertical=8),
                            ),
                            ft.Container(
                                content=ft.Row(
                                    [
                                        ft.ElevatedButton("CSV", icon=ft.Icons.TABLE_CHART, on_click=lambda _: self._export("csv")),
                                        ft.ElevatedButton("JSON", icon=ft.Icons.CODE, on_click=lambda _: self._export("json")),
                                        ft.ElevatedButton("Image", icon=ft.Icons.IMAGE, on_click=lambda _: self._export("image")),
                                        ft.ElevatedButton("Video", icon=ft.Icons.VIDEO_FILE, on_click=lambda _: self._export("video")),
                                    ],
                                    spacing=8,
                                ),
                                padding=ft.Padding.symmetric(horizontal=16),
                                visible=self.app.current_results is not None and len(self.app.current_results) > 0,
                            ),
                        ],
                        spacing=0,
                    ),
                    expand=True,
                ),
                self.app.nav_bar.build() if hasattr(self.app, 'nav_bar') else ft.Container(),
            ],
        )

    def handle_action(self):
        if not self.selected_file:
            self.pick_file()
        elif not self.app.current_results:
            self.run_detection()
        else:
            self.reset()

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
            self._load_file(result[0].path)

    def _load_file(self, path):
        self.selected_file = path
        self.app.current_file = path
        self.app.current_results = []
        try:
            if self.app.file_handler.is_video(path):
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    _, buf = cv2.imencode('.jpg', frame)
                    b64 = base64.b64encode(buf).decode()
                    self.image_display.src = f"data:image/jpeg;base64,{b64}"
                    self.has_image = True
            else:
                frame = cv2.imread(path)
                if frame is not None:
                    _, buf = cv2.imencode('.jpg', frame)
                    b64 = base64.b64encode(buf).decode()
                    self.image_display.src = f"data:image/jpeg;base64,{b64}"
                    self.has_image = True
        except Exception as e:
            print(f"[Upload] Error: {e}")
        self.analyze_button.text = "Analyze"
        self.analyze_button.icon = ft.Icons.PLAY_ARROW
        self.app.page.update()
        self.app.show_snackbar(f"Loaded: {os.path.basename(path)}")

    def run_detection(self):
        if self.selected_file is None or self.app.current_engine is None:
            self.app.show_snackbar("No file or model selected")
            return
        self.progress_bar.visible = True
        self.progress_text.value = "Processing..."
        self.analyze_button.disabled = True
        self.app.page.update()
        threading.Thread(target=self._detect_thread, daemon=True).start()

    def _detect_thread(self):
        try:
            conf = self.app.settings.get("conf", 0.25)
            iou = self.app.settings.get("iou", 0.45)
            if self.app.file_handler.is_video(self.selected_file):
                cap = cv2.VideoCapture(self.selected_file)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise RuntimeError("Failed to read video")
            else:
                frame = cv2.imread(self.selected_file)
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
            total = stats.get("total", 0)
            self.progress_bar.visible = False
            self.progress_text.value = f"Complete - {total} particles"
            self.analyze_button.text = "New File"
            self.analyze_button.icon = ft.Icons.ADD
            self.analyze_button.disabled = False
            self.app.page.update()
            self.app.show_snackbar(f"Found {total} particles")
        except Exception as e:
            self.progress_bar.visible = False
            self.progress_text.value = "Error"
            self.analyze_button.disabled = False
            self.app.page.update()
            self.app.show_snackbar(f"Detection failed: {e}")

    def reset(self):
        self.selected_file = None
        self.app.current_file = None
        self.app.current_results = []
        self.image_display.src = ""
        self.has_image = False
        self.progress_bar.visible = False
        self.progress_text.value = ""
        self.analyze_button.text = "Select File"
        self.analyze_button.icon = ft.Icons.FOLDER_OPEN
        self.app.page.update()

    def _export(self, fmt):
        try:
            if fmt == "csv":
                from core.export import export_csv
                path = self.app.file_handler.get_export_csv_path()
                export_csv(self.app.current_results, path, self.app.settings)
                self.app.show_snackbar(f"CSV saved")
            elif fmt == "json":
                from core.export import export_json_report
                path = self.app.file_handler.get_export_report_path()
                export_json_report(self.app.current_results, path, self.app.settings, self.app.current_file, "Flet Backend")
                self.app.show_snackbar(f"Report saved")
            elif fmt == "image":
                self.app.show_snackbar("Image export")
            elif fmt == "video":
                self.app.show_snackbar("Video export")
        except Exception as e:
            self.app.show_snackbar(f"Export failed: {e}")
