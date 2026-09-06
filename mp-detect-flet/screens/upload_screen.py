# screens/upload_screen.py
import flet as ft
import cv2
import base64
import os
import threading
from components.theme import Colors, GRADIENT_CYAN, card, glass_container, metric_card


class UploadScreen:
    def __init__(self, app):
        self.app = app
        self.selected_file = app.current_file
        PLACEHOLDER = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQq0AAAAAElFTkSuQmCC"
        self.image_display = ft.Image(
            src=PLACEHOLDER, width=400, height=280,
            fit=ft.BoxFit.CONTAIN, visible=False,
            border_radius=8,
        )
        self.progress_bar = ft.ProgressBar(
            visible=False, color=Colors.ACCENT_CYAN,
            bgcolor=Colors.BG_CARD, border_radius=4,
        )
        self.progress_text = ft.Text("", size=12, color=Colors.ACCENT_CYAN)
        self.analyze_button = None
        self.has_image = False

    def build_content(self) -> ft.Column:
        self.analyze_button = ft.Button(
            "Analyze", icon=ft.Icons.PLAY_ARROW,
            on_click=self.handle_action,
            bgcolor=Colors.ACCENT_CYAN, color=Colors.BG_PRIMARY,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
            width=200, height=44,
        )

        return ft.Column(
            [
                # Header
                ft.Container(
                    content=ft.Row([
                        ft.Text("Upload", size=24, weight=ft.FontWeight.BOLD, color=Colors.TEXT_PRIMARY),
                        ft.Text("Micrograph Analysis", size=14, color=Colors.TEXT_SECONDARY),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=ft.Padding.symmetric(horizontal=24, vertical=16),
                ),

                # Main content area
                ft.Container(
                    content=ft.Column(
                        [
                            # Drop zone / Image preview
                            self._build_upload_area(),

                            # Progress
                            ft.Container(
                                content=ft.Column([
                                    self.progress_text,
                                    self.progress_bar,
                                ], spacing=4),
                                padding=ft.Padding.symmetric(horizontal=24),
                                visible=self.progress_bar.visible,
                            ),

                            # Action buttons
                            ft.Container(
                                content=ft.Row(
                                    [self.analyze_button],
                                    alignment=ft.MainAxisAlignment.CENTER,
                                ),
                                padding=ft.Padding.symmetric(vertical=16),
                            ),

                            # Export buttons
                            self._build_export_row(),
                        ],
                        spacing=0,
                        expand=True,
                    ),
                    expand=True,
                ),
            ],
            spacing=0,
            expand=True,
        )

    def _build_upload_area(self):
        """Build the upload/drop zone with gradient border."""
        # Gradient border effect
        return ft.Container(
            content=ft.Stack([
                # Image display
                self.image_display,
                # Placeholder overlay
                ft.Container(
                    content=ft.Column([
                        ft.Container(
                            content=ft.Icon(ft.Icons.CLOUD_UPLOAD, size=40, color=Colors.ACCENT_CYAN),
                            bgcolor=ft.Colors.with_opacity(0.1, Colors.ACCENT_CYAN),
                            border_radius=50,
                            padding=16,
                        ),
                        ft.Text("Drop image or video here", size=16, color=Colors.TEXT_PRIMARY, weight=ft.FontWeight.W_500),
                        ft.Text("or click to browse files", size=12, color=Colors.TEXT_SECONDARY),
                        ft.Container(
                            content=ft.Row([
                                ft.Icon(ft.Icons.IMAGE, size=14, color=Colors.TEXT_MUTED),
                                ft.Text("PNG, JPG, TIF, BMP, MP4, AVI", size=11, color=Colors.TEXT_MUTED),
                            ], spacing=4, alignment=ft.MainAxisAlignment.CENTER),
                            margin=ft.Margin.only(top=8),
                        ),
                    ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=8,
                    ),
                    alignment=ft.Alignment.CENTER,
                    visible=not self.has_image,
                ),
            ], expand=True),
            height=280,
            border=ft.Border.all(2, Colors.GLASS_BORDER_ACTIVE),
            border_radius=12,
            bgcolor=Colors.BG_CARD,
            margin=ft.Margin.symmetric(horizontal=24, vertical=8),
            on_click=lambda _: self.handle_action(),
            animate=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
        )

    def _build_export_row(self):
        """Build export buttons row."""
        visible = self.app.current_results is not None and len(self.app.current_results) > 0
        return ft.Container(
            content=ft.Row(
                [
                    self._export_button("CSV", ft.Icons.TABLE_CHART, "csv"),
                    self._export_button("JSON", ft.Icons.CODE, "json"),
                    self._export_button("Image", ft.Icons.IMAGE, "image"),
                    self._export_button("Video", ft.Icons.VIDEO_FILE, "video"),
                ],
                spacing=8,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            padding=ft.Padding.symmetric(horizontal=24, vertical=8),
            visible=visible,
        )

    def _export_button(self, label, icon, fmt):
        return ft.Button(
            label, icon=icon,
            on_click=lambda _, f=fmt: self._export(f),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                side=ft.BorderSide(1, Colors.GLASS_BORDER),
                bgcolor=Colors.GLASS_BG,
                color=Colors.TEXT_SECONDARY,
            ),
            height=36,
        )

    async def handle_action(self, e=None):
        if not self.selected_file:
            await self.pick_file()
        elif not self.app.current_results:
            self.run_detection()
        else:
            self.reset()

    async def pick_file(self):
        files = await self.app.file_picker.pick_files(
            dialog_title="Select Micrograph",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["png", "jpg", "jpeg", "tif", "tiff", "bmp",
                               "mp4", "avi", "mov", "mkv"],
        )
        if files and len(files) > 0:
            self._load_file(files[0].path)

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
                    self.image_display.visible = True
                    self.has_image = True
            else:
                frame = cv2.imread(path)
                if frame is not None:
                    _, buf = cv2.imencode('.jpg', frame)
                    b64 = base64.b64encode(buf).decode()
                    self.image_display.src = f"data:image/jpeg;base64,{b64}"
                    self.image_display.visible = True
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
            self.app.current_annotated = annotated
            self.app.current_stats = stats
            self.app.current_frame = frame
            _, buf = cv2.imencode('.jpg', annotated)
            b64 = base64.b64encode(buf).decode()
            self.image_display.src = f"data:image/jpeg;base64,{b64}"
            self.image_display.visible = True
            total = stats.get("total", 0)
            self.progress_bar.visible = False
            self.progress_text.value = f"Found {total} particles"
            self.analyze_button.text = "New File"
            self.analyze_button.icon = ft.Icons.ADD
            self.analyze_button.disabled = False
            self.app.page.update()
            self.app.show_snackbar(f"Found {total} particles")
            # Navigate to results
            self.app.go("result")
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
        self.image_display.visible = False
        self.has_image = False
        self.progress_bar.visible = False
        self.progress_text.value = ""
        self.analyze_button.text = "Analyze"
        self.analyze_button.icon = ft.Icons.PLAY_ARROW
        self.app.page.update()

    def _export(self, fmt):
        try:
            if fmt == "csv":
                from core.export import export_csv
                path = self.app.file_handler.get_export_csv_path()
                export_csv(self.app.current_results, path, self.app.settings)
                self.app.show_snackbar(f"CSV saved: {os.path.basename(path)}")
            elif fmt == "json":
                from core.export import export_json_report
                path = self.app.file_handler.get_export_report_path()
                export_json_report(self.app.current_results, path, self.app.settings,
                                  self.app.current_file, "Flet Backend")
                self.app.show_snackbar(f"Report saved: {os.path.basename(path)}")
            elif fmt == "image":
                from core.export import export_annotated_image
                if self.app.current_frame is not None and self.app.current_results:
                    path = self.app.file_handler.get_annotated_image_path()
                    export_annotated_image(self.app.current_frame, self.app.current_results, path)
                    self.app.show_snackbar(f"Image saved: {os.path.basename(path)}")
                else:
                    self.app.show_snackbar("No image to export")
            elif fmt == "video":
                from core.export import export_annotated_video
                if self.app.current_file and self.app.current_results:
                    if self.app.file_handler.is_video(self.app.current_file):
                        path = self.app.file_handler.get_annotated_video_path()
                        export_annotated_video(self.app.current_file, self.app.current_results,
                                              path, self.app.settings)
                        self.app.show_snackbar(f"Video saved: {os.path.basename(path)}")
                    else:
                        self.app.show_snackbar("Source is not a video")
                else:
                    self.app.show_snackbar("No video to export")
        except Exception as e:
            self.app.show_snackbar(f"Export failed: {e}")
