# screens/gallery_screen.py
import flet as ft
import cv2
import base64
import os
from components.theme import Colors


class GalleryScreen:
    def __init__(self, app):
        self.app = app
        self.files = []
        self.filter = "all"

    def build_content(self) -> ft.Column:
        self.load_files()
        return ft.Column(
            [
                # Header
                ft.Container(
                    content=ft.Row([
                        ft.Text("Gallery", size=24, weight=ft.FontWeight.BOLD, color=Colors.TEXT_PRIMARY),
                        ft.Text(f"{len(self.files)} items", size=14, color=Colors.TEXT_SECONDARY),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=ft.Padding.symmetric(horizontal=24, vertical=16),
                ),

                # Filter chips
                ft.Container(
                    content=ft.Row([
                        self._filter_chip("All", "all"),
                        self._filter_chip("Images", "images"),
                        self._filter_chip("Videos", "videos"),
                    ], spacing=8),
                    padding=ft.Padding.symmetric(horizontal=24, vertical=8),
                ),

                # Grid or empty state
                self._build_grid() if self.files else self._build_empty_state(),
            ],
            spacing=0,
            expand=True,
        )

    def _filter_chip(self, label, value):
        is_selected = self.filter == value
        return ft.Container(
            content=ft.Text(label, size=12, color=Colors.ACCENT_CYAN if is_selected else Colors.TEXT_SECONDARY),
            bgcolor=ft.Colors.with_opacity(0.15, Colors.ACCENT_CYAN) if is_selected else Colors.GLASS_BG,
            border=ft.Border.all(1, Colors.GLASS_BORDER_ACTIVE if is_selected else Colors.GLASS_BORDER),
            border_radius=16,
            padding=ft.Padding.symmetric(horizontal=12, vertical=6),
            on_click=lambda _, v=value: self.set_filter(v),
            animate=ft.Animation(150, ft.AnimationCurve.EASE_OUT),
        )

    def _build_grid(self):
        return ft.Container(
            content=ft.GridView(
                runs_count=None,
                max_extent=200,
                child_aspect_ratio=0.85,
                spacing=12,
                run_spacing=12,
                padding=ft.Padding.symmetric(horizontal=24, vertical=8),
                children=[self._build_grid_item(f) for f in self.files],
            ),
            expand=True,
        )

    def _build_grid_item(self, file_path):
        name = os.path.basename(file_path)
        thumbnail_src = self._get_thumbnail(file_path)
        is_video = self.app.file_handler.is_video(file_path)

        return ft.Container(
            content=ft.Column([
                # Thumbnail
                ft.Container(
                    content=ft.Stack([
                        ft.Image(
                            src=thumbnail_src, width=180, height=140,
                            fit=ft.BoxFit.COVER, border_radius=ft.BorderRadius.only(
                                top_left=10, top_right=10,
                            ),
                        ) if thumbnail_src else ft.Container(
                            content=ft.Icon(
                                ft.Icons.VIDEO_FILE if is_video else ft.Icons.IMAGE,
                                size=36, color=Colors.TEXT_MUTED,
                            ),
                            alignment=ft.Alignment.CENTER,
                            expand=True,
                        ),
                        # Video badge
                        ft.Container(
                            content=ft.Icon(ft.Icons.PLAY_ARROW, size=16, color=ft.Colors.WHITE),
                            bgcolor=ft.Colors.with_opacity(0.7, Colors.BG_PRIMARY),
                            border_radius=12,
                            padding=4,
                            top=8, right=8,
                        ) if is_video else ft.Container(),
                    ], expand=True),
                    height=140,
                ),
                # File info
                ft.Container(
                    content=ft.Column([
                        ft.Text(
                            name[:22] + "..." if len(name) > 22 else name,
                            size=11, color=Colors.TEXT_PRIMARY,
                            overflow=ft.TextOverflow.ELLIPSIS,
                        ),
                        ft.Text(
                            "Video" if is_video else "Image",
                            size=10, color=Colors.TEXT_MUTED,
                        ),
                    ], spacing=2),
                    padding=ft.Padding.symmetric(horizontal=8, vertical=6),
                ),
            ], spacing=0),
            bgcolor=Colors.BG_CARD,
            border=ft.Border.all(1, Colors.GLASS_BORDER),
            border_radius=10,
            on_click=lambda _, p=file_path: self.select_file(p),
            animate=ft.Animation(150, ft.AnimationCurve.EASE_OUT),
        )

    def _build_empty_state(self):
        return ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Icon(ft.Icons.PHOTO_LIBRARY, size=48, color=Colors.TEXT_MUTED),
                    bgcolor=Colors.GLASS_BG,
                    border_radius=50,
                    padding=20,
                ),
                ft.Text("No media found", size=16, color=Colors.TEXT_PRIMARY, weight=ft.FontWeight.W_500),
                ft.Text("Upload from the Upload tab", size=12, color=Colors.TEXT_SECONDARY),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
            alignment=ft.Alignment.CENTER,
            expand=True,
        )

    def _get_thumbnail(self, file_path):
        try:
            if self.app.file_handler.is_video(file_path):
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    return ""
            else:
                frame = cv2.imread(file_path)

            if frame is None:
                return ""

            frame = cv2.resize(frame, (200, 150))
            _, buf = cv2.imencode('.jpg', frame)
            return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"
        except Exception as e:
            print(f"[Gallery] Thumbnail error: {e}")
            return ""

    def load_files(self):
        try:
            all_files = self.app.file_handler.list_media_files()
            if self.filter == "images":
                self.files = [f for f in all_files if self.app.file_handler.is_image(f)]
            elif self.filter == "videos":
                self.files = [f for f in all_files if self.app.file_handler.is_video(f)]
            else:
                self.files = all_files
        except Exception as e:
            print(f"[Gallery] Error: {e}")
            self.files = []

    def set_filter(self, filter_type):
        self.filter = filter_type
        self.app._show_screen("gallery")

    def select_file(self, file_path):
        self.app.current_file = file_path
        self.app.show_snackbar(f"Selected: {os.path.basename(file_path)}")
        self.app.go("inference")
