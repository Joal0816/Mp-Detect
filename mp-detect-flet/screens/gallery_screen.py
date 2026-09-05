# screens/gallery_screen.py - Grid view screen
import flet as ft
import os


class GalleryScreen:
    def __init__(self, app):
        self.app = app
        self.files = []
        self.filter = "all"

    def build(self) -> ft.View:
        self.grid_view = self.build_grid_view()
        return ft.View(
            "/gallery",
            [
                ft.AppBar(
                    title=ft.Text("Gallery"),
                    leading=ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=lambda _: self.app.go("/upload")),
                    bgcolor=ft.Colors.SURFACE,
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Row(
                                [
                                    ft.FilterChip(label=ft.Text("All"), selected=self.filter == "all", on_select=lambda _: self.set_filter("all")),
                                    ft.FilterChip(label=ft.Text("Images"), selected=self.filter == "images", on_select=lambda _: self.set_filter("images")),
                                    ft.FilterChip(label=ft.Text("Videos"), selected=self.filter == "videos", on_select=lambda _: self.set_filter("videos")),
                                ],
                                spacing=10,
                            ),
                            self.grid_view,
                        ],
                        spacing=10,
                    ),
                    padding=10,
                    expand=True,
                ),
                self.app.nav_bar.build() if hasattr(self.app, 'nav_bar') else ft.Container(),
            ],
        )

    def build_grid_view(self):
        self.load_files()
        if not self.files:
            return ft.Container(
                content=ft.Column(
                    [
                        ft.Icon(ft.Icons.PHOTO_LIBRARY, size=48, color=ft.Colors.with_opacity(0.5, ft.Colors.ON_SURFACE)),
                        ft.Text("No media found", size=16),
                        ft.Text("Upload from the Upload tab", size=12, color=ft.Colors.with_opacity(0.7, ft.Colors.ON_SURFACE)),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=10,
                ),
                alignment=ft.Alignment.CENTER,
                expand=True,
            )
        return ft.GridView(runs_count=3, child_aspect_ratio=0.8, spacing=10, run_spacing=10, children=[self.build_grid_item(f) for f in self.files])

    def build_grid_item(self, file_path):
        name = os.path.basename(file_path)
        icon = ft.Icons.VIDEO_FILE if self.app.file_handler.is_video(file_path) else ft.Icons.IMAGE
        return ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Icon(icon, size=48, color=ft.Colors.PRIMARY),
                        ft.Text(name[:15] + "..." if len(name) > 15 else name, size=10, text_align=ft.TextAlign.CENTER),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=5,
                ),
                padding=10,
                tooltip=name,
                on_click=lambda _, p=file_path: self.select_file(p),
            ),
            elevation=2,
        )

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
        self.app.page.views.clear()
        self.app.page.views.append(self.build())
        self.app.page.update()

    def select_file(self, file_path):
        self.app.current_file = file_path
        self.app.show_snackbar(f"Selected: {os.path.basename(file_path)}")
        self.app.go("/inference")
