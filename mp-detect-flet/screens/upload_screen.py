# screens/upload_screen.py - File picker screen
import flet as ft
import os


class UploadScreen:
    def __init__(self, app):
        self.app = app
        self.selected_file = app.current_file

    def build(self) -> ft.View:
        return ft.View(
            "/upload",
            [
                ft.AppBar(title=ft.Text("Upload"), bgcolor=ft.Colors.SURFACE),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Icon(ft.Icons.UPLOAD_FILE, size=48, color=ft.Colors.PRIMARY),
                                        ft.Text("Tap to pick file", size=16),
                                        ft.Text("Images: png, jpg, jpeg, tif, bmp", size=12, color=ft.Colors.with_opacity(0.7, ft.Colors.ON_SURFACE)),
                                        ft.Text("Videos: mp4, avi, mov, mkv", size=12, color=ft.Colors.with_opacity(0.7, ft.Colors.ON_SURFACE)),
                                    ],
                                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                    spacing=8,
                                ),
                                width=300,
                                height=200,
                                border=ft.Border.all(2, ft.Colors.OUTLINE),
                                border_radius=10,
                                alignment=ft.Alignment.CENTER,
                                on_click=lambda _: self.pick_file(),
                            ),
                            ft.ElevatedButton("Pick from Gallery", icon=ft.Icons.PHOTO_LIBRARY, on_click=lambda _: self.pick_file()),
                            self.build_selected_file_display(),
                            self.build_recent_files(),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=20,
                    ),
                    padding=20,
                    expand=True,
                ),
                self.app.nav_bar.build() if hasattr(self.app, 'nav_bar') else ft.Container(),
            ],
        )

    def build_selected_file_display(self):
        if self.selected_file is None:
            return ft.Container()
        filename = os.path.basename(self.selected_file)
        return ft.Card(
            content=ft.Container(
                content=ft.Row(
                    [
                        ft.Icon(ft.Icons.IMAGE, color=ft.Colors.PRIMARY),
                        ft.Column([ft.Text("Selected File", size=12), ft.Text(filename, size=14, weight=ft.FontWeight.BOLD)], spacing=4),
                        ft.IconButton(icon=ft.Icons.ARROW_FORWARD, on_click=lambda _: self.app.go("/gallery")),
                    ],
                    spacing=10,
                ),
                padding=10,
            ),
        )

    def build_recent_files(self):
        try:
            files = self.app.file_handler.list_media_files()[:5]
        except Exception:
            files = []
        if not files:
            return ft.Container()
        return ft.Column(
            [
                ft.Text("Recent Files", size=14, weight=ft.FontWeight.BOLD),
                ft.Row([self.build_chip(f) for f in files], wrap=True, spacing=8),
            ],
            spacing=8,
        )

    def build_chip(self, path):
        name = os.path.basename(path)
        icon = ft.Icons.VIDEO_FILE if self.app.file_handler.is_video(path) else ft.Icons.IMAGE
        return ft.Chip(label=ft.Text(name[:20], size=12), leading=ft.Icon(icon, size=16), tooltip=name, on_click=lambda _, p=path: self.select_file(p))

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
            self.select_file(result[0].path)

    def select_file(self, file_path):
        if file_path is None or not os.path.isfile(file_path):
            self.app.show_snackbar("Invalid file selected")
            return
        valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".mp4", ".avi", ".mov", ".mkv")
        if not file_path.lower().endswith(valid_exts):
            self.app.show_snackbar("Unsupported file format")
            return
        self.selected_file = file_path
        self.app.current_file = file_path
        self.app.show_snackbar(f"Selected: {os.path.basename(file_path)}")
        self.app.go("/gallery")
