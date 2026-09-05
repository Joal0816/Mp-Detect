# screens/upload_screen.py - File picker screen
"""Upload screen for MP Detect Flet app."""
import flet as ft


class UploadScreen:
    def __init__(self, app):
        self.app = app
        
    def build(self) -> ft.View:
        return ft.View(
            "/upload",
            [
                ft.AppBar(
                    title=ft.Text("Upload"),
                    bgcolor=ft.colors.SURFACE_VARIANT,
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Icon(ft.icons.UPLOAD_FILE, size=48),
                                        ft.Text("Tap to pick file"),
                                        ft.Text("Images: png, jpg, jpeg, tif, bmp", size=12),
                                        ft.Text("Videos: mp4, avi, mov, mkv", size=12),
                                    ],
                                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                ),
                                width=300,
                                height=200,
                                border=ft.border.all(2, ft.colors.OUTLINE),
                                border_radius=10,
                                alignment=ft.alignment.center,
                            ),
                            ft.ElevatedButton(
                                "Pick from Gallery",
                                icon=ft.icons.PHOTO_LIBRARY,
                            ),
                            ft.ElevatedButton(
                                "Take Photo",
                                icon=ft.icons.CAMERA_ALT,
                            ),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=20,
                    ),
                    padding=20,
                ),
            ],
        )
