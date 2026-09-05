# screens/gallery_screen.py - Grid view screen
"""Gallery screen for MP Detect Flet app."""
import flet as ft


class GalleryScreen:
    def __init__(self, app):
        self.app = app
        
    def build(self) -> ft.View:
        return ft.View(
            "/gallery",
            [
                ft.AppBar(
                    title=ft.Text("Gallery"),
                    leading=ft.IconButton(
                        ft.icons.ARROW_BACK,
                        on_click=lambda _: self.app.go("/upload"),
                    ),
                    bgcolor=ft.colors.SURFACE_VARIANT,
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text("Gallery Grid View"),
                            ft.Text("Coming in Phase 3"),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    alignment=ft.alignment.center,
                    expand=True,
                ),
                # Bottom navigation bar
                self.app.nav_bar.build(),
            ],
        )
