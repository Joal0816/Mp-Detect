# components/nav_bar.py - Bottom navigation bar
"""Bottom navigation bar component for MP Detect Flet app."""
import flet as ft


class NavBar:
    def __init__(self, app):
        self.app = app
        
    def build(self) -> ft.Container:
        return ft.Container(
            content=ft.Row(
                [
                    ft.TextButton(
                        text="Upload",
                        icon=ft.icons.UPLOAD,
                        on_click=lambda _: self.app.go("/upload"),
                    ),
                    ft.TextButton(
                        text="Gallery",
                        icon=ft.icons.PHOTO_LIBRARY,
                        on_click=lambda _: self.app.go("/gallery"),
                    ),
                    ft.TextButton(
                        text="Detect",
                        icon=ft.icons.CAMERA_ALT,
                        on_click=lambda _: self.app.go("/inference"),
                    ),
                    ft.TextButton(
                        text="Models",
                        icon=ft.icons.SETTINGS,
                        on_click=lambda _: self.app.go("/models"),
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_AROUND,
            ),
            bgcolor=ft.colors.SURFACE_VARIANT,
            padding=10,
        )
