# screens/inference_screen.py - Detection view screen
"""Inference screen for MP Detect Flet app."""
import flet as ft


class InferenceScreen:
    def __init__(self, app):
        self.app = app
        
    def build(self) -> ft.View:
        return ft.View(
            "/inference",
            [
                ft.AppBar(
                    title=ft.Text("Inference"),
                    leading=ft.IconButton(
                        ft.icons.ARROW_BACK,
                        on_click=lambda _: self.app.go("/gallery"),
                    ),
                    bgcolor=ft.colors.SURFACE_VARIANT,
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text("Inference Screen"),
                            ft.Text("Coming in Phase 4"),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    alignment=ft.alignment.center,
                    expand=True,
                ),
            ],
        )
