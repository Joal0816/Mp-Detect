# screens/result_screen.py - Results view screen
"""Result screen for MP Detect Flet app."""
import flet as ft


class ResultScreen:
    def __init__(self, app):
        self.app = app
        
    def build(self) -> ft.View:
        return ft.View(
            "/result",
            [
                ft.AppBar(
                    title=ft.Text("Results"),
                    leading=ft.IconButton(
                        ft.icons.ARROW_BACK,
                        on_click=lambda _: self.app.go("/inference"),
                    ),
                    bgcolor=ft.colors.SURFACE_VARIANT,
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text("Result Screen"),
                            ft.Text("Coming in Phase 5"),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    alignment=ft.alignment.center,
                    expand=True,
                ),
            ],
        )
