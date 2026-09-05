# components/nav_bar.py - Bottom navigation bar
import flet as ft


class NavBar:
    def __init__(self, app):
        self.app = app
        self.current_index = 0

    def build(self) -> ft.NavigationBar:
        return ft.NavigationBar(
            selected_index=self.current_index,
            on_change=self.on_nav_change,
            destinations=[
                ft.NavigationBarDestination(icon=ft.Icons.CAMERA_ALT, label="Detect"),
                ft.NavigationBarDestination(icon=ft.Icons.UPLOAD, label="Upload"),
                ft.NavigationBarDestination(icon=ft.Icons.PHOTO_LIBRARY, label="Gallery"),
                ft.NavigationBarDestination(icon=ft.Icons.SETTINGS, label="Models"),
            ],
            bgcolor=ft.Colors.SURFACE,
        )

    def on_nav_change(self, e):
        index = e.control.selected_index
        self.current_index = index
        routes = ["/inference", "/upload", "/gallery", "/models"]
        if 0 <= index < len(routes):
            self.app.go(routes[index])
