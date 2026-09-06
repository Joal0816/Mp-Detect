# main.py - MP Detect Flet App Entry Point
import flet as ft

from core.__version__ import __version__
from core.file_handler import FileHandler
from core.model_manager import ModelManager
from core.settings_manager import load_settings, save_settings


class MPDetectApp:
    NAV_ROUTES = {0: "inference", 1: "upload", 2: "gallery", 3: "models"}
    NAV_INDEX = {v: k for k, v in NAV_ROUTES.items()}

    def __init__(self):
        self.page = None
        self.file_handler = FileHandler()
        self.model_manager = ModelManager()
        self.settings = load_settings()
        self.snackbar = None
        self.nav_bar = None
        self.content_area = None
        self.current_screen_name = "upload"
        self.current_screen = None

        # Current state
        self.current_file = None
        self.current_results = []
        self.current_engine = None
        self.current_annotated = None
        self.current_stats = None
        self.current_frame = None

        # Try to load active model
        try:
            self.current_engine = self.model_manager.get_active_engine()
        except Exception as e:
            print(f"[Init] No active model: {e}")

    def main(self, page: ft.Page):
        self.page = page
        page.title = f"MP Detect v{__version__}"
        page.theme_mode = ft.ThemeMode.DARK
        page.theme = ft.Theme(color_scheme_seed="cyan")
        page.window.width = 1280
        page.window.height = 800
        page.padding = 0

        # Snackbar
        self.snackbar = ft.SnackBar(content=ft.Text(""))
        page.overlay.append(self.snackbar)

        # Content area for screens
        self.content_area = ft.Container(expand=True)

        # Navigation bar
        self.nav_bar = ft.NavigationBar(
            selected_index=1,
            on_change=self._on_nav_change,
            destinations=[
                ft.NavigationBarDestination(icon=ft.Icons.CAMERA_ALT, label="Detect"),
                ft.NavigationBarDestination(icon=ft.Icons.UPLOAD, label="Upload"),
                ft.NavigationBarDestination(icon=ft.Icons.PHOTO_LIBRARY, label="Gallery"),
                ft.NavigationBarDestination(icon=ft.Icons.SETTINGS, label="Models"),
            ],
            bgcolor=ft.Colors.SURFACE,
        )

        # Build initial layout
        page.add(
            ft.Column(
                [
                    self.content_area,
                    self.nav_bar,
                ],
                spacing=0,
                expand=True,
            )
        )

        # Show upload screen
        self._show_screen("upload")

    def _on_nav_change(self, e):
        index = e.control.selected_index
        route = self.NAV_ROUTES.get(index)
        if route:
            self._show_screen(route)

    def _show_screen(self, name):
        # Cleanup previous screen (e.g., stop camera)
        if self.current_screen and hasattr(self.current_screen, 'cleanup'):
            self.current_screen.cleanup()

        self.current_screen_name = name
        self.nav_bar.selected_index = self.NAV_INDEX.get(name, 0)

        # Import and build screen
        screen = None
        if name == "upload":
            from screens.upload_screen import UploadScreen
            screen = UploadScreen(self)
            self.content_area.content = screen.build_content()
        elif name == "gallery":
            from screens.gallery_screen import GalleryScreen
            screen = GalleryScreen(self)
            self.content_area.content = screen.build_content()
        elif name == "inference":
            from screens.inference_screen import InferenceScreen
            screen = InferenceScreen(self)
            self.content_area.content = screen.build_content()
        elif name == "models":
            from screens.model_manager_screen import ModelManagerScreen
            screen = ModelManagerScreen(self)
            self.content_area.content = screen.build_content()
        elif name == "result":
            from screens.result_screen import ResultScreen
            screen = ResultScreen(self)
            self.content_area.content = screen.build_content()

        self.current_screen = screen
        self.page.update()

    def go(self, route):
        name = route.lstrip("/")
        self._show_screen(name)

    def show_snackbar(self, message: str):
        self.snackbar.content = ft.Text(message)
        self.snackbar.open = True
        self.page.update()


app = MPDetectApp()


def main(page: ft.Page):
    app.main(page)


if __name__ == "__main__":
    ft.run(main)
