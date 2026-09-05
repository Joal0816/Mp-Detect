# main.py - MP Detect Flet App Entry Point
import flet as ft

from core.__version__ import __version__
from core.file_handler import FileHandler
from core.model_manager import ModelManager
from core.settings_manager import load_settings


class MPDetectApp:
    def __init__(self):
        self.page = None
        self.file_handler = FileHandler()
        self.model_manager = ModelManager()
        self.settings = load_settings()
        self.snackbar = None
        self.current_screen = None
        self.nav_bar = None

        # Current state
        self.current_file = None
        self.current_results = []
        self.current_engine = None

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

        # Import components
        from components.nav_bar import NavBar
        self.nav_bar = NavBar(self)

        # Import screens
        from screens.upload_screen import UploadScreen
        from screens.gallery_screen import GalleryScreen
        from screens.inference_screen import InferenceScreen
        from screens.result_screen import ResultScreen
        from screens.model_manager_screen import ModelManagerScreen

        self._screens = {
            "upload": UploadScreen,
            "gallery": GalleryScreen,
            "inference": InferenceScreen,
            "result": ResultScreen,
            "models": ModelManagerScreen,
        }

        # Set up routing
        page.on_route_change = self.route_change
        page.go("/upload")

    def route_change(self, route):
        self.page.views.clear()

        if route.route == "/":
            self.page.go("/upload")
            return

        screen_name = route.route.lstrip("/")
        ScreenClass = self._screens.get(screen_name)
        if ScreenClass:
            self.current_screen = ScreenClass(self)
            view = self.current_screen.build()
            self.page.views.append(view)

        self.page.update()

    def go(self, route):
        self.page.go(route)

    def show_snackbar(self, message: str):
        self.snackbar.content = ft.Text(message)
        self.snackbar.open = True
        self.page.update()


app = MPDetectApp()


def main(page: ft.Page):
    app.main(page)


if __name__ == "__main__":
    ft.run(main)
