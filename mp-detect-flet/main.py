# main.py - MP Detect Flet App Entry Point
import flet as ft

from core.__version__ import __version__
from core.file_handler import FileHandler
from core.model_manager import ModelManager
from core.settings_manager import load_settings, save_settings


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

        # Keyboard shortcuts
        page.on_keyboard_event = self._on_keyboard

        # Set up routing
        page.on_route_change = self.route_change
        page.go("/upload")

    def _on_keyboard(self, e):
        """Handle keyboard shortcuts."""
        key = e.key
        ctrl = e.ctrl
        shift = e.shift

        # Space - Toggle detection
        if key == " " and not ctrl:
            if self.current_screen and hasattr(self.current_screen, 'toggle_detection'):
                self.current_screen.toggle_detection()
            return

        # Ctrl+O - Open file
        if ctrl and key == "o":
            if self.current_screen and hasattr(self.current_screen, 'pick_file'):
                self.current_screen.pick_file()
            return

        # Ctrl+E - Quick export
        if ctrl and key == "e":
            if self.current_results:
                try:
                    from core.export import export_csv, export_annotated_image
                    csv_path = self.file_handler.get_export_csv_path()
                    export_csv(self.current_results, csv_path, self.settings)
                    self.show_snackbar(f"CSV exported")
                except Exception as e:
                    self.show_snackbar(f"Export failed: {e}")
            return

        # [ - Decrease confidence
        if key == "[":
            new_val = max(0.01, self.settings.get("conf", 0.25) - 0.05)
            self.settings["conf"] = round(new_val, 2)
            save_settings(self.settings)
            self.show_snackbar(f"Confidence: {self.settings['conf']:.2f}")
            return

        # ] - Increase confidence
        if key == "]":
            new_val = min(0.99, self.settings.get("conf", 0.25) + 0.05)
            self.settings["conf"] = round(new_val, 2)
            save_settings(self.settings)
            self.show_snackbar(f"Confidence: {self.settings['conf']:.2f}")
            return

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
