# main.py - MP Detect Flet App Entry Point
import flet as ft
import os

# Import core modules
from core.file_handler import FileHandler
from core.model_manager import ModelManager
from core.settings_manager import load_settings, save_settings


class MPDetectApp:
    def __init__(self):
        self.page = None
        self.file_handler = FileHandler()
        self.model_manager = ModelManager()
        self.settings = load_settings()
        
        # Current state
        self.current_file = None
        self.current_results = []
        self.current_engine = None
        
    def main(self, page: ft.Page):
        self.page = page
        page.title = "MP Detect"
        page.theme_mode = ft.ThemeMode.DARK
        page.theme = ft.Theme(
            color_scheme_seed=ft.colors.CYAN,
        )
        page.window.width = 400
        page.window.height = 800
        page.padding = 0
        
        # Initialize active model
        try:
            self.current_engine = self.model_manager.get_active_engine()
        except Exception as e:
            print(f"[Init] No active model: {e}")
        
        # Set up routing
        page.on_route_change = self.route_change
        page.go("/")
        
    def route_change(self, route):
        self.page.views.clear()
        
        # Root route - redirect to upload
        if route.route == "/":
            self.page.go("/upload")
            return
            
        # Import screens here to avoid circular imports
        from screens.upload_screen import UploadScreen
        from screens.gallery_screen import GalleryScreen
        from screens.inference_screen import InferenceScreen
        from screens.result_screen import ResultScreen
        from screens.model_manager_screen import ModelManagerScreen
            
        # Upload screen
        if route.route == "/upload":
            view = UploadScreen(self).build()
            self.page.views.append(view)
            
        # Gallery screen
        elif route.route.startswith("/gallery"):
            view = GalleryScreen(self).build()
            self.page.views.append(view)
            
        # Inference screen
        elif route.route.startswith("/inference"):
            view = InferenceScreen(self).build()
            self.page.views.append(view)
            
        # Result screen
        elif route.route.startswith("/result"):
            view = ResultScreen(self).build()
            self.page.views.append(view)
            
        # Model manager screen
        elif route.route == "/models":
            view = ModelManagerScreen(self).build()
            self.page.views.append(view)
            
        self.page.update()
        
    def go(self, route):
        self.page.go(route)
        
    def show_snackbar(self, message: str):
        self.page.snack_bar = ft.SnackBar(
            content=ft.Text(message),
            bgcolor=ft.colors.SURFACE_VARIANT,
        )
        self.page.snack_bar.open = True
        self.page.update()


def main():
    app = MPDetectApp()
    ft.app(target=app.main)


if __name__ == "__main__":
    main()
