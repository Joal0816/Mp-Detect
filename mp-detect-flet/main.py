# main.py - MP Detect Flet App Entry Point
import flet as ft

# Import core modules
from core.file_handler import FileHandler
from core.model_manager import ModelManager
from core.settings_manager import load_settings

# Import screens
from screens.upload_screen import UploadScreen
from screens.gallery_screen import GalleryScreen
from screens.inference_screen import InferenceScreen
from screens.result_screen import ResultScreen
from screens.model_manager_screen import ModelManagerScreen

# Import components
from components.nav_bar import NavBar


class MPDetectApp:
    def __init__(self):
        self.page = None
        self.file_handler = FileHandler()
        self.model_manager = ModelManager()
        self.settings = load_settings()
        self.nav_bar = None
        self.snackbar = None
        self.file_picker = None
        self.current_screen = None
        
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
        
        # Initialize components
        self.nav_bar = NavBar(self)
        self.snackbar = ft.SnackBar(content=ft.Text(""))
        self.file_picker = ft.FilePicker(on_result=self._on_file_picked)
        
        # Add overlay components
        page.overlay.append(self.snackbar)
        page.overlay.append(self.file_picker)
        
        # Initialize active model
        try:
            self.current_engine = self.model_manager.get_active_engine()
        except Exception as e:
            print(f"[Init] No active model: {e}")
        
        # Set up routing
        page.on_route_change = self.route_change
        page.go("/")
        
    def _on_file_picked(self, e):
        """Handle file picker result."""
        if e.files and len(e.files) > 0:
            file_path = e.files[0].path
            if file_path:
                self.current_file = file_path
                # Notify current screen if it has the callback
                if self.current_screen and hasattr(self.current_screen, '_on_file_selected'):
                    self.current_screen._on_file_selected(file_path)
        
    def route_change(self, route):
        self.page.views.clear()
        
        # Root route - redirect to upload
        if route.route == "/":
            self.page.go("/upload")
            return
            
        # Upload screen
        if route.route == "/upload":
            self.current_screen = UploadScreen(self)
            view = self.current_screen.build()
            self.page.views.append(view)
            
        # Gallery screen
        elif route.route.startswith("/gallery"):
            self.current_screen = GalleryScreen(self)
            view = self.current_screen.build()
            self.page.views.append(view)
            
        # Inference screen
        elif route.route.startswith("/inference"):
            self.current_screen = InferenceScreen(self)
            view = self.current_screen.build()
            self.page.views.append(view)
            
        # Result screen
        elif route.route.startswith("/result"):
            self.current_screen = ResultScreen(self)
            view = self.current_screen.build()
            self.page.views.append(view)
            
        # Model manager screen
        elif route.route == "/models":
            self.current_screen = ModelManagerScreen(self)
            view = self.current_screen.build()
            self.page.views.append(view)
            
        self.page.update()
        
    def go(self, route):
        self.page.go(route)
        
    def show_snackbar(self, message: str):
        self.snackbar.content = ft.Text(message)
        self.snackbar.open = True
        self.page.update()
        
    def pick_file(self, dialog_title="Select File", allowed_extensions=None):
        """Open file picker dialog."""
        if allowed_extensions is None:
            allowed_extensions = ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "mp4", "avi", "mov", "mkv"]
        
        self.file_picker.pick_files(
            dialog_title=dialog_title,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=allowed_extensions,
        )


def main():
    app = MPDetectApp()
    ft.app(target=app.main)


if __name__ == "__main__":
    main()
