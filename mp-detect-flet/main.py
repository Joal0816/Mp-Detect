# main.py - MP Detect Flet App Entry Point
import flet as ft

from core.__version__ import __version__
from core.file_handler import FileHandler
from core.model_manager import ModelManager
from core.settings_manager import load_settings
from components.theme import Colors, GRADIENT_CYAN


class MPDetectApp:
    NAV_ROUTES = {0: "inference", 1: "upload", 2: "gallery", 3: "models"}
    NAV_INDEX = {v: k for k, v in NAV_ROUTES.items()}
    NAV_ICONS = {
        "inference": ft.Icons.SCIENCE,
        "upload": ft.Icons.UPLOAD_FILE,
        "gallery": ft.Icons.PHOTO_LIBRARY,
        "models": ft.Icons.SMART_TOY,
    }
    NAV_LABELS = {
        "inference": "Detect",
        "upload": "Upload",
        "gallery": "Gallery",
        "models": "Models",
    }

    def __init__(self):
        self.page = None
        self.file_handler = FileHandler()
        self.model_manager = ModelManager()
        self.settings = load_settings()
        self.snackbar = None
        self.nav_bar = None
        self.nav_rail = None
        self.content_area = None
        self.file_picker = None
        self.current_screen_name = "upload"
        self.current_screen = None
        self.is_desktop = False

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
        page.theme = ft.Theme(
            color_scheme_seed=Colors.ACCENT_CYAN,
            color_scheme=ft.ColorScheme(
                surface=Colors.BG_SURFACE,
                surface_dim=Colors.BG_PRIMARY,
                surface_container=Colors.BG_CARD,
                surface_container_high=Colors.BG_ELEVATED,
                on_surface=Colors.TEXT_PRIMARY,
                primary=Colors.ACCENT_CYAN,
                on_primary=Colors.BG_PRIMARY,
            ),
        )
        page.window.width = 1280
        page.window.height = 800
        page.padding = 0
        page.bgcolor = Colors.BG_PRIMARY

        # Snackbar
        self.snackbar = ft.SnackBar(
            content=ft.Text(""),
            bgcolor=Colors.BG_ELEVATED,
            
        )
        page.overlay.append(self.snackbar)

        # Shared FilePicker
        self.file_picker = ft.FilePicker()
        page.services.append(self.file_picker)

        # Content area
        self.content_area = ft.Container(expand=True, bgcolor=Colors.BG_PRIMARY)

        # Detect screen width and build appropriate nav
        self._build_layout(page)

        # Listen for window resize
        page.on_resize = self._on_resize

        # Show upload screen
        self._show_screen("upload")

    def _build_layout(self, page):
        """Build responsive layout based on screen width."""
        width = page.window.width or 1280
        self.is_desktop = width >= 900

        # Clear existing
        page.controls.clear()

        if self.is_desktop:
            self._build_desktop_layout(page)
        else:
            self._build_mobile_layout(page)

        page.update()

    def _build_desktop_layout(self, page):
        """Desktop: NavigationRail + Content"""
        self.nav_rail = ft.NavigationRail(
            selected_index=self.NAV_INDEX.get(self.current_screen_name, 1),
            on_change=self._on_nav_change,
            label_type=ft.NavigationRailLabelType.ALL,
            min_width=72,
            min_extended_width=200,
            bgcolor=Colors.BG_SURFACE,
            indicator_color=ft.Colors.with_opacity(0.15, Colors.ACCENT_CYAN),
            destinations=[
                ft.NavigationRailDestination(
                    icon=ft.Icon(self.NAV_ICONS[name], color=Colors.TEXT_SECONDARY),
                    selected_icon=ft.Icon(self.NAV_ICONS[name], color=Colors.ACCENT_CYAN),
                    label=ft.Text(self.NAV_LABELS[name], size=11),
                )
                for name in ["inference", "upload", "gallery", "models"]
            ],
            leading=ft.Container(
                content=ft.Column([
                    ft.Icon(ft.Icons.SCIENCE, size=28, color=Colors.ACCENT_CYAN),
                    ft.Text("MP", size=10, color=Colors.TEXT_SECONDARY, weight=ft.FontWeight.BOLD),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
                padding=ft.Padding.symmetric(vertical=12),
            ),
            trailing=ft.Container(
                content=ft.Text(f"v{__version__}", size=9, color=Colors.TEXT_MUTED),
                alignment=ft.Alignment.CENTER,
                padding=8,
            ),
        )

        page.add(
            ft.Row(
                [
                    ft.Container(
                        content=self.nav_rail,
                        width=72,
                        bgcolor=Colors.BG_SURFACE,
                    ),
                    ft.VerticalDivider(width=1, color=Colors.GLASS_BORDER),
                    self.content_area,
                ],
                spacing=0,
                expand=True,
            )
        )

    def _build_mobile_layout(self, page):
        """Mobile: Content + Bottom NavigationBar"""
        self.nav_bar = ft.NavigationBar(
            selected_index=self.NAV_INDEX.get(self.current_screen_name, 1),
            on_change=self._on_nav_change,
            bgcolor=Colors.BG_SURFACE,
            indicator_color=ft.Colors.with_opacity(0.15, Colors.ACCENT_CYAN),
            height=64,
            label_behavior=ft.NavigationBarLabelBehavior.ALWAYS_SHOW,
            destinations=[
                ft.NavigationBarDestination(
                    icon=ft.Icon(self.NAV_ICONS[name], color=Colors.TEXT_SECONDARY),
                    selected_icon=ft.Icon(self.NAV_ICONS[name], color=Colors.ACCENT_CYAN),
                    label=ft.Text(self.NAV_LABELS[name], size=10),
                )
                for name in ["inference", "upload", "gallery", "models"]
            ],
        )

        page.add(
            ft.Column(
                [
                    self.content_area,
                    ft.Divider(height=1, color=Colors.GLASS_BORDER),
                    self.nav_bar,
                ],
                spacing=0,
                expand=True,
            )
        )

    def _on_resize(self, e):
        """Handle window resize — switch between mobile/desktop nav."""
        width = self.page.window.width
        was_desktop = self.is_desktop
        self.is_desktop = width >= 900

        if was_desktop != self.is_desktop:
            self._build_layout(self.page)
            self._show_screen(self.current_screen_name)

    def _on_nav_change(self, e):
        index = e.control.selected_index
        route = self.NAV_ROUTES.get(index)
        if route:
            self._show_screen(route)

    def _show_screen(self, name):
        # Cleanup previous screen
        if self.current_screen and hasattr(self.current_screen, 'cleanup'):
            self.current_screen.cleanup()

        self.current_screen_name = name

        # Update nav selection
        if self.is_desktop and self.nav_rail:
            self.nav_rail.selected_index = self.NAV_INDEX.get(name, 0)
        elif self.nav_bar:
            self.nav_bar.selected_index = self.NAV_INDEX.get(name, 0)

        # Import and build screen
        screen = None
        if name == "upload":
            from screens.upload_screen import UploadScreen
            screen = UploadScreen(self)
        elif name == "gallery":
            from screens.gallery_screen import GalleryScreen
            screen = GalleryScreen(self)
        elif name == "inference":
            from screens.inference_screen import InferenceScreen
            screen = InferenceScreen(self)
        elif name == "models":
            from screens.model_manager_screen import ModelManagerScreen
            screen = ModelManagerScreen(self)
        elif name == "result":
            from screens.result_screen import ResultScreen
            screen = ResultScreen(self)

        if screen:
            self.content_area.content = ft.Container(
                content=screen.build_content(),
                expand=True,
                animate_opacity=200,
                opacity=0,
            )
            # Fade in
            self.content_area.content.opacity = 1
            self.current_screen = screen
            self.page.update()

    def go(self, route):
        name = route.lstrip("/")
        self._show_screen(name)

    def show_snackbar(self, message: str):
        self.snackbar.content = ft.Text(message, color=Colors.TEXT_PRIMARY)
        self.snackbar.open = True
        self.page.update()


app = MPDetectApp()


def main(page: ft.Page):
    app.main(page)


if __name__ == "__main__":
    ft.run(main)
