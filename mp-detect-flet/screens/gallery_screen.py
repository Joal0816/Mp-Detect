# screens/gallery_screen.py - Grid view screen
"""Gallery screen for MP Detect Flet app."""
import flet as ft
import os
from typing import List


class GalleryScreen:
    def __init__(self, app):
        self.app = app
        self.files: List[str] = []
        self.filter = "all"
        self.selected_file = None
        
    def build(self) -> ft.View:
        # Grid view
        self.grid_view = self.build_grid_view()
        
        return ft.View(
            "/gallery",
            [
                ft.AppBar(
                    title=ft.Text("Gallery"),
                    leading=ft.IconButton(
                        icon=ft.icons.ARROW_BACK,
                        on_click=lambda _: self.app.go("/upload"),
                    ),
                    bgcolor=ft.colors.SURFACE_VARIANT,
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            # Filter tabs
                            self.build_filter_tabs(),
                            
                            # Grid view
                            self.grid_view,
                        ],
                        spacing=10,
                    ),
                    padding=10,
                    expand=True,
                ),
                # Bottom navigation bar
                self.app.nav_bar.build(),
            ],
        )
        
    def build_filter_tabs(self):
        """Build filter tabs for All/Images/Videos."""
        return ft.Row(
            [
                ft.FilterChip(
                    label=ft.Text("All"),
                    selected=self.filter == "all",
                    on_select=lambda _: self.set_filter("all"),
                ),
                ft.FilterChip(
                    label=ft.Text("Images"),
                    selected=self.filter == "images",
                    on_select=lambda _: self.set_filter("images"),
                ),
                ft.FilterChip(
                    label=ft.Text("Videos"),
                    selected=self.filter == "videos",
                    on_select=lambda _: self.set_filter("videos"),
                ),
            ],
            spacing=10,
        )
        
    def build_grid_view(self):
        """Build grid view of media files."""
        self.load_files()
        
        if not self.files:
            return ft.Container(
                content=ft.Column(
                    [
                        ft.Icon(ft.icons.PHOTO_LIBRARY, size=48, color=ft.colors.with_opacity(0.5, ft.colors.ON_SURFACE)),
                        ft.Text("No media found", size=16),
                        ft.Text("Upload from the Upload tab", size=12, color=ft.colors.with_opacity(0.7, ft.colors.ON_SURFACE)),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=10,
                ),
                alignment=ft.alignment.center,
                expand=True,
            )
            
        # Create grid items
        grid_items = []
        for file_path in self.files:
            grid_items.append(self.build_grid_item(file_path))
            
        return ft.GridView(
            runs_count=3,
            child_aspect_ratio=0.8,
            spacing=10,
            run_spacing=10,
            children=grid_items,
        )
        
    def build_grid_item(self, file_path: str):
        """Build a single grid item."""
        filename = os.path.basename(file_path)
        is_video = self.app.file_handler.is_video(file_path)
        icon = ft.icons.VIDEO_FILE if is_video else ft.icons.IMAGE
        
        return ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Icon(icon, size=48, color=ft.colors.PRIMARY),
                        ft.Text(
                            filename[:15] + "..." if len(filename) > 15 else filename,
                            size=10,
                            text_align=ft.TextAlign.CENTER,
                        ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=5,
                ),
                padding=10,
                tooltip=filename,
                on_click=lambda _, p=file_path: self.select_file(p),
            ),
            elevation=2,
        )
        
    def load_files(self):
        """Load files based on current filter."""
        try:
            all_files = self.app.file_handler.list_media_files()
            
            if self.filter == "images":
                self.files = [f for f in all_files if self.app.file_handler.is_image(f)]
            elif self.filter == "videos":
                self.files = [f for f in all_files if self.app.file_handler.is_video(f)]
            else:
                self.files = all_files
                
        except Exception as e:
            print(f"[Gallery] Error loading files: {e}")
            self.files = []
            
    def set_filter(self, filter_type: str):
        """Set filter and refresh grid."""
        self.filter = filter_type
        # Rebuild the view
        self.app.page.views.clear()
        self.app.page.views.append(self.build())
        self.app.page.update()
        
    def select_file(self, file_path: str):
        """Select a file and navigate to inference screen."""
        self.selected_file = file_path
        self.app.current_file = file_path
        self.app.show_snackbar(f"Selected: {os.path.basename(file_path)}")
        self.app.go("/inference")
