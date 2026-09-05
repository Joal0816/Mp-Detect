# screens/upload_screen.py - File picker screen
"""Upload screen for MP Detect Flet app."""
import flet as ft
import os


class UploadScreen:
    def __init__(self, app):
        self.app = app
        self.selected_file = app.current_file  # Restore from app state
        
    def build(self) -> ft.View:
        # File display
        self.file_display = self.build_selected_file_display()
        
        return ft.View(
            "/upload",
            [
                ft.AppBar(
                    title=ft.Text("Upload"),
                    bgcolor=ft.colors.SURFACE_VARIANT,
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            # Drop zone
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Icon(ft.icons.UPLOAD_FILE, size=48, color=ft.colors.PRIMARY),
                                        ft.Text("Tap to pick file", size=16),
                                        ft.Text("Images: png, jpg, jpeg, tif, bmp", size=12, color=ft.colors.with_opacity(0.7, ft.colors.ON_SURFACE)),
                                        ft.Text("Videos: mp4, avi, mov, mkv", size=12, color=ft.colors.with_opacity(0.7, ft.colors.ON_SURFACE)),
                                    ],
                                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                    spacing=8,
                                ),
                                width=300,
                                height=200,
                                border=ft.border.all(2, ft.colors.OUTLINE),
                                border_radius=10,
                                alignment=ft.alignment.center,
                                on_click=lambda _: self.pick_file(),
                            ),
                            
                            # Action buttons
                            ft.ElevatedButton(
                                "Pick from Gallery",
                                icon=ft.icons.PHOTO_LIBRARY,
                                on_click=lambda _: self.pick_file(),
                            ),
                            ft.ElevatedButton(
                                "Take Photo",
                                icon=ft.icons.CAMERA_ALT,
                                disabled=True,
                                tooltip="Camera not available in this version",
                            ),
                            
                            # Selected file display
                            self.file_display,
                            
                            # Recent files section
                            ft.Container(
                                content=self.build_recent_files(),
                                padding=ft.padding.only(top=20),
                            ),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=20,
                    ),
                    padding=20,
                    expand=True,
                ),
                # Bottom navigation bar
                self.app.nav_bar.build(),
            ],
        )
        
    def build_selected_file_display(self):
        """Build widget to display selected file."""
        if self.selected_file is None:
            return ft.Container()
            
        filename = os.path.basename(self.selected_file)
        return ft.Card(
            content=ft.Container(
                content=ft.Row(
                    [
                        ft.Icon(ft.icons.IMAGE, color=ft.colors.PRIMARY),
                        ft.Column(
                            [
                                ft.Text("Selected File", size=12, color=ft.colors.with_opacity(0.7, ft.colors.ON_SURFACE)),
                                ft.Text(filename, size=14, weight=ft.FontWeight.BOLD),
                            ],
                            spacing=4,
                        ),
                        ft.IconButton(
                            icon=ft.icons.ARROW_FORWARD,
                            on_click=lambda _: self.go_to_gallery(),
                        ),
                    ],
                    spacing=10,
                ),
                padding=10,
            ),
        )
        
    def build_recent_files(self):
        """Build recent files section."""
        try:
            files = self.app.file_handler.list_media_files()
            recent = files[:5] if files else []
        except Exception as e:
            print(f"[UploadScreen] Error listing recent files: {e}")
            recent = []
            
        if not recent:
            return ft.Container()
            
        return ft.Column(
            [
                ft.Text("Recent Files", size=14, weight=ft.FontWeight.BOLD),
                ft.Row(
                    [
                        self.build_recent_file_chip(f)
                        for f in recent
                    ],
                    wrap=True,
                    spacing=8,
                ),
            ],
            spacing=8,
        )
        
    def build_recent_file_chip(self, file_path):
        """Build a chip for a recent file."""
        filename = os.path.basename(file_path)
        is_video = self.app.file_handler.is_video(file_path)
        icon = ft.icons.VIDEO_FILE if is_video else ft.icons.IMAGE
        
        return ft.Chip(
            label=ft.Text(filename[:20], size=12),
            leading=ft.Icon(icon, size=16),
            tooltip=filename,
            on_click=lambda _, p=file_path: self.select_file(p),
        )
        
    def pick_file(self):
        """Open file picker dialog."""
        self.app.pick_file(
            dialog_title="Select Micrograph",
            allowed_extensions=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "mp4", "avi", "mov", "mkv"],
        )
        
    def _on_file_selected(self, file_path):
        """Handle file selection from file picker."""
        self.select_file(file_path)
        
    def select_file(self, file_path):
        """Select a file and update UI."""
        if file_path is None or not os.path.isfile(file_path):
            self.app.show_snackbar("Invalid file selected")
            return
            
        # Validate extension
        valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".mp4", ".avi", ".mov", ".mkv")
        if not file_path.lower().endswith(valid_exts):
            self.app.show_snackbar("Unsupported file format")
            return
            
        self.selected_file = file_path
        self.app.current_file = file_path
        self.app.show_snackbar(f"Selected: {os.path.basename(file_path)}")
        
        # Navigate to gallery
        self.go_to_gallery()
        
    def go_to_gallery(self):
        """Navigate to gallery screen with selected file."""
        self.app.go("/gallery")
