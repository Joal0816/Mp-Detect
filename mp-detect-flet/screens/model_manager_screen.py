# screens/model_manager_screen.py - Model management screen
"""Model Manager screen for MP Detect Flet app."""
import flet as ft
import os
import threading


class ModelManagerScreen:
    def __init__(self, app):
        self.app = app
        self.model_list = None
        
    def build(self) -> ft.View:
        # Model list
        self.model_list = ft.ListView(
            spacing=10,
            padding=10,
        )
        
        # Load models
        self.load_models()
        
        return ft.View(
            "/models",
            [
                ft.AppBar(
                    title=ft.Text("Models"),
                    leading=ft.IconButton(
                        icon=ft.icons.ARROW_BACK,
                        on_click=lambda _: self.app.go("/inference"),
                    ),
                    actions=[
                        ft.IconButton(
                            icon=ft.icons.ADD,
                            on_click=lambda _: self.show_add_dialog(),
                        ),
                    ],
                    bgcolor=ft.colors.SURFACE_VARIANT,
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            # Active model info
                            self.build_active_model_info(),
                            
                            # Model list
                            self.model_list,
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
        
    def build_active_model_info(self):
        """Build active model info card."""
        active_model = self.app.model_manager.active_model_id
        engine_badge = "No Engine"
        
        if self.app.current_engine:
            fmt = "tflite" if hasattr(self.app.current_engine, "interpreter") else "onnx"
            provider = getattr(self.app.current_engine, "provider_display", "CPU")
            engine_badge = f"{fmt.upper()} {provider}"
            
        return ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text("Active Model", size=14, weight=ft.FontWeight.BOLD),
                        ft.Text(f"Model: {active_model or 'None'}", size=12),
                        ft.Text(f"Backend: {engine_badge}", size=12),
                    ],
                    spacing=5,
                ),
                padding=10,
            ),
        )
        
    def load_models(self):
        """Load models from registry."""
        self.model_list.controls.clear()
        
        models = self.app.model_manager.models
        active_id = self.app.model_manager.active_model_id
        
        if not models:
            self.model_list.controls.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Icon(ft.icons.MODEL_TRAINING, size=48, color=ft.colors.with_opacity(0.5, ft.colors.ON_SURFACE)),
                            ft.Text("No models found", size=16),
                            ft.Text("Add a model using the + button", size=12, color=ft.colors.with_opacity(0.7, ft.colors.ON_SURFACE)),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=10,
                    ),
                    alignment=ft.alignment.center,
                    padding=20,
                )
            )
        else:
            for model in models:
                self.model_list.controls.append(
                    self.build_model_card(model, model["id"] == active_id)
                )
                
        self.app.page.update()
        
    def build_model_card(self, model: dict, is_active: bool):
        """Build a model card."""
        model_id = model.get("id", "Unknown")
        model_name = model.get("name", "Unknown")
        model_format = model.get("format", "").upper()
        
        return ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Text(
                                    f"{'★ ' if is_active else ''}{model_name}",
                                    size=14,
                                    weight=ft.FontWeight.BOLD if is_active else ft.FontWeight.NORMAL,
                                ),
                                ft.Text(
                                    f"[{model_format}]",
                                    size=12,
                                    color=ft.colors.PRIMARY,
                                ),
                            ],
                            spacing=5,
                        ),
                        ft.Text(f"ID: {model_id}", size=10, color=ft.colors.with_opacity(0.7, ft.colors.ON_SURFACE)),
                        ft.Row(
                            [
                                ft.ElevatedButton(
                                    "SELECT",
                                    on_click=lambda _, mid=model_id: self.select_model(mid),
                                    visible=not is_active,
                                ),
                                ft.ElevatedButton(
                                    "VALIDATE",
                                    on_click=lambda _, mid=model_id: self.validate_model(mid),
                                ),
                                ft.IconButton(
                                    icon=ft.icons.DELETE,
                                    on_click=lambda _, mid=model_id: self.delete_model(mid),
                                    visible=not is_active,
                                ),
                            ],
                            spacing=5,
                        ),
                    ],
                    spacing=5,
                ),
                padding=10,
            ),
        )
        
    def show_add_dialog(self):
        """Show add model dialog."""
        # URL input
        url_field = ft.TextField(
            label="Model URL",
            hint_text="https://example.com/model.onnx",
            expand=True,
        )
        
        # Name input
        name_field = ft.TextField(
            label="Model Name",
            hint_text="my_model",
            expand=True,
        )
        
        # Format dropdown
        format_dropdown = ft.Dropdown(
            label="Format",
            options=[
                ft.dropdown.Option("onnx"),
                ft.dropdown.Option("tflite"),
            ],
            value="onnx",
        )
        
        # Labels input
        labels_field = ft.TextField(
            label="Labels (comma-separated)",
            value="HDPE,LDPE,PET,PP,PS,PVC",
            expand=True,
        )
        
        # Progress indicator
        progress_bar = ft.ProgressBar(visible=False)
        progress_text = ft.Text("", visible=False)
        
        def on_add(e):
            url = url_field.value
            name = name_field.value
            fmt = format_dropdown.value
            labels = [l.strip() for l in labels_field.value.split(",") if l.strip()]
            
            if not url or not name:
                self.app.show_snackbar("Please enter URL and name")
                return
                
            # Show progress
            progress_bar.visible = True
            progress_text.visible = True
            progress_text.value = "Downloading model..."
            self.app.page.update()
            
            def download_thread():
                try:
                    self.app.model_manager.add_model_from_url(
                        url=url,
                        name=name,
                        format_=fmt,
                        labels=labels,
                        set_active=True,
                    )
                    
                    # Update UI
                    self.load_models()
                    self.app.page.dialog.open = False
                    self.app.show_snackbar(f"Model added: {name}")
                    
                except Exception as ex:
                    self.app.show_snackbar(f"Failed to add model: {ex}")
                finally:
                    progress_bar.visible = False
                    progress_text.visible = False
                    self.app.page.update()
                    
            threading.Thread(target=download_thread, daemon=True).start()
            
        # Create dialog
        self.app.page.dialog = ft.AlertDialog(
            title=ft.Text("Add Model from URL"),
            content=ft.Container(
                content=ft.Column(
                    [
                        url_field,
                        name_field,
                        format_dropdown,
                        labels_field,
                        progress_bar,
                        progress_text,
                    ],
                    spacing=10,
                    width=300,
                ),
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: self.close_dialog()),
                ft.ElevatedButton("Download", on_click=on_add),
            ],
        )
        
        self.app.page.dialog.open = True
        self.app.page.update()
        
    def close_dialog(self):
        """Close the add dialog."""
        self.app.page.dialog.open = False
        self.app.page.update()
        
    def select_model(self, model_id: str):
        """Select a model as active."""
        try:
            self.app.current_engine = self.app.model_manager.switch_model(model_id)
            self.load_models()
            self.app.show_snackbar(f"Switched to {model_id}")
        except Exception as e:
            self.app.show_snackbar(f"Failed to switch model: {e}")
            
    def validate_model(self, model_id: str):
        """Validate a model."""
        try:
            info = self.app.model_manager._find_model(model_id)
            if info:
                path = os.path.join(os.path.dirname(__file__), "..", info["path"])
                result = self.app.model_manager.validate_model_file(path)
                
                if result.get("valid", False):
                    self.app.show_snackbar(f"Model is valid: {result.get('input_shape')}")
                else:
                    errors = "; ".join(result.get("errors", ["Unknown error"]))
                    self.app.show_snackbar(f"Validation failed: {errors}")
            else:
                self.app.show_snackbar("Model not found")
        except Exception as e:
            self.app.show_snackbar(f"Validation error: {e}")
            
    def delete_model(self, model_id: str):
        """Delete a model."""
        try:
            self.app.model_manager.remove_model(model_id)
            self.load_models()
            self.app.show_snackbar(f"Deleted model: {model_id}")
        except Exception as e:
            self.app.show_snackbar(f"Failed to delete model: {e}")
