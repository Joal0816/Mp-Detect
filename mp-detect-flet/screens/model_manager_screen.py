# screens/model_manager_screen.py
import flet as ft
import os
import threading


class ModelManagerScreen:
    def __init__(self, app):
        self.app = app
        self.model_list = ft.ListView(spacing=10, padding=10, expand=True)

    def build_content(self) -> ft.Column:
        self.load_models()
        return ft.Column(
            [
                ft.AppBar(
                    title=ft.Text("Models"),
                    leading=ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=lambda: self.app.go("inference")),
                    actions=[ft.IconButton(icon=ft.Icons.ADD, on_click=lambda: self.show_add_dialog())],
                    bgcolor=ft.Colors.SURFACE,
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Card(content=ft.Container(content=ft.Column([
                                ft.Text("Active Model", size=14, weight=ft.FontWeight.BOLD),
                                ft.Text(f"Model: {self.app.model_manager.active_model_id or 'None'}", size=12),
                                ft.Text(f"Backend: {'ONNX' if self.app.current_engine and not hasattr(self.app.current_engine, 'interpreter') else 'TFLite' if self.app.current_engine else 'None'}", size=12),
                            ], spacing=5), padding=12)),
                            self.model_list,
                        ],
                        spacing=10,
                        expand=True,
                    ),
                    padding=10,
                    expand=True,
                ),
            ],
            spacing=0,
            expand=True,
        )

    def load_models(self):
        self.model_list.controls.clear()
        models = self.app.model_manager.models
        active_id = self.app.model_manager.active_model_id
        if not models:
            self.model_list.controls.append(ft.Container(content=ft.Column([
                ft.Icon(ft.Icons.MODEL_TRAINING, size=48, color=ft.Colors.with_opacity(0.5, ft.Colors.ON_SURFACE)),
                ft.Text("No models found", size=16),
                ft.Text("Add using + button", size=12, color=ft.Colors.with_opacity(0.7, ft.Colors.ON_SURFACE)),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10), alignment=ft.Alignment.CENTER, padding=20))
        else:
            for model in models:
                mid = model.get("id", "")
                self.model_list.controls.append(self.build_model_card(model, mid == active_id))

    def build_model_card(self, model, is_active):
        mid = model.get("id", "")
        name = model.get("name", "")
        fmt = model.get("format", "").upper()
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Row([ft.Text(f"{'* ' if is_active else ''}{name}", size=14, weight=ft.FontWeight.BOLD if is_active else ft.FontWeight.NORMAL), ft.Text(f"[{fmt}]", size=12, color=ft.Colors.CYAN)], spacing=5),
                    ft.Text(f"ID: {mid}", size=10, color=ft.Colors.with_opacity(0.7, ft.Colors.ON_SURFACE)),
                    ft.Row([
                        ft.ElevatedButton("SELECT", on_click=lambda _, m=mid: self.select_model(m), visible=not is_active),
                        ft.ElevatedButton("VALIDATE", on_click=lambda _, p=model.get("path", ""): self.validate_model(p)),
                        ft.IconButton(icon=ft.Icons.DELETE, on_click=lambda _, m=mid: self.delete_model(m), visible=not is_active),
                    ], spacing=5),
                ], spacing=5),
                padding=12,
            ),
        )

    def show_add_dialog(self):
        url_field = ft.TextField(label="Model URL", hint_text="https://example.com/model.onnx", expand=True)
        name_field = ft.TextField(label="Model Name", hint_text="my_model", expand=True)
        format_dropdown = ft.Dropdown(label="Format", options=[ft.dropdown.Option("onnx"), ft.dropdown.Option("tflite")], value="onnx")
        labels_field = ft.TextField(label="Labels", value="HDPE,LDPE,PET,PP,PS,PVC", expand=True)
        progress_bar = ft.ProgressBar(visible=False)

        def on_add(e):
            if not url_field.value or not name_field.value:
                self.app.show_snackbar("Please enter URL and name")
                return
            progress_bar.visible = True
            self.app.page.update()

            def download():
                try:
                    self.app.model_manager.add_model_from_url(url=url_field.value, name=name_field.value, format_=format_dropdown.value, labels=[l.strip() for l in labels_field.value.split(",")], set_active=True)
                    self.load_models()
                    dialog.open = False
                    self.app.page.update()
                    self.app.show_snackbar(f"Model added: {name_field.value}")
                except Exception as ex:
                    self.app.show_snackbar(f"Failed: {ex}")
                finally:
                    progress_bar.visible = False
                    self.app.page.update()

            threading.Thread(target=download, daemon=True).start()

        dialog = ft.AlertDialog(
            title=ft.Text("Add Model from URL"),
            content=ft.Container(content=ft.Column([url_field, name_field, format_dropdown, labels_field, progress_bar], spacing=10, width=300)),
            actions=[ft.TextButton("Cancel", on_click=lambda _: self.close_dialog(dialog)), ft.ElevatedButton("Download", on_click=on_add)],
        )
        self.app.page.overlay.append(dialog)
        dialog.open = True
        self.app.page.update()

    def close_dialog(self, dialog):
        dialog.open = False
        self.app.page.overlay.remove(dialog)
        self.app.page.update()

    def select_model(self, mid):
        try:
            self.app.current_engine = self.app.model_manager.switch_model(mid)
            self.load_models()
            self.app.show_snackbar(f"Switched to {mid}")
        except Exception as e:
            self.app.show_snackbar(f"Failed: {e}")

    def validate_model(self, path):
        try:
            result = self.app.model_manager.validate_model_file(os.path.join(os.path.dirname(__file__), "..", path))
            if result.get("valid"):
                self.app.show_snackbar(f"Valid: {result.get('input_shape')}")
            else:
                self.app.show_snackbar(f"Invalid: {'; '.join(result.get('errors', []))}")
        except Exception as e:
            self.app.show_snackbar(f"Error: {e}")

    def delete_model(self, mid):
        try:
            self.app.model_manager.remove_model(mid)
            self.load_models()
            self.app.show_snackbar(f"Deleted: {mid}")
        except Exception as e:
            self.app.show_snackbar(f"Failed: {e}")
