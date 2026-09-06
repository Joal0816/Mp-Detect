# screens/model_manager_screen.py
import flet as ft
import os
import threading
from components.theme import Colors


class ModelManagerScreen:
    def __init__(self, app):
        self.app = app
        self.model_list = ft.Column(spacing=8, expand=True)

    def build_content(self) -> ft.Column:
        self.load_models()
        return ft.Column(
            [
                # Header
                ft.Container(
                    content=ft.Row([
                        ft.Text("Models", size=24, weight=ft.FontWeight.BOLD, color=Colors.TEXT_PRIMARY),
                        ft.Row([
                            ft.IconButton(
                                icon=ft.Icons.ADD, tooltip="Add from URL",
                                on_click=lambda: self.show_add_dialog(),
                                icon_color=Colors.ACCENT_CYAN,
                            ),
                            ft.IconButton(
                                icon=ft.Icons.FILE_UPLOAD, tooltip="Add from File",
                                on_click=lambda: self.show_add_from_file_dialog(),
                                icon_color=Colors.TEXT_SECONDARY,
                            ),
                        ], spacing=4),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=ft.Padding.symmetric(horizontal=24, vertical=16),
                ),

                # Active model card
                self._build_active_model_card(),

                # Model list
                ft.Container(
                    content=self.model_list,
                    padding=ft.Padding.symmetric(horizontal=24, vertical=8),
                    expand=True,
                ),
            ],
            spacing=0,
            expand=True,
        )

    def _build_active_model_card(self):
        active_id = self.app.model_manager.active_model_id
        engine = self.app.current_engine
        backend = "ONNX" if engine and not hasattr(engine, "interpreter") else "TFLite" if engine else "None"

        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Container(
                        content=ft.Icon(ft.Icons.CHECK_CIRCLE, size=16, color=Colors.SUCCESS),
                        bgcolor=ft.Colors.with_opacity(0.15, Colors.SUCCESS),
                        border_radius=12,
                        padding=4,
                    ),
                    ft.Text("Active Model", size=12, color=Colors.SUCCESS, weight=ft.FontWeight.W_500),
                ], spacing=6),
                ft.Text(active_id or "No model selected", size=14, color=Colors.TEXT_PRIMARY,
                       weight=ft.FontWeight.W_500),
                ft.Text(f"Backend: {backend}", size=11, color=Colors.TEXT_SECONDARY),
            ], spacing=6),
            bgcolor=Colors.BG_CARD,
            border=ft.Border.all(1, ft.Colors.with_opacity(0.3, Colors.SUCCESS)),
            border_radius=10,
            padding=16,
            margin=ft.Margin.symmetric(horizontal=24, vertical=8),
        )

    def load_models(self):
        self.model_list.controls.clear()
        models = self.app.model_manager.models
        active_id = self.app.model_manager.active_model_id

        if not models:
            self.model_list.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.Icons.MODEL_TRAINING, size=40, color=Colors.TEXT_MUTED),
                        ft.Text("No models found", size=14, color=Colors.TEXT_SECONDARY),
                        ft.Text("Add using + button above", size=11, color=Colors.TEXT_MUTED),
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
                    alignment=ft.Alignment.CENTER,
                    padding=40,
                )
            )
        else:
            for model in models:
                mid = model.get("id", "")
                self.model_list.controls.append(
                    self._build_model_card(model, mid == active_id)
                )

    def _build_model_card(self, model, is_active):
        mid = model.get("id", "")
        name = model.get("name", "")
        fmt = model.get("format", "").upper()
        path = model.get("path", "")

        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text(
                        f"{'★ ' if is_active else ''}{name}",
                        size=14, color=Colors.TEXT_PRIMARY,
                        weight=ft.FontWeight.BOLD if is_active else ft.FontWeight.NORMAL,
                    ),
                    ft.Container(
                        content=ft.Text(fmt, size=10, color=Colors.ACCENT_CYAN),
                        bgcolor=ft.Colors.with_opacity(0.1, Colors.ACCENT_CYAN),
                        border_radius=4,
                        padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                    ),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Text(f"ID: {mid}", size=10, color=Colors.TEXT_MUTED),
                ft.Row([
                    ft.Button("SELECT", visible=not is_active,
                              on_click=lambda _, m=mid: self.select_model(m),
                              style=ft.ButtonStyle(
                                  shape=ft.RoundedRectangleBorder(radius=6),
                                  bgcolor=Colors.ACCENT_CYAN if not is_active else Colors.GLASS_BG,
                                  color=Colors.BG_PRIMARY if not is_active else Colors.TEXT_MUTED,
                              ), height=30),
                    ft.Button("VALIDATE",
                              on_click=lambda _, p=path: self.validate_model(p),
                              style=ft.ButtonStyle(
                                  shape=ft.RoundedRectangleBorder(radius=6),
                                  side=ft.BorderSide(1, Colors.GLASS_BORDER),
                                  bgcolor=Colors.GLASS_BG,
                                  color=Colors.TEXT_SECONDARY,
                              ), height=30),
                    ft.IconButton(icon=ft.Icons.DELETE, visible=not is_active,
                                  on_click=lambda _, m=mid: self.delete_model(m),
                                  icon_color=Colors.ERROR, icon_size=18),
                ], spacing=6),
            ], spacing=6),
            bgcolor=Colors.BG_CARD,
            border=ft.Border.all(1, Colors.GLASS_BORDER_ACTIVE if is_active else Colors.GLASS_BORDER),
            border_radius=10,
            padding=14,
        )

    def show_add_dialog(self):
        url_field = ft.TextField(
            label="Model URL", hint_text="https://example.com/model.onnx",
            expand=True, bgcolor=Colors.BG_CARD, color=Colors.TEXT_PRIMARY,
            border_color=Colors.GLASS_BORDER, focused_border_color=Colors.ACCENT_CYAN,
        )
        name_field = ft.TextField(
            label="Model Name", hint_text="my_model",
            expand=True, bgcolor=Colors.BG_CARD, color=Colors.TEXT_PRIMARY,
            border_color=Colors.GLASS_BORDER, focused_border_color=Colors.ACCENT_CYAN,
        )
        format_dropdown = ft.Dropdown(
            label="Format",
            options=[ft.DropdownOption("onnx"), ft.DropdownOption("tflite")],
            value="onnx",
            bgcolor=Colors.BG_CARD, color=Colors.TEXT_PRIMARY,
        )
        labels_field = ft.TextField(
            label="Labels", value="HDPE,LDPE,PET,PP,PS,PVC",
            expand=True, bgcolor=Colors.BG_CARD, color=Colors.TEXT_PRIMARY,
            border_color=Colors.GLASS_BORDER, focused_border_color=Colors.ACCENT_CYAN,
        )
        progress_bar = ft.ProgressBar(visible=False, color=Colors.ACCENT_CYAN)

        def on_add(e):
            if not url_field.value or not name_field.value:
                self.app.show_snackbar("Please enter URL and name")
                return
            progress_bar.visible = True
            self.app.page.update()

            def download():
                try:
                    self.app.model_manager.add_model_from_url(
                        url=url_field.value, name=name_field.value,
                        format_=format_dropdown.value,
                        labels=[l.strip() for l in labels_field.value.split(",")],
                        set_active=True,
                    )
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
            title=ft.Text("Add Model from URL", color=Colors.TEXT_PRIMARY),
            content=ft.Container(
                content=ft.Column([url_field, name_field, format_dropdown, labels_field, progress_bar], spacing=10, width=320),
                bgcolor=Colors.BG_SURFACE,
            ),
            bgcolor=Colors.BG_CARD,
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: self.close_dialog(dialog)),
                ft.Button("Download", on_click=on_add,
                          bgcolor=Colors.ACCENT_CYAN, color=Colors.BG_PRIMARY),
            ],
        )
        self.app.page.overlay.append(dialog)
        dialog.open = True
        self.app.page.update()

    def close_dialog(self, dialog):
        dialog.open = False
        if dialog in self.app.page.overlay:
            self.app.page.overlay.remove(dialog)
        self.app.page.update()

    async def show_add_from_file_dialog(self):
        files = await self.app.file_picker.pick_files(
            dialog_title="Select Model File",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["onnx", "tflite"],
        )
        if files and len(files) > 0:
            file_path = files[0].path
            try:
                entry = self.app.model_manager.add_model_from_file(file_path, set_active=True)
                self.app.current_engine = self.app.model_manager.get_active_engine()
                self.load_models()
                self.app.show_snackbar(f"Model added: {entry['name']}")
            except Exception as ex:
                self.app.show_snackbar(f"Failed: {ex}")

    def select_model(self, mid):
        try:
            self.app.current_engine = self.app.model_manager.switch_model(mid)
            self.load_models()
            self.app.show_snackbar(f"Switched to {mid}")
        except Exception as e:
            self.app.show_snackbar(f"Failed: {e}")

    def validate_model(self, path):
        try:
            result = self.app.model_manager.validate_model_file(
                os.path.join(os.path.dirname(__file__), "..", path)
            )
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
