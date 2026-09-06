# components/layout.py - Responsive layout engine
import flet as ft


def get_layout_mode(page: ft.Page) -> str:
    """Detect layout mode based on window width."""
    width = page.window.width
    if width < 600:
        return "mobile"
    elif width < 1200:
        return "tablet"
    else:
        return "desktop"


class ResponsiveLayout:
    """Responsive layout that adapts to window size."""

    def __init__(self, app):
        self.app = app
        self.mode = "desktop"

    def build(self, left_sidebar=None, center_content=None, right_inspector=None) -> ft.Row:
        """Build responsive layout with optional panels."""
        self.mode = get_layout_mode(self.app.page)

        if self.mode == "mobile":
            return self._build_mobile(center_content)
        elif self.mode == "tablet":
            return self._build_tablet(left_sidebar, center_content, right_inspector)
        else:
            return self._build_desktop(left_sidebar, center_content, right_inspector)

    def _build_mobile(self, center_content) -> ft.Column:
        """Mobile: single column, everything stacked vertically."""
        controls = []
        if center_content:
            if isinstance(center_content, list):
                controls.extend(center_content)
            else:
                controls.append(center_content)
        return ft.Column(controls, spacing=0, expand=True)

    def _build_tablet(self, left, center, right) -> ft.Row:
        """Tablet: 2 columns (center + right)."""
        left_items = []
        center_items = []
        right_items = []

        if left:
            left_items.append(left)
        if center:
            if isinstance(center, list):
                center_items.extend(center)
            else:
                center_items.append(center)
        if right:
            right_items.append(right)

        # Center takes more space, right takes less
        center_col = ft.Container(
            content=ft.Column(center_items, spacing=0, expand=True),
            expand=True,
        )
        right_col = ft.Container(
            content=ft.Column(right_items, spacing=0),
            width=300,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
        )

        return ft.Row([center_col, right_col], spacing=0, expand=True)

    def _build_desktop(self, left, center, right) -> ft.Row:
        """Desktop: 3 panels (left sidebar + center + right inspector)."""
        left_items = []
        center_items = []
        right_items = []

        if left:
            left_items.append(left)
        if center:
            if isinstance(center, list):
                center_items.extend(center)
            else:
                center_items.append(center)
        if right:
            right_items.append(right)

        left_panel = ft.Container(
            content=ft.Column(left_items, spacing=0),
            width=220,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
        )
        center_panel = ft.Container(
            content=ft.Column(center_items, spacing=0, expand=True),
            expand=True,
        )
        right_panel = ft.Container(
            content=ft.Column(right_items, spacing=0),
            width=280,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
        )

        return ft.Row([left_panel, center_panel, right_panel], spacing=0, expand=True)


class LeftSidebar:
    """Left sidebar with media source and lighting presets (desktop only)."""

    def __init__(self, app):
        self.app = app

    def build(self) -> ft.Column:
        return ft.Column(
            [
                # Media Source Card
                ft.Card(
                    content=ft.Container(
                        content=ft.Column(
                            [
                                ft.Text("Media Source", size=14, weight=ft.FontWeight.BOLD),
                                ft.Row(
                                    [
                                        ft.Button("Open File", icon=ft.Icons.FOLDER_OPEN, on_click=lambda _: self.app.current_screen.pick_file() if hasattr(self.app.current_screen, 'pick_file') else None),
                                        ft.Button("Live Camera", icon=ft.Icons.CAMERA_ALT, on_click=lambda _: self._start_camera()),
                                    ],
                                    spacing=8,
                                ),
                            ],
                            spacing=8,
                        ),
                        padding=12,
                    ),
                ),
                # Lighting Presets Card
                ft.Card(
                    content=ft.Container(
                        content=ft.Column(
                            [
                                ft.Text("Lighting Presets", size=14, weight=ft.FontWeight.BOLD),
                                ft.Button("BLOF", on_click=lambda _: self._set_lighting("blof")),
                                ft.Button("UV 365nm", on_click=lambda _: self._set_lighting("uv_365")),
                                ft.Button("UV 395nm", on_click=lambda _: self._set_lighting("uv_395")),
                            ],
                            spacing=4,
                        ),
                        padding=12,
                    ),
                ),
            ],
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        )

    def _start_camera(self):
        self.app.show_snackbar("Camera started")

    def _set_lighting(self, preset):
        self.app.settings["lighting_mode"] = preset
        from core.settings_manager import save_settings
        save_settings(self.app.settings)
        self.app.show_snackbar(f"Lighting: {preset.upper()}")


class RightInspector:
    """Right panel with all controls (thresholds, counts, analytics, charts, calibration, export)."""

    def __init__(self, app):
        self.app = app
        self.stat_texts = {}
        self.conf_slider = None
        self.iou_slider = None

    def build(self) -> ft.Column:
        self.stat_texts = {
            "total": ft.Text("Total: 0", size=14),
            "avg_conf": ft.Text("Avg Confidence: 0.00", size=12),
            "pet": ft.Text("PET: 0", size=12), "hdpe": ft.Text("HDPE: 0", size=12),
            "pvc": ft.Text("PVC: 0", size=12), "ldpe": ft.Text("LDPE: 0", size=12),
            "pp": ft.Text("PP: 0", size=12), "ps": ft.Text("PS: 0", size=12),
        }
        self.conf_slider = ft.Slider(min=0.01, max=1.0, value=self.app.settings.get("conf", 0.25), label="Conf: {value}", on_change=self._on_conf_change, expand=True)
        self.iou_slider = ft.Slider(min=0.1, max=1.0, value=self.app.settings.get("iou", 0.45), label="IoU: {value}", on_change=self._on_iou_change, expand=True)

        return ft.Column(
            [
                self._build_thresholds_card(),
                self._build_particle_counts_card(),
                self._build_calibration_card(),
                self._build_export_card(),
            ],
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _build_thresholds_card(self) -> ft.Card:
        return ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text("Detection Thresholds", size=14, weight=ft.FontWeight.BOLD),
                        ft.Text("Confidence", size=12, color=ft.Colors.with_opacity(0.7, ft.Colors.ON_SURFACE)),
                        self.conf_slider,
                        ft.Text("IoU", size=12, color=ft.Colors.with_opacity(0.7, ft.Colors.ON_SURFACE)),
                        self.iou_slider,
                    ],
                    spacing=4,
                ),
                padding=12,
            ),
        )

    def _build_particle_counts_card(self) -> ft.Card:
        return ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text("Particle Counts", size=14, weight=ft.FontWeight.BOLD),
                        self.stat_texts["total"],
                        self.stat_texts["avg_conf"],
                        ft.Divider(),
                        ft.Row([self.stat_texts["pet"], self.stat_texts["hdpe"]], spacing=10),
                        ft.Row([self.stat_texts["pvc"], self.stat_texts["ldpe"]], spacing=10),
                        ft.Row([self.stat_texts["pp"], self.stat_texts["ps"]], spacing=10),
                    ],
                    spacing=4,
                ),
                padding=12,
            ),
        )

    def _build_calibration_card(self) -> ft.Card:
        return ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text("Calibration", size=14, weight=ft.FontWeight.BOLD),
                        ft.Row(
                            [
                                ft.Button("4x", on_click=lambda _: self._set_mag("4x")),
                                ft.Button("10x", on_click=lambda _: self._set_mag("10x")),
                                ft.Button("40x", on_click=lambda _: self._set_mag("40x")),
                                ft.Button("100x", on_click=lambda _: self._set_mag("100x")),
                            ],
                            spacing=4,
                        ),
                        ft.Row(
                            [
                                ft.Text("Scale (px/um):", size=12),
                                ft.TextField(value=str(self.app.settings.get("scale_factor", 0.0)), width=80, height=30, text_size=12, on_change=self._on_scale_change),
                                ft.Text("(0=off)", size=10, color=ft.Colors.with_opacity(0.5, ft.Colors.ON_SURFACE)),
                            ],
                            spacing=4,
                        ),
                    ],
                    spacing=8,
                ),
                padding=12,
            ),
        )

    def _build_export_card(self) -> ft.Card:
        return ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text("Export Data", size=14, weight=ft.FontWeight.BOLD),
                        ft.Row(
                            [
                                ft.Button("CSV", icon=ft.Icons.TABLE_CHART, on_click=lambda _: self._export("csv")),
                                ft.Button("JSON", icon=ft.Icons.CODE, on_click=lambda _: self._export("json")),
                            ],
                            spacing=8,
                        ),
                        ft.Row(
                            [
                                ft.Button("Image", icon=ft.Icons.IMAGE, on_click=lambda _: self._export("image")),
                                ft.Button("Video", icon=ft.Icons.VIDEO_FILE, on_click=lambda _: self._export("video")),
                            ],
                            spacing=8,
                        ),
                    ],
                    spacing=8,
                ),
                padding=12,
            ),
        )

    def _on_conf_change(self, e):
        from core.settings_manager import save_settings
        self.app.settings["conf"] = float(e.control.value)
        save_settings(self.app.settings)

    def _on_iou_change(self, e):
        from core.settings_manager import save_settings
        self.app.settings["iou"] = float(e.control.value)
        save_settings(self.app.settings)

    def _on_scale_change(self, e):
        try:
            from core.settings_manager import save_settings
            self.app.settings["scale_factor"] = float(e.control.value)
            save_settings(self.app.settings)
        except ValueError:
            pass

    def _set_mag(self, mag):
        presets = {"4x": 0.15, "10x": 0.40, "40x": 1.60, "100x": 4.00}
        sf = presets.get(mag, 0.0)
        self.app.settings["scale_factor"] = sf
        self.app.settings["magnification"] = mag
        from core.settings_manager import save_settings
        save_settings(self.app.settings)
        self.app.show_snackbar(f"Magnification: {mag} ({sf} px/um)")

    def _export(self, fmt):
        if fmt == "csv":
            try:
                from core.export import export_csv
                path = self.app.file_handler.get_export_csv_path()
                export_csv(self.app.current_results, path, self.app.settings)
                self.app.show_snackbar(f"CSV saved: {path}")
            except Exception as e:
                self.app.show_snackbar(f"Export failed: {e}")
        elif fmt == "json":
            try:
                from core.export import export_json_report
                path = self.app.file_handler.get_export_report_path()
                export_json_report(self.app.current_results, path, self.app.settings, self.app.current_file, "Flet Backend")
                self.app.show_snackbar(f"Report saved: {path}")
            except Exception as e:
                self.app.show_snackbar(f"Export failed: {e}")
        elif fmt == "image":
            self.app.show_snackbar("Image export triggered")
        elif fmt == "video":
            self.app.show_snackbar("Video export triggered")

    def update_stats(self, stats: dict):
        """Update particle counts from detection results."""
        total = stats.get("total", 0)
        self.stat_texts["total"].value = f"Total: {total}"
        self.stat_texts["avg_conf"].value = f"Avg Confidence: {stats.get('avg_conf', 0):.2f}"
        for cls in ["pet", "hdpe", "pvc", "ldpe", "pp", "ps"]:
            count = stats.get("per_class", {}).get(cls.upper(), {}).get("count", 0)
            self.stat_texts[cls].value = f"{cls.upper()}: {count}"
