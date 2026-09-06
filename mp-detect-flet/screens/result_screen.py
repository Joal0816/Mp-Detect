# screens/result_screen.py
import flet as ft
import cv2
import base64
import os
from components.theme import Colors


class ResultScreen:
    def __init__(self, app):
        self.app = app
        PLACEHOLDER = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQq0AAAAAElFTkSuQmCC"
        self.image_display = ft.Image(
            src=PLACEHOLDER, fit=ft.BoxFit.CONTAIN,
            visible=False, border_radius=8,
        )

        # Stats
        self.stat_total = ft.Text("0", size=28, weight=ft.FontWeight.BOLD, color=Colors.ACCENT_CYAN)
        self.stat_avg = ft.Text("0.00", size=14, color=Colors.TEXT_SECONDARY)
        self.stat_pet = ft.Text("PET: 0", size=12, color=Colors.TEXT_PRIMARY)
        self.stat_hdpe = ft.Text("HDPE: 0", size=12, color=Colors.TEXT_PRIMARY)
        self.stat_pvc = ft.Text("PVC: 0", size=12, color=Colors.TEXT_PRIMARY)
        self.stat_ldpe = ft.Text("LDPE: 0", size=12, color=Colors.TEXT_PRIMARY)
        self.stat_pp = ft.Text("PP: 0", size=12, color=Colors.TEXT_PRIMARY)
        self.stat_ps = ft.Text("PS: 0", size=12, color=Colors.TEXT_PRIMARY)

    def build_content(self) -> ft.Column:
        self.load_results()
        return ft.Column(
            [
                # Header with back button
                ft.Container(
                    content=ft.Row([
                        ft.IconButton(
                            icon=ft.Icons.ARROW_BACK,
                            on_click=lambda: self.app.go("inference"),
                            icon_color=Colors.TEXT_SECONDARY,
                        ),
                        ft.Text("Results", size=24, weight=ft.FontWeight.BOLD, color=Colors.TEXT_PRIMARY),
                        ft.Row([
                            ft.IconButton(icon=ft.Icons.SHARE, icon_color=Colors.TEXT_SECONDARY, tooltip="Share"),
                            ft.IconButton(icon=ft.Icons.DOWNLOAD, icon_color=Colors.TEXT_SECONDARY, tooltip="Export"),
                        ], spacing=4),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=ft.Padding.symmetric(horizontal=16, vertical=8),
                ),

                # Main content - split view on desktop
                ft.Row(
                    [
                        # Left: Annotated image
                        ft.Container(
                            content=self.image_display,
                            bgcolor=Colors.BG_CARD,
                            border=ft.Border.all(1, Colors.GLASS_BORDER),
                            border_radius=10,
                            padding=8,
                            expand=True,
                        ),

                        # Right: Stats + Charts
                        ft.Container(
                            content=ft.Column([
                                # Total count card
                                ft.Container(
                                    content=ft.Column([
                                        ft.Text("Total Particles", size=12, color=Colors.TEXT_SECONDARY),
                                        self.stat_total,
                                        self.stat_avg,
                                    ], spacing=2),
                                    bgcolor=Colors.BG_CARD,
                                    border=ft.Border.all(1, Colors.GLASS_BORDER),
                                    border_radius=10,
                                    padding=16,
                                ),

                                # Per-class breakdown
                                ft.Container(
                                    content=ft.Column([
                                        ft.Text("Per Class", size=12, color=Colors.TEXT_SECONDARY,
                                               weight=ft.FontWeight.W_500),
                                        ft.Divider(height=1, color=Colors.GLASS_BORDER),
                                        ft.Row([self.stat_pet, self.stat_hdpe], spacing=8),
                                        ft.Row([self.stat_pvc, self.stat_ldpe], spacing=8),
                                        ft.Row([self.stat_pp, self.stat_ps], spacing=8),
                                    ], spacing=6),
                                    bgcolor=Colors.BG_CARD,
                                    border=ft.Border.all(1, Colors.GLASS_BORDER),
                                    border_radius=10,
                                    padding=12,
                                ),

                                # Class distribution chart
                                ft.Container(
                                    content=ft.Column([
                                        ft.Text("Distribution", size=12, color=Colors.TEXT_SECONDARY,
                                               weight=ft.FontWeight.W_500),
                                        self._build_class_chart(),
                                    ], spacing=8),
                                    bgcolor=Colors.BG_CARD,
                                    border=ft.Border.all(1, Colors.GLASS_BORDER),
                                    border_radius=10,
                                    padding=12,
                                ),

                                # Confidence chart
                                ft.Container(
                                    content=ft.Column([
                                        ft.Text("Confidence", size=12, color=Colors.TEXT_SECONDARY,
                                               weight=ft.FontWeight.W_500),
                                        self._build_confidence_chart(),
                                    ], spacing=8),
                                    bgcolor=Colors.BG_CARD,
                                    border=ft.Border.all(1, Colors.GLASS_BORDER),
                                    border_radius=10,
                                    padding=12,
                                ),
                            ], spacing=8, expand=True),
                            width=280 if hasattr(self.app, "is_desktop") and self.app.is_desktop else None,
                        ),
                    ],
                    spacing=16,
                    expand=True,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),

                # Export buttons
                ft.Container(
                    content=ft.Row([
                        self._export_button("CSV", ft.Icons.TABLE_CHART, "csv"),
                        self._export_button("JSON", ft.Icons.CODE, "json"),
                        self._export_button("Image", ft.Icons.IMAGE, "image"),
                        self._export_button("Video", ft.Icons.VIDEO_FILE, "video"),
                    ], spacing=8, alignment=ft.MainAxisAlignment.CENTER),
                    padding=ft.Padding.symmetric(vertical=12),
                ),
            ],
            spacing=0,
            expand=True,
        )

    def _export_button(self, label, icon, fmt):
        return ft.Button(
            label, icon=icon,
            on_click=lambda _, f=fmt: self._export(f),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                side=ft.BorderSide(1, Colors.GLASS_BORDER),
                bgcolor=Colors.GLASS_BG,
                color=Colors.TEXT_SECONDARY,
            ),
            height=36,
        )

    def load_results(self):
        if self.app.current_annotated is not None:
            try:
                _, buf = cv2.imencode('.jpg', self.app.current_annotated)
                b64 = base64.b64encode(buf).decode()
                self.image_display.src = f"data:image/jpeg;base64,{b64}"
                self.image_display.visible = True
            except Exception as e:
                print(f"[Result] Error: {e}")
        elif self.app.current_file is not None:
            from core.vision import draw_boxes
            try:
                if self.app.file_handler.is_video(self.app.current_file):
                    cap = cv2.VideoCapture(self.app.current_file)
                    ret, frame = cap.read()
                    cap.release()
                else:
                    frame = cv2.imread(self.app.current_file)

                if frame is not None and self.app.current_results:
                    annotated = draw_boxes(frame, self.app.current_results)
                    _, buf = cv2.imencode('.jpg', annotated)
                    b64 = base64.b64encode(buf).decode()
                    self.image_display.src = f"data:image/jpeg;base64,{b64}"
                    self.image_display.visible = True
            except Exception as e:
                print(f"[Result] Error: {e}")

        stats = self.app.current_stats
        if stats:
            self.stat_total.value = str(stats.get("total", 0))
            self.stat_avg.value = f"Avg: {stats.get('avg_conf', 0):.2f}"
            for cls in ["pet", "hdpe", "pvc", "ldpe", "pp", "ps"]:
                count = stats.get("per_class", {}).get(cls.upper(), {}).get("count", 0)
                getattr(self, f"stat_{cls}").value = f"{cls.upper()}: {count}"

    def _build_class_chart(self):
        stats = self.app.current_stats
        if not stats:
            return ft.Container(content=ft.Text("No data", color=Colors.TEXT_MUTED), height=80)

        per_class = stats.get("per_class", {})
        data = []
        colors = [Colors.ERROR, Colors.ACCENT_CYAN, "#E040FB", Colors.WARNING, Colors.SUCCESS, Colors.ACCENT_PURPLE]

        for i, (cls, info) in enumerate(per_class.items()):
            count = info.get("count", 0) if isinstance(info, dict) else 0
            if count > 0:
                data.append(ft.BarChartGroup(
                    x=i,
                    bar_charts=[ft.BarChartRod(to_y=count, color=colors[i % len(colors)], width=16, border_radius=4)],
                ))

        max_y = max((info.get("count", 0) if isinstance(info, dict) else 0 for info in per_class.values()), default=10)
        if max_y == 0:
            max_y = 10

        return ft.BarChart(
            expand=True, bar_groups=data, max_y=max_y,
            left_axis=ft.ChartAxis(labels_size=30),
            bottom_axis=ft.ChartAxis(
                labels=[ft.ChartAxisLabel(label=cls, rotate=-45) for cls in per_class.keys()],
                labels_size=50,
            ),
            height=120,
        )

    def _build_confidence_chart(self):
        results = self.app.current_results
        if not results:
            return ft.Container(content=ft.Text("No data", color=Colors.TEXT_MUTED), height=80)

        buckets = {"0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for _, _, conf in results:
            if conf < 0.2: buckets["0-0.2"] += 1
            elif conf < 0.4: buckets["0.2-0.4"] += 1
            elif conf < 0.6: buckets["0.4-0.6"] += 1
            elif conf < 0.8: buckets["0.6-0.8"] += 1
            else: buckets["0.8-1.0"] += 1

        data = []
        colors = ["#FF5252", "#FF9800", "#FFC107", "#8BC34A", "#4CAF50"]
        for i, (label, count) in enumerate(buckets.items()):
            if count > 0:
                data.append(ft.BarChartGroup(
                    x=i,
                    bar_charts=[ft.BarChartRod(to_y=count, color=colors[i % len(colors)], width=16, border_radius=4)],
                ))

        max_y = max(buckets.values(), default=10)
        if max_y == 0:
            max_y = 10

        return ft.BarChart(
            expand=True, bar_groups=data, max_y=max_y,
            left_axis=ft.ChartAxis(labels_size=30),
            bottom_axis=ft.ChartAxis(
                labels=[ft.ChartAxisLabel(label=label, rotate=-45) for label in buckets.keys()],
                labels_size=50,
            ),
            height=120,
        )

    def _export(self, fmt):
        try:
            if fmt == "csv":
                from core.export import export_csv
                path = self.app.file_handler.get_export_csv_path()
                export_csv(self.app.current_results, path, self.app.settings)
                self.app.show_snackbar(f"CSV saved: {os.path.basename(path)}")
            elif fmt == "json":
                from core.export import export_json_report
                path = self.app.file_handler.get_export_report_path()
                export_json_report(self.app.current_results, path, self.app.settings,
                                  self.app.current_file, "Flet Backend")
                self.app.show_snackbar(f"Report saved: {os.path.basename(path)}")
            elif fmt == "image":
                from core.export import export_annotated_image
                if self.app.current_frame is not None and self.app.current_results:
                    path = self.app.file_handler.get_annotated_image_path()
                    export_annotated_image(self.app.current_frame, self.app.current_results, path)
                    self.app.show_snackbar(f"Image saved: {os.path.basename(path)}")
                else:
                    self.app.show_snackbar("No image to export")
            elif fmt == "video":
                from core.export import export_annotated_video
                if self.app.current_file and self.app.current_results:
                    if self.app.file_handler.is_video(self.app.current_file):
                        path = self.app.file_handler.get_annotated_video_path()
                        export_annotated_video(self.app.current_file, self.app.current_results,
                                              path, self.app.settings)
                        self.app.show_snackbar(f"Video saved: {os.path.basename(path)}")
                    else:
                        self.app.show_snackbar("Source is not a video")
                else:
                    self.app.show_snackbar("No video to export")
        except Exception as e:
            self.app.show_snackbar(f"Export failed: {e}")
