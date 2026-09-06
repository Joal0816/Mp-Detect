# screens/result_screen.py
import flet as ft
import cv2
import base64
import os


class ResultScreen:
    def __init__(self, app):
        self.app = app
        self.image_display = ft.Image(src="", width=400, height=350, fit="contain")

        # Stats display
        self.stat_total = ft.Text("Total: 0", size=14, weight=ft.FontWeight.BOLD)
        self.stat_avg = ft.Text("Avg Confidence: 0.00", size=12)
        self.stat_pet = ft.Text("PET: 0", size=12)
        self.stat_hdpe = ft.Text("HDPE: 0", size=12)
        self.stat_pvc = ft.Text("PVC: 0", size=12)
        self.stat_ldpe = ft.Text("LDPE: 0", size=12)
        self.stat_pp = ft.Text("PP: 0", size=12)
        self.stat_ps = ft.Text("PS: 0", size=12)

    def build_content(self) -> ft.Column:
        self.load_results()
        return ft.Column(
            [
                ft.AppBar(
                    title=ft.Text("Results"),
                    leading=ft.IconButton(icon=ft.Icons.ARROW_BACK,
                                         on_click=lambda: self.app.go("inference")),
                    bgcolor=ft.Colors.SURFACE,
                ),
                # Annotated image
                ft.Container(
                    content=self.image_display,
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.2, ft.Colors.OUTLINE)),
                    border_radius=8,
                    height=360,
                    padding=8,
                ),
                # Stats card
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Detection Results", size=14, weight=ft.FontWeight.BOLD),
                            self.stat_total,
                            self.stat_avg,
                            ft.Divider(),
                            ft.Row([self.stat_pet, self.stat_hdpe], spacing=10),
                            ft.Row([self.stat_pvc, self.stat_ldpe], spacing=10),
                            ft.Row([self.stat_pp, self.stat_ps], spacing=10),
                        ], spacing=4),
                        padding=12,
                    ),
                ),
                # Class distribution chart
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Particle Distribution", size=14, weight=ft.FontWeight.BOLD),
                            self._build_class_chart(),
                        ], spacing=8),
                        padding=12,
                    ),
                ),
                # Confidence distribution chart
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Confidence Distribution", size=14, weight=ft.FontWeight.BOLD),
                            self._build_confidence_chart(),
                        ], spacing=8),
                        padding=12,
                    ),
                ),
                # Export buttons
                ft.Container(
                    content=ft.Column([
                        ft.Text("Export", size=14, weight=ft.FontWeight.BOLD),
                        ft.Row([
                            ft.ElevatedButton("CSV", icon=ft.Icons.TABLE_CHART,
                                             on_click=lambda _: self._export("csv")),
                            ft.ElevatedButton("JSON", icon=ft.Icons.CODE,
                                             on_click=lambda _: self._export("json")),
                            ft.ElevatedButton("Image", icon=ft.Icons.IMAGE,
                                             on_click=lambda _: self._export("image")),
                            ft.ElevatedButton("Video", icon=ft.Icons.VIDEO_FILE,
                                             on_click=lambda _: self._export("video")),
                        ], spacing=8),
                    ], spacing=8),
                    padding=12,
                ),
            ],
            spacing=0,
            expand=True,
        )

    def load_results(self):
        # Use pre-computed annotated image if available
        if self.app.current_annotated is not None:
            try:
                _, buf = cv2.imencode('.jpg', self.app.current_annotated)
                b64 = base64.b64encode(buf).decode()
                self.image_display.src = f"data:image/jpeg;base64,{b64}"
            except Exception as e:
                print(f"[Result] Error encoding annotated image: {e}")
        elif self.app.current_file is not None:
            # Fallback: draw boxes on original image
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
            except Exception as e:
                print(f"[Result] Error: {e}")

        # Update stats display
        stats = self.app.current_stats
        if stats:
            total = stats.get("total", 0)
            self.stat_total.value = f"Total: {total}"
            self.stat_avg.value = f"Avg Confidence: {stats.get('avg_conf', 0):.2f}"
            for cls in ["pet", "hdpe", "pvc", "ldpe", "pp", "ps"]:
                count = stats.get("per_class", {}).get(cls.upper(), {}).get("count", 0)
                getattr(self, f"stat_{cls}").value = f"{cls.upper()}: {count}"

    def _build_class_chart(self):
        """Build bar chart showing particle count per class."""
        stats = self.app.current_stats
        if not stats:
            return ft.Container(content=ft.Text("No data"), height=150)

        per_class = stats.get("per_class", {})
        data = []
        colors = [ft.Colors.RED, ft.Colors.CYAN, ft.Colors.MAGENTA,
                  ft.Colors.YELLOW, ft.Colors.GREEN, ft.Colors.BLUE]

        for i, (cls, info) in enumerate(per_class.items()):
            count = info.get("count", 0) if isinstance(info, dict) else 0
            if count > 0:
                data.append(ft.BarChartGroup(
                    x=i,
                    bar_charts=[ft.BarChartRod(
                        to_y=count,
                        color=colors[i % len(colors)],
                        width=20,
                    )],
                ))

        max_y = max((info.get("count", 0) if isinstance(info, dict) else 0
                     for info in per_class.values()), default=10)
        if max_y == 0:
            max_y = 10

        return ft.BarChart(
            expand=True,
            bar_groups=data,
            max_y=max_y,
            left_axis=ft.ChartAxis(labels_size=40),
            bottom_axis=ft.ChartAxis(
                labels=[ft.ChartAxisLabel(label=cls, rotate=-45)
                        for cls in per_class.keys()],
                labels_size=60,
            ),
            height=200,
        )

    def _build_confidence_chart(self):
        """Build bar chart showing confidence distribution."""
        results = self.app.current_results
        if not results:
            return ft.Container(content=ft.Text("No data"), height=150)

        # Bucket confidence values
        buckets = {"0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for _, _, conf in results:
            if conf < 0.2:
                buckets["0-0.2"] += 1
            elif conf < 0.4:
                buckets["0.2-0.4"] += 1
            elif conf < 0.6:
                buckets["0.4-0.6"] += 1
            elif conf < 0.8:
                buckets["0.6-0.8"] += 1
            else:
                buckets["0.8-1.0"] += 1

        data = []
        colors = [ft.Colors.RED_300, ft.Colors.ORANGE_300, ft.Colors.YELLOW_300,
                  ft.Colors.LIGHT_GREEN_300, ft.Colors.GREEN_300]

        for i, (label, count) in enumerate(buckets.items()):
            if count > 0:
                data.append(ft.BarChartGroup(
                    x=i,
                    bar_charts=[ft.BarChartRod(
                        to_y=count,
                        color=colors[i % len(colors)],
                        width=20,
                    )],
                ))

        max_y = max(buckets.values(), default=10)
        if max_y == 0:
            max_y = 10

        return ft.BarChart(
            expand=True,
            bar_groups=data,
            max_y=max_y,
            left_axis=ft.ChartAxis(labels_size=40),
            bottom_axis=ft.ChartAxis(
                labels=[ft.ChartAxisLabel(label=label, rotate=-45)
                        for label in buckets.keys()],
                labels_size=60,
            ),
            height=200,
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
