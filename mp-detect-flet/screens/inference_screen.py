# screens/inference_screen.py
import flet as ft
import cv2
import base64
import threading
import time
from collections import deque
from components.theme import Colors, card, glass_container, metric_card


class InferenceScreen:
    def __init__(self, app):
        self.app = app
        PLACEHOLDER = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQq0AAAAAElFTkSuQmCC"
        self.image_display = ft.Image(
            src=PLACEHOLDER, fit=ft.BoxFit.CONTAIN,
            visible=False, border_radius=8,
        )
        self.is_processing = False
        self.detect_button = None
        self.view_results_button = None
        self._has_image = False

        # HUD
        self.hud_fps = ft.Text("FPS: --", color=Colors.ACCENT_CYAN, size=11,
                               font_family="monospace")
        self.hud_latency = ft.Text("Latency: --ms", color=Colors.ACCENT_CYAN, size=11,
                                   font_family="monospace")
        self.hud_backend = ft.Text("", color=Colors.SUCCESS, size=11,
                                   font_family="monospace")

        # Camera state
        self.camera_active = False
        self.cap = None
        self.detect_running = False
        self._camera_thread = None
        self._frame_times = deque(maxlen=120)

        # Camera controls
        self.start_camera_button = None
        self.stop_camera_button = None
        self.detect_live_button = None

        # Stats display
        self.stat_total = ft.Text("0", size=24, weight=ft.FontWeight.BOLD, color=Colors.ACCENT_CYAN)
        self.stat_avg = ft.Text("0.00", size=14, color=Colors.TEXT_SECONDARY)
        self.stat_pet = ft.Text("PET: 0", size=12, color=Colors.TEXT_PRIMARY)
        self.stat_hdpe = ft.Text("HDPE: 0", size=12, color=Colors.TEXT_PRIMARY)
        self.stat_pvc = ft.Text("PVC: 0", size=12, color=Colors.TEXT_PRIMARY)
        self.stat_ldpe = ft.Text("LDPE: 0", size=12, color=Colors.TEXT_PRIMARY)
        self.stat_pp = ft.Text("PP: 0", size=12, color=Colors.TEXT_PRIMARY)
        self.stat_ps = ft.Text("PS: 0", size=12, color=Colors.TEXT_PRIMARY)

    def build_content(self) -> ft.Column:
        self.detect_button = ft.Button(
            "Detect", icon=ft.Icons.PLAY_ARROW,
            on_click=self.toggle_detection,
            bgcolor=Colors.ACCENT_CYAN, color=Colors.BG_PRIMARY,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
            width=140, height=40,
        )
        self.view_results_button = ft.Button(
            "Results", icon=ft.Icons.VISIBILITY,
            on_click=lambda _: self.app.go("result"),
            visible=False,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                side=ft.BorderSide(1, Colors.GLASS_BORDER),
                bgcolor=Colors.GLASS_BG,
                color=Colors.TEXT_PRIMARY,
            ),
            height=40,
        )

        # Camera controls
        self.start_camera_button = ft.Button(
            "Start Camera", icon=ft.Icons.CAMERA_ALT,
            on_click=lambda _: self.start_camera(),
            bgcolor=Colors.SUCCESS, color=Colors.BG_PRIMARY,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
            height=40,
        )
        self.stop_camera_button = ft.Button(
            "Stop Camera", icon=ft.Icons.STOP,
            on_click=lambda _: self.stop_camera(),
            bgcolor=Colors.ERROR, color=ft.Colors.WHITE,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
            visible=False, height=40,
        )
        self.detect_live_button = ft.Button(
            "Start Detection", icon=ft.Icons.PLAY_ARROW,
            on_click=lambda _: self.toggle_live_detection(),
            bgcolor=Colors.ACCENT_CYAN, color=Colors.BG_PRIMARY,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
            visible=False, height=40,
        )

        if self.app.current_file:
            self._load_image(self.app.current_file)

        if self.app.current_engine:
            fmt = "TFLite" if hasattr(self.app.current_engine, "interpreter") else "ONNX"
            self.hud_backend.value = f"[{fmt}]"

        return ft.Column(
            [
                # Header
                ft.Container(
                    content=ft.Row([
                        ft.Text("Detect", size=24, weight=ft.FontWeight.BOLD, color=Colors.TEXT_PRIMARY),
                        ft.Text("Real-time Micrograph Analysis", size=14, color=Colors.TEXT_SECONDARY),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=ft.Padding.symmetric(horizontal=24, vertical=16),
                ),

                # Main content
                ft.Row(
                    [
                        # Left: Image + HUD
                        ft.Column(
                            [
                                # Image display with HUD overlay
                                ft.Container(
                                    content=ft.Stack([
                                        self.image_display,
                                        # HUD overlay
                                        ft.Container(
                                            content=ft.Row([
                                                self.hud_fps,
                                                ft.Container(width=1, height=12, bgcolor=Colors.GLASS_BORDER),
                                                self.hud_latency,
                                                ft.Container(width=1, height=12, bgcolor=Colors.GLASS_BORDER),
                                                self.hud_backend,
                                            ], spacing=8),
                                            bgcolor=ft.Colors.with_opacity(0.7, Colors.BG_CARD),
                                            border_radius=6,
                                            padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                                            top=8, left=8,
                                        ),
                                        # Placeholder
                                        ft.Container(
                                            content=ft.Column([
                                                ft.Container(
                                                    content=ft.Icon(ft.Icons.SCIENCE, size=40, color=Colors.ACCENT_CYAN),
                                                    bgcolor=ft.Colors.with_opacity(0.1, Colors.ACCENT_CYAN),
                                                    border_radius=50, padding=16,
                                                ),
                                                ft.Text("No Micrograph Loaded", size=16, color=Colors.TEXT_PRIMARY),
                                                ft.Text("Open a file or start camera", size=12, color=Colors.TEXT_SECONDARY),
                                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
                                            alignment=ft.Alignment.CENTER,
                                            visible=not self._has_image and not self.camera_active,
                                        ),
                                    ], expand=True),
                                    border=ft.Border.all(1, Colors.GLASS_BORDER),
                                    border_radius=10,
                                    bgcolor=Colors.BG_CARD,
                                    expand=True,
                                ),
                                # Controls
                                ft.Container(
                                    content=ft.Row([
                                        # File mode controls
                                        ft.Row([
                                            self.detect_button,
                                            ft.Button("Open File", icon=ft.Icons.FOLDER_OPEN,
                                                      on_click=self._open_file,
                                                      style=ft.ButtonStyle(
                                                          shape=ft.RoundedRectangleBorder(radius=8),
                                                          side=ft.BorderSide(1, Colors.GLASS_BORDER),
                                                          bgcolor=Colors.GLASS_BG,
                                                          color=Colors.TEXT_PRIMARY,
                                                      ), height=40),
                                            self.view_results_button,
                                        ], spacing=8, visible=not self.camera_active),
                                        # Camera mode controls
                                        ft.Row([
                                            self.start_camera_button,
                                            self.stop_camera_button,
                                            self.detect_live_button,
                                        ], spacing=8, visible=self.camera_active),
                                    ], alignment=ft.MainAxisAlignment.CENTER),
                                    padding=ft.Padding.symmetric(vertical=8),
                                ),
                            ],
                            expand=True,
                            spacing=0,
                        ),

                        # Right: Stats panel (desktop only)
                        ft.Container(
                            content=self._build_stats_panel(),
                            width=240,
                            visible=self.app.is_desktop if hasattr(self.app, 'is_desktop') else True,
                        ),
                    ],
                    spacing=16,
                    expand=True,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
            ],
            spacing=0,
            expand=True,
        )

    def _build_stats_panel(self):
        """Build the right stats panel."""
        return ft.Container(
            content=ft.Column([
                # Total count - hero metric
                ft.Container(
                    content=ft.Column([
                        ft.Text("Total", size=12, color=Colors.TEXT_SECONDARY),
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
                        ft.Text("Per Class", size=12, color=Colors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
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
                # Model info
                ft.Container(
                    content=ft.Column([
                        ft.Text("Model", size=12, color=Colors.TEXT_SECONDARY, weight=ft.FontWeight.W_500),
                        ft.Text(self.app.model_manager.active_model_id or "None",
                               size=12, color=Colors.TEXT_PRIMARY),
                        self.hud_backend,
                    ], spacing=4),
                    bgcolor=Colors.BG_CARD,
                    border=ft.Border.all(1, Colors.GLASS_BORDER),
                    border_radius=10,
                    padding=12,
                ),
            ], spacing=8, expand=True),
            padding=ft.Padding(top=0),
        )

    async def _open_file(self, e=None):
        files = await self.app.file_picker.pick_files(
            dialog_title="Select Micrograph",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["png", "jpg", "jpeg", "tif", "tiff", "bmp",
                               "mp4", "avi", "mov", "mkv"],
        )
        if files and len(files) > 0:
            self._load_image(files[0].path)
            self.app.current_file = files[0].path

    def _load_image(self, file_path):
        try:
            if self.app.file_handler.is_video(file_path):
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    _, buf = cv2.imencode('.jpg', frame)
                    b64 = base64.b64encode(buf).decode()
                    self.image_display.src = f"data:image/jpeg;base64,{b64}"
                    self.image_display.visible = True
                    self._has_image = True
            else:
                frame = cv2.imread(file_path)
                if frame is not None:
                    _, buf = cv2.imencode('.jpg', frame)
                    b64 = base64.b64encode(buf).decode()
                    self.image_display.src = f"data:image/jpeg;base64,{b64}"
                    self.image_display.visible = True
                    self._has_image = True
        except Exception as e:
            print(f"[Inference] Error: {e}")

    def toggle_detection(self, e=None):
        if self.app.current_file is None:
            self.app.show_snackbar("No file selected")
            return
        if self.app.current_engine is None:
            self.app.show_snackbar("No model loaded — add a model in Models tab")
            return
        if self.is_processing:
            return
        self.is_processing = True
        self.detect_button.disabled = True
        self.app.page.update()
        threading.Thread(target=self._detection_thread, daemon=True).start()

    def _detection_thread(self):
        try:
            conf = self.app.settings.get("conf", 0.25)
            iou = self.app.settings.get("iou", 0.45)
            if self.app.file_handler.is_video(self.app.current_file):
                cap = cv2.VideoCapture(self.app.current_file)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise RuntimeError("Failed to read video")
            else:
                frame = cv2.imread(self.app.current_file)
                if frame is None:
                    raise RuntimeError("Failed to read image")

            t0 = time.time()
            results = self.app.current_engine.detect(frame, conf_thresh=conf, iou_thresh=iou)
            latency = (time.time() - t0) * 1000

            from core.analytics import compute_stats
            from core.vision import draw_boxes
            stats = compute_stats(results)
            annotated = draw_boxes(frame, results)

            self.app.current_results = results
            self.app.current_annotated = annotated
            self.app.current_stats = stats
            self.app.current_frame = frame

            _, buf = cv2.imencode('.jpg', annotated)
            b64 = base64.b64encode(buf).decode()
            self.image_display.src = f"data:image/jpeg;base64,{b64}"
            self.image_display.visible = True

            # Update stats
            total = stats.get("total", 0)
            self.stat_total.value = str(total)
            self.stat_avg.value = f"Avg: {stats.get('avg_conf', 0):.2f}"
            for cls in ["pet", "hdpe", "pvc", "ldpe", "pp", "ps"]:
                count = stats.get("per_class", {}).get(cls.upper(), {}).get("count", 0)
                getattr(self, f"stat_{cls}").value = f"{cls.upper()}: {count}"

            self.hud_latency.value = f"Latency: {latency:.0f}ms"
            self.view_results_button.visible = True
            self.detect_button.disabled = False
            self.is_processing = False
            self.app.page.update()
            self.app.show_snackbar(f"Found {total} particles")
        except Exception as e:
            print(f"[Inference] Error: {e}")
            self.detect_button.disabled = False
            self.is_processing = False
            self.app.page.update()
            self.app.show_snackbar(f"Detection failed: {e}")

    def start_camera(self):
        if self.camera_active:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.app.show_snackbar("Cannot open camera")
            return
        self.camera_active = True
        self.start_camera_button.visible = False
        self.stop_camera_button.visible = True
        self.detect_live_button.visible = True
        self._has_image = True
        self.app.page.update()
        self._camera_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._camera_thread.start()

    def _capture_loop(self):
        while self.camera_active and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            now = time.time()
            self._frame_times.append(now)
            fps = sum(1 for t in self._frame_times if now - t < 1.0)

            if self.detect_running and self.app.current_engine:
                try:
                    conf = self.app.settings.get("conf", 0.25)
                    iou = self.app.settings.get("iou", 0.45)
                    t0 = time.time()
                    results = self.app.current_engine.detect(frame, conf_thresh=conf, iou_thresh=iou)
                    latency = (time.time() - t0) * 1000
                    from core.vision import draw_boxes
                    from core.analytics import compute_stats
                    display_frame = draw_boxes(frame, results)
                    stats = compute_stats(results)

                    total = stats.get("total", 0)
                    self.stat_total.value = str(total)
                    self.stat_avg.value = f"Avg: {stats.get('avg_conf', 0):.2f}"
                    for cls in ["pet", "hdpe", "pvc", "ldpe", "pp", "ps"]:
                        count = stats.get("per_class", {}).get(cls.upper(), {}).get("count", 0)
                        getattr(self, f"stat_{cls}").value = f"{cls.upper()}: {count}"
                    self.hud_latency.value = f"Latency: {latency:.0f}ms"
                except Exception as e:
                    print(f"[Camera] Detection error: {e}")
                    display_frame = frame
            else:
                display_frame = frame

            _, buf = cv2.imencode('.jpg', display_frame)
            b64 = base64.b64encode(buf).decode()
            self.image_display.src = f"data:image/jpeg;base64,{b64}"
            self.image_display.visible = True
            self.hud_fps.value = f"FPS: {fps}"

            try:
                self.app.page.update()
            except Exception:
                break

            time.sleep(0.033)

    def toggle_live_detection(self):
        if self.app.current_engine is None:
            self.app.show_snackbar("No model loaded")
            return
        self.detect_running = not self.detect_running
        if self.detect_running:
            self.detect_live_button.text = "Stop Detection"
            self.detect_live_button.icon = ft.Icons.STOP
            self.detect_live_button.bgcolor = Colors.WARNING
        else:
            self.detect_live_button.text = "Start Detection"
            self.detect_live_button.icon = ft.Icons.PLAY_ARROW
            self.detect_live_button.bgcolor = Colors.ACCENT_CYAN
        self.app.page.update()

    def stop_camera(self):
        self.camera_active = False
        self.detect_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_camera_button.visible = True
        self.stop_camera_button.visible = False
        self.detect_live_button.visible = False
        self.detect_live_button.text = "Start Detection"
        self.detect_live_button.icon = ft.Icons.PLAY_ARROW
        self.detect_live_button.bgcolor = Colors.ACCENT_CYAN
        self.hud_fps.value = "FPS: --"
        self.hud_latency.value = "Latency: --ms"
        try:
            self.app.page.update()
        except Exception:
            pass

    def cleanup(self):
        self.stop_camera()
