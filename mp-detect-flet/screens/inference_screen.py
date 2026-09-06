# screens/inference_screen.py
import flet as ft
import cv2
import base64
import threading
import time
from collections import deque


class InferenceScreen:
    def __init__(self, app):
        self.app = app
        self.image_display = ft.Image(src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQq0AAAAAElFTkSuQmCC", width=400, height=400, fit="contain", visible=False)
        self.is_processing = False
        self.detect_button = None
        self.view_results_button = None
        self.hud_fps = ft.Text("FPS: --", color=ft.Colors.CYAN, size=12)
        self.hud_latency = ft.Text("Latency: --ms", color=ft.Colors.CYAN, size=12)
        self.hud_backend = ft.Text("", color=ft.Colors.GREEN, size=12)
        self._has_image = False

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
        self.stat_total = ft.Text("Total: 0", size=14, weight=ft.FontWeight.BOLD)
        self.stat_avg = ft.Text("Avg Confidence: 0.00", size=12)
        self.stat_pet = ft.Text("PET: 0", size=12)
        self.stat_hdpe = ft.Text("HDPE: 0", size=12)
        self.stat_pvc = ft.Text("PVC: 0", size=12)
        self.stat_ldpe = ft.Text("LDPE: 0", size=12)
        self.stat_pp = ft.Text("PP: 0", size=12)
        self.stat_ps = ft.Text("PS: 0", size=12)

    def build_content(self) -> ft.Column:
        self.detect_button = ft.Button(
            "Detect", icon=ft.Icons.PLAY_ARROW,
            on_click=self.toggle_detection,
            bgcolor=ft.Colors.CYAN, color=ft.Colors.WHITE, width=120
        )
        self.view_results_button = ft.Button(
            "View Results", icon=ft.Icons.VISIBILITY,
            on_click=lambda _: self.app.go("result"), visible=False
        )

        # Camera controls
        self.start_camera_button = ft.Button(
            "Start Camera", icon=ft.Icons.CAMERA_ALT,
            on_click=lambda _: self.start_camera(),
            bgcolor=ft.Colors.GREEN, color=ft.Colors.WHITE
        )
        self.stop_camera_button = ft.Button(
            "Stop Camera", icon=ft.Icons.STOP,
            on_click=lambda _: self.stop_camera(),
            bgcolor=ft.Colors.RED, color=ft.Colors.WHITE,
            visible=False
        )
        self.detect_live_button = ft.Button(
            "Start Detection", icon=ft.Icons.PLAY_ARROW,
            on_click=lambda _: self.toggle_live_detection(),
            bgcolor=ft.Colors.CYAN, color=ft.Colors.WHITE,
            visible=False
        )

        if self.app.current_file:
            self._load_image(self.app.current_file)

        if self.app.current_engine:
            fmt = "TFLite" if hasattr(self.app.current_engine, "interpreter") else "ONNX"
            self.hud_backend.value = f"[{fmt}]"

        return ft.Column(
            [
                ft.AppBar(title=ft.Text("Detect"), bgcolor=ft.Colors.SURFACE),
                # Image display
                ft.Container(
                    content=ft.Stack([
                        self.image_display,
                        ft.Container(
                            content=ft.Column([
                                ft.Icon(ft.Icons.SCIENCE, size=48,
                                        color=ft.Colors.with_opacity(0.3, ft.Colors.ON_SURFACE)),
                                ft.Text("No Micrograph Loaded", size=16,
                                        color=ft.Colors.with_opacity(0.5, ft.Colors.ON_SURFACE)),
                                ft.Text("Click 'Open File' or 'Start Camera'", size=12,
                                        color=ft.Colors.with_opacity(0.3, ft.Colors.ON_SURFACE)),
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
                            alignment=ft.Alignment.CENTER,
                            visible=not self._has_image and not self.camera_active,
                        ),
                    ], expand=True),
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.2, ft.Colors.OUTLINE)),
                    border_radius=8,
                    height=300,
                    padding=8,
                ),
                # HUD row
                ft.Container(
                    content=ft.Row([self.hud_fps, self.hud_latency, self.hud_backend], spacing=16),
                    padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                ),
                # File mode buttons
                ft.Container(
                    content=ft.Row([
                        self.detect_button,
                        ft.Button("Open File", icon=ft.Icons.FOLDER_OPEN,
                                  on_click=self._open_file),
                        self.view_results_button,
                    ], spacing=8),
                    padding=8,
                    visible=not self.camera_active,
                ),
                # Camera mode buttons
                ft.Container(
                    content=ft.Row([
                        self.start_camera_button,
                        self.stop_camera_button,
                        self.detect_live_button,
                    ], spacing=8),
                    padding=8,
                    visible=self.camera_active,
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
            ],
            spacing=0,
            expand=True,
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

    # ── File-based detection ──────────────────────────────────────────

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

            # Update stats display
            total = stats.get("total", 0)
            self.stat_total.value = f"Total: {total}"
            self.stat_avg.value = f"Avg Confidence: {stats.get('avg_conf', 0):.2f}"
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

    # ── Live camera detection ─────────────────────────────────────────

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

            # FPS calculation
            now = time.time()
            self._frame_times.append(now)
            # Count frames in the last second
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

                    # Update stats
                    total = stats.get("total", 0)
                    self.stat_total.value = f"Total: {total}"
                    self.stat_avg.value = f"Avg Confidence: {stats.get('avg_conf', 0):.2f}"
                    for cls in ["pet", "hdpe", "pvc", "ldpe", "pp", "ps"]:
                        count = stats.get("per_class", {}).get(cls.upper(), {}).get("count", 0)
                        getattr(self, f"stat_{cls}").value = f"{cls.upper()}: {count}"

                    self.hud_latency.value = f"Latency: {latency:.0f}ms"
                except Exception as e:
                    print(f"[Camera] Detection error: {e}")
                    display_frame = frame
            else:
                display_frame = frame

            # Convert to base64 for Flet display
            _, buf = cv2.imencode('.jpg', display_frame)
            b64 = base64.b64encode(buf).decode()
            self.image_display.src = f"data:image/jpeg;base64,{b64}"
            self.image_display.visible = True
            self.hud_fps.value = f"FPS: {fps}"

            try:
                self.app.page.update()
            except Exception:
                break

            time.sleep(0.033)  # Target ~30 FPS

    def toggle_live_detection(self):
        if self.app.current_engine is None:
            self.app.show_snackbar("No model loaded — add a model in Models tab")
            return
        self.detect_running = not self.detect_running
        if self.detect_running:
            self.detect_live_button.text = "Stop Detection"
            self.detect_live_button.icon = ft.Icons.STOP
            self.detect_live_button.bgcolor = ft.Colors.ORANGE
        else:
            self.detect_live_button.text = "Start Detection"
            self.detect_live_button.icon = ft.Icons.PLAY_ARROW
            self.detect_live_button.bgcolor = ft.Colors.CYAN
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
        self.detect_live_button.bgcolor = ft.Colors.CYAN
        self.hud_fps.value = "FPS: --"
        self.hud_latency.value = "Latency: --ms"
        try:
            self.app.page.update()
        except Exception:
            pass

    def cleanup(self):
        """Call when switching away from this screen."""
        self.stop_camera()
