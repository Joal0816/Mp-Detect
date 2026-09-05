# screens/inference_screen.py - Detection view screen with responsive layout
import flet as ft
import cv2
import base64
import threading
from components.layout import ResponsiveLayout, LeftSidebar, RightInspector, get_layout_mode


class InferenceScreen:
    def __init__(self, app):
        self.app = app
        self.image_display = ft.Image(width=400, height=400, fit=ft.ImageFit.CONTAIN)
        self.is_processing = False
        self.detect_button = None
        self.view_results_button = None
        self.hud_fps = ft.Text("FPS: --", color=ft.Colors.CYAN, size=12)
        self.hud_latency = ft.Text("Latency: --ms", color=ft.Colors.CYAN, size=12)
        self.hud_backend = ft.Text("", color=ft.Colors.GREEN, size=12)
        self.left_sidebar = LeftSidebar(app)
        self.right_inspector = RightInspector(app)

        # Camera state
        self._cap = None
        self._inf_active = False
        self._inf_thread_busy = False
        self._last_frame = None

    def build(self) -> ft.View:
        self.detect_button = ft.ElevatedButton("Detect", icon=ft.Icons.PLAY_ARROW, on_click=lambda _: self.toggle_detection(), bgcolor=ft.Colors.PRIMARY, color=ft.Colors.WHITE, width=120)
        self.view_results_button = ft.ElevatedButton("View Results", icon=ft.Icons.VISIBILITY, on_click=lambda _: self.app.go("/result"), visible=False)

        # Load image if file selected
        if self.app.current_file:
            self._load_image(self.app.current_file)

        # Update backend badge
        if self.app.current_engine:
            fmt = "TFLite" if hasattr(self.app.current_engine, "interpreter") else "ONNX"
            self.hud_backend.value = f"[{fmt}]"

        # Center content
        center_content = ft.Column(
            [
                # Image viewport
                ft.Container(
                    content=ft.Stack(
                        [
                            self.image_display,
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Icon(ft.Icons.MICROSCOPE, size=48, color=ft.Colors.with_opacity(0.3, ft.Colors.ON_SURFACE)),
                                        ft.Text("No Micrograph Loaded", size=16, color=ft.Colors.with_opacity(0.5, ft.Colors.ON_SURFACE)),
                                        ft.Text("Click 'Open File' or press Ctrl+O", size=12, color=ft.Colors.with_opacity(0.3, ft.Colors.ON_SURFACE)),
                                    ],
                                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                    spacing=8,
                                ),
                                alignment=ft.Alignment.CENTER,
                                visible=self.image_display.src_base64 is None,
                            ),
                        ],
                        expand=True,
                    ),
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.2, ft.Colors.OUTLINE)),
                    border_radius=8,
                    expand=True,
                ),
                # HUD row
                ft.Container(
                    content=ft.Row([self.hud_fps, self.hud_latency, self.hud_backend], spacing=16),
                    padding=ft.padding.symmetric(horizontal=8, vertical=4),
                ),
                # Toolbar
                ft.Container(
                    content=ft.Row(
                        [
                            self.detect_button,
                            ft.ElevatedButton("Snapshot", icon=ft.Icons.CAMERA, on_click=lambda _: self.take_snapshot()),
                            ft.ElevatedButton("Record", icon=ft.Icons.FIBER_MANUAL_RECORD, on_click=lambda _: self.toggle_recording()),
                            self.view_results_button,
                        ],
                        spacing=8,
                    ),
                    padding=8,
                    bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
                    border_radius=6,
                ),
            ],
            spacing=0,
            expand=True,
        )

        # Build responsive layout
        layout = ResponsiveLayout(self.app)
        content = layout.build(
            left_sidebar=self.left_sidebar.build(),
            center_content=center_content,
            right_inspector=self.right_inspector.build(),
        )

        return ft.View(
            "/inference",
            [
                ft.AppBar(
                    title=ft.Text("Inference"),
                    leading=ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=lambda _: self.app.go("/gallery")),
                    bgcolor=ft.Colors.SURFACE,
                ),
                ft.Container(content=content, expand=True),
                self.app.nav_bar.build() if hasattr(self.app, 'nav_bar') else ft.Container(),
            ],
        )

    def _load_image(self, file_path):
        try:
            if self.app.file_handler.is_video(file_path):
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    _, buf = cv2.imencode('.jpg', frame)
                    self.image_display.src_base64 = base64.b64encode(buf).decode('utf-8')
            else:
                frame = cv2.imread(file_path)
                if frame is not None:
                    _, buf = cv2.imencode('.jpg', frame)
                    self.image_display.src_base64 = base64.b64encode(buf).decode('utf-8')
        except Exception as e:
            print(f"[Inference] Error loading image: {e}")

    def pick_file(self):
        """Open file picker."""
        file_picker = ft.FilePicker()
        self.app.page.overlay.append(file_picker)
        self.app.page.update()
        result = file_picker.pick_files(
            dialog_title="Select Micrograph",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "mp4", "avi", "mov", "mkv"],
        )
        self.app.page.overlay.remove(file_picker)
        self.app.page.update()
        if result and len(result) > 0:
            self.app.current_file = result[0].path
            self._load_image(result[0].path)
            self.app.show_snackbar(f"Loaded: {result[0].path.split('/')[-1]}")

    def toggle_detection(self):
        if self.app.current_file is None:
            self.app.show_snackbar("No file selected")
            return
        if self.app.current_engine is None:
            self.app.show_snackbar("No model loaded")
            return
        if self.is_processing:
            return
        self._run_detection()

    def _run_detection(self):
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

            results = self.app.current_engine.detect(frame, conf_thresh=conf, iou_thresh=iou)
            from core.analytics import compute_stats
            from core.vision import draw_boxes
            stats = compute_stats(results)
            annotated = draw_boxes(frame, results)
            self.app.current_results = results
            self._update_ui(stats, annotated)
        except Exception as e:
            print(f"[Inference] Detection error: {e}")
            self.app.show_snackbar(f"Detection failed: {e}")
        finally:
            self.is_processing = False
            self.detect_button.disabled = False
            self.app.page.update()

    def _update_ui(self, stats, annotated):
        try:
            _, buf = cv2.imencode('.jpg', annotated)
            self.image_display.src_base64 = base64.b64encode(buf).decode('utf-8')
            self.right_inspector.update_stats(stats)
            self.view_results_button.visible = True
            self.app.page.update()
            self.app.show_snackbar(f"Found {stats.get('total', 0)} particles")
        except Exception as e:
            print(f"[Inference] UI error: {e}")

    def take_snapshot(self):
        if self.image_display.src_base64 is None:
            self.app.show_snackbar("No image to snapshot")
            return
        self.app.show_snackbar("Snapshot saved")

    def toggle_recording(self):
        self.app.show_snackbar("Recording toggled")

    def start_camera(self):
        """Start live camera feed."""
        try:
            self._cap = cv2.VideoCapture(0)
            if self._cap.isOpened():
                self._inf_active = True
                threading.Thread(target=self._camera_loop, daemon=True).start()
                self.app.show_snackbar("Camera started")
            else:
                self.app.show_snackbar("Cannot open camera")
        except Exception as e:
            self.app.show_snackbar(f"Camera error: {e}")

    def stop_camera(self):
        self._inf_active = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self.app.show_snackbar("Camera stopped")

    def _camera_loop(self):
        import time
        fps_buf = []
        while self._inf_active and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break
            self._last_frame = frame
            t0 = time.perf_counter()

            if self.app.current_engine and not self._inf_thread_busy:
                self._inf_thread_busy = True
                threading.Thread(target=self._camera_infer, args=(frame.copy(), t0), daemon=True).start()

            _, buf = cv2.imencode('.jpg', frame)
            self.image_display.src_base64 = base64.b64encode(buf).decode('utf-8')

            now = time.perf_counter()
            fps_buf.append(now)
            fps_buf = [t for t in fps_buf if now - t < 1.0]
            self.hud_fps.value = f"FPS: {len(fps_buf)}"

            try:
                self.app.page.update()
            except Exception:
                break
            time.sleep(0.03)

    def _camera_infer(self, frame, t0):
        import time
        try:
            results = self.app.current_engine.detect(frame, conf_thresh=self.app.settings.get("conf", 0.25), iou_thresh=self.app.settings.get("iou", 0.45))
            elapsed_ms = (time.perf_counter() - t0) * 1000
            from core.vision import draw_boxes
            annotated = draw_boxes(frame, results)
            _, buf = cv2.imencode('.jpg', annotated)
            self.image_display.src_base64 = base64.b64encode(buf).decode('utf-8')
            self.hud_latency.value = f"Latency: {elapsed_ms:.1f}ms"
            from core.analytics import compute_stats
            self.right_inspector.update_stats(compute_stats(results))
        except Exception as e:
            print(f"[Camera] Infer error: {e}")
        finally:
            self._inf_thread_busy = False
