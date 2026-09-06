# screens/field_test_screen.py - Primary field-testing screen
"""Camera-first field testing screen for microplastic detection."""
import flet as ft
import cv2
import base64
import threading
import time
from collections import deque
from components.theme import Colors


class FieldTestScreen:
    """Dedicated field-testing screen with camera-first design."""
    
    def __init__(self, app):
        self.app = app
        self._lock = threading.Lock()
        
        # Session state
        self.session_active = False
        self.session_id = None
        self.detection_count = 0
        
        # Camera state
        self.camera_active = False
        self.cap = None
        self.detect_running = False
        self._camera_thread = None
        self._frame_times = deque(maxlen=120)
        
        # UI elements
        self.image_display = ft.Image(
            src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQq0AAAAAElFTkSuQmCC",
            fit=ft.BoxFit.CONTAIN, visible=False, border_radius=8,
        )
        
        # HUD
        self.hud_fps = ft.Text("FPS: --", color=Colors.ACCENT_CYAN, size=11, font_family="monospace")
        self.hud_latency = ft.Text("Latency: --ms", color=Colors.ACCENT_CYAN, size=11, font_family="monospace")
        self.hud_backend = ft.Text("", color=Colors.SUCCESS, size=11, font_family="monospace")
        
        # Session info
        self.session_id_text = ft.Text("No active session", size=12, color=Colors.TEXT_SECONDARY)
        self.location_text = ft.Text("Location: --", size=11, color=Colors.TEXT_MUTED)
        self.particle_count = ft.Text("0", size=28, weight=ft.FontWeight.BOLD, color=Colors.ACCENT_CYAN)
        
        # Controls
        self.start_session_button = None
        self.stop_session_button = None
        self.capture_button = None
        self.detect_toggle = None

    def build_content(self) -> ft.Column:
        self.start_session_button = ft.Button(
            "Start Session", icon=ft.Icons.PLAY_CIRCLE,
            on_click=lambda _: self.start_session(),
            bgcolor=Colors.SUCCESS, color=Colors.BG_PRIMARY,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
            height=44, expand=True,
        )
        self.stop_session_button = ft.Button(
            "Stop Session", icon=ft.Icons.STOP_CIRCLE,
            on_click=lambda _: self.stop_session(),
            bgcolor=Colors.ERROR, color=ft.Colors.WHITE,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
            visible=False, height=44, expand=True,
        )
        self.capture_button = ft.Button(
            "Capture", icon=ft.Icons.CAMERA,
            on_click=lambda _: self.capture_frame(),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                side=ft.BorderSide(1, Colors.GLASS_BORDER),
                bgcolor=Colors.GLASS_BG, color=Colors.TEXT_PRIMARY,
            ),
            visible=False, height=40, expand=True,
        )
        self.detect_toggle = ft.Button(
            "Auto-Detect", icon=ft.Icons.AUTO_FIX_HIGH,
            on_click=lambda _: self.toggle_detection(),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                side=ft.BorderSide(1, Colors.GLASS_BORDER),
                bgcolor=Colors.GLASS_BG, color=Colors.TEXT_PRIMARY,
            ),
            visible=False, height=40, expand=True,
        )

        return ft.Column(
            [
                # Header
                ft.Container(
                    content=ft.Row([
                        ft.Column([
                            ft.Text("Field Test", size=24, weight=ft.FontWeight.BOLD, color=Colors.TEXT_PRIMARY),
                            self.session_id_text,
                        ], spacing=2),
                        ft.Row([
                            ft.Icon(ft.Icons.INFO_OUTLINE, size=16, color=Colors.TEXT_SECONDARY),
                            self.location_text,
                        ], spacing=4),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=ft.Padding.symmetric(horizontal=24, vertical=12),
                ),

                # Camera view (takes most of the screen)
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
                        # Particle count overlay
                        ft.Container(
                            content=ft.Row([
                                ft.Icon(ft.Icons.BUBBLE_CHART, size=20, color=Colors.ACCENT_CYAN),
                                self.particle_count,
                                ft.Text("particles", size=12, color=Colors.TEXT_SECONDARY),
                            ], spacing=4),
                            bgcolor=ft.Colors.with_opacity(0.7, Colors.BG_CARD),
                            border_radius=6,
                            padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                            top=8, right=8,
                        ),
                        # Placeholder
                        ft.Container(
                            content=ft.Column([
                                ft.Container(
                                    content=ft.Icon(ft.Icons.VIDEOCAM, size=48, color=Colors.TEXT_MUTED),
                                    bgcolor=Colors.GLASS_BG, border_radius=50, padding=20,
                                ),
                                ft.Text("Start a session to begin", size=16, color=Colors.TEXT_PRIMARY),
                                ft.Text("Camera will activate automatically", size=12, color=Colors.TEXT_SECONDARY),
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
                            alignment=ft.Alignment.CENTER,
                            visible=not self.camera_active,
                        ),
                    ], expand=True),
                    bgcolor=Colors.BG_CARD,
                    border=ft.Border.all(1, Colors.GLASS_BORDER),
                    border_radius=10,
                    expand=True,
                ),

                # Controls
                ft.Container(
                    content=ft.Column([
                        # Session controls
                        ft.Row([
                            self.start_session_button,
                            self.stop_session_button,
                        ], spacing=8),
                        # Action controls (visible during session)
                        ft.Row([
                            self.capture_button,
                            self.detect_toggle,
                        ], spacing=8, visible=self.session_active),
                    ], spacing=8),
                    padding=ft.Padding.symmetric(horizontal=24, vertical=12),
                ),
            ], spacing=0, expand=True,
        )

    def _safe_update(self):
        try:
            self.app.page.update()
        except Exception:
            pass

    def start_session(self):
        """Start a new detection session."""
        from core.data_models import DetectionSession
        
        with self._lock:
            if self.session_active:
                return
            
            # Create new session
            session = DetectionSession(
                source_type="camera",
                conf_threshold=self.app.settings.get("conf", 0.25),
                iou_threshold=self.app.settings.get("iou", 0.45),
                model_id=self.app.model_manager.active_model_id or "",
            )
            self.session_id = session.session_id
            self.detection_count = 0
            
            # Save to database
            try:
                from core.database import Database
                self.db = Database()
                self.db.create_session(session)
            except Exception as e:
                print(f"[FieldTest] Database error: {e}")
            
            # Get GPS location
            try:
                from core.location import location_service
                import asyncio
                asyncio.ensure_future(location_service.get_current_location())
            except Exception:
                pass
            
            self.session_active = True
        
        # Update UI
        self.session_id_text.value = f"Session: {self.session_id}"
        self.start_session_button.visible = False
        self.stop_session_button.visible = True
        self.capture_button.visible = True
        self.detect_toggle.visible = True
        
        # Start camera
        self._start_camera()
        
        # Update location
        try:
            from core.location import location_service
            self.location_text.value = f"Location: {location_service.format_coordinates()}"
        except Exception:
            self.location_text.value = "Location: unavailable"
        
        self._safe_update()
        self.app.show_snackbar(f"Session started: {self.session_id}")

    def stop_session(self):
        """Stop the current detection session."""
        with self._lock:
            if not self.session_active:
                return
            
            # Stop camera
            self._stop_camera()
            
            # Update database
            try:
                from core.database import Database
                db = Database()
                session = db.get_session(self.session_id)
                if session:
                    session.ended_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                    session.total_particles = self.detection_count
                    db.update_session(session)
            except Exception as e:
                print(f"[FieldTest] Database error: {e}")
            
            self.session_active = False
        
        # Update UI
        self.session_id_text.value = f"Completed: {self.session_id}"
        self.start_session_button.visible = True
        self.stop_session_button.visible = False
        self.capture_button.visible = False
        self.detect_toggle.visible = False
        self._safe_update()
        self.app.show_snackbar(f"Session completed: {self.detection_count} particles")

    def _start_camera(self):
        """Start camera capture."""
        with self._lock:
            if self.camera_active:
                return
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.app.show_snackbar("Cannot open camera")
                    return
                self.camera_active = True
            except Exception as e:
                self.app.show_snackbar(f"Camera error: {e}")
                return
        
        self._camera_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._camera_thread.start()

    def _stop_camera(self):
        """Stop camera capture."""
        with self._lock:
            self.camera_active = False
            self.detect_running = False
            cap = self.cap
            self.cap = None
        
        if cap:
            cap.release()

    def _capture_loop(self):
        """Camera capture loop with configurable FPS."""
        fps_target = self.app.settings.get("camera_fps", 15)
        frame_interval = 1.0 / fps_target
        clahe_enabled = self.app.settings.get("clahe_enabled", True)
        
        while True:
            with self._lock:
                if not self.camera_active:
                    break
                cap = self.cap
            
            if cap is None or not cap.isOpened():
                break
            
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.time()
            now = time.time()
            self._frame_times.append(now)
            fps = sum(1 for t in self._frame_times if now - t < 1.0)

            display_frame = frame
            if self.detect_running and self.app.current_engine:
                try:
                    proc_frame = frame.copy()
                    if clahe_enabled:
                        from core.vision import apply_clahe
                        proc_frame = apply_clahe(proc_frame)
                    
                    conf = self.app.settings.get("conf", 0.25)
                    iou = self.app.settings.get("iou", 0.45)
                    t0 = time.time()
                    results = self.app.current_engine.detect(proc_frame, conf_thresh=conf, iou_thresh=iou)
                    latency = (time.time() - t0) * 1000
                    
                    from core.vision import draw_boxes
                    from core.analytics import compute_stats
                    display_frame = draw_boxes(frame, results)
                    stats = compute_stats(results)
                    
                    self.detection_count += stats.get("total", 0)
                    self.particle_count.value = str(self.detection_count)
                    self.hud_latency.value = f"Latency: {latency:.0f}ms"
                except Exception as e:
                    print(f"[FieldTest] Detection error: {e}")

            _, buf = cv2.imencode('.jpg', display_frame)
            b64 = base64.b64encode(buf).decode()
            self.image_display.src = f"data:image/jpeg;base64,{b64}"
            self.image_display.visible = True
            self.hud_fps.value = f"FPS: {fps}"
            self._safe_update()

            elapsed = time.time() - frame_start
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)

    def capture_frame(self):
        """Capture a single frame for analysis."""
        if not self.camera_active or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Save frame
        try:
            from core.data_models import ParticleDetection
            from core.vision import apply_clahe
            
            clahe_enabled = self.app.settings.get("clahe_enabled", True)
            if clahe_enabled:
                frame = apply_clahe(frame)
            
            conf = self.app.settings.get("conf", 0.25)
            iou = self.app.settings.get("iou", 0.45)
            
            t0 = time.time()
            results = self.app.current_engine.detect(frame, conf_thresh=conf, iou_thresh=iou)
            latency = (time.time() - t0) * 1000
            
            # Create detections
            detections = []
            for i, (bbox, label, confidence) in enumerate(results):
                detection = ParticleDetection(
                    particle_id=i + 1,
                    session_id=self.session_id or "",
                    polymer_type=label,
                    bbox_x1=bbox[0], bbox_y1=bbox[1],
                    bbox_x2=bbox[2], bbox_y2=bbox[3],
                    confidence=confidence,
                    model_id=self.app.model_manager.active_model_id or "",
                    inference_ms=latency,
                )
                detections.append(detection)
            
            # Save to database
            if self.session_id and detections:
                try:
                    from core.database import Database
                    db = Database()
                    db.add_detections(detections)
                except Exception as e:
                    print(f"[FieldTest] Database error: {e}")
            
            self.detection_count += len(detections)
            self.particle_count.value = str(self.detection_count)
            self._safe_update()
            self.app.show_snackbar(f"Captured: {len(detections)} particles")
        except Exception as e:
            self.app.show_snackbar(f"Capture failed: {e}")

    def toggle_detection(self):
        """Toggle auto-detection."""
        with self._lock:
            if self.app.current_engine is None:
                self.app.show_snackbar("No model loaded")
                return
            self.detect_running = not self.detect_running
            running = self.detect_running
        
        if running:
            self.detect_toggle.text = "Stop Detection"
            self.detect_toggle.icon = ft.Icons.STOP
            self.detect_toggle.bgcolor = Colors.WARNING
        else:
            self.detect_toggle.text = "Auto-Detect"
            self.detect_toggle.icon = ft.Icons.AUTO_FIX_HIGH
            self.detect_toggle.bgcolor = Colors.GLASS_BG
        self._safe_update()

    def cleanup(self):
        """Cleanup resources."""
        self._stop_camera()
