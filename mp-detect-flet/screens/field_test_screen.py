# screens/field_test_screen.py - High-performance field testing screen
"""Camera-first field testing screen with optimized performance and modern UI."""
import flet as ft
import cv2
import base64
import threading
import time
from components.theme import Colors


class FieldTestScreen:
    """High-performance field-testing screen with decoupled camera and inference."""
    
    def __init__(self, app):
        self.app = app
        self._lock = threading.Lock()
        
        # Session state
        self.session_active = False
        self.session_id = None
        self.detection_count = 0
        
        # Camera state - using high-performance camera
        self.camera_active = False
        self.camera = None
        self._display_thread = None
        self._detect_running = False
        
        # UI elements
        PLACEHOLDER = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQq0AAAAAElFTkSuQmCC"
        self.image_display = ft.Image(
            src=PLACEHOLDER, fit=ft.BoxFit.COVER,
            visible=True, border_radius=28,
        )
        
        # HUD pills
        self.hud_fps = self._create_hud_pill("FPS: --", ft.Icons.SPEED)
        self.hud_latency = self._create_hud_pill("Latency: --ms", ft.Icons.TIMER)
        self.hud_particles = self._create_hud_pill("0", ft.Icons.BUBBLE_CHART)
        
        # Session info
        self.session_id_text = ft.Text("No active session", size=12, color=Colors.TEXT_SECONDARY)
        self.location_text = ft.Text("Location: --", size=11, color=Colors.TEXT_MUTED)
        
        # Controls
        self.start_session_button = None
        self.stop_session_button = None
        self.capture_button = None
        self.detect_toggle = None

    def _create_hud_pill(self, text: str, icon: ft.Icons) -> ft.Container:
        """Create a glassmorphic HUD pill."""
        text_widget = ft.Text(text, size=11, color=Colors.ACCENT_CYAN, 
                             font_family="monospace", weight=ft.FontWeight.W_500)
        return ft.Container(
            content=ft.Row([
                ft.Icon(icon, size=14, color=Colors.ACCENT_CYAN),
                text_widget,
            ], spacing=4),
            bgcolor=ft.Colors.with_opacity(0.6, Colors.BG_CARD),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.2, Colors.ACCENT_CYAN)),
            border_radius=20,
            padding=ft.Padding.symmetric(horizontal=10, vertical=6),
            data=text_widget,
        )

    def build_content(self) -> ft.Column:
        self.start_session_button = ft.Button(
            "Start Session", icon=ft.Icons.PLAY_CIRCLE,
            on_click=lambda _: self.start_session(),
            bgcolor=Colors.SUCCESS, color=Colors.BG_PRIMARY,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=24)),
            height=48, expand=True,
        )
        self.stop_session_button = ft.Button(
            "Stop Session", icon=ft.Icons.STOP_CIRCLE,
            on_click=lambda _: self.stop_session(),
            bgcolor=Colors.ERROR, color=ft.Colors.WHITE,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=24)),
            visible=False, height=48, expand=True,
        )
        self.capture_button = ft.Button(
            "Capture", icon=ft.Icons.CAMERA,
            on_click=lambda _: self.capture_frame(),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=20),
                bgcolor=ft.Colors.with_opacity(0.15, Colors.ACCENT_CYAN),
                color=Colors.ACCENT_CYAN,
            ),
            visible=False, height=44, expand=True,
        )
        self.detect_toggle = ft.Button(
            "Auto-Detect", icon=ft.Icons.AUTO_FIX_HIGH,
            on_click=lambda _: self.toggle_detection(),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=20),
                bgcolor=ft.Colors.with_opacity(0.15, Colors.ACCENT_CYAN),
                color=Colors.ACCENT_CYAN,
            ),
            visible=False, height=44, expand=True,
        )

        return ft.Column(
            [
                # Header - minimal
                ft.Container(
                    content=ft.Row([
                        ft.Column([
                            ft.Text("Field Test", size=20, weight=ft.FontWeight.BOLD, 
                                   color=Colors.TEXT_PRIMARY),
                            self.session_id_text,
                        ], spacing=2),
                        ft.Row([
                            ft.Icon(ft.Icons.LOCATION_ON, size=14, color=Colors.ACCENT_CYAN),
                            self.location_text,
                        ], spacing=4),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=ft.Padding.symmetric(horizontal=20, vertical=10),
                ),

                # Camera viewport - full bleed with rounded corners
                ft.Container(
                    content=ft.Stack([
                        self.image_display,
                        
                        # Floating HUD pills - top left
                        ft.Container(
                            content=ft.Row([
                                self.hud_fps,
                                self.hud_latency,
                                self.hud_particles,
                            ], spacing=8),
                            top=16, left=16,
                        ),
                        
                        # Capture indicator
                        ft.Container(
                            content=ft.Container(
                                width=12, height=12,
                                bgcolor=Colors.ERROR,
                                border_radius=6,
                                visible=self.session_active,
                            ),
                            top=16, right=16,
                        ),
                        
                        # Placeholder
                        ft.Container(
                            content=ft.Column([
                                ft.Container(
                                    content=ft.Icon(ft.Icons.VIDEOCAM_OFF, size=48, 
                                                   color=Colors.TEXT_MUTED),
                                    bgcolor=ft.Colors.with_opacity(0.1, Colors.BG_CARD),
                                    border_radius=28, padding=20,
                                ),
                                ft.Text("Tap Start Session", size=16, 
                                       color=Colors.TEXT_PRIMARY, weight=ft.FontWeight.W_500),
                                ft.Text("Camera will activate automatically", size=12, 
                                       color=Colors.TEXT_SECONDARY),
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
                            alignment=ft.Alignment.CENTER,
                            visible=not self.camera_active,
                        ),
                    ], expand=True),
                    expand=True,
                    border_radius=28,
                    clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                ),

                # Controls - pill-shaped buttons
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            self.start_session_button,
                            self.stop_session_button,
                        ], spacing=12),
                        ft.Row([
                            self.capture_button,
                            self.detect_toggle,
                        ], spacing=12, visible=self.session_active),
                    ], spacing=10),
                    padding=ft.Padding.symmetric(horizontal=20, vertical=12),
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
        
        # Start high-performance camera
        self._start_camera()
        
        # Update location
        try:
            from core.location import location_service
            self.location_text.value = location_service.format_coordinates()
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
        """Start camera with high-performance capture."""
        from core.camera import HighPerformanceCamera
        
        try:
            fps_target = self.app.settings.get("camera_fps", 15)
            self.camera = HighPerformanceCamera(camera_id=0, target_fps=fps_target)
            
            # Start with display callback
            self.camera.start(process_callback=self._process_frame)
            self.camera_active = True
            
            # Start display update thread
            self._display_thread = threading.Thread(
                target=self._display_loop, daemon=True
            )
            self._display_thread.start()
            
        except Exception as e:
            self.app.show_snackbar(f"Camera error: {e}")

    def _stop_camera(self):
        """Stop camera capture."""
        self.camera_active = False
        if self.camera:
            self.camera.stop()
            self.camera = None

    def _display_loop(self):
        """Display loop - updates UI with latest frame."""
        PLACEHOLDER = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQq0AAAAAElFTkSuQmCC"
        
        while self.camera_active and self.camera:
            frame = self.camera.get_current_frame()
            if frame is not None:
                # Encode to base64 (minimal - only for display)
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(buf).decode()
                self.image_display.src = f"data:image/jpeg;base64,{b64}"
                self.image_display.visible = True
            
            # Update FPS display
            if self.camera:
                fps = self.camera.get_fps()
                hud_fps_text = self.hud_fps.data
                if hud_fps_text:
                    hud_fps_text.value = f"FPS: {fps}"
            
            self._safe_update()
            time.sleep(0.033)  # ~30 FPS display refresh

    def _process_frame(self, frame: np.ndarray):
        """Process frame with ML inference (called from camera thread)."""
        if not self._detect_running or self.app.current_engine is None:
            return
        
        try:
            clahe_enabled = self.app.settings.get("clahe_enabled", True)
            if clahe_enabled:
                from core.vision import apply_clahe
                frame = apply_clahe(frame)
            
            conf = self.app.settings.get("conf", 0.25)
            iou = self.app.settings.get("iou", 0.45)
            
            t0 = time.time()
            results = self.app.current_engine.detect(frame, conf_thresh=conf, iou_thresh=iou)
            latency = (time.time() - t0) * 1000
            
            # Update HUD
            if self.camera:
                self.hud_latency.data.value = f"Latency: {latency:.0f}ms"
            
            # Count particles
            from core.analytics import compute_stats
            stats = compute_stats(results)
            self.detection_count += stats.get("total", 0)
            self.hud_particles.data.value = str(self.detection_count)
            
        except Exception as e:
            print(f"[FieldTest] Detection error: {e}")

    def capture_frame(self):
        """Capture a single frame for analysis."""
        if not self.camera_active or self.camera is None:
            return
        
        frame = self.camera.get_current_frame()
        if frame is None:
            return
        
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
            self.hud_particles.data.value = str(self.detection_count)
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
            self._detect_running = not self._detect_running
            running = self._detect_running
        
        if running:
            self.detect_toggle.text = "Stop Detection"
            self.detect_toggle.icon = ft.Icons.STOP
            self.detect_toggle.bgcolor = ft.Colors.with_opacity(0.15, Colors.WARNING)
            self.detect_toggle.color = Colors.WARNING
        else:
            self.detect_toggle.text = "Auto-Detect"
            self.detect_toggle.icon = ft.Icons.AUTO_FIX_HIGH
            self.detect_toggle.bgcolor = ft.Colors.with_opacity(0.15, Colors.ACCENT_CYAN)
            self.detect_toggle.color = Colors.ACCENT_CYAN
        self._safe_update()

    def cleanup(self):
        """Cleanup resources."""
        self._stop_camera()
