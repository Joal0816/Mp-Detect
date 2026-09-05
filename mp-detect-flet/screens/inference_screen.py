# screens/inference_screen.py - Detection view screen
"""Inference screen for MP Detect Flet app."""
import flet as ft
import cv2
import numpy as np
import base64
import threading
from typing import Optional


class InferenceScreen:
    def __init__(self, app):
        self.app = app
        self.image_display = None
        self.stats_card = None
        self.confidence_slider = None
        self.iou_slider = None
        self.detect_button = None
        self.view_results_button = None
        self.is_processing = False
        
        # Store stat text references for direct updates
        self.stat_texts = {}
        
    def build(self) -> ft.View:
        # Image display
        self.image_display = ft.Image(
            width=350,
            height=350,
            fit=ft.ImageFit.CONTAIN,
        )
        
        # Stat text references
        self.stat_texts = {
            "total": ft.Text("Total: 0"),
            "avg_conf": ft.Text("Avg Confidence: 0.00"),
            "pet": ft.Text("PET: 0"),
            "hdpe": ft.Text("HDPE: 0"),
            "pvc": ft.Text("PVC: 0"),
            "ldpe": ft.Text("LDPE: 0"),
            "pp": ft.Text("PP: 0"),
            "ps": ft.Text("PS: 0"),
        }
        
        # Stats card
        self.stats_card = ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text("Detection Results", size=16, weight=ft.FontWeight.BOLD),
                        self.stat_texts["total"],
                        self.stat_texts["avg_conf"],
                        ft.Divider(),
                        self.stat_texts["pet"],
                        self.stat_texts["hdpe"],
                        self.stat_texts["pvc"],
                        self.stat_texts["ldpe"],
                        self.stat_texts["pp"],
                        self.stat_texts["ps"],
                    ],
                    spacing=5,
                ),
                padding=10,
            ),
        )
        
        # Confidence slider
        self.confidence_slider = ft.Slider(
            min=0.01,
            max=1.0,
            value=self.app.settings.get("conf", 0.25),
            label="Confidence: {value}",
            on_change=self.on_confidence_change,
        )
        
        # IoU slider
        self.iou_slider = ft.Slider(
            min=0.1,
            max=1.0,
            value=self.app.settings.get("iou", 0.45),
            label="IoU: {value}",
            on_change=self.on_iou_change,
        )
        
        # Detect button
        self.detect_button = ft.ElevatedButton(
            "Detect",
            icon=ft.icons.PLAY_ARROW,
            on_click=lambda _: self.run_detection(),
            bgcolor=ft.colors.PRIMARY,
            color=ft.colors.WHITE,
        )
        
        # View results button
        self.view_results_button = ft.ElevatedButton(
            "View Results",
            icon=ft.icons.VISIBILITY,
            on_click=lambda _: self.view_results(),
            visible=False,
        )
        
        # Load current file if available
        if self.app.current_file:
            self.load_image(self.app.current_file)
        
        return ft.View(
            "/inference",
            [
                ft.AppBar(
                    title=ft.Text("Inference"),
                    leading=ft.IconButton(
                        icon=ft.icons.ARROW_BACK,
                        on_click=lambda _: self.app.go("/gallery"),
                    ),
                    bgcolor=ft.colors.SURFACE_VARIANT,
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            # Image display
                            ft.Container(
                                content=self.image_display,
                                border=ft.border.all(1, ft.colors.OUTLINE),
                                border_radius=10,
                            ),
                            
                            # Sliders
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Text("Confidence Threshold"),
                                        self.confidence_slider,
                                        ft.Text("IoU Threshold"),
                                        self.iou_slider,
                                    ],
                                    spacing=5,
                                ),
                                padding=10,
                            ),
                            
                            # Action buttons
                            ft.Row(
                                [
                                    self.detect_button,
                                    self.view_results_button,
                                ],
                                spacing=10,
                                alignment=ft.MainAxisAlignment.CENTER,
                            ),
                            
                            # Stats card
                            self.stats_card,
                        ],
                        spacing=10,
                    ),
                    padding=10,
                    expand=True,
                ),
                # Bottom navigation bar
                self.app.nav_bar.build(),
            ],
        )
        
    def load_image(self, file_path: str):
        """Load and display image from file path."""
        try:
            if self.app.current_file is None:
                return
                
            # Check if it's a video
            if self.app.file_handler.is_video(file_path):
                # Extract first frame from video
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    # Convert to base64 for display
                    _, buffer = cv2.imencode('.jpg', frame)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    self.image_display.src_base64 = img_base64
            else:
                # It's an image
                frame = cv2.imread(file_path)
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    self.image_display.src_base64 = img_base64
                    
        except Exception as e:
            print(f"[Inference] Error loading image: {e}")
            self.app.show_snackbar(f"Error loading image: {e}")
            
    def on_confidence_change(self, e):
        """Handle confidence slider change."""
        from core.settings_manager import save_settings
        value = float(e.control.value)
        self.app.settings["conf"] = value
        save_settings(self.app.settings)
        
    def on_iou_change(self, e):
        """Handle IoU slider change."""
        from core.settings_manager import save_settings
        value = float(e.control.value)
        self.app.settings["iou"] = value
        save_settings(self.app.settings)
        
    def run_detection(self):
        """Run detection on current file."""
        if self.app.current_file is None:
            self.app.show_snackbar("No file selected")
            return
            
        if self.app.current_engine is None:
            self.app.show_snackbar("No model loaded")
            return
            
        if self.is_processing:
            self.app.show_snackbar("Already processing")
            return
            
        self.is_processing = True
        self.detect_button.disabled = True
        self.app.page.update()
        
        # Run detection in background thread
        threading.Thread(
            target=self._detection_thread,
            daemon=True,
        ).start()
        
    def _detection_thread(self):
        """Run detection in background thread."""
        try:
            file_path = self.app.current_file
            conf = self.app.settings.get("conf", 0.25)
            iou = self.app.settings.get("iou", 0.45)
            
            if self.app.file_handler.is_video(file_path):
                # Video detection - extract first frame
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    raise RuntimeError("Failed to read video frame")
            else:
                # Image detection
                frame = cv2.imread(file_path)
                if frame is None:
                    raise RuntimeError("Failed to read image")
                    
            # Run detection - returns list of (bbox, label, conf) tuples
            results = self.app.current_engine.detect(
                frame, conf_thresh=conf, iou_thresh=iou
            )
            
            # Compute stats from results
            from core.analytics import compute_stats
            stats = compute_stats(results)
            
            # Draw boxes on frame
            from core.vision import draw_boxes
            annotated = draw_boxes(frame, results)
            
            # Store results
            self.app.current_results = results
            
            # Update UI
            self._update_ui(stats, annotated, results)
            
        except Exception as e:
            print(f"[Inference] Detection error: {e}")
            self.app.show_snackbar(f"Detection failed: {e}")
        finally:
            self.is_processing = False
            self.detect_button.disabled = False
            self.app.page.update()
            
    def _update_ui(self, stats: dict, annotated_frame: np.ndarray, results: list):
        """Update UI with detection results."""
        try:
            # Update image display
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            self.image_display.src_base64 = img_base64
            
            # Update stats using direct references
            total = stats.get("total", 0)
            avg_conf = stats.get("avg_conf", 0.0)
            
            self.stat_texts["total"].value = f"Total: {total}"
            self.stat_texts["avg_conf"].value = f"Avg Confidence: {avg_conf:.2f}"
            
            # Update per-class counts
            per_class = stats.get("per_class", {})
            for cls_name in ["PET", "HDPE", "PVC", "LDPE", "PP", "PS"]:
                cls_stats = per_class.get(cls_name, {"count": 0, "avg_conf": 0.0})
                count = cls_stats.get("count", 0) if isinstance(cls_stats, dict) else cls_stats[0]
                self.stat_texts[cls_name.lower()].value = f"{cls_name}: {count}"
                        
            # Show view results button
            self.view_results_button.visible = True
            self.app.page.update()
            
            self.app.show_snackbar(f"Detection complete: {total} particles found")
            
        except Exception as e:
            print(f"[Inference] UI update error: {e}")
            
    def view_results(self):
        """Navigate to results screen."""
        self.app.go("/result")
