# screens/result_screen.py - Results view screen
"""Result screen for MP Detect Flet app."""
import flet as ft
import cv2
import base64
import os
import threading


class ResultScreen:
    def __init__(self, app):
        self.app = app
        self.video_display = None
        self.stats_card = None
        self.export_csv_button = None
        self.export_json_button = None
        self.export_image_button = None
        self.export_video_button = None
        self.share_button = None
        
        # Store stat text references for direct updates
        self.stat_texts = {}
        
    def build(self) -> ft.View:
        # Video/Image display
        self.video_display = ft.Image(
            width=350,
            height=300,
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
                        ft.Text("Detection Summary", size=16, weight=ft.FontWeight.BOLD),
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
        
        # Export buttons
        self.export_csv_button = ft.ElevatedButton(
            "Export CSV",
            icon=ft.icons.FILE_DOWNLOAD,
            on_click=lambda _: self.export_csv(),
        )
        
        self.export_json_button = ft.ElevatedButton(
            "Export Report",
            icon=ft.icons.DESCRIPTION,
            on_click=lambda _: self.export_json(),
        )
        
        self.export_image_button = ft.ElevatedButton(
            "Save Image",
            icon=ft.icons.IMAGE,
            on_click=lambda _: self.export_image(),
        )
        
        self.export_video_button = ft.ElevatedButton(
            "Export Video",
            icon=ft.icons.VIDEO_FILE,
            on_click=lambda _: self.export_video(),
            visible=False,
        )
        
        self.share_button = ft.ElevatedButton(
            "Share",
            icon=ft.icons.SHARE,
            on_click=lambda _: self.share(),
        )
        
        # Load results if available
        self.load_results()
        
        return ft.View(
            "/result",
            [
                ft.AppBar(
                    title=ft.Text("Results"),
                    leading=ft.IconButton(
                        icon=ft.icons.ARROW_BACK,
                        on_click=lambda _: self.app.go("/inference"),
                    ),
                    bgcolor=ft.colors.SURFACE_VARIANT,
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            # Video/Image display
                            ft.Container(
                                content=self.video_display,
                                border=ft.border.all(1, ft.colors.OUTLINE),
                                border_radius=10,
                            ),
                            
                            # Stats card
                            self.stats_card,
                            
                            # Export buttons
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Text("Export Options", size=14, weight=ft.FontWeight.BOLD),
                                        ft.Row(
                                            [
                                                self.export_csv_button,
                                                self.export_json_button,
                                            ],
                                            spacing=10,
                                        ),
                                        ft.Row(
                                            [
                                                self.export_image_button,
                                                self.export_video_button,
                                            ],
                                            spacing=10,
                                        ),
                                        self.share_button,
                                    ],
                                    spacing=10,
                                ),
                                padding=10,
                            ),
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
        
    def load_results(self):
        """Load detection results and display."""
        from core.vision import draw_boxes
        
        if self.app.current_file is None:
            return
            
        # Display the annotated image if available
        try:
            if self.app.file_handler.is_video(self.app.current_file):
                # Extract first frame from video
                cap = cv2.VideoCapture(self.app.current_file)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    # Draw boxes on frame
                    annotated = draw_boxes(frame, self.app.current_results)
                    _, buffer = cv2.imencode('.jpg', annotated)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    self.video_display.src_base64 = img_base64
                    
                    # Show video export button, hide image export
                    self.export_video_button.visible = True
            else:
                # It's an image
                frame = cv2.imread(self.app.current_file)
                if frame is not None:
                    annotated = draw_boxes(frame, self.app.current_results)
                    _, buffer = cv2.imencode('.jpg', annotated)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    self.video_display.src_base64 = img_base64
                    
        except Exception as e:
            print(f"[Result] Error loading results: {e}")
            
        # Update stats
        self.update_stats()
        
    def update_stats(self):
        """Update stats display with current results."""
        results = self.app.current_results
        
        if not results:
            return
            
        # Calculate stats
        total = len(results)
        avg_conf = sum(conf for _, _, conf in results) / total if total > 0 else 0.0
        
        # Count per class
        per_class = {"PET": 0, "HDPE": 0, "PVC": 0, "LDPE": 0, "PP": 0, "PS": 0}
        for _, label, _ in results:
            if label in per_class:
                per_class[label] += 1
                
        # Update UI using direct references
        self.stat_texts["total"].value = f"Total: {total}"
        self.stat_texts["avg_conf"].value = f"Avg Confidence: {avg_conf:.2f}"
        self.stat_texts["pet"].value = f"PET: {per_class['PET']}"
        self.stat_texts["hdpe"].value = f"HDPE: {per_class['HDPE']}"
        self.stat_texts["pvc"].value = f"PVC: {per_class['PVC']}"
        self.stat_texts["ldpe"].value = f"LDPE: {per_class['LDPE']}"
        self.stat_texts["pp"].value = f"PP: {per_class['PP']}"
        self.stat_texts["ps"].value = f"PS: {per_class['PS']}"
                    
        self.app.page.update()
        
    def export_csv(self):
        """Export results to CSV."""
        try:
            from core.export import export_csv
            
            output_path = self.app.file_handler.get_export_csv_path()
            export_csv(
                self.app.current_results,
                output_path,
                self.app.settings,
            )
            
            self.app.show_snackbar(f"CSV saved: {os.path.basename(output_path)}")
            
        except Exception as e:
            self.app.show_snackbar(f"Export failed: {e}")
            
    def export_json(self):
        """Export results to JSON report."""
        try:
            from core.export import export_json_report
            
            output_path = self.app.file_handler.get_export_report_path()
            export_json_report(
                self.app.current_results,
                output_path,
                self.app.settings,
                self.app.current_file,
                "Flet Backend",
            )
            
            self.app.show_snackbar(f"Report saved: {os.path.basename(output_path)}")
            
        except Exception as e:
            self.app.show_snackbar(f"Export failed: {e}")
            
    def export_image(self):
        """Export annotated image."""
        try:
            from core.export import export_annotated_image
            
            if self.app.current_file is None:
                self.app.show_snackbar("No file to export")
                return
                
            # Check if it's a video - extract frame if so
            if self.app.file_handler.is_video(self.app.current_file):
                cap = cv2.VideoCapture(self.app.current_file)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    self.app.show_snackbar("Failed to read video frame")
                    return
            else:
                frame = cv2.imread(self.app.current_file)
                if frame is None:
                    self.app.show_snackbar("Failed to read image")
                    return
                
            output_path = self.app.file_handler.get_annotated_image_path()
            export_annotated_image(
                frame,
                self.app.current_results,
                output_path,
            )
            
            self.app.show_snackbar(f"Image saved: {os.path.basename(output_path)}")
            
        except Exception as e:
            self.app.show_snackbar(f"Export failed: {e}")
            
    def export_video(self):
        """Export annotated video."""
        try:
            from core.export import export_annotated_video
            
            if self.app.current_file is None:
                self.app.show_snackbar("No video to export")
                return
                
            if not self.app.file_handler.is_video(self.app.current_file):
                self.app.show_snackbar("Source is not a video")
                return
                
            output_path = self.app.file_handler.get_annotated_video_path()
            
            # Disable button during export
            self.export_video_button.disabled = True
            self.app.page.update()
            
            # Run in background thread
            threading.Thread(
                target=self._export_video_thread,
                args=(output_path,),
                daemon=True,
            ).start()
            
        except Exception as e:
            self.app.show_snackbar(f"Export failed: {e}")
            
    def _export_video_thread(self, output_path: str):
        """Export video in background thread."""
        try:
            from core.export import export_annotated_video
            
            export_annotated_video(
                self.app.current_file,
                self.app.current_results,
                output_path,
                self.app.settings,
            )
            
            # Re-enable button
            self.export_video_button.disabled = False
            self.app.page.update()
            
            self.app.show_snackbar(f"Video saved: {os.path.basename(output_path)}")
            
        except Exception as e:
            # Re-enable button on error
            self.export_video_button.disabled = False
            self.app.page.update()
            
            self.app.show_snackbar(f"Video export failed: {e}")
            
    def share(self):
        """Share results."""
        self.app.show_snackbar("Share functionality coming soon")
