# utils/controllers.py
"""Controller mixins for MPDetectApp: Inference, Upload, Export."""
import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.utils import platform


class InferenceController:
    """Camera inference methods."""

    def _setup_inference_state(self):
        self._inf_cap = None
        self._inf_active = False
        self._inf_recording = False
        self._inf_writer = None
        self._inf_record_path = None
        self._inf_last_frame = None
        self._inf_last_det = None
        self._inf_fps_buf = []
        self._inf_thread_busy = False

    def inference_init(self):
        self._inf_active = False
        self._inf_recording = False
        self._stop_inference_loop()
        if self._inf_cap is not None:
            self._inf_cap.release()
            self._inf_cap = None

        if platform == "win":
            self._inf_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self._inf_cap = cv2.VideoCapture(0)

        if self._inf_cap.isOpened():
            Clock.schedule_interval(self._inf_loop, 1.0 / 30.0)
            self.active_media_type = "camera"
            self._hide_viewport_placeholder()
        else:
            self.show_snackbar("Cannot open camera")

    def inference_cleanup(self):
        self._inf_active = False
        self._inf_recording = False
        self._stop_inference_loop()
        if self._inf_writer:
            self._inf_writer.release()
            self._inf_writer = None
        if self._inf_cap:
            self._inf_cap.release()
            self._inf_cap = None

    def _stop_inference_loop(self):
        Clock.unschedule(self._inf_loop)

    def toggle_inference(self):
        self._inf_active = not self._inf_active
        if not self._inf_active:
            self._inf_reset_table()

    def _inf_loop(self, dt):
        if not self._inf_cap or not self._inf_cap.isOpened():
            return
        ret, frame = self._inf_cap.read()
        if not ret or frame is None:
            return

        if frame.size == 0 or frame.shape[0] < 10 or frame.shape[1] < 10:
            return

        self._inf_last_frame = frame

        if self._inf_active and not self._inf_thread_busy and self.engine is not None:
            self._inf_thread_busy = True
            t0 = time.perf_counter()
            threading.Thread(
                target=self._inf_infer_thread,
                args=(frame.copy(), t0),
                daemon=True,
            ).start()
        else:
            try:
                tex = self._cv2_to_texture(frame)
                scr = self._sm.get_screen("inference")
                if "raw_image" in scr.ids:
                    scr.ids.raw_image.texture = tex
                if "ann_image" in scr.ids:
                    scr.ids.ann_image.texture = tex
            except (RuntimeError, ValueError, cv2.error, OSError) as e:
                print(f"[inf_loop] texture error: {e}")

        if self._inf_recording and self._inf_writer:
            try:
                self._inf_writer.write(frame)
            except (RuntimeError, cv2.error) as e:
                print(f"[inf_loop] writer error: {e}")
                self._stop_inf_recording()

    def _inf_infer_thread(self, frame: np.ndarray, t0: float):
        results = []
        elapsed_ms = 0.0
        try:
            if frame is None or frame.size == 0 or frame.shape[0] < 10 or frame.shape[1] < 10:
                raise ValueError("Invalid frame dimensions for inference")
            results = self.engine.detect(
                frame,
                conf_thresh=self.cfg.get("conf", 0.25),
                iou_thresh=self.cfg.get("iou", 0.45),
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        except (RuntimeError, ValueError, cv2.error, OSError, np.AxisError) as e:
            print(f"[infer_thread] {e}")

        try:
            annotated_frame = self._draw_boxes(frame.copy(), results)
        except (RuntimeError, ValueError, cv2.error) as e:
            print(f"[infer_thread] draw error: {e}")
            annotated_frame = frame.copy()
        self._inf_last_det = results

        Clock.schedule_once(
            lambda dt, r=results, a=annotated_frame, m=elapsed_ms: self._inf_push_ui(r, a, m)
        )
        self._inf_thread_busy = False

    def _inf_push_ui(self, results, annotated_frame, latency_ms):
        if self._inf_last_frame is None:
            return

        raw_tex = self._cv2_to_texture(self._inf_last_frame)
        scr = self._sm.get_screen("inference")

        if "raw_image" in scr.ids and raw_tex:
            scr.ids.raw_image.texture = raw_tex

        now = time.perf_counter()
        self._inf_fps_buf.append(now)
        self._inf_fps_buf = [t for t in self._inf_fps_buf if now - t < 1.0]
        fps = len(self._inf_fps_buf)

        if "hud_fps" in scr.ids:
            scr.ids.hud_fps.text = f"FPS: {fps}"
        if "hud_latency" in scr.ids:
            scr.ids.hud_latency.text = f"Latency: {latency_ms:.1f}ms"
        if "hud_backend" in scr.ids:
            scr.ids.hud_backend.text = self._engine_badge()

        total = len(results)
        avg_conf = sum(c for _, _, c in results) / total if total else 0.0
        stats = self._stats_from_results(results)

        if "total_value" in scr.ids:
            scr.ids.total_value.text = str(total)
        if "avg_value" in scr.ids:
            scr.ids.avg_value.text = f"{avg_conf:.2f}"

        for cname, (cnt, conf) in stats.items():
            k = cname.lower()
            if f"{k}_count_value" in scr.ids:
                scr.ids[f"{k}_count_value"].text = str(cnt)
            if f"{k}_conf" in scr.ids:
                scr.ids[f"{k}_conf"].text = f"{conf:.2f}"

    def _draw_boxes(self, frame, results):
        colors = [
            (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (0, 255, 0), (255, 0, 0),
        ]
        for (x1, y1, x2, y2), label, conf in results:
            idx = CLASSES.index(label) if label in CLASSES else 0
            color = colors[idx % len(colors)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (int(x1), int(y1) - th - 6), (int(x1) + tw + 4, int(y1)), color, -1)
            cv2.putText(
                frame, text,
                (int(x1) + 2, int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )
        return frame

    def _inf_reset_table(self):
        scr = self._sm.get_screen("inference")
        if "total_value" in scr.ids:
            scr.ids.total_value.text = "0"
        if "avg_value" in scr.ids:
            scr.ids.avg_value.text = "0.00"
        for c in CLASSES:
            k = c.lower()
            if f"{k}_count_value" in scr.ids:
                scr.ids[f"{k}_count_value"].text = "0"
            if f"{k}_conf" in scr.ids:
                scr.ids[f"{k}_conf"].text = "0.00"

    def inference_take_snapshot(self):
        if self._inf_last_frame is None:
            self.show_snackbar("No frame captured")
            return
        frame = self._inf_last_frame.copy()
        if self._inf_active and self._inf_last_det:
            frame = self._draw_boxes(frame, self._inf_last_det)
        path = self.files.save_image(frame)
        self.show_snackbar(f"Saved: {os.path.basename(path)}")

    def inference_toggle_recording(self):
        if self._inf_recording:
            self._stop_inf_recording()
        else:
            self._start_inf_recording()

    def _start_inf_recording(self):
        if not self._inf_cap or not self._inf_cap.isOpened():
            return
        w = int(self._inf_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._inf_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._inf_record_path = self.files.get_video_path()
        self._inf_writer = cv2.VideoWriter(
            self._inf_record_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h)
        )
        self._inf_recording = True
        self.show_snackbar("Recording started")

    def _stop_inf_recording(self):
        self._inf_recording = False
        if self._inf_writer:
            self._inf_writer.release()
            self._inf_writer = None
        if self._inf_record_path:
            self.show_snackbar(f"Saved: {os.path.basename(self._inf_record_path)}")
            self._inf_record_path = None


class UploadController:
    """File upload and detection methods."""

    def _setup_upload_state(self):
        self._ug_file_path = None
        self.ug_file_path = ""
        self._ug_annotated = None
        self._ug_temp_video = None
        self._stop_event = threading.Event()
        self._ug_total_frames = 0
        self._ug_current_frame = 0
        self._ug_cached_raw_results = []
        self._ug_cached_annotated_frame = None
        self._ug_cached_elapsed_ms = 0.0
        self._ug_cached_iou = self.cfg.get("iou", 0.45)
        self.ug_scale_factor = DEFAULT_SCALE_FACTOR

    def open_file_manager(self):
        request_android_permissions()
        default_dir = os.path.join(os.path.expanduser("~"), "MP Detect", "Unseen Data")
        if not os.path.isdir(default_dir):
            default_dir = os.path.expanduser("~")
        if open_file_dialog is not None:
            open_file_dialog(
                title="Select Micrograph",
                initial_dir=default_dir,
                filetypes=[
                    ("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                    ("Videos", "*.mp4 *.avi *.mov *.mkv"),
                    ("All files", "*.*"),
                ],
                on_select=self._fm_select,
                on_cancel=lambda: self.show_snackbar("No file selected"),
                zenity_filters=[
                    "Images|*.png *.jpg *.jpeg *.tif *.tiff",
                    "Videos|*.mp4 *.avi *.mov *.mkv",
                ],
            )
        else:
            self._open_file_zenity(default_dir)

    def _open_file_zenity(self, default_dir):
        try:
            result = subprocess.run(
                ["zenity", "--file-selection",
                 "--title=Select Micrograph",
                 f"--filename={default_dir}/",
                 "--file-filter=Images|*.png *.jpg *.jpeg *.tif *.tiff",
                 "--file-filter=Videos|*.mp4 *.avi *.mov *.mkv"],
                capture_output=True, text=True, timeout=30,
            )
            path = result.stdout.strip()
            if path and os.path.isfile(path):
                self._fm_select(path)
            else:
                self.show_snackbar("No file selected")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            self.show_snackbar(f"File picker unavailable: {e}")

    def _fm_select(self, path):
        valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
                      ".mp4", ".mov", ".avi", ".mkv")
        if not path.lower().endswith(valid_exts):
            self.show_snackbar("Unsupported format")
            return
        self._ug_file_path = path
        self.ug_file_path = path
        self.ug_has_image = True
        self.ug_has_result = False
        self.active_media_type = "file"
        self.active_source = "file"
        self._update_source_buttons("file")
        self._show_ug_preview(path)
        self._hide_viewport_placeholder()

    def _show_ug_preview(self, path):
        ext = path.lower().rsplit(".", 1)[-1]
        frame = None
        try:
            if ext in ("mp4", "mov", "avi", "mkv"):
                if not os.path.isfile(path) or os.path.getsize(path) == 0:
                    self.show_snackbar("File is empty or missing")
                    return
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if not ret:
                        frame = None
                else:
                    cap.release()
            else:
                if not os.path.isfile(path) or os.path.getsize(path) == 0:
                    self.show_snackbar("File is empty or missing")
                    return
                frame = cv2.imread(path)
        except (RuntimeError, cv2.error, OSError) as e:
            print(f"[preview] Error reading file: {e}")
            frame = None

        if frame is not None and frame.size > 0 and frame.shape[0] > 0 and frame.shape[1] > 0:
            try:
                tex = self._cv2_to_texture(frame)
                scr = self._sm.get_screen("upload")
                if "ug_image" in scr.ids and tex:
                    scr.ids.ug_image.texture = tex
            except (RuntimeError, ValueError, cv2.error, OSError) as e:
                print(f"[preview] Texture error: {e}")
        else:
            self.show_snackbar("Cannot read file (corrupted or unsupported format)")

    def ug_handle_action(self):
        if not self.ug_has_image:
            self.open_file_manager()
        elif not self.ug_has_result:
            self._run_upload_detection()
        else:
            self.ug_reset()
            self.open_file_manager()

    def _run_upload_detection(self):
        if self._ug_file_path is None or self.engine is None:
            return
        self._stop_event.clear()
        self.ug_processing = True
        self.ug_processing_active = True
        self.is_processing = True
        self.ug_progress = 0
        self.ug_frame_text = "Starting..."
        self.ug_particle_count = 0
        t0 = time.perf_counter()
        threading.Thread(
            target=self._ug_detect_thread,
            args=(self._ug_file_path, t0),
            daemon=True,
        ).start()

    def ug_cancel_processing(self):
        if self.ug_processing_active:
            self._stop_event.set()
            Clock.schedule_once(lambda dt: self._ug_cancelled_ui(), 0)

    def _ug_detect_thread(self, path: str, t0: float):
        cap = None
        writer = None
        try:
            ext = path.lower().rsplit(".", 1)[-1]
            is_video = ext in ("mp4", "mov", "avi", "mkv")
            results = []
            raw_results = []
            annotated = None
            frame_count = 0

            if is_video:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError("Failed to open video file")
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30
                temp_vid = os.path.join(self.files.dir, "temp_processed.mp4")
                writer = cv2.VideoWriter(
                    temp_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    total = 1
                mid = total // 2
                self._ug_total_frames = total

                Clock.schedule_once(
                    lambda dt, tot=total: self._ug_push_progress(0, total)
                )

                while not self._stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    try:
                        raw = self.engine.detect(
                            frame, conf_thresh=0.01,
                            iou_thresh=self.cfg.get("iou", 0.45),
                        )
                    except (RuntimeError, ValueError, cv2.error) as e:
                        print(f"[ug_thread] detect error frame {frame_count}: {e}")
                        raw = []
                    conf_t = self.cfg.get("conf", 0.25)
                    res = [(b, l, c) for b, l, c in raw if c >= conf_t]
                    frame = self._draw_boxes(frame, res)
                    writer.write(frame)
                    frame_count += 1
                    self._ug_current_frame = frame_count

                    if frame_count == mid or annotated is None:
                        annotated = frame.copy()
                        results = res
                        raw_results = raw

                    cur = frame_count
                    tot = total
                    Clock.schedule_once(
                        lambda dt, c=cur, t=tot: self._ug_push_progress(c, t)
                    )

                if cap is not None:
                    cap.release()
                    cap = None
                if writer is not None:
                    writer.release()
                    writer = None
                self._ug_temp_video = temp_vid
            else:
                frame = cv2.imread(path)
                if frame is None:
                    raise RuntimeError("Failed to read image file")
                try:
                    raw = self.engine.detect(
                        frame, conf_thresh=0.01,
                        iou_thresh=self.cfg.get("iou", 0.45),
                    )
                except (RuntimeError, ValueError, cv2.error) as e:
                    print(f"[ug_thread] detect error on image: {e}")
                    raw = []
                conf_t = self.cfg.get("conf", 0.25)
                results = [(b, l, c) for b, l, c in raw if c >= conf_t]
                raw_results = raw
                annotated = self._draw_boxes(frame.copy(), results)
                self._ug_total_frames = 1
                self._ug_current_frame = 1
                Clock.schedule_once(lambda dt: self._ug_push_progress(1, 1))

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_det = len(results)
            avg_conf = sum(c for _, _, c in results) / total_det if total_det else 0.0
            stats = self._stats_from_results(results)

            if self._stop_event.is_set():
                Clock.schedule_once(lambda dt: self._ug_cancelled_ui())
                return

            Clock.schedule_once(
                lambda dt, a=annotated, s=stats, t=total_det, ac=avg_conf, m=elapsed_ms, rr=raw_results:
                    self._ug_push_result(a, s, t, ac, m, rr)
            )
        except (RuntimeError, ValueError, cv2.error, OSError) as e:
            print(f"[ug_thread] {e}")
            Clock.schedule_once(lambda dt, err=str(e): self._ug_error(err))
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass

    def _ug_push_progress(self, current: int, total: int):
        pct = int((current / total) * 100) if total > 0 else 0
        self.ug_progress = pct
        self.ug_frame_text = f"Frame {current} / {total}" if total > 1 else ""

    def _ug_cancelled_ui(self):
        self.ug_processing = False
        self.ug_processing_active = False
        self.is_processing = False
        self.ug_progress = 0
        self.ug_frame_text = "Cancelled"
        self.show_snackbar("Processing cancelled")

    def on_threshold_change(self, instance, value):
        self.cfg["conf"] = float(value)
        save_settings(self.cfg)
        self._refilter_cached_results()

    def on_iou_threshold_change(self, instance, value):
        self.cfg["iou"] = float(value)
        save_settings(self.cfg)
        self._refilter_cached_results()

    def set_scale_factor(self, value):
        try:
            self.ug_scale_factor = float(value)
        except (ValueError, TypeError):
            self.ug_scale_factor = 0.0
        self.cfg["scale_factor"] = self.ug_scale_factor
        save_settings(self.cfg)
        if self._ug_cached_raw_results:
            self._refilter_cached_results()

    def _refilter_cached_results(self):
        if not self._ug_cached_raw_results:
            return
        conf_t = self.cfg.get("conf", 0.25)
        filtered = [(b, l, c) for b, l, c in self._ug_cached_raw_results if c >= conf_t]
        stats = self._stats_from_results(filtered)
        total = len(filtered)
        avg_conf = sum(c for _, _, c in filtered) / total if total else 0.0
        frame = self._read_cached_frame()
        if frame is not None:
            annotated = self._draw_boxes(frame, filtered)
            scr = self._sm.get_screen("upload")
            if "ug_image" in scr.ids:
                scr.ids.ug_image.texture = self._cv2_to_texture(annotated)
        scr = self._sm.get_screen("upload")
        if "ug_total_label" in scr.ids:
            scr.ids.ug_total_label.text = f"Total: {total}"
        if "ug_avg_label" in scr.ids:
            scr.ids.ug_avg_label.text = f"Avg Conf: {avg_conf:.2f}"
        for cname, (cnt, conf) in stats.items():
            k = cname.lower()
            if f"ug_{k}_count" in scr.ids:
                scr.ids[f"ug_{k}_count"].text = str(cnt)
            if f"ug_{k}_conf" in scr.ids:
                scr.ids[f"ug_{k}_conf"].text = f"{conf:.2f}"
        perf = f"Latency: {self._ug_cached_elapsed_ms:.0f}ms  |  Backend: {self._engine_badge()}"
        if "ug_perf_stats" in scr.ids:
            scr.ids.ug_perf_stats.text = perf
        analytics = self._compute_analytics(filtered)
        self._update_analytics_ui(analytics)
        self.ug_particle_count = total

    def _read_cached_frame(self):
        if self._ug_file_path is None:
            return None
        if not os.path.isfile(self._ug_file_path) or os.path.getsize(self._ug_file_path) == 0:
            return None
        ext = self._ug_file_path.lower().rsplit(".", 1)[-1]
        try:
            if ext in ("mp4", "mov", "avi", "mkv"):
                cap = cv2.VideoCapture(self._ug_file_path)
                if cap.isOpened():
                    mid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None and frame.size > 0:
                        return frame
            else:
                frame = cv2.imread(self._ug_file_path)
                if frame is not None and frame.size > 0:
                    return frame
        except (RuntimeError, cv2.error, OSError) as e:
            print(f"[cached_frame] Error: {e}")
        return None

    def _ug_push_result(self, annotated, stats, total, avg_conf, elapsed_ms, raw_results=None):
        scr = self._sm.get_screen("upload")
        if annotated is not None and "ug_image" in scr.ids:
            scr.ids.ug_image.texture = self._cv2_to_texture(annotated)

        if "ug_total_label" in scr.ids:
            scr.ids.ug_total_label.text = f"Total: {total}"
        if "ug_avg_label" in scr.ids:
            scr.ids.ug_avg_label.text = f"Avg Conf: {avg_conf:.2f}"

        for cname, (cnt, conf) in stats.items():
            k = cname.lower()
            if f"ug_{k}_count" in scr.ids:
                scr.ids[f"ug_{k}_count"].text = str(cnt)
            if f"ug_{k}_conf" in scr.ids:
                scr.ids[f"ug_{k}_conf"].text = f"{conf:.2f}"

        perf = f"Latency: {elapsed_ms:.0f}ms  |  Backend: {self._engine_badge()}"
        if "ug_perf_stats" in scr.ids:
            scr.ids.ug_perf_stats.text = perf

        if raw_results is not None:
            self._ug_cached_raw_results = raw_results
        self._ug_cached_annotated_frame = annotated
        self._ug_cached_elapsed_ms = elapsed_ms
        analytics = self._compute_analytics(
            [(b, l, c) for b, l, c in (raw_results or []) if c >= self.cfg.get("conf", 0.25)]
        )
        self._update_analytics_ui(analytics)

        self.ug_particle_count = total
        self.ug_has_result = True
        self.ug_processing = False
        self.ug_processing_active = False
        self.is_processing = False
        self.ug_progress = 100
        self.ug_frame_text = f"Complete - {total} particles"

    def _ug_error(self, msg):
        self.ug_processing = False
        self.ug_processing_active = False
        self.is_processing = False
        self.ug_progress = 0
        self.ug_frame_text = "Error"
        self.ug_particle_count = 0
        self.show_snackbar(f"Detection failed: {msg}")

    def ug_view_in_inference(self):
        if not self.ug_has_result:
            self.show_snackbar("No results to view")
            return
        try:
            raw_frame = self._read_cached_frame()
            annotated_frame = self._ug_cached_annotated_frame

            scr = self._sm.get_screen("inference")

            if raw_frame is not None:
                raw_tex = self._cv2_to_texture(raw_frame)
                if raw_tex and "raw_image" in scr.ids:
                    scr.ids.raw_image.texture = raw_tex

            if annotated_frame is not None:
                ann_tex = self._cv2_to_texture(annotated_frame)
                if ann_tex and "ann_image" in scr.ids:
                    scr.ids.ann_image.texture = ann_tex

            total = self.ug_particle_count
            if "status_text" in scr.ids:
                scr.ids.status_text.text = f"Upload: {total} particles detected"
            if "perf_stats" in scr.ids:
                elapsed = self._ug_cached_elapsed_ms
                scr.ids.perf_stats.text = f"Latency: {elapsed:.0f}ms  |  Backend: {self._engine_badge()}"

            self._sm.transition.direction = "left"
            self._sm.current = "inference"
        except (RuntimeError, ValueError, cv2.error) as e:
            self.show_snackbar(f"Failed to load results: {e}")

    def ug_reset(self):
        self._stop_event.clear()
        self._ug_file_path = None
        self.ug_file_path = ""
        self._ug_annotated = None
        self._ug_temp_video = None
        self._ug_total_frames = 0
        self._ug_current_frame = 0
        self._ug_cached_raw_results = []
        self._ug_cached_annotated_frame = None
        self._ug_cached_elapsed_ms = 0.0
        self._ug_cached_iou = self.cfg.get("iou", 0.45)
        self.ug_has_image = False
        self.ug_has_result = False
        self.ug_progress = 0
        self.ug_frame_text = ""
        self.ug_particle_count = 0
        self.ug_analytics_text = ""
        self.ug_size_bucket_text = ""
        self.ug_morphology_text = ""
        self.ug_illumination_tag = ""
        self.ug_size_counts = [0, 0, 0, 0]
        self.ug_morph_counts = [0, 0, 0, 0]
        self.ug_mean_area = 0.0
        self.ug_min_area = 0.0
        self.ug_max_area = 0.0
        self.ug_mean_ar = 0.0
        self.ug_size_um_text = ""
        scr = self._sm.get_screen("upload")
        if "ug_image" in scr.ids:
            scr.ids.ug_image.texture = None
        if "ug_total_label" in scr.ids:
            scr.ids.ug_total_label.text = "Total: 0"
        if "ug_avg_label" in scr.ids:
            scr.ids.ug_avg_label.text = "Avg Conf: 0.00"
        if "ug_perf_stats" in scr.ids:
            scr.ids.ug_perf_stats.text = ""
        for c in CLASSES:
            k = c.lower()
            if f"ug_{k}_count" in scr.ids:
                scr.ids[f"ug_{k}_count"].text = "0"
            if f"ug_{k}_conf" in scr.ids:
                scr.ids[f"ug_{k}_conf"].text = "0.00"


class ExportController:
    """CSV, image, and video export methods."""

    def _draw_scale_bar(self, frame):
        sf = self.ug_scale_factor
        if sf <= 0:
            return frame
        h, w = frame.shape[:2]
        bar_len_px = int(100 * sf)
        if bar_len_px <= 0 or bar_len_px > w // 3:
            return frame
        bar_um = 100.0
        y0 = h - 20
        x0 = w - bar_len_px - 20
        cv2.rectangle(frame, (x0, y0), (x0 + bar_len_px, y0 + 4), (255, 255, 255), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + bar_len_px, y0 + 4), (0, 0, 0), 1)
        label = f"{bar_um:.0f} um"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tx = x0 + (bar_len_px - tw) // 2
        cv2.putText(frame, label, (tx, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _build_csv_rows(self):
        rows = []
        raw = self._ug_cached_raw_results
        if not raw:
            return rows
        conf_t = self.cfg.get("conf", 0.25)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lighting = self.lighting_preset
        for i, (box, label, conf) in enumerate(raw):
            if conf < conf_t:
                continue
            x1, y1, x2, y2 = box
            bw = abs(x2 - x1)
            bh = abs(y2 - y1)
            morph = self._compute_morphology(box)
            rows.append({
                "particle_id": i + 1,
                "timestamp": ts,
                "class_label": label,
                "confidence": round(conf, 4),
                "x": round(float(x1), 1),
                "y": round(float(y1), 1),
                "width": round(float(bw), 1),
                "height": round(float(bh), 1),
                "area_px": round(morph["area"], 1),
                "area_um2": round(morph["area_um2"], 2) if morph["area_um2"] > 0 else "",
                "aspect_ratio": round(morph["aspect_ratio"], 3),
                "morphology": morph["shape"],
                "illumination": lighting,
                "conf_thresh": self.cfg.get("conf", 0.25),
                "iou_thresh": self.cfg.get("iou", 0.45),
            })
        return rows

    def export_csv_data(self):
        rows = self._build_csv_rows()
        if not rows:
            self.show_snackbar("No data to export")
            return
        path = self.files.get_export_csv_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fieldnames = list(rows[0].keys())
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            self.show_snackbar(f"CSV saved: {os.path.basename(path)}")
        except (OSError, IOError) as e:
            self.show_snackbar(f"CSV export failed: {e}")

    def export_full_report(self):
        raw = self._ug_cached_raw_results
        conf_t = self.cfg.get("conf", 0.25)
        filtered = [(b, l, c) for b, l, c in (raw or []) if c >= conf_t]
        analytics = self._compute_analytics(filtered)
        report = {
            "app": "MP Detect",
            "version": "Phase 6",
            "generated_at": datetime.now().isoformat(),
            "source_file": os.path.basename(self._ug_file_path) if self._ug_file_path else "",
            "model_backend": self._engine_badge(),
            "parameters": {
                "conf_threshold": self.cfg.get("conf", 0.25),
                "iou_threshold": self.cfg.get("iou", 0.45),
                "scale_factor_px_per_um": self.ug_scale_factor,
                "illumination_mode": self.lighting_preset,
            },
            "summary": {
                "total_particles": analytics["total"],
                "mean_area_px": round(analytics["mean_area"], 1),
                "min_area_px": round(analytics["min_area"], 1),
                "max_area_px": round(analytics["max_area"], 1),
                "mean_area_um2": round(analytics["mean_area_um"], 2) if analytics.get("mean_area_um") else None,
                "mean_aspect_ratio": round(analytics["mean_ar"], 3),
            },
            "size_distribution": {
                b[0]: analytics["size_buckets"][i]
                for i, b in enumerate(SIZE_BUCKETS)
            },
            "morphology_distribution": {
                c[0]: analytics["morph_counts"][i]
                for i, c in enumerate(MORPH_CATEGORIES)
            },
            "class_breakdown": {
                k: {"count": v[0], "mean_confidence": round(v[1], 3)}
                for k, v in self._stats_from_results(filtered).items()
            },
        }
        path = self.files.get_export_report_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(report, f, indent=2)
            self.show_snackbar(f"Report saved: {os.path.basename(path)}")
        except (OSError, IOError) as e:
            self.show_snackbar(f"Report export failed: {e}")

    def save_annotated_snapshot(self):
        frame = self._read_cached_frame()
        if frame is None:
            self.show_snackbar("No frame to save")
            return
        raw = self._ug_cached_raw_results
        conf_t = self.cfg.get("conf", 0.25)
        filtered = [(b, l, c) for b, l, c in (raw or []) if c >= conf_t]
        annotated = self._draw_boxes(frame, filtered)
        annotated = self._draw_scale_bar(annotated)
        path = self.files.get_annotated_image_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, annotated)
            self.show_snackbar(f"Annotated image: {os.path.basename(path)}")
        except (OSError, IOError) as e:
            self.show_snackbar(f"Image save failed: {e}")

    def export_annotated_video(self):
        if self._ug_file_path is None:
            self.show_snackbar("No source file for video export")
            return
        if not os.path.isfile(self._ug_file_path):
            self.show_snackbar("Source file not found")
            return
        self.show_snackbar("Exporting annotated video...")
        threading.Thread(target=self._export_video_thread, daemon=True).start()

    def _export_video_thread(self):
        cap = None
        writer = None
        try:
            path = self._ug_file_path
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError("Cannot open source video")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30
            out_path = self.files.get_annotated_video_path()
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                writer.release()
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                writer.release()
                fourcc = cv2.VideoWriter_fourcc(*"X264")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("Cannot create video writer with any codec")
            raw = self._ug_cached_raw_results
            conf_t = self.cfg.get("conf", 0.25)
            filtered = [(b, l, c) for b, l, c in (raw or []) if c >= conf_t]
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                annotated = self._draw_boxes(frame, filtered)
                annotated = self._draw_scale_bar(annotated)
                writer.write(annotated)
                frame_count += 1
            cap.release()
            cap = None
            writer.release()
            writer = None
            Clock.schedule_once(
                lambda dt, p=out_path, fc=frame_count: self._video_export_done(p, fc), 0
            )
        except (RuntimeError, ValueError, cv2.error, OSError) as e:
            Clock.schedule_once(lambda dt, err=str(e): self._video_export_error(err), 0)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass

    def _video_export_done(self, path, frame_count):
        self.show_snackbar(f"Video exported: {os.path.basename(path)} ({frame_count} frames)")

    def _video_export_error(self, msg):
        self.show_snackbar(f"Video export failed: {msg}")

    def ug_share(self):
        target = (
            self._ug_temp_video
            if self._ug_temp_video and os.path.isfile(self._ug_temp_video)
            else self._ug_file_path
        )
        if target and os.path.isfile(target):
            try:
                open_in_system_viewer(target)
            except (OSError, RuntimeError) as e:
                self.show_snackbar(f"Share failed: {e}")
        else:
            self.show_snackbar("No file available to open")
