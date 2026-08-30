# main.py
import os
import sys
import time
import cv2
import numpy as np

from kivy.utils import platform
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import ScreenManager, SlideTransition, Screen
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.properties import StringProperty, BooleanProperty
from kivy.uix.image import Image

# KivyMD imports
from kivymd.app import MDApp
from kivymd.uix.snackbar import MDSnackbar
from kivymd.uix.label import MDLabel
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.list import OneLineAvatarIconListItem, IconLeftWidget

# Import utility modules
try:
    from utils.detector import YOLODetector
    from utils.file_handler import FileHandler
    from utils.settings_manager import load_settings, save_settings
    from utils.permissions import request_android_permissions
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    # Allow app to run even if imports fail, to see error logs on device
    pass


# --- Dummy Detector for Fallback ---
class NullDetector:
    classes = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]
    fixed_size = None

    def set_params(self, **kwargs): pass

    # get_video_capture must handle string source for IP cam
    def get_video_capture(self, index=0, width=640, height=480):
        # index can be 0 or a URL string
        return cv2.VideoCapture(index)

    def _empty_summary(self): return {"per_class": {c: (0, 0.0) for c in self.classes}, "total": 0, "avg_conf": 0.0}

    def add_stats_overlay(self, img, stats): return img

    def detect(self, frame): return self._empty_summary(), frame.copy(), []

    def detect_from_file(self, path):
        img = cv2.imread(path)
        return self._empty_summary(), img if img is not None else np.zeros((480, 640, 3), dtype=np.uint8), []

    def process_video(self, source, dest):
        return self._empty_summary(), np.zeros((100, 100, 3), np.uint8)


class SaveDialogContent(MDBoxLayout):
    default_name = StringProperty("")


# --- Screens ---
class HomeScreen(Screen): pass


class CameraScreen(Screen): pass


class UploadScreen(Screen): pass


class RecentScreen(Screen): pass


class PreviewScreen(Screen): pass


class MenuScreen(Screen): pass


class ParametersScreen(Screen): pass


class AboutScreen(Screen): pass


class WindowManager(ScreenManager): pass


class MPDetectApp(MDApp):
    upload_state = StringProperty("empty")
    detection_active = BooleanProperty(False)
    recording_active = BooleanProperty(False)
    is_processing = BooleanProperty(False)

    def show_snackbar(self, message: str, delay: float = 0.0):
        def _do_snackbar(*_):
            MDSnackbar(
                MDLabel(text=message, theme_text_color="Custom", text_color=(1, 1, 1, 1)),
                y=dp(24),
                pos_hint={"center_x": 0.5},
                size_hint_x=0.9,
                duration=2
            ).open()

        if delay > 0:
            Clock.schedule_once(_do_snackbar, delay)
        else:
            _do_snackbar()

    def build(self):
        self.title = "MP Detect"
        self.icon = "assets/app_icon.png"
        self.cfg = load_settings()
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Red"
        self.theme_cls.primary_hue = "900"

        if platform != "android":
            Window.size = (360, 640)

        # Initialize Detector
        try:
            model_path = os.path.abspath("models/best.onnx")
            self.detector = YOLODetector(
                model_path=model_path,
                conf_thresh=self.cfg["conf"],
                iou_thresh=self.cfg["iou"],
                input_size=self.cfg["imgsz"],
            )
        except Exception as e:
            print(f"Model Warning: {e}")
            self.detector = NullDetector()

        self.files = FileHandler()

        # Load UI
        Builder.load_file("ui.kv")

        # State Variables
        self.cap = None
        self.camera_source = 0  # NEW: Default camera source (0 is built-in)
        self.video_writer = None
        self.current_frame = None
        self.current_stats = None
        self.record_start_time = 0.0
        self.save_dialog = None
        self.temp_frame = None
        self.temp_video_path = None
        self.folder_menu = None
        self.sort_menu = None
        self.file_manager = None
        self.process_start_time = 0.0
        self.recent_current_folder = self.files.dir
        self.recent_sort_mode = "date"

        sm = WindowManager(transition=SlideTransition(duration=0.3))
        screens = [
            HomeScreen(name="home"), CameraScreen(name="camera"),
            UploadScreen(name="upload"), RecentScreen(name="recent"),
            PreviewScreen(name="preview"),
            MenuScreen(name="menu"), ParametersScreen(name="parameters"),
            AboutScreen(name="about")
        ]
        for s in screens: sm.add_widget(s)
        return sm

    def on_start(self):
        # Trigger permissions prompt immediately on app start
        request_android_permissions()

    def go(self, screen_name, direction="left"):
        self.root.transition.direction = direction
        self.root.current = screen_name
        if screen_name != "preview":
            scr = self.root.get_screen("preview")
            container = scr.ids.preview_container
            container.clear_widgets()

    # --- File Manager ---
    def open_file_manager(self, from_results=False):
        # Re-request permissions just in case
        request_android_permissions()

        start_path = os.path.abspath(self.files.dir)
        if not os.path.exists(start_path):
            start_path = os.path.expanduser("~")

        self.file_manager = MDFileManager(
            exit_manager=self.exit_file_manager,
            select_path=self.select_file_path,
            preview=False,
            selector="file",
            ext=[".png", ".jpg", ".jpeg", ".mp4", ".mov", ".avi", ".mkv"],
        )
        self.file_manager.show(start_path)

    def select_file_path(self, path):
        self.exit_file_manager()
        if not os.path.exists(path): return

        valid_exts = (".png", ".jpg", ".jpeg", ".mp4", ".mov", ".avi", ".mkv")
        if not path.lower().endswith(valid_exts):
            self.show_snackbar("Invalid file format")
            return

        self.selected_file_path = path
        self._reset_upload_labels()

        try:
            if path.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                cap.release()
                if ret: self._show_texture("upload_image", frame)
            else:
                img = cv2.imread(path)
                if img is not None: self._show_texture("upload_image", img)

            self.upload_state = "selected"

        except Exception as e:
            self.show_snackbar(f"Load Error: {e}")

    def exit_file_manager(self, *args):
        if self.file_manager:
            self.file_manager.close()

    # --- Upload & Detect ---
    def on_upload_state(self, instance, value):
        scr = self.root.get_screen("upload")
        toolbar = scr.ids.upload_toolbar
        if value == "detected":
            save_type = "video" if self.temp_video_path else "image"
            toolbar.right_action_items = [["content-save", lambda x: self.show_custom_save_dialog(save_type)]]
        else:
            toolbar.right_action_items = []

    def handle_upload_action(self):
        if self.upload_state == "empty":
            self.open_file_manager()
        elif self.upload_state == "selected":
            self.start_upload_detection()
        elif self.upload_state == "detected":
            self.reset_upload_screen()
            self.open_file_manager()

    def start_upload_detection(self):
        if not self.selected_file_path: return
        self.is_processing = True
        self.process_start_time = time.time()
        Clock.schedule_once(self._perform_detection, 0.1)

    def _perform_detection(self, dt):
        try:
            is_video = self.selected_file_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
            stats_text = ""

            if is_video:
                temp_vid = os.path.join(self.files.dir, "temp_processed.mp4")

                cap = cv2.VideoCapture(self.selected_file_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: fps = 30

                writer = cv2.VideoWriter(temp_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                middle_index = total_frames // 2 if total_frames > 0 else -1
                frame_count = 0

                thumbnail = None
                middle_frame_stats = None

                while True:
                    ret, frame = cap.read()
                    if not ret: break

                    # Detect
                    det, annotated, _ = self.detector.detect(frame)
                    # Overlay
                    final_frame = self.detector.add_stats_overlay(annotated, det)

                    # Capture Middle Frame Data
                    if frame_count == middle_index:
                        thumbnail = final_frame.copy()
                        middle_frame_stats = det

                    writer.write(final_frame)
                    frame_count += 1

                cap.release()
                writer.release()

                if thumbnail is None: thumbnail = np.zeros((100, 100, 3), dtype=np.uint8)
                if middle_frame_stats is None: middle_frame_stats = self.detector._empty_summary()

                # Stats Logic
                process_end = time.time()
                duration_proc = process_end - self.process_start_time
                vid_len = frame_count / fps if fps > 0 else 0
                stats_text = f"Video Length: {vid_len:.2f}s | Process Time: {duration_proc:.2f}s"

                self.current_upload_stats = middle_frame_stats
                self.temp_video_path = temp_vid
                self.temp_frame = thumbnail

            else:
                # Image Logic
                det, annotated, _ = self.detector.detect_from_file(self.selected_file_path)
                annotated_with_overlay = self.detector.add_stats_overlay(annotated.copy(), det)

                self.current_upload_stats = det
                self.temp_frame = annotated_with_overlay
                self.temp_video_path = None

                process_end = time.time()
                duration_proc = process_end - self.process_start_time
                stats_text = f"Process Time: {duration_proc:.2f}s"

            self._show_texture("upload_image", self.temp_frame)
            self._update_upload_metrics(self.current_upload_stats)

            if "performance_stats" in self.root.get_screen("upload").ids:
                self.root.get_screen("upload").ids.performance_stats.text = stats_text

            self.upload_state = "detected"

        except Exception as e:
            self.show_snackbar(f"Failed: {e}")
            print(e)
        finally:
            self.is_processing = False

    def reset_upload_screen(self):
        self.selected_file_path = None
        self.upload_state = "empty"
        self.temp_video_path = None

        scr = self.root.get_screen("upload")
        scr.ids.upload_image.texture = None
        if "performance_stats" in scr.ids:
            scr.ids.performance_stats.text = ""
        self._reset_upload_labels()

    def _reset_upload_labels(self):
        scr = self.root.get_screen("upload")
        if "total_label" in scr.ids: scr.ids.total_label.text = "Total: 0"
        if "avg_label" in scr.ids: scr.ids.avg_label.text = "Avg Conf: 0.00"
        for cls in self.detector.classes:
            key = cls.lower()
            if f"{key}_count" in scr.ids: scr.ids[f"{key}_count"].text = "0"
            if f"{key}_conf" in scr.ids: scr.ids[f"{key}_conf"].text = "0.00"

    def _update_upload_metrics(self, det):
        scr = self.root.get_screen("upload")
        scr.ids.total_label.text = f"Total: {det['total']}"
        scr.ids.avg_label.text = f"Avg Conf: {det['avg_conf']:.2f}"
        for cls_name, (count, avg) in det["per_class"].items():
            key = cls_name.lower()
            if f"{key}_count" in scr.ids: scr.ids[f"{key}_count"].text = str(count)
            if f"{key}_conf" in scr.ids: scr.ids[f"{key}_conf"].text = f"{avg:.2f}"

    # --- Camera ---

    # REVISED: camera_init now calls the selection dialog
    def camera_init(self):
        request_android_permissions()
        self.show_camera_select_dialog()

    # NEW METHOD: Dialog to choose camera source
    def show_camera_select_dialog(self):

        def select_camera_source(source):
            self.camera_source = source  # 0 (int) or URL (str)
            dialog.dismiss()
            self._start_camera_after_selection()  # Proceed to start camera

        # Dialog Content
        content = MDBoxLayout(orientation="vertical", spacing="12dp", padding="16dp", size_hint_y=None, height=dp(200))

        content.add_widget(MDLabel(text="Select Camera Source", font_style="H6", halign="center"))
        content.add_widget(
            MDLabel(text="Choose between the built-in camera (index 0) or an IP webcam URL.", font_style="Caption",
                    halign="center"))

        # Laptop Camera Button
        content.add_widget(MDFlatButton(
            text="Laptop Camera (Index 0)",
            on_release=lambda x: select_camera_source(0)
        ))

        # IP Webcam Button
        content.add_widget(MDFlatButton(
            text="IP Webcam (http://192.168.5.178:8080/video)",
            on_release=lambda x: select_camera_source("https://192.168.95.23:8080/video")
        ))

        dialog = MDDialog(
            title="Choose Camera",
            type="custom",
            content_cls=content,
            auto_dismiss=False,
        )
        dialog.open()

    # NEW METHOD: Logic to start camera after selection
    def _start_camera_after_selection(self):
        self.camera_active = True
        self.detection_active = False
        self.recording_active = False

        # Use the chosen source: either 0 or the URL string
        self.cap = self.detector.get_video_capture(self.camera_source)

        if self.cap is None or not self.cap.isOpened():
            self.show_snackbar("Error: Failed to open camera source.", 0.1)
            self.camera_cleanup()
            return

        Clock.schedule_interval(self._camera_loop, 1.0 / 30.0)

    def camera_cleanup(self):
        self.camera_active = False
        if self.recording_active:
            self.stop_recording()
        Clock.unschedule(self._camera_loop)
        if self.cap:
            self.cap.release()
            self.cap = None

    def toggle_detection_state(self):
        self.detection_active = not self.detection_active
        if not self.detection_active:
            self._reset_camera_table()

    def _reset_camera_table(self):
        scr = self.root.get_screen("camera")
        if "total_label" in scr.ids: scr.ids.total_label.text = "Total: 0"
        if "avg_label" in scr.ids: scr.ids.avg_label.text = "Avg: 0.00"
        for cls in self.detector.classes:
            key = cls.lower()
            if f"{key}_count" in scr.ids: scr.ids[f"{key}_count"].text = "0"
            if f"{key}_conf" in scr.ids: scr.ids[f"{key}_conf"].text = "0.00"

    # --- SAVE DIALOG ---
    def show_custom_save_dialog(self, type_="image"):
        default_name = time.strftime("IMG_%Y%m%d_%H%M%S") if type_ == "image" else time.strftime("VID_%Y%m%d_%H%M%S")
        self.save_dialog = MDDialog(
            title="Save Capture",
            type="custom",
            content_cls=SaveDialogContent(default_name=default_name),
            buttons=[
                MDFlatButton(text="DISCARD", on_release=self.close_save_dialog),
                MDFlatButton(text="SAVE", on_release=lambda x: self.process_custom_save(type_))
            ],
            auto_dismiss=False
        )
        self.save_dialog.open()

    def show_folder_selection(self, text_field):
        folders = self.files.list_folders()
        menu_items = []
        if not folders:
            self.show_snackbar("No subfolders found")
        else:
            for f in folders:
                folder_name = os.path.basename(f)
                menu_items.append({
                    "text": folder_name,
                    "viewclass": "OneLineListItem",
                    "on_release": lambda x=folder_name: self._set_folder_text(text_field, x)
                })
            self.folder_menu = MDDropdownMenu(
                caller=text_field,
                items=menu_items,
                width_mult=4,
            )
            self.folder_menu.open()

    def _set_folder_text(self, field, text):
        field.text = text
        if self.folder_menu: self.folder_menu.dismiss()

    def close_save_dialog(self, *args):
        if self.save_dialog: self.save_dialog.dismiss()
        if self.root.current != "upload":
            self.temp_frame = None
            if self.temp_video_path and os.path.exists(self.temp_video_path):
                try:
                    os.remove(self.temp_video_path)
                except:
                    pass
            self.temp_video_path = None

    def process_custom_save(self, type_):
        content = self.save_dialog.content_cls
        name = content.ids.file_name.text
        folder = content.ids.folder_name.text
        if not name or not folder:
            self.show_snackbar("Please enter name and folder")
            return

        try:
            if type_ == "image" and self.temp_frame is not None:
                path = self.files.save_custom_image(self.temp_frame, name, folder)
                self.show_snackbar(f"Saved to {folder}")

            elif type_ == "video" and self.temp_video_path:
                path = self.files.save_custom_video(self.temp_video_path, name, folder)
                self.show_snackbar(f"Saved to {folder}")

            if self.save_dialog: self.save_dialog.dismiss()
        except Exception as e:
            self.show_snackbar(f"Error saving: {e}")

    # --- RECORDING ---
    def take_snapshot(self):
        if self.current_frame is not None:
            frame_to_save = self.current_frame.copy()
            if self.detection_active and self.current_stats:
                frame_to_save = self.detector.add_stats_overlay(frame_to_save, self.current_stats)
            self.temp_frame = frame_to_save
            self.show_custom_save_dialog("image")

    def toggle_recording(self):
        if self.recording_active:
            self.stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if not self.cap: return
        self.temp_video_path = os.path.join(self.files.dir, "temp_recording.mp4")
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_writer = cv2.VideoWriter(self.temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
        self.recording_active = True
        self.record_start_time = time.time()
        self.show_snackbar("Recording Started")

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording_active = False
            self.show_custom_save_dialog("video")

    def _camera_loop(self, dt):
        if not self.camera_active or not self.cap: return
        ret, frame = self.cap.read()
        if not ret: return

        display_frame = frame

        if self.detection_active:
            det, annotated, _ = self.detector.detect(frame)
            self.current_stats = det
            display_frame = annotated

            self._update_table_ui(det)

            if self.recording_active and self.video_writer:
                record_frame = self.detector.add_stats_overlay(annotated.copy(), det)
                self.video_writer.write(record_frame)
        else:
            if self.recording_active and self.video_writer:
                self.video_writer.write(frame)

        self.current_frame = display_frame

        if self.recording_active:
            elapsed = int(time.time() - self.record_start_time)
            scr = self.root.get_screen("camera")
            if "timer_label" in scr.ids:
                scr.ids.timer_label.text = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
                scr.ids.timer_label.opacity = 1
        else:
            try:
                self.root.get_screen("camera").ids.timer_label.opacity = 0
            except:
                pass

        self._show_texture("camera_image", display_frame)

    def _update_table_ui(self, det):
        scr = self.root.get_screen("camera")
        if "total_label" in scr.ids:
            scr.ids.total_label.text = f"Total: {det['total']}"
            scr.ids.avg_label.text = f"Avg: {det['avg_conf']:.2f}"
            for cls_name, (count, avg) in det["per_class"].items():
                key = cls_name.lower()
                if f"{key}_count" in scr.ids: scr.ids[f"{key}_count"].text = str(count)
                if f"{key}_conf" in scr.ids: scr.ids[f"{key}_conf"].text = f"{avg:.2f}"

    # --- Parameters ---
    def parameters_fill(self):
        scr = self.root.get_screen("parameters")
        scr.ids.conf_input.text = str(self.cfg['conf'])
        scr.ids.iou_input.text = str(self.cfg['iou'])
        if self.detector.fixed_size:
            scr.ids.imgsz_input.text = str(self.detector.fixed_size)
        else:
            scr.ids.imgsz_input.text = str(self.cfg['imgsz'])

    def show_imgsz_menu(self):
        if self.detector.fixed_size is not None:
            self.show_snackbar(f"Model requires fixed size: {self.detector.fixed_size}")
            return
        sizes = [320, 416, 512, 640, 960, 1280]
        field = self.root.get_screen("parameters").ids.imgsz_input
        menu_items = [{"text": str(s), "on_release": lambda x=s: self.select_imgsz(x)} for s in sizes]
        w_mult = field.width / dp(48)
        self.imgsz_menu = MDDropdownMenu(caller=field, items=menu_items, width_mult=w_mult, position="bottom")
        self.imgsz_menu.open()

    def select_imgsz(self, size):
        self.imgsz_menu.dismiss()
        self.root.get_screen("parameters").ids.imgsz_input.text = str(size)

    def parameters_save(self, conf, iou, imgsz):
        try:
            final_imgsz = self.detector.fixed_size if self.detector.fixed_size else int(imgsz)
            self.cfg = {"conf": float(conf), "iou": float(iou), "imgsz": final_imgsz}
            save_settings(self.cfg)
            self.detector.set_params(self.cfg['conf'], self.cfg['iou'], self.cfg['imgsz'])
            self.show_snackbar("Settings Saved")
        except:
            self.show_snackbar("Invalid Input")

    # --- Recent & Preview ---
    def recent_refresh(self):
        self._load_recent_content(self.recent_current_folder)

    def recent_back_nav(self):
        if self.recent_current_folder != self.files.dir:
            parent = os.path.dirname(self.recent_current_folder)
            if self.files.dir in parent:
                self.recent_current_folder = parent
                self._load_recent_content(self.recent_current_folder)
            else:
                self.recent_current_folder = self.files.dir
                self._load_recent_content(self.recent_current_folder)
        else:
            self.go("home", "right")

    def show_sort_menu(self):
        menu_items = [
            {"text": "Sort by Date", "viewclass": "OneLineListItem", "on_release": lambda: self.set_sort("date")},
            {"text": "Sort by Name", "viewclass": "OneLineListItem", "on_release": lambda: self.set_sort("name")}
        ]
        self.sort_menu = MDDropdownMenu(
            caller=self.root.get_screen("recent").ids.recent_toolbar,
            items=menu_items,
            width_mult=3,
        )
        self.sort_menu.open()

    def set_sort(self, mode):
        if self.sort_menu: self.sort_menu.dismiss()
        self.recent_sort_mode = mode
        self.show_snackbar(f"Sorting by {mode.title()}")
        self._load_recent_content(self.recent_current_folder)

    def _enter_folder(self, path):
        self.recent_current_folder = path
        self._load_recent_content(path)

    def _load_recent_content(self, path):
        scr = self.root.get_screen("recent")
        scr.ids.recent_list.clear_widgets()

        if path == self.files.dir:
            scr.ids.recent_toolbar.title = "Recent Results"
        else:
            scr.ids.recent_toolbar.title = os.path.basename(path)

        if not os.path.exists(path): return

        all_items = os.listdir(path)
        folders = []
        files = []

        for item in all_items:
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                folders.append(full_path)
            elif item.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4', 'avi', '.mov')):
                files.append(full_path)

        if self.recent_sort_mode == "date":
            folders.sort(key=os.path.getmtime, reverse=True)
            files.sort(key=os.path.getmtime, reverse=True)
        else:
            folders.sort(key=lambda x: os.path.basename(x).lower())
            files.sort(key=lambda x: os.path.basename(x).lower())

        for f in folders:
            name = os.path.basename(f)
            item = OneLineAvatarIconListItem(text=name)
            item.add_widget(IconLeftWidget(icon="folder"))
            item.bind(on_release=lambda x, p=f: self._enter_folder(p))
            scr.ids.recent_list.add_widget(item)

        for f in files:
            name = os.path.basename(f)
            item = OneLineAvatarIconListItem(text=name)
            ext = f.split('.')[-1].lower()
            is_video = ext in ['mp4', 'avi', 'mov']
            icon = "filmstrip" if is_video else "image"
            item.add_widget(IconLeftWidget(icon=icon))

            if is_video:
                item.bind(on_release=lambda x: self.show_snackbar("Video saved."))
            else:
                item.bind(on_release=lambda x, p=f: self._open_internal_preview(p))

            scr.ids.recent_list.add_widget(item)

        if not folders and not files:
            scr.ids.recent_list.add_widget(OneLineAvatarIconListItem(text="Empty Folder"))

    def _open_internal_preview(self, path):
        if path.lower().endswith((".mp4", ".mov", ".avi")):
            self.show_snackbar("Video playback not supported in-app")
            return

        self.go("preview")
        scr = self.root.get_screen("preview")
        container = scr.ids.preview_container
        container.clear_widgets()
        scr.ids.preview_toolbar.title = os.path.basename(path)

        img = Image(source=path, allow_stretch=True, keep_ratio=True)
        container.add_widget(img)

    def _show_texture(self, widget_id, img):
        if img is None: return
        buf = cv2.flip(img, 0).tobytes()
        tex = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        tex.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        scr = self.root.get_screen(self.root.current)
        if widget_id in scr.ids:
            scr.ids[widget_id].texture = tex


if __name__ == "__main__":
    MPDetectApp().run()