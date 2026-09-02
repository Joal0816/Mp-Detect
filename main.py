# main.py
import ast
for _attr in ('Str', 'Num', 'Bytes', 'NameConstant', 'Ellipsis'):
    if not hasattr(ast, _attr):
        setattr(ast, _attr, ast.Constant)
if not hasattr(ast.Constant, 's'):
    ast.Constant.s = property(lambda self: self.value)
if not hasattr(ast.Constant, 'n'):
    ast.Constant.n = property(lambda self: self.value)

import os
os.environ["KIVY_NO_ARGS"] = "1"
from kivy.config import Config
Config.set("input", "mouse", "mouse,multitouch_on_demand")
Config.set("graphics", "multisamples", "0")

import csv
import json
import subprocess
import threading
import time
from datetime import datetime
from functools import partial

import cv2
import numpy as np
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, StringProperty
from kivy.utils import platform
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDButton, MDButtonText
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.snackbar import MDSnackbar, MDSnackbarText

# Phase 5: Native Kivy bar chart widget
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle

# ── KivyMD 2.0 Compatibility Shims ────────────────────────────────
from kivy.factory import Factory
from kivymd.uix.selectioncontrol import MDCheckbox
from kivymd.uix.behaviors.toggle_behavior import MDToggleButtonBehavior


class MDRadioGroup(MDBoxLayout):
    pass


class MDRadioButton(MDCheckbox):
    pass


class MDToggleButton(MDButton, MDToggleButtonBehavior):
    pass


Factory.classes["MDRadioGroup"] = {
    "module": None, "cls": MDRadioGroup, "is_template": False,
    "baseclasses": None, "filename": None,
}
Factory.classes["MDRadioButton"] = {
    "module": None, "cls": MDRadioButton, "is_template": False,
    "baseclasses": None, "filename": None,
}
Factory.classes["MDToggleButton"] = {
    "module": None, "cls": MDToggleButton, "is_template": False,
    "baseclasses": None, "filename": None,
}

# ── Graceful Fallback for Utility Modules ─────────────────────────────
try:
    from utils.model_manager import ModelManager
except ImportError:
    class ModelManager:
        def __init__(self):
            self.models = []
            self.active_model_id = None

        def get_active_engine(self):
            return None

        def switch_model(self, model_id):
            return None

        def validate_model_file(self, path):
            return {"valid": False, "errors": ["Module not found"]}

try:
    from utils.file_handler import FileHandler
except ImportError:
    class FileHandler:
        def __init__(self):
            self.dir = os.path.expanduser("~")

        def save_image(self, frame):
            p = os.path.join(self.dir, f"capture_{int(time.time())}.jpg")
            cv2.imwrite(p, frame)
            return p

        def get_video_path(self):
            return os.path.join(self.dir, f"record_{int(time.time())}.mp4")

try:
    from utils.media_dispatcher import open_in_system_viewer
except ImportError:
    def open_in_system_viewer(path):
        if platform == "win":
            os.startfile(path)
        else:
            print(f"[Viewer Request] {path}")

try:
    from utils.settings_manager import load_settings, save_settings, reset_settings, update_setting
except ImportError:
    def load_settings():
        return {"conf": 0.25, "iou": 0.45, "imgsz": 640}

    def save_settings(settings):
        pass

    def reset_settings():
        return {"conf": 0.25, "iou": 0.45, "imgsz": 640}

    def update_setting(cfg, key, value):
        cfg[key] = value
        return cfg

try:
    from utils.detector import detect_available_providers, get_provider_display_name
except ImportError:
    def detect_available_providers():
        return ["CPUExecutionProvider"]

    def get_provider_display_name(p):
        return p

try:
    from utils.permissions import request_android_permissions
except ImportError:
    def request_android_permissions():
        pass

try:
    from utils.file_dialog import open_file_dialog
except ImportError:
    open_file_dialog = None


CLASSES = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]
HUD_BACKEND_BADGE = {
    "onnx": "[ONNX]",
    "tflite": "[TFLite]",
}
_DEFAULTS = {"conf": 0.25, "iou": 0.45, "imgsz": 640}

# Phase 5: Size bucket boundaries (pixels²)
SIZE_BUCKETS = [
    ("< 2500 px", 0, 2500),
    ("2500-10k px", 2500, 10000),
    ("10k-40k px", 10000, 40000),
    ("> 40k px", 40000, float("inf")),
]
# Phase 5: Morphology shape categories (aspect ratio thresholds)
# Fragment: equant/irregular, Film: flat broad, Pellet: elongated, Fiber: highly elongated
MORPH_CATEGORIES = [
    ("Fragment", 1.0, 1.5),
    ("Film", 1.5, 2.0),
    ("Pellet", 2.0, 3.0),
    ("Fiber", 3.0, float("inf")),
]
# Phase 5: Scale factor (pixels per micrometer). Set > 0 to enable physical units.
DEFAULT_SCALE_FACTOR = 0.0


class InferenceScreen(MDScreen):
    def on_enter(self, *args):
        app = MDApp.get_running_app()
        if app:
            app.inference_init()

    def on_leave(self, *args):
        app = MDApp.get_running_app()
        if app:
            app.inference_cleanup()

    def update_split(self, value):
        app = MDApp.get_running_app()
        if app:
            app.update_split(value)


class ModelManagerScreen(MDScreen):
    pass


class UploadGatewayScreen(MDScreen):
    pass


class GalleryScreen(MDScreen):
    pass


class DistributionBarChart(Widget):
    """Phase 5: Lightweight horizontal bar chart rendered via Kivy Canvas + child labels."""
    values = ListProperty([0, 0, 0, 0])
    labels = ListProperty(["A", "B", "C", "D"])
    bar_colors = ListProperty([
        (0, 0.89, 1, 1), (0, 0.7, 0.9, 1),
        (0, 0.5, 0.8, 1), (0, 0.3, 0.7, 1),
    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(values=self._schedule_redraw, labels=self._schedule_redraw,
                  size=self._schedule_redraw, pos=self._schedule_redraw)
        Clock.schedule_once(lambda dt: self._draw_bars(), 0)

    def _schedule_redraw(self, *args):
        Clock.schedule_once(lambda dt: self._draw_bars(), 0)

    def _draw_bars(self):
        self.canvas.clear()
        old = [w for w in self.children if getattr(w, '_chart_lbl', False)]
        for w in old:
            self.remove_widget(w)
        n = min(len(self.values), len(self.labels))
        if n == 0 or self.width <= 0 or self.height <= 0:
            return
        max_val = max(self.values[:n]) if any(v > 0 for v in self.values[:n]) else 1
        label_w = dp(52)
        value_w = dp(32)
        bar_area_w = max(1, self.width - label_w - value_w - dp(8))
        bar_h = max(2, (self.height - (n - 1) * 2) / n)
        for i in range(n):
            val = self.values[i]
            frac = val / max_val if max_val > 0 else 0
            bar_w = max(1, frac * bar_area_w)
            y = self.y + self.height - (i + 1) * (bar_h + 2)
            c = self.bar_colors[i % len(self.bar_colors)]
            with self.canvas:
                Color(*c)
                Rectangle(
                    pos=(self.x + label_w + 4, y),
                    size=(bar_w, bar_h),
                )
            lbl = MDLabel(
                text=str(self.labels[i]),
                role="small",
                text_color=(0.7, 0.7, 0.7, 1),
                size_hint=(None, None),
                size=(label_w, bar_h),
                pos=(self.x, y),
                halign="right",
            )
            lbl._chart_lbl = True
            self.add_widget(lbl)
            val_lbl = MDLabel(
                text=str(val),
                role="small",
                font_name="RobotoMono",
                text_color=(1, 1, 1, 0.9),
                size_hint=(None, None),
                size=(value_w, bar_h),
                pos=(self.x + label_w + 4 + bar_w + 4, y),
                halign="left",
            )
            val_lbl._chart_lbl = True
            self.add_widget(val_lbl)


class MPDetectApp(MDApp):
    upload_state = StringProperty("empty")
    detection_active = BooleanProperty(False)
    recording_active = BooleanProperty(False)
    is_processing = BooleanProperty(False)

    ug_has_image = BooleanProperty(False)
    ug_has_result = BooleanProperty(False)
    ug_processing = BooleanProperty(False)
    ug_file_path = StringProperty("")

    _inf_fps = NumericProperty(0.0)
    _inf_latency = NumericProperty(0.0)

    # Phase 4A: Real-time HUD progress properties
    ug_progress = NumericProperty(0)
    ug_frame_text = StringProperty("")
    ug_particle_count = NumericProperty(0)
    ug_processing_active = BooleanProperty(False)

    # Phase 5: Analytics & diagnostics properties
    ug_analytics_text = StringProperty("")
    ug_size_bucket_text = StringProperty("")
    ug_morphology_text = StringProperty("")
    ug_illumination_tag = StringProperty("")
    ug_size_counts = ListProperty([0, 0, 0, 0])
    ug_morph_counts = ListProperty([0, 0, 0, 0])
    ug_mean_area = NumericProperty(0.0)
    ug_min_area = NumericProperty(0.0)
    ug_max_area = NumericProperty(0.0)
    ug_mean_ar = NumericProperty(0.0)
    ug_scale_factor = NumericProperty(0.0)
    ug_size_um_text = StringProperty("")

    # Phase 2A: Lighting & source state
    active_source = StringProperty("camera")
    lighting_preset = StringProperty("blof")
    active_media_type = StringProperty("none")

    # Phase 7: Status ribbon & diagnostics properties
    status_model_name = StringProperty("No Model")
    status_provider = StringProperty("CPU")
    status_memory = StringProperty("-- MB")
    status_ready = BooleanProperty(False)
    status_model_list = ListProperty([])

    # Phase 8: Keyboard shortcut & calibration properties
    _viewport_mode = StringProperty("annotated")
    _calibration_active = BooleanProperty(False)
    _calibration_p1 = NumericProperty(0)
    _calibration_p2 = NumericProperty(0)
    _calibration_um = NumericProperty(0)
    _magnification = StringProperty("10x")

    def build(self):
        self.title = "MP Detect"
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Cyan"

        if platform != "android":
            Window.size = (360, 640)

        self.cfg = load_settings() if callable(load_settings) else _DEFAULTS.copy()
        self.files = FileHandler()

        # Phase 7: Restore lighting preset from config
        self.lighting_preset = self.cfg.get("lighting_mode", "blof")

        try:
            self.mm = ModelManager()
            # Phase 7: Restore active model from config if available
            saved_model_id = self.cfg.get("active_model_id", "")
            if saved_model_id and saved_model_id != self.mm.active_model_id:
                try:
                    self.mm.switch_model(saved_model_id)
                except (KeyError, RuntimeError):
                    pass
            self.engine = self.mm.get_active_engine()
        except (RuntimeError, ValueError, OSError) as e:
            print(f"[ModelManager Init Warning] {e}")
            self.mm = None
            self.engine = None

        # Phase 7: Update status ribbon with model/provider info
        self._update_status_ribbon()

        self._setup_inference_state()
        self._setup_upload_state()

        sm = MDScreenManager()
        from kivy.lang import Builder
        Builder.load_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpdetect.kv"))
        sm.add_widget(InferenceScreen(name="inference"))
        sm.add_widget(ModelManagerScreen(name="models"))
        sm.add_widget(UploadGatewayScreen(name="upload"))
        sm.add_widget(GalleryScreen(name="gallery"))

        from kivy.uix.boxlayout import BoxLayout
        from kivy.factory import Factory
        root = BoxLayout(orientation="vertical")
        self._sm = sm
        root.add_widget(sm)
        root.add_widget(Factory.BottomNavBar())
        return root

    def on_start(self):
        request_android_permissions()
        Clock.schedule_once(self._populate_model_manager, 0.5)
        Clock.schedule_once(self._populate_gallery, 0.6)
        Clock.schedule_once(self._init_source_group, 0.7)
        # Phase 7: Scan models directory and update status ribbon
        Clock.schedule_once(self._phase7_init, 1.0)
        Clock.schedule_once(lambda dt: setattr(self, "_gl_ready", True), 1.0)
        # Auto-load default micrograph on startup
        Clock.schedule_once(self._auto_load_startup_image, 1.5)

    def _auto_load_startup_image(self, dt):
        pass

    def _phase7_init(self, dt):
        """Phase 7: Initialize model scanning, status ribbon, and config persistence."""
        self._scan_models_directory()
        self._update_status_ribbon()
        # Bind lighting preset to persist on change
        self.bind(lighting_preset=self._on_lighting_persist)
        # Phase 8: Bind keyboard shortcuts
        Window.bind(on_keyboard=self._on_keyboard)
        # Phase 8: Restore calibration from config
        self.ug_scale_factor = self.cfg.get("scale_factor", 0.0)
        self._magnification = self.cfg.get("magnification", "10x")

    def _init_source_group(self, dt):
        self._update_source_buttons(self.active_source)

    def on_stop(self):
        """Phase 8: Production hardening - clean up all resources on exit."""
        self.inference_cleanup()
        # Cancel any ongoing upload detection
        if self.ug_processing_active and hasattr(self, '_stop_event'):
            self._stop_event.set()
        # Clean up temp video files
        if getattr(self, '_ug_temp_video', None) and os.path.isfile(self._ug_temp_video):
            try:
                os.remove(self._ug_temp_video)
            except OSError:
                pass
        # Release any lingering video captures
        if hasattr(self, '_inf_cap') and self._inf_cap is not None:
            try:
                self._inf_cap.release()
            except Exception:
                pass
        # Release video writers
        if hasattr(self, '_inf_writer') and self._inf_writer is not None:
            try:
                self._inf_writer.release()
            except Exception:
                pass
        # Unbind keyboard
        try:
            Window.unbind(on_keyboard=self._on_keyboard)
        except (ValueError, TypeError, AttributeError):
            pass

    def on_source_change(self, instance, value):
        self.active_source = value
        self._update_source_buttons(value)
        self.show_snackbar(f"Source: {value}")

    def _update_source_buttons(self, active_source):
        try:
            scr = self._sm.get_screen("inference")
            btn_file = scr.ids.get("btn_source_file")
            btn_cam = scr.ids.get("btn_source_camera")
            if active_source == "file":
                if btn_file:
                    btn_file.style = "filled"
                    btn_file.md_bg_color = (0, 0.8863, 1, 1)
                if btn_cam:
                    btn_cam.style = "outlined"
                    btn_cam.md_bg_color = (0.102, 0.102, 0.102, 1)
            else:
                if btn_cam:
                    btn_cam.style = "filled"
                    btn_cam.md_bg_color = (0, 0.8863, 1, 1)
                if btn_file:
                    btn_file.style = "outlined"
                    btn_file.md_bg_color = (0.102, 0.102, 0.102, 1)
        except (KeyError, AttributeError):
            pass

    # ── Gallery Methods ─────────────────────────────────────────
    def _populate_gallery(self, dt):
        """Populate the gallery screen with files from MP Detect directory."""
        if self.files is None:
            return
        scr = self._sm.get_screen("gallery")
        if "gallery_list" not in scr.ids:
            return

        scr.ids.gallery_list.clear_widgets()
        files = self.files.list_files()
        supported_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
                          ".mp4", ".mov", ".avi", ".mkv")

        for fpath in files:
            if not fpath.lower().endswith(supported_exts):
                continue
            fname = os.path.basename(fpath)
            ext = fname.rsplit(".", 1)[-1].upper() if "." in fname else "FILE"
            is_video = ext.lower() in ("mp4", "mov", "avi", "mkv")
            icon = "video" if is_video else "image"

            card = MDCard(
                orientation="horizontal",
                size_hint_y=None,
                height=dp(64),
                padding=dp(12),
                spacing=dp(12),
                elevation=1,
                style="filled",
                md_bg_color=(0.15, 0.15, 0.15, 1),
            )

            icon_widget = MDIcon(
                icon=icon,
                theme_text_color="Custom",
                text_color=COLOR_ACCENT_PRIMARY,
                font_size="24sp",
                size_hint_x=None,
                width=dp(32),
                halign="center",
                valign="center",
            )
            card.add_widget(icon_widget)

            text_box = MDBoxLayout(orientation="vertical", spacing=dp(2))
            name_label = MDLabel(
                text=fname[:30],
                role="medium",
                text_color=COLOR_TEXT_PRIMARY,
            )
            text_box.add_widget(name_label)
            fmt_label = MDLabel(
                text=ext,
                role="small",
                text_color=COLOR_TEXT_SECONDARY,
            )
            text_box.add_widget(fmt_label)
            card.add_widget(text_box)

            card.bind(on_release=partial(self._on_gallery_select, fpath))
            scr.ids.gallery_list.add_widget(card)

        if not scr.ids.gallery_list.children:
            empty_label = MDLabel(
                text="No images or videos found",
                role="medium",
                halign="center",
                theme_text_color="Custom",
                text_color=COLOR_TEXT_SECONDARY,
            )
            scr.ids.gallery_list.add_widget(empty_label)

    def _on_gallery_select(self, file_path, _instance):
        """Handle gallery file selection - route to upload screen."""
        try:
            self._fm_select(file_path)
            self._sm.transition.direction = "left"
            self._sm.current = "upload"
        except Exception as e:
            self.show_snackbar(f"Failed to load file: {e}")

    # ── Phase 7: Status Ribbon & Diagnostics ────────────────────
    def _update_status_ribbon(self):
        """Update the status ribbon properties from current engine state."""
        if self.engine is not None:
            fmt = "tflite" if hasattr(self.engine, "interpreter") else "onnx"
            self.status_model_name = self.mm.active_model_id if self.mm else "Unknown"
            provider = getattr(self.engine, "provider_display", "CPU")
            self.status_provider = provider
        else:
            self.status_model_name = "No Model"
            self.status_provider = "CPU"
        self._update_memory_display()
        self.status_ready = self.engine is not None

    def _update_memory_display(self):
        """Update memory usage display (best-effort, cross-platform)."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            self.status_memory = f"{mem_mb:.0f} MB"
        except (ImportError, OSError):
            self.status_memory = "N/A"

    def _scan_models_directory(self):
        """Scan the models/ directory for .onnx and .tflite files and update status_model_list."""
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        found = []
        if os.path.isdir(models_dir):
            for f in os.listdir(models_dir):
                if f.lower().endswith((".onnx", ".tflite")):
                    found.append(f)
        self.status_model_list = found
        # Also refresh model manager list
        Clock.schedule_once(self._populate_model_manager, 0.1)

    def _on_lighting_persist(self, instance, value):
        """Persist lighting mode to config on change."""
        self.cfg["lighting_mode"] = value
        save_settings(self.cfg)

    def reset_all_settings(self):
        """Reset all settings to factory defaults."""
        self.cfg = reset_settings()
        self.lighting_preset = self.cfg.get("lighting_mode", "blof")
        self.show_snackbar("Settings reset to defaults")
        self._update_status_ribbon()

    def set_hardware_provider(self, provider):
        """Manually set the hardware provider preference."""
        self.cfg["hardware_provider"] = provider
        save_settings(self.cfg)
        self.show_snackbar(f"Provider preference: {provider}")

    # ── Phase 8: Keyboard Shortcuts & Accessibility ──────────────
    def _on_keyboard(self, window, key, scancode, codepoint, modifiers):
        """Global keyboard shortcut handler. Returns True to consume the event."""
        # Skip if text input has focus
        focused = None
        try:
            focused = Window.keyboard_target
        except Exception:
            pass
        if focused is not None:
            return False

        is_ctrl = 'ctrl' in modifiers or 'meta' in modifiers

        if key == 32:  # Spacebar
            self._kb_toggle_detection()
            return True
        elif key == 27:  # Escape
            self._kb_cancel_inference()
            return True
        elif is_ctrl and key == 111:  # Ctrl+O / Cmd+O
            self.open_file_manager()
            return True
        elif is_ctrl and key == 101:  # Ctrl+E / Cmd+E
            self._kb_quick_export()
            return True
        elif key == 9:  # Tab
            self._kb_cycle_viewport()
            return True
        elif key == 91:  # [ (left bracket)
            self._kb_adjust_confidence(-0.05)
            return True
        elif key == 93:  # ] (right bracket)
            self._kb_adjust_confidence(0.05)
            return True
        elif codepoint and codepoint.lower() == 'r':
            self._kb_reset_viewport()
            return True
        return False

    def _kb_toggle_detection(self):
        current = self._sm.current if self.root else ""
        if current == "inference":
            self.toggle_inference()
        elif current == "upload":
            self.ug_handle_action()
        else:
            self.show_snackbar("Space: Start/Pause Detection")

    def _kb_cancel_inference(self):
        if self.ug_processing_active:
            self.ug_cancel_processing()
        elif hasattr(self, '_inf_active') and self._inf_active:
            self.toggle_inference()
        else:
            self.show_snackbar("Escape: Cancel Inference")

    def _kb_quick_export(self):
        if self.ug_has_result:
            self.export_csv_data()
            self.save_annotated_snapshot()
        else:
            self.show_snackbar("No results to export")

    def _kb_cycle_viewport(self):
        current = self._sm.current if self.root else ""
        if current == "inference":
            try:
                scr = self._sm.get_screen("inference")
                raw_img = scr.ids.get("raw_image")
                ann_img = scr.ids.get("ann_image")
                if raw_img and ann_img:
                    if self._viewport_mode == "annotated":
                        if raw_img.opacity < 1:
                            raw_img.opacity = 1
                            raw_img.source = ""
                        self._viewport_mode = "raw"
                    else:
                        raw_img.opacity = 1
                        self._viewport_mode = "annotated"
                    self.show_snackbar(f"View: {self._viewport_mode}")
            except (KeyError, AttributeError):
                pass

    def _kb_adjust_confidence(self, delta):
        new_val = max(0.01, min(0.99, self.cfg.get("conf", 0.25) + delta))
        self.cfg["conf"] = round(new_val, 2)
        save_settings(self.cfg)
        # Update UI slider if visible
        try:
            scr = self._sm.get_screen("inference")
            slider = scr.ids.get("threshold_slider")
            if slider and hasattr(slider, 'slider'):
                slider.slider.value = self.cfg["conf"]
            inp = slider.ids.get("numeric_input") if slider else None
            if inp:
                inp.text = f"{self.cfg['conf']:.2f}"
        except (KeyError, AttributeError):
            pass
        self._refilter_cached_results()
        self.show_snackbar(f"Confidence: {self.cfg['conf']:.2f}")

    def _kb_reset_viewport(self):
        """Reset viewport pan & zoom to 1:1."""
        try:
            scr = self._sm.get_screen("inference")
            scatter = None
            for child in scr.walk():
                if child.__class__.__name__ == 'Scatter':
                    scatter = child
                    break
            if scatter:
                scatter.scale = 1.0
                scatter.pos = scatter.parent.pos if scatter.parent else (0, 0)
                self.show_snackbar("Viewport reset: 1:1")
        except (KeyError, AttributeError):
            pass

    # ── Phase 8: Optical Micron Calibration Tool ─────────────────
    # Predefined magnification to px/µm calibration presets
    # These are approximate defaults; users should calibrate per objective
    CALIBRATION_PRESETS = {
        "4x":  0.15,   # ~0.15 px/µm at 4x (low magnification)
        "10x": 0.40,   # ~0.40 px/µm at 10x
        "40x": 1.60,   # ~1.60 px/µm at 40x
        "100x": 4.00,  # ~4.00 px/µm at 100x
    }

    def set_magnification(self, mag):
        """Set scale factor from a predefined magnification preset."""
        self._magnification = mag
        sf = self.CALIBRATION_PRESETS.get(mag, 0.0)
        self.ug_scale_factor = sf
        self.cfg["scale_factor"] = sf
        self.cfg["magnification"] = mag
        save_settings(self.cfg)
        self.show_snackbar(f"Magnification: {mag} ({sf} px/µm)")
        if self._ug_cached_raw_results:
            self._refilter_cached_results()

    def calibrate_from_scale_bar(self, known_um):
        """Calibrate from a known scale bar length (pixels already measured)."""
        if known_um <= 0:
            self.show_snackbar("Scale bar length must be > 0")
            return
        # User should provide the known µm value; pixels are computed from the UI
        # This is a helper for manual calibration
        self._calibration_um = known_um
        self._calibration_active = True
        self.show_snackbar(f"Draw a line on the viewport equal to {known_um} µm")

    def apply_calibration(self, pixel_length):
        """Apply calibration from a measured pixel length and known µm value."""
        if pixel_length <= 0 or self._calibration_um <= 0:
            self.show_snackbar("Invalid calibration values")
            return
        sf = pixel_length / self._calibration_um
        self.ug_scale_factor = sf
        self.cfg["scale_factor"] = sf
        self._calibration_active = False
        save_settings(self.cfg)
        self.show_snackbar(f"Calibrated: {sf:.4f} px/µm")
        if self._ug_cached_raw_results:
            self._refilter_cached_results()

    def set_lighting_preset(self, preset):
        self.lighting_preset = preset
        self.cfg["lighting_mode"] = preset
        save_settings(self.cfg)
        self.show_snackbar(f"Lighting: {preset}")

    def go(self, screen_name, direction="left"):
        self._sm.transition.direction = direction
        self._sm.current = screen_name
        if screen_name == "gallery":
            Clock.schedule_once(self._populate_gallery, 0.1)

    def show_snackbar(self, message: str, delay: float = 0.0):
        def _do(_):
            # Always log safely to console and HUD first
            print(f"[HUD Notice] {message}")
            try:
                # Only attempt popup if graphics are ready
                if self.root and getattr(self, "_gl_ready", False):
                    snackbar = MDSnackbar(
                        MDSnackbarText(text=message),
                        y=dp(24),
                        pos_hint={"center_x": 0.5},
                        size_hint_x=0.9,
                    )
                    snackbar.open()
            except Exception as e:
                # Catches ALL OpenGL / FBO crashes safely
                pass

        Clock.schedule_once(_do, max(0.1, delay))

    # ── Helpers ───────────────────────────────────────────────────
    def _hide_viewport_placeholder(self):
        """Hide the 'No Micrograph Loaded' placeholder when media is active."""
        self.active_media_type = "loaded"
        try:
            scr = self._sm.get_screen("inference")
            ph = scr.ids.get("viewport_placeholder")
            if ph:
                ph.opacity = 0
                ph.disabled = True
        except (KeyError, AttributeError):
            pass

    def _cv2_to_texture(self, bgr: np.ndarray) -> Texture:
        """Converts BGR OpenCV array to Kivy Texture. Must run on Main Thread."""
        if bgr is None or bgr.size == 0:
            return None
        buf = cv2.flip(bgr, 0).tobytes()
        tex = Texture.create(size=(bgr.shape[1], bgr.shape[0]), colorfmt="bgr")
        tex.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        return tex

    def _engine_badge(self) -> str:
        if self.engine is None:
            return "[NO ENGINE]"
        fmt = "tflite" if hasattr(self.engine, "interpreter") else "onnx"
        base = HUD_BACKEND_BADGE.get(fmt, fmt.upper())
        provider = getattr(self.engine, "provider_display", "")
        if provider:
            return f"{base} {provider}"
        return base

    def _empty_stats(self) -> dict:
        return {c: (0, 0.0) for c in CLASSES}

    def _stats_from_results(self, results: list) -> dict:
        per = {c: [0, 0.0] for c in CLASSES}
        for box, label, conf in results:
            if label in per:
                per[label][0] += 1
                per[label][1] += conf
        return {c: (v[0], v[1] / v[0] if v[0] else 0.0) for c, v in per.items()}

    # ── Phase 5: Morphology & Analytics ──────────────────────────
    def _compute_morphology(self, box) -> dict:
        x1, y1, x2, y2 = box
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        area = w * h
        min_dim = min(w, h) if min(w, h) > 0 else 1.0
        aspect_ratio = max(w, h) / min_dim
        if aspect_ratio < 1.5:
            shape = "Fragment"
        elif aspect_ratio < 2.0:
            shape = "Film"
        elif aspect_ratio < 3.0:
            shape = "Pellet"
        else:
            shape = "Fiber"
        sf = self.ug_scale_factor if self.ug_scale_factor > 0 else 0.0
        area_um2 = area / (sf * sf) if sf > 0 else 0.0
        return {
            "area": area,
            "aspect_ratio": aspect_ratio,
            "shape": shape,
            "area_um2": area_um2,
        }

    def _compute_analytics(self, results: list):
        n = len(results)
        sf = self.ug_scale_factor if self.ug_scale_factor > 0 else 0.0
        if n == 0:
            return {
                "total": 0,
                "mean_area": 0.0,
                "min_area": 0.0,
                "max_area": 0.0,
                "mean_ar": 0.0,
                "mean_area_um": 0.0,
                "min_area_um": 0.0,
                "max_area_um": 0.0,
                "size_buckets": [0, 0, 0, 0],
                "morph_counts": [0, 0, 0, 0],
                "morph_summary": "",
                "size_summary": "",
            }
        morphs = [self._compute_morphology(box) for box, _, _ in results]
        areas = [m["area"] for m in morphs]
        ars = [m["aspect_ratio"] for m in morphs]
        areas_um = [m["area_um2"] for m in morphs] if sf > 0 else []
        size_buckets = [0] * len(SIZE_BUCKETS)
        morph_counts = [0] * len(MORPH_CATEGORIES)
        for m in morphs:
            for i, (_, lo, hi) in enumerate(SIZE_BUCKETS):
                if lo <= m["area"] < hi:
                    size_buckets[i] += 1
                    break
            for i, (_, lo, hi) in enumerate(MORPH_CATEGORIES):
                if lo <= m["aspect_ratio"] < hi:
                    morph_counts[i] += 1
                    break
        morph_labels = [c[0] for c in MORPH_CATEGORIES]
        morph_summary = "  ".join(
            f"{morph_labels[i]}: {morph_counts[i]}" for i in range(len(morph_labels))
        )
        size_labels = [b[0] for b in SIZE_BUCKETS]
        size_summary = "  ".join(
            f"{size_labels[i]}: {size_buckets[i]}" for i in range(len(size_labels))
        )
        mean_area_um = sum(areas_um) / n if areas_um else 0.0
        min_area_um = min(areas_um) if areas_um else 0.0
        max_area_um = max(areas_um) if areas_um else 0.0
        return {
            "total": n,
            "mean_area": sum(areas) / n,
            "min_area": min(areas),
            "max_area": max(areas),
            "mean_ar": sum(ars) / n,
            "mean_area_um": mean_area_um,
            "min_area_um": min_area_um,
            "max_area_um": max_area_um,
            "size_buckets": size_buckets,
            "morph_counts": morph_counts,
            "morph_summary": morph_summary,
            "size_summary": size_summary,
        }

    def _update_analytics_ui(self, analytics: dict):
        self.ug_mean_area = analytics["mean_area"]
        self.ug_min_area = analytics["min_area"]
        self.ug_max_area = analytics["max_area"]
        self.ug_mean_ar = analytics["mean_ar"]
        self.ug_size_counts = analytics["size_buckets"]
        self.ug_morph_counts = analytics["morph_counts"]
        self.ug_size_bucket_text = analytics["size_summary"]
        self.ug_morphology_text = analytics["morph_summary"]
        lighting = self.lighting_preset.upper().replace("_", " ")
        self.ug_illumination_tag = f"Lighting: {lighting}"
        sf = self.ug_scale_factor
        has_um = sf > 0 and analytics["total"] > 0
        if has_um:
            self.ug_size_um_text = (
                f"Size: {analytics['min_area_um']:.1f}–{analytics['max_area_um']:.1f} µm²"
            )
        else:
            self.ug_size_um_text = ""
        lines = []
        if analytics["total"] > 0:
            lines.append(f"Particles: {analytics['total']}")
            lines.append(f"Mean Area: {analytics['mean_area']:.0f} px²")
            if has_um:
                lines.append(f"Mean Area: {analytics['mean_area_um']:.1f} µm²")
            lines.append(f"Size Range: {analytics['min_area']:.0f}–{analytics['max_area']:.0f} px²")
            lines.append(f"Mean Aspect Ratio: {analytics['mean_ar']:.2f}")
        self.ug_analytics_text = "\n".join(lines) if lines else "No particles detected"

    # ── Inference Screen ──────────────────────────────────────────
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

        # Phase 8: Guard against invalid image dimensions
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
            # Phase 8: Guard against invalid frame dimensions
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
                frame,
                text,
                (int(x1) + 2, int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
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
            # New layout uses metric_value IDs
            if f"{k}_count_value" in scr.ids:
                scr.ids[f"{k}_count_value"].text = "0"
            if f"{k}_conf" in scr.ids:
                scr.ids[f"{k}_conf"].text = "0.00"

    def update_split(self, value):
        scr = self._sm.get_screen("inference")
        raw_w = max(0.05, float(value))
        # New layout uses single canvas; split controls sidebar width instead
        if "left_sidebar" in scr.ids:
            scr.ids.left_sidebar.width = dp(220) if value > 0.2 else dp(0)
        if "right_inspector" in scr.ids:
            scr.ids.right_inspector.width = dp(280) if value > 0.8 else dp(0)

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

    # ── Model Manager Screen ──────────────────────────────────────
    def _populate_model_manager(self, dt):
        if self.mm is None:
            return
        scr = self._sm.get_screen("models")
        if "mm_model_list" not in scr.ids:
            return

        scr.ids.mm_model_list.clear_widgets()
        for m in getattr(self.mm, "models", []):
            card = MDCard(
                orientation="vertical",
                size_hint_y=None,
                height=dp(110),
                padding=dp(12),
                spacing=dp(8),
                elevation=1,
                style="filled",
                md_bg_color=(0.15, 0.15, 0.15, 1),
            )
            is_active = m.get("id") == self.mm.active_model_id
            accent = (0.0, 0.898, 1.0, 1.0) if is_active else (0.7, 0.7, 0.7, 1.0)

            # Format badge
            fmt = m.get("format", "").upper()
            fmt_color = (0.0, 0.898, 1.0, 1.0) if fmt == "ONNX" else (0.4627, 1, 0.0118, 1.0)
            fmt_badge = f" [{fmt}]" if fmt else ""

            badge_text = (
                f"[ACTIVE]{fmt_badge} {m.get('name', 'Model')}"
                if is_active
                else f"{fmt_badge} {m.get('name', 'Model')}"
            )
            badge = MDLabel(
                text=badge_text,
                role="medium",
                bold=is_active,
                text_color=accent,
            )
            card.add_widget(badge)

            detail = MDLabel(
                text=f"ID: {m.get('id', 'N/A')}  |  {m.get('path', '')}",
                role="small",
                text_color=(0.5, 0.5, 0.5, 1.0),
            )
            card.add_widget(detail)

            btn_row = MDBoxLayout(size_hint_y=None, height=dp(36), spacing=dp(8))
            model_id = m.get("id")
            model_path = m.get("path", "")

            if not is_active:
                btn = MDButton(
                    MDButtonText(text="SWITCH"),
                    style="filled",
                    size_hint_x=0.3,
                    height=dp(32),
                    on_release=partial(self._on_switch_pressed, model_id),
                )
                btn_row.add_widget(btn)

            val_btn = MDButton(
                MDButtonText(text="VALIDATE"),
                style="outlined",
                size_hint_x=0.3,
                height=dp(32),
                on_release=partial(self._on_validate_pressed, model_path),
            )
            btn_row.add_widget(val_btn)

            if not is_active:
                rm_btn = MDButton(
                    MDButtonText(text="REMOVE"),
                    style="outlined",
                    size_hint_x=0.3,
                    height=dp(32),
                    md_bg_color=(1, 0.2, 0.2, 1),
                    on_release=partial(self._on_remove_model, model_id),
                )
                btn_row.add_widget(rm_btn)

            card.add_widget(btn_row)
            scr.ids.mm_model_list.add_widget(card)

    def _on_switch_pressed(self, model_id, _instance):
        self._mm_switch(model_id)

    def _on_validate_pressed(self, model_path, _instance):
        self._mm_validate(model_path)

    def _on_remove_model(self, model_id, _instance):
        if self.mm is None:
            return
        try:
            self.mm.remove_model(model_id)
            self._populate_model_manager(0)
            self._update_status_ribbon()
            self.show_snackbar(f"Removed model: {model_id}")
        except (KeyError, ValueError) as e:
            self.show_snackbar(f"Failed to remove: {e}")

    def _mm_switch(self, model_id):
        try:
            self.engine = self.mm.switch_model(model_id)
            # Phase 7: Persist active model ID to config
            self.cfg["active_model_id"] = model_id
            save_settings(self.cfg)
            self._update_status_ribbon()
            self.show_snackbar(f"Switched to {model_id}")
            Clock.schedule_once(self._populate_model_manager, 0.1)
        except (RuntimeError, ValueError, OSError) as e:
            self.show_snackbar(f"Switch failed: {e}")

    def _mm_validate(self, rel_path):
        resolved = os.path.join(os.path.dirname(__file__), rel_path)
        result = self.mm.validate_model_file(resolved)
        if result.get("valid", False):
            msg = f"Valid | input={result.get('input_shape')} classes={result.get('num_classes')}"
        else:
            msg = "Invalid: " + "; ".join(result.get("errors", ["Unknown validation issue"]))
        self.show_snackbar(msg)

    def mm_add_model_dialog(self):
        def _on_file_selected(path):
            if not path:
                return
            try:
                entry = self.mm.add_model_from_file(path)
                self._populate_model_manager(0)
                self.show_snackbar(f"Added: {entry.get('name', os.path.basename(path))}")
            except (ValueError, OSError, FileNotFoundError) as e:
                self.show_snackbar(f"Failed to add model: {e}")

        default_dir = os.path.join(os.path.expanduser("~"), "MP Detect", "Models")
        if not os.path.isdir(default_dir):
            default_dir = os.path.expanduser("~")
        if open_file_dialog is not None:
            open_file_dialog(
                title="Select Model File",
                initial_dir=default_dir,
                filetypes=[
                    ("ONNX Model", "*.onnx"),
                    ("TFLite Model", "*.tflite"),
                    ("All files", "*.*"),
                ],
                on_select=_on_file_selected,
                on_cancel=lambda: None,
            )
        else:
            self._open_file_zenity(default_dir)

    # ── Upload Gateway Screen ─────────────────────────────────────
    def _setup_upload_state(self):
        self._ug_file_path = None
        self.ug_file_path = ""
        self._ug_annotated = None
        self._ug_temp_video = None
        self._stop_event = threading.Event()
        self._ug_total_frames = 0
        self._ug_current_frame = 0
        # Phase 5: Cached raw results for threshold recalculation
        self._ug_cached_raw_results = []
        self._ug_cached_annotated_frame = None
        self._ug_cached_elapsed_ms = 0.0
        self._ug_cached_iou = self.cfg.get("iou", 0.45)
        # Phase 5: Scale factor (pixels per μm)
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
                    # Phase 5: Capture at low threshold for re-filtering
                    try:
                        raw = self.engine.detect(
                            frame,
                            conf_thresh=0.01,
                            iou_thresh=self.cfg.get("iou", 0.45),
                        )
                    except (RuntimeError, ValueError, cv2.error) as e:
                        print(f"[ug_thread] detect error frame {frame_count}: {e}")
                        raw = []
                    # Apply current confidence threshold for display
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
                # Phase 5: Capture at low threshold for re-filtering
                try:
                    raw = self.engine.detect(
                        frame,
                        conf_thresh=0.01,
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

    # Phase 5: Instant threshold recalculation from cached results
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
        # Redraw annotated frame from cached raw results
        frame = self._read_cached_frame()
        if frame is not None:
            annotated = self._draw_boxes(frame, filtered)
            scr = self._sm.get_screen("upload")
            if "ug_image" in scr.ids:
                scr.ids.ug_image.texture = self._cv2_to_texture(annotated)
        # Update stats UI
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
        # Update analytics
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

        # Phase 5: Cache raw results and compute analytics
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
        """Load upload detection results into the inference screen viewport."""
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

            # Update results text on inference screen
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
        # Phase 5: Clear cached results and analytics
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

    # ── Phase 6: Structured Data Export & Annotated Media Saving ──
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


if __name__ == "__main__":
    MPDetectApp().run()
