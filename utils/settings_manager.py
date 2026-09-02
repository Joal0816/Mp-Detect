# utils/settings_manager.py
import json
import os

CONFIG_PATH = "config.json"

# Phase 7: Comprehensive defaults for all runtime parameters
DEFAULTS = {
    # Detection thresholds
    "conf": 0.25,
    "iou": 0.45,
    "imgsz": 640,
    # Lighting mode: "blof", "uv_365", "uv_395"
    "lighting_mode": "blof",
    # Active model path (relative to project root)
    "active_model_path": "models/best.onnx",
    # Active model ID in registry
    "active_model_id": "default_onnx",
    # Hardware provider: "auto", "CUDAExecutionProvider", "CPUExecutionProvider", etc.
    "hardware_provider": "auto",
    # Scale factor for microscopy (pixels per micrometer)
    "scale_factor": 0.0,
    # UI preferences
    "show_overlay": True,
    "show_scale_bar": True,
    "default_export_dir": "",
}


def load_settings():
    """Load settings from config.json, merging with defaults for any missing keys."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in DEFAULTS.items():
                data.setdefault(k, v)
            return data
        except (json.JSONDecodeError, OSError, IOError):
            return DEFAULTS.copy()
    return DEFAULTS.copy()


def save_settings(cfg: dict):
    """Save the full configuration dict to config.json."""
    # Ensure all default keys are present before saving
    for k, v in DEFAULTS.items():
        cfg.setdefault(k, v)
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
    except (OSError, IOError) as e:
        print(f"[SettingsManager] Failed to save settings: {e}")


def reset_settings():
    """Reset config.json to factory defaults."""
    save_settings(DEFAULTS.copy())
    return DEFAULTS.copy()


def get_setting(cfg: dict, key: str):
    """Get a single setting value, returning the default if missing."""
    return cfg.get(key, DEFAULTS.get(key))


def update_setting(cfg: dict, key: str, value) -> dict:
    """Update a single setting and return the updated config."""
    cfg[key] = value
    save_settings(cfg)
    return cfg