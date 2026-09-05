# core/settings_manager.py - Settings management
"""Settings manager for MP Detect Flet app."""
import json
import os

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "settings.json")

DEFAULT_SETTINGS = {
    "conf": 0.25,
    "iou": 0.45,
    "imgsz": 640,
    "lighting_mode": "blof",
    "active_model_id": "",
    "scale_factor": 0.0,
    "magnification": "10x",
}


def load_settings() -> dict:
    """Load settings from file, or return defaults."""
    if not os.path.exists(SETTINGS_PATH):
        return DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_PATH, "r") as f:
            settings = json.load(f)
        # Merge with defaults for missing keys
        for key, value in DEFAULT_SETTINGS.items():
            if key not in settings:
                settings[key] = value
        return settings
    except (json.JSONDecodeError, OSError):
        return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> None:
    """Save settings to file."""
    try:
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")
    except OSError as e:
        print(f"[Settings] Failed to save: {e}")


def reset_settings() -> dict:
    """Reset settings to defaults and save."""
    settings = DEFAULT_SETTINGS.copy()
    save_settings(settings)
    return settings


def update_setting(settings: dict, key: str, value) -> dict:
    """Update a single setting and save."""
    settings[key] = value
    save_settings(settings)
    return settings
