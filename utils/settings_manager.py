# utils/settings_manager.py
import json
import os

CONFIG_PATH = "config.json"
# Defaults (Dark mode removed)
DEFAULTS = {"conf": 0.25, "iou": 0.45, "imgsz": 640}

def load_settings():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in DEFAULTS.items():
                data.setdefault(k, v)
            return data
        except Exception:
            return DEFAULTS.copy()
    return DEFAULTS.copy()

def save_settings(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)