# utils/__init__.py
"""
Utility package for MP Detect.

- detector.py: ONNX YOLO engine
- file_handler.py: file saving + recent list items
- settings_manager.py: JSON settings load/save
- permissions.py: Android permission requests (Camera, Storage)
"""
__all__ = ["detector", "file_handler", "settings_manager", "permissions"]