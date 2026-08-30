# utils/file_handler.py
import os
import time
import shutil
import cv2
from kivy.utils import platform

class FileHandler:
    def __init__(self, results_dir=""):
        # Base DCIM path for Android
        if platform == "android":
            self.base_dir = "/storage/emulated/0/DCIM"
            # Explicitly named "MP Detect" per user request
            self.dir = os.path.join(self.base_dir, "MP Detect")
        else:
            # Desktop fallback
            self.base_dir = os.getcwd()
            self.dir = os.path.join(self.base_dir, "MP Detect")

        if not os.path.exists(self.dir):
            try:
                os.makedirs(self.dir, exist_ok=True)
            except OSError:
                pass

    def _ensure_folder(self, folder_name):
        # Ensure folders are always inside the MP Detect root
        # If folder_name is "MP Detect", we just return root
        if folder_name.strip() == "MP Detect":
            return self.dir

        target_dir = os.path.join(self.dir, folder_name)
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir, exist_ok=True)
            except OSError:
                return self.dir
        return target_dir

    def save_image(self, bgr_image) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.dir, f"IMG_{ts}.jpg")
        cv2.imwrite(path, bgr_image)
        return path

    def save_custom_image(self, bgr_image, filename, folder_name) -> str:
        save_dir = self._ensure_folder(folder_name)
        if not filename.lower().endswith((".jpg", ".png")):
            filename += ".jpg"
        path = os.path.join(save_dir, filename)
        cv2.imwrite(path, bgr_image)
        return path

    def save_custom_video(self, temp_path, filename, folder_name) -> str:
        save_dir = self._ensure_folder(folder_name)
        if not filename.lower().endswith(".mp4"):
            filename += ".mp4"
        final_path = os.path.join(save_dir, filename)
        shutil.move(temp_path, final_path)
        return final_path

    def get_video_path(self) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.dir, f"VID_{ts}.mp4")

    def list_folders(self):
        """List folders only inside the MP Detect directory"""
        if not os.path.exists(self.dir): return []
        try:
            items = (os.path.join(self.dir, d) for d in os.listdir(self.dir))
            folders = [d for d in items if os.path.isdir(d)]
            return sorted(folders)
        except Exception:
            return []

    def list_files(self, folder_path=None):
        """List files in specific folder (or default MP Detect dir)"""
        target = folder_path if folder_path else self.dir
        if not os.path.exists(target): return []
        try:
            files = (os.path.join(target, f) for f in os.listdir(target))
            files = [f for f in files if os.path.isfile(f)]
            return sorted(files, key=os.path.getmtime, reverse=True)
        except Exception:
            return []