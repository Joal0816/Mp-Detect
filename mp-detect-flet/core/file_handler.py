# core/file_handler.py - File I/O (adapted for Flet)
"""File handler for MP Detect Flet app."""
import os
import time
import shutil
import cv2
import sys


def get_platform() -> str:
    """Detect current platform."""
    if sys.platform == "linux" and os.path.exists("/system/build.prop"):
        return "android"
    elif sys.platform == "darwin":
        return "ios" if os.path.exists("/System/Library/CoreServices/SystemVersion.plist") else "macos"
    elif sys.platform == "win32":
        return "windows"
    else:
        return "linux"


class FileHandler:
    def __init__(self):
        platform = get_platform()
        
        if platform == "android":
            self.base_dir = "/storage/emulated/0/DCIM"
            self.dir = os.path.join(self.base_dir, "MP Detect")
        else:
            self.base_dir = os.getcwd()
            self.dir = os.path.join(self.base_dir, "MP Detect")

        if not os.path.exists(self.dir):
            try:
                os.makedirs(self.dir, exist_ok=True)
            except OSError:
                pass
        self._ensure_results_dirs()

    def _ensure_results_dirs(self):
        for sub in ("Results", "Results/Images", "Results/Videos"):
            d = os.path.join(self.dir, sub)
            if not os.path.exists(d):
                try:
                    os.makedirs(d, exist_ok=True)
                except OSError:
                    pass

    def _ensure_folder(self, folder_name):
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

    def get_export_csv_path(self) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.dir, "Results", f"MP_Detect_Report_{ts}.csv")

    def get_export_report_path(self) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.dir, "Results", f"MP_Detect_Summary_{ts}.json")

    def get_annotated_image_path(self, suffix="annotated") -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.dir, "Results", "Images", f"IMG_{suffix}_{ts}.jpg")

    def get_annotated_video_path(self, suffix="annotated") -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.dir, "Results", "Videos", f"VID_{suffix}_{ts}.mp4")

    def list_folders(self):
        if not os.path.exists(self.dir):
            return []
        try:
            items = (os.path.join(self.dir, d) for d in os.listdir(self.dir))
            folders = [d for d in items if os.path.isdir(d)]
            return sorted(folders)
        except Exception:
            return []

    def list_files(self, folder_path=None):
        target = folder_path if folder_path else self.dir
        if not os.path.exists(target):
            return []
        try:
            files = (os.path.join(target, f) for f in os.listdir(target))
            files = [f for f in files if os.path.isfile(f)]
            return sorted(files, key=os.path.getmtime, reverse=True)
        except Exception:
            return []

    def list_media_files(self, folder_path=None):
        """List only media files (images and videos)."""
        all_files = self.list_files(folder_path)
        supported_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
                          ".mp4", ".mov", ".avi", ".mkv")
        return [f for f in all_files if f.lower().endswith(supported_exts)]

    def is_video(self, file_path: str) -> bool:
        ext = file_path.lower().rsplit(".", 1)[-1] if "." in file_path else ""
        return ext in ("mp4", "mov", "avi", "mkv")

    def is_image(self, file_path: str) -> bool:
        ext = file_path.lower().rsplit(".", 1)[-1] if "." in file_path else ""
        return ext in ("png", "jpg", "jpeg", "tif", "tiff", "bmp")
