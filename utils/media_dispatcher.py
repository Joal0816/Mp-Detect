# utils/media_dispatcher.py
import os
import platform
import subprocess
import sys
from typing import Optional

_ANDROID_ARGUMENT = os.environ.get("ANDROID_ARGUMENT")
_IS_ANDROID = _ANDROID_ARGUMENT is not None


def _open_android(file_path: str) -> None:
    try:
        from jnius import autoclass, cast
    except ImportError:
        print("[media_dispatcher] jnius not available on this Android build")
        return

    Intent = autoclass("android.content.Intent")
    FileProvider = autoclass("androidx.core.content.FileProvider")
    PythonActivity = autoclass("org.kivy.android.PythonActivity")

    context = PythonActivity.mActivity
    authority = "org.joal0816.mpdetect.fileprovider"
    uri = FileProvider.getUriForFile(context, authority, file_path)

    intent = Intent(Intent.ACTION_VIEW)
    mime = _guess_mime(file_path)
    intent.setDataAndType(uri, mime)
    intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)

    context.startActivity(intent)


def _guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }.get(ext, "*/*")


def open_in_system_viewer(file_path: str) -> None:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    if _IS_ANDROID:
        _open_android(file_path)
        return

    if sys.platform == "linux":
        subprocess.Popen(["xdg-open", file_path])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", file_path])
    elif sys.platform == "win32":
        os.startfile(file_path)
    else:
        raise OSError(f"Unsupported platform: {sys.platform}")
