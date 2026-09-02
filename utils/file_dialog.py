# utils/file_dialog.py
"""Shared file dialog: tkinter with zenity fallback."""
import os
import subprocess
from typing import Callable, List, Optional, Tuple


def open_file_dialog(
    title: str = "Select File",
    initial_dir: Optional[str] = None,
    filetypes: Optional[List[Tuple[str, str]]] = None,
    on_select: Optional[Callable[[str], None]] = None,
    on_cancel: Optional[Callable[[], None]] = None,
    zenity_filters: Optional[List[str]] = None,
) -> Optional[str]:
    """Open a file dialog using tkinter, with zenity fallback on Linux.

    Args:
        title: Dialog window title.
        initial_dir: Starting directory for the dialog.
        filetypes: List of (label, pattern) pairs for tkinter filetypes.
        on_select: Called with the selected path if a file is chosen.
        on_cancel: Called if the user cancels or no file is selected.
        zenity_filters: Zenity --file-filter strings for fallback.

    Returns:
        Selected file path, or None if cancelled/error.
    """
    if initial_dir is None:
        initial_dir = os.path.join(os.path.expanduser("~"), "MP Detect")
        if not os.path.isdir(initial_dir):
            initial_dir = os.path.expanduser("~")

    if filetypes is None:
        filetypes = [
            ("All files", "*.*"),
        ]

    # Try tkinter first
    try:
        import tkinter as _tk
        from tkinter import filedialog as _filedialog

        root = _tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = _filedialog.askopenfilename(
            initialdir=initial_dir,
            title=title,
            filetypes=filetypes,
        )
        root.destroy()
        if path and os.path.isfile(path):
            if on_select:
                on_select(path)
            return path
        else:
            if on_cancel:
                on_cancel()
            return None
    except Exception:
        pass

    # Zenity fallback (Linux)
    return _open_file_zenity(
        title=title,
        initial_dir=initial_dir,
        filters=zenity_filters,
        on_select=on_select,
        on_cancel=on_cancel,
    )


def _open_file_zenity(
    title: str = "Select File",
    initial_dir: Optional[str] = None,
    filters: Optional[List[str]] = None,
    on_select: Optional[Callable[[str], None]] = None,
    on_cancel: Optional[Callable[[], None]] = None,
) -> Optional[str]:
    """Fallback file dialog using zenity (Linux only)."""
    if initial_dir is None:
        initial_dir = os.path.expanduser("~")

    cmd = [
        "zenity", "--file-selection",
        f"--title={title}",
        f"--filename={initial_dir}/",
    ]
    if filters:
        for f in filters:
            cmd.append(f"--file-filter={f}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
        path = result.stdout.strip()
        if path and os.path.isfile(path):
            if on_select:
                on_select(path)
            return path
        else:
            if on_cancel:
                on_cancel()
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        if on_cancel:
            on_cancel()
        return None
