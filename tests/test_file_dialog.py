# tests/test_file_dialog.py
"""Tests for utils/file_dialog.py — shared file dialog."""
import os
from unittest.mock import MagicMock, patch

from utils.file_dialog import open_file_dialog


class TestOpenFileDialog:
    @patch("utils.file_dialog._open_file_zenity")
    @patch("utils.file_dialog.os.path.isfile", return_value=True)
    def test_tkinter_success_calls_on_select(self, mock_isfile, mock_zenity):
        mock_root = MagicMock()
        mock_path = "/tmp/test_image.png"

        with patch("utils.file_dialog.os.path.isdir", return_value=True):
            with patch("tkinter.Tk", return_value=mock_root):
                with patch("tkinter.filedialog.askopenfilename", return_value=mock_path):
                    on_select = MagicMock()
                    open_file_dialog(
                        title="Test",
                        initial_dir="/tmp",
                        filetypes=[("Images", "*.png")],
                        on_select=on_select,
                    )
                    on_select.assert_called_once_with(mock_path)

    @patch("utils.file_dialog._open_file_zenity")
    @patch("utils.file_dialog.os.path.isfile", return_value=False)
    def test_tkinter_cancel_calls_on_cancel(self, mock_isfile, mock_zenity):
        mock_root = MagicMock()

        with patch("utils.file_dialog.os.path.isdir", return_value=True):
            with patch("tkinter.Tk", return_value=mock_root):
                with patch("tkinter.filedialog.askopenfilename", return_value=""):
                    on_cancel = MagicMock()
                    open_file_dialog(
                        title="Test",
                        initial_dir="/tmp",
                        on_cancel=on_cancel,
                    )
                    on_cancel.assert_called_once()

    @patch("utils.file_dialog._open_file_zenity")
    def test_tkinter_import_error_falls_back_to_zenity(self, mock_zenity):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tkinter":
                raise ImportError("No tkinter")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with patch("utils.file_dialog.os.path.isdir", return_value=True):
                open_file_dialog(title="Test", initial_dir="/tmp")
                mock_zenity.assert_called_once()

    @patch("utils.file_dialog.subprocess.run")
    def test_zenity_success(self, mock_run):
        mock_run.return_value = MagicMock(stdout="/tmp/test.png\n", returncode=0)
        on_select = MagicMock()

        with patch("utils.file_dialog.os.path.isfile", return_value=True):
            result = open_file_dialog(
                title="Test",
                initial_dir="/tmp",
                on_select=on_select,
            )
            assert result == "/tmp/test.png"
            on_select.assert_called_once_with("/tmp/test.png")

    @patch("utils.file_dialog.subprocess.run")
    def test_zenity_cancel(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=1)
        on_cancel = MagicMock()

        with patch("utils.file_dialog.os.path.isfile", return_value=False):
            result = open_file_dialog(
                title="Test",
                initial_dir="/tmp",
                on_cancel=on_cancel,
            )
            assert result is None
            on_cancel.assert_called_once()

    @patch("utils.file_dialog.subprocess.run", side_effect=FileNotFoundError)
    def test_zenity_unavailable(self, mock_run):
        on_cancel = MagicMock()
        result = open_file_dialog(
            title="Test",
            initial_dir="/tmp",
            on_cancel=on_cancel,
        )
        assert result is None
        on_cancel.assert_called_once()

    @patch("utils.file_dialog._open_file_zenity")
    def test_default_initial_dir(self, mock_zenity):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tkinter":
                raise ImportError("No tkinter")
            return real_import(name, *args, **kwargs)

        def mock_isdir(path):
            if "MP Detect" in path:
                return True
            return False

        with patch("builtins.__import__", side_effect=mock_import):
            with patch("utils.file_dialog.os.path.isdir", side_effect=mock_isdir):
                open_file_dialog(title="Test")
                mock_zenity.assert_called_once()
                call_args = mock_zenity.call_args
                assert call_args[1]["initial_dir"].endswith("MP Detect")
