# tests/test_file_handler.py
"""Tests for utils/file_handler.py — file management."""
import os
import tempfile
from unittest.mock import patch

from utils.file_handler import FileHandler


class TestFileHandler:
    def test_init_creates_mp_detect_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("utils.file_handler.os.path.expanduser", return_value=tmpdir):
                with patch("utils.file_handler.platform", "linux"):
                    fh = FileHandler()
                    assert os.path.isdir(fh.dir)

    def test_list_files_returns_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mp_dir = os.path.join(tmpdir, "MP Detect")
            os.makedirs(mp_dir)
            open(os.path.join(mp_dir, "b.txt"), "w").close()
            open(os.path.join(mp_dir, "a.txt"), "w").close()
            open(os.path.join(mp_dir, "c.txt"), "w").close()
            fh = FileHandler()
            fh.dir = mp_dir
            files = fh.list_files()
            assert len(files) == 3
            basenames = [os.path.basename(f) for f in files]
            assert set(basenames) == {"a.txt", "b.txt", "c.txt"}

    def test_list_files_excludes_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mp_dir = os.path.join(tmpdir, "MP Detect")
            os.makedirs(mp_dir)
            open(os.path.join(mp_dir, "file.txt"), "w").close()
            os.makedirs(os.path.join(mp_dir, "subdir"))
            fh = FileHandler()
            fh.dir = mp_dir
            files = fh.list_files()
            assert len(files) == 1

    def test_list_files_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mp_dir = os.path.join(tmpdir, "MP Detect")
            os.makedirs(mp_dir)
            fh = FileHandler()
            fh.dir = mp_dir
            assert fh.list_files() == []

    def test_list_folders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mp_dir = os.path.join(tmpdir, "MP Detect")
            os.makedirs(os.path.join(mp_dir, "folder1"))
            os.makedirs(os.path.join(mp_dir, "folder2"))
            open(os.path.join(mp_dir, "file.txt"), "w").close()
            fh = FileHandler()
            fh.dir = mp_dir
            folders = fh.list_folders()
            assert len(folders) == 2

    def test_get_video_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fh = FileHandler()
            fh.dir = tmpdir
            path = fh.get_video_path()
            assert path.endswith(".mp4")
            assert "VID_" in path

    def test_get_export_csv_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fh = FileHandler()
            fh.dir = tmpdir
            path = fh.get_export_csv_path()
            assert "Results" in path
            assert path.endswith(".csv")
            assert "MP_Detect_Report_" in path

    def test_get_export_report_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fh = FileHandler()
            fh.dir = tmpdir
            path = fh.get_export_report_path()
            assert "Results" in path
            assert path.endswith(".json")
            assert "MP_Detect_Summary_" in path

    def test_get_annotated_image_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fh = FileHandler()
            fh.dir = tmpdir
            path = fh.get_annotated_image_path()
            assert "Results" in path
            assert "Images" in path
            assert path.endswith(".jpg")

    def test_get_annotated_video_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fh = FileHandler()
            fh.dir = tmpdir
            path = fh.get_annotated_video_path()
            assert "Results" in path
            assert "Videos" in path
            assert path.endswith(".mp4")

    def test_ensure_results_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fh = FileHandler()
            fh.dir = tmpdir
            fh._ensure_results_dirs()
            assert os.path.isdir(os.path.join(tmpdir, "Results"))
            assert os.path.isdir(os.path.join(tmpdir, "Results", "Images"))
            assert os.path.isdir(os.path.join(tmpdir, "Results", "Videos"))
