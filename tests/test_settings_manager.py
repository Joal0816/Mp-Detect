# tests/test_settings_manager.py
"""Tests for utils/settings_manager.py — configuration persistence."""
import json
import os
import tempfile
from unittest.mock import patch

from utils.settings_manager import (
    DEFAULTS,
    get_setting,
    load_settings,
    reset_settings,
    save_settings,
    update_setting,
)


class TestLoadSettings:
    def test_load_returns_defaults_when_no_file(self):
        with patch("utils.settings_manager.os.path.exists", return_value=False):
            result = load_settings()
            assert result == DEFAULTS

    def test_load_merges_missing_keys(self):
        partial = {"conf": 0.5}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(partial, f)
            f.flush()
            fname = f.name
        try:
            with patch("utils.settings_manager.CONFIG_PATH", fname):
                result = load_settings()
                assert result["conf"] == 0.5
                for k, v in DEFAULTS.items():
                    assert k in result
        finally:
            os.unlink(fname)

    def test_load_handles_corrupt_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            f.flush()
            fname = f.name
        try:
            with patch("utils.settings_manager.CONFIG_PATH", fname):
                result = load_settings()
                assert result == DEFAULTS
        finally:
            os.unlink(fname)


class TestSaveSettings:
    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            with patch("utils.settings_manager.CONFIG_PATH", path):
                cfg = DEFAULTS.copy()
                cfg["conf"] = 0.75
                save_settings(cfg)
                assert os.path.exists(path)
                with open(path) as f:
                    saved = json.load(f)
                assert saved["conf"] == 0.75

    def test_save_adds_all_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            with patch("utils.settings_manager.CONFIG_PATH", path):
                save_settings({})
                with open(path) as f:
                    saved = json.load(f)
                for k in DEFAULTS:
                    assert k in saved


class TestResetSettings:
    def test_reset_restores_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            with patch("utils.settings_manager.CONFIG_PATH", path):
                cfg = DEFAULTS.copy()
                cfg["conf"] = 0.99
                save_settings(cfg)
                result = reset_settings()
                assert result == DEFAULTS
                with open(path) as f:
                    saved = json.load(f)
                assert saved["conf"] == DEFAULTS["conf"]


class TestGetSetting:
    def test_get_existing_key(self):
        cfg = {"conf": 0.75}
        assert get_setting(cfg, "conf") == 0.75

    def test_get_missing_key_returns_default(self):
        assert get_setting({}, "conf") == DEFAULTS["conf"]


class TestUpdateSetting:
    def test_update_and_persist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            with patch("utils.settings_manager.CONFIG_PATH", path):
                cfg = DEFAULTS.copy()
                result = update_setting(cfg, "iou", 0.6)
                assert result["iou"] == 0.6
                with open(path) as f:
                    saved = json.load(f)
                assert saved["iou"] == 0.6
