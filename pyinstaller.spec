# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for MP Detect - Desktop Standalone Build
# Usage: pyinstaller pyinstaller.spec

import os
import sys

block_cipher = None

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(SPEC))
SRC_DIR = BASE_DIR

# ── Data files to bundle ──────────────────────────────────────────
datas = [
    (os.path.join(SRC_DIR, 'mpdetect.kv'), '.'),
    (os.path.join(SRC_DIR, 'config.example.json'), '.'),
    (os.path.join(SRC_DIR, 'models'), 'models'),
    (os.path.join(SRC_DIR, 'assets'), 'assets'),
]

# ── Hidden imports for KivyMD and ONNX Runtime ───────────────────
hiddenimports = [
    'kivymd',
    'kivymd.uix',
    'kivymd.uix.button',
    'kivymd.uix.card',
    'kivymd.uix.label',
    'kivymd.uix.screen',
    'kivymd.uix.screenmanager',
    'kivymd.uix.boxlayout',
    'kivymd.uix.filemanager',
    'kivymd.uix.snackbar',
    'kivymd.uix.selectioncontrol',
    'kivymd.uix.behaviors',
    'kivymd.uix.behaviors.toggle_behavior',
    'kivymd.app',
    'kivy.core.window',
    'kivy.core.image',
    'kivy.graphics',
    'kivy.properties',
    'kivy.clock',
    'kivy.factory',
    'kivy.uix.boxlayout',
    'kivy.uix.scrollview',
    'kivy.uix.image',
    'onnxruntime',
    'numpy',
    'cv2',
    'PIL',
    'utils.model_manager',
    'utils.file_handler',
    'utils.settings_manager',
    'utils.detector',
    'utils.inference_engine',
    'utils.media_dispatcher',
    'utils.permissions',
]

# ── Analysis ──────────────────────────────────────────────────────
a = Analysis(
    [os.path.join(SRC_DIR, 'main.py')],
    pathex=[SRC_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'pytest',
        'unittest',
        'test',
        'distutils',
        'setuptools',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ── PYZ (Python archive) ─────────────────────────────────────────
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── EXE ──────────────────────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MPDetect',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Windowed app, no console
    icon=None,  # Set to icon.ico if available
)

# ── COLLECT (one-folder distribution) ────────────────────────────
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MPDetect',
)

# ── BUNDLE (one-file distribution, macOS .app) ──────────────────
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='MPDetect.app',
        icon=None,  # Set to icon.icns if available
        bundle_identifier='org.mpdetect.app',
        info_plist={
            'CFBundleDisplayName': 'MP Detect',
            'CFBundleShortVersionString': '8.1.0',
            'NSHighResolutionCapable': True,
        },
    )
