[app]

# ── App Metadata ──────────────────────────────────────────────────
title = MP Detect
package.name = mpdetect
package.domain = org.mpdetect
source.dir = .
source.include_exts = py,kv,json,spec,txt
source.include_patterns = assets/*,models/*.onnx,utils/*.py
source.exclude_dirs = tests,venv,.venv,__pycache__,.git,build,dist

# ── Versioning ────────────────────────────────────────────────────
version = 8.0.0

# ── Requirements ──────────────────────────────────────────────────
requirements = python3,kivy,kivymd,opencv-python-headless,onnxruntime,numpy,pillow

# ── Android Permissions ───────────────────────────────────────────
android.permissions = CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,INTERNET

# ── Android API Settings ──────────────────────────────────────────
android.api = 31
android.minapi = 24
android.ndk = 25b
android.sdk = 33
android.accept_sdk_license = True

# ── Build Configuration ──────────────────────────────────────────
android.arch = arm64-v8a
android.release_artifact = apk
android.debug_artifact = debug

# ── Python Bootstrap ──────────────────────────────────────────────
bootstrap = sdl2
fullscreen = 0
orientation = portrait
android.enter_fps = 60
android.exit_fps = 0
android.add_pct = 0
android.ant = None

# ── P4a Recipe (for onnxruntime compatibility) ───────────────────
p4a.branch = develop

# ── Presplash / Icon ──────────────────────────────────────────────
#presplash.filename = %(source.dir)s/assets/presplash.png
#icon.filename = %(source.dir)s/assets/icon.png
#icon.filename = %(source.dir)s/assets/icon_512.png

# ── Window Settings ──────────────────────────────────────────────
window = 0

# ── Log Level ────────────────────────────────────────────────────
log_level = 2

# ── iOS (not used, but kept for reference) ───────────────────────
ios.kivy_ios_url = https://github.com/kivy/kivy-ios
ios.kivy_ios_branch = master
ios.swift_deps = None
ios.objc_deps = None

# ── Desktop-specific overrides ────────────────────────────────────
# For desktop packaging with PyInstaller, see pyinstaller.spec
