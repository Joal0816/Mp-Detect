# MP Detect - Flet Edition

Microparticle Detection App built with Flet (Flutter) for Android and iOS.

## Features

- **Upload Screen**: Pick images and videos from device
- **Gallery Screen**: Browse media files with grid view and filters
- **Inference Screen**: Run YOLO detection with adjustable parameters
- **Result Screen**: View annotated results and export data
- **Model Manager**: Add, select, and validate detection models

## Installation

### Prerequisites

- Python 3.10+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run App

```bash
python main.py
```

## Build for Android

```bash
# Install Flet CLI
pip install flet-cli

# Build APK
flet build apk

# Output: build/android/app/release/app-release.apk
```

## Build for iOS (macOS only)

```bash
# Build iOS
flet build ios

# Output: build/ios/Runner.ipa
```

## Project Structure

```
mp-detect-flet/
├── main.py                 # App entry point
├── requirements.txt        # Dependencies
├── pyproject.toml          # Project config
├── flet.toml               # Flet build config
│
├── screens/                # UI screens
│   ├── upload_screen.py
│   ├── gallery_screen.py
│   ├── inference_screen.py
│   ├── result_screen.py
│   └── model_manager_screen.py
│
├── components/             # Reusable UI components
│   └── nav_bar.py
│
├── core/                   # Business logic
│   ├── inference_engine.py
│   ├── detector.py
│   ├── model_manager.py
│   ├── vision.py
│   ├── file_handler.py
│   ├── settings_manager.py
│   ├── analytics.py
│   └── export.py
│
├── models/                 # Model files
│   └── model_registry.json
│
└── assets/                 # App assets
    └── icons/
```

## Supported Formats

### Images
- PNG, JPG, JPEG, TIF, TIFF, BMP

### Videos
- MP4, AVI, MOV, MKV

### Models
- ONNX (.onnx)
- TFLite (.tflite)

## Detection Classes

- HDPE (High-Density Polyethylene)
- LDPE (Low-Density Polyethylene)
- PET (Polyethylene Terephthalate)
- PP (Polypropylene)
- PS (Polystyrene)
- PVC (Polyvinyl Chloride)

## License

MIT License
