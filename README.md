# MP-Detect: Microplastic Detection & Quantification GUI

[![Tests](https://github.com/MP-DETECT-CODE/Mp-Detect/actions/workflows/test.yml/badge.svg)](https://github.com/MP-DETECT-CODE/Mp-Detect/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime-green.svg)](https://onnxruntime.ai/)
[![Kivy Framework](https://img.shields.io/badge/GUI-Kivy-orange.svg)](https://kivy.org/)

**MP-Detect** is an interactive desktop computer vision application designed to automate the detection, localization, and quantification of microplastic particles from images and video streams. Built using the **Kivy** cross-platform framework and powered by **ONNX Runtime**, it provides real-time, high-accuracy inference across varied lighting conditions—including standard optical illumination and UV fluorescence imaging.

---

## Key Features

- **Cross-Platform Interactive GUI**: Modern, touch-ready desktop interface built with Kivy and declarative KV styling (`mpdetect.kv`).
- **High-Performance Inference**: Accelerated execution powered by `ONNX Runtime` (`best.onnx`), optimizing CPU/GPU throughput without heavy PyTorch dependencies in deployment.
- **Multi-Modal Illumination Support**: Evaluated and calibrated for both **Bright Light Optical Filtering (BLOF)** and **Ultraviolet (UV)** fluorescence imaging environments.
- **Batch Image & Video Processing**: Supports static microscope captures as well as continuous stream analysis (`.mp4`, `.avi`) with annotated frame rendering.
- **Configurable Detection Parameters**: Fine-tune confidence thresholds, Non-Maximum Suppression (NMS) IoU thresholds, input tensor sizing, and camera indices via `config.json`.
- **Export & Analytics**: Output automated particle counts, bounding-box coordinate maps, and annotated detection media directly to disk.

---

## Repository Structure

```text
Mp-Detect/
├── assets/                          # App icons and presplash images
├── models/
│   ├── best.onnx                    # Pretrained microplastic detection weights
│   └── model_registry.json          # Model registry (auto-managed)
├── utils/
│   ├── __init__.py
│   ├── controllers.py               # Inference, Upload, and Export controller mixins
│   ├── detector.py                  # Legacy ONNX tensor pre/post-processing & NMS
│   ├── file_dialog.py               # Shared tkinter/zenity file dialog
│   ├── file_handler.py              # IO utilities for images, videos, and logging
│   ├── inference_engine.py          # ONNX/TFLite inference backends
│   ├── media_dispatcher.py          # System file sharing
│   ├── model_manager.py             # Model registry, validation, and switching
│   ├── permissions.py               # OS-level camera & storage permission handlers
│   ├── settings_manager.py          # Configuration parser for runtime adjustments
│   └── vision.py                    # Shared vision utilities (letterbox, NMS, IoU)
├── tests/
│   ├── test_controllers.py          # Tests for controller logic
│   ├── test_file_dialog.py          # Tests for file dialog (mocked)
│   ├── test_file_handler.py         # Tests for file handler
│   ├── test_inference_engine.py     # Tests for ONNX/TFLite backends
│   ├── test_settings_manager.py     # Tests for config persistence
│   └── test_vision.py               # Tests for letterbox, NMS, IoU
├── MP Detect/
│   ├── Results/                     # Sample detection outputs
│   └── Unseen Data/                 # Test datasets
├── .github/workflows/test.yml       # CI/CD: GitHub Actions test workflow
├── buildozer.spec                   # Android build config
├── pyinstaller.spec                 # Desktop build config
├── config.example.json              # Default configuration template
├── main.py                          # Application entry point
├── mpdetect.kv                      # Kivy layout & visual styling
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/MP-DETECT-CODE/Mp-Detect.git
cd Mp-Detect
```

### 2. Create and Activate a Virtual Environment
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note (Linux users):** Kivy relies on system-level OpenGL libraries. On Debian/Ubuntu systems, make sure to install:
> ```bash
> sudo apt install libgl1-mesa-dev libgles2-mesa-dev
> ```

---

## Usage

### Launching the Application
```bash
python main.py
```

### Workflow
1. **Load Media**: Select an individual micrograph image or load a recorded video file (`.mp4`) using the file browser.
2. **Adjust Thresholds**: Access the settings panel to tune the **Confidence Threshold** (e.g., `0.25` - `0.60`) and **NMS IoU Threshold**.
3. **Execute Detection**: Click **Detect** to run inference across frames. Microplastics will be outlined with class tags and individual confidence scores.
4. **Inspect & Export**: Review total particle count summaries and save annotated media or CSV logs to the designated results folder.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

The test suite covers:
- **Vision utilities** (`test_vision.py`): letterbox, IoU, NMS, coordinate conversion
- **File dialog** (`test_file_dialog.py`): tkinter/zenity fallback (fully mocked)
- **File handler** (`test_file_handler.py`): path generation, file listing
- **Settings manager** (`test_settings_manager.py`): config load/save/reset
- **Inference engine** (`test_inference_engine.py`): ONNX/TFLite backends
- **Controllers** (`test_controllers.py`): morphology, analytics, CSV export

---

## Configuration

Runtime behavioral properties can be modified through the in-app settings panel or directly within `config.json`:

```json
{
  "conf": 0.25,
  "iou": 0.45,
  "imgsz": 640,
  "lighting_mode": "blof",
  "active_model_path": "models/best.onnx",
  "active_model_id": "default_onnx",
  "hardware_provider": "auto",
  "scale_factor": 0.0,
  "show_overlay": true,
  "show_scale_bar": true
}
```

See `config.example.json` for the full default configuration.

---

## Detection Capabilities & Performance

The model handles diverse particulate morphologies and illumination contexts:

| Lighting Setup | Evaluation Focus | Typical Challenge Handled |
| :--- | :--- | :--- |
| **BLOF (Bright-Light Optical)** | High-contrast particulate morphology | Shadowing, translucent polymers, sediment noise |
| **UV Fluorescence** | Fluorescent emission response | Uneven fluorescence intensities, overlapping fragments |

---

## Dependencies

- **Python 3.10+**
- **Kivy**: Desktop GUI interface & event management
- **KivyMD**: Material Design components for Kivy
- **ONNX Runtime**: Lightweight neural network execution engine
- **OpenCV (`opencv-python-headless`)**: Video streaming, color conversions, and image rendering
- **NumPy**: Matrix preprocessing, tensor reshaping, and vector math
- **Pillow**: Image processing utilities

---

## Contributing

Contributions, bug reports, and enhancements are welcome:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/Optimization`).
3. Commit your modifications (`git commit -m "Add optimized bounding box post-processing"`).
4. Push to your branch (`git push origin feature/Optimization`).
5. Open a Pull Request.

---

## License

This project is licensed under the [MIT License](LICENSE).
