# MP-Detect: Microplastic Detection & Quantification GUI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)[cite: 1]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)[cite: 1]
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime-green.svg)](https://onnxruntime.ai/)[cite: 1]
[![Kivy Framework](https://img.shields.io/badge/GUI-Kivy-orange.svg)](https://kivy.org/)[cite: 1]

**MP-Detect** is an interactive desktop computer vision application designed to automate the detection, localization, and quantification of microplastic particles from images and video streams[cite: 1]. Built using the **Kivy** cross-platform framework and powered by **ONNX Runtime**, it provides real-time, high-accuracy inference across varied lighting conditions—including standard optical illumination and UV fluorescence imaging[cite: 1].

---

## Key Features

- **Cross-Platform Interactive GUI**: Modern, touch-ready desktop interface built with Kivy and declarative KV styling (`ui.kv`)[cite: 1].
- **High-Performance Inference**: Accelerated execution powered by `ONNX Runtime` (`best.onnx`), optimizing CPU/GPU throughput without heavy PyTorch dependencies in deployment[cite: 1].
- **Multi-Modal Illumination Support**: Evaluated and calibrated for both **Bright Light Optical Filtering (BLOF)** and **Ultraviolet (UV)** fluorescence imaging environments[cite: 1].
- **Batch Image & Video Processing**: Supports static microscope captures as well as continuous stream analysis (`.mp4`, `.avi`) with annotated frame rendering[cite: 1].
- **Configurable Detection Parameters**: Fine-tune confidence thresholds, Non-Maximum Suppression (NMS) IoU thresholds, input tensor sizing, and camera indices via `config.json`[cite: 1].
- **Export & Analytics**: Output automated particle counts, bounding-box coordinate maps, and annotated detection media directly to disk[cite: 1].

---

## Repository Structure

```text
MPDetect-GUI/
├── assets/
│   └── app_icon.png              # Application logo and window branding
├── models/
│   └── best.onnx                 # Pretrained microplastic detection weights
├── utils/
│   ├── __init__.py
│   ├── detector.py               # ONNX tensor pre/post-processing & NMS logic
│   ├── file_handler.py           # IO utilities for images, videos, and logging
│   ├── permissions.py            # OS-level camera & storage permission handlers
│   └── settings_manager.py       # Configuration parser for runtime adjustments
├── MP Detect/
│   ├── Results/                  # Sample detection outputs and processed videos
│   └── Unseen Data/              # Test datasets (BLOF & UV test images/videos)
├── config.json                   # User settings & detection thresholds
├── main.py                       # Application lifecycle & event controller
├── ui.kv                         # Kivy layout architecture & visual styling
├── requirements.txt              # Production dependencies
└── README.md
```[cite: 1]

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Joal0816/Mp-Detect.git](https://github.com/Joal0816/Mp-Detect.git)
cd Mp-Detect
```[cite: 1]

### 2. Create and Activate a Virtual Environment
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```[cite: 1]

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```[cite: 1]

> **Note (Linux users):** Kivy relies on system-level OpenGL libraries[cite: 1]. On Debian/Ubuntu systems, make sure to install[cite: 1]:
> ```bash
> sudo apt install libgl1-mesa-dev libgles2-mesa-dev
> ```[cite: 1]

---

## Usage

### Launching the Application
Execute the primary entry script[cite: 1]:
```bash
python main.py
```[cite: 1]

### Workflow
1. **Load Media**: Select an individual micrograph image or load a recorded video file (`.mp4`) using the file browser[cite: 1].
2. **Adjust Thresholds**: Access the settings panel to tune the **Confidence Threshold** (e.g., `0.35` - `0.60`) and **NMS IoU Threshold**[cite: 1].
3. **Execute Detection**: Click **Detect** to run inference across frames[cite: 1]. Microplastics will be outlined with class tags and individual confidence scores[cite: 1].
4. **Inspect & Export**: Review total particle count summaries and save annotated media or CSV logs to the designated results folder[cite: 1].

---

## Configuration (`config.json`)

Runtime behavioral properties can be modified through the in-app settings panel or directly within `config.json`[cite: 1]:

```json
{
  "confidence_threshold": 0.45,
  "nms_iou_threshold": 0.50,
  "input_size": [640, 640],
  "model_path": "models/best.onnx",
  "save_annotated_results": true,
  "output_directory": "MP Detect/Results"
}
```[cite: 1]

---

## Detection Capabilities & Performance

The model handles diverse particulate morphologies and illumination contexts[cite: 1]:

| Lighting Setup | Evaluation Focus | Typical Challenge Handled |
| :--- | :--- | :--- |
| **BLOF (Bright-Light Optical)** | High-contrast particulate morphology | Shadowing, translucent polymers, sediment noise |
| **UV Fluorescence** | Fluorescent emission response | Uneven fluorescence intensities, overlapping fragments |[cite: 1]

---

## Dependencies

- **Python 3.10+**[cite: 1]
- **Kivy**: Desktop GUI interface & event management[cite: 1]
- **ONNX Runtime**: Lightweight neural network execution engine[cite: 1]
- **OpenCV (`opencv-python`)**: Video streaming, color conversions, and image rendering[cite: 1]
- **NumPy**: Matrix preprocessing, tensor reshaping, and vector math[cite: 1]

---

## Contributing

Contributions, bug reports, and enhancements are welcome[cite: 1]:
1. Fork the repository[cite: 1].
2. Create your feature branch (`git checkout -b feature/Optimization`)[cite: 1].
3. Commit your modifications (`git commit -m "Add optimized bounding box post-processing"`)[cite: 1].
4. Push to your branch (`git push origin feature/Optimization`)[cite: 1].
5. Open a Pull Request[cite: 1].

---

## License

This project is licensed under the [MIT License](LICENSE)[cite: 1].