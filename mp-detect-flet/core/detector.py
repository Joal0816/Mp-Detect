# core/detector.py - YOLO detection (ported from utils/detector.py)
"""YOLO detection with auto-provider selection."""
import os
import cv2
import numpy as np
from typing import List, Tuple

from core.vision import cv2_letterbox, nms, xywh_to_xyxy, CLASSES

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Execution provider priority order
_EXECUTION_PROVIDER_PRIORITY = [
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
    "DirectMLExecutionProvider",
    "CPUExecutionProvider",
]

_PROVIDER_DISPLAY_NAMES = {
    "CUDAExecutionProvider": "CUDA (GPU)",
    "TensorrtExecutionProvider": "TensorRT (GPU)",
    "DirectMLExecutionProvider": "DirectML (GPU)",
    "CPUExecutionProvider": "CPU",
}


def detect_available_providers() -> List[str]:
    """Detect which ONNX Runtime execution providers are available."""
    if ort is None:
        return ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    return [p for p in _EXECUTION_PROVIDER_PRIORITY if p in available]


def select_best_provider() -> str:
    """Auto-select the optimal execution provider with graceful fallback to CPU."""
    available = detect_available_providers()
    if not available:
        return "CPUExecutionProvider"
    for provider in _EXECUTION_PROVIDER_PRIORITY:
        if provider in available:
            print(f"[Detector] Selected execution provider: {provider}")
            return provider
    return "CPUExecutionProvider"


def get_provider_display_name(provider: str) -> str:
    """Return a human-friendly display name for an execution provider."""
    return _PROVIDER_DISPLAY_NAMES.get(provider, provider)


class YOLODetector:
    def __init__(self, model_path: str, conf_thresh=0.25, iou_thresh=0.45, input_size=640):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing ONNX: {model_path}")

        if ort is None:
            raise ImportError("onnxruntime is required for YOLODetector")
        self.execution_provider = select_best_provider()
        providers_to_use = [self.execution_provider]
        if self.execution_provider != "CPUExecutionProvider":
            providers_to_use.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(model_path, providers=providers_to_use)

        active_providers = self.session.get_providers()
        if self.execution_provider in active_providers:
            self.active_provider = self.execution_provider
        else:
            self.active_provider = active_providers[0] if active_providers else "CPUExecutionProvider"

        self.provider_display = get_provider_display_name(self.active_provider)
        print(f"[Detector] Active provider: {self.provider_display}")

        model_inputs = self.session.get_inputs()[0]
        self.input_name = model_inputs.name
        self.input_shape = model_inputs.shape

        self.is_dynamic = False
        self.fixed_size = None

        if len(self.input_shape) == 4 and isinstance(self.input_shape[2], int) and isinstance(self.input_shape[3], int):
            self.fixed_size = self.input_shape[2]
            print(f"[Detector] Model has STATIC input size: {self.fixed_size}")
        else:
            self.is_dynamic = True
            print("[Detector] Model supports DYNAMIC input size.")

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        if self.fixed_size is not None:
            self.input_size = self.fixed_size
        else:
            self.input_size = input_size

        self.classes = CLASSES

    def set_params(self, conf_thresh, iou_thresh, input_size):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        if self.is_dynamic:
            self.input_size = input_size
        elif self.fixed_size is not None and input_size != self.fixed_size:
            print(f"[Detector Warning] Request for {input_size} ignored. Model requires {self.fixed_size}.")

    def _empty_summary(self):
        return {"per_class": {c: (0, 0.0) for c in self.classes}, "total": 0, "avg_conf": 0.0}

    def detect(self, bgr_frame, conf_thresh=None, iou_thresh=None):
        _conf = conf_thresh if conf_thresh is not None else self.conf_thresh
        _iou = iou_thresh if iou_thresh is not None else self.iou_thresh
        original_frame = bgr_frame.copy()

        rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        padded, ratio, pad = cv2_letterbox(rgb, new_shape=self.input_size)
        x = padded.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[np.newaxis, :]

        out = self.session.run(None, {self.input_name: x})[0]

        if out.ndim != 3:
            return self._empty_summary(), original_frame, []
        if out.shape[1] < out.shape[2]:
            out = np.transpose(out, (0, 2, 1))
        pred = out[0]

        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]
        if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
            cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))

        confs = cls_scores.max(axis=1)
        cls_ids = cls_scores.argmax(axis=1)

        m = confs >= _conf
        boxes_xywh = boxes_xywh[m]
        confs = confs[m]
        cls_ids = cls_ids[m]

        per_class = {c: [0, 0.0] for c in self.classes}
        total = 0
        conf_sum = 0.0
        raw_detections = []

        if boxes_xywh.size > 0:
            boxes_xywh[:, [0, 2]] *= self.input_size
            boxes_xywh[:, [1, 3]] *= self.input_size
            boxes_xyxy = xywh_to_xyxy(boxes_xywh)
            boxes_xyxy[:, [0, 2]] -= pad[0]
            boxes_xyxy[:, [1, 3]] -= pad[1]
            boxes_xyxy[:, [0, 2]] /= ratio[0]
            boxes_xyxy[:, [1, 3]] /= ratio[1]

            h, w = original_frame.shape[:2]
            boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w - 1)
            boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h - 1)

            keep = nms(boxes_xyxy, confs, _iou)
            boxes_xyxy = boxes_xyxy[keep]
            confs = confs[keep]
            cls_ids = cls_ids[keep]

            for ((x1, y1, x2, y2), c, cid) in zip(boxes_xyxy, confs, cls_ids):
                cid = int(cid)
                if cid < 0 or cid >= len(self.classes):
                    continue
                label = self.classes[cid]
                per_class[label][0] += 1
                per_class[label][1] += float(c)
                total += 1
                conf_sum += float(c)
                raw_detections.append(([int(x1), int(y1), int(x2), int(y2)], label, float(c)))

        else:
            boxes_xyxy = np.array([])
            confs = np.array([])
            cls_ids = np.array([])

        avg_per_class = {k: (v[0], (v[1] / v[0] if v[0] else 0.0)) for k, v in per_class.items()}
        overall = (conf_sum / total) if total else 0.0
        det_stats = {"per_class": avg_per_class, "total": total, "avg_conf": overall}

        return raw_detections


Detector = YOLODetector
