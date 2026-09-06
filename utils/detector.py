# utils/detector.py
import os
import cv2
import time
import numpy as np
from typing import List, Tuple

from utils.vision import COLORS, BOX_THICKNESS, FONT_SCALE, LABEL_THICKNESS, cv2_letterbox, _iou, nms, xywh_to_xyxy

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Phase 7: Execution provider priority order
_EXECUTION_PROVIDER_PRIORITY = [
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
    "DirectMLExecutionProvider",
    "CPUExecutionProvider",
]

# Friendly display names for providers
_PROVIDER_DISPLAY_NAMES = {
    "CUDAExecutionProvider": "CUDA (GPU)",
    "TensorrtExecutionProvider": "TensorRT (GPU)",
    "DirectMLExecutionProvider": "DirectML (GPU)",
    "CPUExecutionProvider": "CPU",
}


def detect_available_providers() -> List[str]:
    """Detect which ONNX Runtime execution providers are available on this system."""
    if ort is None:
        return ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    return [p for p in _EXECUTION_PROVIDER_PRIORITY if p in available]


def select_best_provider() -> str:
    """Auto-select the optimal execution provider with graceful fallback to CPU."""
    available = detect_available_providers()
    if not available:
        return "CPUExecutionProvider"
    # Pick highest priority available
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

        # Phase 7: Auto-detect and use the best available execution provider
        if ort is None:
            raise ImportError("onnxruntime is required for YOLODetector")
        self.execution_provider = select_best_provider()
        providers_to_use = [self.execution_provider]
        # Always include CPU as fallback
        if self.execution_provider != "CPUExecutionProvider":
            providers_to_use.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(model_path, providers=providers_to_use)

        # Verify which provider is actually active (ORT may fall back silently)
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

        self.classes = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]

    def set_params(self, conf_thresh, iou_thresh, input_size):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        if self.is_dynamic:
            self.input_size = input_size
        elif self.fixed_size is not None and input_size != self.fixed_size:
            print(f"[Detector Warning] Request for {input_size} ignored. Model requires {self.fixed_size}.")

    def get_video_capture(self, index=0, width=1920, height=1080):
        cap = cv2.VideoCapture(index)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        except Exception:
            pass
        return cap

    def _empty_summary(self):
        return {"per_class": {c: (0, 0.0) for c in self.classes}, "total": 0, "avg_conf": 0.0}

    def add_stats_overlay(self, img, stats):
        if stats is None:
            stats = self._empty_summary()

        h, w = img.shape[:2]
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        display_order = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS"]
        rows = []
        for cls_name in display_order:
            cnt, avg = stats["per_class"].get(cls_name, (0, 0.0))
            rows.append((cls_name, str(cnt), f"{avg:.2f}"))

        total_row = ("TOTAL", str(stats['total']), f"{stats['avg_conf']:.2f}")

        line_height = 28
        num_lines = 1 + 1 + 1 + len(self.classes) + 1 + 1
        box_height = (num_lines * line_height) + 30
        box_width = 300

        x1, y1 = 20, h - box_height - 20
        x2, y2 = x1 + box_width, h - 20

        col1_x = x1 + 15
        col2_center = x1 + 140
        col3_center = x1 + 240

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        color = (255, 255, 255)
        thick = 1
        aa = cv2.LINE_AA

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        y_text = y1 + 30

        cv2.putText(img, timestamp, (col1_x, y_text), font, scale, color, thick, aa)
        y_text += line_height

        cv2.putText(img, "TYPE", (col1_x, y_text), font, scale, color, thick, aa)
        w_txt, _ = cv2.getTextSize("COUNT", font, scale, thick)[0]
        cv2.putText(img, "COUNT", (col2_center - w_txt // 2, y_text), font, scale, color, thick, aa)
        w_txt, _ = cv2.getTextSize("AVG CONF", font, scale, thick)[0]
        cv2.putText(img, "AVG CONF", (col3_center - w_txt // 2, y_text), font, scale, color, thick, aa)
        y_text += line_height

        cv2.putText(img, "-" * 17, (col1_x, y_text), font, scale, color, thick, aa)
        y_text += line_height

        for (name, cnt, conf) in rows:
            cv2.putText(img, name, (col1_x, y_text), font, scale, color, thick, aa)
            w_txt, _ = cv2.getTextSize(cnt, font, scale, thick)[0]
            cv2.putText(img, cnt, (col2_center - w_txt // 2, y_text), font, scale, color, thick, aa)
            w_txt, _ = cv2.getTextSize(conf, font, scale, thick)[0]
            cv2.putText(img, conf, (col3_center - w_txt // 2, y_text), font, scale, color, thick, aa)
            y_text += line_height

        cv2.putText(img, "-" * 17, (col1_x, y_text), font, scale, color, thick, aa)
        y_text += line_height

        t_name, t_cnt, t_conf = total_row
        cv2.putText(img, t_name, (col1_x, y_text), font, scale, color, thick, aa)
        w_txt, _ = cv2.getTextSize(t_cnt, font, scale, thick)[0]
        cv2.putText(img, t_cnt, (col2_center - w_txt // 2, y_text), font, scale, color, thick, aa)
        w_txt, _ = cv2.getTextSize(t_conf, font, scale, thick)[0]
        cv2.putText(img, t_conf, (col3_center - w_txt // 2, y_text), font, scale, color, thick, aa)

        return img

    def _draw_detections(self, frame, boxes_xyxy, confs, cls_ids):
        annotated = frame.copy()

        for ((x1, y1, x2, y2), conf, cls_id) in zip(boxes_xyxy.astype(int), confs, cls_ids):
            cls_id = int(cls_id)
            if cls_id < 0 or cls_id >= len(self.classes):
                continue

            label = self.classes[cls_id]
            color = COLORS[cls_id % len(COLORS)]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=BOX_THICKNESS)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, LABEL_THICKNESS)
            cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), LABEL_THICKNESS,
                        cv2.LINE_AA)

        return annotated

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

                # Append raw: ( [x1, y1, x2, y2], class_name, conf )
                raw_detections.append(([int(x1), int(y1), int(x2), int(y2)], label, float(c)))

        else:
            boxes_xyxy = np.array([])
            confs = np.array([])
            cls_ids = np.array([])

        avg_per_class = {k: (v[0], (v[1] / v[0] if v[0] else 0.0)) for k, v in per_class.items()}
        overall = (conf_sum / total) if total else 0.0
        det_stats = {"per_class": avg_per_class, "total": total, "avg_conf": overall}

        annotated = self._draw_detections(original_frame, boxes_xyxy, confs, cls_ids)

        # RETURN: stats, image, AND raw_detections
        return det_stats, annotated, raw_detections

    def detect_from_file(self, path: str):
        pl = path.lower()
        if pl.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            img = cv2.imread(path)
            if img is None: raise RuntimeError("Failed to read image")
            # Image logic doesn't need tracking, but matches signature
            return self.detect(img)
        else:
            raise ValueError("Unsupported file type")

    def process_video(self, source_path, dest_path):
        """
        Original process_video kept for compatibility, but main logic
        will now handle tracking in main.py loop to update UI in real-time.
        This function returns summary stats for the whole video (without tracking logic applied here).
        """
        cap = cv2.VideoCapture(source_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_index = total_frames // 2 if total_frames > 0 else -1

        writer = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        total_stats = {"per_class": {c: [0, 0.0] for c in self.classes}, "total": 0, "avg_conf": 0.0}
        frame_count = 0
        thumbnail_frame = None
        last_frame = None

        while True:
            ret, frame = cap.read()
            if not ret: break

            det, annotated, _ = self.detect(frame)
            final_frame = self.add_stats_overlay(annotated, det)

            if frame_count == middle_index:
                thumbnail_frame = final_frame.copy()

            writer.write(final_frame)
            last_frame = final_frame

            for c, (cnt, conf) in det["per_class"].items():
                total_stats["per_class"][c][0] += cnt
                total_stats["per_class"][c][1] += conf
            total_stats["total"] += det["total"]
            total_stats["avg_conf"] += det["avg_conf"]

            frame_count += 1

        cap.release()
        writer.release()

        if frame_count > 0:
            final_summary = {"per_class": {}, "total": 0, "avg_conf": 0.0}
            final_summary["total"] = int(total_stats["total"] / frame_count)
            final_summary["avg_conf"] = total_stats["avg_conf"] / frame_count

            for c, vals in total_stats["per_class"].items():
                avg_cnt = int(vals[0] / frame_count)
                avg_conf = vals[1] / frame_count
                final_summary["per_class"][c] = (avg_cnt, avg_conf)

            if thumbnail_frame is None:
                thumbnail_frame = last_frame if last_frame is not None else np.zeros((height, width, 3), dtype=np.uint8)

            return final_summary, thumbnail_frame
        else:
            return self._empty_summary(), np.zeros((height, width, 3), dtype=np.uint8)


Detector = YOLODetector

# Provide explicit infer alias if only detect exists
if not hasattr(YOLODetector, 'infer') and hasattr(YOLODetector, 'detect'):
    YOLODetector.infer = YOLODetector.detect