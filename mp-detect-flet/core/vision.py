# core/vision.py - Shared vision utilities (ported from utils/vision.py)
"""Shared vision utilities: letterbox, NMS, IoU, coordinate conversion."""
from typing import List, Tuple

import cv2
import numpy as np

# Drawing constants
BOX_THICKNESS = 2
FONT_SCALE = 0.5
LABEL_THICKNESS = 1

COLORS = [
    (0, 0, 255),    # Red     - HDPE
    (255, 255, 0),  # Cyan    - LDPE
    (255, 0, 255),  # Magenta - PET
    (0, 255, 255),  # Yellow  - PP
    (0, 255, 0),    # Green   - PS
    (255, 0, 0),    # Blue    - PVC
]

CLASSES = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]


def letterbox(
    img: np.ndarray,
    new_shape: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    h0, w0 = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, (r, r), (left, top)


cv2_letterbox = letterbox


def _iou(b1: np.ndarray, b_arr: np.ndarray) -> np.ndarray:
    x1 = np.maximum(b1[0], b_arr[:, 0])
    y1 = np.maximum(b1[1], b_arr[:, 1])
    x2 = np.minimum(b1[2], b_arr[:, 2])
    y2 = np.minimum(b1[3], b_arr[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b_arr[:, 2] - b_arr[:, 0]) * (b_arr[:, 3] - b_arr[:, 1])
    return inter / (a1 + a2 - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    idxs = scores.argsort()[::-1]
    keep: List[int] = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(int(i))
        if idxs.size == 1:
            break
        ious = _iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious <= iou_thr]
    return keep


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def draw_boxes(frame: np.ndarray, results: list) -> np.ndarray:
    """Draw bounding boxes on frame."""
    annotated = frame.copy()
    for (x1, y1, x2, y2), label, conf in results:
        idx = CLASSES.index(label) if label in CLASSES else 0
        color = COLORS[idx % len(COLORS)]
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, BOX_THICKNESS)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, LABEL_THICKNESS)
        cv2.rectangle(annotated, (int(x1), int(y1) - th - 6), (int(x1) + tw + 4, int(y1)), color, -1)
        cv2.putText(
            annotated, text,
            (int(x1) + 2, int(y1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), LABEL_THICKNESS, cv2.LINE_AA,
        )
    return annotated
