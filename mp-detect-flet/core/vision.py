# core/vision.py - Shared vision utilities
"""Shared vision utilities: letterbox, NMS, IoU, coordinate conversion, drawing."""
from typing import List, Tuple, Optional

import cv2
import numpy as np

# Drawing constants
BOX_THICKNESS = 2
FONT_SCALE = 0.5
LABEL_THICKNESS = 1

# Default colors for detection classes
DEFAULT_COLORS = [
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (128, 0, 255),  # Purple
    (255, 128, 0),  # Orange
]

# Default classes (overridden by model registry)
DEFAULT_CLASSES = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]


def letterbox(
    img: np.ndarray,
    new_shape: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """Resize image with aspect ratio preservation and padding."""
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


def apply_clahe(frame: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Enhances contrast for low-light field conditions.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _iou(b1: np.ndarray, b_arr: np.ndarray) -> np.ndarray:
    """Compute IoU between a box and an array of boxes."""
    x1 = np.maximum(b1[0], b_arr[:, 0])
    y1 = np.maximum(b1[1], b_arr[:, 1])
    x2 = np.minimum(b1[2], b_arr[:, 2])
    y2 = np.minimum(b1[3], b_arr[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b_arr[:, 2] - b_arr[:, 0]) * (b_arr[:, 3] - b_arr[:, 1])
    return inter / (a1 + a2 - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Non-Maximum Suppression using OpenCV's C++ backend when available."""
    if len(boxes) == 0:
        return []
    
    # Try OpenCV's fast NMS first
    try:
        boxes_list = boxes.tolist()
        scores_list = scores.tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_list, scores_list,
            score_threshold=0.0,
            nms_threshold=iou_thr
        )
        if len(indices) > 0:
            return indices.flatten().tolist()
    except Exception:
        pass
    
    # Fallback to numpy NMS
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
    """Convert center-format boxes to corner-format."""
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def draw_boxes(frame: np.ndarray, results: list, 
               classes: Optional[List[str]] = None) -> np.ndarray:
    """Draw bounding boxes on frame with labels and confidence.
    
    Args:
        frame: Input image (BGR)
        results: List of (bbox, label, confidence) tuples
        classes: Optional list of class names for color mapping
    """
    if classes is None:
        classes = DEFAULT_CLASSES
    
    annotated = frame.copy()
    colors = DEFAULT_COLORS
    
    for (x1, y1, x2, y2), label, conf in results:
        idx = classes.index(label) if label in classes else 0
        color = colors[idx % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, BOX_THICKNESS)
        
        # Draw label background
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, LABEL_THICKNESS)
        cv2.rectangle(annotated, (int(x1), int(y1) - th - 6), (int(x1) + tw + 4, int(y1)), color, -1)
        
        # Draw label text
        cv2.putText(
            annotated, text,
            (int(x1) + 2, int(y1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), LABEL_THICKNESS, cv2.LINE_AA,
        )
    
    return annotated


def draw_boxes_with_morphology(frame: np.ndarray, detections: list,
                                classes: Optional[List[str]] = None) -> np.ndarray:
    """Draw bounding boxes with morphology information.
    
    Args:
        frame: Input image (BGR)
        detections: List of ParticleDetection objects
        classes: Optional list of class names for color mapping
    """
    if classes is None:
        classes = DEFAULT_CLASSES
    
    annotated = frame.copy()
    colors = DEFAULT_COLORS
    
    for d in detections:
        idx = classes.index(d.polymer_type) if d.polymer_type in classes else 0
        color = colors[idx % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(annotated, (d.bbox_x1, d.bbox_y1), (d.bbox_x2, d.bbox_y2), color, BOX_THICKNESS)
        
        # Draw label with morphology
        text = f"{d.polymer_type} {d.confidence:.2f}"
        if d.morphology and d.morphology != "Unknown":
            text += f" [{d.morphology[:4]}]"
        
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, LABEL_THICKNESS)
        cv2.rectangle(annotated, (d.bbox_x1, d.bbox_y1 - th - 6), 
                      (d.bbox_x1 + tw + 4, d.bbox_y1), color, -1)
        cv2.putText(
            annotated, text,
            (d.bbox_x1 + 2, d.bbox_y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), LABEL_THICKNESS, cv2.LINE_AA,
        )
    
    return annotated
