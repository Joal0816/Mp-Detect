# core/export.py - CSV, JSON, video export
"""Export module for MP Detect."""
import csv
import json
import os
import time
from datetime import datetime
from typing import List, Tuple, Dict

import cv2
import numpy as np

from core.vision import draw_boxes, CLASSES


def export_csv(
    results: List[Tuple],
    output_path: str,
    settings: Dict,
    lighting_preset: str = "blof",
) -> str:
    """
    Export detection results to CSV.
    
    Args:
        results: List of (bbox, label, confidence) tuples
        output_path: Path to save CSV
        settings: Detection settings
        lighting_preset: Current lighting mode
        
    Returns:
        Path to saved CSV
    """
    rows = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for i, (box, label, conf) in enumerate(results):
        x1, y1, x2, y2 = box
        bw = abs(x2 - x1)
        bh = abs(y2 - y1)
        area = bw * bh
        
        rows.append({
            "particle_id": i + 1,
            "timestamp": ts,
            "class_label": label,
            "confidence": round(conf, 4),
            "x": round(float(x1), 1),
            "y": round(float(y1), 1),
            "width": round(float(bw), 1),
            "height": round(float(bh), 1),
            "area_px": round(area, 1),
            "illumination": lighting_preset,
            "conf_thresh": settings.get("conf", 0.25),
            "iou_thresh": settings.get("iou", 0.45),
        })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return output_path


def export_json_report(
    results: List[Tuple],
    output_path: str,
    settings: Dict,
    file_path: str,
    engine_badge: str,
    lighting_preset: str = "blof",
) -> str:
    """
    Export detection results to JSON report.
    
    Args:
        results: List of (bbox, label, confidence) tuples
        output_path: Path to save JSON
        settings: Detection settings
        file_path: Source file path
        engine_badge: Engine description string
        lighting_preset: Current lighting mode
        
    Returns:
        Path to saved JSON
    """
    from core.analytics import compute_stats
    
    stats = compute_stats(results)
    
    report = {
        "app": "MP Detect",
        "version": "2.0.0",
        "generated_at": datetime.now().isoformat(),
        "source_file": os.path.basename(file_path) if file_path else "",
        "model_backend": engine_badge,
        "parameters": {
            "conf_threshold": settings.get("conf", 0.25),
            "iou_threshold": settings.get("iou", 0.45),
            "illumination_mode": lighting_preset,
        },
        "summary": {
            "total_particles": stats["total"],
            "avg_confidence": round(stats["avg_conf"], 3),
        },
        "class_breakdown": {
            k: {"count": v["count"], "mean_confidence": round(v["avg_conf"], 3)}
            for k, v in stats["per_class"].items()
        },
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return output_path


def export_annotated_image(
    frame: "np.ndarray",
    results: List[Tuple],
    output_path: str,
) -> str:
    """
    Export annotated image with bounding boxes.
    
    Args:
        frame: Original image (BGR)
        results: List of (bbox, label, confidence) tuples
        output_path: Path to save image
        
    Returns:
        Path to saved image
    """
    annotated = draw_boxes(frame.copy(), results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    return output_path


def export_annotated_video(
    source_path: str,
    results: List[Tuple],
    output_path: str,
    settings: Dict,
) -> str:
    """
    Export annotated video with bounding boxes.
    
    Args:
        source_path: Path to source video
        results: List of (bbox, label, confidence) tuples
        output_path: Path to save annotated video
        settings: Detection settings
        
    Returns:
        Path to saved video
    """
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open source video")
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Try different codecs
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not writer.isOpened():
        writer.release()
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not writer.isOpened():
        writer.release()
        fourcc = cv2.VideoWriter_fourcc(*"X264")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not writer.isOpened():
        raise RuntimeError("Cannot create video writer with any codec")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        annotated = draw_boxes(frame, results)
        writer.write(annotated)
        frame_count += 1
    
    cap.release()
    writer.release()
    
    return output_path
