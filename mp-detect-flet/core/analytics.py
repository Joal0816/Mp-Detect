# core/analytics.py - Detection analytics
"""Analytics module for microplastic detection statistics."""
from typing import List, Dict, Tuple
from collections import defaultdict

from core.vision import DEFAULT_CLASSES
from core.data_models import (
    ParticleDetection, classify_morphology, classify_size_mm,
    MORPHOLOGY_TYPES, SIZE_CLASSES
)


def compute_stats(results: List[Tuple]) -> Dict:
    """Compute detection statistics from raw results.
    
    Args:
        results: List of (bbox, label, confidence) tuples
    
    Returns:
        Dictionary with detection statistics
    """
    if not results:
        return {
            "total": 0,
            "avg_conf": 0.0,
            "per_class": {c: {"count": 0, "avg_conf": 0.0} for c in DEFAULT_CLASSES},
        }
    
    total = len(results)
    conf_sum = 0.0
    
    per_class: Dict[str, Dict] = {c: {"count": 0, "conf_sum": 0.0} for c in DEFAULT_CLASSES}
    
    for _, label, conf in results:
        # Handle both old format (string) and new format (dict)
        if isinstance(label, dict):
            polymer = label.get("polymer_type", "Unknown")
        else:
            polymer = label
        
        if polymer not in per_class:
            per_class[polymer] = {"count": 0, "conf_sum": 0.0}
        per_class[polymer]["count"] += 1
        per_class[polymer]["conf_sum"] += conf
        conf_sum += conf
    
    # Calculate averages
    for c in per_class:
        count = per_class[c]["count"]
        per_class[c]["avg_conf"] = per_class[c]["conf_sum"] / count if count > 0 else 0.0
        del per_class[c]["conf_sum"]
    
    return {
        "total": total,
        "avg_conf": conf_sum / total if total > 0 else 0.0,
        "per_class": per_class,
    }


def compute_detailed_stats(detections: List[ParticleDetection]) -> Dict:
    """Compute detailed statistics from ParticleDetection objects.
    
    Includes polymer type, morphology, and size class breakdowns.
    """
    if not detections:
        return {
            "total": 0,
            "avg_conf": 0.0,
            "per_class": {},
            "per_morphology": {},
            "per_size": {},
        }
    
    total = len(detections)
    conf_sum = 0.0
    
    per_class: Dict[str, int] = defaultdict(int)
    per_morphology: Dict[str, int] = defaultdict(int)
    per_size: Dict[str, int] = defaultdict(int)
    
    for d in detections:
        per_class[d.polymer_type] += 1
        per_morphology[d.morphology] += 1
        per_size[d.size_class] += 1
        conf_sum += d.confidence
    
    return {
        "total": total,
        "avg_conf": conf_sum / total if total > 0 else 0.0,
        "per_class": dict(per_class),
        "per_morphology": dict(per_morphology),
        "per_size": dict(per_size),
    }


def compute_area_stats(results: List[Tuple]) -> Dict:
    """Compute area-based statistics for detected particles.
    
    Args:
        results: List of (bbox, label, confidence) tuples
    
    Returns:
        Dictionary with area statistics
    """
    if not results:
        return {
            "mean_area": 0.0,
            "min_area": 0.0,
            "max_area": 0.0,
        }
    
    areas = []
    for (x1, y1, x2, y2), _, _ in results:
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        areas.append(w * h)
    
    return {
        "mean_area": sum(areas) / len(areas),
        "min_area": min(areas),
        "max_area": max(areas),
    }


def compute_size_distribution(detections: List[ParticleDetection], 
                               scale_factor: float = 0.0) -> Dict[str, int]:
    """Compute size class distribution.
    
    Args:
        detections: List of ParticleDetection objects
        scale_factor: Pixels per millimeter (0 = uncalibrated)
    
    Returns:
        Dictionary with size class counts
    """
    size_counts: Dict[str, int] = defaultdict(int)
    
    for d in detections:
        if d.size_class and d.size_class != "Unknown":
            size_counts[d.size_class] += 1
        elif scale_factor > 0 and d.area_px > 0:
            size_class = classify_size_mm(d.area_px, scale_factor)
            size_counts[size_class] += 1
    
    return dict(size_counts)


def compute_morphology_distribution(detections: List[ParticleDetection]) -> Dict[str, int]:
    """Compute morphology distribution.
    
    Args:
        detections: List of ParticleDetection objects
    
    Returns:
        Dictionary with morphology counts
    """
    morph_counts: Dict[str, int] = defaultdict(int)
    
    for d in detections:
        if d.morphology and d.morphology != "Unknown":
            morph_counts[d.morphology] += 1
    
    return dict(morph_counts)


def compute_confidence_distribution(results: List[Tuple], 
                                     bins: int = 5) -> Dict[str, int]:
    """Compute confidence score distribution.
    
    Args:
        results: List of (bbox, label, confidence) tuples
        bins: Number of bins for distribution
    
    Returns:
        Dictionary with bin labels and counts
    """
    if not results:
        return {}
    
    bin_width = 1.0 / bins
    bin_labels = [f"{i*bin_width:.1f}-{(i+1)*bin_width:.1f}" for i in range(bins)]
    bin_counts = [0] * bins
    
    for _, _, conf in results:
        bin_idx = min(int(conf / bin_width), bins - 1)
        bin_counts[bin_idx] += 1
    
    return dict(zip(bin_labels, bin_counts))


def compute_aspect_ratio_stats(results: List[Tuple]) -> Dict:
    """Compute aspect ratio statistics (useful for fiber detection).
    
    Args:
        results: List of (bbox, label, confidence) tuples
    
    Returns:
        Dictionary with aspect ratio statistics
    """
    if not results:
        return {
            "mean_aspect_ratio": 1.0,
            "min_aspect_ratio": 1.0,
            "max_aspect_ratio": 1.0,
            "fiber_like_count": 0,
        }
    
    aspect_ratios = []
    for (x1, y1, x2, y2), _, _ in results:
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if w > 0 and h > 0:
            ar = max(w, h) / min(w, h)
            aspect_ratios.append(ar)
    
    if not aspect_ratios:
        return {
            "mean_aspect_ratio": 1.0,
            "min_aspect_ratio": 1.0,
            "max_aspect_ratio": 1.0,
            "fiber_like_count": 0,
        }
    
    return {
        "mean_aspect_ratio": sum(aspect_ratios) / len(aspect_ratios),
        "min_aspect_ratio": min(aspect_ratios),
        "max_aspect_ratio": max(aspect_ratios),
        "fiber_like_count": sum(1 for ar in aspect_ratios if ar > 5.0),
    }
