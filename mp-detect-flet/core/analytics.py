# core/analytics.py - Simplified stats computation
"""Analytics module for MP Detect (simplified for mobile)."""
from typing import List, Tuple, Dict

from core.vision import CLASSES


def compute_stats(results: List[Tuple]) -> Dict:
    """
    Compute simplified detection statistics.
    
    Args:
        results: List of (bbox, label, confidence) tuples
        
    Returns:
        Dictionary with stats
    """
    if not results:
        return {
            "total": 0,
            "avg_conf": 0.0,
            "per_class": {c: {"count": 0, "avg_conf": 0.0} for c in CLASSES},
        }
    
    total = len(results)
    conf_sum = sum(conf for _, _, conf in results)
    avg_conf = conf_sum / total if total > 0 else 0.0
    
    per_class = {c: {"count": 0, "conf_sum": 0.0} for c in CLASSES}
    
    for _, label, conf in results:
        if label in per_class:
            per_class[label]["count"] += 1
            per_class[label]["conf_sum"] += conf
    
    # Calculate averages
    for c in CLASSES:
        count = per_class[c]["count"]
        if count > 0:
            per_class[c]["avg_conf"] = per_class[c]["conf_sum"] / count
        else:
            per_class[c]["avg_conf"] = 0.0
        del per_class[c]["conf_sum"]
    
    return {
        "total": total,
        "avg_conf": avg_conf,
        "per_class": per_class,
    }


def compute_area_stats(results: List[Tuple]) -> Dict:
    """
    Compute area-based statistics for detected particles.
    
    Args:
        results: List of (bbox, label, confidence) tuples
        
    Returns:
        Dictionary with area stats
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
