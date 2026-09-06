# core/data_models.py - ISO/TR 21960 compliant data models
"""Data models for microplastic detection following ISO/TR 21960 and NOAA standards."""
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import uuid
from datetime import datetime


# ── ISO/TR 21960 Standard Classifications ──────────────────────────────

POLYMER_TYPES = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC", "PA", "PU"]

MORPHOLOGY_TYPES = [
    "Fiber",      # Filaments, threads
    "Fragment",   # Rigid, irregular pieces
    "Film",       # Thin planes (bags, packaging)
    "Foam",       # Expanded, porous structures
    "Pellet",     # Pre-production spherical beads
    "Bead",       # Microbeads (cosmetics)
]

SIZE_CLASSES = {
    "Nanoplastic":  (0, 1e-3),      # < 1 µm (0.001 mm)
    "Small MP":     (1e-3, 1.0),    # 1 µm – 1 mm
    "Large MP":     (1.0, 5.0),     # 1 mm – 5 mm
    "Mesoplastic":  (5.0, 25.0),    # 5 mm – 25 mm
}

ILLUMINATION_MODES = ["BLOF", "UV365", "UV395", "Natural"]
MAGNIFICATION_MODES = ["4x", "10x", "40x", "100x"]


def classify_size_mm(area_px: float, scale_factor: float) -> str:
    """Classify particle size based on bounding box area and scale factor.
    
    Args:
        area_px: Bounding box area in pixels
        scale_factor: Pixels per millimeter (0 = uncalibrated)
    
    Returns:
        Size class string
    """
    if scale_factor <= 0:
        return "Unknown"
    
    # Estimate diameter from area (assuming roughly circular)
    import math
    diameter_mm = 2 * math.sqrt(area_px / math.pi) / scale_factor
    
    for size_name, (min_mm, max_mm) in SIZE_CLASSES.items():
        if min_mm <= diameter_mm < max_mm:
            return size_name
    
    if diameter_mm >= 25.0:
        return "Macro"
    return "Unknown"


def classify_morphology(width: int, height: int) -> str:
    """Classify morphology based on bounding box aspect ratio.
    
    Args:
        width: Bounding box width in pixels
        height: Bounding box height in pixels
    
    Returns:
        Morphology type string
    """
    if width <= 0 or height <= 0:
        return "Unknown"
    
    aspect_ratio = max(width, height) / min(width, height)
    
    if aspect_ratio > 5.0:
        return "Fiber"
    elif aspect_ratio > 3.0:
        return "Filament"
    else:
        return "Fragment"


# ── Data Models ─────────────────────────────────────────────────────────

@dataclass
class ParticleDetection:
    """Single particle detection record following ISO/TR 21960."""
    
    # Identity
    particle_id: int
    session_id: str
    
    # Classification (ISO/TR 21960)
    polymer_type: str          # HDPE, LDPE, PET, PP, PS, PVC, PA, PU
    morphology: str = "Unknown"  # Fiber, Fragment, Film, Foam, Pellet, Bead
    size_class: str = "Unknown"  # Nanoplastic, Small MP, Large MP, Mesoplastic
    
    # Geometry (pixels)
    bbox_x1: int = 0
    bbox_y1: int = 0
    bbox_x2: int = 0
    bbox_y2: int = 0
    width_px: int = 0
    height_px: int = 0
    area_px: float = 0.0
    aspect_ratio: float = 1.0
    size_mm: Optional[float] = None  # Physical size if calibrated
    
    # Confidence
    confidence: float = 0.0
    
    # Context
    timestamp: str = ""
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    illumination: str = "BLOF"
    magnification: str = "10x"
    
    # Model info
    model_id: str = ""
    backend: str = ""  # ONNX, TFLite
    inference_ms: float = 0.0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        # Auto-compute derived fields
        if self.width_px > 0 and self.height_px > 0:
            self.area_px = self.width_px * self.height_px
            self.aspect_ratio = max(self.width_px, self.height_px) / min(self.width_px, self.height_px)
    
    @classmethod
    def from_detection(cls, particle_id: int, session_id: str, bbox: list, 
                       label: str, confidence: float, **kwargs) -> "ParticleDetection":
        """Create from raw detection tuple."""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        morphology = kwargs.pop("morphology", classify_morphology(width, height))
        scale_factor = kwargs.pop("scale_factor", 0.0)
        size_class = classify_size_mm(width * height, scale_factor)
        
        return cls(
            particle_id=particle_id,
            session_id=session_id,
            polymer_type=label,
            morphology=morphology,
            size_class=size_class,
            bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
            width_px=width, height_px=height,
            confidence=confidence,
            **kwargs,
        )


@dataclass
class DetectionSession:
    """Detection session record."""
    
    session_id: str = ""
    started_at: str = ""
    ended_at: Optional[str] = None
    
    # Source
    source_type: str = "image"  # "camera", "image", "video"
    source_path: Optional[str] = None
    
    # Location
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    location_name: Optional[str] = None
    
    # Settings
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    model_id: str = ""
    
    # Results summary
    total_particles: int = 0
    per_class: Dict[str, int] = field(default_factory=dict)
    per_morphology: Dict[str, int] = field(default_factory=dict)
    per_size: Dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    
    # Export status
    csv_exported: bool = False
    json_exported: bool = False
    pdf_exported: bool = False
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())[:8]
        if not self.started_at:
            self.started_at = datetime.now().isoformat()
    
    def update_summary(self, detections: List[ParticleDetection]):
        """Update summary statistics from detections."""
        self.total_particles = len(detections)
        
        if not detections:
            return
        
        # Per-class counts
        self.per_class = {}
        self.per_morphology = {}
        self.per_size = {}
        conf_sum = 0.0
        
        for d in detections:
            self.per_class[d.polymer_type] = self.per_class.get(d.polymer_type, 0) + 1
            self.per_morphology[d.morphology] = self.per_morphology.get(d.morphology, 0) + 1
            self.per_size[d.size_class] = self.per_size.get(d.size_class, 0) + 1
            conf_sum += d.confidence
        
        self.avg_confidence = conf_sum / len(detections)


@dataclass
class ExportOptions:
    """Export configuration."""
    include_csv: bool = True
    include_json: bool = True
    include_pdf: bool = False
    include_image: bool = True
    include_video: bool = False
    output_dir: str = ""
