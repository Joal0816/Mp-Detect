# core/database.py - SQLite database for persistent storage
"""SQLite database manager for microplastic detection sessions and results."""
import sqlite3
import os
from contextlib import contextmanager
from typing import Optional, List, Dict
from datetime import datetime

from core.data_models import (
    ParticleDetection, DetectionSession, 
    POLYMER_TYPES, MORPHOLOGY_TYPES, SIZE_CLASSES
)


# ── Schema ──────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    source_type TEXT NOT NULL DEFAULT 'image',
    source_path TEXT,
    gps_lat REAL,
    gps_lon REAL,
    location_name TEXT,
    conf_threshold REAL DEFAULT 0.25,
    iou_threshold REAL DEFAULT 0.45,
    model_id TEXT DEFAULT '',
    total_particles INTEGER DEFAULT 0,
    avg_confidence REAL DEFAULT 0.0,
    csv_exported INTEGER DEFAULT 0,
    json_exported INTEGER DEFAULT 0,
    pdf_exported INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    particle_id INTEGER NOT NULL,
    polymer_type TEXT NOT NULL,
    morphology TEXT DEFAULT 'Unknown',
    size_class TEXT DEFAULT 'Unknown',
    bbox_x1 INTEGER DEFAULT 0,
    bbox_y1 INTEGER DEFAULT 0,
    bbox_x2 INTEGER DEFAULT 0,
    bbox_y2 INTEGER DEFAULT 0,
    width_px INTEGER DEFAULT 0,
    height_px INTEGER DEFAULT 0,
    area_px REAL DEFAULT 0.0,
    aspect_ratio REAL DEFAULT 1.0,
    size_mm REAL,
    confidence REAL NOT NULL,
    timestamp TEXT NOT NULL,
    gps_lat REAL,
    gps_lon REAL,
    illumination TEXT DEFAULT 'BLOF',
    magnification TEXT DEFAULT '10x',
    model_id TEXT DEFAULT '',
    backend TEXT DEFAULT '',
    inference_ms REAL DEFAULT 0.0,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_detections_session ON detections(session_id);
CREATE INDEX IF NOT EXISTS idx_detections_polymer ON detections(polymer_type);
CREATE INDEX IF NOT EXISTS idx_detections_morphology ON detections(morphology);
CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
"""


# ── Database Manager ────────────────────────────────────────────────────

class Database:
    """SQLite database for persistent detection storage."""
    
    def __init__(self, db_path: str = "mp_detect.db"):
        self.db_path = db_path
        self._init_db()
    
    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._connection() as conn:
            conn.executescript(SCHEMA)
    
    # ── Session Operations ──────────────────────────────────────────────
    
    def create_session(self, session: DetectionSession) -> str:
        """Create a new detection session."""
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO sessions 
                   (session_id, started_at, source_type, source_path,
                    gps_lat, gps_lon, location_name,
                    conf_threshold, iou_threshold, model_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session.session_id, session.started_at, session.source_type,
                 session.source_path, session.gps_lat, session.gps_lon,
                 session.location_name, session.conf_threshold,
                 session.iou_threshold, session.model_id)
            )
        return session.session_id
    
    def update_session(self, session: DetectionSession):
        """Update session with final statistics."""
        with self._connection() as conn:
            conn.execute(
                """UPDATE sessions SET
                   ended_at=?, total_particles=?, avg_confidence=?,
                   per_class_json=?, per_morphology_json=?, per_size_json=?,
                   csv_exported=?, json_exported=?, pdf_exported=?
                   WHERE session_id=?""",
                (session.ended_at, session.total_particles, session.avg_confidence,
                 str(session.per_class), str(session.per_morphology), str(session.per_size),
                 int(session.csv_exported), int(session.json_exported), int(session.pdf_exported),
                 session.session_id)
            )
    
    def get_session(self, session_id: str) -> Optional[DetectionSession]:
        """Get a session by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if row:
                return self._row_to_session(row)
        return None
    
    def list_sessions(self, limit: int = 50, offset: int = 0) -> List[DetectionSession]:
        """List recent sessions."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
            return [self._row_to_session(row) for row in rows]
    
    def delete_session(self, session_id: str):
        """Delete a session and its detections."""
        with self._connection() as conn:
            conn.execute("DELETE FROM detections WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
    
    def _row_to_session(self, row) -> DetectionSession:
        """Convert database row to DetectionSession."""
        return DetectionSession(
            session_id=row["session_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            source_type=row["source_type"],
            source_path=row["source_path"],
            gps_lat=row["gps_lat"],
            gps_lon=row["gps_lon"],
            location_name=row["location_name"],
            conf_threshold=row["conf_threshold"],
            iou_threshold=row["iou_threshold"],
            model_id=row["model_id"],
            total_particles=row["total_particles"],
            avg_confidence=row["avg_confidence"],
            csv_exported=bool(row["csv_exported"]),
            json_exported=bool(row["json_exported"]),
            pdf_exported=bool(row["pdf_exported"]),
        )
    
    # ── Detection Operations ────────────────────────────────────────────
    
    def add_detection(self, detection: ParticleDetection):
        """Add a single detection record."""
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO detections 
                   (session_id, particle_id, polymer_type, morphology, size_class,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    width_px, height_px, area_px, aspect_ratio, size_mm,
                    confidence, timestamp, gps_lat, gps_lon,
                    illumination, magnification, model_id, backend, inference_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (detection.session_id, detection.particle_id, detection.polymer_type,
                 detection.morphology, detection.size_class,
                 detection.bbox_x1, detection.bbox_y1, detection.bbox_x2, detection.bbox_y2,
                 detection.width_px, detection.height_px, detection.area_px,
                 detection.aspect_ratio, detection.size_mm,
                 detection.confidence, detection.timestamp,
                 detection.gps_lat, detection.gps_lon,
                 detection.illumination, detection.magnification,
                 detection.model_id, detection.backend, detection.inference_ms)
            )
    
    def add_detections(self, detections: List[ParticleDetection]):
        """Add multiple detection records."""
        with self._connection() as conn:
            conn.executemany(
                """INSERT INTO detections 
                   (session_id, particle_id, polymer_type, morphology, size_class,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    width_px, height_px, area_px, aspect_ratio, size_mm,
                    confidence, timestamp, gps_lat, gps_lon,
                    illumination, magnification, model_id, backend, inference_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(d.session_id, d.particle_id, d.polymer_type,
                  d.morphology, d.size_class,
                  d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2,
                  d.width_px, d.height_px, d.area_px,
                  d.aspect_ratio, d.size_mm,
                  d.confidence, d.timestamp,
                  d.gps_lat, d.gps_lon,
                  d.illumination, d.magnification,
                  d.model_id, d.backend, d.inference_ms) for d in detections]
            )
    
    def get_detections(self, session_id: str) -> List[ParticleDetection]:
        """Get all detections for a session."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM detections WHERE session_id=? ORDER BY particle_id",
                (session_id,)
            ).fetchall()
            return [self._row_to_detection(row) for row in rows]
    
    def get_detection_count(self, session_id: str) -> int:
        """Get count of detections for a session."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM detections WHERE session_id=?",
                (session_id,)
            ).fetchone()
            return row["cnt"] if row else 0
    
    def _row_to_detection(self, row) -> ParticleDetection:
        """Convert database row to ParticleDetection."""
        return ParticleDetection(
            particle_id=row["particle_id"],
            session_id=row["session_id"],
            polymer_type=row["polymer_type"],
            morphology=row["morphology"],
            size_class=row["size_class"],
            bbox_x1=row["bbox_x1"],
            bbox_y1=row["bbox_y1"],
            bbox_x2=row["bbox_x2"],
            bbox_y2=row["bbox_y2"],
            width_px=row["width_px"],
            height_px=row["height_px"],
            area_px=row["area_px"],
            aspect_ratio=row["aspect_ratio"],
            size_mm=row["size_mm"],
            confidence=row["confidence"],
            timestamp=row["timestamp"],
            gps_lat=row["gps_lat"],
            gps_lon=row["gps_lon"],
            illumination=row["illumination"],
            magnification=row["magnification"],
            model_id=row["model_id"],
            backend=row["backend"],
            inference_ms=row["inference_ms"],
        )
    
    # ── Analytics ───────────────────────────────────────────────────────
    
    def get_polymer_stats(self, session_id: str) -> Dict[str, int]:
        """Get polymer type breakdown for a session."""
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT polymer_type, COUNT(*) as cnt 
                   FROM detections WHERE session_id=? 
                   GROUP BY polymer_type""",
                (session_id,)
            ).fetchall()
            return {row["polymer_type"]: row["cnt"] for row in rows}
    
    def get_morphology_stats(self, session_id: str) -> Dict[str, int]:
        """Get morphology breakdown for a session."""
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT morphology, COUNT(*) as cnt 
                   FROM detections WHERE session_id=? 
                   GROUP BY morphology""",
                (session_id,)
            ).fetchall()
            return {row["morphology"]: row["cnt"] for row in rows}
    
    def get_size_stats(self, session_id: str) -> Dict[str, int]:
        """Get size class breakdown for a session."""
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT size_class, COUNT(*) as cnt 
                   FROM detections WHERE session_id=? 
                   GROUP BY size_class""",
                (session_id,)
            ).fetchall()
            return {row["size_class"]: row["cnt"] for row in rows}
    
    def get_total_sessions(self) -> int:
        """Get total number of sessions."""
        with self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM sessions").fetchone()
            return row["cnt"] if row else 0
    
    def get_total_detections(self) -> int:
        """Get total number of detections across all sessions."""
        with self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM detections").fetchone()
            return row["cnt"] if row else 0
