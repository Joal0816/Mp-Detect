# core/camera.py - High-performance camera capture with frame dropping
"""Camera capture system with non-blocking frame buffer and inference decoupling."""
import cv2
import numpy as np
import threading
import time
from collections import deque
from typing import Optional, Callable


class FrameBuffer:
    """Non-blocking frame buffer with automatic dropping."""
    
    def __init__(self, max_size: int = 1):
        self._buffer = None
        self._lock = threading.Lock()
        self._max_size = max_size
        self._drop_count = 0
    
    def put(self, frame: np.ndarray) -> bool:
        """Put frame in buffer, drop if full. Returns True if frame was kept."""
        with self._lock:
            if self._buffer is not None:
                self._drop_count += 1
                return False
            self._buffer = frame
            return True
    
    def get(self) -> Optional[np.ndarray]:
        """Get frame from buffer (non-blocking)."""
        with self._lock:
            frame = self._buffer
            self._buffer = None
            return frame
    
    def get_drop_count(self) -> int:
        """Get number of dropped frames."""
        with self._lock:
            count = self._drop_count
            self._drop_count = 0
            return count


class HighPerformanceCamera:
    """Camera capture with decoupled processing and frame dropping."""
    
    def __init__(self, camera_id: int = 0, target_fps: int = 15):
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        
        # Frame buffer - only keeps latest frame
        self._frame_buffer = FrameBuffer(max_size=1)
        
        # Processing callback
        self._process_callback: Optional[Callable] = None
        
        # Performance tracking
        self._fps_counter = deque(maxlen=60)
        self._last_frame_time = 0
        self._frame_interval = 1.0 / target_fps
        
        # State
        self._lock = threading.Lock()
        self._current_frame: Optional[np.ndarray] = None
    
    def start(self, process_callback: Optional[Callable] = None):
        """Start camera capture."""
        with self._lock:
            if self._running:
                return
            
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_id}")
            
            # Optimize camera settings for performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            
            self._process_callback = process_callback
            self._running = True
        
        # Start capture thread
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._capture_thread.start()
        
        # Start processing thread
        if process_callback:
            self._process_thread = threading.Thread(
                target=self._process_loop, daemon=True
            )
            self._process_thread.start()
    
    def stop(self):
        """Stop camera capture."""
        with self._lock:
            self._running = False
            cap = self.cap
            self.cap = None
        
        if cap:
            cap.release()
        
        # Wait for threads to finish
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._process_thread:
            self._process_thread.join(timeout=2.0)
    
    def _capture_loop(self):
        """High-speed capture loop - only captures frames."""
        while True:
            with self._lock:
                if not self._running:
                    break
                cap = self.cap
            
            if cap is None or not cap.isOpened():
                break
            
            frame_start = time.time()
            
            # Capture frame (non-blocking with buffer=1)
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.001)
                continue
            
            # Put frame in buffer (drops if full)
            self._frame_buffer.put(frame)
            self._current_frame = frame
            
            # Track FPS
            self._fps_counter.append(time.time())
            
            # Adaptive sleep to target FPS
            elapsed = time.time() - frame_start
            sleep_time = max(0, self._frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _process_loop(self):
        """Processing loop - runs inference on latest frame."""
        while True:
            with self._lock:
                if not self._running:
                    break
            
            # Get latest frame (non-blocking)
            frame = self._frame_buffer.get()
            if frame is None:
                time.sleep(0.001)
                continue
            
            # Process frame
            if self._process_callback:
                try:
                    self._process_callback(frame)
                except Exception as e:
                    print(f"[Camera] Process error: {e}")
    
    def get_fps(self) -> float:
        """Get current FPS."""
        now = time.time()
        self._fps_counter = deque(
            [t for t in self._fps_counter if now - t < 1.0],
            maxlen=60
        )
        return len(self._fps_counter)
    
    def get_drop_count(self) -> int:
        """Get number of dropped frames."""
        return self._frame_buffer.get_drop_count()
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current frame (for display)."""
        return self._current_frame
    
    @property
    def is_running(self) -> bool:
        return self._running
