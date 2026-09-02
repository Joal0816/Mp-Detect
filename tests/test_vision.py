# tests/test_vision.py
"""Tests for utils/vision.py — shared vision utilities."""
import numpy as np
import pytest

from utils.vision import (
    BOX_THICKNESS,
    COLORS,
    FONT_SCALE,
    LABEL_THICKNESS,
    _iou,
    cv2_letterbox,
    letterbox,
    nms,
    xywh_to_xyxy,
)


# ── letterbox ─────────────────────────────────────────────────────

class TestLetterbox:
    def test_square_image_no_resize(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        padded, ratio, pad = letterbox(img, new_shape=640)
        assert padded.shape[:2] == (640, 640)
        assert ratio == (1.0, 1.0)
        assert pad == (0, 0)

    def test_rectangular_image_preserves_aspect(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, ratio, pad = letterbox(img, new_shape=640)
        assert padded.shape[0] == 640
        assert padded.shape[1] == 640
        assert ratio[0] == ratio[1]

    def test_small_image_upscaled(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        padded, ratio, pad = letterbox(img, new_shape=640)
        assert padded.shape[0] == 640
        assert padded.shape[1] == 640
        assert ratio[0] > 1.0

    def test_tall_image_letterboxed(self):
        img = np.zeros((800, 200, 3), dtype=np.uint8)
        padded, ratio, pad = letterbox(img, new_shape=640)
        assert padded.shape[0] == 640
        assert padded.shape[1] == 640

    def test_cv2_letterbox_is_alias(self):
        assert cv2_letterbox is letterbox

    def test_custom_color(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        padded, _, _ = letterbox(img, new_shape=640, color=(255, 0, 0))
        assert padded.shape[:2] == (640, 640)


# ── _iou ──────────────────────────────────────────────────────────

class TestIoU:
    def test_identical_boxes_iou_1(self):
        b1 = np.array([10, 10, 50, 50], dtype=np.float32)
        b2 = np.array([[10, 10, 50, 50]], dtype=np.float32)
        result = _iou(b1, b2)
        assert abs(result[0] - 1.0) < 1e-6

    def test_no_overlap_iou_0(self):
        b1 = np.array([0, 0, 10, 10], dtype=np.float32)
        b2 = np.array([[20, 20, 30, 30]], dtype=np.float32)
        result = _iou(b1, b2)
        assert abs(result[0]) < 1e-6

    def test_partial_overlap(self):
        b1 = np.array([0, 0, 20, 20], dtype=np.float32)
        b2 = np.array([[10, 10, 30, 30]], dtype=np.float32)
        result = _iou(b1, b2)
        # overlap area = 10*10 = 100, union = 400+400-100 = 700
        assert abs(result[0] - 100.0 / 700.0) < 1e-5

    def test_multiple_boxes(self):
        b1 = np.array([0, 0, 20, 20], dtype=np.float32)
        b2 = np.array([
            [0, 0, 20, 20],
            [30, 30, 40, 40],
        ], dtype=np.float32)
        result = _iou(b1, b2)
        assert len(result) == 2
        assert abs(result[0] - 1.0) < 1e-6
        assert abs(result[1]) < 1e-6


# ── nms ───────────────────────────────────────────────────────────

class TestNMS:
    def test_single_box(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9])
        keep = nms(boxes, scores, iou_thr=0.5)
        assert keep == [0]

    def test_no_overlap_keeps_all(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [40, 40, 50, 50],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7])
        keep = nms(boxes, scores, iou_thr=0.5)
        assert len(keep) == 3

    def test_full_overlap_suppresses(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [0, 0, 10, 10],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        keep = nms(boxes, scores, iou_thr=0.5)
        assert len(keep) == 1
        assert keep[0] == 0

    def test_partial_overlap_threshold(self):
        boxes = np.array([
            [0, 0, 20, 20],
            [10, 10, 30, 30],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        keep_high = nms(boxes, scores, iou_thr=0.1)
        keep_low = nms(boxes, scores, iou_thr=0.9)
        assert len(keep_high) == 1
        assert len(keep_low) == 2


# ── xywh_to_xyxy ──────────────────────────────────────────────────

class TestXywhToXyxy:
    def test_center_based_conversion(self):
        xywh = np.array([[50, 50, 20, 10]], dtype=np.float32)
        xyxy = xywh_to_xyxy(xywh)
        expected = np.array([[40, 45, 60, 55]], dtype=np.float32)
        np.testing.assert_array_almost_equal(xyxy, expected)

    def test_multiple_boxes(self):
        xywh = np.array([
            [10, 10, 4, 2],
            [20, 20, 6, 6],
        ], dtype=np.float32)
        xyxy = xywh_to_xyxy(xywh)
        assert xyxy.shape == (2, 4)
        np.testing.assert_array_almost_equal(xyxy[0], [8, 9, 12, 11])
        np.testing.assert_array_almost_equal(xyxy[1], [17, 17, 23, 23])

    def test_zero_size_box(self):
        xywh = np.array([[10, 10, 0, 0]], dtype=np.float32)
        xyxy = xywh_to_xyxy(xywh)
        np.testing.assert_array_almost_equal(xyxy, [[10, 10, 10, 10]])


# ── constants ─────────────────────────────────────────────────────

class TestConstants:
    def test_colors_count(self):
        assert len(COLORS) == 6

    def test_drawing_constants(self):
        assert BOX_THICKNESS == 2
        assert FONT_SCALE == 0.5
        assert LABEL_THICKNESS == 1
