# utils/inference_engine.py
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter, load_delegate  # type: ignore[no-redef]
    except ImportError:
        Interpreter = None
        load_delegate = None

BOX_THICKNESS = 2
FONT_SCALE = 0.5
LABEL_THICKNESS = 1

COLORS = [
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
]


# ── shared helpers ────────────────────────────────────────────────

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


# ── abstract base ─────────────────────────────────────────────────

class BaseInferenceEngine(ABC):
    def __init__(self) -> None:
        self.input_size: int = 640
        self.labels: List[str] = []

    @abstractmethod
    def load_model(self, path: str) -> None: ...

    @abstractmethod
    def preprocess(self, frame: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def infer(self, tensor: np.ndarray) -> Any: ...

    @abstractmethod
    def postprocess(
        self, raw_output: Any, conf_thresh: float, iou_thresh: float
    ) -> list: ...

    def preprocess_letterbox(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        padded, ratio, pad = letterbox(rgb, new_shape=self.input_size)
        tensor = padded.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, :]
        return tensor, ratio, pad

    def rescale_boxes(
        self,
        boxes_xyxy: np.ndarray,
        ratio: Tuple[float, float],
        pad: Tuple[int, int],
        orig_h: int,
        orig_w: int,
    ) -> np.ndarray:
        if boxes_xyxy.size == 0:
            return boxes_xyxy
        boxes = boxes_xyxy.copy()
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes[:, [0, 2]] /= ratio[0]
        boxes[:, [1, 3]] /= ratio[1]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)
        return boxes

    def _parse_yolo_output(
        self,
        pred: np.ndarray,
        ratio: Tuple[float, float],
        pad: Tuple[int, int],
        orig_h: int,
        orig_w: int,
        conf_thresh: float,
        iou_thresh: float,
    ) -> list:
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]

        if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
            cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))

        confs = cls_scores.max(axis=1)
        cls_ids = cls_scores.argmax(axis=1)

        mask = confs >= conf_thresh
        boxes_xywh = boxes_xywh[mask]
        confs = confs[mask]
        cls_ids = cls_ids[mask]

        if boxes_xywh.size == 0:
            return []

        boxes_xywh[:, [0, 2]] *= self.input_size
        boxes_xywh[:, [1, 3]] *= self.input_size
        boxes_xyxy = xywh_to_xyxy(boxes_xywh)
        boxes_xyxy = self.rescale_boxes(boxes_xyxy, ratio, pad, orig_h, orig_w)

        keep = nms(boxes_xyxy, confs, iou_thresh)
        results: list = []
        for idx in keep:
            cid = int(cls_ids[idx])
            label = self.labels[cid] if cid < len(self.labels) else str(cid)
            results.append((
                [int(boxes_xyxy[idx, 0]), int(boxes_xyxy[idx, 1]),
                 int(boxes_xyxy[idx, 2]), int(boxes_xyxy[idx, 3])],
                label,
                float(confs[idx]),
            ))
        return results


# ── ONNX backend ──────────────────────────────────────────────────

class OnnxBackend(BaseInferenceEngine):
    def __init__(self) -> None:
        super().__init__()
        self.session: Optional["ort.InferenceSession"] = None
        self.input_name: str = ""
        self.input_shape: list = []
        self.is_dynamic: bool = False
        self.fixed_size: Optional[int] = None

    def load_model(self, path: str) -> None:
        if ort is None:
            raise ImportError("onnxruntime is not installed")
        # Phase 7: Auto-detect best execution provider
        from utils.detector import select_best_provider, get_provider_display_name
        best_provider = select_best_provider()
        providers_to_use = [best_provider]
        if best_provider != "CPUExecutionProvider":
            providers_to_use.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(path, providers=providers_to_use)
        active = self.session.get_providers()
        self.execution_provider = best_provider if best_provider in active else (active[0] if active else "CPUExecutionProvider")
        self.provider_display = get_provider_display_name(self.execution_provider)
        print(f"[OnnxBackend] Active provider: {self.provider_display}")
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape

        if (
            len(self.input_shape) == 4
            and isinstance(self.input_shape[2], int)
            and isinstance(self.input_shape[3], int)
        ):
            self.fixed_size = self.input_shape[2]
            self.is_dynamic = False
        else:
            self.is_dynamic = True

        self.input_size = self.fixed_size or 640

    def set_input_size(self, size: int) -> None:
        if self.is_dynamic:
            self.input_size = size
        elif self.fixed_size is not None and size != self.fixed_size:
            self.input_size = self.fixed_size

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        tensor, _, _ = self.preprocess_letterbox(frame)
        return tensor

    def infer(self, tensor: np.ndarray) -> Any:
        if self.session is None:
            raise RuntimeError("Model not loaded")
        return self.session.run(None, {self.input_name: tensor})[0]

    def postprocess(
        self, raw_output: Any, conf_thresh: float, iou_thresh: float
    ) -> list:
        orig_h, orig_w = self._last_frame.shape[:2]
        ratio, pad = self._last_ratio, self._last_pad
        return self._parse_yolo_output(
            raw_output, ratio, pad, orig_h, orig_w, conf_thresh, iou_thresh
        )

    def detect(
        self, frame: np.ndarray, conf_thresh: float = 0.25, iou_thresh: float = 0.45
    ) -> list:
        self._last_frame = frame
        tensor, self._last_ratio, self._last_pad = self.preprocess_letterbox(frame)
        raw = self.infer(tensor)
        return self.postprocess(raw, conf_thresh, iou_thresh)


# ── TFLite backend ────────────────────────────────────────────────

class TFLiteBackend(BaseInferenceEngine):
    def __init__(self, use_nnapi: bool = False, delegate_path: Optional[str] = None) -> None:
        super().__init__()
        self.interpreter: Optional["Interpreter"] = None
        self.input_details: list = []
        self.output_details: list = []
        self._use_nnapi = use_nnapi
        self._delegate_path = delegate_path

    def load_model(self, path: str) -> None:
        if Interpreter is None:
            raise ImportError(
                "Neither tflite-runtime nor tensorflow is installed"
            )
        delegates = []
        if self._use_nnapi and load_delegate is not None and self._delegate_path:
            delegates.append(load_delegate(self._delegate_path))

        self.interpreter = Interpreter(
            model_path=path,
            experimental_delegates=delegates if delegates else None,
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        shape = self.input_details[0]["shape"]
        if len(shape) == 4:
            self.input_size = int(max(shape[1], shape[2]))
        else:
            self.input_size = 640

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        tensor, _, _ = self.preprocess_letterbox(frame)
        return tensor

    def infer(self, tensor: np.ndarray) -> Any:
        if self.interpreter is None:
            raise RuntimeError("Model not loaded")
        inp_detail = self.input_details[0]
        if inp_detail["dtype"] == np.float16:
            tensor = tensor.astype(np.float16)
        self.interpreter.set_tensor(inp_detail["index"], tensor)
        self.interpreter.invoke()
        return [
            self.interpreter.get_tensor(od["index"])
            for od in self.output_details
        ]

    def postprocess(
        self, raw_output: Any, conf_thresh: float, iou_thresh: float
    ) -> list:
        out = raw_output[0] if isinstance(raw_output, list) else raw_output
        orig_h, orig_w = self._last_frame.shape[:2]
        ratio, pad = self._last_ratio, self._last_pad
        return self._parse_yolo_output(
            out, ratio, pad, orig_h, orig_w, conf_thresh, iou_thresh
        )

    def detect(
        self, frame: np.ndarray, conf_thresh: float = 0.25, iou_thresh: float = 0.45
    ) -> list:
        self._last_frame = frame
        tensor, self._last_ratio, self._last_pad = self.preprocess_letterbox(frame)
        raw = self.infer(tensor)
        return self.postprocess(raw, conf_thresh, iou_thresh)
