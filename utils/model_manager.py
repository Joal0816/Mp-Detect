# utils/model_manager.py
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from utils.inference_engine import BaseInferenceEngine, OnnxBackend, TFLiteBackend

_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_registry.json")
_DUMMY_FRAME = np.zeros((640, 640, 3), dtype=np.uint8)


def _load_registry() -> dict:
    with open(_REGISTRY_PATH, "r") as f:
        return json.load(f)


def _save_registry(data: dict) -> None:
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


class ModelManager:
    def __init__(self, registry_path: Optional[str] = None) -> None:
        self._registry_path = registry_path or _REGISTRY_PATH
        self._registry = self._load()
        self._engines: Dict[str, BaseInferenceEngine] = {}

    def _load(self) -> dict:
        with open(self._registry_path, "r") as f:
            return json.load(f)

    def _save(self) -> None:
        with open(self._registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)
            f.write("\n")

    @property
    def active_model_id(self) -> str:
        return self._registry.get("active_model_id", "")

    @property
    def models(self) -> List[dict]:
        return self._registry.get("models", [])

    def _find_model(self, model_id: str) -> Optional[dict]:
        for m in self.models:
            if m["id"] == model_id:
                return m
        return None

    def _build_engine(self, model_info: dict) -> BaseInferenceEngine:
        fmt = model_info["format"].lower()
        if fmt == "onnx":
            engine = OnnxBackend()
        elif fmt == "tflite":
            engine = TFLiteBackend()
        else:
            raise ValueError(f"Unknown format: {fmt}")

        resolved = os.path.join(os.path.dirname(__file__), "..", model_info["path"])
        engine.load_model(resolved)
        engine.labels = list(model_info.get("labels", []))
        return engine

    def get_active_engine(self) -> BaseInferenceEngine:
        mid = self.active_model_id
        if mid in self._engines:
            return self._engines[mid]
        info = self._find_model(mid)
        if info is None:
            raise KeyError(f"Active model '{mid}' not found in registry")
        engine = self._build_engine(info)
        self._engines[mid] = engine
        return engine

    def get_engine(self, model_id: str) -> BaseInferenceEngine:
        if model_id in self._engines:
            return self._engines[model_id]
        info = self._find_model(model_id)
        if info is None:
            raise KeyError(f"Model '{model_id}' not found in registry")
        engine = self._build_engine(info)
        self._engines[model_id] = engine
        return engine

    def validate_model_file(self, file_path: str) -> dict:
        result: dict = {
            "valid": True,
            "path": file_path,
            "exists": os.path.isfile(file_path),
            "input_shape": None,
            "output_shape": None,
            "num_classes": None,
            "bbox_layout": None,
            "errors": [],
        }
        if not result["exists"]:
            result["valid"] = False
            result["errors"].append(f"File not found: {file_path}")
            return result

        ext = Path(file_path).suffix.lower()
        try:
            if ext == ".onnx":
                result = self._validate_onnx(file_path, result)
            elif ext == ".tflite":
                result = self._validate_tflite(file_path, result)
            else:
                result["valid"] = False
                result["errors"].append(f"Unsupported format: {ext}")
        except Exception as e:
            result["valid"] = False
            result["errors"].append(str(e))

        return result

    def _validate_onnx(self, file_path: str, result: dict) -> dict:
        try:
            import onnxruntime as ort
        except ImportError:
            result["valid"] = False
            result["errors"].append("onnxruntime is not installed")
            return result

        sess = ort.InferenceSession(file_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]

        result["input_shape"] = inp.shape
        result["output_shape"] = out.shape

        shape = out.shape
        if len(shape) == 3:
            nc = min(shape[1], shape[2]) - 4
            result["num_classes"] = int(nc) if nc > 0 else None

        try:
            raw = sess.run(None, {inp.name: np.random.randn(*[d if isinstance(d, int) else 1 for d in inp.shape]).astype(np.float32)})
            pred = raw[0]
            if pred.ndim == 3:
                nc = min(pred.shape[1], pred.shape[2]) - 4
                result["bbox_layout"] = "xywh" if nc > 0 else "unknown"
            elif pred.ndim == 2:
                nc = min(pred.shape[0], pred.shape[1]) - 4
                result["bbox_layout"] = "xywh" if nc > 0 else "unknown"
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Dry-run inference failed: {e}")

        return result

    def _validate_tflite(self, file_path: str, result: dict) -> dict:
        try:
            from utils.inference_engine import Interpreter
        except ImportError:
            result["valid"] = False
            result["errors"].append("Neither tflite-runtime nor tensorflow is installed")
            return result

        interp = Interpreter(model_path=file_path)
        interp.allocate_tensors()
        inp_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]

        result["input_shape"] = inp_det["shape"].tolist()
        result["output_shape"] = out_det["shape"].tolist()

        shape = out_det["shape"]
        if len(shape) == 3:
            nc = min(shape[1], shape[2]) - 4
            result["num_classes"] = int(nc) if nc > 0 else None
        elif len(shape) == 2:
            nc = min(shape[0], shape[1]) - 4
            result["num_classes"] = int(nc) if nc > 0 else None

        try:
            interp.invoke()
            out = interp.get_tensor(out_det["index"])
            if out.ndim == 3:
                nc = min(out.shape[1], out.shape[2]) - 4
                result["bbox_layout"] = "xywh" if nc > 0 else "unknown"
            elif out.ndim == 2:
                nc = min(out.shape[0], out.shape[1]) - 4
                result["bbox_layout"] = "xywh" if nc > 0 else "unknown"
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Dry-run inference failed: {e}")

        return result

    def switch_model(self, model_id: str) -> BaseInferenceEngine:
        if model_id == self.active_model_id and model_id in self._engines:
            return self._engines[model_id]

        info = self._find_model(model_id)
        if info is None:
            raise KeyError(f"Model '{model_id}' not found in registry")

        if model_id not in self._engines:
            engine = self._build_engine(info)
            self._engines[model_id] = engine
        else:
            engine = self._engines[model_id]

        self._registry["active_model_id"] = model_id
        self._save()
        return engine

    def remove_model(self, model_id: str) -> None:
        self._registry["models"] = [m for m in self.models if m["id"] != model_id]
        self._engines.pop(model_id, None)
        if self.active_model_id == model_id and self.models:
            self._registry["active_model_id"] = self.models[0]["id"]
        self._save()

    def add_model(
        self,
        model_id: str,
        name: str,
        format_: str,
        path: str,
        input_shape: List[int],
        labels: List[str],
        set_active: bool = False,
    ) -> dict:
        entry = {
            "id": model_id,
            "name": name,
            "format": format_,
            "path": path,
            "input_shape": input_shape,
            "labels": labels,
        }
        existing = self._find_model(model_id)
        if existing:
            existing.update(entry)
        else:
            self._registry["models"].append(entry)

        if set_active:
            self._registry["active_model_id"] = model_id
        self._save()
        return entry
