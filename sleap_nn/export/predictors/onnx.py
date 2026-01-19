"""ONNX Runtime predictor."""

from __future__ import annotations

from typing import Dict, Iterable, Optional
import time

import numpy as np

from sleap_nn.export.predictors.base import ExportPredictor


class ONNXPredictor(ExportPredictor):
    """ONNX Runtime inference with provider selection."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        providers: Optional[Iterable[str]] = None,
    ) -> None:
        """Initialize ONNX predictor with execution providers.

        Args:
            model_path: Path to the ONNX model file.
            device: Device for inference ("auto", "cpu", or "cuda").
            providers: ONNX Runtime execution providers. Auto-selected if None.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNXPredictor. Install with "
                "`pip install onnxruntime` or `onnxruntime-gpu`."
            ) from exc

        # Preload CUDA/cuDNN libraries from pip-installed nvidia packages
        # This is required for onnxruntime-gpu to find the CUDA libraries
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()

        self.ort = ort
        if providers is None:
            providers = _select_providers(device, ort.get_available_providers())

        self.session = ort.InferenceSession(model_path, providers=list(providers))
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_type = input_info.type
        self.input_dtype = _onnx_type_to_numpy(self.input_type)
        self.output_names = [out.name for out in self.session.get_outputs()]

    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on a batch of images."""
        image = _as_numpy(image, expected_dtype=self.input_dtype)
        outputs = self.session.run(None, {self.input_name: image})
        return dict(zip(self.output_names, outputs))

    def benchmark(
        self, image: np.ndarray, n_warmup: int = 50, n_runs: int = 200
    ) -> Dict[str, float]:
        """Benchmark inference latency and throughput."""
        image = _as_numpy(image, expected_dtype=self.input_dtype)
        for _ in range(n_warmup):
            self.session.run(None, {self.input_name: image})

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            self.session.run(None, {self.input_name: image})
            times.append(time.perf_counter() - start)

        times_ms = np.array(times) * 1000.0
        mean_ms = float(times_ms.mean())
        p50_ms = float(np.percentile(times_ms, 50))
        p95_ms = float(np.percentile(times_ms, 95))
        fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

        return {
            "latency_ms_mean": mean_ms,
            "latency_ms_p50": p50_ms,
            "latency_ms_p95": p95_ms,
            "fps": fps,
        }


def _select_providers(device: str, available: Iterable[str]) -> Iterable[str]:
    device = device.lower()
    available = list(available)

    if device in ("cpu", "host"):
        return ["CPUExecutionProvider"]

    if device.startswith("cuda") or device == "auto":
        # Note: We don't include TensorrtExecutionProvider here because:
        # 1. We have a dedicated TensorRTPredictor for native TRT inference
        # 2. ORT's TensorRT provider requires TRT libs in LD_LIBRARY_PATH
        preferred = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        return [p for p in preferred if p in available] or available

    if device in ("directml", "dml"):
        preferred = ["DmlExecutionProvider", "CPUExecutionProvider"]
        return [p for p in preferred if p in available] or available

    return available


def _onnx_type_to_numpy(type_str: str | None) -> np.dtype | None:
    if not type_str:
        return None
    if type_str.startswith("tensor(") and type_str.endswith(")"):
        key = type_str[len("tensor(") : -1]
    else:
        return None

    mapping = {
        "float": np.float32,
        "float16": np.float16,
        "double": np.float64,
        "uint8": np.uint8,
        "int8": np.int8,
        "uint16": np.uint16,
        "int16": np.int16,
        "uint32": np.uint32,
        "int32": np.int32,
        "uint64": np.uint64,
        "int64": np.int64,
    }
    return mapping.get(key)


def _as_numpy(image: np.ndarray, expected_dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(image, np.ndarray):
        data = image
    else:
        try:
            import torch

            if isinstance(image, torch.Tensor):
                data = image.detach().cpu().numpy()
            else:
                data = np.asarray(image)
        except ImportError:
            data = np.asarray(image)

    if expected_dtype is not None:
        if data.dtype != expected_dtype:
            data = data.astype(expected_dtype)
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    return data
