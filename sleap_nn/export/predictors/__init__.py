"""Predictors for exported models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from sleap_nn.export.predictors.base import ExportPredictor
from sleap_nn.export.predictors.onnx import ONNXPredictor
from sleap_nn.export.predictors.tensorrt import TensorRTPredictor


def detect_runtime(model_path: str | Path) -> str:
    """Auto-detect runtime from file extension or folder contents."""
    model_path = Path(model_path)
    if model_path.is_dir():
        onnx_path = model_path / "exported" / "model.onnx"
        trt_path = model_path / "exported" / "model.trt"
        if trt_path.exists():
            return "tensorrt"
        if onnx_path.exists():
            return "onnx"
        raise ValueError(f"No exported model found in {model_path}")

    ext = model_path.suffix.lower()
    if ext == ".onnx":
        return "onnx"
    if ext in (".trt", ".engine"):
        return "tensorrt"

    raise ValueError(f"Unknown model format: {ext}")


def load_exported_model(
    model_path: str | Path,
    runtime: str = "auto",
    device: str = "auto",
    **kwargs,
) -> ExportPredictor:
    """Load an exported model and return a predictor instance."""
    if runtime == "auto":
        runtime = detect_runtime(model_path)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if runtime == "onnx":
        return ONNXPredictor(str(model_path), device=device, **kwargs)
    if runtime == "tensorrt":
        return TensorRTPredictor(str(model_path), device=device, **kwargs)

    raise ValueError(f"Unknown runtime: {runtime}")


__all__ = [
    "ExportPredictor",
    "ONNXPredictor",
    "TensorRTPredictor",
    "detect_runtime",
    "load_exported_model",
]
