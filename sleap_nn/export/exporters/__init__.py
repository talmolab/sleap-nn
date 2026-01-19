"""Exporters for serialized model formats."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch

from sleap_nn.export.exporters.onnx_exporter import export_to_onnx
from sleap_nn.export.exporters.tensorrt_exporter import export_to_tensorrt


def export_model(
    model: torch.nn.Module,
    save_path: str | Path,
    fmt: str = "onnx",
    input_shape: Iterable[int] = (1, 1, 512, 512),
    opset_version: int = 17,
    output_names: Optional[list] = None,
    verify: bool = True,
    **kwargs,
) -> Path:
    """Export a model to the requested format."""
    fmt = fmt.lower()
    if fmt == "onnx":
        return export_to_onnx(
            model,
            save_path,
            input_shape=input_shape,
            opset_version=opset_version,
            output_names=output_names,
            verify=verify,
        )
    if fmt == "tensorrt":
        return export_to_tensorrt(model, save_path, input_shape=input_shape, **kwargs)
    if fmt == "both":
        export_to_onnx(
            model,
            save_path,
            input_shape=input_shape,
            opset_version=opset_version,
            output_names=output_names,
            verify=verify,
        )
        return export_to_tensorrt(model, save_path, input_shape=input_shape, **kwargs)

    raise ValueError(f"Unknown export format: {fmt}")


__all__ = ["export_model", "export_to_onnx", "export_to_tensorrt"]
