"""ONNX export utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch


def export_to_onnx(
    model: torch.nn.Module,
    save_path: str | Path,
    input_shape: Iterable[int] = (1, 1, 512, 512),
    input_dtype: torch.dtype = torch.uint8,
    opset_version: int = 17,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    do_constant_folding: bool = True,
    verify: bool = True,
) -> Path:
    """Export a PyTorch model to ONNX."""
    save_path = Path(save_path)
    model.eval()

    if input_names is None:
        input_names = ["image"]
    if dynamic_axes is None:
        dynamic_axes = {"image": {0: "batch", 2: "height", 3: "width"}}

    device = None
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    if input_dtype.is_floating_point:
        dummy_input = torch.randn(*input_shape, device=device, dtype=input_dtype)
    else:
        dummy_input = torch.randint(
            0, 256, input_shape, device=device, dtype=input_dtype
        )

    if output_names is None:
        with torch.no_grad():
            test_out = model(dummy_input)
        output_names = _infer_output_names(test_out)

    common = dict(
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    try:
        # Default to the legacy TorchScript exporter: it is fast and exports every
        # current wrapper (including the multi-stage top-down / multi-class ones that
        # the torch.export-based exporter cannot trace).
        torch.onnx.export(
            model,
            dummy_input,
            save_path.as_posix(),
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            dynamo=False,
            **common,
        )
    except torch.onnx.errors.UnsupportedOperatorError:
        # A few ops have no symbolic in the legacy exporter — notably the antialiased
        # resize (``aten::_upsample_bilinear2d_aa``) that single-instance / other
        # downscaling wrappers use to match the PyTorch inference resize. The
        # torch.export-based exporter supports them, so fall back to it for those
        # models (it needs opset >= 18 for the antialias Resize attribute).
        torch.onnx.export(
            model,
            dummy_input,
            save_path.as_posix(),
            opset_version=max(opset_version, 18),
            dynamo=True,
            **common,
        )

    if verify:
        _verify_onnx(save_path)

    return save_path


def _infer_output_names(output) -> List[str]:
    if isinstance(output, dict):
        return list(output.keys())
    if isinstance(output, (list, tuple)):
        return [f"output_{idx}" for idx in range(len(output))]
    return ["output_0"]


def _verify_onnx(path: Path) -> None:
    import onnx

    model = onnx.load(path.as_posix())
    onnx.checker.check_model(model)
