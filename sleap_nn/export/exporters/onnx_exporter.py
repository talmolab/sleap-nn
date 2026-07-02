"""ONNX export utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from loguru import logger


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
    numerical_check: bool = False,
    numerical_atol: float = 1e-3,
    numerical_rtol: float = 1e-3,
) -> Path:
    """Export a PyTorch model to ONNX.

    Args:
        model: The PyTorch module to export.
        save_path: Destination path for the ``.onnx`` file.
        input_shape: Shape of the dummy input used to trace the graph.
        input_dtype: Dtype of the dummy input (``uint8`` matches the inference
            wrappers, which normalize in-graph).
        opset_version: ONNX opset for the (default) TorchScript exporter.
        dynamic_axes: Dynamic-axis spec; defaults to batch/height/width dynamic on
            the ``image`` input.
        input_names: ONNX input names; defaults to ``["image"]``.
        output_names: ONNX output names; inferred from a reference forward if
            ``None``.
        do_constant_folding: Whether to constant-fold during export.
        verify: If ``True``, run the structural ``onnx.checker`` on the result.
        numerical_check: If ``True``, additionally run the exported graph through
            onnxruntime on the export dummy input and assert output parity against
            PyTorch (atol/rtol below). ``_verify_onnx`` alone is structural only
            (``onnx.checker``); a numerical check catches graphs that are valid but
            wrong — a known failure mode for transformer backbones (e.g. Swin) whose
            ops trace but disagree numerically. Requires onnxruntime (the ``export``
            extra); degrades to a warning if it is unavailable.
        numerical_atol: Absolute tolerance for the numerical parity check.
        numerical_rtol: Relative tolerance for the numerical parity check.
    """
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

    # A reference forward is needed to infer output names and/or for parity.
    test_out = None
    if output_names is None or numerical_check:
        with torch.no_grad():
            test_out = model(dummy_input)
    if output_names is None:
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

    if numerical_check:
        _verify_onnx_numerical(
            save_path,
            dummy_input,
            test_out,
            input_names[0],
            output_names,
            atol=numerical_atol,
            rtol=numerical_rtol,
        )

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


def _verify_onnx_numerical(
    path: Path,
    dummy_input: torch.Tensor,
    reference_output,
    input_name: str,
    output_names: List[str],
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    """Assert onnxruntime output parity against a PyTorch reference.

    Complements the structural ``_verify_onnx`` check for cases where a graph is
    valid but numerically wrong (a known transformer-backbone failure mode). Lazily
    imports onnxruntime (the ``export`` extra) and degrades to a warning if it is
    unavailable rather than failing the export.
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - onnxruntime is an optional extra
        logger.warning(
            "onnxruntime is not installed; skipping ONNX numerical-parity check. "
            "Install it with `pip install 'sleap-nn[export]'` to enable it."
        )
        return

    session = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {input_name: dummy_input.detach().cpu().numpy()})

    if isinstance(reference_output, dict):
        ref_list = [reference_output[name] for name in output_names]
    elif isinstance(reference_output, (list, tuple)):
        ref_list = list(reference_output)
    else:
        ref_list = [reference_output]

    for name, ref, got in zip(output_names, ref_list, ort_outputs):
        ref_np = ref.detach().cpu().numpy()
        max_abs = float(np.abs(ref_np - got).max())
        if not np.allclose(ref_np, got, atol=atol, rtol=rtol):
            message = (
                f"ONNX numerical-parity check failed for output '{name}': max abs "
                f"diff {max_abs:.3g} exceeds atol={atol}, rtol={rtol}. The exported "
                f"graph is structurally valid but numerically disagrees with "
                f"PyTorch (common for transformer backbones)."
            )
            logger.error(message)
            raise AssertionError(message)
        logger.info(
            f"ONNX numerical-parity OK for '{name}' (max abs diff {max_abs:.3g})."
        )
