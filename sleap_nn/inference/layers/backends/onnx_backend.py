"""``ONNXBackend`` — runs an exported ONNX model under the ``ModelBackend`` protocol.

The existing wrappers in ``sleap_nn/export/wrappers/`` bake several
postprocessing steps into the ONNX graph itself (uint8 normalization,
input-scale resize, peak finding, optional PAF line scoring). When an
``InferenceLayer`` runs against this backend, ``does_baked_postproc`` is
``True`` so the layer skips its Python peak finding and just applies the
coord-transform ladder to whatever the session returned.

This backend is a thin shim around ``onnxruntime.InferenceSession``: it
selects the right execution providers, runs the session, and converts
numpy outputs to torch tensors so downstream layer code is dtype/device
uniform with the ``TorchBackend`` path.

PR 11 (#519) deletes the legacy ``sleap_nn/export/predictors/onnx.py``
runtime wrapper after this backend is wired through the predictor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import attrs
import numpy as np
import torch


def _select_providers(device: str, available: Iterable[str]) -> list[str]:
    """Pick the right ONNX Runtime execution providers for ``device``."""
    device = device.lower()
    available = list(available)
    if device in ("cpu", "host"):
        return ["CPUExecutionProvider"]
    if device.startswith("cuda") or device == "auto":
        # Skip TensorrtExecutionProvider — we ship a dedicated TensorRTBackend
        # for native TRT, and the ORT-TRT EP needs system-level TRT libs.
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return [p for p in preferred if p in available] or available
    if device in ("directml", "dml"):
        preferred = ["DmlExecutionProvider", "CPUExecutionProvider"]
        return [p for p in preferred if p in available] or available
    return available


def _onnx_dtype_to_numpy(type_str: Optional[str]) -> Optional[np.dtype]:
    """Convert an ONNX ``tensor(...)`` type string to a numpy dtype."""
    if not type_str or not (type_str.startswith("tensor(") and type_str.endswith(")")):
        return None
    key = type_str[len("tensor(") : -1]
    return {
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
    }.get(key)


@attrs.define(eq=False, slots=False)
class ONNXBackend:
    """ONNX Runtime backend conforming to :class:`ModelBackend`.

    Args:
        model_path: Path to an exported ``.onnx`` file.
        device: ``"cpu"`` / ``"cuda"`` / ``"auto"`` / ``"directml"``. Used
            to pick onnxruntime execution providers.
        providers: Explicit override for the execution-provider list. If
            ``None``, providers are auto-selected from ``device``.

    Notes:
        ``does_baked_postproc=True`` — the ONNX wrappers in
        ``sleap_nn/export/wrappers/`` bake peak finding, normalization,
        and (top-down) crop extraction into the graph. Layer postprocess
        methods take the ``"peaks"`` / ``"peak_vals"`` keys directly from
        the session output instead of running Python peak finding.
    """

    model_path: str
    device: str = "auto"
    providers: Optional[Iterable[str]] = None

    _session: object = attrs.field(default=None, init=False, repr=False)
    _input_name: str = attrs.field(default="", init=False, repr=False)
    _input_dtype: Optional[np.dtype] = attrs.field(default=None, init=False, repr=False)
    _output_names: list[str] = attrs.field(factory=list, init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        """Load the ONNX session and cache I/O metadata."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNXBackend. Install with "
                "`pip install onnxruntime` (or `onnxruntime-gpu` for CUDA)."
            ) from exc

        if hasattr(ort, "preload_dlls"):
            # Auto-load CUDA/cuDNN libs from pip-installed nvidia-* packages.
            ort.preload_dlls()

        providers = (
            list(self.providers)
            if self.providers is not None
            else _select_providers(self.device, ort.get_available_providers())
        )
        self._session = ort.InferenceSession(self.model_path, providers=providers)

        in_info = self._session.get_inputs()[0]
        self._input_name = in_info.name
        self._input_dtype = _onnx_dtype_to_numpy(in_info.type)
        self._output_names = [out.name for out in self._session.get_outputs()]

    # ──────────────────────────────────────────────────────────────────
    # ModelBackend protocol surface
    # ──────────────────────────────────────────────────────────────────

    @property
    def does_baked_postproc(self) -> bool:
        """ONNX wrappers bake peak finding into the graph."""
        return True

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the ONNX session.

        Args:
            x: Input tensor. Cast to the session's expected dtype before
                handoff (most exported wrappers expect ``uint8``).

        Returns:
            Dict mapping the session's output names to torch tensors.
            Layer postprocess methods then look up ``"peaks"`` /
            ``"peak_vals"`` via the ``does_baked_postproc=True`` branch.
        """
        np_x = x.detach().cpu().numpy()
        if self._input_dtype is not None and np_x.dtype != self._input_dtype:
            np_x = np_x.astype(self._input_dtype)

        outputs = self._session.run(None, {self._input_name: np_x})
        return {
            name: torch.from_numpy(np.asarray(out))
            for name, out in zip(self._output_names, outputs)
        }

    def warmup(self, input_shape: Tuple[int, ...]) -> None:
        """Run a single dummy forward to prime the runtime / GPU caches."""
        dummy = np.zeros(input_shape, dtype=self._input_dtype or np.float32)
        self._session.run(None, {self._input_name: dummy})

    # ──────────────────────────────────────────────────────────────────
    # Convenience constructor
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_export_dir(
        cls, export_dir: Union[str, Path], device: str = "auto"
    ) -> "ONNXBackend":
        """Load an ONNX backend from an export directory containing ``model.onnx``.

        Args:
            export_dir: Directory written by ``sleap_nn export``.
            device: Device hint for execution-provider selection.

        Returns:
            A configured ``ONNXBackend``.
        """
        export_dir = Path(export_dir)
        candidates = sorted(export_dir.glob("*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"No .onnx file found in {export_dir}")
        return cls(model_path=str(candidates[0]), device=device)
