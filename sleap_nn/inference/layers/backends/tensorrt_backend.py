"""``TensorRTBackend`` — runs a serialized TensorRT engine.

Like :class:`ONNXBackend` but the runtime is native TensorRT instead of
ONNX Runtime. Same protocol surface: ``does_baked_postproc=True`` because
the wrappers in ``sleap_nn/export/wrappers/`` produce engines whose
output names are ``"peaks"``, ``"peak_vals"``, etc.

CUDA-only. Constructing on a non-CUDA host raises a clear error;
``import tensorrt`` is deferred to ``__attrs_post_init__`` so importing
this module is cheap (and the import error doesn't break test
collection on CPU-only hosts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import attrs
import torch


@attrs.define(eq=False, slots=False)
class TensorRTBackend:
    """Native TensorRT engine backend conforming to :class:`ModelBackend`.

    Args:
        engine_path: Path to a serialized TRT engine file (``.trt``).
        device: Must be ``"cuda"`` or ``"auto"``. Other values raise.

    Notes:
        Constructing this backend imports ``tensorrt`` lazily. On a host
        without CUDA / ``tensorrt`` installed, the constructor raises with
        a clear pointer at the right install extra (``[tensorrt]``).
    """

    engine_path: str
    device: str = "cuda"

    _engine: object = attrs.field(default=None, init=False, repr=False)
    _context: object = attrs.field(default=None, init=False, repr=False)
    _input_names: list[str] = attrs.field(factory=list, init=False, repr=False)
    _output_names: list[str] = attrs.field(factory=list, init=False, repr=False)
    _trt: object = attrs.field(default=None, init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        """Load the TRT engine + create an execution context."""
        if self.device not in ("cuda", "auto"):
            raise ValueError(
                f"TensorRTBackend only supports CUDA; got device={self.device!r}"
            )
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                "tensorrt is required for TensorRTBackend. Install with "
                "`pip install sleap-nn[tensorrt]` (Linux/Windows only)."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TensorRTBackend requires a CUDA device; none is available."
            )

        self._trt = trt
        engine_path = Path(self.engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self._engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self._context = self._engine.create_execution_context()
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)

    # ──────────────────────────────────────────────────────────────────
    # ModelBackend protocol surface
    # ──────────────────────────────────────────────────────────────────

    @property
    def does_baked_postproc(self) -> bool:
        """TRT engines exported from our wrappers bake peak finding."""
        return True

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Execute the TRT engine on ``x`` (must be on CUDA already).

        Args:
            x: ``(B, C, H, W)`` input tensor on CUDA. Auto-cast to the
                engine's expected dtype (``uint8`` for most exported
                wrappers).

        Returns:
            Dict mapping engine output names to torch tensors on CUDA.
        """
        trt = self._trt
        input_name = self._input_names[0]
        expected = self._engine.get_tensor_dtype(input_name)

        x = x.to("cuda", non_blocking=True)
        if expected == trt.DataType.UINT8:
            if x.dtype != torch.uint8:
                x = x.to(torch.uint8)
        elif x.dtype == torch.uint8:
            x = x.to(torch.float32)
        x = x.contiguous()

        self._context.set_input_shape(input_name, tuple(x.shape))

        bindings: Dict[str, int] = {input_name: x.data_ptr()}
        outputs: Dict[str, torch.Tensor] = {}
        for name in self._output_names:
            shape = tuple(self._context.get_tensor_shape(name))
            dtype = self._trt_dtype_to_torch(self._engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(shape, dtype=dtype, device="cuda")
            bindings[name] = outputs[name].data_ptr()
        for name, ptr in bindings.items():
            self._context.set_tensor_address(name, ptr)

        stream = torch.cuda.current_stream().cuda_stream
        if not self._context.execute_async_v3(stream):
            raise RuntimeError("TensorRT inference failed")
        torch.cuda.current_stream().synchronize()
        return outputs

    def warmup(self, input_shape: Tuple[int, ...]) -> None:
        """Run a single dummy forward to prime the engine + GPU caches."""
        trt = self._trt
        input_name = self._input_names[0]
        expected = self._engine.get_tensor_dtype(input_name)
        dtype = torch.uint8 if expected == trt.DataType.UINT8 else torch.float32
        dummy = torch.zeros(input_shape, dtype=dtype, device="cuda")
        self(dummy)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _trt_dtype_to_torch(self, trt_dtype) -> torch.dtype:
        """Map a TRT dtype to its torch counterpart (defaults to float32)."""
        trt = self._trt
        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }
        return mapping.get(trt_dtype, torch.float32)

    @classmethod
    def from_export_dir(cls, export_dir: Union[str, Path]) -> "TensorRTBackend":
        """Load a TRT backend from an export directory containing ``*.trt``."""
        export_dir = Path(export_dir)
        candidates = sorted(export_dir.glob("*.trt"))
        if not candidates:
            raise FileNotFoundError(f"No .trt file found in {export_dir}")
        return cls(engine_path=str(candidates[0]))
