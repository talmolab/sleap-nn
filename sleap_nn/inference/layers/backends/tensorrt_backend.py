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
from typing import Callable, Dict, Optional, Tuple, Union

import attrs
import torch


def run_in_batch_chunks(
    run: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
    x: torch.Tensor,
    max_batch: Optional[int],
) -> Dict[str, torch.Tensor]:
    """``run(x)``, or ``run`` on chunks of at most ``max_batch`` along axis 0.

    Outputs of the chunks are concatenated along axis 0 (every exported
    wrapper's outputs are batch-first). ``max_batch`` ``None`` or a batch within
    it runs ``x`` whole.
    """
    if max_batch is None or x.shape[0] <= max_batch:
        return run(x)
    starts = range(0, x.shape[0], max_batch)
    parts = [run(x[i : i + max_batch]) for i in starts]
    for i, part in zip(starts, parts):
        n = min(max_batch, x.shape[0] - i)
        for k, v in part.items():
            if v.ndim == 0 or v.shape[0] != n:
                raise RuntimeError(
                    f"Output '{k}' (shape {tuple(v.shape)}) is not batch-first for a "
                    f"chunk of {n}, so chunks of this engine cannot be joined. Run "
                    f"batches of at most {max_batch}, or re-export with "
                    f"--max-batch-size >= {x.shape[0]}."
                )
    return {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}


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
    # Largest batch the engine's optimization profile admits (``None``: unknown).
    _max_batch: Optional[int] = attrs.field(default=None, init=False, repr=False)

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
            raise RuntimeError(_load_failure_message(engine_path, trt.__version__))

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(
                f"Could not create a TensorRT execution context for {engine_path}. "
                "The likely cause is not enough free GPU memory: an engine reserves "
                "memory for its optimization profile's maximum shape (sleap-nn export "
                "--max-batch-size). Free GPU memory, or re-export with a smaller "
                "--max-batch-size."
            )
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)
        try:
            max_shape = self._engine.get_tensor_profile_shape(self._input_names[0], 0)[
                2
            ]
            self._max_batch = int(max_shape[0]) if int(max_shape[0]) > 0 else None
        except Exception:  # noqa: BLE001 -- static engines / older TensorRT: unknown
            self._max_batch = None

    # ──────────────────────────────────────────────────────────────────
    # ModelBackend protocol surface
    # ──────────────────────────────────────────────────────────────────

    @property
    def does_baked_postproc(self) -> bool:
        """TRT engines exported from our wrappers bake peak finding."""
        return True

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Execute the TRT engine on ``x`` (must be on CUDA already).

        A batch larger than the engine's optimization profile admits
        (``sleap-nn export --max-batch-size``) runs in chunks of that size and
        the outputs are concatenated along the batch axis (larger batches used to
        fail in ``set_input_shape``).

        Args:
            x: ``(B, C, H, W)`` input tensor on CUDA. Auto-cast to the
                engine's expected dtype (``uint8`` for most exported
                wrappers).

        Returns:
            Dict mapping engine output names to torch tensors on CUDA.
        """
        return run_in_batch_chunks(self._run, x, self._max_batch)

    def _run(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """One engine execution on a batch the profile admits."""
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

        if not self._context.set_input_shape(input_name, tuple(x.shape)):
            try:
                lo, _, hi = self._engine.get_tensor_profile_shape(input_name, 0)
                bounds = f" (profile min {tuple(lo)}, max {tuple(hi)})"
            except Exception:  # noqa: BLE001 -- static engines / older TensorRT
                bounds = ""
            raise ValueError(
                f"Input shape {tuple(x.shape)} is outside the TensorRT engine's "
                f"optimization profile{bounds} for '{input_name}' "
                f"({self.engine_path}). Re-export at the size the frames or crops "
                "have here, with a --max-batch-size that covers the batch."
            )

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


def _load_failure_message(engine_path: Path, trt_version: str) -> str:
    """Explain why TensorRT could not deserialize an engine file.

    An engine runs only on the TensorRT version (and GPU) that built it. The engine's
    ``model.trt.metadata.json`` names that version when sleap-nn wrote it.
    """
    import json

    built_with = None
    meta = engine_path.with_name(engine_path.name + ".metadata.json")
    if meta.exists():
        try:
            built_with = json.loads(meta.read_text()).get("tensorrt_version")
        except (OSError, ValueError):
            built_with = None
    rebuild = (
        "Export it again in this environment (`sleap-nn export <model> ... --format "
        "tensorrt`), or run the directory's model.onnx with `--runtime onnx`."
    )
    if built_with and str(built_with) != str(trt_version):
        return (
            f"Failed to load TensorRT engine {engine_path}: it was built with TensorRT "
            f"{built_with}, and this is TensorRT {trt_version}; an engine runs only on "
            f"the TensorRT version that built it. {rebuild}"
        )
    return (
        f"Failed to load TensorRT engine {engine_path} with TensorRT {trt_version}. "
        "An engine runs only on the TensorRT version and GPU that built it"
        + (
            f" (built with TensorRT {built_with})."
            if built_with
            else " (its metadata does not name the building version)."
        )
        + f" {rebuild}"
    )
