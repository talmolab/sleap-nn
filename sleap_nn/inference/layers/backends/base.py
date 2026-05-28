"""``ModelBackend`` protocol — the contract every runtime backend implements.

Decouples *model-type logic* (preprocess, peak finding, PAF grouping —
owned by ``InferenceLayer`` subclasses) from *runtime logic* (PyTorch,
ONNX, TensorRT — owned by backend classes). One backend serves every
model type; one layer hosts every backend.

The protocol is intentionally tiny: a property pair (``device``,
``does_baked_postproc``), a forward pass (``__call__``), and a warmup
hook. Anything more belongs in the layer or the backend's own __init__.
"""

from typing import Dict, Protocol, Tuple, runtime_checkable

import torch


@runtime_checkable
class ModelBackend(Protocol):
    """Runtime-agnostic forward-pass contract.

    Any object that satisfies this protocol can power any
    ``InferenceLayer`` subclass. Verify with ``isinstance(obj, ModelBackend)``
    at construction time.
    """

    @property
    def device(self) -> str:
        """Device the backend runs on. ``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"mps"``."""
        ...

    @property
    def does_baked_postproc(self) -> bool:
        """``True`` if this backend already performs peak finding internally.

        ONNX and TensorRT export wrappers bake normalization + peak finding
        + (optionally) PAF scoring into the graph and return precomputed
        peaks. When this property is ``True``, the wrapping ``InferenceLayer``
        must skip its own Python-side peak finding and only apply coordinate
        transforms to whatever the backend returns.

        For pure PyTorch (``TorchBackend``) this is always ``False``.
        """
        ...

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the model forward pass.

        Args:
            x: Preprocessed input. Shape ``(B, C, H, W)``. The backend
                accepts whatever the upstream layer hands it — e.g.
                ``TorchBackend`` accepts uint8 because the Lightning
                module normalizes internally; an ONNX backend may also
                accept uint8 because normalization is baked.

        Returns:
            Dict of output tensors. Keys depend on the wrapped model:

            - Torch (``does_baked_postproc=False``)::

                {"SingleInstanceConfmapsHead": (B, N, H, W), ...}

            - ONNX/TRT (``does_baked_postproc=True``)::

                {"peaks": (B, I, N, 2), "peak_vals": (B, I, N), ...}
        """
        ...

    def warmup(self, input_shape: Tuple[int, ...]) -> None:
        """Run dummy forward passes to prime the backend.

        Particularly important on MPS (≈73× cold-start ratio per the
        benchmark suite) and CUDA-with-compile (where the first call
        triggers JIT compilation).

        Args:
            input_shape: Shape of the dummy tensor to allocate.
        """
        ...
