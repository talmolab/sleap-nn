"""``InferenceLayer`` — abstract base for every model-type layer.

Each ``InferenceLayer`` subclass:

1. Owns a ``ModelBackend`` (the runtime — PyTorch / ONNX / TensorRT)
2. Knows the model-type-specific preprocess + postprocess steps
3. Exposes a uniform ``predict(image) -> Outputs`` API

Direct numpy input is the headline new capability vs. today's pipeline:
``layer.predict(np.ndarray)`` works without going through ``sio.Video``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import torch

from sleap_nn.inference.layers.backends.base import ModelBackend
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.preprocess_info import PreprocInfo

# Anything ``predict()`` will coerce into a ``(B, C, H, W)`` float tensor.
ImageInput = Union[np.ndarray, torch.Tensor]


class InferenceLayer(ABC):
    """Abstract base for model-type-specific inference layers.

    Subclasses implement ``preprocess`` (image → tensor + ``PreprocInfo``),
    ``postprocess`` (raw backend output + ``PreprocInfo`` → ``Outputs``),
    and may override ``predict`` for composed layers (top-down). The
    default ``predict`` is preprocess → backend → postprocess.

    Attributes:
        backend: The runtime backend (``TorchBackend`` etc.).
        preprocess_config: Knobs governing input transformation.
        postprocess_config: Knobs governing peak decoding and what
            intermediate tensors to keep.
        output_stride: Confmap → input-pixel stride. Read from the model's
            head config at construction.
    """

    def __init__(
        self,
        backend: ModelBackend,
        preprocess_config: PreprocessConfig,
        postprocess_config: PostprocessConfig,
        output_stride: int,
    ) -> None:
        """Validate the backend protocol and stash configs."""
        if not isinstance(backend, ModelBackend):
            raise TypeError(
                f"backend must satisfy ModelBackend, got {type(backend).__name__}"
            )
        self.backend = backend
        self.preprocess_config = preprocess_config
        self.postprocess_config = postprocess_config
        self.output_stride = output_stride

    # ──────────────────────────────────────────────────────────────────
    # Subclass contract
    # ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def preprocess(self, image: ImageInput) -> Tuple[torch.Tensor, PreprocInfo]:
        """Coerce raw input to ``(B, C, H, W)`` and capture coord-undo info."""

    @abstractmethod
    def postprocess(self, raw_out: dict, info: PreprocInfo) -> Outputs:
        """Turn the backend's raw dict into a structured ``Outputs``."""

    # ──────────────────────────────────────────────────────────────────
    # Default forward — subclasses override for composed layers
    # ──────────────────────────────────────────────────────────────────

    def predict(self, image: ImageInput) -> Outputs:
        """Run the full preprocess → backend → postprocess pipeline."""
        x, info = self.preprocess(image)
        raw = self.backend(x)
        return self.postprocess(raw, info)

    def __call__(self, image: ImageInput) -> Outputs:
        """Alias for :meth:`predict`."""
        return self.predict(image)

    # ──────────────────────────────────────────────────────────────────
    # Warmup helper — subclasses define ``warmup_input_shape``
    # ──────────────────────────────────────────────────────────────────

    def warmup(self, sample_shape: Tuple[int, ...] | None = None) -> None:
        """Prime the backend with a dummy forward.

        Args:
            sample_shape: Input shape for the dummy. If ``None``, falls back
                to the subclass ``warmup_input_shape`` property.
        """
        shape = sample_shape if sample_shape is not None else self.warmup_input_shape
        self.backend.warmup(shape)

    @property
    def warmup_input_shape(self) -> Tuple[int, ...]:
        """Default warmup shape — subclasses can override."""
        return (1, 1, 64, 64)

    # ──────────────────────────────────────────────────────────────────
    # Helpers shared by every subclass
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_4d_float_tensor(image: ImageInput) -> torch.Tensor:
        """Coerce an image input to ``(B, C, H, W)`` float32.

        Accepts:

        - ``(H, W)`` grayscale numpy/torch
        - ``(H, W, C)`` channel-last numpy/torch
        - ``(B, H, W, C)`` channel-last
        - ``(C, H, W)`` channel-first
        - ``(B, C, H, W)`` channel-first

        Always returns ``(B, C, H, W)`` ``torch.float32``.
        """
        if isinstance(image, np.ndarray):
            t = torch.from_numpy(image)
        elif isinstance(image, torch.Tensor):
            t = image
        else:
            raise TypeError(
                f"image must be np.ndarray or torch.Tensor, got {type(image).__name__}"
            )

        if t.ndim == 2:  # (H, W)
            t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif t.ndim == 3:
            # Heuristic: smaller-trailing-dim → channel-last; otherwise
            # already channel-first single sample.
            if t.shape[-1] <= 4 and t.shape[0] > 4:  # (H, W, C)
                t = t.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            else:  # (C, H, W)
                t = t.unsqueeze(0)
        elif t.ndim == 4:
            # Same heuristic for batched: trailing channel-last → permute.
            if t.shape[-1] <= 4 and t.shape[1] > 4:
                t = t.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"unexpected image rank {t.ndim}: shape {tuple(t.shape)}")

        return t.float()
