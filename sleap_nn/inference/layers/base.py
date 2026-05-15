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
        """Prime the backend by running ``predict()`` on a synthesized frame.

        The synthesized frame goes through the layer's full ``preprocess``
        chain (sizematcher → input_scale → ensure_rgb/grayscale → pad →
        n_samples wrap) so the model receives an input with the same
        rank / channel-count / device contract as real inference, and
        cuDNN's algorithm cache is primed for the right shape.

        Pre-PR-27 this called ``backend.warmup(shape=(1, 1, 64, 64))``
        directly, bypassing ``preprocess`` entirely. On torch 2.9.1+cu128
        the bottom-up / centroid Lightning ``forward`` ops do
        ``torch.squeeze(img, dim=1)`` unconditionally, collapsing the
        4D dummy to 3D ``(1, 64, 64)``. cuDNN cached an algorithm for
        that degenerate shape; the next real ``(4, 1, 3, 384, 384)`` batch
        re-used the cached algorithm and crashed mid-decoder with
        ``input[B=1, C=36, 32, 16] expected 72 channels``. Audit:
        ``scratch/2026-04-30-inference-refactor-implementation/cuda_bench/
        channel_bug_*.log``.

        Args:
            sample_shape: Legacy escape hatch. When provided, dispatches
                straight to ``backend.warmup`` exactly as before. Prefer
                the default (synthesized real frame) on cuda / mps.
        """
        if sample_shape is not None:
            self.backend.warmup(sample_shape)
            return
        if self.backend.device == "cpu":
            return  # warmup is a no-op on CPU; first forward is already cold-start
        # Synthesize a tiny 3-channel uint8 frame in raw-video shape
        # (H, W, C). ``preprocess`` will route it through sizematcher (when
        # ``max_height``/``max_width`` are set), channel coercion, input
        # scale, stride pad, and the n_samples wrap — producing the exact
        # post-preprocess shape real inference uses.
        cfg = self.preprocess_config
        h = min(cfg.max_height or 96, 256)
        w = min(cfg.max_width or 96, 256)
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        try:
            self.predict(dummy)
        except Exception:  # noqa: BLE001 — warmup is best-effort
            pass
        if self.backend.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif self.backend.device == "mps":
            torch.mps.synchronize()

    @property
    def warmup_input_shape(self) -> Tuple[int, ...]:
        """Legacy warmup shape — only used when ``sample_shape`` is passed.

        Retained for callers passing ``sample_shape`` to ``warmup``.
        The default ``warmup()`` path ignores this and synthesizes a real
        raw frame instead.
        """
        return (1, 1, 64, 64)

    # ──────────────────────────────────────────────────────────────────
    # Helpers shared by every subclass
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_4d_tensor(image: ImageInput) -> torch.Tensor:
        """Coerce an image input to ``(B, C, H, W)``, preserving dtype.

        Accepts:

        - ``(H, W)`` grayscale numpy/torch
        - ``(H, W, C)`` channel-last numpy/torch
        - ``(B, H, W, C)`` channel-last
        - ``(C, H, W)`` channel-first
        - ``(B, C, H, W)`` channel-first

        Returns ``(B, C, H, W)`` with the same dtype as the input. uint8
        inputs stay uint8 so subsequent ``tvf.resize`` calls produce
        clean integer outputs (legacy parity — the eager float
        conversion produced 255.00006... values that diverged from
        legacy's clean uint8 path).
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

        return t

    @classmethod
    def _to_4d_float_tensor(cls, image: ImageInput) -> torch.Tensor:
        """Coerce to ``(B, C, H, W)`` ``torch.float32``.

        Thin wrapper over :meth:`_to_4d_tensor` for backward compat with
        callers that explicitly want float32 (older test fixtures, ONNX
        backends that don't accept uint8). New layer ``preprocess()``
        methods use ``_to_4d_tensor`` so the legacy uint8 → ``tvf.resize``
        path is preserved bit-for-bit.
        """
        return cls._to_4d_tensor(image).float()

    # ──────────────────────────────────────────────────────────────────
    # Shared raw-frame preprocessing chain
    # ──────────────────────────────────────────────────────────────────

    def _apply_full_preprocess(
        self,
        x: torch.Tensor,
        *,
        max_stride: int = 1,
        unsqueeze_n_samples: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Run the legacy-parity preprocessing chain on a (B, C, H, W) tensor.

        Mirrors the legacy ``_make_pipeline_inputs`` per-frame chain
        (``sleap_nn/inference/predictors.py:530-563``) plus the
        ``apply_pad_to_stride`` hop from ``_run_inference_on_batch``.
        Each step short-circuits when its config field is the identity
        (``None``/``False``/``1.0``), so a raw-frame layer running on a
        properly-sized batch sees zero extra ops.

        Stages applied in order:

        1. ``ensure_rgb`` / ``ensure_grayscale`` — channel coercion
           (matches legacy lines 545-553).
        2. Per-sample ``apply_sizematcher`` to
           ``(preprocess_config.max_height, preprocess_config.max_width)``,
           returning a per-sample ``eff_scale`` for the coord-undo ladder
           (matches legacy line 538-555).
        3. ``resize_image`` by ``preprocess_config.scale`` — global input
           scale (matches legacy line 600-604).
        4. ``apply_pad_to_stride`` to ``max_stride`` (matches legacy line
           607). Use the model's max_stride; ``1`` is a no-op.
        5. ``unsqueeze(dim=1)`` to add the ``n_samples`` axis so the
           Lightning forward's unconditional ``squeeze(dim=1)`` resolves
           to the expected rank. Skip when the layer's forward accepts
           4D directly (``single_instance`` has an ``ndim==5`` guard, so
           we still wrap to match legacy bit-for-bit).

        Args:
            x: ``(B, C, H, W)`` float32 tensor from :meth:`_to_4d_float_tensor`.
            max_stride: Model's required input stride; the input is padded
                bottom-right to a multiple of this. ``1`` is the identity.
            unsqueeze_n_samples: When ``True`` (the legacy default for
                multi-instance layers) wraps with a ``(B, 1, C, H, W)``
                ``n_samples`` axis. Top-down crops feed
                :class:`CenteredInstanceLayer` post-crop and don't need
                sizematcher — those callers pass ``False``.

        Returns:
            ``(processed_tensor, eff_scale, original_HW)``:

            * ``processed_tensor``: ``(B, 1, C, H', W')`` if
              ``unsqueeze_n_samples`` else ``(B, C, H', W')``.
            * ``eff_scale``: ``(B,)`` per-sample sizematcher scale factor.
              All ones when no sizematcher is configured.
            * ``original_HW``: ``(H, W)`` of the input before any resize.
        """
        # Local imports avoid a circular base.py → data.* → ... → base.py path.
        from sleap_nn.data.normalization import convert_to_grayscale, convert_to_rgb
        from sleap_nn.data.resizing import (
            apply_pad_to_stride,
            apply_sizematcher,
            resize_image,
        )

        cfg = self.preprocess_config
        B, _C, H, W = x.shape
        orig_hw = (H, W)

        # 1. Channel coercion.
        if cfg.ensure_grayscale and x.shape[-3] != 1:
            x = convert_to_grayscale(x)
        elif cfg.ensure_rgb and x.shape[-3] != 3:
            x = convert_to_rgb(x)

        # 2. Per-sample sizematcher → eff_scale.
        if cfg.max_height is not None or cfg.max_width is not None:
            resized_frames: list = []
            eff_scales: list = []
            for b in range(B):
                # apply_sizematcher accepts (C, H, W); preserves device.
                r, scale = apply_sizematcher(x[b], cfg.max_height, cfg.max_width)
                resized_frames.append(r)
                eff_scales.append(float(scale))
            x = torch.stack(resized_frames, dim=0)
            eff_scale = torch.tensor(eff_scales, dtype=torch.float32, device=x.device)
        else:
            eff_scale = torch.ones(B, dtype=torch.float32, device=x.device)

        # 3. Input scale.
        if cfg.scale != 1.0:
            x = resize_image(x, cfg.scale)

        # 4. Pad to stride.
        if max_stride != 1:
            x = apply_pad_to_stride(x, max_stride)

        # 5. n_samples wrap.
        if unsqueeze_n_samples:
            x = x.unsqueeze(1)

        return x, eff_scale, orig_hw
