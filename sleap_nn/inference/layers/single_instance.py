"""``SingleInstanceLayer`` — proof-of-pattern for the InferenceLayer abstraction.

Single-instance models predict one pose per frame from a confmap-only head.
The layer:

1. Accepts ``np.ndarray`` or ``torch.Tensor`` directly (real-time / notebook
   use cases that don't want to spin up a ``sio.Video``).
2. Runs the backend (PyTorch / ONNX / TensorRT) — same code path; the
   ``ModelBackend`` protocol abstracts the runtime.
3. Decodes confmaps to keypoints via :mod:`sleap_nn.inference.ops.peaks`.
4. Reverses the coord ladder via :mod:`sleap_nn.inference.ops.coord` so
   ``Outputs.pred_keypoints`` is in original-image space.

Parity test: this layer's output on a fixed input matches the corresponding
slice of the PR 0 ``single_instance.pkl`` golden bit-for-bit.
"""

from __future__ import annotations

from typing import Optional, Tuple

import attrs
import torch

from sleap_nn.inference.layers.backends.base import ModelBackend
from sleap_nn.inference.layers.base import ImageInput, InferenceLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.ops.coord import (
    undo_eff_scale,
    undo_input_scale,
    undo_stride,
)
from sleap_nn.inference.ops.peaks import find_global_peaks
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.preprocess_info import PreprocInfo

# Lightning's SingleInstance forward returns a Tensor; TorchBackend wraps it
# as ``{"output": ...}``. ONNX/TRT wrappers (PR 7) emit baked peak fields.
_TORCH_OUTPUT_KEY = "output"
_HEAD_OUTPUT_KEY = "SingleInstanceConfmapsHead"


class SingleInstanceLayer(InferenceLayer):
    """Single-pose-per-frame inference layer.

    Args:
        backend: Runtime backend (e.g. ``TorchBackend(model=lightning_module)``).
        preprocess_config: Pre-forward transformation knobs.
        postprocess_config: Peak decoding + intermediate-return knobs.
        output_stride: Stride between confmap and input pixels (read from
            the head config at construction).
    """

    def __init__(
        self,
        backend: ModelBackend,
        output_stride: int,
        preprocess_config: Optional[PreprocessConfig] = None,
        postprocess_config: Optional[PostprocessConfig] = None,
    ) -> None:
        """Compose the layer with default empty configs when omitted."""
        super().__init__(
            backend=backend,
            preprocess_config=preprocess_config or PreprocessConfig(),
            postprocess_config=postprocess_config or PostprocessConfig(),
            output_stride=output_stride,
        )

    # ──────────────────────────────────────────────────────────────────
    # Preprocess
    # ──────────────────────────────────────────────────────────────────

    def preprocess(self, image: ImageInput) -> Tuple[torch.Tensor, PreprocInfo]:
        """Coerce to ``(B, C, H, W)`` and record reverse-ladder info."""
        x = self._to_4d_float_tensor(image)
        B, _C, H, W = x.shape

        info = PreprocInfo(
            original_size=(H, W),
            processed_size=(H, W),
            eff_scale=torch.ones(B),
            input_scale=self.preprocess_config.scale,
            output_stride=self.output_stride,
            pad_amount=(0, 0),
            crop_offsets=None,
        )
        return x, info

    # ──────────────────────────────────────────────────────────────────
    # Postprocess
    # ──────────────────────────────────────────────────────────────────

    def postprocess(self, raw_out: dict, info: PreprocInfo) -> Outputs:
        """Decode confmaps → keypoints, reverse coord ladder, build ``Outputs``.

        On a baked-postproc backend (ONNX/TRT in PR 7) ``raw_out`` already
        contains ``peaks`` + ``peak_vals``; we skip ``find_global_peaks``
        and only apply the coord ladder.
        """
        if self.backend.does_baked_postproc:
            peaks = raw_out["peaks"]
            vals = raw_out["peak_vals"]
            confmaps = raw_out.get("confmaps")
        else:
            confmaps = self._extract_confmaps(raw_out)
            refinement = (
                self.postprocess_config.refinement
                if self.postprocess_config.refinement != "none"
                else None
            )
            peaks, vals = find_global_peaks(
                confmaps.detach(),
                threshold=self.postprocess_config.peak_threshold,
                refinement=refinement,
                integral_patch_size=self.postprocess_config.integral_patch_size,
            )

        # Coord ladder: confmap pixels → input pixels → original-image pixels.
        peaks = undo_stride(peaks, info.output_stride)
        peaks = undo_input_scale(peaks, info.input_scale)
        peaks = undo_eff_scale(peaks, info.eff_scale)

        # ``find_global_peaks`` returns (B, N, 2) — single-instance has I=1.
        # Reshape to the canonical (B, I=1, N, 2) / (B, I=1, N) Outputs shape.
        peaks_BIN2 = peaks.unsqueeze(1)
        vals_BIN = vals.unsqueeze(1)

        outputs = Outputs(
            pred_keypoints=peaks_BIN2,
            pred_peak_values=vals_BIN,
            preprocess_info=info,
        )
        if self.postprocess_config.return_confmaps and confmaps is not None:
            outputs = attrs.evolve(outputs, pred_confmaps=confmaps.detach())
        return outputs

    @staticmethod
    def _extract_confmaps(raw_out: dict) -> torch.Tensor:
        """Pull the confmap tensor out of the backend's dict.

        ``TorchBackend`` wraps a tensor-returning Lightning forward under
        ``"output"``; if the model returned a dict directly, we look for
        the canonical head name.
        """
        if _TORCH_OUTPUT_KEY in raw_out:
            return raw_out[_TORCH_OUTPUT_KEY]
        if _HEAD_OUTPUT_KEY in raw_out:
            return raw_out[_HEAD_OUTPUT_KEY]
        # Fall back to the single tensor in the dict, if there's exactly one.
        tensors = [v for v in raw_out.values() if isinstance(v, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0]
        raise KeyError(
            f"SingleInstanceLayer.postprocess could not find confmaps in raw_out "
            f"keys={list(raw_out.keys())}; expected '{_TORCH_OUTPUT_KEY}' or "
            f"'{_HEAD_OUTPUT_KEY}'."
        )
