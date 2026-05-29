"""``SingleInstanceLayer`` — single-pose-per-frame inference.

Single-instance models predict one pose per frame from a confmap-only head.
The layer:

1. Accepts ``np.ndarray`` or ``torch.Tensor`` directly (real-time / notebook
   use cases that don't want to spin up a ``sio.Video``).
2. Runs the backend (PyTorch / ONNX / TensorRT) — same code path; the
   ``ModelBackend`` protocol abstracts the runtime.
3. Decodes confmaps to keypoints via :mod:`sleap_nn.inference.ops.peaks`.
4. Reverses the coord ladder via :mod:`sleap_nn.inference.ops.coord` so
   ``Outputs.pred_keypoints`` is in original-image space.
"""

from __future__ import annotations

from typing import Optional

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


class SingleInstanceLayer(InferenceLayer):
    """Single-pose-per-frame inference layer.

    Args:
        backend: Runtime backend (e.g. ``TorchBackend(model=lightning_module)``).
        preprocess_config: Pre-forward transformation knobs.
        postprocess_config: Peak decoding + intermediate-return knobs.
        output_stride: Stride between confmap and input pixels (read from
            the head config at construction).
        max_stride: Backbone-network stride; inputs are padded bottom-right
            to a multiple of this in ``preprocess``. Default ``1`` (no pad).
    """

    _HEAD_OUTPUT_KEY: str = "SingleInstanceConfmapsHead"

    def __init__(
        self,
        backend: ModelBackend,
        output_stride: int,
        max_stride: int = 1,
        preprocess_config: Optional[PreprocessConfig] = None,
        postprocess_config: Optional[PostprocessConfig] = None,
    ) -> None:
        """Compose the layer with default empty configs when omitted."""
        super().__init__(
            backend=backend,
            preprocess_config=preprocess_config or PreprocessConfig(),
            postprocess_config=postprocess_config or PostprocessConfig(),
            output_stride=output_stride,
            max_stride=max_stride,
        )

    # ──────────────────────────────────────────────────────────────────
    # Postprocess
    # ──────────────────────────────────────────────────────────────────

    def postprocess(self, raw_out: dict, info: PreprocInfo) -> Outputs:
        """Decode confmaps → keypoints, reverse coord ladder, build ``Outputs``.

        On a baked-postproc backend (ONNX/TRT) ``raw_out`` already
        contains ``peaks`` + ``peak_vals``; we skip ``find_global_peaks``
        and only apply the coord ladder.
        """
        if self.backend.does_baked_postproc:
            peaks = raw_out["peaks"]
            vals = raw_out["peak_vals"]
            confmaps = raw_out.get("confmaps")
        else:
            confmaps = self._extract_confmaps(raw_out)
            peaks, vals = find_global_peaks(
                confmaps.detach(),
                threshold=self.postprocess_config.peak_threshold,
                refinement=self.postprocess_config.effective_refinement,
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
