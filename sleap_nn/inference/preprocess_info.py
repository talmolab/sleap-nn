"""``PreprocInfo`` — preprocessing metadata used to reverse coordinate transforms.

Produced by ``InferenceLayer.preprocess()`` and consumed by ``.postprocess()``
to undo the size-match scale, input-scale resize, stride downsample, and
(top-down only) crop offset that took a frame from original-image space into
confmap pixel space.

This struct is the *single* place these numbers are recorded — replacing the
current ad-hoc dict that ``predictors.py`` shuttles between methods. Frozen
because it's a value type: never mutate after capture.
"""

from typing import Optional, Tuple

import attrs
import torch


@attrs.frozen(eq=False, repr=False)
class PreprocInfo:
    """Metadata captured during preprocessing for coordinate-transform reversal.

    Fields default to identity values so an "untouched" image (no resize,
    no scale, stride 1) produces a no-op coordinate ladder. Each field is
    consumed by exactly one op in :mod:`sleap_nn.inference.ops.coord`.

    Attributes:
        original_size: ``(height, width)`` of the original frame before any
            resizing. Used for plotting / un-cropping. ``(0, 0)`` is sentinel.
        processed_size: ``(height, width)`` of the post-preprocess input
            handed to the model.
        eff_scale: Per-sample size-matcher scale factor. Stored as a 1-D
            tensor ``(B,)``; broadcast against ``(B, ...)`` coords.
        input_scale: Scalar applied via :func:`apply_input_scale` before the
            forward pass. ``1.0`` is identity.
        output_stride: Confmap → input-pixel stride. ``>= 1``.
        pad_amount: ``(pad_h, pad_w)`` padding added to reach a stride-aligned
            shape. Currently informational; not used in coord ops.
        crop_offsets: ``(B*I, 2)`` top-left corner of each crop bbox, only
            populated by top-down stage 2. ``None`` for non-top-down layers.
    """

    original_size: Tuple[int, int] = (0, 0)
    processed_size: Tuple[int, int] = (0, 0)
    eff_scale: torch.Tensor = attrs.field(factory=lambda: torch.tensor([1.0]))
    input_scale: float = 1.0
    output_stride: int = 1
    pad_amount: Tuple[int, int] = (0, 0)
    crop_offsets: Optional[torch.Tensor] = None

    def __repr__(self) -> str:
        """Compact summary — never prints tensor contents."""
        eff_shape = tuple(self.eff_scale.shape)
        crops = (
            f"crop_offsets=Tensor{tuple(self.crop_offsets.shape)}"
            if self.crop_offsets is not None
            else "crop_offsets=None"
        )
        return (
            f"PreprocInfo(orig={self.original_size}, proc={self.processed_size}, "
            f"eff_scale=Tensor{eff_shape}, input_scale={self.input_scale}, "
            f"output_stride={self.output_stride}, pad={self.pad_amount}, {crops})"
        )
