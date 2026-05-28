"""``PreprocessConfig`` / ``PostprocessConfig`` — value types parameterizing layers.

Both are ``attrs.frozen`` so they hash, serialize, and never silently mutate
after a layer has been constructed. ``PostprocessConfig`` is shared by every
layer subclass; ``PreprocessConfig`` mirrors the ``data_config.preprocessing``
section of the training config so layer factories can populate it directly.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import attrs


@attrs.frozen
class PreprocessConfig:
    """Preprocessing knobs applied before the model forward pass.

    Defaults are the no-op identity for every field — calling the layer
    on an already-correctly-shaped batch produces zero extra work.

    Attributes:
        ensure_rgb: ``True`` forces 3-channel RGB; ``False`` forces 1-channel
            grayscale; ``None`` leaves channels untouched.
        ensure_grayscale: Inverse of ``ensure_rgb``. Mutually exclusive.
        max_height: Resize so height ≤ this (preserves aspect ratio). ``None``
            means no max.
        max_width: Same for width.
        scale: Multiplicative input-scale factor applied via
            :func:`apply_input_scale` after size matching. ``1.0`` is identity.
        crop_size: Top-down stage 2 only — square crop side length.
    """

    ensure_rgb: Optional[bool] = None
    ensure_grayscale: Optional[bool] = None
    max_height: Optional[int] = None
    max_width: Optional[int] = None
    scale: float = 1.0
    crop_size: Optional[Tuple[int, int]] = None


@attrs.frozen
class PostprocessConfig:
    """Knobs that govern how raw model outputs become keypoints.

    Distinct from the post-inference ``FilterConfig``: this struct governs
    the *decoding* step (peak finding, integral refinement, NMS), while
    ``FilterConfig`` filters the keypoints that come out the other side.
    ``peak_threshold`` here decides which confmap pixels become peaks;
    ``min_peak_value`` in ``FilterConfig`` filters peaks the decoder
    already returned.

    Attributes:
        peak_threshold: Minimum confmap activation to consider a peak.
        refinement: ``"integral"`` runs sub-pixel integral regression around
            each rough peak. ``"none"`` returns grid-aligned peaks.
        integral_patch_size: Side length of the refinement patch.
        max_instances: Cap on instances per frame (centroid layer only).
        return_confmaps: Keep ``(B, N, H, W)`` confmaps on the ``Outputs``
            (heavy — opt-in for visualization / debugging).
        return_pafs: Keep ``(B, 2E, H, W)`` PAFs (bottom-up only; heavy).
        return_paf_graph: Keep the bottom-up PAF graph tuple (opt-in).
        return_class_maps: Keep ``(B, C, H, W)`` class maps (multi-class
            bottom-up; heavy).
        return_class_vectors: Keep ``(B, I, N, C)`` class logits
            (multi-class top-down).
    """

    peak_threshold: float = 0.2
    refinement: Literal["integral", "none"] = "integral"
    integral_patch_size: int = 5
    max_instances: Optional[int] = None

    return_confmaps: bool = False
    return_pafs: bool = False
    return_paf_graph: bool = False
    return_class_maps: bool = False
    return_class_vectors: bool = False

    @property
    def effective_refinement(self) -> Optional[str]:
        """Return the refinement string or ``None`` when ``"none"``.

        Every postprocess site needs ``refinement=None`` (not the string
        ``"none"``) to disable refinement. This property centralises that
        coercion.
        """
        return self.refinement if self.refinement != "none" else None
