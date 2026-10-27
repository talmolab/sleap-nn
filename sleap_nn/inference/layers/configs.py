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
        scale: Multiplicative input-scale factor applied (after size matching)
            via :func:`sleap_nn.data.resizing.resize_image` (``tvf.resize``) on
            the live ckpt path. ``1.0`` is identity.
        crop_size: Top-down stage 2 only — square crop side length.
        resize_mode: How frames are fitted to ``(max_height, max_width)``:
            ``"letterbox"`` (one scale, then pad) or ``"squash"`` (resize to
            exactly that size, a separate scale per axis, so ``eff_scale`` is
            ``(B, 2)`` ``(x, y)``). Squash needs both ``max_height`` and
            ``max_width``.
    """

    ensure_rgb: Optional[bool] = None
    ensure_grayscale: Optional[bool] = None
    max_height: Optional[int] = None
    max_width: Optional[int] = None
    scale: float = 1.0
    crop_size: Optional[Tuple[int, int]] = None
    resize_mode: str = "letterbox"

    def __attrs_post_init__(self) -> None:
        """Reject contradictory channel or resize settings."""
        if self.ensure_rgb and self.ensure_grayscale:
            raise ValueError(
                "ensure_rgb and ensure_grayscale cannot both be True; choose one "
                "(or leave both None to keep the source channel count)."
            )
        if self.resize_mode not in ("letterbox", "squash"):
            raise ValueError(
                f"resize_mode must be 'letterbox' or 'squash', got {self.resize_mode!r}."
            )
        if self.resize_mode == "squash" and (
            self.max_height is None or self.max_width is None
        ):
            raise ValueError("resize_mode='squash' needs max_height and max_width.")


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
        refinement: Sub-pixel peak refinement. ``"auto"`` (default) is
            ``"dark"`` for the top-down keypoint stage (centered-instance and
            multi-class top-down layers) and ``"integral"`` for every other
            layer. ``"dark"`` runs DARK (a Newton step on the blurred log
            confmap); ``"local"`` moves each peak a quarter pixel towards its
            higher neighbor; ``"integral"`` runs integral regression over an
            ``integral_patch_size`` patch; ``"none"`` returns grid-aligned
            peaks. ``"dark"`` / ``"local"`` need one global peak per channel:
            the centroid layer falls back to ``"integral"`` for them and the
            bottom-up layers raise. See
            :func:`sleap_nn.inference.ops.peaks.resolve_refinement`.
        integral_patch_size: Side length of the refinement patch.
        dark_blur_sigma: Std (confmap pixels) of the DARK modulation blur. The
            loaders set it to the head's training ``sigma``.
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
    refinement: Literal["auto", "dark", "integral", "local", "none"] = "auto"
    integral_patch_size: int = 5
    dark_blur_sigma: float = 1.0
    max_instances: Optional[int] = None

    return_confmaps: bool = False
    return_pafs: bool = False
    return_paf_graph: bool = False
    return_class_maps: bool = False
    return_class_vectors: bool = False

    def resolve_refinement(
        self, auto: str = "integral", peaks: str = "global"
    ) -> Optional[str]:
        """Return the refinement a peak finder runs (``None`` to disable).

        Resolves ``"auto"`` to ``auto`` and ``"none"`` to ``None``, and
        applies the global-peak-only rule for ``"dark"`` / ``"local"``; see
        :func:`sleap_nn.inference.ops.peaks.resolve_refinement`.
        """
        from sleap_nn.inference.ops.peaks import resolve_refinement

        return resolve_refinement(self.refinement, auto=auto, peaks=peaks)
