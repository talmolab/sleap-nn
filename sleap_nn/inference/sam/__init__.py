"""SAM-powered prompted instance segmentation for INFERENCE (PR-A).

The pivot (PLAN / README): SAM is used to *predict* per-instance masks for an
existing pose/centroid ``.slp`` so a human can review/correct them in the GUI,
then train — **not** to auto-generate training GT. This package is the SAM1
prompted producer + its backend interface; SAM3, reconciliation, and the
mask-native tracker land in later PRs behind the same surfaces.

Public surface
--------------
* :func:`get_mask_backend` — **explicit, no-default** backend selection
  (PLAN L2). ``"sam"`` builds a SAM1 :class:`~.backends.SamBackend`; an unknown
  or omitted name raises. ``"sam3"`` is reserved for PR-B and currently errors
  with a pointer.
* :func:`run_sam_segmentation` — end-to-end orchestration: load a pose ``.slp``,
  run the chosen backend with the chosen prompt mode, emit
  ``sio.PredictedSegmentationMask`` (raw score + ``instance=``/``track=``
  populated, PLAN L8) onto each frame, and optionally save an embedded ``.slp``
  + a review overlay PNG.

Everything heavy (``segment-anything``) is imported lazily inside the backend,
so importing this package on a default install is cheap and dependency-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sleap_nn.inference.sam.backends import MaskBackend, SamBackend
from sleap_nn.inference.sam.mask_layer import (
    FindInstanceMaskSAM,
    SamSegmentationLayer,
)
from sleap_nn.inference.sam.prompts import PROMPT_MODES, SamPrompt

__all__ = [
    "MaskBackend",
    "SamBackend",
    "SamSegmentationLayer",
    "FindInstanceMaskSAM",
    "SamPrompt",
    "PROMPT_MODES",
    "MASK_BACKENDS",
    "get_mask_backend",
    "run_sam_segmentation",
]

#: Registered explicit ``mask_backend`` names (PLAN L2 — no default).
MASK_BACKENDS = ("sam", "sam3")


def get_mask_backend(
    mask_backend: str,
    *,
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_h",
    device: str = "cuda",
    **kwargs,
) -> MaskBackend:
    """Build a mask backend by **explicit** name (no default; PLAN L2).

    Args:
        mask_backend: The backend name. ``"sam"`` builds a SAM1
            :class:`~.backends.SamBackend`; ``"sam3"`` is reserved for PR-B.
            There is no default — the caller must name one.
        sam_checkpoint: Path to the SAM1 checkpoint (required for ``"sam"``).
        sam_model_type: SAM1 model registry key.
        device: Torch device for the model.
        **kwargs: Forwarded to the backend constructor (e.g. ``clahe``).

    Returns:
        A ready :class:`~.backends.MaskBackend`.

    Raises:
        ValueError: If ``mask_backend`` is not a registered name.
        NotImplementedError: If ``mask_backend == "sam3"`` (lands in PR-B).
    """
    if mask_backend is None:
        raise ValueError(
            "mask_backend is required and has no default; pass one of "
            f"{MASK_BACKENDS} (PLAN L2)."
        )
    name = str(mask_backend).lower()
    if name == "sam":
        return SamBackend.from_checkpoint(
            sam_checkpoint,
            model_type=sam_model_type,
            device=device,
            **kwargs,
        )
    if name == "sam3":
        raise NotImplementedError(
            "mask_backend='sam3' is added in PR-B (the SAM3 backend behind the "
            "'sam3' extra). Use mask_backend='sam' for SAM1."
        )
    raise ValueError(
        f"Unknown mask_backend {mask_backend!r}; expected one of {MASK_BACKENDS}."
    )


def run_sam_segmentation(
    source,
    mask_backend: str,
    *,
    prompt_mode: str = "pose",
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_h",
    device: str = "cuda",
    anchor_ind: Optional[int] = None,
    disjointify_masks: bool = False,
    backend: Optional[MaskBackend] = None,
    output_path: Optional[str] = None,
    overlay_path: Optional[str] = None,
):
    """Predict per-instance masks for a pose ``.slp`` with a SAM backend.

    Loads (or accepts) a ``sio.Labels`` whose frames carry pose/centroid
    instances, runs the chosen backend with the chosen prompt mode, attaches one
    ``sio.PredictedSegmentationMask`` per instance (raw score + ``instance=`` /
    ``track=`` populated, PLAN L8), and returns a new ``sio.Labels``. The
    instances are retained alongside the masks (correction needs the pose).

    Args:
        source: A path to a pose ``.slp``/``.pkg.slp`` (with image data) or an
            in-memory ``sio.Labels``.
        mask_backend: **Explicit** backend name (PLAN L2): ``"sam"`` / ``"sam3"``.
        prompt_mode: ``"pose"`` / ``"centroid"`` / ``"box"`` (full-frame). The
            crop-center-pixel mode is the top-down seam
            (:class:`~.mask_layer.FindInstanceMaskSAM`), not this entry point.
        sam_checkpoint: SAM1 checkpoint path (required for ``"sam"`` unless a
            pre-built ``backend`` is passed).
        sam_model_type: SAM1 model registry key.
        device: Torch device for the model.
        anchor_ind: Optional centroid anchor node index for ``"centroid"``.
        disjointify_masks: Make per-frame masks disjoint when >=2 instances.
        backend: A pre-built :class:`~.backends.MaskBackend` to use directly
            (skips loading); when given, ``mask_backend`` is still validated for
            the name but the checkpoint/device args are ignored.
        output_path: Optional path to save the result embedded (``.slp``).
        overlay_path: Optional path to write a review overlay PNG of the first
            frame.

    Returns:
        A new ``sio.Labels`` with per-frame ``PredictedSegmentationMask`` (and the
        original pose instances retained).
    """
    import sleap_io as sio

    from sleap_nn.inference.outputs import Outputs
    from sleap_nn.inference.sam.overlay import save_mask_overlay

    if isinstance(source, sio.Labels):
        labels = source
    else:
        labels = sio.load_slp(Path(source).expanduser().as_posix())

    if backend is None:
        backend = get_mask_backend(
            mask_backend,
            sam_checkpoint=sam_checkpoint,
            sam_model_type=sam_model_type,
            device=device,
        )
    elif mask_backend not in MASK_BACKENDS:
        raise ValueError(
            f"Unknown mask_backend {mask_backend!r}; expected one of {MASK_BACKENDS}."
        )

    layer = SamSegmentationLayer(
        backend,
        prompt_mode=prompt_mode,
        anchor_ind=anchor_ind,
        disjointify_masks=disjointify_masks,
    )

    new_lfs = []
    for lf in labels.labeled_frames:
        frame_masks = layer.masks_for_frame(lf.image, lf.instances)
        if not frame_masks:
            continue
        # Reuse the standard packaging path (build_predicted_segmentation_mask).
        masks = Outputs(pred_masks=[frame_masks]).to_masks(0)
        new_lfs.append(
            sio.LabeledFrame(
                video=lf.video,
                frame_idx=lf.frame_idx,
                instances=list(lf.instances),  # retain poses for correction
                masks=masks,
            )
        )

    out = sio.Labels(
        videos=list(labels.videos),
        skeletons=list(labels.skeletons),
        labeled_frames=new_lfs,
    )

    if output_path is not None:
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(out_path.as_posix(), embed=True)
    if overlay_path is not None:
        save_mask_overlay(out, overlay_path)

    return out
