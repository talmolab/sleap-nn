"""Prompt builders for SAM-prompted instance segmentation (PR-A).

A *prompt* is the geometric hint handed to a SAM backend
(:mod:`sleap_nn.inference.sam.backends`) for one instance: positive point
coordinates, an optional box, and the keypoint-box used by the candidate
rejection heuristic (:func:`sleap_nn.inference.sam.backends._pick`). All
coordinates are in the pixel space of the image the backend will encode (the
full frame for the pose/centroid/box modes, the crop for the top-down
crop-center-pixel mode).

Four prompt modes (v1 = prompted-only, PLAN L3), each validated by a prototype:

* :func:`pose_prompt` — every visible keypoint as a positive point **plus** the
  padded keypoint box (the locked exp-07 recipe; cleanest — P1).
* :func:`centroid_prompt` — a single positive point (the predicted centroid /
  anchor, or the mean of the visible keypoints); ``multimask_output`` is
  essential and the keypoint box is kept only for candidate rejection (P1/P2).
* :func:`box_prompt` — the padded pose/crop box as the only prompt, no points
  (leaks between adjacent animals; secondary — P1/P2).
* :func:`crop_center_prompt` — the naive crop center ``(w/2, h/2)`` of an
  instance-centered crop; needs only a centroid, no pose (the top-down seam,
  byte-identical packaging — P2).

The product rule (PLAN L3 / P2): use the pose prompt when an instance has
visible keypoints, else fall back to the centroid/crop-center point.
:func:`prompt_for_instance` implements exactly that dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Locked SAM prompt recipe constants (harvested from #642 / exp-07; PLAN §1).
# These are the single source of truth for the default prompt recipe.
# ---------------------------------------------------------------------------
#: Keypoint-box margin: ``max(SAM_BOX_MARGIN_MIN, SAM_BOX_MARGIN_FRAC * side)``.
SAM_BOX_MARGIN_FRAC: float = 0.6
SAM_BOX_MARGIN_MIN: float = 15.0

#: Available prompt modes (v1; PLAN L3). ``"pose"`` falls back to a center point
#: per-instance when no keypoints are visible (the product rule).
PROMPT_MODES: Tuple[str, ...] = ("pose", "centroid", "box", "crop_center")


@dataclass
class SamPrompt:
    """A built SAM prompt for one instance.

    Attributes:
        point_coords: ``(n, 2)`` float32 positive-point xy, or ``None`` when the
            prompt is box-only.
        point_labels: ``(n,)`` int32 labels (all ``1`` — positive; automatic
            prompting uses no negatives, PLAN §2.2). ``None`` iff
            ``point_coords`` is ``None``.
        box: ``[x0, y0, x1, y1]`` float32 box prompt, or ``None`` when the prompt
            is point-only.
        reject_box: ``[x0, y0, x1, y1]`` float32 box used **only** by the
            candidate-rejection heuristic (:func:`backends._pick`) — never passed
            to SAM. Always populated so :func:`backends._pick` can size-reject the
            whole-arena candidate even in point-only modes.
        mode: The originating mode tag (``"pose"`` / ``"centroid"`` / ``"box"`` /
            ``"crop_center"``), carried for diagnostics / overlays.
    """

    point_coords: Optional[np.ndarray]
    point_labels: Optional[np.ndarray]
    box: Optional[np.ndarray]
    reject_box: np.ndarray
    mode: str


def visible_keypoints(points: np.ndarray) -> np.ndarray:
    """Return the finite ``(m, 2)`` rows of an ``(n, 2)`` keypoint array.

    Drops any keypoint with a non-finite (``NaN`` / ``inf``) x or y, matching the
    prototype's ``k[np.isfinite(k).all(1)]`` visibility filter.

    Args:
        points: ``(n, 2)`` xy keypoints (may carry ``NaN`` for missing nodes).

    Returns:
        ``(m, 2)`` float32 array of the visible keypoints (``m`` may be 0).
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return pts[np.isfinite(pts).all(axis=1)]


def kpt_box(
    pos: np.ndarray,
    hw: Tuple[int, int],
    margin_frac: float = SAM_BOX_MARGIN_FRAC,
    margin_min: float = SAM_BOX_MARGIN_MIN,
) -> np.ndarray:
    """Padded keypoint bounding box ``[x0, y0, x1, y1]`` clamped to ``hw``.

    Harvested verbatim from #642 ``_kpt_box`` (the locked exp-07 recipe): the
    per-axis margin is ``max(margin_min, margin_frac * side)``, so the box grows
    with the instance but never collapses below ``margin_min`` px for a small or
    degenerate (single-point) instance.

    Args:
        pos: ``(n, 2)`` visible xy keypoints (no NaNs; ``n >= 1``).
        hw: ``(height, width)`` of the image the box lives in.
        margin_frac: Box margin as a fraction of the box side length.
        margin_min: Minimum box margin (px) per axis.

    Returns:
        ``float32`` array ``[x0, y0, x1, y1]`` clamped to ``[0, w-1] x [0, h-1]``.
    """
    pos = np.asarray(pos, dtype=np.float32).reshape(-1, 2)
    x0, y0 = pos.min(0)
    x1, y1 = pos.max(0)
    mx = max(margin_min, margin_frac * (x1 - x0))
    my = max(margin_min, margin_frac * (y1 - y0))
    return np.array(
        [
            max(0.0, x0 - mx),
            max(0.0, y0 - my),
            min(hw[1] - 1.0, x1 + mx),
            min(hw[0] - 1.0, y1 + my),
        ],
        np.float32,
    )


def pose_prompt(
    keypoints: np.ndarray,
    hw: Tuple[int, int],
    margin_frac: float = SAM_BOX_MARGIN_FRAC,
    margin_min: float = SAM_BOX_MARGIN_MIN,
) -> SamPrompt:
    """Pose prompt: visible keypoints as positive points + the padded kpt-box.

    The strongest prompt (P1: containment 1.0, 72% single-component, no
    blow-ups). No negative points (automatic prompting only; PLAN §2.2).

    Args:
        keypoints: ``(n, 2)`` xy keypoints; non-finite rows are dropped here.
        hw: ``(height, width)`` of the image being prompted.
        margin_frac: Keypoint-box margin fraction (:func:`kpt_box`).
        margin_min: Keypoint-box minimum margin (:func:`kpt_box`).

    Returns:
        A :class:`SamPrompt` with both ``point_coords`` and ``box`` set.

    Raises:
        ValueError: If no keypoints are visible (caller should fall back to a
            center point — see :func:`prompt_for_instance`).
    """
    pts = visible_keypoints(keypoints)
    if len(pts) == 0:
        raise ValueError(
            "pose_prompt requires at least one visible keypoint; the caller "
            "should fall back to a centroid/crop-center point."
        )
    box = kpt_box(pts, hw, margin_frac=margin_frac, margin_min=margin_min)
    return SamPrompt(
        point_coords=pts.astype(np.float32),
        point_labels=np.ones(len(pts), np.int32),
        box=box,
        reject_box=box,
        mode="pose",
    )


def centroid_prompt(
    point: np.ndarray,
    hw: Tuple[int, int],
    keypoints: Optional[np.ndarray] = None,
    margin_frac: float = SAM_BOX_MARGIN_FRAC,
    margin_min: float = SAM_BOX_MARGIN_MIN,
) -> SamPrompt:
    """Centroid prompt: a single positive point, ``multimask_output`` essential.

    The box is **not** passed to SAM (a lone point is maximally ambiguous —
    P1/P2); it is computed only for the candidate-rejection heuristic. When
    ``keypoints`` is given the keypoint box is used; otherwise a fixed margin box
    around the point is used as the reject box.

    Args:
        point: The ``(2,)`` anchor xy (predicted centroid / mean point).
        hw: ``(height, width)`` of the image being prompted.
        keypoints: Optional ``(n, 2)`` keypoints to size the reject box; when
            absent a ``margin_min``-radius box around ``point`` is used.
        margin_frac: Reject-box margin fraction.
        margin_min: Reject-box minimum margin.

    Returns:
        A :class:`SamPrompt` with ``point_coords`` set and ``box`` ``None``.
    """
    pt = np.asarray(point, dtype=np.float32).reshape(1, 2)
    vis = visible_keypoints(keypoints) if keypoints is not None else np.empty((0, 2))
    if len(vis) > 0:
        reject = kpt_box(vis, hw, margin_frac=margin_frac, margin_min=margin_min)
    else:
        cx, cy = float(pt[0, 0]), float(pt[0, 1])
        reject = np.array(
            [
                max(0.0, cx - margin_min),
                max(0.0, cy - margin_min),
                min(hw[1] - 1.0, cx + margin_min),
                min(hw[0] - 1.0, cy + margin_min),
            ],
            np.float32,
        )
    return SamPrompt(
        point_coords=pt,
        point_labels=np.ones(1, np.int32),
        box=None,
        reject_box=reject,
        mode="centroid",
    )


def box_prompt(
    keypoints: np.ndarray,
    hw: Tuple[int, int],
    margin_frac: float = SAM_BOX_MARGIN_FRAC,
    margin_min: float = SAM_BOX_MARGIN_MIN,
) -> SamPrompt:
    """Box prompt: the padded pose/crop box as the only prompt, no points.

    Secondary mode (P1/P2: leaks between adjacent animals). The same box is both
    the SAM prompt and the reject box.

    Args:
        keypoints: ``(n, 2)`` xy keypoints; non-finite rows are dropped here.
        hw: ``(height, width)`` of the image being prompted.
        margin_frac: Keypoint-box margin fraction (:func:`kpt_box`).
        margin_min: Keypoint-box minimum margin (:func:`kpt_box`).

    Returns:
        A :class:`SamPrompt` with ``box`` set and ``point_coords`` ``None``.

    Raises:
        ValueError: If no keypoints are visible.
    """
    pts = visible_keypoints(keypoints)
    if len(pts) == 0:
        raise ValueError("box_prompt requires at least one visible keypoint.")
    box = kpt_box(pts, hw, margin_frac=margin_frac, margin_min=margin_min)
    return SamPrompt(
        point_coords=None,
        point_labels=None,
        box=box,
        reject_box=box,
        mode="box",
    )


def crop_center_prompt(crop_hw: Tuple[int, int]) -> SamPrompt:
    """Top-down crop-center-pixel prompt: the naive crop center ``(w/2, h/2)``.

    For an instance-centered crop, the crop center *is* the centroid (the crop
    math centers it there). Needs only a centroid — no pose (P2). The reject box
    is the full crop extent.

    Args:
        crop_hw: ``(crop_h, crop_w)`` of the crop being prompted.

    Returns:
        A :class:`SamPrompt` with a single center ``point_coords`` and the full
        crop as the reject box.
    """
    ch, cw = int(crop_hw[0]), int(crop_hw[1])
    center = np.array([[cw / 2.0, ch / 2.0]], np.float32)
    reject = np.array([0.0, 0.0, cw - 1.0, ch - 1.0], np.float32)
    return SamPrompt(
        point_coords=center,
        point_labels=np.ones(1, np.int32),
        box=None,
        reject_box=reject,
        mode="crop_center",
    )


def prompt_for_instance(
    mode: str,
    hw: Tuple[int, int],
    keypoints: Optional[np.ndarray] = None,
    centroid: Optional[np.ndarray] = None,
    margin_frac: float = SAM_BOX_MARGIN_FRAC,
    margin_min: float = SAM_BOX_MARGIN_MIN,
) -> SamPrompt:
    """Dispatch to the right prompt builder, applying the L3 product rule.

    The product rule (PLAN L3 / P2): for ``mode="pose"`` use the pose prompt when
    the instance has visible keypoints, else fall back to a center point (the
    centroid if given, else an error). For ``mode="crop_center"`` use the richer
    crop-local pose prompt when crop-local keypoints are present, else the naive
    center pixel. ``"centroid"`` / ``"box"`` always use their named builder.

    Args:
        mode: One of :data:`PROMPT_MODES`.
        hw: ``(height, width)`` of the image being prompted. For ``crop_center``
            this is the crop size.
        keypoints: ``(n, 2)`` xy keypoints (image- or crop-local space depending
            on the call site). Required for ``pose`` / ``box``.
        centroid: ``(2,)`` anchor xy. Required for ``centroid``; used as the
            pose-fallback point when ``mode="pose"`` and no keypoint is visible.
        margin_frac: Keypoint-box margin fraction.
        margin_min: Keypoint-box minimum margin.

    Returns:
        A built :class:`SamPrompt`.

    Raises:
        ValueError: If ``mode`` is unknown, or a required prompt source is
            missing (e.g. ``mode="box"`` with no visible keypoints, or
            ``mode="pose"`` with neither keypoints nor a centroid fallback).
    """
    if mode not in PROMPT_MODES:
        raise ValueError(
            f"Unknown prompt mode {mode!r}; expected one of {PROMPT_MODES}."
        )

    if mode == "crop_center":
        if keypoints is not None and len(visible_keypoints(keypoints)) > 0:
            # Pose-if-exists rule, in crop-local space.
            return pose_prompt(
                keypoints, hw, margin_frac=margin_frac, margin_min=margin_min
            )
        return crop_center_prompt(hw)

    if mode == "centroid":
        if centroid is None:
            if keypoints is not None and len(visible_keypoints(keypoints)) > 0:
                centroid = visible_keypoints(keypoints).mean(0)
            else:
                raise ValueError(
                    "centroid prompt requires a centroid (or visible keypoints "
                    "to average)."
                )
        return centroid_prompt(
            centroid,
            hw,
            keypoints=keypoints,
            margin_frac=margin_frac,
            margin_min=margin_min,
        )

    if mode == "box":
        return box_prompt(keypoints, hw, margin_frac=margin_frac, margin_min=margin_min)

    # mode == "pose": pose-if-exists-else-center (the L3 product rule).
    has_kpts = keypoints is not None and len(visible_keypoints(keypoints)) > 0
    if has_kpts:
        return pose_prompt(
            keypoints, hw, margin_frac=margin_frac, margin_min=margin_min
        )
    if centroid is not None:
        return centroid_prompt(
            centroid,
            hw,
            keypoints=None,
            margin_frac=margin_frac,
            margin_min=margin_min,
        )
    raise ValueError(
        "pose prompt has no visible keypoints and no centroid fallback was " "provided."
    )
