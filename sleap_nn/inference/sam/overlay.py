"""Review/debug overlay rendering for predicted segmentation masks.

A v1 deliverable is *review-style* overlays (PLAN L4): masks are produced now;
the GUI correction round-trip is gated on upstream frontend work. This module
renders a colored per-instance mask overlay PNG so a human can eyeball the
predictions before importing the ``.slp``. Harvested from #642 ``_save_overlay``
and repurposed for ``PredictedSegmentationMask`` (which carries scale/offset, so
masks are decoded to the image grid first).

``cv2`` is imported lazily inside the function so importing this module never
pulls OpenCV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

#: Distinct per-instance overlay colors (RGB), cycled by instance index.
_COLORS = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 80, 255),
    (255, 255, 80),
    (255, 80, 255),
    (80, 255, 255),
    (255, 160, 80),
    (160, 80, 255),
]

#: Fixed warning color (RGB) for low-score masks flagged for human review.
_WARNING_COLOR = (255, 0, 0)


def save_mask_overlay(
    labels,
    path,
    frame_index: int = 0,
    low_score_threshold: Optional[float] = None,
) -> Optional[Path]:
    """Render image + colored per-instance mask overlay for one labeled frame.

    When ``low_score_threshold`` is given, any mask whose ``.score`` falls below
    it is rendered as a *low-score flag* so a human reviewer knows to scrutinize
    it: its contour is drawn in a fixed warning color (:data:`_WARNING_COLOR`)
    with a thicker outline, and a ``"!{score:.2f}"`` text label is drawn near the
    mask centroid. Masks at/above the threshold (and all masks when the threshold
    is ``None``) render normally with their cycled per-instance :data:`_COLORS`.

    Args:
        labels: A ``sio.Labels`` whose frames carry
            ``sio.PredictedSegmentationMask`` objects.
        path: Output PNG path.
        frame_index: Which labeled frame to render (default the first).
        low_score_threshold: Optional score floor; masks with ``.score`` strictly
            below it are flagged in the warning style (see above). ``None``
            (default) disables flagging and renders every mask normally.

    Returns:
        The written ``Path`` on success, or ``None`` if there were no frames /
        masks to render.
    """
    import cv2

    from sleap_nn.inference.segmentation_convert import decode_mask_to_image_res

    frames = list(labels.labeled_frames)
    if not frames or frame_index >= len(frames):
        return None
    lf = frames[frame_index]

    img = np.asarray(lf.image)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.ndim == 2:
        rgb = np.stack([img] * 3, axis=-1).astype(np.float32)
    else:
        rgb = img.astype(np.float32)
    H, W = rgb.shape[:2]

    masks = list(getattr(lf, "masks", []) or [])
    if not masks:
        return None

    for i, m in enumerate(masks):
        decoded = decode_mask_to_image_res(m)
        # decode_mask_to_image_res can return a +/-1 px or differently-sized
        # canvas (offset padding / scale rounding); clamp to the frame extent.
        mm = np.zeros((H, W), bool)
        hh, ww = min(H, decoded.shape[0]), min(W, decoded.shape[1])
        mm[:hh, :ww] = decoded[:hh, :ww]

        score = float(getattr(m, "score", 0.0))
        low_score = low_score_threshold is not None and score < low_score_threshold

        c = np.array(_COLORS[i % len(_COLORS)], dtype=np.float32)
        rgb[mm] = 0.5 * rgb[mm] + 0.5 * c
        cnts, _ = cv2.findContours(
            mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if low_score:
            # Flag for review: thicker warning-colored outline + score label so a
            # reviewer can spot masks below the per-model nominal floor.
            cv2.drawContours(rgb, cnts, -1, _WARNING_COLOR, 4)
            ys, xs = np.nonzero(mm)
            if xs.size:
                cy, cx = int(ys.mean()), int(xs.mean())
                cv2.putText(
                    rgb,
                    f"!{score:.2f}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    _WARNING_COLOR,
                    1,
                    cv2.LINE_AA,
                )
        else:
            cv2.drawContours(rgb, cnts, -1, tuple(int(x) for x in c), 2)

    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path.as_posix(), rgb[..., ::-1].astype(np.uint8))  # RGB->BGR
    return out_path
