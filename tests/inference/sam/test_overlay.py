"""CPU-only tests for the low-score review flag in ``save_mask_overlay``.

No SAM / torch model: a tiny synthetic image and two
``sio.PredictedSegmentationMask`` objects (one above, one below a chosen
threshold) built through the codebase's standard packaging path
(``build_predicted_segmentation_mask``) exercise §3.3 — ``pred_iou_min`` consumed
as a per-model review signal. We assert the overlay PNG is written with and
without the flag, and that turning the flag on changes the rendered pixels
(more warning-red).
"""

from pathlib import Path

import numpy as np

import sleap_io as sio

from sleap_nn.inference.segmentation_convert import build_predicted_segmentation_mask
from sleap_nn.inference.sam.overlay import save_mask_overlay


def _disk(h, w, cy, cx, r):
    """A boolean ``(h, w)`` disk of radius ``r`` centered at ``(cy, cx)``."""
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2


def _labels_two_masks(tmp_path, high_score=0.95, low_score=0.40):
    """A ``sio.Labels`` with one frame carrying two masks at distinct scores.

    A tiny PNG is written to ``tmp_path`` and loaded as a ``sio.Video`` (so
    ``lf.image`` yields real pixels without any heavyweight fixture), and the two
    masks are built through the codebase's standard
    ``build_predicted_segmentation_mask`` path so they are genuine
    ``sio.PredictedSegmentationMask`` objects with full-res scale/offset.

    Args:
        tmp_path: Pytest tmp dir to write the backing image into.
        high_score: Score for the first (above-threshold) mask.
        low_score: Score for the second (below-threshold) mask.
    """
    import cv2

    h, w = 64, 64
    img_path = Path(tmp_path) / "frame.png"
    cv2.imwrite(img_path.as_posix(), np.zeros((h, w, 3), dtype=np.uint8))
    video = sio.Video.from_filename(img_path.as_posix())

    m_high = build_predicted_segmentation_mask(_disk(h, w, 18, 18, 10), high_score)
    m_low = build_predicted_segmentation_mask(_disk(h, w, 46, 46, 10), low_score)

    lf = sio.LabeledFrame(video=video, frame_idx=0, masks=[m_high, m_low])
    return sio.Labels(videos=[video], skeletons=[], labeled_frames=[lf])


def _red_pixels(img_bgr):
    """Count strongly-red BGR pixels (warning color is RGB (255, 0, 0))."""
    b, g, r = img_bgr[..., 0], img_bgr[..., 1], img_bgr[..., 2]
    return int(np.count_nonzero((r > 200) & (g < 80) & (b < 80)))


def test_save_mask_overlay_no_threshold_writes_png(tmp_path):
    """``low_score_threshold=None`` writes a PNG and renders both masks."""
    import cv2

    labels = _labels_two_masks(tmp_path)
    out_path = tmp_path / "overlay.png"

    result = save_mask_overlay(labels, out_path, low_score_threshold=None)
    assert result is not None
    assert result.exists()

    img = cv2.imread(result.as_posix())
    assert img is not None
    assert img.ndim == 3 and img.shape[-1] == 3


def test_save_mask_overlay_threshold_between_scores_writes_png(tmp_path):
    """A threshold between the two scores still writes a PNG without error."""
    import cv2

    labels = _labels_two_masks(tmp_path, high_score=0.95, low_score=0.40)
    out_path = tmp_path / "overlay_flagged.png"

    # 0.5 sits between the two scores, so exactly one mask is flagged.
    result = save_mask_overlay(labels, out_path, low_score_threshold=0.5)
    assert result is not None
    assert result.exists()

    img = cv2.imread(result.as_posix())
    assert img is not None and img.ndim == 3 and img.shape[-1] == 3


def test_save_mask_overlay_flag_changes_rendering(tmp_path):
    """Flagging a low-score mask adds warning-red pixels vs the unflagged render."""
    import cv2

    labels = _labels_two_masks(tmp_path, high_score=0.95, low_score=0.40)

    off_path = tmp_path / "off.png"
    on_path = tmp_path / "on.png"

    save_mask_overlay(labels, off_path, low_score_threshold=None)
    save_mask_overlay(labels, on_path, low_score_threshold=0.5)

    off = cv2.imread(off_path.as_posix())
    on = cv2.imread(on_path.as_posix())
    assert off is not None and on is not None

    # The flagged render must differ from the unflagged one and carry strictly
    # more warning-red pixels (thicker red contour + the "!{score}" label).
    assert not np.array_equal(off, on)
    assert _red_pixels(on) > _red_pixels(off)


def test_save_mask_overlay_returns_path(tmp_path):
    """The returned value is the written ``Path`` and points at the PNG."""
    labels = _labels_two_masks(tmp_path)
    out_path = tmp_path / "ret.png"
    result = save_mask_overlay(labels, out_path, low_score_threshold=0.5)
    assert isinstance(result, Path)
    assert result == out_path
