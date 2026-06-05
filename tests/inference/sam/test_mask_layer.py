"""Unit tests for the SAM mask layer (CPU; a fake backend, no real SAM).

Covers:
* :class:`SamSegmentationLayer` full-frame placement + ``instance=``/``track=``
  population + SLP round-trip.
* ``mask_backend`` is explicit / required (no default).
"""

import numpy as np
import pytest

import sleap_io as sio

from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.sam.mask_layer import SamSegmentationLayer


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2


class FakeBackend:
    """A :class:`MaskBackend`-shaped fake that returns a fixed mask per prompt.

    ``mask_fn(image, prompt)`` produces one ``(H, W)`` bool mask; the score is a
    fixed value. Records each encoded image for assertions.
    """

    pred_iou_min = 0.88

    def __init__(self, mask_fn, score=0.77):
        self._mask_fn = mask_fn
        self._score = score
        self.encoded = []

    def masks(self, image, prompts):
        image = np.asarray(image)
        self.encoded.append(image.shape)
        out_masks, out_scores = [], []
        for p in prompts:
            out_masks.append(self._mask_fn(image, p).astype(bool))
            out_scores.append(float(self._score))
        return out_masks, out_scores


# --------------------------------------------------------------------------- SamSegmentationLayer


def _labels_with_one_pose(h=100, w=120):
    skel = sio.Skeleton([sio.Node("a"), sio.Node("b"), sio.Node("c")])
    pts = np.array([[40.0, 50.0], [60.0, 70.0], [np.nan, np.nan]])
    track = sio.Track("m0")
    inst = sio.PredictedInstance.from_numpy(
        points_data=pts,
        skeleton=skel,
        point_scores=np.array([0.9, 0.8, 0.0]),
        score=0.85,
    )
    inst.track = track
    inst.tracking_score = 0.95
    video = sio.Video.from_filename("dummy.mp4")
    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
    # Attach a synthetic image so masks_for_frame has something to encode.
    img = (np.random.rand(h, w) * 255).astype(np.uint8)
    return skel, video, lf, img, track, inst


def test_sam_segmentation_layer_populates_instance_and_track():
    skel, video, lf, img, track, inst = _labels_with_one_pose()

    def kpt_disk(image, prompt):
        # A disk over the keypoints.
        pts = prompt.point_coords
        c = pts.mean(0)
        return _disk(image.shape[0], image.shape[1], c[1], c[0], 12)

    layer = SamSegmentationLayer(FakeBackend(kpt_disk, score=0.66), prompt_mode="pose")
    frame_masks = layer.masks_for_frame(img, [inst])
    assert len(frame_masks) == 1
    d = frame_masks[0]
    assert d["scale"] == (1.0, 1.0) and d["offset"] == (0.0, 0.0)
    assert abs(d["score"] - 0.66) < 1e-6
    # PLAN L8: instance + track populated.
    assert d["instance"] is inst
    assert d["track"] is track
    assert abs(d["tracking_score"] - 0.95) < 1e-6

    # Package via the standard path; the sio mask carries the provenance.
    masks = Outputs(pred_masks=[frame_masks]).to_masks(0)
    assert len(masks) == 1
    m = masks[0]
    assert isinstance(m, sio.PredictedSegmentationMask)
    assert m.instance is inst
    assert m.track is track
    assert abs(m.tracking_score - 0.95) < 1e-6
    assert abs(m.score - 0.66) < 1e-6


def test_sam_segmentation_layer_gt_instance_not_set_as_instance():
    """A GT ``sio.Instance`` (not a prediction) is NOT pinned to ``mask.instance``."""
    skel = sio.Skeleton([sio.Node("a"), sio.Node("b")])
    gt = sio.Instance.from_numpy(np.array([[40.0, 50.0], [60.0, 70.0]]), skeleton=skel)
    img = (np.random.rand(100, 120) * 255).astype(np.uint8)

    layer = SamSegmentationLayer(
        FakeBackend(lambda image, p: _disk(image.shape[0], image.shape[1], 60, 50, 12)),
        prompt_mode="pose",
    )
    frame_masks = layer.masks_for_frame(img, [gt])
    assert len(frame_masks) == 1
    assert frame_masks[0]["instance"] is None


def test_sam_segmentation_layer_skips_instances_without_prompt():
    skel = sio.Skeleton([sio.Node("a")])
    img = (np.random.rand(40, 40) * 255).astype(np.uint8)
    # An instance with no visible keypoints and no centroid -> skipped.
    empty = sio.PredictedInstance.from_numpy(
        points_data=np.array([[np.nan, np.nan]]),
        skeleton=skel,
        point_scores=np.array([0.0]),
        score=0.0,
    )
    layer = SamSegmentationLayer(
        FakeBackend(lambda image, p: np.ones((40, 40), bool)), prompt_mode="pose"
    )
    assert layer.masks_for_frame(img, [empty]) == []


def test_sam_segmentation_layer_rejects_unknown_mode():
    with pytest.raises(ValueError):
        SamSegmentationLayer(object(), prompt_mode="nope")


def test_sam_segmentation_disjointify_multi_instance():
    skel = sio.Skeleton([sio.Node("a")])
    img = (np.random.rand(50, 50) * 255).astype(np.uint8)
    i0 = sio.PredictedInstance.from_numpy(
        points_data=np.array([[15.0, 25.0]]),
        skeleton=skel,
        point_scores=np.array([0.9]),
        score=0.9,
    )
    i1 = sio.PredictedInstance.from_numpy(
        points_data=np.array([[35.0, 25.0]]),
        skeleton=skel,
        point_scores=np.array([0.9]),
        score=0.9,
    )

    def big_disk(image, prompt):
        c = prompt.point_coords[0]
        return _disk(50, 50, c[1], c[0], 18)  # overlapping

    layer = SamSegmentationLayer(
        FakeBackend(big_disk), prompt_mode="pose", disjointify_masks=True
    )
    out = layer.masks_for_frame(img, [i0, i1])
    assert len(out) == 2
    # After disjointify the two masks do not overlap.
    assert not (out[0]["mask"] & out[1]["mask"]).any()


def test_sam_segmentation_slp_round_trip(tmp_path):
    skel, video, lf, img, track, inst = _labels_with_one_pose()

    def kpt_disk(image, prompt):
        c = prompt.point_coords.mean(0)
        return _disk(image.shape[0], image.shape[1], c[1], c[0], 12)

    layer = SamSegmentationLayer(FakeBackend(kpt_disk), prompt_mode="pose")
    frame_masks = layer.masks_for_frame(img, [inst])
    masks = Outputs(pred_masks=[frame_masks]).to_masks(0)
    out = sio.Labels(
        videos=[video],
        skeletons=[skel],
        labeled_frames=[
            sio.LabeledFrame(video=video, frame_idx=0, instances=[inst], masks=masks)
        ],
    )
    out.tracks = [track]
    path = tmp_path / "sam_masks.slp"
    out.save(path.as_posix())
    reloaded = sio.load_slp(path.as_posix())
    assert len(reloaded.labeled_frames) == 1
    rf = reloaded.labeled_frames[0]
    assert len(rf.masks) == 1
    rm = rf.masks[0]
    assert isinstance(rm, sio.PredictedSegmentationMask)
    assert np.isfinite(rm.score)
    # The track round-trips (referenced by name on reload).
    assert rm.track is not None
    assert rm.track.name == "m0"


# --------------------------------------------------------------------------- mask_backend required


def test_get_mask_backend_required_no_default():
    from sleap_nn.inference.sam import get_mask_backend

    with pytest.raises(ValueError):
        get_mask_backend(None)


def test_get_mask_backend_unknown_name():
    from sleap_nn.inference.sam import get_mask_backend

    with pytest.raises(ValueError):
        get_mask_backend("not-a-backend")


def test_get_mask_backend_sam3_not_yet():
    from sleap_nn.inference.sam import get_mask_backend

    with pytest.raises(NotImplementedError):
        get_mask_backend("sam3")


def test_run_predict_mask_backend_rejects_model_paths():
    from sleap_nn.inference.run import predict

    with pytest.raises(ValueError, match="do not also pass"):
        predict(
            "dummy.slp",
            mask_backend="sam",
            model_paths=["/tmp/some_model"],
            device="cpu",
        )
