"""Unit tests for the SAM mask layers (CPU; a fake backend, no real SAM).

Covers:
* :class:`FindInstanceMaskSAM` ``.predict(crops)`` contract (shapes ``(N,1,h,w)``
  / ``(N,1)``), and its parity packaging through ``TopDownSegmentationLayer.
  _run_stage_2`` -> ``decode_mask_to_image_res`` (the crop-center seam lands at
  the centroid, byte-identical to the trained-model seg layer).
* :class:`SamSegmentationLayer` full-frame placement + ``instance=``/``track=``
  population + SLP round-trip.
* ``mask_backend`` is explicit / required (no default).
"""

import numpy as np
import pytest
import torch

import sleap_io as sio

from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.sam.mask_layer import (
    FindInstanceMaskSAM,
    SamSegmentationLayer,
    place_full,
)
from sleap_nn.inference.segmentation_convert import decode_mask_to_image_res


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


# --------------------------------------------------------------------------- helpers


def test_place_full_inverse_of_crop_extract():
    crop = _disk(20, 20, 10, 10, 5)
    full = place_full(crop, (30, 40), (100, 100))
    ys, xs = np.nonzero(full)
    # Crop center (10,10) at tl (30,40) -> image (40, 50).
    assert abs(xs.mean() - 40.0) <= 1.0
    assert abs(ys.mean() - 50.0) <= 1.0


def test_place_full_clips_off_frame():
    # A 20x20 crop at tl (-5, -5) into a 30x30 frame: the bottom-right 15x15 of
    # the crop survives at the frame origin; the rest is clipped off-frame.
    crop = np.ones((20, 20), bool)
    full = place_full(crop, (-5, -5), (30, 30))
    assert full[:15, :15].all()
    assert full.sum() == 15 * 15
    # Nothing outside the placed 15x15 block.
    assert not full[15:, :].any()
    assert not full[:, 15:].any()


# --------------------------------------------------------------------------- FindInstanceMaskSAM


def test_find_instance_mask_sam_predict_contract():
    h = w = 64

    def center_disk(image, prompt):
        # The prompt is the crop center; return a disk there.
        cx, cy = prompt.point_coords[0]
        return _disk(h, w, cy, cx, 8)

    layer = FindInstanceMaskSAM(FakeBackend(center_disk))
    crops = torch.zeros(3, 1, h, w)
    out = layer.predict(crops)
    assert out.crops.shape == (3, 1, h, w)
    assert out.instance_scores.shape == (3, 1)
    # Masks are {0,1} float; non-empty (the center disk).
    assert out.crops.max() == 1.0 and out.crops.min() == 0.0
    assert out.crops[0, 0].sum() > 0


def test_find_instance_mask_sam_empty_crops():
    layer = FindInstanceMaskSAM(FakeBackend(lambda i, p: np.zeros((8, 8), bool)))
    out = layer.predict(torch.zeros(0, 1, 8, 8))
    assert out.crops.shape == (0, 1, 8, 8)
    assert out.instance_scores.shape == (0, 1)


def test_find_instance_mask_sam_rejects_wrong_ndim():
    layer = FindInstanceMaskSAM(FakeBackend(lambda i, p: np.zeros((8, 8), bool)))
    with pytest.raises(ValueError):
        layer.predict(torch.zeros(8, 8))


def _make_topdown_with_sam(backend, crop_size):
    """Build a TopDownSegmentationLayer with a FindInstanceMaskSAM stage 2."""
    from sleap_nn.inference.layers.topdown_segmentation import TopDownSegmentationLayer

    layer = TopDownSegmentationLayer.__new__(TopDownSegmentationLayer)
    layer.centroid_layer = None
    layer.centered_instance_layer = FindInstanceMaskSAM(backend)
    layer.crop_size = crop_size
    layer.centroid_nms = False
    layer.centroid_nms_threshold = 0.5
    layer.return_crops = False
    layer.mask_output = "mask"
    layer.polygon_epsilon = 0.01
    return layer


def test_sam_crop_center_seam_lands_at_centroid_through_decode():
    """The crop-center SAM mask packs to the centroid (top-down seam parity).

    Mirrors ``test_topdown_seg_offset_placement_through_decode`` but with the SAM
    stage 2 — the inherited ``_run_stage_2`` offset/scale contract must place the
    crop-center mask at the full-frame centroid, NOT at the origin.
    """
    ch = cw = 64
    cx, cy = 200.0, 150.0

    def center_disk(image, prompt):
        px, py = prompt.point_coords[0]
        return _disk(image.shape[0], image.shape[1], py, px, 6)

    layer = _make_topdown_with_sam(FakeBackend(center_disk, score=0.77), (ch, cw))
    image = torch.zeros(1, 1, 512, 512)
    centroids = torch.tensor([[[cx, cy]]])
    out = layer._run_stage_2(
        image,
        centroids,
        torch.tensor([[0.5]]),
        torch.tensor([[True]]),
        eff_scale=torch.tensor([1.0]),
    )
    inst = out.pred_masks[0][0]
    # SAM emits at crop-pixel res -> output_stride=1, input_scale=1 -> scale=(1,1).
    assert inst["scale"] == (1.0, 1.0)
    # Score is the backend's raw SAM score (not the centroid confidence).
    assert abs(inst["score"] - 0.77) < 1e-6
    # offset = floored crop top-left ~ (cx - cw/2, cy - ch/2).
    ox, oy = inst["offset"]
    assert abs(ox - (cx - cw / 2)) <= 1.0
    assert abs(oy - (cy - ch / 2)) <= 1.0

    m = sio.PredictedSegmentationMask.from_numpy(
        inst["mask"], score=inst["score"], scale=inst["scale"], offset=inst["offset"]
    )
    decoded = decode_mask_to_image_res(m)
    ys, xs = np.nonzero(decoded)
    assert xs.size > 0
    assert abs(xs.mean() - cx) <= 4.0 and abs(ys.mean() - cy) <= 4.0
    # Decisively not at the origin.
    assert xs.mean() > cw / 4 and ys.mean() > ch / 4


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


def test_sam_segmentation_layer_rejects_crop_center_mode():
    with pytest.raises(ValueError):
        SamSegmentationLayer(object(), prompt_mode="crop_center")


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
