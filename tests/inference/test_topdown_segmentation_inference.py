"""Inference tests for TOP-DOWN (crop-centered) instance segmentation (#622).

Two layers of testing:

* A *deterministic offset-placement unit test*: build the composed
  ``TopDownSegmentationLayer`` with a stubbed stage-2 backend returning a known
  crop mask at a known centroid, then assert the emitted ``sio`` mask — decoded
  through :func:`decode_mask_to_image_res` exactly as eval does — lands at the
  correct FULL-FRAME location, and that two crops at different centroids do not
  collide. This validates the DQ5 offset/scale contract end-to-end through the
  decode path (not just that the dict has the right numbers).

* A full ``train -> predict`` e2e exercising the
  ``loaders -> TopDownSegmentationLayer -> Predictor -> sio.Labels`` plumbing via
  the GT-centroid fallback path.
"""

import numpy as np
import torch

import sleap_io as sio

from sleap_nn.inference.layers.topdown_segmentation import (
    CenteredInstanceMaskLayer,
    TopDownSegmentationLayer,
)
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.segmentation_convert import decode_mask_to_image_res


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2


class _StubMaskLayer:
    """Stand-in for :class:`CenteredInstanceMaskLayer` that returns fixed masks.

    ``predict(crops)`` returns ``Outputs(crops=...)`` where ``crops`` is a
    pre-set ``(n_valid, 1, h, w)`` boolean-as-float tensor — one per crop, in
    the same row order ``_run_stage_2`` builds the crops (sorted by valid (b, i)).
    """

    def __init__(self, crop_masks, output_stride):
        self._crop_masks = crop_masks  # (n_valid, 1, h, w) float
        self.output_stride = output_stride
        self.use_gt_peaks = False

    def predict(self, crops):
        return Outputs(crops=self._crop_masks)


def _make_layer(crop_masks, crop_size, output_stride):
    """Build a ``TopDownSegmentationLayer`` via ``__new__`` with a stub stage 2."""
    layer = TopDownSegmentationLayer.__new__(TopDownSegmentationLayer)
    layer.centroid_layer = None  # not used by _run_stage_2
    layer.centered_instance_layer = _StubMaskLayer(crop_masks, output_stride)
    layer.crop_size = crop_size
    layer.centroid_nms = False
    layer.centroid_nms_threshold = 0.5
    layer.return_crops = False
    layer.mask_output = "mask"
    layer.polygon_epsilon = 0.01
    return layer


def test_topdown_seg_offset_placement_through_decode():
    """A crop mask at a known centroid decodes to the correct full-frame location.

    Builds a single 1-instance "frame": a centroid at image (cx, cy), a crop of
    size (ch, cw), and a known foreground disk drawn in the crop at output
    stride. The emitted sio mask, decoded via ``decode_mask_to_image_res`` (the
    eval path), must have its foreground centered at (cx, cy) in full-frame
    coordinates — NOT at the origin.
    """
    s = 2
    ch = cw = 64
    cx, cy = 200.0, 150.0  # image-space centroid (eff_scale = 1)

    # Crop-stride mask: a disk centered in the crop (the model "found" the
    # centered instance). Crop center at crop px (cw/2, ch/2) -> stride
    # (cw/2/s, ch/2/s).
    mh, mw = ch // s, cw // s
    crop_mask = _disk(mh, mw, mh // 2, mw // 2, 6).astype(np.float32)
    crop_masks = torch.from_numpy(crop_mask)[None, None]  # (1, 1, mh, mw)

    layer = _make_layer(crop_masks, crop_size=(ch, cw), output_stride=s)

    image = torch.zeros(1, 1, 512, 512)  # sized-space frame (eff_scale = 1)
    centroids = torch.tensor([[[cx, cy]]])  # (B=1, max_inst=1, 2)
    centroid_vals = torch.tensor([[0.77]])
    valid_mask = torch.tensor([[True]])
    out = layer._run_stage_2(
        image, centroids, centroid_vals, valid_mask, eff_scale=torch.tensor([1.0])
    )

    assert out.pred_masks is not None
    assert len(out.pred_masks) == 1  # one frame
    assert len(out.pred_masks[0]) == 1  # one instance
    inst = out.pred_masks[0][0]
    # scale = eff/stride per axis; offset = crop_topleft (eff=1).
    assert inst["scale"] == (1.0 / s, 1.0 / s)
    assert abs(inst["score"] - 0.77) < 1e-6
    # crop top-left in image px ~ (cx - cw/2, cy - ch/2).
    ox, oy = inst["offset"]
    assert abs(ox - (cx - cw / 2)) <= 1.0
    assert abs(oy - (cy - ch / 2)) <= 1.0

    # Decode through the eval path and check the foreground center-of-mass lands
    # at the image-space centroid (NOT at the origin).
    m = sio.PredictedSegmentationMask.from_numpy(
        inst["mask"], score=inst["score"], scale=inst["scale"], offset=inst["offset"]
    )
    decoded = decode_mask_to_image_res(m)
    ys, xs = np.nonzero(decoded)
    assert ys.size > 0
    com_y, com_x = ys.mean(), xs.mean()
    # Within a few px of the true centroid (stride/interp rounding).
    assert abs(com_x - cx) <= 4.0
    assert abs(com_y - cy) <= 4.0
    # And decisively NOT at the origin.
    assert com_x > cw / 4 and com_y > ch / 4


def test_topdown_seg_two_crops_do_not_collide():
    """Two crops at different centroids decode to two distinct full-frame blobs."""
    s = 2
    ch = cw = 64
    mh, mw = ch // s, cw // s
    # Two centroids far apart in a 512x512 frame.
    c0 = (120.0, 100.0)
    c1 = (400.0, 380.0)

    disk = _disk(mh, mw, mh // 2, mw // 2, 6).astype(np.float32)
    crop_masks = torch.from_numpy(disk)[None, None].repeat(2, 1, 1, 1)  # (2,1,mh,mw)

    layer = _make_layer(crop_masks, crop_size=(ch, cw), output_stride=s)

    image = torch.zeros(1, 1, 512, 512)
    centroids = torch.tensor([[list(c0), list(c1)]])  # (1, 2, 2)
    centroid_vals = torch.tensor([[0.9, 0.8]])
    valid_mask = torch.tensor([[True, True]])
    out = layer._run_stage_2(
        image, centroids, centroid_vals, valid_mask, eff_scale=torch.tensor([1.0])
    )

    assert len(out.pred_masks[0]) == 2
    coms = []
    for inst in out.pred_masks[0]:
        m = sio.PredictedSegmentationMask.from_numpy(
            inst["mask"],
            score=inst["score"],
            scale=inst["scale"],
            offset=inst["offset"],
        )
        decoded = decode_mask_to_image_res(m)
        ys, xs = np.nonzero(decoded)
        coms.append((xs.mean(), ys.mean()))

    # The two blobs land near their respective centroids and far from each other.
    (x0, y0), (x1, y1) = coms
    assert abs(x0 - c0[0]) <= 4 and abs(y0 - c0[1]) <= 4
    assert abs(x1 - c1[0]) <= 4 and abs(y1 - c1[1]) <= 4
    assert (abs(x0 - x1) + abs(y0 - y1)) > 100  # decisively non-colliding


def test_topdown_seg_empty_batch_emits_empty_frames():
    """No valid centroids -> one empty mask list per frame (to_labels-safe)."""
    layer = _make_layer(torch.zeros(0, 1, 8, 8), crop_size=(64, 64), output_stride=2)
    centroids = torch.full((2, 1, 2), float("nan"))
    centroid_vals = torch.zeros(2, 1)
    valid_mask = torch.zeros(2, 1, dtype=torch.bool)
    out = layer._run_stage_2(
        centroids,
        centroids,
        centroid_vals,
        valid_mask,
        eff_scale=torch.tensor([1.0, 1.0]),
    )
    assert out.pred_masks == [[], []]


def test_centered_instance_mask_layer_postprocess_thresholds():
    """The stage-2 mask layer sigmoids + thresholds logits into a bool mask."""
    from sleap_nn.inference.preprocess_info import PreprocInfo

    layer = CenteredInstanceMaskLayer.__new__(CenteredInstanceMaskLayer)
    layer.fg_threshold = 0.5
    layer.output_stride = 2
    layer._HEAD_OUTPUT_KEY = "SegmentationHead"
    layer._TORCH_OUTPUT_KEY = "output"

    # Logits: strongly positive in a 4x4 block, strongly negative elsewhere.
    logits = torch.full((1, 1, 8, 8), -10.0)
    logits[0, 0, 2:6, 2:6] = 10.0
    info = PreprocInfo(
        original_size=(16, 16),
        processed_size=(16, 16),
        eff_scale=torch.tensor([1.0]),
        input_scale=1.0,
        output_stride=2,
    )
    out = layer.postprocess({"output": logits}, info)
    assert out.crops.shape == (1, 1, 8, 8)
    mask = out.crops[0, 0].bool().numpy()
    assert mask[2:6, 2:6].all()
    assert not mask[0, 0]
    # Score is the mean foreground probability over the predicted mask (~1.0).
    assert float(out.instance_scores[0, 0]) > 0.99


# ---------------------------------------------------------------------------
# End-to-end: train a tiny top-down seg model, predict via GT-centroid fallback
# ---------------------------------------------------------------------------


def _topdown_seg_infer_config(seg_path, tmp_path):
    """One-epoch CPU training config for a top-down (crop) seg model."""
    from omegaconf import OmegaConf

    # Reuse the bottom-up seg trainer config and swap in the crop-seg head.
    from tests.inference.test_segmentation_inference import _seg_train_config

    cfg = _seg_train_config(seg_path, tmp_path, max_epochs=1)
    cfg.data_config.preprocessing.crop_size = 160
    cfg.model_config.head_configs = OmegaConf.create(
        {
            "single_instance": None,
            "centroid": None,
            "bottomup": None,
            "centered_instance": None,
            "multi_class_bottomup": None,
            "multi_class_topdown": None,
            "bottomup_segmentation": None,
            "centered_instance_segmentation": {
                "segmentation": {
                    "output_stride": 2,
                    "loss_weight": 1.0,
                    "anchor_part": None,
                },
            },
        }
    )
    cfg.trainer_config.run_name = "topdown_seg_infer"
    return cfg


def test_topdown_seg_train_predict_gt_centroid_wiring(minimal_instance_seg, tmp_path):
    """Full train -> load -> predict via the GT-centroid fallback (seg dir only).

    Exercises ``loaders -> TopDownSegmentationLayer -> Predictor -> sio.Labels``.
    With only the seg dir, the centroid stage reads GT user-instance centroids.
    The 1-epoch model is not expected to produce good masks; placement +
    plumbing are the point.
    """
    from sleap_nn.training.model_trainer import ModelTrainer
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.loaders import load_model_assets

    cfg = _topdown_seg_infer_config(minimal_instance_seg.as_posix(), tmp_path)
    ModelTrainer.get_model_trainer_from_config(cfg).train()
    run_dir = (tmp_path / "topdown_seg_infer").as_posix()

    # loaders detect the new model type and build a TopDownSegmentationLayer.
    assets, model_types = load_model_assets([run_dir], device="cpu")
    assert model_types == ["centered_instance_segmentation"]

    pred = Predictor.from_model_paths(
        [run_dir], peak_threshold=0.05, fg_threshold=0.3, device="cpu"
    )
    assert isinstance(pred.layer, TopDownSegmentationLayer)
    # The composed layer is recognized as a segmentation layer (gates
    # tracking / no-skeleton / mask-count).
    assert pred._is_segmentation_layer()
    # GT-centroid fallback => predicts only labeled frames.
    assert pred._needs_gt_instances()
    # fg_threshold threads through to the stage-2 mask layer.
    assert abs(pred.layer.centered_instance_layer.fg_threshold - 0.3) < 1e-9

    out = pred.predict(minimal_instance_seg.as_posix(), make_labels=True)
    assert isinstance(out, sio.Labels)
    # A segmentation model must emit masks ONLY — no phantom keypoint instances
    # (those would also break mask-tracking auto-detection). Regression guard.
    for lf in out:
        assert len(lf.instances) == 0
    for m in out.masks:
        assert isinstance(m, sio.PredictedSegmentationMask)
        assert np.isfinite(m.score)
        # Mask is placed in-frame (non-negative origin within the frame).
        ox, oy = m.offset
        assert ox >= -1 and oy >= -1

    # The output saves + reloads as a valid .slp.
    out_path = tmp_path / "topdown_seg_preds.slp"
    out.save(out_path.as_posix())
    sio.load_slp(out_path.as_posix())
