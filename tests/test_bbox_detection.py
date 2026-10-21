"""Tests for the bounding-box detection model types (``bbox_centernet``, ``bbox_ssd``).

Covers the config presets, ground-truth derivation through sleap-io, box geometry
through augmentation, both target encoders, the anchors/head layout contract, the
losses, the decoders (encode -> decode round trips), the COCO-style metrics, and a
train -> predict -> eval run per detection style.
"""

import math

import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf

from sleap_nn.config.get_config import get_head_configs
from sleap_nn.config.model_config import ssd_level_strides
from sleap_nn.data.bboxes import (
    CenterNetTargetEncoder,
    SSDTargetEncoder,
    add_bboxes_to_labels,
    auto_anchor_sizes,
    bboxes_to_corners,
    clip_bboxes,
    corners_to_bboxes,
    decode_boxes,
    encode_boxes,
    frame_bbox_arrays,
    gaussian_radius,
    labels_bbox_categories,
    make_anchors,
    match_anchors,
    resolve_bbox_source,
)
from sleap_nn.evaluation import compute_bbox_metrics
from sleap_nn.inference.bbox import BBoxDecoder, decode_centernet, decode_ssd


def _head_configs(model_type, **leaf_kwargs):
    hc = OmegaConf.structured(get_head_configs(model_type))
    leaf = hc[model_type]["center" if model_type == "bbox_centernet" else "ssd"]
    for k, v in leaf_kwargs.items():
        leaf[k] = v
    return hc


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_bbox_head_config_presets():
    """String and dict presets build the two detector head configs."""
    from sleap_nn.config.utils import get_model_type_from_cfg

    cn = get_head_configs("bbox_centernet")
    assert cn.bbox_centernet.center.output_stride == 4
    assert cn.bbox_centernet.size.loss_weight == 0.1
    assert cn.bbox_centernet.offset.loss_weight == 1.0

    ssd = get_head_configs({"bbox_ssd": {"ssd": {"num_levels": 2, "output_stride": 4}}})
    assert ssd.bbox_ssd.ssd.num_levels == 2
    assert ssd_level_strides(ssd.bbox_ssd.ssd) == [4, 8]
    # Tuple defaults land as lists in an OmegaConf config.
    s = OmegaConf.structured(ssd)
    assert list(s.bbox_ssd.ssd.aspect_ratios) == [0.5, 1.0, 2.0]

    for name, hc in (("bbox_centernet", cn), ("bbox_ssd", ssd)):
        cfg = OmegaConf.create(
            {"model_config": {"head_configs": OmegaConf.structured(hc)}}
        )
        assert get_model_type_from_cfg(cfg) == name


def _stride_cfg(model_type, max_stride, **leaf_kwargs):
    return OmegaConf.create(
        {
            "model_config": {
                "backbone_config": {
                    "unet": {"max_stride": max_stride, "output_stride": 1}
                },
                "head_configs": _head_configs(model_type, **leaf_kwargs),
            }
        }
    )


def test_check_output_strides_bbox():
    """SSD raises max_stride to its deepest level; CenterNet needs one stride."""
    from sleap_nn.config.utils import check_output_strides

    cfg = check_output_strides(
        _stride_cfg("bbox_ssd", 16, output_stride=8, num_levels=3)
    )
    assert cfg.model_config.backbone_config.unet.max_stride == 32
    assert cfg.model_config.backbone_config.unet.output_stride == 8

    cfg = _stride_cfg("bbox_centernet", 16)
    cfg.model_config.head_configs.bbox_centernet.size.output_stride = 2
    with pytest.raises(ValueError, match="share one output_stride"):
        check_output_strides(cfg)


def test_bbox_data_config_validates_source():
    """``bbox_source`` only accepts the known sources."""
    from sleap_nn.config.data_config import DataConfig

    assert DataConfig().bbox_source == "auto"
    with pytest.raises(ValueError):
        DataConfig(bbox_source="poses")


# --------------------------------------------------------------------------- #
# Ground truth from labels
# --------------------------------------------------------------------------- #
def test_add_bboxes_from_instances(minimal_instance):
    """Pose labels get one tight user box per instance (via sleap-io)."""
    labels = sio.load_slp(minimal_instance)
    assert resolve_bbox_source(labels, "auto") == "instances"
    assert add_bboxes_to_labels(labels, source="auto", padding=2.0) == "instances"
    lf = labels[0]
    assert len(lf.bboxes) == len(lf.user_instances) == 2
    for inst, box in zip(lf.user_instances, lf.bboxes):
        pts = inst.numpy()
        assert box.instance is inst
        np.testing.assert_allclose(
            box.xyxy, [*(pts.min(0) - 2.0), *(pts.max(0) + 2.0)], atol=1e-5
        )
    # Labels now carrying boxes resolve to them; re-deriving is a no-op.
    assert resolve_bbox_source(labels, "auto") == "bboxes"
    assert add_bboxes_to_labels(labels, source="auto") == "bboxes"
    assert len(labels[0].bboxes) == 2

    xyxy, classes = frame_bbox_arrays(lf, ["object"])
    assert xyxy.shape == (2, 4) and classes.tolist() == [0, 0]
    assert labels_bbox_categories([labels]) == ["object"]


def test_bbox_categories_and_masks(minimal_instance):
    """Box categories become classes; masks derive boxes from their extent."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    lf.bboxes = [
        sio.UserBoundingBox.from_xyxy(10, 10, 50, 40, category="mouse"),
        sio.UserBoundingBox.from_xyxy(60, 10, 90, 40, category="fly"),
    ]
    assert labels_bbox_categories([labels]) == ["mouse", "fly"]
    xyxy, classes = frame_bbox_arrays(lf, ["fly", "mouse"])
    assert classes.tolist() == [1, 0]

    mask = np.zeros((384, 384), dtype=bool)
    mask[100:150, 20:80] = True
    lf.masks = [sio.UserSegmentationMask.from_numpy(mask)]
    add_bboxes_to_labels(labels, source="masks")
    assert len(lf.bboxes) == 1
    x1, y1, x2, y2 = lf.bboxes[0].xyxy
    assert (x1, y1) == (20, 100) and x2 >= 79 and y2 >= 149


def test_corners_roundtrip_and_clip():
    """Corners re-box to the same box; clipping drops out-of-frame boxes."""
    xyxy = np.array([[10, 20, 50, 80], [-30, 0, 10, 40]], dtype=np.float32)
    corners = torch.from_numpy(bboxes_to_corners(xyxy))
    assert corners.shape == (2, 4, 2)
    np.testing.assert_allclose(corners_to_bboxes(corners).numpy(), xyxy)

    boxes = torch.tensor([[10.0, 20, 50, 80], [-30, 0, 10, 40], [200, 0, 300, 10]])
    clipped, keep = clip_bboxes(boxes, (100, 100))
    assert keep.tolist() == [True, True, False]  # 1/4 visible -> kept; outside -> no
    assert clipped[1, 0] == 0
    _, keep = clip_bboxes(boxes, (100, 100), min_visible_fraction=0.3)
    assert keep.tolist() == [True, False, False]


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def test_gaussian_radius_grows_with_box():
    assert 0 < gaussian_radius(10, 10) < gaussian_radius(40, 40)


def test_centernet_targets():
    """A peak of exactly 1 at each center cell plus size/offset at that cell."""
    enc = CenterNetTargetEncoder(num_classes=2, output_stride=4)
    boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0], [100.0, 40.0, 140.0, 100.0]])
    t = enc(boxes, torch.tensor([0, 1]), (128, 160))
    assert t["center_heatmap"].shape == (2, 32, 40)
    # Box 0: center (30, 40) px -> (7.5, 10) cells -> cell (7, 10).
    assert t["center_heatmap"][0, 10, 7] == 1.0
    assert t["center_heatmap"][1, 10, 7] < 1.0
    torch.testing.assert_close(t["size_target"][:, 10, 7], torch.tensor([10.0, 10.0]))
    torch.testing.assert_close(t["offset_target"][:, 10, 7], torch.tensor([0.5, 0.0]))
    assert t["reg_mask"].sum() == 2
    assert (t["center_heatmap"] == 1.0).sum() == 2

    # A fixed sigma gives a narrower peak than CenterNet's adaptive radius here.
    narrow = CenterNetTargetEncoder(1, 4, sigma=1.0)(
        boxes[:1], torch.tensor([0]), (128, 160)
    )
    assert (narrow["center_heatmap"] > 0.1).sum() < (t["center_heatmap"][0] > 0.1).sum()


def test_anchor_layout_matches_ssd_head():
    """Anchors line up one-to-one with the SSD head's flattened predictions."""
    from sleap_nn.architectures.model import Model
    from sleap_nn.config.model_config import UNetConfig

    hc = _head_configs(
        "bbox_ssd", output_stride=8, num_levels=3, anchor_sizes=[16, 32, 64]
    )
    enc = SSDTargetEncoder(
        1,
        ssd_level_strides(hc.bbox_ssd.ssd),
        [16, 32, 64],
        [1.0, 1.26],
        [0.5, 1.0, 2.0],
    )
    hc.bbox_ssd.ssd.anchor_scales = [1.0, 1.26]
    backbone = OmegaConf.structured(
        UNetConfig(filters=8, max_stride=32, output_stride=8)
    )
    model = Model("unet", backbone, hc.bbox_ssd, "bbox_ssd")
    out = model(torch.rand(1, 1, 64, 96))["SSDHead"]
    anchors = enc.anchors((64, 96))
    assert out.shape == (1, anchors.shape[0], 2 + 4)
    # First level, first cell, first anchor shape: centered at (4, 4) px.
    a0 = anchors[0]
    assert ((a0[0] + a0[2]) / 2, (a0[1] + a0[3]) / 2) == (4.0, 4.0)
    assert enc.anchors_per_location == 6


def test_box_coding_roundtrip():
    anchors = torch.tensor([[0.0, 0.0, 20.0, 20.0], [10.0, 10.0, 50.0, 30.0]])
    boxes = torch.tensor([[2.0, 3.0, 18.0, 25.0], [12.0, 8.0, 60.0, 33.0]])
    deltas = encode_boxes(boxes, anchors, [0.1, 0.1, 0.2, 0.2])
    torch.testing.assert_close(
        decode_boxes(deltas, anchors, [0.1, 0.1, 0.2, 0.2]), boxes
    )


def test_match_anchors_forces_best_anchor():
    """A box overlapping every anchor poorly is still matched to its best anchor."""
    anchors = torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 100.0, 100.0]])
    boxes = torch.tensor([[0.0, 0.0, 30.0, 30.0]])  # IoU 0.11 with anchor 0
    idx, state = match_anchors(boxes, anchors, 0.5, 0.4)
    assert state.tolist() == [1, 0]
    assert idx[0] == 0
    idx, state = match_anchors(torch.zeros((0, 4)), anchors)
    assert state.tolist() == [0, 0]


def test_ssd_targets():
    enc = SSDTargetEncoder(2, [8], [16.0], [1.0], [1.0])
    t = enc(torch.tensor([[0.0, 0.0, 16.0, 16.0]]), torch.tensor([1]), (32, 32))
    cls = t["ssd_cls_target"]
    assert cls.shape == (16,)
    assert (cls == 2).sum() >= 1  # class 1 -> logit index 2
    pos = cls == 2
    decoded = decode_boxes(
        t["ssd_box_target"][pos], enc.anchors((32, 32))[pos], enc.variances
    )
    torch.testing.assert_close(
        decoded, torch.tensor([[0.0, 0.0, 16.0, 16.0]]).expand_as(decoded)
    )


def test_auto_anchor_sizes():
    sizes = auto_anchor_sizes(np.linspace(40, 200, 100), 3, [8, 16, 32])
    assert len(sizes) == 3 and sizes[0] < sizes[1] < sizes[2]
    assert sizes[0] == pytest.approx(48.0, rel=0.05)
    assert auto_anchor_sizes(np.zeros(0), 2, [8, 16]) == [32.0, 64.0]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_type", ["bbox_centernet", "bbox_ssd"])
@pytest.mark.parametrize("augment", [False, True])
def test_bbox_dataset(minimal_instance, model_type, augment):
    """Samples carry padded GT boxes and the style's targets, also under aug."""
    from sleap_nn.data.bboxes import get_bbox_target_encoder
    from sleap_nn.data.custom_datasets import BBoxDataset

    labels = sio.load_slp(minimal_instance)
    add_bboxes_to_labels(labels)
    hc = _head_configs(model_type, categories=["object"])
    if model_type == "bbox_ssd":
        hc.bbox_ssd.ssd.anchor_sizes = [32.0, 64.0, 128.0]
    ds = BBoxDataset(
        labels=[labels],
        target_encoder=get_bbox_target_encoder(model_type, hc),
        categories=["object"],
        max_stride=32,
        geometric_aug=(
            {
                "rotation_min": -30.0,
                "rotation_max": 30.0,
                "affine_p": 1.0,
                "flip_p": 1.0,
            }
            if augment
            else None
        ),
        apply_aug=augment,
    )
    s = ds[0]
    assert s["image"].shape == (1, 1, 384, 384)
    assert s["bboxes"].shape == (2, 4) and s["num_instances"] == 2
    assert s["bbox_classes"].tolist() == [0, 0]
    boxes = s["bboxes"]
    assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()
    assert boxes.min() >= 0 and boxes.max() <= 383
    if not augment:
        gt = torch.from_numpy(frame_bbox_arrays(labels[0], ["object"])[0])
        torch.testing.assert_close(boxes, gt)
    if model_type == "bbox_centernet":
        assert s["center_heatmap"].shape == (1, 96, 96)
        assert s["reg_mask"].sum() == 2
    else:
        assert s["ssd_cls_target"].ndim == 1
        assert (s["ssd_cls_target"] > 0).sum() >= 2


# --------------------------------------------------------------------------- #
# Model + losses
# --------------------------------------------------------------------------- #
def test_centernet_model_heads():
    from sleap_nn.architectures.model import Model
    from sleap_nn.config.model_config import UNetConfig

    hc = _head_configs("bbox_centernet", categories=["a", "b"])
    backbone = OmegaConf.structured(
        UNetConfig(filters=8, max_stride=16, output_stride=4)
    )
    out = Model("unet", backbone, hc.bbox_centernet, "bbox_centernet")(
        torch.rand(2, 1, 64, 64)
    )
    assert out["BBoxCenterHead"].shape == (2, 2, 16, 16)
    assert out["BBoxSizeHead"].shape == (2, 2, 16, 16)
    assert out["BBoxOffsetHead"].shape == (2, 2, 16, 16)


def test_ssd_head_rejects_missing_strides():
    from sleap_nn.architectures.model import Model
    from sleap_nn.config.model_config import UNetConfig

    hc = _head_configs("bbox_ssd", output_stride=4, num_levels=4)
    backbone = OmegaConf.structured(
        UNetConfig(filters=8, max_stride=16, output_stride=4)
    )
    with pytest.raises(ValueError, match="missing"):
        Model("unet", backbone, hc.bbox_ssd, "bbox_ssd")


def test_bbox_losses():
    from sleap_nn.training.losses import (
        compute_centernet_focal_loss,
        compute_masked_l1,
        compute_ssd_loss,
    )

    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 3, 4] = 1.0
    perfect = torch.where(target > 0, torch.tensor(20.0), torch.tensor(-20.0))
    assert compute_centernet_focal_loss(perfect, target) < 1e-4
    assert compute_centernet_focal_loss(torch.zeros_like(target), target) > 0.1

    mask = torch.zeros(1, 1, 8, 8)
    mask[0, 0, 3, 4] = 1.0
    pred, gt = torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8)
    gt[0, :, 3, 4] = torch.tensor([2.0, 4.0])
    assert compute_masked_l1(pred, gt, mask) == pytest.approx(3.0, rel=1e-3)

    cls_t = torch.zeros(1, 10, dtype=torch.long)
    cls_t[0, 0] = 1
    cls_t[0, 9] = -1  # ignored
    raw = torch.zeros(1, 10, 2 + 4, requires_grad=True)
    loss, cls_loss, box_loss = compute_ssd_loss(raw, cls_t, torch.zeros(1, 10, 4))
    # 1 positive + 3 hard negatives, uniform logits: 4 * log(2) / 1 positive.
    assert cls_loss.item() == pytest.approx(4 * math.log(2), rel=1e-4)
    assert box_loss.item() == 0.0
    loss.backward()
    assert raw.grad[0, 9].abs().sum() == 0  # ignored anchors get no gradient


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
GT_BOXES = torch.tensor([[12.0, 20.0, 52.0, 68.0], [90.0, 30.0, 150.0, 70.0]])


def test_decode_centernet_inverts_targets():
    """Decoding perfect targets recovers the boxes (to sub-cell precision)."""
    t = CenterNetTargetEncoder(1, 4)(GT_BOXES, torch.tensor([0, 0]), (128, 160))
    boxes, scores, classes = decode_centernet(
        t["center_heatmap"][None],
        t["size_target"][None],
        t["offset_target"][None],
        output_stride=4,
        score_threshold=0.5,
        max_detections=10,
    )
    valid = torch.isfinite(scores[0])
    assert valid.sum() == 2
    got = boxes[0][valid]
    got = got[got[:, 0].argsort()]
    torch.testing.assert_close(got, GT_BOXES, atol=1e-4, rtol=0)
    assert classes[0][valid].tolist() == [0, 0]
    assert boxes.shape == (1, 10, 4) and torch.isnan(boxes[0][~valid]).all()


def test_decode_ssd_inverts_targets():
    """Perfect SSD logits/deltas decode (after NMS) to the GT boxes."""
    enc = SSDTargetEncoder(1, [8, 16], [32.0, 64.0], [1.0, 1.26], [0.5, 1.0, 2.0])
    t = enc(GT_BOXES, torch.tensor([0, 0]), (128, 160))
    cls = t["ssd_cls_target"]
    logits = torch.full((cls.shape[0], 2), -10.0)
    logits[cls <= 0, 0] = 10.0
    logits[cls == 1, 1] = 10.0
    raw = torch.cat([logits, t["ssd_box_target"]], dim=-1)[None]
    boxes, scores, _ = decode_ssd(
        raw,
        enc.anchors((128, 160)),
        enc.variances,
        score_threshold=0.5,
        iou_threshold=0.45,
    )
    valid = torch.isfinite(scores[0])
    assert valid.sum() == 2
    got = boxes[0][valid]
    got = got[got[:, 0].argsort()]
    torch.testing.assert_close(got, GT_BOXES, atol=1e-3, rtol=0)


def test_bbox_decoder_from_config():
    hc = _head_configs("bbox_ssd", categories=["a"], anchor_sizes=[32.0, 64.0, 128.0])
    dec = BBoxDecoder.from_head_configs("bbox_ssd", hc, score_threshold=0.3)
    assert dec.iou_threshold == 0.45 and dec.categories == ["a"]
    hc = _head_configs("bbox_centernet")
    dec = BBoxDecoder.from_head_configs("bbox_centernet", hc)
    assert dec.iou_threshold is None and dec.categories == ["object"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _frame(gt, pred, scores, gt_cls=None, pred_cls=None):
    gt, pred = np.asarray(gt, float).reshape(-1, 4), np.asarray(pred, float).reshape(
        -1, 4
    )
    return {
        "gt_bboxes": gt,
        "gt_classes": np.zeros(len(gt), int) if gt_cls is None else np.asarray(gt_cls),
        "pred_bboxes": pred,
        "pred_scores": np.asarray(scores, float),
        "pred_classes": (
            np.zeros(len(pred), int) if pred_cls is None else np.asarray(pred_cls)
        ),
    }


def test_compute_bbox_metrics_perfect_and_misses():
    box = [0, 0, 10, 10]
    perfect = compute_bbox_metrics([_frame([box], [box], [0.9])])
    assert perfect["bbox_mAP"] == pytest.approx(1.0)
    assert perfect["bbox_f1"] == pytest.approx(1.0)
    assert perfect["bbox_mean_iou"] == pytest.approx(1.0)

    # One hit + one miss + one false positive (low score, ranked last).
    m = compute_bbox_metrics(
        [
            _frame([box, [20, 20, 30, 30]], [box, [50, 50, 60, 60]], [0.9, 0.1]),
        ]
    )
    assert m["bbox_n_tp"] == 1 and m["bbox_n_fn"] == 1
    assert m["bbox_n_fp"] == 0  # best-F1 threshold stops before the FP
    assert m["bbox_recall"] == pytest.approx(0.5)
    assert m["bbox_AP50"] == pytest.approx(0.5, abs=0.01)

    # A shifted box: IoU 0.68 counts at 0.5 but not at 0.75.
    shifted = compute_bbox_metrics([_frame([box], [[1, 1, 11, 11]], [0.9])])
    assert shifted["bbox_AP50"] == pytest.approx(1.0)
    assert shifted["bbox_AP75"] == pytest.approx(0.0)

    # A detection of the wrong class is a FP and a FN.
    wrong = compute_bbox_metrics(
        [_frame([box], [box], [0.9], gt_cls=[0], pred_cls=[1])], num_classes=2
    )
    assert wrong["bbox_AP50"] == pytest.approx(0.0)
    empty = compute_bbox_metrics([_frame([box], np.zeros((0, 4)), [])])
    assert empty["bbox_recall"] == 0.0 and empty["bbox_n_fn"] == 1


def test_run_evaluation_bbox(minimal_instance, tmp_path):
    """``match_method="bbox"`` scores predicted boxes against pose-derived GT."""
    from sleap_nn.evaluation import run_evaluation

    gt = sio.load_slp(minimal_instance)
    lf = gt[0]
    pred_boxes = []
    for inst in lf.user_instances:
        x1, y1, x2, y2 = inst.to_bbox().xyxy
        pred_boxes.append(
            sio.PredictedBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, score=0.8)
        )
    pr = sio.Labels(
        videos=gt.videos,
        labeled_frames=[
            sio.LabeledFrame(video=lf.video, frame_idx=lf.frame_idx, bboxes=pred_boxes)
        ],
    )
    pred_path = tmp_path / "pred.slp"
    pr.save(pred_path.as_posix())
    m = run_evaluation(
        minimal_instance.as_posix(),
        pred_path.as_posix(),
        match_method="auto",
        save_metrics=(tmp_path / "metrics.npz").as_posix(),
    )
    assert m["bbox_mAP"] == pytest.approx(1.0)
    assert (tmp_path / "metrics.json").exists()


def test_outputs_to_bboxes_categories():
    from sleap_nn.inference.outputs import Outputs

    out = Outputs(
        pred_bboxes=torch.tensor([[[0.0, 0, 10, 10], [float("nan")] * 4]]),
        pred_bbox_scores=torch.tensor([[0.7, float("nan")]]),
        pred_bbox_classes=torch.tensor([[1, -1]]),
    )
    boxes = out.to_bboxes(0, categories=["mouse", "fly"])
    assert len(boxes) == 1 and boxes[0].category.name == "fly"
    assert boxes[0].score == pytest.approx(0.7)
    assert out.to_bboxes(0, categories=["object"])[0].category is None
    labels = out.to_labels(skeleton=None, bbox_categories=["mouse", "fly"])
    assert len(labels[0].bboxes) == 1 and labels.skeletons == []


# --------------------------------------------------------------------------- #
# Train -> predict -> eval
# --------------------------------------------------------------------------- #
def _bbox_train_config(config, model_type, tmp_path):
    cfg = config.copy()
    cfg.data_config.preprocessing.crop_size = None
    cfg.data_config.augmentation_config.geometric["flip_p"] = 0.5
    cfg.model_config.backbone_config.unet.max_stride = 32
    cfg.model_config.backbone_config.unet.output_stride = 4
    cfg.model_config.head_configs = OmegaConf.structured(get_head_configs(model_type))
    if model_type == "bbox_ssd":
        cfg.model_config.head_configs.bbox_ssd.ssd.output_stride = 8
    cfg.trainer_config.trainer_accelerator = "cpu"
    cfg.trainer_config.max_epochs = 2
    cfg.trainer_config.min_train_steps_per_epoch = 2
    cfg.trainer_config.save_ckpt = True
    cfg.trainer_config.ckpt_dir = str(tmp_path)
    cfg.trainer_config.run_name = model_type
    cfg.trainer_config.visualize_preds_during_training = True
    cfg.trainer_config.eval = {"enabled": True, "frequency": 1}
    return cfg


@pytest.mark.parametrize("model_type", ["bbox_centernet", "bbox_ssd"])
def test_bbox_train_predict_eval(config, minimal_instance, tmp_path, model_type):
    """Train on pose labels (boxes derived), reload, predict boxes, evaluate."""
    from sleap_nn.evaluation import run_evaluation
    from sleap_nn.inference.layers.bbox import BBoxLayer
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.training.model_trainer import ModelTrainer

    cfg = _bbox_train_config(config, model_type, tmp_path)
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    run_dir = tmp_path / model_type
    saved = OmegaConf.load(run_dir / "training_config.yaml")
    leaf = (
        saved.model_config.head_configs.bbox_centernet.center
        if model_type == "bbox_centernet"
        else saved.model_config.head_configs.bbox_ssd.ssd
    )
    assert list(leaf.categories) == ["object"]
    if model_type == "bbox_ssd":
        assert len(leaf.anchor_sizes) == leaf.num_levels  # auto-sized + saved
    log = (run_dir / "training_log.csv").read_text()
    assert "eval/val/bbox_mAP" in log

    pred = Predictor.from_model_paths(
        [run_dir.as_posix()], device="cpu", peak_threshold=0.0
    )
    assert isinstance(pred.layer, BBoxLayer)
    labels = pred.predict(minimal_instance.as_posix())
    assert len(labels) == 1
    boxes = labels[0].bboxes
    assert boxes and all(isinstance(b, sio.PredictedBoundingBox) for b in boxes)
    assert len(boxes) <= 100
    assert all(0 <= b.x1 < b.x2 <= 384 and 0 <= b.y1 < b.y2 <= 384 for b in boxes)
    out = tmp_path / "pred.slp"
    labels.save(out.as_posix())
    reloaded = sio.load_slp(out.as_posix())
    assert len(reloaded[0].bboxes) == len(boxes)

    m = run_evaluation(minimal_instance.as_posix(), out.as_posix(), match_method="bbox")
    assert m is not None and 0.0 <= m["bbox_mAP"] <= 1.0 and m["n_gt"] == 2

    # The NMS knob reaches the decoder; tracking is refused (boxes only).
    nms = Predictor.from_model_paths(
        [run_dir.as_posix()], device="cpu", bbox_nms_iou=0.3
    )
    assert nms.layer.decoder.iou_threshold == pytest.approx(0.3)
    default_iou = 0.45 if model_type == "bbox_ssd" else None
    assert pred.layer.decoder.iou_threshold == default_iou
    from sleap_nn.inference.tracking import TrackerConfig

    nms.tracker_config = TrackerConfig()
    with pytest.raises(ValueError, match="not supported for bounding-box"):
        nms.predict(minimal_instance.as_posix())

    _check_onnx_export_parity(run_dir, minimal_instance, labels, tmp_path)


def _check_onnx_export_parity(run_dir, slp_path, torch_labels, tmp_path):
    """ONNX export decodes (with the shared BBoxDecoder) to the PyTorch boxes."""
    pytest.importorskip("onnxruntime")
    from click.testing import CliRunner

    from sleap_nn.evaluation import bbox_iou_matrix
    from sleap_nn.export.cli import export
    from sleap_nn.inference.layers.exported import ExportedBBoxLayer
    from sleap_nn.inference.predictor import Predictor

    out_dir = tmp_path / "exported"
    result = CliRunner().invoke(
        export, [str(run_dir), "-o", str(out_dir), "--peak-threshold", "0.0"]
    )
    assert result.exit_code == 0, result.output
    meta = __import__("json").loads((out_dir / "export_metadata.json").read_text())
    assert meta["class_names"] == ["object"]

    pred = Predictor.from_export_dir(out_dir, runtime="onnx", device="cpu")
    assert isinstance(pred.layer, ExportedBBoxLayer)
    onnx_labels = pred.predict(slp_path.as_posix())
    a = np.array([b.xyxy for b in torch_labels[0].bboxes]).reshape(-1, 4)
    b = np.array([b.xyxy for b in onnx_labels[0].bboxes]).reshape(-1, 4)
    assert len(a) == len(b)
    if len(a):
        # Each exported box has a near-identical native box.
        assert (bbox_iou_matrix(b, a).max(axis=1) > 0.99).mean() >= 0.9


def _synthetic_two_class_labels(tmp_path, n_frames=6):
    """Image-sequence labels: a bright square ("a") and a dim one ("b") per frame,
    plus one user-confirmed negative frame with nothing in it."""
    from PIL import Image

    rng = np.random.default_rng(0)
    paths, lfs = [], []
    for i in range(n_frames + 1):
        img = rng.integers(0, 20, size=(128, 128), dtype=np.uint8)
        boxes = []
        if i < n_frames:
            for name, value in (("a", 250), ("b", 120)):
                x, y = rng.integers(5, 90, size=2)
                s = int(rng.integers(16, 30))
                img[y : y + s, x : x + s] = value
                boxes.append(
                    sio.UserBoundingBox.from_xyxy(
                        float(x),
                        float(y),
                        float(x + s - 1),
                        float(y + s - 1),
                        category=name,
                    )
                )
        path = tmp_path / f"frame_{i:02d}.png"
        Image.fromarray(img).save(path)
        paths.append(path.as_posix())
        lfs.append((i, boxes))
    video = sio.Video.from_filename(paths)
    frames = []
    for i, boxes in lfs:
        lf = sio.LabeledFrame(video=video, frame_idx=i, bboxes=boxes)
        if not boxes:
            lf.is_negative = True
        frames.append(lf)
    labels = sio.Labels(videos=[video], labeled_frames=frames)
    out = tmp_path / "boxes.slp"
    labels.save(out.as_posix())
    return out


@pytest.mark.parametrize("model_type", ["bbox_centernet", "bbox_ssd"])
def test_bbox_multiclass_and_negatives(config, tmp_path, model_type):
    """Categories become classes end to end; negative frames are background samples."""
    from sleap_nn.data.custom_datasets import get_train_val_datasets
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.training.model_trainer import ModelTrainer

    path = _synthetic_two_class_labels(tmp_path)
    cfg = _bbox_train_config(config, model_type, tmp_path)
    cfg.data_config.train_labels_path = [path.as_posix()]
    cfg.data_config.val_labels_path = [path.as_posix()]
    cfg.data_config.use_negative_frames = True
    cfg.data_config.augmentation_config = None
    cfg.trainer_config.max_epochs = 1
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()

    saved = OmegaConf.load(tmp_path / model_type / "training_config.yaml")
    leaf = (
        saved.model_config.head_configs.bbox_centernet.center
        if model_type == "bbox_centernet"
        else saved.model_config.head_configs.bbox_ssd.ssd
    )
    assert list(leaf.categories) == ["a", "b"]

    train_ds, _ = get_train_val_datasets(
        [sio.load_slp(path.as_posix())], [sio.load_slp(path.as_posix())], saved
    )
    assert len(train_ds) == 7  # 6 positives + 1 negative
    neg = train_ds[6]
    assert neg["num_instances"] == 0 and neg["is_negative"]
    pos = train_ds[0]
    assert sorted(pos["bbox_classes"][: pos["num_instances"]].tolist()) == [0, 1]
    if model_type == "bbox_centernet":
        assert pos["center_heatmap"].shape[0] == 2
        assert neg["center_heatmap"].sum() == 0
    else:
        assert (neg["ssd_cls_target"] > 0).sum() == 0
        assert set(pos["ssd_cls_target"].unique().tolist()) >= {0, 1, 2}

    pred = Predictor.from_model_paths(
        [(tmp_path / model_type).as_posix()], device="cpu", peak_threshold=0.0
    )
    assert pred.layer.categories == ["a", "b"]
    # Raw outputs: an untrained size head may regress ~0-size boxes, which the
    # decoder drops after clipping, so check class ids before box filtering.
    outputs = pred.predict(path.as_posix(), make_labels=False)
    classes = torch.cat([o.pred_bbox_classes.flatten() for o in outputs])
    scores = torch.cat([o.pred_bbox_scores.flatten() for o in outputs])
    assert set(classes[torch.isfinite(scores)].tolist()) <= {0, 1}


@pytest.mark.parametrize("n_classes", [1, 2])
def test_centernet_module_loss_is_per_sample(n_classes):
    """Perfect logits for a batch of DIFFERENT targets give ~zero losses.

    Guards against broadcasting a (B, h, w) target against (B, 1, h, w) logits,
    which scores every image against every other image's heatmap.
    """
    from sleap_nn.training.lightning_modules import BBoxCenterNetLightningModule

    cats = ["a", "b"][:n_classes]
    hc = _head_configs("bbox_centernet", categories=cats)
    module = BBoxCenterNetLightningModule(
        model_type="bbox_centernet",
        backbone_type="unet",
        backbone_config={"unet": {"filters": 8, "max_stride": 16, "output_stride": 4}},
        head_configs=hc,
    )
    enc = CenterNetTargetEncoder(n_classes, 4)
    targets = [
        enc(torch.tensor([[8.0, 8.0, 30.0, 30.0]]), torch.tensor([0]), (64, 64)),
        enc(
            torch.tensor([[34.0, 30.0, 60.0, 62.0]]),
            torch.tensor([n_classes - 1]),
            (64, 64),
        ),
    ]
    batch = {k: torch.stack([t[k] for t in targets]) for k in targets[0]}
    logits = torch.where(batch["center_heatmap"] == 1.0, 20.0, -20.0)
    raw = {
        "BBoxCenterHead": logits,
        "BBoxSizeHead": batch["size_target"],
        "BBoxOffsetHead": batch["offset_target"],
    }
    loss, terms = module._compute_losses(raw, batch)
    assert terms["center_loss"] < 1e-4
    assert loss < 1e-3
