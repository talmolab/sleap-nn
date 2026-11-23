"""The experimental keypoint branch of ``bbox_detr``: boxes and poses in one pass.

Covers the data path (points stay with their boxes through every augmentation),
the keypoint loss on synthetic matches, a tiny CPU training run that saves, loads
strictly and predicts instances through the inference layer and ``Predictor``,
the refusals (export, box distillation) and that a plain ``bbox_detr`` is
unchanged (parameters, random draws, saved config).
"""

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf

from sleap_nn.architectures.detr.model import BBoxDetrHead
from sleap_nn.data.bboxes import (
    SetTargetEncoder,
    add_bboxes_to_labels,
    collate_set_targets,
)
from sleap_nn.data.detr_augmentation import (
    detr_train_transform,
    hflip,
    mirror_order,
    random_resize_crop,
    resize_frame,
    rot90,
    vflip,
    zoom_out,
)
from sleap_nn.training.detr import SetCriterion
from sleap_nn.training.detr.matcher import HungarianMatcher

from tests.test_bbox_detr_training import TINY_DETR, _detr_train_config


# --------------------------------------------------------------------------- #
# Augmentation: points move with the pixels
# --------------------------------------------------------------------------- #
def _blob_image(hw, points, radius=2):
    """``(1, K, H, W)`` float image: channel ``k`` has a square blob at point ``k``."""
    h, w = hw
    img = torch.zeros(1, len(points), h, w)
    for k, (x, y) in enumerate(points):
        xi, yi = int(round(x)), int(round(y))
        img[
            0,
            k,
            max(yi - radius, 0) : yi + radius + 1,
            max(xi - radius, 0) : xi + radius + 1,
        ] = 255.0
    return img


def _centroid(img, channel, near=None, window=6):
    """Intensity centroid ``(x, y)`` of one channel (around ``near`` if given)."""
    ch = img[0, channel].float()
    if near is not None:
        x, y = (int(round(float(v))) for v in near)
        mask = torch.zeros_like(ch)
        mask[
            max(y - window, 0) : y + window + 1, max(x - window, 0) : x + window + 1
        ] = 1
        ch = ch * mask
    total = ch.sum()
    if total <= 0:
        return torch.full((2,), float("nan"))  # no blob there
    ys, xs = torch.meshgrid(
        torch.arange(ch.shape[0], dtype=torch.float32),
        torch.arange(ch.shape[1], dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([(xs * ch).sum() / total, (ys * ch).sum() / total])


POINTS = [(20.0, 30.0), (52.0, 34.0), (40.0, 60.0)]


def _points_tensor(points=POINTS):
    pts = torch.tensor(points, dtype=torch.float32)[None]  # (1, K, 2)
    weight = torch.arange(1, len(points) + 1, dtype=torch.float32)[None, :, None]
    return torch.cat([pts, weight], -1)  # (1, K, 3): x, y, weight


def _assert_on_blobs(img, points, order=None, tol=1.0):
    order = order or list(range(points.shape[1]))
    for k in range(points.shape[1]):
        xy = points[0, k, :2]
        if not torch.isfinite(xy).all():
            continue
        c = _centroid(img, order[k], near=xy)
        assert torch.allclose(c, xy, atol=tol), (k, c, xy)


def test_flips_and_rotations_move_points_exactly_and_swap_pairs():
    hw = (80, 72)
    img = _blob_image(hw, POINTS, radius=0)
    pts = _points_tensor()
    boxes = torch.tensor([[10.0, 20.0, 60.0, 70.0]])
    order = mirror_order(3, [(0, 1)])
    assert order == [1, 0, 2]
    for fn in (hflip, vflip):
        out, b, p = fn(img, boxes, pts, order)
        # Exact: one bright pixel per channel, at the swapped node's point.
        _assert_on_blobs(out, p, order=order, tol=1e-5)
        assert torch.equal(p[..., 2], pts[..., 2][:, order])  # weights follow
        plain = fn(img, boxes)
        assert len(plain) == 2 and torch.equal(plain[1], b)
    for clockwise in (True, False):
        out, b, p = rot90(img, boxes, clockwise, pts)
        _assert_on_blobs(out, p, tol=1e-5)
        assert torch.equal(p[..., 2], pts[..., 2])


@pytest.mark.parametrize("mode", ["squash", "letterbox"])
def test_resize_crop_zoom_move_points(mode):
    torch.manual_seed(3)
    hw = (90, 80)
    img = _blob_image(hw, POINTS)
    pts = _points_tensor()
    boxes = torch.tensor([[10.0, 20.0, 60.0, 70.0]])

    out, b, _, p = resize_frame(img, boxes, (64, 64), mode, pts)
    _assert_on_blobs(out, p, tol=1.0)
    out2, _, p2 = zoom_out(out, b, 0.5, (0.3, 0.7), p)
    _assert_on_blobs(out2, p2, tol=1.0)

    torch.manual_seed(0)
    crop, cb = random_resize_crop(img, boxes, (60,), (40, 50))
    torch.manual_seed(0)
    crop2, cb2, cp = random_resize_crop(img, boxes, (60,), (40, 50), pts)
    assert torch.equal(crop, crop2) and torch.equal(cb, cb2)  # same draws
    for k in range(3):
        xy = cp[0, k, :2]
        inside = (xy >= -0.5).all() and xy[0] <= crop.shape[-1] - 0.5
        inside = inside and xy[1] <= crop.shape[-2] - 0.5
        assert bool(torch.isfinite(xy).all()) == bool(inside)
    _assert_on_blobs(crop2, cp, tol=1.5)


def test_resize_drops_points_off_the_frame():
    pts = torch.tensor([[[5.0, 5.0, 1.0], [50.0, 5.0, 1.0], [-3.0, 5.0, 1.0]]])
    _, _, _, p = resize_frame(
        torch.zeros(1, 1, 20, 20), torch.zeros(0, 4), (40, 40), "letterbox", pts
    )
    assert torch.isfinite(p[0, 0, :2]).all()
    assert torch.isnan(p[0, 1:, :2]).all() and (p[0, :, 2] == 1).all()


def test_detr_train_transform_same_draws_and_points_on_blobs():
    hw = (90, 120)
    img = _blob_image(hw, POINTS)
    pts = _points_tensor()
    boxes = torch.tensor([[10.0, 20.0, 60.0, 70.0]])
    kwargs = dict(
        out_hw=(64, 64),
        mode="letterbox",
        crop_p=0.5,
        shortest_sides=(80, 90),
        crop_range=(60, 80),
        flip_p=0.5,
        vflip_p=0.5,
        rot90_p=0.5,
        zoom_out_p=0.5,
        zoom_out_range=(0.5, 1.0),
    )
    for seed in range(12):
        torch.manual_seed(seed)
        image, b = detr_train_transform(img, boxes, **kwargs)
        torch.manual_seed(seed)
        image2, b2, p = detr_train_transform(
            img, boxes, points=pts, flip_order=[0, 1, 2], **kwargs
        )
        assert torch.equal(image, image2) and torch.equal(b, b2)
        _assert_on_blobs(image2, p, tol=1.5)


# --------------------------------------------------------------------------- #
# Dataset: points stay with their boxes through the whole training chain
# --------------------------------------------------------------------------- #
def _blob_labels(tmp_path, n_frames=3, with_box_only=False):
    """Labels on PNG frames: two animals of 4 nodes, each node a blob.

    Node ``k``'s blob is in channel ``CHANNELS[k]`` (``T`` and ``H`` share channel 2,
    far apart). Nodes ``L`` and ``R`` are a symmetric pair, so a mirror flip swaps
    them: after it, node ``L``'s point lies on channel 1's blob.
    """
    import imageio.v3 as iio

    skeleton = sio.Skeleton(list(NAMES), edges=[("L", "R")], symmetries=[("L", "R")])
    rng = np.random.default_rng(0)
    frames, paths = [], []
    for f in range(n_frames):
        img = np.zeros((96, 128, 3), np.uint8)
        instances = []
        for a in range(2):
            cx, cy = 30 + 60 * a + rng.integers(-5, 5), 48 + rng.integers(-10, 10)
            pts = np.array(
                [[cx - 10, cy - 6], [cx + 10, cy - 6], [cx, cy + 12], [cx, cy - 16]],
                float,
            )
            for k, (x, y) in enumerate(pts):
                img[int(y) - 2 : int(y) + 3, int(x) - 2 : int(x) + 3, CHANNELS[k]] = 255
            instances.append(sio.Instance.from_numpy(pts, skeleton=skeleton))
        path = tmp_path / f"frame_{f}.png"
        iio.imwrite(path, img)
        paths.append(path.as_posix())
        frames.append(instances)
    video = sio.Video.from_filename(paths)
    lfs = [
        sio.LabeledFrame(video=video, frame_idx=i, instances=inst)
        for i, inst in enumerate(frames)
    ]
    labels = sio.Labels(labeled_frames=lfs, videos=[video], skeletons=[skeleton])
    add_bboxes_to_labels(labels, source="instances", padding=4.0)
    if with_box_only:  # One extra box with no pose instance.
        lfs[0].bboxes.append(sio.UserBoundingBox.from_xyxy(2, 2, 12, 12))
    return labels


def _check_sample(sample, order=(1, 0, 2, 3), tol=1.6):
    """Every target's points sit on their blobs and inside their box.

    A target's nodes may be in the mirrored order (a flip, or a pasted animal's
    own flip). Points without weight (covered by a pasted animal or a shrunk frame
    under ``supervise_occluded: false``: their blob may be painted over) are not
    checked. Returns ``(targets checked, points checked)``.
    """
    names = NAMES
    image = sample["image"].reshape(1, -1, *sample["image"].shape[-2:]).float()
    h, w = image.shape[-2:]
    n = int(sample["set_count"])
    size = torch.tensor([w, h], dtype=torch.float32)
    checked = n_points = 0
    for t in range(n):
        vis = (sample["set_keypoint_visible"][t] > 0) & (
            sample["set_keypoint_weights"][t] > 0
        )
        if not vis.any():
            continue
        pts = sample["set_keypoints"][t] * size
        cx, cy, bw, bh = sample["set_boxes"][t] * torch.cat([size, size])
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        for k in range(len(names)):
            if vis[k]:
                x, y = pts[k]
                assert x1 - 1 <= x <= x2 + 1 and y1 - 1 <= y <= y2 + 1, (t, k)
        fits = []
        for perm in (list(range(len(names))), list(order)):
            ok = all(
                torch.allclose(
                    _centroid(image, CHANNELS[perm[k]], pts[k], window=4),
                    pts[k],
                    atol=tol,
                )
                for k in range(len(names))
                if vis[k]
            )
            fits.append(ok)
        assert any(fits), (t, pts)
        checked += 1
        n_points += int(vis.sum())
    return checked, n_points


NAMES = ["L", "R", "T", "H"]
CHANNELS = [0, 1, 2, 2]


AUG_CASES = {
    "letterbox": dict(),
    "crop_flips_rot90": dict(crop_p=1.0, flip_p=0.5, vflip_p=0.5, rot90_p=0.5),
    "zoom_out_zero": dict(zoom_out_p=1.0, zoom_out_min=0.5, zoom_out_max=0.8),
    "zoom_out_frame": dict(
        zoom_out_p=1.0, zoom_out_min=0.5, zoom_out_max=0.8, zoom_out_canvas="frame"
    ),
    "copy_paste": dict(
        paste_p=1.0, paste_max=3, paste_dilate=0.6, paste_feather=0.0, flip_p=0.5
    ),
}


@pytest.mark.parametrize("case", sorted(AUG_CASES))
def test_dataset_keeps_points_with_their_boxes(tmp_path, case):
    from sleap_nn.data.custom_datasets import DetrBBoxDataset

    labels = _blob_labels(tmp_path)
    aug = {
        "crop_p": 0.0,
        "shortest_sides": [96],
        "crop_range": [64, 96],
        "flip_p": 0.0,
        **AUG_CASES[case],
    }
    ds = DetrBBoxDataset(
        labels=[labels],
        target_encoder=SetTargetEncoder(),
        categories=["object"],
        max_stride=32,
        ensure_rgb=True,
        apply_aug=True,
        max_hw=(96, 128),
        resize_mode="letterbox",
        detr_aug=aug,
        # Covered points lose their weight, so painted-over blobs are not checked.
        keypoints={"part_names": NAMES, "supervise_occluded": False},
    )
    assert ds._kpt_flip_order == [1, 0, 2, 3] and ds.symmetric_inds == [(4, 5)]
    targets = points = most = 0
    for seed in range(8):
        torch.manual_seed(seed)
        sample = ds[seed % len(ds)]
        assert sample["set_keypoints"].shape[1:] == (4, 2)
        t, p = _check_sample(sample)
        targets, points = targets + t, points + p
        most = max(most, int(sample["set_count"]))
    assert targets >= 8 and points >= 2 * targets
    if case in ("zoom_out_frame", "copy_paste"):
        assert most > 2  # animals were added, with their points


def test_dataset_stock_geometric_aug_keeps_points(tmp_path):
    from sleap_nn.data.custom_datasets import DetrBBoxDataset

    labels = _blob_labels(tmp_path)
    ds = DetrBBoxDataset(
        labels=[labels],
        target_encoder=SetTargetEncoder(),
        categories=["object"],
        max_stride=32,
        ensure_rgb=True,
        apply_aug=True,
        max_hw=(96, 128),
        resize_mode="letterbox",
        geometric_aug={
            "rotation_p": 1.0,
            "rotation_min": -20,
            "rotation_max": 20,
            "flip_p": 0.5,
        },
        keypoints={"part_names": NAMES},
    )
    for seed in range(4):
        torch.manual_seed(seed)
        np.random.seed(seed)
        assert _check_sample(ds[0], tol=2.0)[0] >= 1


def test_dataset_box_only_and_masked_nodes_get_zero_weight(tmp_path):
    from sleap_nn.data.custom_datasets import DetrBBoxDataset

    labels = _blob_labels(tmp_path, n_frames=1, with_box_only=True)
    labels.provenance["labeled_nodes"] = ["L", "T"]
    ds = DetrBBoxDataset(
        labels=[labels],
        target_encoder=SetTargetEncoder(),
        categories=["object"],
        max_stride=32,
        ensure_rgb=True,
        max_hw=(96, 128),
        keypoints={"part_names": ["T", "L", "R"]},  # any order, matched by name
    )
    sample = ds[0]
    assert int(sample["set_count"]) == 3
    w = sample["set_keypoint_weights"][:3]
    # labeled_nodes ["L", "T"]: "R" is unlabeled in this file.
    assert w[:2].tolist() == [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    assert w[2].tolist() == [0.0, 0.0, 0.0]  # the box without a pose
    assert sample["set_keypoint_visible"][2].sum() == 0
    # Node order follows part_names: "T" is the third skeleton node.
    pts = sample["set_keypoints"][0] * torch.tensor([128.0, 96.0])
    inst = labels[0].instances[0].numpy()
    assert torch.allclose(pts[0], torch.tensor(inst[2], dtype=torch.float32), atol=1e-3)
    # node_loss_mask: false turns the file's node mask off (boxes without a pose
    # still train boxes only).
    unmasked = DetrBBoxDataset(
        labels=[labels],
        target_encoder=SetTargetEncoder(),
        categories=["object"],
        max_stride=32,
        ensure_rgb=True,
        max_hw=(96, 128),
        keypoints={"part_names": ["T", "L", "R"], "node_loss_mask": False},
    )[0]
    w = unmasked["set_keypoint_weights"][:3]
    assert w.tolist() == [[1.0] * 3, [1.0] * 3, [0.0] * 3]


# --------------------------------------------------------------------------- #
# Loss on synthetic matches
# --------------------------------------------------------------------------- #
def _synthetic_outputs(q=4, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "pred_logits": torch.randn(1, q, 2, generator=g),
        "pred_boxes": torch.tensor(
            [
                [
                    [0.3, 0.3, 0.2, 0.2],
                    [0.7, 0.7, 0.2, 0.4],
                    [0.5, 0.5, 0.1, 0.1],
                    [0.1, 0.9, 0.1, 0.1],
                ]
            ]
        ),
        "pred_keypoints": torch.rand(1, q, k, 2, generator=g).requires_grad_(),
        "pred_keypoint_logits": torch.randn(1, q, k, generator=g).requires_grad_(),
    }


def test_keypoint_loss_value_gradient_and_masks():
    from sleap_nn.training.detr.criterion import KEYPOINT_MIN_BOX_SIZE

    out = _synthetic_outputs()
    target = {
        "labels": torch.tensor([0, 0]),
        "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.4]]),
        "keypoints": torch.tensor(
            [
                [[0.25, 0.3], [0.35, 0.3], [0.0, 0.0]],
                [[0.7, 0.6], [0.7, 0.8], [0.6, 0.7]],
            ]
        ),
        "keypoint_visible": torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
        "keypoint_weights": torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
    }
    crit = SetCriterion(1, matcher=HungarianMatcher(), losses=("keypoints",))
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    num_boxes = torch.tensor(2.0)
    from sleap_nn.training.detr.criterion import LOSS_TERMS

    losses = LOSS_TERMS["keypoints"](crit, out, [target], indices, num_boxes)

    pred = out["pred_keypoints"][0, :2].detach()
    size = target["boxes"][:, 2:].clamp(min=KEYPOINT_MIN_BOX_SIZE)[:, None]
    err = ((pred - target["keypoints"]).abs() / size).sum(-1)
    on = target["keypoint_visible"] * target["keypoint_weights"]
    # Weighted means: by the summed weights of the located points / of all points.
    expect_l1 = (err * on).sum() / on.sum()
    logits = out["pred_keypoint_logits"][0, :2].detach()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target["keypoint_visible"], reduction="none"
    )
    expect_vis = (bce * target["keypoint_weights"]).sum() / target[
        "keypoint_weights"
    ].sum()
    torch.testing.assert_close(losses["loss_keypoints"], expect_l1)
    torch.testing.assert_close(losses["loss_keypoint_vis"], expect_vis)

    (losses["loss_keypoints"] + losses["loss_keypoint_vis"]).backward()
    gp = out["pred_keypoints"].grad[0]
    gv = out["pred_keypoint_logits"].grad[0]
    assert gp[2:].abs().sum() == 0 and gv[2:].abs().sum() == 0  # unmatched queries
    assert gp[0, 2].abs().sum() == 0  # invisible point: no L1
    assert gp[1, 1].abs().sum() == 0 and gv[1, 1] == 0  # masked node
    assert gp[0, 0].abs().sum() > 0 and gv[0, 2] != 0  # invisible: still BCE
    # Box-only targets: both terms are zero.
    box_only = {k: target[k] for k in ("labels", "boxes")}
    zero = LOSS_TERMS["keypoints"](crit, out, [box_only], indices, num_boxes)
    assert float(zero["loss_keypoints"]) == 0 and float(zero["loss_keypoint_vis"]) == 0
    # Layers without keypoint outputs (encoder proposals) have no keypoint terms.
    assert (
        LOSS_TERMS["keypoints"](
            crit, {"pred_boxes": out["pred_boxes"]}, [target], indices, num_boxes
        )
        == {}
    )


def test_set_targets_carry_keypoints():
    enc = SetTargetEncoder(max_targets=3)
    out = enc(
        torch.tensor([[0.0, 0.0, 50.0, 100.0]]),
        torch.tensor([0]),
        (100, 200),
        keypoints=torch.tensor([[[10.0, 20.0], [float("nan"), float("nan")]]]),
        keypoint_weights=torch.tensor([[1.0, 0.5]]),
    )
    assert out["set_keypoints"].shape == (3, 2, 2)
    torch.testing.assert_close(out["set_keypoints"][0, 0], torch.tensor([0.05, 0.2]))
    assert out["set_keypoints"][0, 1].tolist() == [0.0, 0.0]
    assert out["set_keypoint_visible"][0].tolist() == [1.0, 0.0]
    assert out["set_keypoint_weights"][:, 1].tolist() == [0.5, 0.0, 0.0]
    targets = collate_set_targets(torch.utils.data.default_collate([out]))
    assert targets[0]["keypoints"].shape == (1, 2, 2)
    plain = enc(torch.zeros(1, 4), torch.tensor([0]), (10, 10))
    assert "set_keypoints" not in plain
    assert (
        "keypoints"
        not in collate_set_targets(torch.utils.data.default_collate([plain]))[0]
    )


# --------------------------------------------------------------------------- #
# Model: the plain detector is unchanged
# --------------------------------------------------------------------------- #
HEAD = dict(
    in_channels=8,
    hidden_dim=16,
    num_queries=6,
    group_detr=2,
    dec_layers=2,
    sa_nheads=2,
    ca_nheads=4,
    dim_feedforward=32,
)


def test_plain_head_unchanged_and_keypoint_head_adds_only_its_modules():
    torch.manual_seed(0)
    plain = BBoxDetrHead(**HEAD)
    torch.manual_seed(0)
    kpt = BBoxDetrHead(**HEAD, num_keypoints=5)
    assert plain.num_keypoints == 0
    assert not any("keypoint" in n for n, _ in plain.named_parameters())
    p_sd, k_sd = plain.state_dict(), kpt.state_dict()
    extra = sorted(set(k_sd) - set(p_sd))
    assert extra and all(n.startswith("keypoint_") for n in extra)
    # The box detector's weights are drawn exactly as without keypoints.
    for name, value in p_sd.items():
        assert torch.equal(value, k_sd[name]), name

    feats = [torch.randn(2, 8, 4, 4)]
    plain.eval(), kpt.eval()
    with torch.no_grad():
        a, b = plain(feats), kpt(feats)
    assert set(a) == {"pred_logits", "pred_boxes", "aux_outputs", "enc_outputs"}
    assert torch.equal(a["pred_boxes"], b["pred_boxes"])
    assert b["pred_keypoints"].shape == (2, 6, 5, 2)
    assert b["pred_keypoint_logits"].shape == (2, 6, 5)
    assert all("pred_keypoints" in x for x in b["aux_outputs"])
    assert "pred_keypoints" not in b["enc_outputs"]
    # Zero-initialized offsets: every point starts at its box center.
    centers = b["pred_boxes"][..., None, :2].expand_as(b["pred_keypoints"])
    torch.testing.assert_close(b["pred_keypoints"], centers)


def test_keypoint_head_trains_points_not_boxes():
    torch.manual_seed(0)
    head = BBoxDetrHead(**HEAD, num_keypoints=2)
    hidden = torch.randn(3, 2, 16)
    boxes = torch.rand(3, 2, 4).requires_grad_()
    for p in head.keypoint_embed.layers[-1].parameters():
        torch.nn.init.normal_(p)
    pts, vis = head.keypoint_outputs(hidden, boxes)
    pts.sum().backward()
    assert boxes.grad is None  # the box is detached
    # xy = c + d * wh
    d = head.keypoint_embed(hidden).reshape(3, 2, 2, 2)
    expect = boxes.detach()[..., None, :2] + d * boxes.detach()[..., None, 2:]
    torch.testing.assert_close(pts, expect)


# --------------------------------------------------------------------------- #
# Config: off by default, nothing new saved
# --------------------------------------------------------------------------- #
def test_config_keypoints_off_by_default_and_not_saved():
    from sleap_nn.config.get_config import get_head_configs
    from sleap_nn.config.model_config import config_to_save
    from sleap_nn.config.training_job_config import TrainingJobConfig

    heads = OmegaConf.structured(get_head_configs("bbox_detr"))
    assert heads.bbox_detr.detr.keypoints is None
    cfg = OmegaConf.structured(TrainingJobConfig())
    cfg.model_config.head_configs = heads
    saved = config_to_save(cfg)
    assert "keypoints" not in saved.model_config.head_configs.bbox_detr.detr
    assert "keypoints" in cfg.model_config.head_configs.bbox_detr.detr  # copy only
    lines = OmegaConf.to_yaml(cfg).splitlines()
    assert OmegaConf.to_yaml(saved).splitlines() == [
        line for line in lines if line.strip() != "keypoints: null"
    ]
    # A config without a DETR head is returned as is.
    other = OmegaConf.structured(TrainingJobConfig())
    assert config_to_save(other) is other
    # `keypoints: {}` turns the branch on with its defaults.
    on = OmegaConf.merge(
        OmegaConf.structured(get_head_configs("bbox_detr")),
        {"bbox_detr": {"detr": {"keypoints": {}}}},
    )
    kp = on.bbox_detr.detr.keypoints
    assert kp.part_names is None and kp.loss_weight == 5.0
    assert kp.visibility_loss_weight == 1.0 and kp.visibility_threshold == 0.5
    assert (
        "keypoints"
        in config_to_save(
            OmegaConf.merge(cfg, {"model_config": {"head_configs": on}})
        ).model_config.head_configs.bbox_detr.detr
    )


# --------------------------------------------------------------------------- #
# End to end on CPU
# --------------------------------------------------------------------------- #
def _kpt_config(config, tmp_path, run_name="bbox_detr_kpt", **detr_aug):
    cfg = _detr_train_config(config, tmp_path, run_name=run_name, **detr_aug)
    # As a user's YAML `keypoints: {}` reads: the section with its defaults.
    cfg = OmegaConf.merge(
        cfg,
        {"model_config": {"head_configs": {"bbox_detr": {"detr": {"keypoints": {}}}}}},
    )
    cfg.trainer_config.ema = {"enabled": False}
    return cfg


def test_keypoint_detr_trains_saves_loads_and_predicts(
    config, minimal_instance, tmp_path
):
    from sleap_nn.inference.layers.bbox import BBoxLayer
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.training.lightning_modules import BBoxDetrLightningModule
    from sleap_nn.training.model_trainer import ModelTrainer

    cfg = _kpt_config(
        config, tmp_path, zoom_out_p=0.5, paste_p=0.5, flip_p=0.5, rot90_p=0.5
    )
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    logged = trainer.trainer.callback_metrics
    for stage in ("train", "val"):
        for term in ("loss_keypoints", "loss_keypoint_vis"):
            for suffix in ("", "_0"):
                assert f"{stage}/{term}{suffix}" in logged
        assert f"{stage}/loss_keypoints_enc" not in logged

    run_dir = tmp_path / "bbox_detr_kpt"
    saved = OmegaConf.load(run_dir / "training_config.yaml")
    kp = saved.model_config.head_configs.bbox_detr.detr.keypoints
    assert list(kp.part_names) == ["A", "B"]
    assert [n["name"] for n in saved.data_config.skeletons[0]["nodes"]] == ["A", "B"]

    # The checkpoint loads strictly into a model built from the saved config.
    ckpt = torch.load(run_dir / "best.ckpt", map_location="cpu", weights_only=False)
    module = BBoxDetrLightningModule(
        model_type="bbox_detr",
        backbone_type="dinov2_windowed",
        backbone_config=saved.model_config.backbone_config,
        head_configs=saved.model_config.head_configs,
    )
    module.load_state_dict(ckpt["state_dict"], strict=True)
    assert module.model.detr.num_keypoints == 2

    pred = Predictor.from_model_paths(
        [run_dir.as_posix()], device="cpu", peak_threshold=0.0
    )
    assert isinstance(pred.layer, BBoxLayer)
    pred.layer.decoder.visibility_threshold = 0.0  # untrained: keep every point
    labels = pred.predict(minimal_instance.as_posix())
    lf = labels[0]
    assert lf.instances and len(lf.instances) == len(lf.bboxes)
    assert labels.skeletons[0].node_names == ["A", "B"]
    for inst, box in zip(lf.instances, lf.bboxes):
        assert isinstance(inst, sio.PredictedInstance) and box.instance is inst
        pts = inst.numpy()
        assert pts.shape == (2, 2) and np.isfinite(pts).all()
        scores = inst.points["score"]
        assert ((scores >= 0) & (scores <= 1)).all()
        assert math.isclose(inst.score, box.score, rel_tol=1e-6)
    out = tmp_path / "pred.slp"
    labels.save(out.as_posix())
    assert len(sio.load_slp(out.as_posix())[0].instances) == len(lf.instances)

    # The inference layer on its own: one instance per box slot.
    outputs = pred.layer.predict(
        torch.from_numpy(sio.load_slp(minimal_instance.as_posix())[0].image).permute(
            2, 0, 1
        )[None]
    )
    assert outputs.pred_keypoints.shape[:2] == outputs.pred_bboxes.shape[:2]
    assert outputs.pred_keypoints.shape[2:] == (2, 2)

    # Export refuses the keypoint branch with a clear message.
    from click.testing import CliRunner

    from sleap_nn.export.cli import export as export_cmd

    result = CliRunner().invoke(
        export_cmd, [run_dir.as_posix(), "-o", (tmp_path / "exp").as_posix()]
    )
    assert result.exit_code != 0 and "keypoints" in result.output
    from sleap_nn.export.wrappers.bbox import BBoxONNXWrapper

    with pytest.raises(ValueError, match="keypoint"):
        BBoxONNXWrapper(module.model, model_type="bbox_detr")


def test_box_distillation_refuses_a_keypoint_student():
    from sleap_nn.training.distillation import _attach_box_distillation

    class Student:
        model_type = "bbox_detr"

    torch.manual_seed(0)
    Student.model = torch.nn.Module()
    Student.model.detr = BBoxDetrHead(**HEAD, num_keypoints=2)
    with pytest.raises(ValueError, match="keypoints"):
        _attach_box_distillation(Student(), lambda key, default=None: default)


# --------------------------------------------------------------------------- #
# Review follow-ups
# --------------------------------------------------------------------------- #
def _negative_file(tmp_path):
    """A one-frame file whose frame is a confirmed negative (no animals, no boxes)."""
    neg = _blob_labels(tmp_path / "neg_dir", n_frames=1)
    neg[0].instances = []
    neg[0].bboxes = []
    neg[0].is_negative = True
    return neg


@pytest.mark.parametrize("paste", [False, True], ids=["zoom_frame", "zoom_frame_paste"])
def test_negative_canvas_frames_with_keypoints(tmp_path, paste):
    """Zoom-out onto a frame (and copy-paste) may draw a negative frame: no crash."""
    from sleap_nn.data.custom_datasets import DetrBBoxDataset

    (tmp_path / "neg_dir").mkdir()
    labels, neg = _blob_labels(tmp_path), _negative_file(tmp_path)
    aug = {
        "crop_p": 0.0,
        "shortest_sides": [96],
        "crop_range": [64, 96],
        "flip_p": 0.0,
        "zoom_out_p": 1.0,
        "zoom_out_min": 0.5,
        "zoom_out_max": 0.8,
        "zoom_out_canvas": "frame",
        **(dict(paste_p=1.0, paste_dilate=0.6, paste_feather=0.0) if paste else {}),
    }
    ds = DetrBBoxDataset(
        labels=[labels, neg],
        target_encoder=SetTargetEncoder(),
        categories=["object"],
        max_stride=32,
        ensure_rgb=True,
        apply_aug=True,
        max_hw=(96, 128),
        resize_mode="letterbox",
        detr_aug=aug,
        use_negative_frames=True,
        keypoints={"part_names": NAMES, "supervise_occluded": False},
    )
    assert any(bool(ds.lf_idx_list[i]["is_negative"]) for i in range(len(ds)))
    for seed in range(16):
        torch.manual_seed(seed)
        sample = ds[seed % len(ds)]
        _check_sample(sample)


def test_unweight_covered_points():
    from sleap_nn.data.custom_datasets import (
        _unweight_points_in_rect,
        _unweight_points_under,
    )

    pts = torch.tensor([[[5.0, 5.0, 1.0], [20.0, 20.0, 1.0], [float("nan"), 0, 1.0]]])
    out = _unweight_points_in_rect(pts, (0, 0, 10, 10))
    assert out[0, :, 2].tolist() == [0.0, 1.0, 1.0]
    assert torch.equal(out[0, :2, :2], pts[0, :2, :2])  # locations kept
    alpha = torch.zeros(10, 10)
    alpha[:5, :5] = 1.0
    out = _unweight_points_under(pts, alpha, 3, 3)
    assert out[0, :, 2].tolist() == [0.0, 1.0, 1.0]


@pytest.mark.parametrize("supervise_occluded", [True, False])
def test_covered_points_are_occluded_points(tmp_path, supervise_occluded):
    """Points covered by a pasted animal or a shrunk frame keep their location.

    Default: they are trained as located (visibility target 1, weight 1), like a
    labelled occluded point. ``supervise_occluded: false``: they get weight 0.
    Every point of these frames is labelled and inside the input, so every
    visibility target is 1 either way.
    """
    from sleap_nn.data.custom_datasets import DetrBBoxDataset

    labels = _blob_labels(tmp_path)
    ds = DetrBBoxDataset(
        labels=[labels],
        target_encoder=SetTargetEncoder(),
        categories=["object"],
        max_stride=32,
        ensure_rgb=True,
        apply_aug=True,
        max_hw=(96, 128),
        resize_mode="letterbox",
        detr_aug={
            "crop_p": 0.0,
            "shortest_sides": [96],
            "crop_range": [64, 96],
            "flip_p": 0.0,
            "paste_p": 1.0,
            "paste_max": 6,
            "paste_overlap": 1.0,
            "zoom_out_p": 1.0,
            "zoom_out_canvas": "frame",
        },
        keypoints={"part_names": NAMES, "supervise_occluded": supervise_occluded},
    )
    zero_weight = 0
    for seed in range(8):
        torch.manual_seed(seed)
        sample = ds[seed % len(ds)]
        n = int(sample["set_count"])
        w = sample["set_keypoint_weights"][:n]
        assert (sample["set_keypoint_visible"][:n] == 1).all()  # locations kept
        assert ((w == 0) | (w == 1)).all()
        zero_weight += int((w == 0).sum())
    if supervise_occluded:
        assert zero_weight == 0
    else:
        assert zero_weight > 0  # covered points lost their weight


def test_unlinked_boxes_warn_and_training_set_without_poses_raises(tmp_path):
    from sleap_nn.data.custom_datasets import DetrBBoxDataset

    labels = _blob_labels(tmp_path)
    for lf in labels:  # Boxes no longer linked to their instances.
        for box in lf.bboxes:
            box.instance = None
    common = dict(
        labels=[labels],
        target_encoder=SetTargetEncoder(),
        categories=["object"],
        max_stride=32,
        ensure_rgb=True,
        max_hw=(96, 128),
    )
    ds = DetrBBoxDataset(**common, keypoints={"part_names": NAMES})
    assert ds._kpt_box_counts == [[6, 0, 0]]
    assert float(ds[0]["set_keypoint_weights"].sum()) == 0
    with pytest.raises(ValueError, match="no training box is linked"):
        DetrBBoxDataset(
            **common, keypoints={"part_names": NAMES, "require_poses": True}
        )
    (tmp_path / "linked").mkdir()
    linked = _blob_labels(tmp_path / "linked")
    ds = DetrBBoxDataset(
        **{**common, "labels": [linked, labels]},
        keypoints={"part_names": NAMES, "require_poses": True},
    )
    assert ds._kpt_box_counts == [[6, 6, 6], [6, 0, 0]]
    # Linked, but labeled_nodes leaves no keypoint node labelled: another cause.
    (tmp_path / "masked").mkdir()
    masked = _blob_labels(tmp_path / "masked")
    masked.provenance["labeled_nodes"] = ["X"]
    with pytest.raises(ValueError, match="linked to pose instances, but every"):
        DetrBBoxDataset(
            **{**common, "labels": [masked]},
            keypoints={"part_names": NAMES, "require_poses": True},
        )


def test_integer_labeled_nodes_read_against_the_files_own_skeleton(tmp_path):
    from sleap_nn.data.custom_datasets import _labeled_node_mask

    labels = _blob_labels(tmp_path, n_frames=1)  # skeleton L, R, T, H
    labels.provenance["labeled_nodes"] = [0, 2]  # L and T in the file's skeleton
    assert _labeled_node_mask(labels, ["T", "R", "L"]).tolist() == [1.0, 0.0, 1.0]
    labels.provenance["labeled_nodes"] = ["H", 1]
    assert _labeled_node_mask(labels, ["H", "R", "L"]).tolist() == [1.0, 1.0, 0.0]
    del labels.provenance["labeled_nodes"]
    assert _labeled_node_mask(labels, ["H"]).tolist() == [1.0]


def test_keypoint_decode_gives_one_detection_per_query():
    from sleap_nn.inference.bbox import decode_set, decode_set_keypoints

    logits = torch.full((1, 3, 3), -10.0)  # 2 classes + the extra logit
    logits[0, 0, :2] = torch.tensor([2.0, 1.0])  # query 0: both classes > 0.2
    logits[0, 1, 1] = 3.0  # query 1: class 1
    boxes = torch.tensor(
        [[[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1]]]
    )
    kpts = torch.rand(1, 3, 2, 2)
    vis = torch.full((1, 3, 2), 5.0)
    b, sc, cl = decode_set(logits, boxes, (100, 100), score_threshold=0.2)
    assert int(torch.isfinite(sc).sum()) == 3  # (query 0, class 0 and 1), query 1
    kb, ks, kc, kp, kv = decode_set_keypoints(
        logits, boxes, kpts, vis, (100, 100), score_threshold=0.2
    )
    n = int(torch.isfinite(ks).sum())
    assert n == 2 and kc[0, :n].tolist() == [1, 0]  # by score: query 1, query 0
    torch.testing.assert_close(kp[0, 1], kpts[0, 0] * 100)
    # With one class both decoders select the same detections.
    one = logits[..., [0, 2]]
    b1, s1, c1 = decode_set(one, boxes, (100, 100), score_threshold=0.0)
    b2, s2, c2, _, _ = decode_set_keypoints(
        one, boxes, kpts, vis, (100, 100), score_threshold=0.0
    )
    assert torch.equal(b1.nan_to_num(-1), b2.nan_to_num(-1))
    assert torch.equal(s1.nan_to_num(-1), s2.nan_to_num(-1)) and torch.equal(c1, c2)


def _write_variant(minimal_instance, tmp_path, names, name):
    """``minimal_instance`` with skeleton ``names`` (A and B matched by name)."""
    labels = sio.load_slp(minimal_instance.as_posix())
    skeleton = sio.Skeleton(list(names), name=name)
    for lf in labels:
        new = []
        for inst in lf.instances:
            src = dict(zip(inst.skeleton.node_names, inst.numpy()))
            pts = np.array([src.get(n, [np.nan, np.nan]) for n in names], float)
            new.append(sio.Instance.from_numpy(pts, skeleton=skeleton))
        lf.instances = new
    labels.skeletons = [skeleton]
    path = tmp_path / f"{name}.pkg.slp"
    labels.save(path.as_posix(), embed=True)
    return path.as_posix()


@pytest.mark.parametrize("case", ["reordered", "subset", "two_skeletons"])
def test_trainer_matches_keypoint_nodes_by_name(
    config, minimal_instance, tmp_path, case
):
    from sleap_nn.training.model_trainer import ModelTrainer

    cfg = _kpt_config(config, tmp_path, run_name=case)
    cfg.trainer_config.max_epochs = 1
    cfg.trainer_config.visualize_preds_during_training = False
    kp = cfg.model_config.head_configs.bbox_detr.detr.keypoints
    expect = {"reordered": ["B", "A"], "subset": ["A"], "two_skeletons": ["A", "B"]}
    if case != "two_skeletons":
        kp.part_names = expect[case]
    else:  # default part_names: the first file's; the second has C, B, A
        other = _write_variant(minimal_instance, tmp_path, ["C", "B", "A"], "other")
        cfg.data_config.train_labels_path = [minimal_instance.as_posix(), other]
        cfg.data_config.val_labels_path = [other]
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    saved = OmegaConf.load(tmp_path / case / "training_config.yaml")
    names = expect[case]
    assert (
        list(saved.model_config.head_configs.bbox_detr.detr.keypoints.part_names)
        == names
    )
    assert [n["name"] for n in saved.data_config.skeletons[0]["nodes"]] == names
    # The training data the trainer builds puts every point on its node by name.
    from sleap_nn.data.custom_datasets import DetrBBoxDataset, detr_dataset_kwargs

    _, kwargs = detr_dataset_kwargs(trainer.config)  # no augmentation
    assert kwargs["keypoints"]["part_names"] == names
    ds = DetrBBoxDataset(
        labels=trainer.train_labels,
        target_encoder=SetTargetEncoder(),
        categories=["object"],
        max_stride=32,
        ensure_rgb=True,
        max_hw=(128, 128),
        **kwargs,
    )
    checked = 0
    for i in range(len(ds)):
        sample = ds[i]
        entry = ds.lf_idx_list[i]
        lf = trainer.train_labels[entry["labels_idx"]][entry["lf_idx"]]
        scale = float(sample["eff_scale"][0])
        for t, box in enumerate(lf.bboxes):
            inst = box.instance
            own = dict(zip(inst.skeleton.node_names, inst.numpy()))
            got = sample["set_keypoints"][t] * 128.0
            for k, name in enumerate(names):
                expect = torch.tensor(
                    own.get(name, [np.nan, np.nan]), dtype=torch.float32
                )
                if torch.isfinite(expect).all():
                    torch.testing.assert_close(
                        got[k], expect * scale, atol=1e-3, rtol=0
                    )
                    checked += 1
                else:
                    assert sample["set_keypoint_visible"][t, k] == 0
    assert checked >= 2 * len(names)


@pytest.mark.parametrize("mode", ["letterbox", "squash"])
def test_predicted_points_map_back_like_boxes(config, minimal_instance, tmp_path, mode):
    """Zeroed offsets: every predicted point lands on its box center in the frame."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.training.model_trainer import ModelTrainer

    cfg = _kpt_config(config, tmp_path)
    cfg.data_config.preprocessing.resize_mode = mode
    cfg.data_config.preprocessing.max_width = 160
    cfg.trainer_config.max_epochs = 1
    cfg.trainer_config.visualize_preds_during_training = False
    ModelTrainer.get_model_trainer_from_config(cfg).train()
    pred = Predictor.from_model_paths(
        [(tmp_path / "bbox_detr_kpt").as_posix()], device="cpu", peak_threshold=0.0
    )
    detr = next(
        m for m in pred.layer.backend.model.modules() if hasattr(m, "keypoint_embed")
    )
    with torch.no_grad():
        detr.keypoint_embed.layers[-1].weight.zero_()
        detr.keypoint_embed.layers[-1].bias.zero_()
        detr.keypoint_visibility_embed.weight.zero_()
        detr.keypoint_visibility_embed.bias.fill_(10.0)
    frame = sio.load_slp(minimal_instance.as_posix())[0].image
    img = torch.from_numpy(frame).permute(2, 0, 1)[None][..., 40:340]  # 384 x 300
    out = pred.layer.predict(img)
    b, p = out.pred_bboxes[0], out.pred_keypoints[0]
    ok = torch.isfinite(b).all(-1)
    # Boxes clipped to the frame have moved centers; compare boxes well inside it.
    inside = ok & (b[:, 0] > 1) & (b[:, 1] > 1) & (b[:, 2] < 298) & (b[:, 3] < 382)
    assert inside.any()
    centers = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], -1)
    for k in range(p.shape[1]):
        torch.testing.assert_close(p[inside, k], centers[inside], atol=1e-3, rtol=0)


def test_labeled_node_mask_warns_like_the_node_weights(tmp_path):
    from loguru import logger

    from sleap_nn.data.custom_datasets import _labeled_node_mask

    labels = _blob_labels(tmp_path, n_frames=1)
    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        labels.provenance["labeled_nodes"] = ["L", "Tial"]  # a misspelled name
        assert _labeled_node_mask(labels, NAMES).tolist() == [1.0, 0.0, 0.0, 0.0]
        assert any("Tial" in m for m in messages)
        labels.provenance["labeled_nodes"] = ["H"]
        assert _labeled_node_mask(labels, ["L", "R"]).tolist() == [0.0, 0.0]
        assert any("matches no keypoint node" in m for m in messages)
    finally:
        logger.remove(sink)
