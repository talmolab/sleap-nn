"""Tests for the single-warp centered-instance data path (`crop_pipeline="warp"`)."""

import cv2
import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf

from sleap_nn.data.confidence_maps import generate_confmaps
from sleap_nn.data.instance_cropping import (
    BBoxCropSpec,
    bbox_crop_windows,
    crop_windows,
    points_to_crop,
)
from sleap_nn.data.topdown_warp import (
    BalancedBatchSampler,
    BalancedDataLoader,
    WarpAug,
    InstanceFilter,
    WarpCenteredInstanceDataset,
    render_confmaps,
    merged_skeleton,
    resolve_batch_counts,
)

NODES = ["head", "left_hand", "right_hand", "tail"]


def _labels(tmp_path, name="a", nodes=NODES, labeled=None, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 2)
    path = str(tmp_path / f"{name}.png")
    cv2.imwrite(path, img)
    nodes = list(nodes)
    sk = sio.Skeleton(nodes=list(nodes), edges=[(nodes[0], nodes[-1])])
    if "left_hand" in nodes and "right_hand" in nodes:
        sk.add_symmetry("left_hand", "right_hand")
    video = sio.Video.from_filename([path])
    pts = {
        "head": (60.0, 30.0),
        "left_hand": (90.0, 50.0),
        "right_hand": (40.0, 55.0),
        "tail": (65.0, 90.0),
        "extra": (70.0, 60.0),
    }
    inst = sio.Instance.from_numpy(np.array([pts[n] for n in nodes]), skeleton=sk)
    tiny = sio.Instance.from_numpy(
        np.array([[5.0, 5.0]] + [[np.nan, np.nan]] * (len(nodes) - 1)), skeleton=sk
    )
    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst, tiny])
    labels = sio.Labels([lf], skeletons=[sk], videos=[video])
    if labeled is not None:
        labels.provenance["labeled_nodes"] = labeled
    return labels, img[:, :, ::-1].copy()  # RGB


def _head(nodes=NODES):
    return OmegaConf.create(
        {"part_names": nodes, "sigma": 1.5, "output_stride": 2, "anchor_part": None}
    )


def test_bbox_crop_windows_square_tuple_unchanged():
    boxes = torch.tensor([[10.0, 20.0, 50.0, 100.0], [0.0, 0.0, 3.0, 2.0]])
    for aspect in ("square", "stretch", "letterbox"):
        a = bbox_crop_windows(boxes, 64, aspect, 0.1, 8.0)
        b = bbox_crop_windows(boxes, (64, 64), aspect, 0.1, 8.0)
        for x, y in zip(a, b):
            assert torch.equal(x, y)


def test_bbox_crop_windows_non_square():
    boxes = torch.tensor([[10.0, 20.0, 50.0, 100.0], [0.0, 0.0, 200.0, 40.0]])
    windows, scales, _ = bbox_crop_windows(boxes, (64, 48), "square", 0.0)
    sides = windows[:, 2:] - windows[:, :2]
    # Crop aspect (w / h = 0.75), one scale for both axes, and the box fits.
    torch.testing.assert_close(sides[:, 0] / sides[:, 1], torch.full((2,), 0.75))
    torch.testing.assert_close(scales[:, 0], scales[:, 1])
    assert (sides[:, 0] >= boxes[:, 2] - boxes[:, 0] - 1e-4).all()
    assert (sides[:, 1] >= boxes[:, 3] - boxes[:, 1] - 1e-4).all()
    # Tall box fills the height, wide box fills the width.
    torch.testing.assert_close(sides[0, 1], torch.tensor(80.0))
    torch.testing.assert_close(sides[1, 0], torch.tensor(200.0))


def test_warp_crop_matches_inference_crop(tmp_path):
    labels, img = _labels(tmp_path)
    spec = BBoxCropSpec(padding=0.2, bbox_source="instances")
    ds = WarpCenteredInstanceDataset([labels], (64, 48), _head(), bbox_crop=spec)
    assert len(ds) == 2
    s = ds[0]
    assert s["instance_image"].shape == (1, 3, 64, 48)
    assert s["instance_image"].dtype == torch.uint8
    box = torch.from_numpy(ds.table.box[:1])
    win, sc, _ = spec.windows(box, (64, 48))
    frame = torch.from_numpy(img).permute(2, 0, 1)[None].float()
    ref = crop_windows(frame, win, torch.tensor([0]), (64, 48))[0]
    diff = (s["instance_image"][0].float() - ref).abs()
    assert diff[:, 2:-2, 2:-2].max() <= 4  # cv2 fixed-point bilinear + rounding
    assert diff.mean() < 1.0
    pts = torch.from_numpy(ds.table.kpts[:1])
    torch.testing.assert_close(
        s["instance"], points_to_crop(pts, win, sc), atol=1e-3, rtol=0
    )
    cms = generate_confmaps(s["instance"], img_hw=(64, 48), sigma=1.5, output_stride=2)
    assert "confidence_maps" in s
    torch.testing.assert_close(s["confidence_maps"], cms)
    torch.testing.assert_close(render_confmaps(s["instance"], (64, 48), 1.5, 2), cms)


def test_warp_flip_swaps_symmetric_nodes(tmp_path):
    labels, _ = _labels(tmp_path)
    spec = BBoxCropSpec(padding=0.2, bbox_source="instances")
    plain = WarpCenteredInstanceDataset([labels], (64, 48), _head(), bbox_crop=spec)
    geo = {"rotation_p": 0.0, "scale_p": 0.0, "flip_p": 1.0}
    flip = WarpCenteredInstanceDataset(
        [labels],
        (64, 48),
        _head(),
        bbox_crop=spec,
        geometric_aug=geo,
        apply_aug=True,
        return_flip=True,
        gpu_confmaps=True,
    )
    a, b = plain[0], flip[0]
    assert b["flipped"] and "confidence_maps" not in b
    # Image is the mirror of the un-flipped crop.
    torch.testing.assert_close(b["instance_image"], a["instance_image"].flip(-1))
    # Left/right swapped, x mirrored about the crop center.
    ka, kb = a["instance"][0], b["instance"][0]
    i_l, i_r = NODES.index("left_hand"), NODES.index("right_hand")
    torch.testing.assert_close(kb[i_l, 0], 47 - ka[i_r, 0])
    torch.testing.assert_close(kb[i_r, 0], 47 - ka[i_l, 0])
    torch.testing.assert_close(kb[i_l, 1], ka[i_r, 1])
    torch.testing.assert_close(kb[0], torch.stack([47 - ka[0, 0], ka[0, 1]]))


def test_warp_mixed_skeletons_and_weights(tmp_path):
    la, _ = _labels(tmp_path, "a")
    lb, _ = _labels(
        tmp_path, "b", nodes=["head", "tail", "extra"], labeled=["head", "extra"]
    )
    nodes = NODES + ["extra"]
    ds = WarpCenteredInstanceDataset(
        [la, lb],
        (32, 32),
        _head(nodes),
        bbox_crop=BBoxCropSpec(bbox_source="instances"),
        instance_filter=InstanceFilter(min_labeled_nodes=2),
    )
    # The single-point instances are filtered out; one person per file is left.
    assert ds.table.labels_idx.tolist() == [0, 1]
    assert ds.table.filtered["few_labeled_nodes"] == 2
    torch.testing.assert_close(ds[0]["node_weights"], torch.tensor([[1, 1, 1, 1, 0.0]]))
    # File b: no hands, and `tail` exists but is not in labeled_nodes.
    torch.testing.assert_close(ds[1]["node_weights"], torch.tensor([[1, 0, 0, 0, 1.0]]))
    assert torch.isnan(ds[1]["instance"][0, NODES.index("tail")]).all()
    sk = merged_skeleton([la, lb], nodes)
    assert sk.node_names == nodes and len(sk.symmetries) == 1


def test_warp_unlabeled_points_ignore(tmp_path):
    la, _ = _labels(tmp_path, "a")
    # Drop the tail of the main instance (a COCO v=0 point).
    inst = la.labeled_frames[0].instances[0]
    pts = inst.numpy()
    pts[NODES.index("tail")] = np.nan
    la.labeled_frames[0].instances[0] = sio.Instance.from_numpy(
        pts, skeleton=la.skeletons[0]
    )
    kw = dict(
        crop_size=(32, 32),
        confmap_head_config=_head(),
        bbox_crop=BBoxCropSpec(bbox_source="instances"),
        instance_filter=InstanceFilter(min_labeled_nodes=2),
    )
    absent = WarpCenteredInstanceDataset([la], **kw)
    ignore = WarpCenteredInstanceDataset([la], unlabeled_points="ignore", **kw)
    # Fully labeled file + "absent": no weights, so the loss is the plain MSE.
    assert "node_weights" not in absent[0]
    torch.testing.assert_close(
        ignore[0]["node_weights"], torch.tensor([[1, 1, 1, 0.0]])
    )
    off = WarpCenteredInstanceDataset(
        [la], unlabeled_points="ignore", node_loss_mask=False, **kw
    )
    assert "node_weights" not in off[0]
    with pytest.raises(ValueError):
        WarpCenteredInstanceDataset([la], unlabeled_points="drop", **kw)


def test_warp_aug_stream_depends_on_rank(tmp_path):
    la, _ = _labels(tmp_path, "a")
    geo = OmegaConf.create(
        {
            "rotation_min": -40.0,
            "rotation_max": 40.0,
            "rotation_p": 1.0,
            "scale_min": 0.7,
            "scale_max": 1.3,
            "scale_p": 1.0,
        }
    )
    crops = []
    for rank in (0, 0, 1):
        ds = WarpCenteredInstanceDataset(
            [la],
            (32, 32),
            _head(),
            bbox_crop=BBoxCropSpec(bbox_source="instances"),
            geometric_aug=geo,
            apply_aug=True,
            base_seed=7,
        )
        ds.rank = rank
        crops.append(ds[0]["instance"])
    torch.testing.assert_close(crops[0], crops[1], equal_nan=True)
    assert not torch.allclose(torch.nan_to_num(crops[0]), torch.nan_to_num(crops[2]))


def test_balanced_batch_sampler():
    groups = np.array([0] * 5 + [1] * 100)
    s = BalancedBatchSampler(groups, [3, 5], batches_per_epoch=7, seed=1, rank=0)
    batches = list(s)
    assert len(s) == 7 and len(batches) == 7
    for b in batches:
        assert len(b) == 8
        assert (groups[b] == 0).sum() == 3 and (groups[b] == 1).sum() == 5
    # Small group cycles through all its samples (oversampled).
    assert set(np.concatenate(batches)[groups[np.concatenate(batches)] == 0]) == set(
        range(5)
    )
    other = list(BalancedBatchSampler(groups, [3, 5], 7, seed=1, rank=1))
    assert other != batches
    assert resolve_batch_counts(None, [1, 3], 10, 2) == [3, 7]
    assert resolve_batch_counts(None, None, 9, 2) in ([5, 4], [4, 5])
    with pytest.raises(ValueError):
        resolve_batch_counts([1, 2, 3], None, 6, 2)


def test_balanced_batch_sampler_resumes():
    groups = np.array([0] * 5 + [1] * 13)
    full = BalancedBatchSampler(groups, [3, 4], batches_per_epoch=4, seed=2, rank=1)
    stream = []
    for _ in range(3):  # three epochs, as a loader would deliver them
        for b in full:
            stream.append(b)
            full.position += 1
    assert len(stream) == 12
    # A fresh sampler at the saved position yields the rest of that epoch, then
    # continues the same stream.
    resumed = BalancedBatchSampler(groups, [3, 4], 4, seed=2, rank=1)
    resumed.load_state_dict({"position": 6})
    rest = list(resumed)
    assert rest == stream[6:8]
    resumed.position = 8
    assert list(resumed) == stream[8:12]
    assert full.state_dict() == {"position": 12}


def test_balanced_data_loader_counts_delivered_batches():
    groups = np.array([0] * 4 + [1] * 6)
    sampler = BalancedBatchSampler(groups, [1, 2], batches_per_epoch=3, seed=0)
    loader = BalancedDataLoader(torch.arange(10), batch_sampler=sampler, num_workers=0)
    first = [b.tolist() for b in loader]
    assert len(first) == 3 and loader.state_dict() == {"position": 3}
    second = [b.tolist() for b in loader]
    other = BalancedDataLoader(
        torch.arange(10),
        batch_sampler=BalancedBatchSampler(groups, [1, 2], 3, seed=0),
    )
    other.load_state_dict({"position": 3})
    assert [b.tolist() for b in other] == second


def test_warp_contrast_pivots_on_mid_grey():
    aug = WarpAug(contrast=(3.0, 3.0), contrast_p=1.0)
    crop = np.array([[[0], [64], [127], [128], [200]]], dtype=np.uint8)
    out = aug.intensity(crop, np.random.default_rng(0))
    # (x - 127.5) * 3 + 127.5, clipped (no abs() fold-back of dark pixels).
    assert out[0, :, 0].tolist() == [0, 0, 126, 129, 255]
    bright = WarpAug(brightness=(2.0, 2.0), brightness_p=1.0)
    assert bright.intensity(crop, np.random.default_rng(0))[0, :, 0].tolist() == [
        0,
        128,
        254,
        255,
        255,
    ]


def test_warp_shift_is_a_fraction_of_the_padded_box():
    aug = WarpAug(translate=(0.5, 0.25), translate_p=1.0)
    rng = np.random.default_rng(0)
    shifts = np.array(
        [aug.sample(rng, 64, 48, box_wh=(20.0, 40.0))[0][:2, 2] for _ in range(500)]
    )
    assert np.abs(shifts[:, 0]).max() <= 10.0 and np.abs(shifts[:, 0]).max() > 9.0
    assert np.abs(shifts[:, 1]).max() <= 10.0 and np.abs(shifts[:, 1]).max() > 9.0


def test_warp_dataset_shift_uses_padded_box(tmp_path):
    labels, _ = _labels(tmp_path)
    spec = BBoxCropSpec(padding=0.25, bbox_source="instances")
    ds = WarpCenteredInstanceDataset([labels], (64, 48), _head(), bbox_crop=spec)
    box = ds.table.box[0]
    w, h = box[2] - box[0], box[3] - box[1]
    margin = 2 * 0.25 * max(w, h)
    np.testing.assert_allclose(
        ds.box_wh[0], np.array([w + margin, h + margin]) * ds.scales[0], rtol=1e-5
    )


def test_crop_size_hw_config_validation():
    from sleap_nn.config.data_config import PreprocessingConfig

    cfg = PreprocessingConfig(
        crop_size=[256, 192], crop_mode="bbox", crop_pipeline="warp"
    )
    assert list(cfg.crop_size) == [256, 192]
    PreprocessingConfig(crop_size=128)
    with pytest.raises(ValueError):
        PreprocessingConfig(crop_size=[256, 192], crop_mode="bbox")
    for bad in ([256, 192, 3], [256.5, 192], [256, True], "256"):
        with pytest.raises(ValueError):
            PreprocessingConfig(crop_size=bad, crop_mode="bbox", crop_pipeline="warp")


# --------------------------------------------------------------------------- #
# Train -> predict
# --------------------------------------------------------------------------- #
def _rescaled_file(labels, out_dir, name, hw, nodes):
    """One frame of ``labels`` resized to ``hw`` as a PNG-backed labels file."""
    lf = labels[0]
    img = lf.image[..., 0]
    h, w = hw
    path = str(out_dir / f"{name}.png")
    cv2.imwrite(path, cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
    s = np.array([w / img.shape[1], h / img.shape[0]])
    sk = sio.Skeleton(nodes=list(nodes), edges=[(nodes[0], nodes[1])])
    video = sio.Video.from_filename([path])
    insts = []
    for inst in lf.user_instances:
        pts = (inst.numpy() + 0.5) * s - 0.5
        extra = [pts.mean(axis=0)] * (len(nodes) - 2)
        insts.append(sio.Instance.from_numpy(np.vstack([pts, *extra]), skeleton=sk))
    return sio.Labels(
        [sio.LabeledFrame(video=video, frame_idx=0, instances=insts)],
        skeletons=[sk],
        videos=[video],
    )


@pytest.fixture(scope="module")
def warp_run(tmp_path_factory, pytestconfig):
    """A tiny warp-trained model on two files with different skeletons and sizes."""
    from pathlib import Path

    from sleap_nn.training.model_trainer import ModelTrainer
    from tests.fixtures.datasets import config as config_fixture

    tmp = tmp_path_factory.mktemp("warp_run")
    data_dir = Path(pytestconfig.rootdir) / "tests/assets/datasets"
    pkg = data_dir / "minimal_instance.pkg.slp"
    small = _rescaled_file(sio.load_slp(pkg), tmp, "small", (288, 240), "ABC")
    small_path = str(tmp / "small.slp")
    small.save(small_path)

    cfg = config_fixture.__wrapped__(data_dir).copy()
    cfg.data_config.train_labels_path = [str(pkg), small_path]
    cfg.data_config.val_labels_path = [str(pkg), small_path]
    cfg.data_config.unlabeled_points = "ignore"
    cfg.data_config.augmentation_config.geometric.translate_width = 0.1
    cfg.data_config.augmentation_config.geometric.translate_height = 0.1
    cfg.data_config.augmentation_config.geometric.flip_p = 0.5
    pre = cfg.data_config.preprocessing
    pre.crop_mode = "bbox"
    pre.crop_pipeline = "warp"
    pre.crop_size = [64, 48]
    pre.gpu_confmaps = True
    head = cfg.model_config.head_configs.centered_instance.confmaps
    head.part_names = None
    head.anchor_part = None
    head.centroid_method = "bbox_center"
    tc = cfg.trainer_config
    tc.trainer_accelerator = "cpu"
    tc.max_epochs = 3
    tc.train_steps_per_epoch = 4
    tc.train_data_loader.batch_size = 2
    tc.val_data_loader.batch_size = 2
    tc.balanced_sampler = {"counts": [1, 1]}
    tc.optimizer = {"lr": 1e-3}
    tc.save_ckpt = True
    tc.ckpt_dir = str(tmp)
    tc.run_name = "warp"
    tc.visualize_preds_during_training = True
    tc.eval = {"enabled": True, "frequency": 1}
    ModelTrainer.get_model_trainer_from_config(cfg).train()
    return {"dir": tmp / "warp", "pkg": pkg, "tmp": tmp}


def test_warp_training_saves_merged_skeleton_and_hw_crop(warp_run):
    saved = OmegaConf.load(warp_run["dir"] / "training_config.yaml")
    pre = saved.data_config.preprocessing
    assert list(pre.crop_size) == [64, 48]
    # No size matcher: frames keep their native size.
    assert pre.max_height is None and pre.max_width is None
    head = saved.model_config.head_configs.centered_instance.confmaps
    assert list(head.part_names) == ["A", "B", "C"]
    skeleton = saved.data_config.skeletons[0]
    assert [n["name"] for n in skeleton["nodes"]] == ["A", "B", "C"]
    assert not skeleton["name"].endswith("+")


def test_warp_training_checkpoint_holds_sampler_position(warp_run):
    ckpt = torch.load(
        warp_run["dir"] / "last.ckpt", map_location="cpu", weights_only=False
    )
    states = ckpt["loops"]["fit_loop"]["state_dict"]["combined_loader"]
    # 3 epochs x 4 batches delivered.
    assert states == [{"position": 12}]


def test_warp_model_predicts_mixed_frame_sizes(warp_run):
    """Inference without a size matcher over frames of different sizes."""
    from sleap_nn.inference.predictor import Predictor

    labels = sio.load_slp(warp_run["pkg"])
    big = _rescaled_file(labels, warp_run["tmp"], "big", (384, 384), "ABC")
    small = _rescaled_file(labels, warp_run["tmp"], "small2", (200, 312), "ABC")
    mixed = sio.Labels(
        [*big.labeled_frames, *small.labeled_frames],
        skeletons=big.skeletons,
        videos=[*big.videos, *small.videos],
    )
    for lf in small.labeled_frames:
        for inst in lf.instances:
            inst.skeleton = big.skeletons[0]
    pred = Predictor.from_model_paths(
        [warp_run["dir"].as_posix()], device="cpu", peak_threshold=0.0, batch_size=2
    )
    assert pred.layer.bbox_crop is not None
    assert tuple(pred.layer.crop_size) == (64, 48)
    out = pred.predict(mixed, make_labels=True)
    assert len(out.labeled_frames) == 2
    for lf, (h, w) in zip(out.labeled_frames, [(384, 384), (200, 312)]):
        assert len(lf.instances) == 2
        for inst in lf.instances:
            pts = inst.numpy()
            assert pts.shape == (3, 2)
            finite = pts[np.isfinite(pts).all(axis=1)]
            assert len(finite) > 0
            assert (finite[:, 0] >= -1).all() and (finite[:, 0] <= w).all()
            assert (finite[:, 1] >= -1).all() and (finite[:, 1] <= h).all()
