"""Tests for the warp path's augmentations (half-body, photometric, dropout, stages)."""

import math
import multiprocessing as mp

import cv2
import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from sleap_nn.config.data_config import GeometricConfig
from sleap_nn.data.augmentation import (
    apply_geometric_augmentation,
    apply_intensity_augmentation,
)
from sleap_nn.data.custom_datasets import _get_warp_centered_instance_datasets
from sleap_nn.data.instance_cropping import BBoxCropSpec, bbox_crop_windows
from sleap_nn.data.topdown_warp import (
    WarpAug,
    WarpCenteredInstanceDataset,
    center_size_window,
)
from sleap_nn.training.callbacks import AugmentationStageCallback

COCO = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]  # fmt: skip
# A person facing the camera (their left = image right), 200 x 160 frame.
PTS = np.array(
    [
        [80, 30], [85, 25], [75, 25], [90, 28], [70, 28],
        [100, 50], [60, 50], [108, 75], [52, 75], [112, 98], [48, 98],
        [93, 105], [67, 105], [95, 140], [65, 140], [96, 175], [64, 175],
    ],
    dtype=np.float64,
)  # fmt: skip
UPPER = list(range(11))
LOWER = list(range(11, 17))
CROP = (64, 48)


def _person_labels(tmp_path, pts=PTS, name="p"):
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(
        rng.integers(0, 255, (200, 160, 3), dtype=np.uint8), (5, 5), 2
    )
    path = str(tmp_path / f"{name}.png")
    cv2.imwrite(path, img)
    sk = sio.Skeleton(nodes=list(COCO))
    for a in COCO:
        if a.startswith("left_"):
            sk.add_symmetry(a, "right_" + a[5:])
    video = sio.Video.from_filename([path])
    inst = sio.Instance.from_numpy(np.asarray(pts, np.float64), skeleton=sk)
    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
    return sio.Labels([lf], skeletons=[sk], videos=[video])


def _head():
    return OmegaConf.create(
        {"part_names": COCO, "sigma": 1.5, "output_stride": 2, "anchor_part": None}
    )


def _ds(labels, geo=None, inten=None, seed=0, **kw):
    return WarpCenteredInstanceDataset(
        [labels],
        CROP,
        _head(),
        bbox_crop=BBoxCropSpec(padding=0.125, bbox_source="instances"),
        geometric_aug=geo,
        intensity_aug=inten,
        apply_aug=True,
        return_flip=True,
        gpu_confmaps=True,
        base_seed=seed,
        **kw,
    )


def _inside(k, H, W):
    return (k[:, 0] >= 0) & (k[:, 1] >= 0) & (k[:, 0] <= W - 1) & (k[:, 1] <= H - 1)


def test_center_size_window_matches_bbox_crop_windows():
    boxes = np.array([[10.0, 20.0, 50.0, 100.0], [0.0, 0.0, 200.0, 40.0]])
    for crop in [(64, 48), (32, 32)]:
        w_ref, s_ref, _ = bbox_crop_windows(
            torch.from_numpy(boxes), crop, "square", 0.0, 8.0
        )
        for j, b in enumerate(boxes):
            w, s = center_size_window(
                (b[:2] + b[2:]) / 2, b[2:] - b[:2], crop, "square", 8.0
            )
            np.testing.assert_allclose(w, w_ref[j].numpy(), atol=1e-4)
            np.testing.assert_allclose(s, s_ref[j].numpy(), atol=1e-5)


@pytest.mark.parametrize("upper_prob,half", [(1.0, UPPER), (0.0, LOWER)])
def test_half_body_box_contains_chosen_joints(tmp_path, upper_prob, half):
    labels = _person_labels(tmp_path)
    geo = {"half_body_p": 1.0, "half_body_upper_prob": upper_prob, "rotation_p": 0.0,
           "scale_p": 0.0}  # fmt: skip
    ds = _ds(labels, geo)
    assert ds.aug.lower_idx.tolist() == LOWER
    s = ds[0]
    assert s["half_body"]
    k = s["instance"][0].numpy()
    other = LOWER if half is UPPER else UPPER
    # Every chosen joint is in the crop, inside the 1.5x padded half box.
    assert np.isfinite(k[half]).all() and _inside(k[half], *CROP).all()
    H, W = CROP
    # The half's extent is at most 1/1.5 of the crop, exactly so on the binding axis.
    frac = (k[half].max(0) - k[half].min(0)) / np.array([W, H])
    assert (frac <= 1 / 1.5 + 1e-3).all() and np.isclose(frac.max(), 1 / 1.5, rtol=1e-3)
    # Window center = mean of the chosen joints -> crop center.
    np.testing.assert_allclose(k[half].mean(0), [(W - 1) / 2, (H - 1) / 2], atol=1e-3)
    # Joints of the other half that left the crop are NaN (no target).
    assert np.isnan(k[other]).any()


def test_half_body_needs_enough_joints(tmp_path):
    pts = PTS.copy()
    pts[[0, 1, 2, 3, 4, 7, 8, 9, 10]] = np.nan  # 8 visible < min_total 9
    labels = _person_labels(tmp_path, pts)
    ds = _ds(labels, {"half_body_p": 1.0, "rotation_p": 0.0, "scale_p": 0.0})
    assert not ds[0]["half_body"]
    # With a lower threshold: upper has 2 (shoulders) >= 2, lower has 6.
    ds = _ds(
        labels,
        {"half_body_p": 1.0, "half_body_min_total_keypoints": 8, "rotation_p": 0.0,
         "scale_p": 0.0, "half_body_upper_prob": 1.0},
    )  # fmt: skip
    s = ds[0]
    assert s["half_body"]
    k = s["instance"][0].numpy()
    assert _inside(k[[5, 6]], *CROP).all()


def test_half_body_with_flip_swaps(tmp_path):
    labels = _person_labels(tmp_path)
    base = {"half_body_p": 1.0, "half_body_upper_prob": 1.0, "rotation_p": 0.0,
            "scale_p": 0.0}  # fmt: skip
    a = _ds(labels, base)[0]
    b = _ds(labels, {**base, "flip_p": 1.0})[0]
    assert b["flipped"] and a["half_body"] and b["half_body"]
    ka, kb = a["instance"][0].numpy(), b["instance"][0].numpy()
    W = CROP[1]
    for i, n in enumerate(COCO):
        if n.startswith("left_"):
            j = COCO.index("right_" + n[5:])
            if np.isfinite(ka[j]).all():
                np.testing.assert_allclose(
                    kb[i], [W - 1 - ka[j, 0], ka[j, 1]], atol=1e-3
                )
    torch.testing.assert_close(b["instance_image"], a["instance_image"].flip(-1))


def test_dropout_and_erase_keep_targets(tmp_path):
    labels = _person_labels(tmp_path)
    geo = {"rotation_min": -30.0, "rotation_max": 30.0, "rotation_p": 1.0,
           "flip_p": 0.5}  # fmt: skip
    plain = _ds(labels, geo, seed=3, unlabeled_points="ignore")[0]
    drop = _ds(labels, {**geo, "coarse_dropout_p": 1.0, "erase_p": 1.0,
                        "erase_scale_min": 0.05, "erase_scale_max": 0.1}, seed=3,
               unlabeled_points="ignore")[0]  # fmt: skip
    # Same geometry draws -> same keypoints and weights; the image has a black hole.
    torch.testing.assert_close(drop["instance"], plain["instance"], equal_nan=True)
    torch.testing.assert_close(drop["node_weights"], plain["node_weights"])
    img = drop["instance_image"][0]
    changed = (img != plain["instance_image"][0]).any(0)
    H, W = CROP
    assert changed.float().mean() >= 0.2 * 0.2 * 0.5  # at least ~one hole's worth
    zero = (img == 0).all(0)
    ys, xs = np.nonzero(zero.numpy())
    assert (ys.max() - ys.min() + 1) >= int(0.2 * H) and (
        xs.max() - xs.min() + 1
    ) >= int(0.2 * W)


def test_coarse_dropout_hole_sizes():
    a = WarpAug(coarse_dropout_p=1.0, coarse_dropout_holes=(1, 1),
                coarse_dropout_size=(0.2, 0.4))  # fmt: skip
    rng = np.random.default_rng(0)
    for _ in range(50):
        crop = np.full((64, 48, 3), 200, np.uint8)
        out = a.intensity(crop, rng)
        ys, xs = np.nonzero((out == 0).all(-1))
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        assert int(0.2 * 64) <= h <= int(0.4 * 64) and int(0.2 * 48) <= w <= int(
            0.4 * 48
        )
        assert (out == 0).all(-1).sum() == h * w  # one rectangle


def test_photometric_ops_shapes_and_effects():
    rng = np.random.default_rng(0)
    crop = cv2.GaussianBlur(
        rng.integers(0, 255, (64, 48, 3), dtype=np.uint8), (3, 3), 1
    )
    for kw in [
        {"blur_p": 1.0},
        {"median_blur_p": 1.0},
        {"gaussian_blur_p": 1.0, "gaussian_blur_sigma": (1.0, 1.0)},
        {"uniform_noise_p": 1.0, "uniform_noise": (0.05, 0.1)},
        {"hsv_p": 1.0, "hsv_delta": (10.0, 50.0, 50.0)},
    ]:
        outs = [
            WarpAug(**kw).intensity(crop, np.random.default_rng(s)) for s in range(4)
        ]
        for out in outs:
            assert out.shape == crop.shape and out.dtype == np.uint8
        assert any((o != crop).any() for o in outs), kw
    gray = crop[:, :, :1].copy()
    out = WarpAug(blur_p=1.0, median_blur_p=1.0).intensity(gray, rng)
    assert out.shape == gray.shape


def test_hsv_matches_mmdet_reference():
    delta = (5.0, 30.0, 30.0)
    crop = np.random.default_rng(1).integers(0, 255, (16, 12, 3), dtype=np.uint8)
    for seed in range(10):
        out = WarpAug(hsv_p=1.0, hsv_delta=delta)._hsv(
            crop, np.random.default_rng(seed)
        )
        # mmdet YOLOXHSVRandomAug.transform, on RGB.
        rng = np.random.default_rng(seed)
        gains = rng.uniform(-1, 1, 3) * np.array(delta)
        gains = (gains * rng.integers(0, 2, 3)).astype(np.int16)
        if not gains.any():  # mmdet still round-trips (lossy); we skip the no-op.
            np.testing.assert_array_equal(out, crop)
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + gains[0]) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + gains[1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + gains[2], 0, 255)
        ref = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        np.testing.assert_array_equal(out, ref)


def test_mmpose_affine_sampling():
    a = WarpAug.from_configs(
        {"affine_sampling": "mmpose", "scale_min": 0.6, "scale_max": 1.4,
         "scale_p": 1.0, "rotation_min": -80.0, "rotation_max": 80.0,
         "rotation_p": 1.0},
        None,
    )  # fmt: skip
    rng = np.random.default_rng(0)
    s, th = [], []
    for _ in range(2000):
        A, _ = a.sample(rng, 64, 48)
        s.append(math.hypot(A[0, 0], A[1, 0]))
        th.append(math.degrees(math.atan2(A[1, 0], A[0, 0])))
    s, th = np.array(s), np.array(th)
    assert s.min() >= 1 / 1.4 - 1e-6 and s.max() <= 1 / 0.6 + 1e-6
    assert np.abs(th).max() <= 80 + 1e-6
    # Truncated normal on [-1, 1]: std ~0.54 of the half range (uniform: 0.577).
    assert 0.50 < np.std(th) / 80 < 0.57
    assert abs(np.median(1 / s) - 1.0) < 0.03


def test_deterministic_given_seed(tmp_path):
    labels = _person_labels(tmp_path)
    geo = {"rotation_min": -40.0, "rotation_max": 40.0, "rotation_p": 0.6,
           "scale_min": 0.7, "scale_max": 1.3, "flip_p": 0.5, "half_body_p": 0.5,
           "half_body_min_total_keypoints": 9, "coarse_dropout_p": 0.5,
           "erase_p": 0.5}  # fmt: skip
    inten = {"hsv_p": 1.0, "blur_p": 0.3, "median_blur_p": 0.3}

    def run(seed):
        torch.manual_seed(0)
        ds = _ds(labels, geo, inten, seed=seed)
        return [ds[0] for _ in range(6)]

    a, b, c = run(1), run(1), run(2)
    for x, y in zip(a, b):
        torch.testing.assert_close(x["instance_image"], y["instance_image"])
        torch.testing.assert_close(x["instance"], y["instance"], equal_nan=True)
    assert any(
        not torch.equal(x["instance_image"], y["instance_image"]) for x, y in zip(a, c)
    )


def test_mixup_raises_in_warp_and_validates_in_config(tmp_path):
    labels = _person_labels(tmp_path)
    with pytest.raises(ValueError, match="mixup"):
        _ds(labels, {"mixup_p": 0.5})
    GeometricConfig(mixup_p=0.5)  # warns (loguru), does not raise


def test_stock_pipeline_accepts_warp_only_keys():
    img = torch.randint(0, 255, (1, 3, 32, 32), dtype=torch.uint8)
    inst = torch.rand(1, 1, 2, 2) * 30
    out, _ = apply_geometric_augmentation(
        img, inst, rotation_p=0.0, scale_p=0.0, half_body_p=0.3,
        coarse_dropout_p=1.0, affine_sampling="mmpose",
    )  # fmt: skip
    assert out.shape == img.shape
    out, _ = apply_intensity_augmentation(img, inst, hsv_p=1.0, blur_p=0.1)
    assert out.shape == img.shape


def _stage_ds(labels):
    return _ds(
        labels,
        {"rotation_p": 0.0, "scale_p": 0.0, "flip_p": 0.0},
        stage2_aug=({"rotation_p": 0.0, "scale_p": 0.0, "flip_p": 1.0}, None),
    )


def _first(x):
    return x[0]


@pytest.mark.parametrize(
    "context",
    [
        pytest.param(
            "fork",
            marks=pytest.mark.skipif(
                "fork" not in mp.get_all_start_methods(),
                reason="no fork start method (Windows)",
            ),
        ),
        "spawn",
    ],
)
def test_stage_switch_reaches_persistent_workers(tmp_path, context):
    labels = _person_labels(tmp_path)
    ds = _stage_ds(labels)
    loader = DataLoader(
        ds,
        batch_size=1,
        sampler=list(range(len(ds))) * 4,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=1,
        collate_fn=_first,
        multiprocessing_context=context,
    )
    first = [s["aug_stage"] for s in loader]
    assert first == [0] * 4
    ds.set_aug_stage(1)
    second = list(loader)  # same persistent workers, new epoch
    assert [s["aug_stage"] for s in second] == [1] * 4
    assert all(s["flipped"] for s in second)


def test_augmentation_stage_callback(tmp_path):
    labels = _person_labels(tmp_path)
    ds = _stage_ds(labels)

    class _Trainer:
        current_epoch = 0
        global_step = 0
        train_dataloader = DataLoader(ds)

    cb = AugmentationStageCallback(start_epoch=2)
    t = _Trainer()
    cb.on_train_epoch_start(t, None)
    assert int(ds.aug_stage) == 0 and not ds[0]["flipped"]
    t.current_epoch = 1
    cb.on_train_epoch_start(t, None)
    assert int(ds.aug_stage) == 0
    cb.on_train_epoch_end(t, None)  # switches ahead of epoch 2's prefetch
    assert int(ds.aug_stage) == 1
    t.current_epoch = 2
    cb.on_train_epoch_start(t, None)
    assert int(ds.aug_stage) == 1 and ds[0]["flipped"]
    # Resuming mid-run: stateless rule.
    cb2 = AugmentationStageCallback(start_step=100)
    ds.set_aug_stage(0)
    t.global_step = 150
    cb2.on_train_batch_start(t, None, None, 0)
    assert int(ds.aug_stage) == 1
    with pytest.raises(ValueError):
        AugmentationStageCallback()


def test_stage2_config_merges_over_stage1(tmp_path):
    labels = _person_labels(tmp_path)
    cfg = OmegaConf.create(
        {
            "data_config": {
                "preprocessing": {
                    "crop_size": list(CROP), "scale": 1.0, "crop_mode": "bbox",
                    "crop_bbox_padding": 0.125, "ensure_rgb": True,
                    "ensure_grayscale": False, "gpu_confmaps": True,
                },
                "bbox_source": "instances",
                "use_augmentations_train": True,
                "user_instances_only": True,
                "augmentation_config": {
                    "geometric": {"rotation_min": -80.0, "rotation_max": 80.0,
                                  "rotation_p": 0.6, "coarse_dropout_p": 1.0,
                                  "half_body_p": 0.3},
                    "intensity": {"hsv_p": 1.0},
                    "stage2": {"start_epoch": 3,
                               "geometric": {"rotation_min": -60.0,
                                             "rotation_max": 60.0,
                                             "coarse_dropout_p": 0.5}},
                },
            },
            "model_config": {
                "head_configs": {"centered_instance": {"confmaps": dict(_head())}},
                "backbone_config": {"unet": {"max_stride": 16}},
            },
            "trainer_config": {"seed": 0},
        }
    )  # fmt: skip
    tr, va = _get_warp_centered_instance_datasets([labels], [labels], cfg, "unet", {})
    s1, s2 = tr.augs
    assert s1.rotation == (-80.0, 80.0) and s2.rotation == (-60.0, 60.0)
    assert s1.coarse_dropout_p == 1.0 and s2.coarse_dropout_p == 0.5
    assert s2.half_body_p == 0.3 and s2.hsv_p == 1.0 and s2.rotation_p == 0.6
    assert len(va.augs) == 1 and not va.apply_aug


def test_mmpose_shift_moves_the_box_before_rotation():
    """mmpose shifts the box center, then rotates the crop about it."""
    a = WarpAug(
        affine_sampling="mmpose",
        translate=(0.5, 0.0),
        translate_p=1.0,
        rotation=(90.0, 90.0),
        rotation_p=1.0,
    )
    A, _ = a.sample(np.random.default_rng(0), 64, 64, box_wh=(40.0, 40.0))
    c = np.array([31.5, 31.5, 1.0])
    # Some box point lands on the crop center: the shifted box center, which is
    # off-center along x only (before the 90 degree rotation).
    src = np.linalg.solve(A, c)
    assert abs(src[1] - 31.5) < 1e-6 and 0 < abs(src[0] - 31.5) <= 20.0 + 1e-6
    uniform = WarpAug(translate=(0.5, 0.0), translate_p=1.0, rotation=(90.0, 90.0),
                      rotation_p=1.0)  # fmt: skip
    B, _ = uniform.sample(np.random.default_rng(0), 64, 64, box_wh=(40.0, 40.0))
    # Uniform: the shift is applied after rotation, in crop x.
    center = B @ c
    assert abs(center[1] - 31.5) < 1e-6 and 0 < abs(center[0] - 31.5) <= 20.0 + 1e-6


def test_half_body_shift_uses_the_half_body_box(tmp_path):
    labels = _person_labels(tmp_path)
    ds = _ds(labels, {"half_body_p": 1.0, "half_body_upper_prob": 1.0})
    hb = ds.aug.half_body_window(
        np.random.default_rng(0),
        ds.table.kpts[0],
        np.isfinite(ds.table.kpts[0]).all(-1),
        CROP,
    )
    assert hb is not None
    window, scale, box_wh = hb
    upper = ds.table.kpts[0][UPPER]
    size = (upper.max(0) - upper.min(0)) * 1.5
    np.testing.assert_allclose(box_wh, size * scale, rtol=1e-5)


def _train_cfg(tmp_path, pytestconfig, **geometric):
    from pathlib import Path

    from tests.fixtures.datasets import config as config_fixture

    data_dir = Path(pytestconfig.rootdir) / "tests/assets/datasets"
    cfg = config_fixture.__wrapped__(data_dir).copy()
    pre = cfg.data_config.preprocessing
    pre.crop_mode = "bbox"
    pre.crop_pipeline = "warp"
    pre.crop_size = [48, 40]
    pre.gpu_confmaps = True
    cfg.data_config.unlabeled_points = "ignore"
    geo = cfg.data_config.augmentation_config.geometric
    for key, value in geometric.items():
        geo[key] = value
    head = cfg.model_config.head_configs.centered_instance.confmaps
    head.anchor_part = None
    head.centroid_method = "bbox_center"
    tc = cfg.trainer_config
    tc.trainer_accelerator = "cpu"
    tc.max_epochs = 3
    tc.train_steps_per_epoch = 2
    tc.save_ckpt = True
    tc.ckpt_dir = str(tmp_path)
    tc.run_name = "aug"
    return cfg


def test_stage2_switch_in_training(tmp_path, pytestconfig):
    """A short warp run switches to stage 2 at its start epoch."""
    from sleap_nn.training.model_trainer import ModelTrainer

    cfg = _train_cfg(
        tmp_path,
        pytestconfig,
        affine_sampling="mmpose",
        half_body_p=0.5,
        half_body_min_total_keypoints=2,
        half_body_min_upper_keypoints=1,
        half_body_min_lower_keypoints=1,
        half_body_lower_nodes=["B"],
        coarse_dropout_p=0.5,
        erase_p=0.5,
    )
    cfg.data_config.augmentation_config.intensity.hsv_p = 0.5
    cfg.data_config.augmentation_config.intensity.blur_p = 0.5
    cfg.data_config.augmentation_config.stage2 = {
        "start_epoch": 2,
        "geometric": {"rotation_min": -10.0, "rotation_max": 10.0},
    }
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    callbacks = [
        c for c in trainer.trainer.callbacks if isinstance(c, AugmentationStageCallback)
    ]
    assert len(callbacks) == 1 and callbacks[0].start_epoch == 2
    ds = trainer.trainer.train_dataloader.dataset
    assert len(ds.augs) == 2 and ds.augs[1].rotation == (-10.0, 10.0)
    assert int(ds.aug_stage) == 1


def test_stage2_refused_outside_warp(tmp_path, pytestconfig):
    from sleap_nn.training.model_trainer import ModelTrainer

    cfg = _train_cfg(tmp_path, pytestconfig)
    cfg.data_config.preprocessing.crop_pipeline = "legacy"
    cfg.data_config.preprocessing.crop_size = 48
    cfg.data_config.augmentation_config.stage2 = {"start_epoch": 1}
    with pytest.raises(ValueError, match="stage2"):
        ModelTrainer.get_model_trainer_from_config(cfg).train()
