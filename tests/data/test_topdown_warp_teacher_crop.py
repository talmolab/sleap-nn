"""Tests for the warp path's dual-resolution teacher crop (distillation)."""

import os

import cv2
import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from sleap_nn.data.custom_datasets import (
    _get_warp_centered_instance_datasets,
    get_train_val_datasets,
)
from sleap_nn.data.instance_cropping import BBoxCropSpec
from sleap_nn.data.topdown_warp import WarpAug, WarpCenteredInstanceDataset
from sleap_nn.training.distillation import TEACHER_IMAGE_KEY

NODES = ["nose", "left_hand", "right_hand", "left_foot", "right_foot"]
PTS = np.array([[80, 40], [105, 90], [55, 90], [95, 170], [65, 170]], np.float64)
CROP = (64, 48)

GEO = {
    "rotation_min": -30.0,
    "rotation_max": 30.0,
    "rotation_p": 1.0,
    "scale_min": 0.8,
    "scale_max": 1.2,
    "scale_p": 1.0,
    "translate_width": 0.1,
    "translate_height": 0.1,
    "translate_p": 1.0,
    "flip_p": 0.5,
    "half_body_p": 0.3,
    "half_body_min_total_keypoints": 3,
    "half_body_min_upper_keypoints": 1,
    "half_body_min_lower_keypoints": 1,
    "coarse_dropout_p": 0.7,
    "coarse_dropout_min_holes": 1,
    "coarse_dropout_max_holes": 3,
    "coarse_dropout_min_size": 0.1,
    "coarse_dropout_max_size": 0.3,
    "erase_p": 0.7,
    "erase_scale_min": 0.02,
    "erase_scale_max": 0.1,
}
INTENSITY = {
    "contrast_min": 0.7,
    "contrast_max": 1.3,
    "contrast_p": 0.5,
    "brightness_min": 0.8,
    "brightness_max": 1.2,
    "brightness_p": 0.5,
    "hsv_p": 0.5,
    "gaussian_blur_p": 0.3,
}


def _labels(tmp_path, blob=None):
    """One person in a smooth 200 x 160 RGB frame (+ a white blob at ``blob``)."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 160, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 6)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    if blob is not None:
        yy, xx = np.mgrid[:200, :160]
        g = np.exp(-((xx - blob[0]) ** 2 + (yy - blob[1]) ** 2) / (2 * 4.0**2))
        img = (img * 0.3).astype(np.float64)
        img = np.clip(img + 255 * g[..., None], 0, 255).astype(np.uint8)
    path = str(tmp_path / "frame.png")
    cv2.imwrite(path, img)
    sk = sio.Skeleton(nodes=list(NODES))  # (it converts the list in place)
    sk.add_symmetry("left_hand", "right_hand")
    sk.add_symmetry("left_foot", "right_foot")
    video = sio.Video.from_filename([path])
    inst = sio.Instance.from_numpy(PTS, skeleton=sk)
    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
    return sio.Labels([lf], skeletons=[sk], videos=[video])


def _head():
    return OmegaConf.create(
        {"part_names": NODES, "sigma": 1.5, "output_stride": 2, "anchor_part": None}
    )


def _ds(labels, teacher=None, geo=GEO, inten=INTENSITY, seed=0, **kw):
    return WarpCenteredInstanceDataset(
        [labels],
        CROP,
        _head(),
        bbox_crop=BBoxCropSpec(padding=0.25, bbox_source="instances"),
        geometric_aug=geo,
        intensity_aug=inten,
        apply_aug=True,
        return_flip=True,
        gpu_confmaps=True,
        base_seed=seed,
        teacher_crop_size=teacher,
        **kw,
    )


def _hwc(t):
    return t[0].numpy().transpose(1, 2, 0).astype(np.float32)


def _draws(ds, n):
    out = []
    for s in range(n):
        ds._rng = np.random.default_rng(s)
        ds._teacher_rng = np.random.default_rng(1000 + s)
        ds._pid = os.getpid()
        out.append(ds[0])
    return out


def test_teacher_image_key_is_shared_with_distillation():
    from sleap_nn.data import topdown_warp

    assert TEACHER_IMAGE_KEY is topdown_warp.TEACHER_IMAGE_KEY


@pytest.mark.parametrize("ratio", [2.0, 1.5])
def test_teacher_crop_is_the_student_window(tmp_path, ratio):
    """Downsampled to the student's size, the teacher crop is the student crop,
    with every geometric and photometric augmentation on (flips, half-body,
    dropout holes and erases included)."""
    teacher = (int(CROP[0] * ratio), int(CROP[1] * ratio))
    ds = _ds(_labels(tmp_path), teacher=teacher)
    flips = holes = 0
    for s in _draws(ds, 40):
        st, te = _hwc(s["instance_image"]), _hwc(s[TEACHER_IMAGE_KEY])
        assert te.shape == (*teacher, 3) and s[TEACHER_IMAGE_KEY].dtype == torch.uint8
        down = cv2.resize(te, CROP[::-1], interpolation=cv2.INTER_AREA)
        err = np.abs(down - st)
        # The residual is resampling (bilinear point samples vs. the area mean of
        # finer ones: ~2/255 with no augmentation at all), far below what a
        # one-pixel misregistration of the window gives.
        assert err.mean() < 3.5, err.mean()
        inner = (slice(1, -1), slice(1, -1))
        aligned = np.abs(down[inner] - st[inner]).mean()
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            moved = np.roll(down, (dy, dx), axis=(0, 1))
            assert aligned < 0.5 * np.abs(moved[inner] - st[inner]).mean()
        # Dropout / erase boxes cover the same part of the window.
        zero = (st == 0).all(-1)
        if zero.sum() > 20:
            holes += 1
            core = cv2.erode(zero.astype(np.uint8), np.ones((3, 3), np.uint8))
            assert (down[core.astype(bool)] < 1.0).all()
        flips += bool(s["flipped"])
    assert flips and holes  # the draws exercised flips and holes


def test_teacher_crop_keypoint_geometry(tmp_path):
    """A blob on the nose lands at ``(k + 0.5) * r - 0.5`` in the teacher crop,
    where ``k`` is the nose in the student crop, under rotation / scale / shift /
    flip."""
    geo = {k: v for k, v in GEO.items() if not k.startswith(("coarse", "erase"))}
    geo["half_body_p"] = 0.0
    ds = _ds(_labels(tmp_path, blob=PTS[0]), teacher=(128, 96), geo=geo, inten=None)
    r = 2.0
    for s in _draws(ds, 20):
        k = s["instance"][0, 0].numpy()  # the nose (its own flip partner)
        if not np.isfinite(k).all():
            continue
        te = _hwc(s[TEACHER_IMAGE_KEY]).mean(-1)
        w = np.clip(te - 0.6 * te.max(), 0, None)
        yy, xx = np.mgrid[: te.shape[0], : te.shape[1]]
        peak = np.array([(w * xx).sum(), (w * yy).sum()]) / w.sum()
        np.testing.assert_allclose(peak, (k + 0.5) * r - 0.5, atol=1.0)


def test_student_samples_unchanged_by_teacher_crop(tmp_path):
    """With a teacher crop the student's sample is byte-identical (noise on too),
    and without one there is no `teacher_image`."""
    labels = _labels(tmp_path)
    inten = dict(INTENSITY, uniform_noise_p=0.5, gaussian_noise_p=0.5)
    inten.update(gaussian_noise_std=0.05, median_blur_p=0.3)
    inten["blur_p"] = 0.3
    plain = _draws(_ds(labels, inten=inten), 15)
    dual = _draws(_ds(labels, teacher=(96, 72), inten=inten), 15)
    for a, b in zip(plain, dual):
        assert TEACHER_IMAGE_KEY not in a and TEACHER_IMAGE_KEY in b
        assert set(b) - set(a) == {TEACHER_IMAGE_KEY}
        for key in a:
            if torch.is_tensor(a[key]):
                torch.testing.assert_close(
                    a[key], b[key], rtol=0, atol=0, equal_nan=True, msg=key
                )
            else:
                assert a[key] == b[key], key


def test_teacher_crop_without_augmentation_is_the_inference_window(tmp_path):
    """Un-augmented, the teacher crop is the box window cut at the teacher's size."""
    labels = _labels(tmp_path)
    ds = _ds(labels, teacher=(96, 72))
    ds.apply_aug = False
    big = WarpCenteredInstanceDataset(
        [labels],
        (96, 72),
        _head(),
        bbox_crop=BBoxCropSpec(padding=0.25, bbox_source="instances"),
    )
    assert torch.equal(ds[0][TEACHER_IMAGE_KEY], big[0]["instance_image"])


def test_teacher_photometry_maps_spatial_draws():
    """Holes, erase boxes and blur kernels scale to the teacher grid; LUT ops are
    identical."""
    rng = np.random.default_rng(3)
    crop = np.full((40, 30, 3), 128, np.uint8)
    teacher = np.full((80, 60, 3), 128, np.uint8)
    aug = WarpAug(coarse_dropout_p=1.0, coarse_dropout_size=(0.2, 0.4))
    st, te = aug.intensity(crop, rng, teacher, np.random.default_rng(0))
    zs, zt = (st == 0).all(-1), (te == 0).all(-1)
    assert zs.any() and zt.sum() == 4 * zs.sum()
    np.testing.assert_array_equal(zt[::2, ::2], zs)
    aug = WarpAug(contrast=(1.5, 1.5), contrast_p=1.0)
    st, te = aug.intensity(crop + 10, rng, teacher + 10)
    assert st[0, 0, 0] == te[0, 0, 0] != 138


def test_teacher_erase_that_does_not_fit_leaves_both_crops(tmp_path):
    crop = np.full((40, 30, 3), 128, np.uint8)
    teacher = np.full((80, 60, 3), 128, np.uint8)
    aug = WarpAug(erase_p=1.0, erase_scale=(0.9, 0.9), erase_ratio=(4.0, 4.0))
    st, te = aug.intensity(crop, np.random.default_rng(0), teacher)
    np.testing.assert_array_equal(st, crop)
    np.testing.assert_array_equal(te, teacher)


@pytest.mark.parametrize(
    "gray_frames,kw,channels",
    [(False, {"ensure_grayscale": True}, 1), (True, {"ensure_rgb": True}, 3)],
)
def test_teacher_crop_follows_the_channel_settings(tmp_path, gray_frames, kw, channels):
    labels = _labels(tmp_path)
    if gray_frames:
        path = labels.videos[0].filename[0]
        cv2.imwrite(path, cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    s = _draws(_ds(labels, teacher=(96, 72), **kw), 1)[0]
    assert s["instance_image"].shape == (1, channels, 64, 48)
    assert s[TEACHER_IMAGE_KEY].shape == (1, channels, 96, 72)
    down = cv2.resize(_hwc(s[TEACHER_IMAGE_KEY]), (48, 64), cv2.INTER_AREA)
    st = _hwc(s["instance_image"])
    assert np.abs(down.reshape(st.shape) - st).mean() < 4.0


def test_teacher_crop_size_validation(tmp_path):
    labels = _labels(tmp_path)
    for bad in [(96, 96), 64, (0, 0)]:  # an int is square, unlike the 64x48 crop
        with pytest.raises(ValueError, match="aspect ratio"):
            _ds(labels, teacher=bad)
    assert _ds(labels, teacher=(32, 24)).teacher_hw == (32, 24)


def test_teacher_crop_collates_through_a_dataloader(tmp_path):
    ds = _ds(_labels(tmp_path), teacher=(96, 72))
    batch = next(iter(DataLoader(ds, batch_size=3, sampler=[0, 0, 0])))
    assert batch[TEACHER_IMAGE_KEY].shape == (3, 1, 3, 96, 72)
    assert batch["instance_image"].shape == (3, 1, 3, 64, 48)


# ------------------------------------------------------------------ config


def _cfg(pytestconfig):
    from pathlib import Path

    from tests.fixtures.datasets import config as config_fixture

    data_dir = Path(pytestconfig.rootdir) / "tests/assets/datasets"
    cfg = config_fixture.__wrapped__(data_dir).copy()
    pre = cfg.data_config.preprocessing
    pre.crop_mode = "bbox"
    pre.crop_pipeline = "warp"
    pre.crop_size = [64, 48]
    head = cfg.model_config.head_configs.centered_instance.confmaps
    head.anchor_part = None
    return cfg


def _minimal_labels(pytestconfig):
    return sio.load_slp(
        f"{pytestconfig.rootdir}/tests/assets/datasets/minimal_instance.pkg.slp"
    )


def test_builder_gives_the_teacher_crop_to_training_only(pytestconfig):
    cfg, labels = _cfg(pytestconfig), _minimal_labels(pytestconfig)
    train, val = get_train_val_datasets(
        [labels], [labels], cfg, teacher_crop_size=[96, 72]
    )
    assert train.teacher_hw == (96, 72) and val.teacher_hw is None
    assert TEACHER_IMAGE_KEY in train[0] and TEACHER_IMAGE_KEY not in val[0]
    # Only what the caller passes counts: a distillation config alone (as the
    # visualization datasets see it) adds no teacher crop.
    cfg.trainer_config.distillation = {"teacher": "t", "teacher_crop_size": [96, 72]}
    for train in (
        get_train_val_datasets([labels], [labels], cfg)[0],
        _get_warp_centered_instance_datasets([labels], [labels], cfg, "unet", {})[0],
    ):
        assert train.teacher_hw is None and TEACHER_IMAGE_KEY not in train[0]


def test_teacher_crop_of_embedded_gray_frames_with_ensure_rgb(pytestconfig):
    """Frames read through the labels (an embedded grayscale .pkg.slp)."""
    cfg, labels = _cfg(pytestconfig), _minimal_labels(pytestconfig)
    cfg.data_config.preprocessing.ensure_rgb = True
    train, _ = get_train_val_datasets(
        [labels], [labels], cfg, teacher_crop_size=[96, 72]
    )
    s = train[0]
    assert s["instance_image"].shape == (1, 3, 64, 48)
    assert s[TEACHER_IMAGE_KEY].shape == (1, 3, 96, 72)


def test_teacher_crop_size_needs_the_warp_path(pytestconfig):
    cfg, labels = _cfg(pytestconfig), _minimal_labels(pytestconfig)
    cfg.data_config.preprocessing.crop_pipeline = "legacy"
    cfg.data_config.preprocessing.crop_size = 96
    with pytest.raises(ValueError, match="crop_pipeline='warp'"):
        get_train_val_datasets([labels], [labels], cfg, teacher_crop_size=144)
