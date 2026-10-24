"""Tests for detector-box crops in the warp data path (`crop_bbox_source`, jitter)."""

import cv2
import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf

from sleap_nn.data.instance_cropping import BBoxCropSpec
from sleap_nn.data.topdown_warp import (
    WarpCenteredInstanceDataset,
    build_instance_table,
)

NODES = ["head", "left_hand", "right_hand", "tail"]
PTS = np.array([[60.0, 30.0], [90.0, 50.0], [40.0, 55.0], [65.0, 90.0]])


def _labels(tmp_path):
    """One frame, two people: each with a GT person box; detector boxes added later."""
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(
        rng.integers(0, 255, (120, 160, 3), dtype=np.uint8), (5, 5), 2
    )
    path = str(tmp_path / "a.png")
    cv2.imwrite(path, img)
    sk = sio.Skeleton(nodes=list(NODES), edges=[("head", "tail")])
    sk.add_symmetry("left_hand", "right_hand")
    video = sio.Video.from_filename([path])
    a = sio.Instance.from_numpy(PTS, skeleton=sk)
    b = sio.Instance.from_numpy(PTS * 0.5 + 70, skeleton=sk)
    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[a, b])
    lf.bboxes.append(sio.UserBoundingBox(35, 25, 95, 95, instance=a, category="person"))
    lf.bboxes.append(
        sio.UserBoundingBox(85, 80, 120, 120, instance=b, category="person")
    )
    return sio.Labels([lf], skeletons=[sk], videos=[video]), a, b


def _det(inst, xyxy, score=0.9, source="det", category="person"):
    return sio.PredictedBoundingBox(
        *xyxy, instance=inst, category=category, score=score, source=source
    )


def _table(labels, **kw):
    return build_instance_table(
        [labels], NODES, bbox_source="bboxes", bbox_categories=["person"], **kw
    )


def test_gt_source_ignores_predicted_boxes(tmp_path):
    labels, a, _ = _labels(tmp_path)
    ref = _table(labels)
    labels[0].bboxes.append(_det(a, (10, 10, 100, 110)))
    labels[0].bboxes.append(_det(None, (0, 0, 50, 50)))
    t = _table(labels)
    np.testing.assert_array_equal(t.box, ref.box)
    np.testing.assert_array_equal(t.box[0], [35, 25, 95, 95])
    assert t.box_sources == {"gt": 2}


def test_detector_source_uses_linked_box_and_falls_back(tmp_path):
    labels, a, _ = _labels(tmp_path)
    lf = labels[0]
    lf.bboxes.append(_det(a, (30, 20, 100, 100), score=0.6))
    lf.bboxes.append(_det(a, (32, 22, 98, 97), score=0.8))  # highest score wins
    lf.bboxes.append(_det(None, (0, 0, 50, 50), score=0.99))  # unlinked: ignored
    t = _table(labels, crop_bbox_source="detector")
    np.testing.assert_array_equal(t.box[0], [32, 22, 98, 97])
    np.testing.assert_array_equal(t.box[1], [85, 80, 120, 120])  # GT fallback
    assert t.box_sources == {"gt": 1, "detector": 1}
    assert t.filtered["no_detector_box"] == 0

    t = _table(labels, crop_bbox_source="detector", crop_bbox_fallback="drop")
    assert len(t) == 1 and t.filtered["no_detector_box"] == 1
    np.testing.assert_array_equal(t.box[0], [32, 22, 98, 97])


def test_detector_source_filters(tmp_path):
    labels, a, b = _labels(tmp_path)
    lf = labels[0]
    lf.bboxes.append(_det(a, (30, 20, 100, 100), score=0.4, source="x"))
    lf.bboxes.append(_det(b, (80, 75, 125, 125), score=0.9, source="y"))
    lf.bboxes.append(_det(b, (81, 76, 124, 124), score=0.95, category="head"))
    # min score: a's box is too weak -> GT.
    t = _table(labels, crop_bbox_source="detector", crop_bbox_min_score=0.5)
    np.testing.assert_array_equal(t.box[0], [35, 25, 95, 95])
    np.testing.assert_array_equal(t.box[1], [80, 75, 125, 125])  # not the head box
    # detector source string.
    t = _table(labels, crop_bbox_source="detector", crop_bbox_detector="x")
    np.testing.assert_array_equal(t.box[0], [30, 20, 100, 100])
    np.testing.assert_array_equal(t.box[1], [85, 80, 120, 120])
    with pytest.raises(ValueError):
        _table(labels, crop_bbox_source="predicted")
    with pytest.raises(ValueError):
        _table(labels, crop_bbox_source="detector", crop_bbox_fallback="keypoints")


def test_detector_box_crop_is_the_window_of_that_box(tmp_path):
    labels, a, _ = _labels(tmp_path)
    labels[0].bboxes.append(_det(a, (20, 15, 110, 105)))
    spec = BBoxCropSpec(padding=0.1, bbox_source="bboxes")
    head = OmegaConf.create(
        {"part_names": NODES, "sigma": 1.5, "output_stride": 2, "anchor_part": None}
    )
    ds = WarpCenteredInstanceDataset(
        [labels],
        (64, 48),
        head,
        bbox_crop=spec,
        bbox_categories=["person"],
        crop_bbox_source="detector",
    )
    win, _, _ = spec.windows(torch.tensor([[20.0, 15.0, 110.0, 105.0]]), (64, 48))
    np.testing.assert_allclose(ds.windows[0], win[0].numpy())
    s = ds[0]
    assert s["instance_image"].shape == (1, 3, 64, 48)
    torch.testing.assert_close(s["crop_window"], win)


def _jitter_ds(labels, jitter, apply_aug):
    head = OmegaConf.create(
        {"part_names": NODES, "sigma": 1.5, "output_stride": 2, "anchor_part": None}
    )
    return WarpCenteredInstanceDataset(
        [labels],
        (64, 48),
        head,
        bbox_crop=BBoxCropSpec(padding=0.1, bbox_source="bboxes"),
        bbox_categories=["person"],
        apply_aug=apply_aug,
        crop_bbox_jitter=jitter,
    )


def test_box_jitter_train_only(tmp_path):
    labels, _, _ = _labels(tmp_path)
    ref = _jitter_ds(labels, 0.0, apply_aug=False)[0]
    # Validation never jitters.
    val = _jitter_ds(labels, 0.2, apply_aug=False)
    assert val.crop_bbox_jitter == 0.0
    torch.testing.assert_close(val[0]["instance"], ref["instance"])
    # Training without augmentation keys and no jitter is the plain crop.
    plain = _jitter_ds(labels, 0.0, apply_aug=True)
    torch.testing.assert_close(plain[0]["instance"], ref["instance"])
    # Jitter moves the crop, and keypoints follow the image.
    ds = _jitter_ds(labels, 0.1, apply_aug=True)
    outs = torch.stack([ds[0]["instance"] for _ in range(20)]).numpy()
    # (Points a jittered crop pushes out are NaN.)
    assert np.nanmax(np.nanstd(outs, axis=0)) > 0.5
    with pytest.raises(ValueError):
        _jitter_ds(labels, -0.1, apply_aug=True)


def test_jittered_window_follows_the_window_rule(tmp_path):
    labels, _, _ = _labels(tmp_path)
    ds = _jitter_ds(labels, 1e-9, apply_aug=True)
    win, sc, wh = ds._jittered_window(0, np.random.default_rng(0))
    np.testing.assert_allclose(win, ds.windows[0], atol=1e-3)
    np.testing.assert_allclose(sc, ds.scales[0], rtol=1e-5)
    np.testing.assert_allclose(wh, ds.box_wh[0], rtol=1e-5)
    # A huge jitter never produces an inverted or empty box.
    ds = _jitter_ds(labels, 5.0, apply_aug=True)
    rng = np.random.default_rng(1)
    for _ in range(50):
        win, sc, _ = ds._jittered_window(0, rng)
        assert (win[2:] > win[:2]).all() and np.isfinite(sc).all()


def test_crop_bbox_source_config_validation():
    from sleap_nn.config.data_config import PreprocessingConfig

    cfg = PreprocessingConfig()
    assert cfg.crop_bbox_source == "gt" and cfg.crop_bbox_jitter == 0.0
    assert cfg.crop_bbox_fallback == "gt" and cfg.crop_bbox_detector is None
    PreprocessingConfig(
        crop_mode="bbox",
        crop_pipeline="warp",
        crop_bbox_source="detector",
        crop_bbox_jitter=0.05,
    )
    with pytest.raises(ValueError):
        PreprocessingConfig(crop_mode="bbox", crop_bbox_source="detector")
    with pytest.raises(ValueError):
        PreprocessingConfig(crop_mode="bbox", crop_bbox_jitter=0.1)
    with pytest.raises(ValueError):
        PreprocessingConfig(
            crop_mode="bbox", crop_pipeline="warp", crop_bbox_source="pred"
        )
    with pytest.raises(ValueError):
        PreprocessingConfig(
            crop_mode="bbox", crop_pipeline="warp", crop_bbox_fallback="none"
        )


def test_config_keys_reach_the_warp_datasets(tmp_path, pytestconfig):
    from pathlib import Path

    from sleap_nn.data.custom_datasets import _get_warp_centered_instance_datasets
    from tests.fixtures.datasets import config as config_fixture

    labels, a, _ = _labels(tmp_path)
    labels[0].bboxes.append(_det(a, (20, 15, 110, 105)))
    data_dir = Path(pytestconfig.rootdir) / "tests/assets/datasets"
    cfg = config_fixture.__wrapped__(data_dir).copy()
    pre = cfg.data_config.preprocessing
    pre.crop_mode = "bbox"
    pre.crop_pipeline = "warp"
    pre.crop_size = [64, 48]
    pre.crop_bbox_categories = ["person"]
    pre.crop_bbox_source = "detector"
    pre.crop_bbox_fallback = "drop"
    pre.crop_bbox_jitter = 0.05
    cfg.data_config.bbox_source = "bboxes"
    head = cfg.model_config.head_configs.centered_instance.confmaps
    head.part_names = NODES
    head.anchor_part = None
    train, val = _get_warp_centered_instance_datasets(
        [labels], [labels], cfg, "unet", {}
    )
    for ds in (train, val):
        assert len(ds) == 1
        assert ds.table.box_sources == {"gt": 0, "detector": 1}
        np.testing.assert_array_equal(ds.table.box[0], [20, 15, 110, 105])
    assert train.crop_bbox_jitter == 0.05 and val.crop_bbox_jitter == 0.0
