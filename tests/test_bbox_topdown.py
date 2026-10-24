"""Bounding-box detectors as the first stage of top-down models.

Covers the box-scaled crop geometry shared by training and inference
(``preprocessing.crop_mode="bbox"``), the centered-instance datasets that cut those
crops, the trainer's crop sizing, and the detector -> crop model pipelines
(``Predictor.from_model_paths([detector, centered_instance])``), including the
GT-box fallback and the combinations that must be refused.
"""

import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf

from sleap_nn.config.get_config import get_head_configs
from sleap_nn.data.bboxes import instance_crop_bbox
from sleap_nn.data.custom_datasets import CenteredInstanceDataset
from sleap_nn.data.instance_cropping import (
    BBoxCropSpec,
    bbox_crop_windows,
    crop_windows,
    letterbox_crops,
    points_from_crop,
    points_to_crop,
    pre_crop_size,
    window_corners,
)
from sleap_nn.data.skia_augmentation import crop_and_resize_skia
from sleap_nn.inference.layers.bbox import BBoxStageLayer, gt_boxes_from_instances


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_bbox_crop_windows_square_stretch_and_floor():
    """Windows: padded, squared (or per-axis), centered, floored at min_size."""
    boxes = torch.tensor([[10.0, 20.0, 50.0, 40.0], [5.0, 5.0, 5.0, 5.0]])
    win, scale, keep = bbox_crop_windows(boxes, 64, "square", 0.25, min_size=16)
    # Square on the padded longest side: 40 + 2 * 0.25 * 40 = 60.
    np.testing.assert_allclose((win[0, 2:] - win[0, :2]).numpy(), [60, 60])
    np.testing.assert_allclose(scale[0].numpy(), [64 / 60, 64 / 60])
    # Padded box (60 x 40) kept around the crop center, in crop pixels.
    np.testing.assert_allclose(keep[0].numpy(), [32, 40 * 64 / 60 / 2], rtol=1e-6)
    # Centered on the box center (edge coordinates = keypoint + 0.5).
    np.testing.assert_allclose(((win[0, :2] + win[0, 2:]) / 2).numpy(), [30.5, 30.5])
    # A degenerate (single-point) box is floored at min_size.
    np.testing.assert_allclose((win[1, 2:] - win[1, :2]).numpy(), [16, 16])

    win, scale, _ = bbox_crop_windows(boxes[:1], 64, "stretch", 0.25)
    np.testing.assert_allclose((win[0, 2:] - win[0, :2]).numpy(), [60, 40])
    np.testing.assert_allclose(scale[0].numpy(), [64 / 60, 64 / 40])
    with pytest.raises(ValueError, match="crop_bbox_aspect"):
        bbox_crop_windows(boxes, 64, "diagonal")


@pytest.mark.parametrize("aspect", ["square", "stretch"])
def test_points_crop_round_trip(aspect):
    """Keypoints map into the crop and back exactly."""
    boxes = torch.tensor([[10.0, 20.0, 50.0, 40.0]])
    win, scale, _ = bbox_crop_windows(boxes, 48, aspect, 0.1)
    pts = torch.tensor([[[10.0, 20.0], [50.0, 40.0], [30.0, 30.0]]])
    crop = points_to_crop(pts, win, scale)
    # The box center lands on the crop center.
    np.testing.assert_allclose(crop[0, 2].numpy(), [23.5, 23.5], atol=1e-5)
    np.testing.assert_allclose(
        points_from_crop(crop, win, scale).numpy(), pts.numpy(), atol=1e-5
    )


def test_inference_crops_match_skia_training_crops():
    """roi_align crops (inference) == Skia crops (training), frame edges included."""
    torch.manual_seed(0)
    image = (torch.rand(1, 1, 120, 160) * 255).to(torch.uint8)
    boxes = torch.tensor(
        [[40.0, 30.0, 90.0, 60.0], [2.0, 2.0, 12.0, 50.0], [140.0, 100.0, 159.0, 119.0]]
    )
    for aspect in ("square", "stretch"):
        for size in (32, 47):
            win, _, _ = bbox_crop_windows(boxes, size, aspect, 0.2)
            ours = crop_windows(image, win, torch.zeros(3, dtype=torch.long), size)
            for i in range(3):
                ref = crop_and_resize_skia(
                    image, window_corners(win[i : i + 1]), (size, size)
                )
                assert (ref.float() - ours[i : i + 1]).abs().max() <= 2.0


def test_letterbox_and_pre_crop_size():
    """Letterbox keeps the centered box; pre-crops keep the crop's parity."""
    crops = torch.ones(1, 1, 10, 10)
    out = letterbox_crops(crops, torch.tensor([[5.0, 2.0]]))
    assert out[0, 0].sum() == 10 * 4  # full width, the 4 central rows
    assert out[0, 0, 5].sum() == 10 and out[0, 0, 0].sum() == 0
    for c in (64, 65, 96, 191):
        m = pre_crop_size(c)
        assert m >= c * np.sqrt(2) and (m - c) % 2 == 0


def test_bbox_crop_spec_from_config():
    """The crop spec is read from a config (None for fixed crops)."""
    cfg = OmegaConf.create({"data_config": {"preprocessing": {"crop_mode": "fixed"}}})
    assert BBoxCropSpec.from_config(cfg) is None
    # Configs written before the field existed are fixed-crop.
    assert BBoxCropSpec.from_config(OmegaConf.create({"data_config": {}})) is None
    cfg = OmegaConf.create(
        {
            "data_config": {
                "bbox_padding": 3.0,
                "preprocessing": {
                    "crop_mode": "bbox",
                    "crop_bbox_aspect": "letterbox",
                    "crop_bbox_padding": 0.2,
                    "crop_bbox_min_size": 12,
                },
            }
        }
    )
    spec = BBoxCropSpec.from_config(cfg)
    assert spec == BBoxCropSpec("letterbox", 0.2, 12.0, "auto", 3.0)


# --------------------------------------------------------------------------- #
# Training boxes
# --------------------------------------------------------------------------- #
def test_instance_crop_bbox_sources(minimal_instance):
    """Training boxes follow the detector's GT rule per bbox_source."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    inst = lf.instances[0]
    tight = np.asarray(inst.to_bbox(mode="tight", padding=2.0).xyxy)
    np.testing.assert_allclose(instance_crop_bbox(lf, inst, "instances", 2.0), tight)
    # A user box linked to the instance wins (as is, like a detector's GT) ...
    lf.bboxes = [sio.UserBoundingBox(x1=1, y1=2, x2=30, y2=40, instance=inst)]
    np.testing.assert_allclose(
        instance_crop_bbox(lf, inst, "bboxes", 2.0), [1, 2, 30, 40]
    )
    # ... and an instance without one falls back to its keypoints.
    other = lf.instances[1]
    np.testing.assert_allclose(
        instance_crop_bbox(lf, other, "bboxes", 0.0),
        np.asarray(other.to_bbox(mode="tight").xyxy),
    )


def test_gt_boxes_from_instances_match_sleap_io(minimal_instance):
    """GT-box fallback boxes equal sleap-io's tight padded boxes."""
    labels = sio.load_slp(minimal_instance)
    insts = labels[0].instances
    pts = torch.full((1, 3, 2, 2), float("nan"))
    for i, inst in enumerate(insts):
        pts[0, i] = torch.from_numpy(inst.numpy())
    boxes, scores, classes = gt_boxes_from_instances(pts, padding=4.0)
    for i, inst in enumerate(insts):
        np.testing.assert_allclose(
            boxes[0, i].numpy(), inst.to_bbox(mode="tight", padding=4.0).xyxy, rtol=1e-6
        )
    assert torch.isnan(boxes[0, 2]).all() and torch.isnan(scores[0, 2])
    assert classes[0].tolist() == [0, 0, -1]


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def _head():
    return OmegaConf.create(
        {
            "part_names": ["A", "B"],
            "anchor_part": None,
            "centroid_method": "bbox_center",
            "sigma": 1.5,
            "output_stride": 2,
        }
    )


@pytest.mark.parametrize("aspect", ["square", "letterbox", "stretch"])
def test_dataset_crop_equals_inference_crop(minimal_instance, aspect):
    """Without augmentation a training crop IS the inference crop of its box."""
    labels = sio.load_slp(minimal_instance)
    spec = BBoxCropSpec(aspect=aspect, padding=0.1, bbox_padding=1.0)
    ds = CenteredInstanceDataset(
        labels=[labels],
        crop_size=64,
        confmap_head_config=_head(),
        max_stride=8,
        max_hw=(384, 384),
        bbox_crop=spec,
    )
    for index in range(len(ds)):
        sample = ds[index]
        meta = ds.instance_idx_list[index]
        lf = labels[meta["lf_idx"]]
        inst = lf.instances[meta["inst_idx"]]
        np.testing.assert_allclose(
            meta["bbox"], inst.to_bbox(mode="tight", padding=1.0).xyxy, rtol=1e-6
        )
        frame = torch.from_numpy(lf.image).permute(2, 0, 1)[None]
        win, scale, keep = spec.windows(torch.as_tensor(meta["bbox"])[None], 64)
        ref = crop_windows(frame, win, torch.zeros(1, dtype=torch.long), 64)
        if aspect == "letterbox":
            ref = letterbox_crops(ref, keep)
        assert sample["instance_image"].shape == (1, 1, 64, 64)
        assert (sample["instance_image"][0].float() - ref[0]).abs().max() <= 1.0
        kps = points_to_crop(torch.from_numpy(inst.numpy()).float()[None], win, scale)
        np.testing.assert_allclose(sample["instance"].numpy(), kps.numpy(), atol=1e-4)
        np.testing.assert_allclose(sample["crop_window"].numpy(), win.numpy())
        assert sample["confidence_maps"].shape == (1, 2, 32, 32)
        if aspect == "letterbox":
            outside = torch.ones(1, 1, 64, 64) - letterbox_crops(
                torch.ones(1, 1, 64, 64), keep
            )
            assert (sample["instance_image"].float() * outside).sum() == 0


def test_fixed_crop_dataset_unchanged(minimal_instance):
    """Fixed crops keep their sample layout (no box-crop keys)."""
    labels = sio.load_slp(minimal_instance)
    ds = CenteredInstanceDataset(
        labels=[labels], crop_size=64, confmap_head_config=_head(), max_stride=8
    )
    sample = ds[0]
    assert "crop_window" not in sample and "bbox" not in ds.instance_idx_list[0]
    assert sample["instance_image"].shape == (1, 1, 64, 64)


def test_dataset_augmented_bbox_crop_runs(minimal_instance):
    """Augmented box-scaled samples keep their shape and targets."""
    labels = sio.load_slp(minimal_instance)
    ds = CenteredInstanceDataset(
        labels=[labels],
        crop_size=48,
        confmap_head_config=_head(),
        max_stride=8,
        apply_aug=True,
        geometric_aug={
            "rotation_min": -180.0,
            "rotation_max": 180.0,
            "rotation_p": 1.0,
        },
        intensity_aug={"contrast_p": 1.0},
        bbox_crop=BBoxCropSpec(),
    )
    sample = ds[0]
    assert sample["instance_image"].shape == (1, 1, 48, 48)
    assert torch.isfinite(sample["confidence_maps"]).all()


# --------------------------------------------------------------------------- #
# Train -> predict -> eval
# --------------------------------------------------------------------------- #
def _train_config(config, tmp_path, model_type, run_name, **pre):
    """Tiny CPU training config for one model of the pair."""
    cfg = config.copy()
    cfg.trainer_config.trainer_accelerator = "cpu"
    cfg.trainer_config.max_epochs = 2
    cfg.trainer_config.min_train_steps_per_epoch = 2
    cfg.trainer_config.save_ckpt = True
    cfg.trainer_config.ckpt_dir = str(tmp_path)
    cfg.trainer_config.run_name = run_name
    cfg.trainer_config.visualize_preds_during_training = False
    if model_type in ("bbox_centernet", "bbox_ssd"):
        # Long enough for real boxes on the one-frame fixture (seconds on CPU).
        cfg.trainer_config.max_epochs = 30
        cfg.trainer_config.min_train_steps_per_epoch = 4
        cfg.trainer_config.optimizer = {"lr": 1e-3}
        cfg.data_config.preprocessing.crop_size = None
        cfg.model_config.backbone_config.unet.max_stride = 32
        cfg.model_config.backbone_config.unet.output_stride = 4
        cfg.model_config.head_configs = OmegaConf.structured(
            get_head_configs(model_type)
        )
        if model_type == "bbox_ssd":
            cfg.model_config.head_configs.bbox_ssd.ssd.output_stride = 8
    else:
        cfg.trainer_config.max_epochs = 10
        cfg.trainer_config.min_train_steps_per_epoch = 4
        cfg.trainer_config.optimizer = {"lr": 1e-3}
        cfg.model_config.head_configs.centered_instance.confmaps.anchor_part = None
        cfg.model_config.head_configs.centered_instance.confmaps.centroid_method = (
            "bbox_center"
        )
        cfg.trainer_config.eval = {"enabled": True, "frequency": 1}
    for key, value in pre.items():
        cfg.data_config.preprocessing[key] = value
    return cfg


@pytest.fixture(scope="module")
def trained_pair(tmp_path_factory, pytestconfig):
    """A CenterNet detector + box-scaled and fixed-crop centered-instance models."""
    from pathlib import Path

    from sleap_nn.training.model_trainer import ModelTrainer
    from tests.fixtures.datasets import config as config_fixture

    tmp_path = tmp_path_factory.mktemp("bbox_topdown")
    data_dir = Path(pytestconfig.rootdir) / "tests/assets/datasets"
    base = config_fixture.__wrapped__(data_dir)
    dirs = {}
    for name, model_type, pre in (
        ("det", "bbox_centernet", {}),
        (
            "ci_bbox",
            "centered_instance",
            {"crop_mode": "bbox", "crop_size": None, "crop_bbox_aspect": "square"},
        ),
        ("ci_fixed", "centered_instance", {}),
    ):
        cfg = _train_config(base, tmp_path, model_type, name, **pre)
        trainer = ModelTrainer.get_model_trainer_from_config(cfg)
        trainer.train()
        dirs[name] = tmp_path / name
    return dirs


def test_trainer_sizes_box_crops(trained_pair, minimal_instance):
    """crop_size / crop_bbox_min_size are auto-sized and saved."""
    saved = OmegaConf.load(trained_pair["ci_bbox"] / "training_config.yaml")
    pre = saved.data_config.preprocessing
    assert pre.crop_mode == "bbox"
    labels = sio.load_slp(minimal_instance)
    sides = []
    for lf in labels:
        for inst in lf.user_instances:
            w, h = np.ptp(inst.numpy(), axis=0)
            sides.append(max(w, h) * 1.2)
    median = float(np.median(sides))
    assert pre.crop_size == int(np.ceil(median / 8) * 8)  # max_stride 8
    assert pre.crop_bbox_min_size == pytest.approx(median / 4, abs=0.01)


def test_detector_feeds_box_scaled_crops(trained_pair, minimal_instance):
    """Detector + box-scaled model: boxes seed crops and are kept, linked."""
    from sleap_nn.inference.layers.topdown import TopDownLayer
    from sleap_nn.inference.predictor import Predictor

    pred = Predictor.from_model_paths(
        [trained_pair["det"].as_posix(), trained_pair["ci_bbox"].as_posix()],
        device="cpu",
        peak_threshold=0.0,
    )
    layer = pred.layer
    assert isinstance(layer, TopDownLayer)
    assert isinstance(layer.centroid_layer, BBoxStageLayer)
    assert not layer.centroid_layer.use_gt_boxes
    assert layer.bbox_crop is not None and layer.bbox_crop.aspect == "square"
    assert not pred._needs_gt_instances()
    labels = pred.predict(sio.load_slp(minimal_instance), make_labels=True)
    lf = labels[0]
    assert len(lf.instances) >= 1
    # The detections that seeded the crops are kept and linked to their instances.
    assert len(lf.bboxes) >= len(lf.instances)
    linked = [b for b in lf.bboxes if b.instance is not None]
    assert {id(b.instance) for b in linked} == {id(i) for i in lf.instances}


def test_box_scaled_model_alone_uses_gt_boxes(trained_pair, minimal_instance):
    """A box-scaled model alone runs on GT boxes (post-train eval path)."""
    from sleap_nn.inference.predictor import Predictor

    pred = Predictor.from_model_paths(
        [trained_pair["ci_bbox"].as_posix()], device="cpu", peak_threshold=0.0
    )
    stage_1 = pred.layer.centroid_layer
    assert isinstance(stage_1, BBoxStageLayer) and stage_1.use_gt_boxes
    assert pred._needs_gt_instances()
    labels = pred.predict(sio.load_slp(minimal_instance), make_labels=True)
    assert len(labels[0].instances) == 2
    # GT-derived boxes are not packaged as predictions.
    assert len(labels[0].bboxes) == 0


def test_gt_box_lift_is_exact(trained_pair, minimal_instance, monkeypatch):
    """A crop-center keypoint lifts back to the box center, through the window."""
    from sleap_nn.inference.outputs import Outputs
    from sleap_nn.inference.predictor import Predictor

    pred = Predictor.from_model_paths(
        [trained_pair["ci_bbox"].as_posix()], device="cpu", peak_threshold=0.0
    )
    layer = pred.layer
    crop = layer.crop_size[0]

    def center_keypoints(crops):
        n = crops.shape[0]
        kp = torch.full((n, 1, 2, 2), (crop - 1) / 2)
        return Outputs(pred_keypoints=kp, pred_peak_values=torch.ones(n, 1, 2))

    monkeypatch.setattr(layer.centered_instance_layer, "predict", center_keypoints)
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    image = torch.from_numpy(lf.image).permute(2, 0, 1)[None]
    insts = torch.full((1, 2, 2, 2), float("nan"))
    for i, inst in enumerate(lf.instances):
        insts[0, i] = torch.from_numpy(inst.numpy())
    out = layer.predict(image, instances=insts)
    padding = OmegaConf.load(
        trained_pair["ci_bbox"] / "training_config.yaml"
    ).data_config.bbox_padding
    for i, inst in enumerate(lf.instances):
        x1, y1, x2, y2 = inst.to_bbox(mode="tight", padding=padding).xyxy
        np.testing.assert_allclose(
            out.pred_keypoints[0, i, 0].numpy(),
            [(x1 + x2) / 2, (y1 + y2) / 2],
            atol=1e-3,
        )


def test_detector_with_fixed_crop_model(trained_pair, minimal_instance):
    """A detector also seeds fixed crops (centered on box centers)."""
    from sleap_nn.inference.predictor import Predictor

    pred = Predictor.from_model_paths(
        [trained_pair["det"].as_posix(), trained_pair["ci_fixed"].as_posix()],
        device="cpu",
        peak_threshold=0.0,
    )
    assert isinstance(pred.layer.centroid_layer, BBoxStageLayer)
    assert pred.layer.bbox_crop is None
    labels = pred.predict(sio.load_slp(minimal_instance), make_labels=True)
    assert len(labels[0].instances) >= 1


def test_refused_combinations(trained_pair, minimal_instance_centroid_ckpt):
    """Centroid + box-scaled model, and two first stages, are refused."""
    from sleap_nn.inference.predictor import Predictor

    with pytest.raises(ValueError, match="first stage must be a bbox_centernet"):
        Predictor.from_model_paths(
            [minimal_instance_centroid_ckpt, trained_pair["ci_bbox"].as_posix()],
            device="cpu",
        )
    with pytest.raises(ValueError, match="both a centroid model and a bounding-box"):
        Predictor.from_model_paths(
            [
                minimal_instance_centroid_ckpt,
                trained_pair["det"].as_posix(),
                trained_pair["ci_fixed"].as_posix(),
            ],
            device="cpu",
        )


def test_trainer_refuses_box_crops_for_unsupported_types(config, tmp_path):
    """Box-scaled crops are refused for other model types."""
    from sleap_nn.training.model_trainer import ModelTrainer

    cfg = _train_config(config, tmp_path, "bbox_centernet", "x", crop_mode="bbox")
    with pytest.raises(ValueError, match="crop_mode='bbox' is supported"):
        ModelTrainer.get_model_trainer_from_config(cfg)._setup_bbox_crop(
            BBoxCropSpec(), (None, None)
        )


def test_predict_and_eval_cli(trained_pair, minimal_instance, tmp_path):
    """sleap-nn predict -m detector -m ci, then sleap-nn eval."""
    from click.testing import CliRunner

    from sleap_nn.cli import cli

    out = tmp_path / "pred.slp"
    result = CliRunner().invoke(
        cli,
        [
            "predict",
            "-i",
            str(minimal_instance),
            "-m",
            trained_pair["det"].as_posix(),
            "-m",
            trained_pair["ci_bbox"].as_posix(),
            "-o",
            str(out),
            "--peak_threshold",
            "0.0",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    labels = sio.load_slp(out)
    assert len(labels[0].instances) >= 1 and len(labels[0].bboxes) >= 1
    result = CliRunner().invoke(
        cli,
        [
            "eval",
            "--ground_truth_path",
            str(minimal_instance),
            "--predicted_path",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output


def test_multiclass_dataset_box_scaled_crops(minimal_instance):
    """The multi-class top-down dataset cuts the same box-scaled crops."""
    from sleap_nn.data.custom_datasets import TopDownCenteredInstanceMultiClassDataset

    labels = sio.load_slp(minimal_instance)
    for i, inst in enumerate(labels[0].instances):
        inst.track = sio.Track(f"{i}")
    labels.update()
    spec = BBoxCropSpec(aspect="square", padding=0.1)
    ds = TopDownCenteredInstanceMultiClassDataset(
        labels=[labels],
        crop_size=64,
        confmap_head_config=_head(),
        class_vectors_head_config=OmegaConf.create({"classes": ["0", "1"]}),
        max_stride=8,
        bbox_crop=spec,
    )
    plain = CenteredInstanceDataset(
        labels=[sio.load_slp(minimal_instance)],
        crop_size=64,
        confmap_head_config=_head(),
        max_stride=8,
        bbox_crop=spec,
    )
    for index in range(len(ds)):
        sample, ref = ds[index], plain[index]
        assert torch.equal(sample["instance_image"], ref["instance_image"])
        np.testing.assert_allclose(sample["instance"], ref["instance"])
        assert sample["class_vectors"].shape == (2,)


def test_detector_with_multiclass_topdown(
    trained_pair, minimal_instance, minimal_instance_multi_class_topdown_ckpt
):
    """A detector seeds a multi-class top-down model."""
    from sleap_nn.inference.layers.topdown_multiclass import TopDownMultiClassLayer
    from sleap_nn.inference.predictor import Predictor

    pred = Predictor.from_model_paths(
        [
            trained_pair["det"].as_posix(),
            str(minimal_instance_multi_class_topdown_ckpt),
        ],
        device="cpu",
        peak_threshold=0.0,
    )
    assert isinstance(pred.layer, TopDownMultiClassLayer)
    assert isinstance(pred.layer.centroid_layer, BBoxStageLayer)
    labels = pred.predict(sio.load_slp(minimal_instance), make_labels=True)
    assert len(labels[0].instances) >= 1


def test_export_refuses_detector_pair(trained_pair, tmp_path):
    """Exporting a detector + top-down pair is refused clearly."""
    from click.testing import CliRunner

    from sleap_nn.export.cli import export

    result = CliRunner().invoke(
        export,
        [
            trained_pair["det"].as_posix(),
            trained_pair["ci_bbox"].as_posix(),
            "--output",
            str(tmp_path / "exp"),
        ],
    )
    assert result.exit_code != 0
    assert "cannot be exported as one bundle" in result.output


def test_detector_with_topdown_segmentation(
    trained_pair, minimal_instance_seg, tmp_path
):
    """A detector seeds fixed crops for a top-down segmentation model."""
    from sleap_nn.inference.layers.topdown_segmentation import TopDownSegmentationLayer
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.train import train

    train(
        train_labels_path=[minimal_instance_seg.as_posix()],
        use_same_data_for_val=True,
        head_configs="centered_instance_segmentation",
        backbone_config={
            "unet": {
                "in_channels": 1,
                "filters": 8,
                "filters_rate": 1.5,
                "max_stride": 16,
                "output_stride": 2,
            }
        },
        ensure_grayscale=True,
        crop_size=160,
        batch_size=1,
        max_epochs=1,
        min_train_steps_per_epoch=1,
        num_workers=0,
        save_ckpt=True,
        ckpt_dir=tmp_path.as_posix(),
        run_name="seg",
        trainer_accelerator="cpu",
        visualize_preds_during_training=False,
    )
    pred = Predictor.from_model_paths(
        [trained_pair["det"].as_posix(), (tmp_path / "seg").as_posix()],
        device="cpu",
        peak_threshold=0.0,
    )
    assert isinstance(pred.layer, TopDownSegmentationLayer)
    assert isinstance(pred.layer.centroid_layer, BBoxStageLayer)
    labels = pred.predict(sio.load_slp(minimal_instance_seg), make_labels=True)
    assert len(labels) >= 1
    assert all(isinstance(m, sio.PredictedSegmentationMask) for m in labels.masks)


def test_gt_box_fallback_uses_training_box_rule(
    trained_pair, minimal_instance, monkeypatch
):
    """The GT-box fallback derives boxes as training did (linked user boxes)."""
    import dataclasses

    from sleap_nn.inference.outputs import Outputs
    from sleap_nn.inference.predictor import Predictor

    saved = OmegaConf.load(trained_pair["ci_bbox"] / "training_config.yaml")
    # `auto` is frozen to what it meant for the (pose-only) training labels.
    assert saved.data_config.bbox_source == "instances"
    pred = Predictor.from_model_paths(
        [trained_pair["ci_bbox"].as_posix()], device="cpu", peak_threshold=0.0
    )
    pred.layer.bbox_crop = dataclasses.replace(
        pred.layer.bbox_crop, bbox_source="bboxes"
    )
    crop = pred.layer.crop_size[0]

    def center_keypoints(crops):
        n = crops.shape[0]
        kp = torch.full((n, 1, 2, 2), (crop - 1) / 2)
        return Outputs(pred_keypoints=kp, pred_peak_values=torch.ones(n, 1, 2))

    monkeypatch.setattr(pred.layer.centered_instance_layer, "predict", center_keypoints)
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    linked, other = lf.instances
    lf.bboxes = [sio.UserBoundingBox(x1=40, y1=60, x2=200, y2=220, instance=linked)]
    out = pred.predict(labels, make_labels=True)
    centers = sorted(tuple(np.round(i.numpy()[0], 2)) for i in out[0].instances)
    x1, y1, x2, y2 = other.to_bbox(mode="tight").xyxy
    expected = sorted(
        [(120.0, 140.0), (round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2))]
    )
    np.testing.assert_allclose(centers, expected, atol=1e-2)


def test_more_refused_combinations(
    trained_pair, minimal_instance, minimal_instance_centroid_ckpt
):
    """Legacy pipeline, export and a stage-2-less detector pair are refused."""
    from click.testing import CliRunner

    from sleap_nn.export.cli import export
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.legacy_predict import run_inference

    with pytest.raises(ValueError, match="no top-down model"):
        Predictor.from_model_paths(
            [trained_pair["det"].as_posix(), str(minimal_instance_centroid_ckpt)],
            device="cpu",
        )
    with pytest.raises(ValueError, match="box-scaled crops"):
        run_inference(
            data_path=str(minimal_instance),
            model_paths=[trained_pair["ci_bbox"].as_posix()],
            make_labels=True,
            device="cpu",
        )
    result = CliRunner().invoke(
        export,
        [str(minimal_instance_centroid_ckpt), trained_pair["ci_bbox"].as_posix()],
    )
    assert result.exit_code != 0 and "crop_mode='bbox'" in result.output


def test_filters_drop_seeding_boxes():
    """Filtering an instance out also drops the detection that seeded it."""
    from sleap_nn.inference.filters import FilterPipeline
    from sleap_nn.inference.outputs import Outputs

    out = Outputs(
        pred_keypoints=torch.zeros(1, 2, 3, 2),
        pred_peak_values=torch.ones(1, 2, 3),
        instance_scores=torch.tensor([[0.9, 0.1]]),
        pred_bboxes=torch.tensor([[[0.0, 0.0, 5.0, 5.0], [1.0, 1.0, 6.0, 6.0]]]),
        pred_bbox_scores=torch.tensor([[0.9, 0.1]]),
        pred_bbox_classes=torch.tensor([[0, 0]]),
    )
    dropped = FilterPipeline._nan_out_where(torch.tensor([[False, True]]), out)
    assert torch.isfinite(dropped.pred_bboxes[0, 0]).all()
    assert torch.isnan(dropped.pred_bboxes[0, 1]).all()
    assert dropped.pred_bbox_classes[0].tolist() == [0, -1]
