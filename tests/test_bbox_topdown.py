"""Bounding-box detectors as the first stage of top-down models.

Covers the box-scaled crop geometry shared by training and inference
(``preprocessing.crop_mode="bbox"``), the centered-instance datasets that cut those
crops, the trainer's crop sizing, and the detector -> crop model pipelines
(``Predictor.from_model_paths([detector, centered_instance])``), including the
GT-box fallback and the combinations that must be refused.
"""

from pathlib import Path

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
def trained_pair(bbox_det_pair, tmp_path_factory, pytestconfig):
    """A CenterNet detector + box-scaled and fixed-crop centered-instance models.

    The detector and the box-scaled model are the session's shared pair
    (``tests/fixtures/bbox_models.py``, same training settings); only the fixed-crop
    model is trained here.
    """
    from pathlib import Path

    from sleap_nn.training.model_trainer import ModelTrainer
    from tests.fixtures.datasets import config as config_fixture

    tmp_path = tmp_path_factory.mktemp("bbox_topdown")
    data_dir = Path(pytestconfig.rootdir) / "tests/assets/datasets"
    base = config_fixture.__wrapped__(data_dir)
    dirs = dict(bbox_det_pair)
    cfg = _train_config(base, tmp_path, "centered_instance", "ci_fixed")
    ModelTrainer.get_model_trainer_from_config(cfg).train()
    dirs["ci_fixed"] = tmp_path / "ci_fixed"
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


def _labels_with_linked_user_box(path, video=None):
    """The fixture's frame with a user box linked to one instance, off its keypoints.

    With ``video``, the same instances sit on frame 0 of that video file instead of
    the fixture's embedded frame.
    """
    labels = sio.load_slp(path)
    lf = labels[0]
    if video is not None:
        lf = sio.LabeledFrame(
            video=sio.Video.from_filename(str(video)),
            frame_idx=0,
            instances=lf.instances,
        )
        labels = sio.Labels(
            labeled_frames=[lf], videos=[lf.video], skeletons=labels.skeletons
        )
    lf.bboxes = [
        sio.UserBoundingBox(x1=40, y1=60, x2=200, y2=220, instance=lf.instances[0])
    ]
    return labels


@pytest.mark.parametrize(
    "source",
    ["labels_provider", "video_selection", "list_of_providers", "multi_provider"],
)
def test_gt_boxes_follow_the_box_rule_for_any_frame_source(
    trained_pair, minimal_instance, centered_instance_video, monkeypatch, source
):
    """boxes='gt' crops the same boxes whoever built the frame source.

    A ``LabelsProvider`` built by the caller (as the command line does for its .slp
    frame filters) or by a ``VideoSelection`` must get the model's GT-box rule, as the
    provider the predictor builds from a ``sio.Labels`` does; without it the crops fell
    back to tight keypoint boxes and ignored the labels' own boxes.
    """
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import (
        LabelsProvider,
        MultiVideoProvider,
        VideoSelection,
    )

    pred = Predictor.from_model_paths(
        [trained_pair["ci_fixed"].as_posix()],
        device="cpu",
        peak_threshold=0.0,
        boxes="gt",
    )
    stage_1 = pred.layer.centroid_layer
    seen = []
    stage_1_predict = stage_1.predict

    def spy(image, instances=None, boxes=None, **kwargs):
        out = stage_1_predict(image, instances=instances, boxes=boxes, **kwargs)
        seen.append(out.pred_bboxes.reshape(-1, 4).numpy())
        return out

    monkeypatch.setattr(stage_1, "predict", spy)

    def crop_boxes(src):
        seen.clear()
        labels = pred.predict(src, make_labels=True)
        points = np.stack([inst.numpy() for inst in labels[0].instances])
        return np.concatenate(seen), points

    # A VideoSelection reads a video with its own file.
    labels = _labels_with_linked_user_box(
        minimal_instance,
        video=centered_instance_video if source == "video_selection" else None,
    )
    expected_boxes, expected_points = crop_boxes(labels)
    # The reference route: the linked user box as is, the other instance's keypoint box.
    tight = labels[0].instances[1].to_bbox(mode="tight").xyxy
    np.testing.assert_allclose(
        expected_boxes, [[40, 60, 200, 220], list(tight)], atol=1e-3
    )

    def provider():
        return LabelsProvider(labels=labels, batch_size=4, only_labeled_frames=True)

    if source == "labels_provider":
        src = provider()
    elif source == "video_selection":
        src = VideoSelection(labels=labels, video_index=0)
    elif source == "list_of_providers":
        src = [provider()]
    else:
        src = MultiVideoProvider(providers=[provider()], video_offsets=[0])
    boxes, points = crop_boxes(src)
    np.testing.assert_allclose(boxes, expected_boxes, atol=1e-3)
    np.testing.assert_allclose(points, expected_points, atol=1e-3)

    # A box rule the caller set on its own provider is kept.
    def own_rule(lf, inst):
        return np.array([10, 20, 110, 120], dtype=np.float32)

    src = LabelsProvider(labels=labels, batch_size=4, instance_box_fn=own_rule)
    boxes, _ = crop_boxes(src)
    np.testing.assert_allclose(boxes, [[10, 20, 110, 120]] * 2, atol=1e-3)
    assert src.instance_box_fn is own_rule


@pytest.mark.parametrize("source", ["in_memory", "old_file"])
def test_gt_box_rule_counts_occluded_keypoints_for_a_caller_built_provider(
    trained_pair, minimal_instance, monkeypatch, source
):
    """A box-scaled model alone boxes occluded keypoints too, for any frame source.

    The model's rule counts a keypoint that is marked not visible but has
    coordinates (an occluded estimate); the tight box of the visible keypoints,
    which a caller-built provider used to get, leaves it out. Labels read from an
    SLP file older than format 3.0 cannot hold occluded estimates: where sleap-nn
    and sleap-io can tell such labels apart, a hidden point's stored coordinates
    are stale and do not count.
    """
    from sleap_nn.data import providers as data_providers
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import LabelsProvider

    loaded = sio.load_slp(minimal_instance)
    if source == "in_memory":
        # Labels built in memory (no source file): occluded estimates are real.
        labels = sio.Labels(
            labeled_frames=loaded.labeled_frames,
            videos=loaded.videos,
            skeletons=loaded.skeletons,
        )
    else:
        labels = loaded
        if not (
            hasattr(data_providers, "has_legacy_hidden_points")
            and data_providers.has_legacy_hidden_points(labels)
        ):
            pytest.skip("this sleap-nn or sleap-io does not tell old-layout files")
    occluded, other = labels[0].instances
    occluded.points["visible"][1] = False
    located = occluded.numpy(invisible_as_nan=False)
    assert np.isnan(occluded.numpy()[1]).all() and np.isfinite(located).all()
    counted = located if source == "in_memory" else occluded.numpy()[[0]]

    pred = Predictor.from_model_paths(
        [trained_pair["ci_bbox"].as_posix()], device="cpu", peak_threshold=0.0
    )
    padding = OmegaConf.load(
        trained_pair["ci_bbox"] / "training_config.yaml"
    ).data_config.bbox_padding
    stage_1 = pred.layer.centroid_layer
    seen = []
    stage_1_predict = stage_1.predict

    def spy(image, instances=None, boxes=None, **kwargs):
        out = stage_1_predict(image, instances=instances, boxes=boxes, **kwargs)
        seen.append(out.pred_bboxes.reshape(-1, 4).numpy())
        return out

    monkeypatch.setattr(stage_1, "predict", spy)
    expected = [
        [*(counted.min(axis=0) - padding), *(counted.max(axis=0) + padding)],
        list(other.to_bbox(mode="tight", padding=padding).xyxy),
    ]
    for src in (
        labels,
        LabelsProvider(labels=labels, batch_size=4, only_labeled_frames=True),
    ):
        seen.clear()
        pred.predict(src, make_labels=True)
        np.testing.assert_allclose(np.concatenate(seen), expected, atol=1e-3)


def test_reused_provider_gets_each_predictors_gt_box_rule(
    trained_pair, minimal_instance
):
    """A provider reused with another predictor does not keep the first one's rule."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import LabelsProvider

    labels = _labels_with_linked_user_box(minimal_instance)
    lf = labels[0]
    src = LabelsProvider(labels=labels, batch_size=4, only_labeled_frames=True)

    def rule_boxes(model, **kwargs):
        pred = Predictor.from_model_paths(
            [trained_pair[model].as_posix()],
            device="cpu",
            peak_threshold=0.0,
            **kwargs,
        )
        pred.predict(src, make_labels=True)
        return [src.instance_box_fn(lf, inst) for inst in lf.instances]

    # boxes="gt": the labels' own boxes where they have them.
    np.testing.assert_allclose(
        rule_boxes("ci_fixed", boxes="gt")[0], [40, 60, 200, 220], atol=1e-3
    )
    # A box-scaled model trained on keypoint boxes: its own rule, not the first one.
    padding = OmegaConf.load(
        trained_pair["ci_bbox"] / "training_config.yaml"
    ).data_config.bbox_padding
    expected = [
        inst.to_bbox(mode="tight", padding=padding).xyxy for inst in lf.instances
    ]
    np.testing.assert_allclose(rule_boxes("ci_bbox"), expected, atol=1e-3)


def test_predict_cli_gt_boxes_same_with_frame_filters(
    trained_pair, minimal_instance, tmp_path
):
    """`sleap-nn predict --boxes gt` crops the same boxes with --only_labeled_frames."""
    from click.testing import CliRunner

    from sleap_nn.cli import cli

    src = tmp_path / "linked_box.pkg.slp"
    _labels_with_linked_user_box(minimal_instance).save(src, embed=True)
    lf = sio.load_slp(src)[0]
    assert lf.bboxes[0].instance is lf.instances[0]

    def run(name, *extra):
        out = tmp_path / f"{name}.slp"
        result = CliRunner().invoke(
            cli,
            ["predict", "-i", str(src), "-m", trained_pair["ci_fixed"].as_posix()]
            + ["--boxes", "gt", "--no-rescore", "--peak_threshold", "0"]
            + ["--device", "cpu", "-o", str(out), *extra],
        )
        assert result.exit_code == 0, result.output
        frame = sio.load_slp(out)[0]
        return np.stack([inst.numpy() for inst in frame.predicted_instances])

    plain = run("plain")
    assert plain.shape[0] == 2 and np.isfinite(plain).all()
    np.testing.assert_allclose(
        run("filtered", "--only_labeled_frames"), plain, atol=1e-3
    )


def _labels_with_predictions(path, video, tmp_path):
    """A ``.slp`` on a video: user and predicted instances, predicted-only, empty.

    Frame 0 holds the fixture's two user instances (a user box linked to the first)
    and a predicted instance; frame 1 a predicted instance only; frames 2 and 3 are
    labeled frames without instances. Frames 1 and 2 are suggestions.
    """
    labels = _labels_with_linked_user_box(path, video=video)
    lf = labels[0]
    skeleton = labels.skeletons[0]
    lf.instances.append(
        sio.PredictedInstance.from_numpy(
            lf.instances[1].numpy() + 60, skeleton=skeleton, score=1.0
        )
    )
    labels.append(
        sio.LabeledFrame(
            video=lf.video,
            frame_idx=1,
            instances=[
                sio.PredictedInstance.from_numpy(
                    lf.instances[0].numpy(), skeleton=skeleton, score=1.0
                )
            ],
        )
    )
    for idx in (2, 3):
        labels.append(sio.LabeledFrame(video=lf.video, frame_idx=idx, instances=[]))
    labels.suggestions = [
        sio.SuggestionFrame(video=lf.video, frame_idx=idx) for idx in (1, 2)
    ]
    out = tmp_path / "with_predictions.slp"
    labels.save(out)
    return out


def _invoke_counting_skips(args):
    """Run the command line; return the result and its frame-skip warnings."""
    from click.testing import CliRunner
    from loguru import logger

    from sleap_nn.cli import cli

    messages = []
    handler = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        result = CliRunner().invoke(cli, args)
    finally:
        logger.remove(handler)
    return result, [m for m in messages if "are skipped" in m]


def _predicted_points(path):
    """``{frame_idx: (n, nodes, 2)}`` of the predicted instances in a ``.slp``."""
    return {
        lf.frame_idx: np.stack([inst.numpy() for inst in lf.predicted_instances])
        for lf in sio.load_slp(path)
        if lf.predicted_instances
    }


def test_predict_cli_only_predicted_frames_gt_boxes_from_user_instances(
    trained_pair, minimal_instance, centered_instance_video, tmp_path
):
    """`--boxes gt --only_predicted_frames` crops user instances only, as predict().

    Predicted instances are never taken as GT boxes, and a selected frame without
    user instances is skipped, as on the route where the predictor builds the frame
    source from the .slp itself.
    """
    from click.testing import CliRunner

    from sleap_nn.cli import cli
    from sleap_nn.inference.run import predict

    src = _labels_with_predictions(minimal_instance, centered_instance_video, tmp_path)
    model = trained_pair["ci_fixed"].as_posix()
    reference = tmp_path / "reference.slp"
    predict(
        str(src),
        model_paths=[model],
        device="cpu",
        peak_threshold=0.0,
        boxes="gt",
        output_path=str(reference),
    )
    expected = _predicted_points(reference)
    assert list(expected) == [0] and expected[0].shape[0] == 2

    out = tmp_path / "filtered.slp"
    result = CliRunner().invoke(
        cli,
        ["predict", "-i", str(src), "-m", model, "--boxes", "gt", "--no-rescore"]
        + ["--peak_threshold", "0", "--device", "cpu", "-o", str(out)]
        + ["--only_predicted_frames"],
    )
    assert result.exit_code == 0, result.output
    got = _predicted_points(out)
    assert got[0].shape[0] == 2 and list(got) == [0]
    np.testing.assert_allclose(got[0], expected[0], atol=1e-3)


def test_gt_box_scope_of_a_reused_provider(trained_pair, minimal_instance, tmp_path):
    """The user-instance scope is undone for a predictor that does not crop GT boxes."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import LabelsProvider

    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    lf.instances.append(
        sio.PredictedInstance.from_numpy(
            lf.instances[1].numpy() + 60, skeleton=labels.skeletons[0], score=1.0
        )
    )
    labels.append(
        sio.LabeledFrame(video=lf.video, frame_idx=lf.frame_idx + 1, instances=[])
    )
    src = LabelsProvider(labels=labels, batch_size=4)
    assert src.num_frames() == 2

    pred = Predictor.from_model_paths(
        [trained_pair["ci_fixed"].as_posix()], device="cpu", boxes="gt"
    )
    pred._attach_gt_box_fn(src)
    assert src.num_frames() == 1
    assert len(src._frame_instances(lf)) == 2

    pred = Predictor.from_model_paths(
        [trained_pair["det"].as_posix(), trained_pair["ci_bbox"].as_posix()],
        device="cpu",
    )
    pred._attach_gt_box_fn(src)
    assert src.num_frames() == 2
    assert len(src._frame_instances(lf)) == 3


def test_stream_to_file_box_scaled_model_skips_frames_without_user_instances(
    trained_pair, minimal_instance, centered_instance_video, tmp_path
):
    """A box-scaled model alone streams the frames with user instances only.

    It used to stop with "requires `instances`" on a batch of frames without any
    instance, and to crop predicted instances as GT; it now runs what the in-memory
    route runs, and reports the skipped frames once.
    """
    src = _labels_with_predictions(minimal_instance, centered_instance_video, tmp_path)
    model = trained_pair["ci_bbox"].as_posix()

    def run(name, flag):
        out = tmp_path / f"{name}.slp"
        result, skips = _invoke_counting_skips(
            ["predict", "-i", str(src), "-m", model, "--peak_threshold", "0"]
            + ["--device", "cpu", "--batch_size", "1", flag, str(out)],
        )
        assert result.exit_code == 0, result.output
        return _predicted_points(out), skips

    expected, _ = run("in_memory", "-o")
    assert list(expected) == [0] and expected[0].shape[0] == 2
    got, skips = run("streamed", "--stream-to-file")
    assert list(got) == [0]
    np.testing.assert_allclose(got[0], expected[0], atol=1e-3)
    assert len(skips) == 1 and "3 of 4" in skips[0]


@pytest.mark.parametrize(
    "route", ["boxes_gt_in_memory", "box_scaled_in_memory", "box_scaled_stream"]
)
@pytest.mark.parametrize(
    "flag, n_selected",
    [("--exclude_user_labeled", 3), ("--only_suggested_frames", 2)],
)
def test_gt_boxes_on_frames_without_user_instances_give_empty_output(
    trained_pair,
    minimal_instance,
    centered_instance_video,
    tmp_path,
    route,
    flag,
    n_selected,
):
    """Filters that select no frame with user instances: an empty output, one warning.

    Labelled boxes come from user instances only, so nothing runs. Before, the
    predicted instance on such a frame was cropped (`--boxes gt`), or the run stopped
    with "requires `instances`" (a box-scaled model alone).
    """
    src = _labels_with_predictions(minimal_instance, centered_instance_video, tmp_path)
    out = tmp_path / "out.slp"
    if route == "boxes_gt_in_memory":
        model_args = ["-m", trained_pair["ci_fixed"].as_posix(), "--boxes", "gt"]
        model_args += ["--no-rescore"]
    else:
        model_args = ["-m", trained_pair["ci_bbox"].as_posix()]
    out_flag = "--stream-to-file" if route.endswith("stream") else "-o"
    result, skips = _invoke_counting_skips(
        ["predict", "-i", str(src), *model_args, "--peak_threshold", "0"]
        + ["--device", "cpu", "--batch_size", "1", flag, out_flag, str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists() and _predicted_points(out) == {}
    assert len(skips) == 1
    assert f"{n_selected} of {n_selected} selected frame(s)" in skips[0]


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


# --------------------------------------------------------------------------- #
# Given boxes (sleap-nn predict --boxes)
# --------------------------------------------------------------------------- #
class _NoBackend:
    """A ``ModelBackend`` stand-in for stage layers that never call it."""

    device = "cpu"
    does_baked_postproc = False

    def __call__(self, x):
        raise AssertionError("not called")

    def warmup(self, input_shape):
        return None


def test_given_boxes_stage_layer():
    """Given boxes pass through with their scores; weak and padded ones drop."""
    stage = BBoxStageLayer(
        backend=_NoBackend(),
        decoder=None,
        use_given_boxes=True,
        box_score_threshold=0.3,
    )
    assert not stage.use_gt_centroids  # needs no GT instances
    boxes = torch.tensor(
        [[[0.0, 0.0, 10.0, 20.0], [5.0, 5.0, 9.0, 9.0], [float("nan")] * 4]]
    )
    out = stage.predict(
        torch.zeros(1, 1, 32, 32),
        boxes=boxes,
        box_scores=torch.tensor([[0.9, 0.1, float("nan")]]),
    )
    np.testing.assert_allclose(out.pred_centroids[0, 0].numpy(), [5.0, 10.0])
    assert out.pred_centroid_values[0, 0].item() == pytest.approx(0.9)
    assert torch.isnan(out.pred_bboxes[0, 1:]).all()
    assert out.pred_bbox_classes[0].tolist() == [0, -1, -1]
    # A batch with no box source at all has no detections.
    out = stage.predict(torch.zeros(2, 1, 32, 32))
    assert (
        out.pred_centroids.shape == (2, 1, 2) and torch.isnan(out.pred_centroids).all()
    )
    with pytest.raises(ValueError, match="exclusive"):
        BBoxStageLayer(_NoBackend(), None, use_gt_boxes=True, use_given_boxes=True)


def test_labels_provider_frame_boxes(minimal_instance):
    """frame_box_fn puts each frame's predicted boxes on the batch, NaN-padded."""
    from sleap_nn.inference.providers import LabelsProvider, predicted_frame_boxes

    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    lf.bboxes = [
        sio.PredictedBoundingBox(x1=1, y1=2, x2=3, y2=4, score=0.7),
        sio.UserBoundingBox(x1=0, y1=0, x2=9, y2=9),  # GT: skipped
        sio.PredictedBoundingBox(x1=5, y1=6, x2=7, y2=8, score=0.2),
    ]
    boxes, scores = predicted_frame_boxes(lf)
    np.testing.assert_allclose(boxes, [[1, 2, 3, 4], [5, 6, 7, 8]])
    np.testing.assert_allclose(scores, [0.7, 0.2])
    batch = next(
        iter(
            LabelsProvider(
                labels=labels, frame_box_fn=predicted_frame_boxes, prefetch=False
            )
        )
    )
    assert batch.frame_boxes.shape == (1, 2, 4)
    np.testing.assert_allclose(batch.frame_box_scores[0], [0.7, 0.2])


def _labels_with_boxes(path, scores=(0.8, 0.6)):
    """The fixture's frame with one predicted box per instance (padded)."""
    labels = sio.load_slp(path)
    lf = labels[0]
    for inst, score in zip(lf.instances, scores):
        x1, y1, x2, y2 = inst.to_bbox(mode="tight", padding=4.0).xyxy
        lf.bboxes.append(
            sio.PredictedBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, score=score)
        )
    return labels


@pytest.mark.parametrize("model", ["ci_bbox", "ci_fixed"])
def test_predictor_seeds_from_given_boxes(trained_pair, minimal_instance, model):
    """boxes='given': one instance per box, scored by it, the box kept linked."""
    from sleap_nn.inference.predictor import Predictor

    pred = Predictor.from_model_paths(
        [trained_pair[model].as_posix()],
        device="cpu",
        peak_threshold=0.0,
        boxes="given",
        box_score_threshold=0.7,
    )
    stage_1 = pred.layer.centroid_layer
    assert isinstance(stage_1, BBoxStageLayer) and stage_1.use_given_boxes
    assert not pred._needs_gt_instances()
    labels = pred.predict(_labels_with_boxes(minimal_instance), make_labels=True)
    lf = labels[0]
    # The 0.6 box is below the threshold.
    assert len(lf.instances) == 1
    assert lf.instances[0].score == pytest.approx(0.8)
    assert len(lf.bboxes) == 1 and lf.bboxes[0].instance is lf.instances[0]


def test_predictor_seeds_fixed_crops_from_gt_boxes(trained_pair, minimal_instance):
    """boxes='gt' seeds a fixed-crop model from GT boxes (not GT centroids)."""
    from sleap_nn.inference.predictor import Predictor

    pred = Predictor.from_model_paths(
        [trained_pair["ci_fixed"].as_posix()],
        device="cpu",
        peak_threshold=0.0,
        boxes="gt",
    )
    stage_1 = pred.layer.centroid_layer
    assert isinstance(stage_1, BBoxStageLayer) and stage_1.use_gt_boxes
    assert pred.layer.bbox_crop is None
    labels = pred.predict(sio.load_slp(minimal_instance), make_labels=True)
    assert len(labels[0].instances) == 2


def test_box_seeding_refusals(trained_pair, minimal_instance):
    """Given boxes replace stage 1, so a detector alongside is refused."""
    from sleap_nn.inference.predictor import Predictor

    with pytest.raises(ValueError, match="replaces the top-down stage 1"):
        Predictor.from_model_paths(
            [trained_pair["det"].as_posix(), trained_pair["ci_bbox"].as_posix()],
            device="cpu",
            boxes="given",
        )
    with pytest.raises(ValueError, match="needs a top-down crop model"):
        Predictor.from_model_paths(
            [trained_pair["det"].as_posix()], device="cpu", boxes="gt"
        )
    pred = Predictor.from_model_paths(
        [trained_pair["ci_bbox"].as_posix()], device="cpu", boxes="given"
    )
    with pytest.raises(ValueError, match="frames of a .slp / sio.Labels"):
        pred.predict(np.zeros((1, 64, 64, 1), dtype=np.uint8))


def test_predict_cli_with_boxes_slp(trained_pair, minimal_instance, tmp_path):
    """sleap-nn predict --boxes boxes.slp copies the boxes, rescores, and evals."""
    from click.testing import CliRunner

    from sleap_nn.cli import cli

    boxes_path = tmp_path / "boxes.slp"
    _labels_with_boxes(minimal_instance).save(boxes_path)
    out = tmp_path / "pred.slp"
    args = [
        "predict",
        "-i",
        str(minimal_instance),
        "-m",
        trained_pair["ci_bbox"].as_posix(),
        "--boxes",
        str(boxes_path),
        "--peak_threshold",
        "0",
        "--oks_nms",
        "0.9",
        "--device",
        "cpu",
        "-o",
        str(out),
    ]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    lf = sio.load_slp(out)[0]
    assert len(lf.instances) == 2
    # Rescoring is on by default with --boxes: box score x mean keypoint score.
    for inst in lf.instances:
        box = next(b for b in lf.bboxes if b.instance is inst)
        vals = inst.numpy(scores=True)[:, 2]
        above = vals[vals > 0.2]
        kpt = above.mean() if above.size else 0.0
        assert inst.score == pytest.approx(box.score * kpt, rel=1e-4)
    result = CliRunner().invoke(
        cli,
        ["eval", "-g", str(minimal_instance), "-p", str(out), "--coco"],
    )
    assert result.exit_code == 0, result.output

    # Usage errors, before any model loads.
    result = CliRunner().invoke(cli, args + ["--stream-to-file", str(out)])
    assert result.exit_code != 0 and "--stream-to-file" in result.output
    video_args = list(args)
    video_args[2] = "video.mp4"
    result = CliRunner().invoke(cli, video_args)
    assert result.exit_code != 0 and "needs a .slp input" in result.output


def _export(src, dst, *args):
    from click.testing import CliRunner

    from sleap_nn.export.cli import export

    result = CliRunner().invoke(export, [str(src), "-o", str(dst), *args])
    assert result.exit_code == 0, result.output
    return dst


@pytest.mark.parametrize("input_dtype", ["float32", "uint8"])
def test_network_only_exports_in_checkpoint_pipeline(
    trained_pair, minimal_instance, tmp_path, input_dtype
):
    """A detector export + a --network-only box-scaled export predict as the pair."""
    pytest.importorskip("onnxruntime")
    import json

    from sleap_nn.inference.exported_network import ExportedNetwork, is_network_export
    from sleap_nn.inference.predictor import Predictor

    det = _export(trained_pair["det"], tmp_path / "det", "--input-dtype", input_dtype)
    ci = _export(
        trained_pair["ci_bbox"],
        tmp_path / "ci",
        "--network-only",
        "--input-dtype",
        input_dtype,
    )
    assert is_network_export(det) and is_network_export(ci)
    assert not is_network_export(trained_pair["ci_bbox"])
    meta = json.loads((ci / "export_metadata.json").read_text())
    assert meta["network_only"] is True
    assert meta["output_names"] == ["CenteredInstanceConfmapsHead"]
    assert meta["input_dtype"] == input_dtype

    # The top detections only: below them, the barely trained detector's
    # near-zero plateaus make the set of local maxima hinge on 1e-7 differences.
    kwargs = dict(device="cpu", peak_threshold=0.0, max_instances=2)
    ref = Predictor.from_model_paths(
        [trained_pair["det"].as_posix(), trained_pair["ci_bbox"].as_posix()], **kwargs
    )
    exp = Predictor.from_model_paths(
        [det.as_posix(), ci.as_posix()], export_runtime="onnx", **kwargs
    )
    # The exported networks sit where the checkpoints' modules sit; every other
    # step (decoder, box-scaled crops, peaks) is the checkpoint path's.
    assert isinstance(exp.layer.centroid_layer.backend.model, ExportedNetwork)
    assert isinstance(exp.layer.centered_instance_layer.backend.model, ExportedNetwork)
    a = ref.predict(sio.load_slp(minimal_instance), make_labels=True)[0]
    b = exp.predict(sio.load_slp(minimal_instance), make_labels=True)[0]
    assert len(a.instances) == len(b.instances) >= 1
    ba = np.array([x.xyxy for x in a.bboxes])
    bb = np.array([x.xyxy for x in b.bboxes])
    pa = np.stack([i.numpy() for i in a.instances])
    pb = np.stack([i.numpy() for i in b.instances])
    # float32 graphs get the checkpoint path's own inputs; uint8 graphs a rounded
    # crop (up to half a grey level).
    tol = 1e-3 if input_dtype == "float32" else 1.0
    np.testing.assert_allclose(bb, ba, atol=tol)
    np.testing.assert_allclose(pb, pa, atol=tol)


def test_network_only_refusals(trained_pair, tmp_path):
    """--network-only is for crop models; exported networks take no node filter."""
    pytest.importorskip("onnxruntime")
    from click.testing import CliRunner

    from sleap_nn.export.cli import export
    from sleap_nn.inference.predictor import Predictor

    result = CliRunner().invoke(
        export,
        [trained_pair["det"].as_posix(), "-o", str(tmp_path / "x"), "--network-only"],
    )
    assert result.exit_code != 0 and "centered_instance" in result.output
    result = CliRunner().invoke(
        export,
        [
            trained_pair["ci_fixed"].as_posix(),
            "-o",
            str(tmp_path / "y"),
            "--input-dtype",
            "float32",
        ],
    )
    assert result.exit_code != 0 and "--input-dtype float32" in result.output

    det = _export(trained_pair["det"], tmp_path / "det")
    ci = _export(trained_pair["ci_bbox"], tmp_path / "ci", "--network-only")
    with pytest.raises(ValueError, match="cannot slice an exported network"):
        Predictor.from_model_paths(
            [det.as_posix(), ci.as_posix()],
            device="cpu",
            export_runtime="onnx",
            node_filter=["A"],
        )


def test_predict_runtime_reaches_every_export_directory(
    trained_pair, minimal_instance, tmp_path
):
    """`--runtime` applies to export directories among several --model_paths (#857).

    The exports hold ONNX graphs only: `--runtime onnx` predicts, and `--runtime
    tensorrt` must fail on the missing engine instead of silently running "auto".
    """
    pytest.importorskip("onnxruntime")
    from pathlib import Path

    from click.testing import CliRunner

    from sleap_nn.cli import cli

    det = _export(trained_pair["det"], tmp_path / "det")
    ci = _export(trained_pair["ci_bbox"], tmp_path / "ci", "--network-only")
    base = [
        "predict",
        "-i",
        str(minimal_instance),
        "-m",
        det.as_posix(),
        "-m",
        ci.as_posix(),
        "--device",
        "cpu",
    ]
    out = tmp_path / "p.slp"
    result = CliRunner().invoke(cli, base + ["-o", str(out), "--runtime", "onnx"])
    assert result.exit_code == 0, result.output
    labels = sio.load_slp(out)
    assert len(labels.labeled_frames) >= 1
    # The provenance names the runtime each exported graph ran on.
    ran = labels.provenance["exported_models"]
    assert sorted(Path(e["file"]).parent.name for e in ran) == ["ci", "det"]
    assert {e["runtime"] for e in ran} == {"onnx"}
    result = CliRunner().invoke(
        cli, base + ["-o", str(tmp_path / "q.slp"), "--runtime", "tensorrt"]
    )
    assert result.exit_code != 0
    assert "TensorRT model not found" in str(result.exception) + result.output


@pytest.fixture(scope="module")
def lone_det_export(trained_pair, tmp_path_factory):
    """The session's CenterNet detector, exported once for the lone-detector tests."""
    pytest.importorskip("onnxruntime")
    return _export(trained_pair["det"], tmp_path_factory.mktemp("lone_det") / "det")


def _lone_detector_boxes(det, data, out, *extra):
    """Run ``sleap-nn predict`` on a lone exported detector; (score, xyxy) per box."""
    from click.testing import CliRunner

    from sleap_nn.cli import cli

    result = CliRunner().invoke(
        cli,
        ["predict", "-i", str(data), "-m", Path(det).as_posix()]
        + ["--device", "cpu", "-o", str(out), *extra],
    )
    assert result.exit_code == 0, result.output
    boxes = [
        (float(b.score), tuple(float(v) for v in b.xyxy))
        for lf in sio.load_slp(out)
        for b in lf.bboxes
    ]
    return sorted(boxes, reverse=True)


def test_lone_exported_detector_takes_the_thresholds(
    trained_pair, lone_det_export, minimal_instance, tmp_path
):
    """A lone exported box detector decodes with a typed --peak_threshold / --bbox_nms_iou.

    Without the flags it keeps the score threshold stored at export (0.2 by default)
    and the detector style's default suppression (none for CenterNet), as before.
    """

    def run(name, *extra):
        return _lone_detector_boxes(
            lone_det_export, minimal_instance, tmp_path / f"{name}.slp", *extra
        )

    def scores(boxes):
        return [s for s, _ in boxes]

    def flat(boxes):
        return [v for s, box in boxes for v in (s, *box)]

    # A typed threshold of 0 keeps every candidate the decoder ranks.
    everything = run("all", "--peak_threshold", "0.0")
    all_scores = scores(everything)
    distinct = sorted(set(all_scores))
    assert len(distinct) >= 2

    # Without the flags: the exported threshold (0.2), exactly as before.
    kept_by_default = [s for s in all_scores if s >= 0.2]
    assert len(kept_by_default) < len(everything)
    assert scores(run("default")) == pytest.approx(kept_by_default)

    # A typed score threshold between the two best scores keeps the best only.
    thr = (distinct[-2] + distinct[-1]) / 2
    expected = [s for s in all_scores if s >= thr]
    assert 0 < len(expected) < len(everything)
    assert scores(run("typed", "--peak_threshold", repr(thr))) == pytest.approx(
        expected
    )

    # The untyped cut is the threshold stored at export, whatever it is: an export
    # made with `--peak-threshold` between the two best scores keeps the best only,
    # and a typed threshold still replaces it.
    stored = _export(
        trained_pair["det"], tmp_path / "det_stored", "--peak-threshold", repr(thr)
    )

    def run_stored(name, *extra):
        return scores(
            _lone_detector_boxes(
                stored, minimal_instance, tmp_path / f"{name}.slp", *extra
            )
        )

    assert run_stored("stored_default") == pytest.approx(expected)
    assert run_stored("stored_typed", "--peak_threshold", "0.0") == pytest.approx(
        all_scores
    )

    # A typed suppression threshold. 1.0 disables it (as CenterNet's default does); a
    # low one keeps the best box and only boxes the unsuppressed run also has. Whether
    # it removes any of this toy detector's boxes depends on what it learned (its boxes
    # may not overlap), so suppression itself is checked on built outputs below
    # (`test_lone_exported_detector_suppresses_overlapping_boxes`).
    nms_off = run("nms_off", "--peak_threshold", "0.0", "--bbox_nms_iou", "1.0")
    assert flat(nms_off) == pytest.approx(flat(everything))
    suppressed = run("nms", "--peak_threshold", "0.0", "--bbox_nms_iou", "0.05")
    assert 0 < len(suppressed) <= len(everything)
    assert flat(suppressed[:1]) == pytest.approx(flat(everything[:1]))
    for box in suppressed:
        assert any(flat([box]) == pytest.approx(flat([other])) for other in everything)


# Scores of the three boxes `_use_overlapping_detector_outputs` builds: the first two
# overlap (IoU 0.5), the third overlaps neither.
_BUILT_SCORES = (0.9, 0.6, 0.4)


def _use_overlapping_detector_outputs(monkeypatch):
    """Feed a lone exported CenterNet detector's decoder three built boxes.

    The exported graph still runs; only its head outputs are replaced, before the
    detector's own decoder (score threshold, IoU suppression) and the coordinate
    mapping, so the thresholds are applied by the code under test to boxes whose
    overlap is fixed by construction rather than learned.
    """
    from sleap_nn.inference.layers.bbox import ExportedBBoxLayer

    postprocess = ExportedBBoxLayer.postprocess

    def built(self, raw_out, info):
        assert self.decoder.model_type == "bbox_centernet"
        heat = torch.zeros_like(raw_out["BBoxCenterHead"])
        size = torch.zeros_like(raw_out["BBoxSizeHead"])
        _, _, h, w = heat.shape
        # Room for the three boxes inside the grid (this fixture's frames are not
        # letterboxed, so none reaches into padding and gets clipped).
        assert h >= 16 and w >= 32, (h, w)
        y = h // 2
        # Two 6x6-cell boxes 2 cells apart (IoU 24 / 48 = 0.5), as separate peaks of
        # the 3x3 peak NMS; a third one far from both.
        for x, score in zip((w // 4, w // 4 + 2, 3 * w // 4), _BUILT_SCORES):
            heat[:, 0, y, x] = score
            size[:, :, y, x] = 6.0
        out = dict(raw_out, BBoxCenterHead=heat, BBoxSizeHead=size)
        if out.get("BBoxOffsetHead") is not None:
            out["BBoxOffsetHead"] = torch.zeros_like(out["BBoxOffsetHead"])
        return postprocess(self, out, info)

    monkeypatch.setattr(ExportedBBoxLayer, "postprocess", built)


@pytest.fixture
def overlapping_detector_outputs(monkeypatch):
    """:func:`_use_overlapping_detector_outputs` for one test."""
    _use_overlapping_detector_outputs(monkeypatch)


def test_lone_exported_detector_suppresses_overlapping_boxes(
    lone_det_export, minimal_instance, tmp_path, overlapping_detector_outputs
):
    """A typed --bbox_nms_iou suppresses a lone exported detector's overlapping boxes."""

    def scores(name, *extra):
        boxes = _lone_detector_boxes(
            lone_det_export, minimal_instance, tmp_path / f"{name}.slp", *extra
        )
        return [s for s, _ in boxes]

    high, low, apart = _BUILT_SCORES
    # CenterNet's default: no suppression.
    assert scores("default") == pytest.approx([high, low, apart])
    assert scores("off", "--bbox_nms_iou", "1.0") == pytest.approx([high, low, apart])
    # Below the pair's IoU (0.5) the weaker of the two goes; above it both stay.
    assert scores("low", "--bbox_nms_iou", "0.3") == pytest.approx([high, apart])
    assert scores("high", "--bbox_nms_iou", "0.7") == pytest.approx([high, low, apart])
    # The score threshold applies as well.
    assert scores("both", "--peak_threshold", "0.5", "--bbox_nms_iou", "0.3") == (
        pytest.approx([high])
    )


def _lone_detector_decoder(monkeypatch, det, data, out, *flags):
    """Run a lone exported detector on the command line; what reached its decoder.

    Returns the two threshold keywords ``Predictor.from_export_dir`` was called with
    (only those present) and the decoder it built.
    """
    from sleap_nn.inference.predictor import Predictor

    seen = {}
    build = Predictor.from_export_dir.__func__

    def spy(cls, export_dir, **kwargs):
        seen.update(kwargs)
        predictor = build(cls, export_dir, **kwargs)
        seen["decoder"] = predictor.layer.decoder
        return predictor

    monkeypatch.setattr(Predictor, "from_export_dir", classmethod(spy))
    _lone_detector_boxes(det, data, out, *flags)
    decoder = seen.pop("decoder")
    typed = {k: seen[k] for k in ("peak_conf_threshold", "bbox_nms_iou") if k in seen}
    return typed, decoder


@pytest.mark.parametrize(
    "flags, expected",
    [
        ((), {}),
        (("--peak_threshold", "0.6"), {"peak_conf_threshold": 0.6}),
        (("--bbox_nms_iou", "0.3"), {"bbox_nms_iou": 0.3}),
        (
            ("--peak_threshold", "0.2", "--bbox_nms_iou", "0.3"),
            {"peak_conf_threshold": 0.2, "bbox_nms_iou": 0.3},
        ),
    ],
)
def test_lone_exported_detector_forwards_only_typed_thresholds(
    lone_det_export, minimal_instance, tmp_path, monkeypatch, flags, expected
):
    """Only thresholds typed on the command line reach the lone detector's decoder."""
    typed, decoder = _lone_detector_decoder(
        monkeypatch, lone_det_export, minimal_instance, tmp_path / "o.slp", *flags
    )
    assert typed == pytest.approx(expected)
    assert decoder.score_threshold == pytest.approx(
        expected.get("peak_conf_threshold", 0.2)
    )
    assert decoder.iou_threshold == expected.get("bbox_nms_iou")


@pytest.mark.parametrize("flags, iou", [((), 0.45), (("--bbox_nms_iou", "0.3"), 0.3)])
def test_lone_exported_ssd_detector_suppression(
    config, minimal_instance, tmp_path, monkeypatch, flags, iou
):
    """An SSD detector's default suppression (0.45) stays unless --bbox_nms_iou is typed."""
    pytest.importorskip("onnxruntime")
    from sleap_nn.training.model_trainer import ModelTrainer

    cfg = _train_config(config, tmp_path, "bbox_ssd", "det_ssd")
    cfg.trainer_config.max_epochs = 2
    ModelTrainer.get_model_trainer_from_config(cfg).train()
    det = _export(tmp_path / "det_ssd", tmp_path / "det_ssd_export")
    typed, decoder = _lone_detector_decoder(
        monkeypatch, det, minimal_instance, tmp_path / "o.slp", *flags
    )
    assert typed == ({"bbox_nms_iou": iou} if flags else {})
    assert decoder.iou_threshold == pytest.approx(iou)
    assert decoder.score_threshold == pytest.approx(0.2)


def test_lone_exported_detector_thresholds_from_python(
    lone_det_export, minimal_instance, monkeypatch
):
    """The Python calls take the same two thresholds for a lone exported detector."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.run import predict

    p = Predictor.from_export_dir(lone_det_export, runtime="onnx", device="cpu")
    assert p.layer.decoder.score_threshold == pytest.approx(0.2)
    assert p.layer.decoder.iou_threshold is None
    p = Predictor.from_export_dir(
        lone_det_export,
        runtime="onnx",
        device="cpu",
        peak_conf_threshold=0.5,
        bbox_nms_iou=0.123,
    )
    assert p.layer.decoder.score_threshold == pytest.approx(0.5)
    assert p.layer.decoder.iou_threshold == pytest.approx(0.123)

    def n_boxes(peak_threshold=0.0, **kwargs):
        labels = predict(
            str(minimal_instance),
            export_dir=str(lone_det_export),
            device="cpu",
            peak_threshold=peak_threshold,
            **kwargs,
        )
        return sum(len(lf.bboxes) for lf in labels)

    # On the trained detector, a smoke check only: whether a low IoU threshold
    # removes a box depends on what the toy detector learned.
    assert 0 < n_boxes(bbox_nms_iou=0.05) <= n_boxes()
    # On built outputs (scores 0.9, 0.6, 0.4; the first two overlap with IoU 0.5),
    # both thresholds reach `predict`'s decoder.
    _use_overlapping_detector_outputs(monkeypatch)
    assert n_boxes() == 3
    assert n_boxes(bbox_nms_iou=0.3) == 2
    assert n_boxes(peak_threshold=0.5) == 2
    assert n_boxes(peak_threshold=0.5, bbox_nms_iou=0.3) == 1


def test_lone_network_only_export_runs_as_its_checkpoint(
    trained_pair, minimal_instance, tmp_path
):
    """A lone --network-only crop export takes its checkpoint's route (#857)."""
    pytest.importorskip("onnxruntime")
    from click.testing import CliRunner

    from sleap_nn.cli import cli

    ci = _export(trained_pair["ci_bbox"], tmp_path / "ci", "--network-only")
    outs = {}
    for name, path in (("export", ci), ("checkpoint", trained_pair["ci_bbox"])):
        result = CliRunner().invoke(
            cli,
            ["predict", "-i", str(minimal_instance), "-m", path.as_posix()]
            + ["--device", "cpu", "-o", str(tmp_path / f"{name}.slp")],
        )
        outs[name] = (result.exit_code, type(result.exception).__name__)
    # Same outcome as the checkpoint alone, not the fused-export route's error.
    assert outs["export"] == outs["checkpoint"], outs


@pytest.mark.parametrize("legacy", [True, False])
def test_gt_boxes_ignore_stale_hidden_points_of_old_files(
    minimal_instance, monkeypatch, legacy
):
    """Boxes from labelled keypoints skip hidden points' stored coordinates for
    labels read from a pre-3.0 file (as training does); the labels are unchanged."""
    from types import SimpleNamespace

    from sleap_nn.data import providers
    from sleap_nn.inference.predictor import Predictor

    monkeypatch.setattr(providers, "has_legacy_hidden_points", lambda labels: legacy)
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    inst = lf.user_instances[0]
    inst.points["visible"][0] = False  # hidden, stale coordinates kept
    stale_xy = inst.points["xy"][0].copy()
    predictor = Predictor.__new__(Predictor)
    predictor.layer = SimpleNamespace(
        centroid_layer=SimpleNamespace(use_gt_boxes=True),
        bbox_crop=SimpleNamespace(bbox_source="instances", bbox_padding=0.0),
    )
    box = predictor._gt_box_fn(labels)(lf, inst)
    visible_xy = inst.points["xy"][1]
    if legacy:
        # Only the visible point is left: a degenerate box on it.
        np.testing.assert_allclose(box, [*visible_xy, *visible_xy], atol=1e-4)
    else:
        assert box[0] <= min(stale_xy[0], visible_xy[0]) + 1e-4
    np.testing.assert_array_equal(inst.points["xy"][0], stale_xy)  # unchanged
