"""End to end on CPU: a ``bbox_detr`` detector as the first stage of top-down pose.

The configs come from the config generator's ``topdown_bbox_detr`` pipeline (the
same defaults as the mouse sample config), with a tiny random DINOv3 ViT in place
of the pretrained one: the detector trains on boxes derived from the poses, the
centered-instance model on the same padded boxes, and ``sleap-nn predict`` runs
the pair, then ``sleap-nn eval`` scores it.
"""

import pytest
import sleap_io as sio
from click.testing import CliRunner
from omegaconf import OmegaConf

from sleap_nn.config_generator.generator import ConfigGenerator


def _cpu(cfg, tmp_path, run_name, epochs):
    trainer = cfg.trainer_config
    trainer.trainer_accelerator = "cpu"
    trainer.trainer_devices = 1
    trainer.max_epochs = epochs
    trainer.min_train_steps_per_epoch = 4
    trainer.precision = "32"
    trainer.ckpt_dir = str(tmp_path)
    trainer.run_name = run_name
    trainer.train_data_loader.batch_size = 2
    trainer.val_data_loader.batch_size = 2
    trainer.visualize_preds_during_training = False
    cfg.data_config.val_labels_path = []
    cfg.data_config.validation_fraction = 0.0
    cfg.data_config.use_same_data_for_val = True
    return cfg


@pytest.fixture(scope="module")
def detr_pair(tmp_path_factory):
    """A tiny bbox_detr detector + box-scaled centered-instance model."""
    pytest.importorskip("transformers")
    from transformers import DINOv3ViTConfig

    from sleap_nn.training.model_trainer import ModelTrainer

    tmp_path = tmp_path_factory.mktemp("detr_topdown")
    vit_dir = tmp_path / "tiny_dinov3"
    DINOv3ViTConfig(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=64,
        patch_size=16,
        num_register_tokens=4,
    ).save_pretrained(vit_dir)

    gen = (
        ConfigGenerator.from_slp("tests/assets/datasets/minimal_instance.pkg.slp")
        .auto()
        .pipeline("topdown_bbox_detr")
        .detr(max_side=256)
    )
    det = _cpu(gen.build_detector(), tmp_path, "det", epochs=3)
    vit = det.model_config.backbone_config.pretrained
    vit.model_name = vit_dir.as_posix()
    vit.weights = False
    vit.layers = [1, 2, 3, 4]
    vit.image_mean = [0.485, 0.456, 0.406]
    vit.image_std = [0.229, 0.224, 0.225]
    det.trainer_config.ema.warmup_steps = 1
    ci = _cpu(gen.build_centered_instance(), tmp_path, "ci", epochs=3)
    ci.trainer_config.optimizer.lr = 1e-3

    for cfg in (det, ci):
        ModelTrainer.get_model_trainer_from_config(cfg).train()
    return {"det": tmp_path / "det", "ci": tmp_path / "ci", "gen": gen}


def test_detector_trained_on_padded_pose_boxes(detr_pair):
    saved = OmegaConf.load(detr_pair["det"] / "training_config.yaml")
    assert saved.model_config.head_configs.bbox_detr.detr.categories == ["object"]
    assert saved.model_config.backbone_config.pretrained.mode == "tokens"
    assert saved.data_config.bbox_source == "auto"
    ci = OmegaConf.load(detr_pair["ci"] / "training_config.yaml")
    assert ci.data_config.bbox_padding == saved.data_config.bbox_padding > 0
    assert ci.data_config.preprocessing.crop_mode == "bbox"
    assert (detr_pair["det"] / "best.ckpt").exists()


def test_predict_and_eval_with_detr_stage_1(detr_pair, tmp_path):
    """sleap-nn predict -m <bbox_detr> -m <centered_instance>, then eval."""
    from sleap_nn.cli import cli

    labels_path = "tests/assets/datasets/minimal_instance.pkg.slp"
    out = tmp_path / "pred.slp"
    result = CliRunner().invoke(
        cli,
        [
            "predict",
            "-i",
            labels_path,
            "-m",
            detr_pair["det"].as_posix(),
            "-m",
            detr_pair["ci"].as_posix(),
            "-o",
            str(out),
            "--peak_threshold",
            "0.0",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    lf = sio.load_slp(out)[0]
    # Every pose instance was cropped from a detected box, which stays linked.
    assert lf.instances and len(lf.bboxes) >= len(lf.instances)
    linked = {id(b.instance) for b in lf.bboxes if b.instance is not None}
    assert linked == {id(inst) for inst in lf.instances}
    assert all(isinstance(b, sio.PredictedBoundingBox) for b in lf.bboxes)

    result = CliRunner().invoke(
        cli, ["eval", "--ground_truth_path", labels_path, "--predicted_path", str(out)]
    )
    assert result.exit_code == 0, result.output


def test_detect_boxes_with_animal_detector(detr_pair, tmp_path):
    """sleap-nn detect-boxes writes an animal detector's boxes into the labels.

    The single-class detector's class ("object") is used as the requested
    category, and its boxes are matched to the pose instances' keypoint extents.
    """
    from sleap_nn.cli import cli

    out = tmp_path / "train.mouse.slp"
    result = CliRunner().invoke(
        cli,
        [
            "detect-boxes",
            "-i",
            "tests/assets/datasets/minimal_instance.pkg.slp",
            "-m",
            detr_pair["det"].as_posix(),
            "-o",
            str(out),
            "--category",
            "mouse",
            "--score_threshold",
            "0.0",
            "--min_iou",
            "0.0",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    labels = sio.load_slp(out)
    boxes = [b for lf in labels for b in lf.bboxes]
    assert boxes and all(isinstance(b, sio.PredictedBoundingBox) for b in boxes)
    assert {b.category.name for b in boxes} == {"mouse"}
    assert any(b.instance is not None for b in boxes)
    assert labels.provenance["detector_boxes"]
