"""Box-detector distillation (`trainer_config.distillation` on `bbox_*` models)."""

from pathlib import Path

import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf

from sleap_nn.config.training_job_config import verify_training_cfg
from sleap_nn.data.bboxes import add_bboxes_to_labels, get_bbox_target_encoder
from sleap_nn.training.distillation import (
    BoxDistiller,
    attach_distillation,
    load_teacher_model,
)
from sleap_nn.training.lightning_modules import LightningModel

SSD_ANCHORS = [32.0, 64.0, 128.0]


def _detr_cfg(config, tmp_path, run_name):
    from tests.test_bbox_detr_training import _detr_train_config

    cfg = _detr_train_config(config, tmp_path, run_name=run_name)
    cfg.trainer_config.eval = {"enabled": False}
    cfg.trainer_config.visualize_preds_during_training = False
    cfg.trainer_config.ema = {"enabled": False}
    return cfg


def _ssd_cfg(config, tmp_path, run_name="ssd"):
    from tests.test_bbox_detection import _bbox_train_config

    cfg = _bbox_train_config(config, "bbox_ssd", tmp_path)
    cfg.model_config.head_configs.bbox_ssd.ssd.anchor_sizes = SSD_ANCHORS
    cfg.trainer_config.run_name = run_name
    cfg.trainer_config.eval = {"enabled": False}
    cfg.trainer_config.visualize_preds_during_training = False
    return cfg


def _save_model_dir(cfg, out_dir, seed=0):
    """A model directory (best.ckpt + training_config.yaml) of a fresh model."""
    torch.manual_seed(seed)
    cfg = verify_training_cfg(cfg)
    module = LightningModel.get_lightning_model_from_config(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": module.state_dict()}, out_dir / "best.ckpt")
    OmegaConf.save(cfg, out_dir / "training_config.yaml")
    return out_dir


def _ssd_batch(minimal_instance, b=2):
    """A collated `bbox_ssd` training batch (the minimal frame, b times)."""
    from sleap_nn.config.get_config import get_head_configs
    from sleap_nn.data.custom_datasets import BBoxDataset

    labels = sio.load_slp(minimal_instance)
    add_bboxes_to_labels(labels)
    hc = OmegaConf.structured(get_head_configs("bbox_ssd"))
    hc.bbox_ssd.ssd.output_stride = 8
    hc.bbox_ssd.ssd.anchor_sizes = SSD_ANCHORS
    ds = BBoxDataset(
        labels=[labels],
        target_encoder=get_bbox_target_encoder("bbox_ssd", hc),
        categories=["object"],
        max_stride=32,
    )
    return torch.utils.data.default_collate([ds[0]] * b)


def test_box_student_with_a_detr_teacher(config, tmp_path, minimal_instance):
    """A DETR teaches an SSD: box term from the teacher's boxes, feature term."""
    teacher = _save_model_dir(_detr_cfg(config, tmp_path, "t"), tmp_path / "teacher")
    torch.manual_seed(0)
    student = LightningModel.get_lightning_model_from_config(
        verify_training_cfg(_ssd_cfg(config, tmp_path))
    )
    d = attach_distillation(
        student, {"teacher": str(teacher), "teacher_score_threshold": 0.0}
    )
    assert isinstance(d, BoxDistiller) and student.distiller is d
    assert d.teacher_type == "bbox_detr" and d.class_map == [0]
    assert d.projector is not None and not d.teacher.training
    assert all(not p.requires_grad for p in d.teacher.parameters())

    batch = _ssd_batch(minimal_instance)
    X = torch.rand(2, 1, 384, 384)
    detections, feat = d.run_box_teacher(X)
    assert len(detections) == 2 and feat is not None
    for boxes, classes in detections:
        assert boxes.shape[1] == 4 and len(boxes) == len(classes) > 0
        assert (boxes[:, 2:] > boxes[:, :2]).all() and boxes.max() <= 384
        assert (classes == 0).all()
    targets = d.pseudo_targets(detections, (384, 384), X.device)
    assert targets["ssd_cls_target"].shape == batch["ssd_cls_target"].shape
    assert (targets["ssd_cls_target"] > 0).any()

    torch.manual_seed(1)
    total = student.training_step(batch, 0)
    distiller, student.distiller = student.distiller, None
    try:
        torch.manual_seed(1)
        supervised = student.training_step(batch, 0)
    finally:
        student.distiller = distiller
    assert torch.isfinite(total) and total.item() != pytest.approx(supervised.item())
    total.backward()
    assert d.projector.weight.grad is not None
    assert all(p.grad is None for p in d.teacher.parameters())
    # Without the feature term: no projector, box term only.
    student.distiller = None
    d2 = attach_distillation(
        student, {"teacher": str(teacher), "alpha": 0.0, "teacher_score_threshold": 0}
    )
    assert d2.projector is None
    terms, n_boxes = d2.box_loss_terms(student, X, student.model(X), None)
    assert set(terms) == {"box"} and n_boxes > 0


def test_box_distillation_errors(
    config, tmp_path, minimal_instance_centered_instance_ckpt
):
    teacher = _save_model_dir(_detr_cfg(config, tmp_path, "t"), tmp_path / "teacher")
    student = LightningModel.get_lightning_model_from_config(
        verify_training_cfg(_ssd_cfg(config, tmp_path))
    )
    with pytest.raises(ValueError, match="stage 1"):
        attach_distillation(student, {"teacher": str(teacher), "stage": 2})
    with pytest.raises(ValueError, match="box-detector teacher"):
        attach_distillation(
            student, {"teacher": str(minimal_instance_centered_instance_ckpt)}
        )
    with pytest.raises(ValueError, match="teacher_crop_size"):
        attach_distillation(
            student, {"teacher": str(teacher), "teacher_crop_size": [64, 64]}
        )
    # Categories are matched by name once either side has several.
    cfg = _ssd_cfg(config, tmp_path)
    cfg.model_config.head_configs.bbox_ssd.ssd.categories = ["a", "b"]
    two = LightningModel.get_lightning_model_from_config(verify_training_cfg(cfg))
    with pytest.raises(ValueError, match="share no box category"):
        attach_distillation(two, {"teacher": str(teacher)})
    cfg.model_config.head_configs.bbox_ssd.ssd.categories = ["b", "object"]
    named = LightningModel.get_lightning_model_from_config(verify_training_cfg(cfg))
    assert attach_distillation(named, {"teacher": str(teacher)}).class_map == [1]


def test_detr_student_trains_with_a_detr_teacher(config, tmp_path):
    """End to end: a DETR distilled from another; checkpoints load plainly."""
    from sleap_nn.training.model_trainer import ModelTrainer

    teacher = _save_model_dir(_detr_cfg(config, tmp_path, "t"), tmp_path / "teacher")
    cfg = _detr_cfg(config, tmp_path, "student")
    cfg.model_config.head_configs.bbox_detr.detr.dec_layers = 1  # a smaller student
    cfg.trainer_config.max_epochs = 1
    cfg.trainer_config.distillation = {
        "teacher": str(teacher),
        "teacher_score_threshold": 0.0,
    }
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    assert isinstance(trainer.lightning_model.distiller, BoxDistiller)
    run_dir = tmp_path / "student"
    metrics = trainer.trainer.callback_metrics
    for key in ("box", "feat", "teacher_boxes", "weight"):
        assert f"train/distill_{key}" in metrics
    assert metrics["train/distill_teacher_boxes"] > 0
    saved = OmegaConf.load(run_dir / "training_config.yaml")
    assert saved.trainer_config.distillation.teacher_score_threshold == 0.0
    # The projector is not in the state dict: the student loads as a plain model.
    best = torch.load(run_dir / "best.ckpt", map_location="cpu", weights_only=False)
    assert "distillation" in best
    assert not any(k.startswith("distiller.") for k in best["state_dict"])
    model, _ = load_teacher_model(str(run_dir))  # strict load
    assert model.model_type == "bbox_detr"
