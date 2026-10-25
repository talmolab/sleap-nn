"""Tests for DWPose-style distillation (`trainer_config.distillation`)."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from sleap_nn.config.trainer_config import DistillationConfig, TrainerConfig
from sleap_nn.config.training_job_config import verify_training_cfg
from sleap_nn.training.distillation import (
    CHECKPOINT_KEY,
    TEACHER_IMAGE_KEY,
    Distiller,
    attach_distillation,
    load_teacher_model,
    shared_node_channels,
)
from sleap_nn.training.lightning_modules import LightningModel

requires_transformers = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="needs the `transformers` package (the `backbones` extra)",
)

HEAD = "CenteredInstanceConfmapsHead"


def _ci_config(ckpt_dir, part_names=None, filters=None):
    cfg = OmegaConf.load(Path(ckpt_dir) / "training_config.yaml")
    if part_names is not None:
        cfg.model_config.head_configs.centered_instance.confmaps.part_names = list(
            part_names
        )
    if filters is not None:
        cfg.model_config.backbone_config.unet.filters = filters
    return cfg


def _student(ckpt_dir, part_names=None, filters=8, seed=0):
    torch.manual_seed(seed)
    cfg = _ci_config(ckpt_dir, part_names=part_names, filters=filters)
    return LightningModel.get_lightning_model_from_config(cfg)


def _batch(n_nodes, b=2, size=96, node_weights=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    batch = {
        "instance_image": torch.rand(b, 1, 1, size, size, generator=g),
        "confidence_maps": torch.rand(b, 1, n_nodes, size // 2, size // 2, generator=g),
    }
    if node_weights is not None:
        batch["node_weights"] = torch.as_tensor(node_weights, dtype=torch.float32)
    return batch


def _supervised(student, batch):
    """The student's own training step with distillation switched off."""
    distiller, student.distiller = student.distiller, None
    try:
        return student.training_step(batch, 0)
    finally:
        student.distiller = distiller


# ------------------------------------------------------------------ config


def test_config_defaults_off():
    assert TrainerConfig().distillation is None
    cfg = DistillationConfig(teacher="t")
    assert (cfg.stage, cfg.alpha, cfg.beta, cfg.gamma) == (1, 1.0, 1.0, 1.0)
    assert cfg.supervised_weight is None and cfg.decay is None
    assert cfg.mask_unlabeled_nodes and cfg.teacher_input_scale == 1.0
    with pytest.raises(ValueError, match="needs a `teacher`"):
        DistillationConfig()
    DistillationConfig(stage=2)  # self-distillation needs no teacher
    with pytest.raises(ValueError):
        DistillationConfig(teacher="t", stage=3)


def test_config_teacher_crop_size():
    assert DistillationConfig(teacher="t").teacher_crop_size is None
    assert DistillationConfig(teacher="t", teacher_crop_size=384).teacher_crop_size
    cfg = DistillationConfig(teacher="t", teacher_crop_size=[384, 288])
    assert list(cfg.teacher_crop_size) == [384, 288]
    for bad in ([384], [384, 0], "big", [1.5, 2]):
        with pytest.raises(ValueError, match="int or"):
            DistillationConfig(teacher="t", teacher_crop_size=bad)
    with pytest.raises(ValueError, match="stage-1 option"):
        DistillationConfig(stage=2, teacher_crop_size=[384, 288])
    with pytest.raises(ValueError, match="not both"):
        DistillationConfig(
            teacher="t", teacher_crop_size=[384, 288], teacher_input_scale=1.5
        )


def test_config_key_validates(config):
    cfg = config.copy()
    cfg.trainer_config.distillation = {"teacher": "x", "alpha": 0.5}
    out = verify_training_cfg(cfg)
    assert out.trainer_config.distillation.alpha == 0.5
    assert out.trainer_config.distillation.trainable is None
    cfg.trainer_config.distillation = {"stage": 1}
    with pytest.raises(ValueError, match="needs a `teacher`"):
        verify_training_cfg(cfg)


# ------------------------------------------------------------------ pieces


def test_model_returns_backbone_outputs(minimal_instance_centered_instance_ckpt):
    model = _student(minimal_instance_centered_instance_ckpt).model.eval()
    x = torch.rand(2, 1, 96, 96)
    plain = model(x)
    out, bb = model(x, return_backbone_outputs=True)
    assert torch.equal(plain[HEAD], out[HEAD])
    assert {"outputs", "strides", "middle_output"} <= set(bb)
    assert torch.equal(model.forward_heads(bb)[HEAD], plain[HEAD])


def test_shared_node_channels_by_name():
    assert shared_node_channels(["B", "C", "A"], ["A", "B"]) == ([0, 2], [1, 0])
    with pytest.raises(ValueError, match="share no node"):
        shared_node_channels(["x"], ["A", "B"])


def test_load_teacher_is_frozen_and_exact(minimal_instance_centered_instance_ckpt):
    ckpt = minimal_instance_centered_instance_ckpt
    teacher, cfg = load_teacher_model(str(ckpt))
    assert not teacher.training
    assert not any(p.requires_grad for p in teacher.parameters())
    sd = torch.load(ckpt / "best.ckpt", map_location="cpu", weights_only=False)
    for k, v in teacher.state_dict().items():
        assert torch.equal(v, sd["state_dict"][f"model.{k}"]), k
    # A `.ckpt` path works too.
    load_teacher_model(str(ckpt / "best.ckpt"))
    with pytest.raises(FileNotFoundError):
        load_teacher_model(str(ckpt / "nope.ckpt"))


# ------------------------------------------------------------------ stage 1


def test_stage1_teacher_outside_module_tree(minimal_instance_centered_instance_ckpt):
    ckpt = minimal_instance_centered_instance_ckpt
    student = _student(ckpt, part_names=["B", "C", "A"])
    n_params = sum(1 for _ in student.parameters())
    d = attach_distillation(student, {"teacher": str(ckpt)})
    assert student.distiller is d
    assert (d.student_idx, d.teacher_idx) == ([0, 2], [1, 0])
    # Only the projector joins the student; the teacher stays out of
    # parameters / state_dict / modules.
    assert sum(1 for _ in student.parameters()) == n_params + 2
    keys = [k for k in student.state_dict() if k.startswith("distiller.")]
    assert keys == ["distiller.projector.weight", "distiller.projector.bias"]
    teacher_params = {id(p) for p in d.teacher.parameters()}
    assert not teacher_params & {id(p) for p in student.parameters()}
    assert all(m is not d.teacher for m in student.modules())
    # Mode and device/dtype follow the student, but eval/frozen stays.
    student.train()
    assert not d.teacher.training
    student.to(torch.float64)
    assert next(d.teacher.parameters()).dtype == torch.float64


def test_stage1_loss_and_gradients(minimal_instance_centered_instance_ckpt):
    ckpt = minimal_instance_centered_instance_ckpt
    student = _student(ckpt, part_names=["B", "C", "A"])
    d = attach_distillation(student, {"teacher": str(ckpt), "alpha": 0.5})
    batch = _batch(3)
    supervised = _supervised(student, batch)
    total = student.training_step(batch, 0)

    x = torch.squeeze(batch["instance_image"], dim=1)
    out, bb = student.model(x, return_backbone_outputs=True)
    terms = d.loss_terms(x, out, bb, batch, None)
    t_out, t_bb = d.teacher(x, return_backbone_outputs=True)
    expected_hm = F.mse_loss(out[HEAD][:, [0, 2]], t_out[HEAD][:, [1, 0]])
    torch.testing.assert_close(terms["heatmap"], expected_hm)
    expected_feat = F.mse_loss(d.projector(bb["middle_output"]), t_bb["middle_output"])
    torch.testing.assert_close(terms["feat"], expected_feat)
    torch.testing.assert_close(
        total, supervised + 0.5 * terms["feat"] + 1.0 * terms["heatmap"]
    )

    total.backward()
    assert d.projector.weight.grad is not None
    assert d.projector.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in d.teacher.parameters())


def test_stage1_identical_teacher_gives_zero_heatmap_term(
    minimal_instance_centered_instance_ckpt,
):
    """A student that IS the teacher (same weights, same nodes) matches exactly."""
    ckpt = minimal_instance_centered_instance_ckpt
    student = LightningModel.get_lightning_model_from_config(_ci_config(ckpt))
    sd = torch.load(ckpt / "best.ckpt", map_location="cpu", weights_only=False)
    student.load_state_dict(sd["state_dict"])
    d = attach_distillation(student, {"teacher": str(ckpt), "alpha": 0.0})
    assert d.projector is None
    student.eval()
    x = torch.rand(2, 1, 96, 96)
    out, bb = student.model(x, return_backbone_outputs=True)
    terms = d.loss_terms(x, out, bb, {}, None)
    assert set(terms) == {"heatmap"}
    assert terms["heatmap"].item() == pytest.approx(0.0, abs=1e-12)


def test_heatmap_term_respects_node_weights(minimal_instance_centered_instance_ckpt):
    ckpt = minimal_instance_centered_instance_ckpt
    student = _student(ckpt, part_names=["B", "C", "A"])
    d = attach_distillation(student, {"teacher": str(ckpt), "alpha": 0.0})
    x = torch.rand(2, 1, 96, 96)
    out, bb = student.model(x, return_backbone_outputs=True)
    t = d.teacher(x)[HEAD][:, [1, 0]]
    s = out[HEAD][:, [0, 2]]
    # Node "A" (student channel 2) is unlabeled in sample 0.
    w = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    masked = d.loss_terms(x, out, bb, {}, w)["heatmap"]
    per = ((s - t) ** 2).mean(dim=(2, 3))  # (B, shared)
    expected = (per[0, 0] + per[1, 0] + per[1, 1]) / 3
    torch.testing.assert_close(masked, expected)
    d.mask_unlabeled_nodes = False
    torch.testing.assert_close(d.loss_terms(x, out, bb, {}, w)["heatmap"], per.mean())
    # Through the training step, from the batch's `node_weights`.
    d.mask_unlabeled_nodes = True
    student.training_step(_batch(3, node_weights=w[:, None, :]), 0)


def test_teacher_input_scale_and_teacher_image(minimal_instance_centered_instance_ckpt):
    ckpt = minimal_instance_centered_instance_ckpt
    student = _student(ckpt)
    d = attach_distillation(student, {"teacher": str(ckpt), "teacher_input_scale": 1.5})
    x = torch.rand(2, 1, 96, 96)
    xt = d.teacher_input(x, {})
    assert xt.shape[-2:] == (144, 144)  # 96 * 1.5, a multiple of max_stride 8
    d.teacher_input_scale = 1.4
    assert d.teacher_input(x, {}).shape[-2:] == (136, 136)  # rounded to stride
    # A dual-resolution crop in the batch wins over resampling.
    image = torch.full((2, 1, 1, 128, 128), 255, dtype=torch.uint8)
    xt = d.teacher_input(x, {TEACHER_IMAGE_KEY: image})
    assert xt.shape == (2, 1, 128, 128) and xt.max() == 1.0
    # Outputs on the teacher's bigger grid are resized back to the student's.
    out, bb = student.model(x, return_backbone_outputs=True)
    terms = d.loss_terms(x, out, bb, {}, None)
    assert terms["heatmap"].ndim == 0 and terms["feat"].ndim == 0


def test_teacher_crop_size_uses_the_dual_resolution_crop(
    minimal_instance_centered_instance_ckpt,
):
    ckpt = minimal_instance_centered_instance_ckpt
    student = _student(ckpt)
    d = attach_distillation(
        student, {"teacher": str(ckpt), "teacher_crop_size": [144, 136]}, 96
    )
    assert d.teacher_crop_size == (144, 136) and d.teacher_input_scale == 1.0
    x = torch.rand(2, 1, 96, 96)
    image = torch.full((2, 1, 1, 144, 136), 51, dtype=torch.uint8)
    xt = d.teacher_input(x, {TEACHER_IMAGE_KEY: image})
    assert xt.shape == (2, 1, 144, 136)
    assert torch.allclose(xt, torch.full_like(xt, 0.2))
    out, bb = student.model(x, return_backbone_outputs=True)
    terms = d.loss_terms(x, out, bb, {TEACHER_IMAGE_KEY: image}, None)
    assert terms["heatmap"].ndim == 0 and terms["feat"].ndim == 0
    # Never a silent fallback to the student's crop at the wrong resolution.
    with pytest.raises(ValueError, match="no `teacher_image`"):
        d.teacher_input(x, {})
    with pytest.raises(ValueError, match="not distillation.teacher_crop_size"):
        d.teacher_input(x, {TEACHER_IMAGE_KEY: image[..., :128, :128]})
    # The teacher crop must fit the teacher's stride.
    with pytest.raises(ValueError, match="multiple of the teacher's max_stride 8"):
        attach_distillation(
            _student(ckpt), {"teacher": str(ckpt), "teacher_crop_size": 140}
        )


def test_decay_schedule(minimal_instance_centered_instance_ckpt):
    ckpt = minimal_instance_centered_instance_ckpt
    d = attach_distillation(_student(ckpt), {"teacher": str(ckpt)})
    assert d.decay == "linear"
    assert d.weight(0, 100) == 1.0
    assert d.weight(25, 100) == pytest.approx(0.75)
    assert d.weight(150, 100) == 0.0
    assert d.weight(10, None) == 1.0  # no trainer: no schedule
    d.decay = "none"
    assert d.weight(50, 100) == 1.0


def test_unsupported_model_type(minimal_instance_centroid_ckpt):
    cfg = OmegaConf.load(Path(minimal_instance_centroid_ckpt) / "training_config.yaml")
    model = LightningModel.get_lightning_model_from_config(cfg)
    with pytest.raises(ValueError, match="Distillation supports"):
        attach_distillation(model, {"teacher": str(minimal_instance_centroid_ckpt)})


# -------------------------------------------------------------- checkpoints


def test_checkpoint_strips_projector_and_resume_restores(
    minimal_instance_centered_instance_ckpt,
):
    ckpt = minimal_instance_centered_instance_ckpt
    student = _student(ckpt)
    d = attach_distillation(student, {"teacher": str(ckpt)})
    with torch.no_grad():
        d.projector.weight.fill_(0.25)
    checkpoint = {"state_dict": student.state_dict()}
    student.on_save_checkpoint(checkpoint)
    assert not any(k.startswith("distiller.") for k in checkpoint["state_dict"])
    saved = checkpoint[CHECKPOINT_KEY]["state_dict"]
    assert torch.all(saved["distiller.projector.weight"] == 0.25)
    # A plain model loads the checkpoint strictly.
    plain = _student(ckpt)
    plain.on_load_checkpoint(checkpoint)
    plain.load_state_dict(checkpoint["state_dict"], strict=True)

    # Resuming with distillation puts the trained projector back.
    resumed = _student(ckpt, seed=1)
    attach_distillation(resumed, {"teacher": str(ckpt)})
    ck = {
        "state_dict": dict(checkpoint["state_dict"]),
        **{CHECKPOINT_KEY: checkpoint[CHECKPOINT_KEY]},
    }
    resumed.on_load_checkpoint(ck)
    resumed.load_state_dict(ck["state_dict"], strict=True)
    assert torch.all(resumed.distiller.projector.weight == 0.25)

    # Starting distillation from a plain checkpoint keeps the fresh projector.
    fresh = _student(ckpt, seed=2)
    attach_distillation(fresh, {"teacher": str(ckpt)})
    w0 = fresh.distiller.projector.weight.clone()
    ck = {"state_dict": plain.state_dict()}
    fresh.on_load_checkpoint(ck)
    fresh.load_state_dict(ck["state_dict"], strict=True)
    assert torch.equal(fresh.distiller.projector.weight, w0)


# ------------------------------------------------------------------ stage 2


def test_stage2_head_self_distillation(minimal_instance_centered_instance_ckpt):
    ckpt = minimal_instance_centered_instance_ckpt
    student = LightningModel.get_lightning_model_from_config(_ci_config(ckpt))
    x = torch.rand(2, 1, 96, 96)
    sd = torch.load(ckpt / "best.ckpt", map_location="cpu", weights_only=False)
    student.load_state_dict(sd["state_dict"])
    student.eval()
    with torch.no_grad():
        before = student.model(x)[HEAD]

    d = attach_distillation(student, {"stage": 2, "teacher": str(ckpt)})
    assert d.stage == 2 and d.shared_features and d.frozen_backbone
    assert d.supervised_weight == 0.0 and d.projector is None
    assert d.decay == "none"  # the KD term is the whole loss
    trainable = [n for n, p in student.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("model.head_layers.") for n in trainable)
    # The head was re-initialized; the teacher keeps the old head.
    student.eval()
    with torch.no_grad():
        out, bb = student.model(x, return_backbone_outputs=True)
    assert not torch.allclose(out[HEAD], before)
    terms = d.loss_terms(x, out, bb, {}, None)
    torch.testing.assert_close(terms["heatmap"], F.mse_loss(out[HEAD], before))
    # The frozen backbone stays in eval mode through `.train()`.
    student.train()
    assert student.model.head_layers.training
    assert not any(m.training for m in student.model.backbone.modules())
    # Loss = gamma * heatmap (no supervised loss by default).
    batch = _batch(2)
    total = student.training_step(batch, 0)
    out = student.model(torch.squeeze(batch["instance_image"], dim=1))
    t = student.distiller.teacher.forward_heads(
        student.model(torch.squeeze(batch["instance_image"], 1), True)[1]
    )
    torch.testing.assert_close(total, F.mse_loss(out[HEAD], t[HEAD]))


def test_stage2_trainable_backbone_runs_full_teacher(
    minimal_instance_centered_instance_ckpt,
):
    ckpt = minimal_instance_centered_instance_ckpt
    student = _student(ckpt, filters=None)
    d = attach_distillation(
        student,
        {
            "stage": 2,
            "trainable": ["head", "backbone.decoders.*"],
            "supervised_weight": 0.5,
        },
    )
    assert not d.shared_features and not d.frozen_backbone
    assert d.supervised_weight == 0.5
    names = [n for n, p in student.named_parameters() if p.requires_grad]
    assert any(n.startswith("model.backbone.decoders.") for n in names)
    assert not any(n.startswith("model.backbone.encoders.") for n in names)
    batch = _batch(2)
    loss = student.training_step(batch, 0)
    loss.backward()
    with pytest.raises(ValueError, match="match no parameter"):
        attach_distillation(_student(ckpt), {"stage": 2, "trainable": ["nothing.*"]})


def test_stage2_teacher_must_match(minimal_instance_centered_instance_ckpt):
    ckpt = minimal_instance_centered_instance_ckpt
    with pytest.raises(ValueError, match="does not match the student"):
        attach_distillation(
            _student(ckpt, filters=8), {"stage": 2, "teacher": str(ckpt)}
        )


# ------------------------------------------------------------------ ViT path


@requires_transformers
def test_vit_last_tap_is_the_distilled_feature():
    """A plain ViT exposes its last tapped token map; the projector maps it."""
    from sleap_nn.architectures.model import Model

    pretrained = dict(
        source="hf",
        model_name="facebook/dinov2-with-registers-small",
        weights=False,
        mode="decoder",
        in_channels=3,
        output_stride=4,
        max_stride=32,
        vit_decoder="classic",
    )
    head = OmegaConf.create(
        {"confmaps": {"part_names": ["a", "b"], "sigma": 2.0, "output_stride": 4}}
    )
    model = Model(
        backbone_type="pretrained",
        backbone_config=OmegaConf.create(pretrained),
        head_configs=head,
        model_type="centered_instance",
    ).eval()
    _, bb = model(torch.rand(1, 3, 64, 64), return_backbone_outputs=True)
    assert bb["last_tap"].shape == (1, 384, 4, 4)
    assert bb["last_tap"] is bb["middle_output"]  # direct decoder: same map


# ---------------------------------------------------------------- end to end


def _e2e_config(config, tmp_path, ckpt):
    cfg = config.copy()
    cfg.data_config.preprocessing.crop_size = 96
    cfg.model_config.head_configs.centered_instance = OmegaConf.create(
        OmegaConf.to_container(
            _ci_config(ckpt).model_config.head_configs.centered_instance
        )
    )
    cfg.model_config.head_configs.single_instance = None
    cfg.model_config.backbone_config.unet.filters = 8
    cfg.model_config.backbone_config.unet.max_stride = 8
    cfg.model_config.backbone_config.unet.output_stride = 2
    cfg.trainer_config.max_epochs = 1
    cfg.trainer_config.min_train_steps_per_epoch = 2
    cfg.trainer_config.save_ckpt = True
    cfg.trainer_config.ckpt_dir = str(tmp_path)
    cfg.trainer_config.run_name = "kd"
    cfg.trainer_config.trainer_accelerator = "cpu"
    cfg.trainer_config.distillation = {"teacher": str(ckpt), "alpha": 0.5}
    return cfg


def test_model_trainer_trains_with_distillation(
    config, tmp_path, minimal_instance_centered_instance_ckpt
):
    """`trainer_config.distillation` alone drives a run; checkpoints stay plain."""
    from sleap_nn.training.model_trainer import ModelTrainer

    ckpt = minimal_instance_centered_instance_ckpt
    cfg = _e2e_config(config, tmp_path, ckpt)
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    lm = trainer.lightning_model
    assert isinstance(lm.distiller, Distiller)
    metrics = trainer.trainer.callback_metrics
    assert {"train/distill_heatmap", "train/distill_feat"} <= set(metrics)
    best = torch.load(
        tmp_path / "kd" / "best.ckpt", map_location="cpu", weights_only=False
    )
    assert not any(k.startswith("distiller.") for k in best["state_dict"])
    assert CHECKPOINT_KEY in best
    # The run's checkpoint loads into a plain model.
    plain = LightningModel.get_lightning_model_from_config(_ci_config(tmp_path / "kd"))
    plain.load_state_dict(best["state_dict"], strict=True)


def test_model_trainer_warp_dual_resolution_teacher(
    config, tmp_path, minimal_instance_centered_instance_ckpt, monkeypatch
):
    """A warp run with `teacher_crop_size` feeds the teacher its own crop in every
    training step; validation batches carry none."""
    from sleap_nn.training.model_trainer import ModelTrainer

    ckpt = minimal_instance_centered_instance_ckpt
    cfg = _e2e_config(config, tmp_path, ckpt)
    pre = cfg.data_config.preprocessing
    pre.crop_mode = "bbox"
    pre.crop_pipeline = "warp"
    pre.crop_size = [64, 48]
    pre.gpu_confmaps = True
    head = cfg.model_config.head_configs.centered_instance.confmaps
    head.anchor_part = None
    head.centroid_method = "bbox_center"
    cfg.trainer_config.distillation = {
        "teacher": str(ckpt),
        "alpha": 0.5,
        "teacher_crop_size": [96, 72],
    }
    seen = []
    teacher_input = Distiller.teacher_input

    def spy(self, x, batch):
        seen.append((tuple(x.shape[-2:]), tuple(batch[TEACHER_IMAGE_KEY].shape)))
        return teacher_input(self, x, batch)

    monkeypatch.setattr(Distiller, "teacher_input", spy)
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    assert trainer.lightning_model.distiller.teacher_crop_size == (96, 72)
    assert seen and all(
        xs == (64, 48) and ts[0] > 0 and ts[1:] == (1, 1, 96, 72) for xs, ts in seen
    )
    assert "train/distill_heatmap" in trainer.trainer.callback_metrics
    assert TEACHER_IMAGE_KEY not in trainer.trainer.val_dataloaders.dataset[0]


# ---------------------------------------------------------------------- DDP

_WORKER = Path(__file__).parent / "ddp_distillation_worker.py"


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="gloo DDP smoke test runs on Linux"
)
def test_distillation_under_ddp(tmp_path, minimal_instance_centered_instance_ckpt):
    """2 gloo ranks: each holds its own frozen teacher outside DDP; the student and
    projector stay in sync; the teacher is unchanged."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_", "SLURM"))
    }
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", WANDB_MODE="disabled")
    proc = subprocess.run(
        [
            sys.executable,
            _WORKER.as_posix(),
            tmp_path.as_posix(),
            Path(minimal_instance_centered_instance_ckpt).as_posix(),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    r0, r1 = (json.loads((tmp_path / f"rank{k}.json").read_text()) for k in (0, 1))
    assert (r0["rank"], r1["rank"], r0["world_size"]) == (0, 1, 2)
    for r in (r0, r1):
        assert r["strategy_wraps_teacher"] is False
        assert r["teacher_in_ddp_params"] is False
        assert r["teacher_unchanged"] is True
        assert r["teacher_training"] is False
        assert r["steps"] == 4
    # DDP all-reduced the projector's (and the student's) gradients.
    assert r0["projector_sum"] == pytest.approx(r1["projector_sum"], rel=0, abs=0)
    assert r0["student_sum"] == pytest.approx(r1["student_sum"], rel=0, abs=0)
    assert r0["projector_changed"] and r1["projector_changed"]


def test_model_trainer_guards(minimal_instance_centered_instance_ckpt):
    """Stage 2 refuses a freeze schedule, and a resume without its teacher."""
    from types import SimpleNamespace

    from sleap_nn.training.model_trainer import ModelTrainer

    ckpt = minimal_instance_centered_instance_ckpt

    def setup(**trainer_config):
        cfg = OmegaConf.create({"trainer_config": trainer_config})
        host = SimpleNamespace(config=cfg, lightning_model=_student(ckpt))
        ModelTrainer._setup_distillation(host)
        return host.lightning_model

    assert setup(distillation=None).distiller is None
    with pytest.raises(ValueError, match="remove trainer_config.freeze_schedule"):
        setup(distillation={"stage": 2}, freeze_schedule={"targets": ["encoder"]})
    with pytest.raises(ValueError, match="Resuming distillation stage 2"):
        setup(distillation={"stage": 2}, resume_ckpt_path="last.ckpt")
    assert setup(distillation={"teacher": str(ckpt)}).distiller.stage == 1
