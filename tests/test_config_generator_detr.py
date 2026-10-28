"""Config generator, recommender and TUI support for ``bbox_detr`` detectors.

Covers the ``bbox_detr`` / ``topdown_bbox_detr`` pipelines (the golden config lives
next to the other generator goldens in ``tests/assets/generated_configs/``), the
presets and their parameter / memory estimates, the token-width resolution rule,
the GPU-aware recommendation and the TUI model picker / configure step.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio
import yaml
from omegaconf import OmegaConf

from sleap_nn.config_generator.architecture_estimates import (
    DETR_PRESETS,
    estimate_detr_head_params,
    estimate_detr_params,
    estimate_vit_params,
    recommend_detr_resolution,
)
from sleap_nn.config_generator.generator import DETR_TRAINER_DEFAULTS, ConfigGenerator
from sleap_nn.config_generator.memory import estimate_detr_memory

FIXTURE_SLP = "tests/assets/datasets/minimal_instance.pkg.slp"
GOLDEN_DIR = Path("tests/assets/generated_configs")
PLACEHOLDER_PATH = "/path/to/labels.slp"


def _normalized(cfg) -> dict:
    cfg.data_config.train_labels_path = [PLACEHOLDER_PATH]
    return yaml.safe_load(OmegaConf.to_yaml(cfg))


def _detr_generator(pipeline="bbox_detr"):
    return ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline(pipeline)


# --------------------------------------------------------------------------- #
# The bbox_detr pipeline
# --------------------------------------------------------------------------- #
def test_bbox_detr_matches_golden():
    """The generated bbox_detr config is pinned.

    To intentionally update the golden, write ``ConfigGenerator.from_slp(
    FIXTURE_SLP).auto().pipeline("bbox_detr").build()`` (normalized like
    ``_normalized``) to ``tests/assets/generated_configs/bbox_detr.yaml``.
    """
    golden = yaml.safe_load((GOLDEN_DIR / "bbox_detr.yaml").read_text())
    assert _normalized(_detr_generator().build()) == golden


def test_bbox_detr_defaults():
    """DINOv3 ViT-S, one class, 100 queries, letterbox, RF-DETR's trainer."""
    from sleap_nn.config.training_job_config import verify_training_cfg
    from sleap_nn.config.utils import get_model_type_from_cfg

    gen = _detr_generator()
    assert gen.is_bbox and gen.is_detr and not gen.is_topdown
    cfg = verify_training_cfg(gen.build())
    assert get_model_type_from_cfg(cfg) == "bbox_detr"

    data = cfg.data_config
    pre = data.preprocessing
    assert pre.resize_mode == "letterbox" and pre.scale == 1.0
    assert pre.ensure_rgb  # the fixture is grayscale; DINO reads RGB
    assert pre.max_height % 32 == 0 and pre.max_width % 32 == 0
    res = gen.detr_resolution()
    assert (pre.max_height, pre.max_width) == (res.max_height, res.max_width)
    # Boxes from the poses, padded by 5% of the median animal size.
    assert data.bbox_source == "auto"
    assert data.bbox_padding == round(0.05 * gen.stats.median_bbox_size)
    aug = data.augmentation_config
    assert aug.geometric is None  # no rotation: it would inflate the boxes
    short = min(pre.max_height, pre.max_width)
    assert list(aug.detr.shortest_sides) == [short]
    assert list(aug.detr.crop_range) == [round(short * 2 / 3), short]
    assert aug.detr.multi_scale and aug.detr.flip_p == 0.5

    backbone = cfg.model_config.backbone_config
    assert backbone.unet is None
    vit = backbone.pretrained
    assert vit.model_name == "timm/vit_small_patch16_dinov3.lvd1689m" and vit.weights
    assert vit.mode == "tokens" and list(vit.layers) == [3, 6, 9, 12]
    assert (vit.output_stride, vit.max_stride) == (16, 32)
    detr = cfg.model_config.head_configs.bbox_detr.detr
    assert detr.categories is None  # one class, filled from the boxes
    assert detr.num_queries == detr.num_select == 100
    assert detr.dec_layers == 3 and detr.group_detr == 13
    assert cfg.model_config.init_from is None

    trainer = cfg.trainer_config
    assert trainer.max_epochs == 200 and trainer.min_train_steps_per_epoch == 1
    assert trainer.optimizer_name == "AdamW" and trainer.optimizer.lr == 1e-4
    assert trainer.optimizer.encoder_lr_scale == 1.5
    assert trainer.optimizer.layer_decay == 0.8
    assert trainer.optimizer.weight_decay == 1e-4
    sched = trainer.lr_scheduler
    assert sched is None or all(
        sched[k] is None
        for k in ("step_lr", "reduce_lr_on_plateau", "cosine_annealing_warmup")
    )
    assert not trainer.early_stopping.stop_training_on_plateau
    assert trainer.ema.enabled and trainer.ema.decay == 0.993
    assert trainer.gradient_clip_val == 0.1 and trainer.precision == "bf16-mixed"
    assert trainer.eval.enabled


def test_other_pipelines_keep_their_trainer():
    """The DETR trainer, augmentation and head never leak into other pipelines."""
    for pipeline in ("bbox_centernet", "bottomup", "centroid"):
        cfg = ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline(pipeline).build()
        assert "ema" not in cfg.trainer_config
        assert "precision" not in cfg.trainer_config
        assert "bbox_detr" not in cfg.model_config.head_configs
        assert "resize_mode" not in cfg.data_config.preprocessing


@pytest.mark.parametrize("preset", sorted(DETR_PRESETS))
def test_detr_presets(preset):
    """Each preset builds a valid config with its backbone and decoder depth."""
    from sleap_nn.config.training_job_config import verify_training_cfg

    p = DETR_PRESETS[preset]
    cfg = verify_training_cfg(_detr_generator().detr(preset=preset).build())
    backbone = cfg.model_config.backbone_config
    detr = cfg.model_config.head_configs.bbox_detr.detr
    assert detr.dec_layers == p.dec_layers
    if p.backbone == "pretrained":
        assert backbone.pretrained.model_name == p.model_name
        assert backbone.dinov2_windowed is None
    else:
        win = backbone.dinov2_windowed
        assert backbone.pretrained is None
        # Fresh decoder on plain DINOv2 weights (no import given).
        assert win.weights == "facebook/dinov2-small"
        assert win.pos_embed_grid == p.input_size // 16
        assert cfg.data_config.preprocessing.resize_mode == "letterbox"


def test_rfdetr_import_finetune():
    """An RF-DETR preset with an imported checkpoint squashes and loads it."""
    from sleap_nn.config.training_job_config import verify_training_cfg

    gen = _detr_generator().detr(preset="rfdetr_m", init_from="models/rfdetr-medium")
    cfg = verify_training_cfg(gen.build())
    pre = cfg.data_config.preprocessing
    assert pre.resize_mode == "squash" and pre.max_height == pre.max_width == 576
    assert cfg.model_config.init_from.path == "models/rfdetr-medium"
    # The whole checkpoint replaces the plain DINOv2 init.
    assert cfg.model_config.backbone_config.dinov2_windowed.weights is None
    detr = cfg.model_config.head_configs.bbox_detr.detr
    assert detr.dec_layers == 4
    # 300 queries, so the import's 300 x 13 query embeddings load.
    assert detr.num_queries == detr.num_select == 300
    assert (
        gen.detr(num_queries=50)
        .build()
        .model_config.head_configs.bbox_detr.detr.num_queries
        == 50
    )


def test_detr_setter_validates_and_sets_side():
    gen = _detr_generator()
    with pytest.raises(ValueError, match="Unknown bbox_detr preset"):
        gen.detr(preset="rfdetr_xl")
    cfg = gen.detr(num_queries=30, max_side=256).build()
    pre = cfg.data_config.preprocessing
    assert pre.max_height == pre.max_width == 256  # square fixture frames
    assert cfg.model_config.head_configs.bbox_detr.detr.num_queries == 30


def test_bbox_detr_round_trips_through_model_trainer():
    pytest.importorskip("torch")
    from sleap_nn.training.model_trainer import ModelTrainer

    cfg = _detr_generator().build()
    cfg.data_config.val_labels_path = []
    ModelTrainer(cfg)


# --------------------------------------------------------------------------- #
# Top-down from DETR boxes
# --------------------------------------------------------------------------- #
def test_topdown_bbox_detr_pair():
    """Stage 1 is the standalone DETR config; stage 2 the box-crop CI model.

    The CI keeps the pose trainer and augmentation and crops the same padded
    boxes the detector learns.
    """
    from sleap_nn.config.training_job_config import verify_training_cfg
    from sleap_nn.config.utils import get_model_type_from_cfg

    gen = _detr_generator("topdown_bbox_detr")
    assert (
        gen.is_topdown and gen.is_topdown_bbox and gen.detector_pipeline == "bbox_detr"
    )
    golden = yaml.safe_load((GOLDEN_DIR / "bbox_detr.yaml").read_text())
    assert _normalized(gen.build_detector()) == golden
    assert _normalized(gen.build()) == golden

    det = verify_training_cfg(gen.build_detector())
    ci = verify_training_cfg(gen.build_centered_instance())
    assert get_model_type_from_cfg(ci) == "centered_instance"
    assert ci.data_config.preprocessing.crop_mode == "bbox"
    for key in ("bbox_source", "bbox_padding"):
        assert ci.data_config[key] == det.data_config[key]
    assert det.data_config.bbox_padding > 0
    # Only the padding differs from the CenterNet pair's CI golden.
    ci_golden = yaml.safe_load(
        (GOLDEN_DIR / "topdown_bbox_centered_instance.yaml").read_text()
    )
    ci_golden["data_config"]["bbox_padding"] = det.data_config.bbox_padding
    assert _normalized(gen.build_centered_instance()) == ci_golden
    # The DETR trainer is the detector's alone, and building it leaves the
    # generator's (CI) state as it was.
    assert ci.trainer_config.optimizer_name == "Adam"
    assert (
        "ema"
        not in OmegaConf.to_container(gen.build_centered_instance())["trainer_config"]
    )
    state = {key: getattr(gen, key) for key in DETR_TRAINER_DEFAULTS}
    gen.build_detector()
    assert {key: getattr(gen, key) for key in DETR_TRAINER_DEFAULTS} == state
    assert gen._optimizer_name == "Adam" and gen._early_stopping


def test_topdown_bbox_detr_save_names(tmp_path):
    gen = _detr_generator("topdown_bbox_detr")
    gen.save(str(tmp_path / "td.yaml"))
    assert (tmp_path / "td_bbox_detr.yaml").exists()
    assert (tmp_path / "td_centered_instance.yaml").exists()
    assert "# === DETECTOR CONFIG ===" in gen.to_yaml()
    assert "bbox_detr" in gen.summary() and "Input:" in gen.summary()


def test_cli_config_pipeline_detr(tmp_path):
    from click.testing import CliRunner

    from sleap_nn.cli import config as config_cmd
    from sleap_nn.config.utils import get_model_type_from_cfg

    out = tmp_path / "out.yaml"
    result = CliRunner().invoke(
        config_cmd,
        [FIXTURE_SLP, "--auto", "--pipeline", "topdown_bbox_detr", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    det = OmegaConf.load(tmp_path / "out_bbox_detr.yaml")
    assert get_model_type_from_cfg(det) == "bbox_detr"
    result = CliRunner().invoke(
        config_cmd, [FIXTURE_SLP, "--auto", "--gpu_memory_gb", "lots", "-o", str(out)]
    )
    assert result.exit_code != 0 and "--gpu_memory_gb" in result.output
    # A GPU size without bbox_detr-friendly data keeps the plain recommendation.
    result = CliRunner().invoke(
        config_cmd, [FIXTURE_SLP, "--auto", "--gpu_memory_gb", "1", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out_centroid.yaml").exists()


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
def _touching_stats(**changes):
    from sleap_nn.config_generator.analyzer import analyze_slp

    return dataclasses.replace(analyze_slp(FIXTURE_SLP), **changes)


def test_recommender_suggests_detr_for_touching_animals():
    from sleap_nn.config_generator.recommender import recommend_pipeline

    stats = _touching_stats(touch_frequency=0.4)
    # Unknown GPU: centroid, with bbox_detr suggested.
    rec = recommend_pipeline(stats)
    assert rec.recommended == "centroid"
    assert "topdown_bbox_detr" in rec.alternatives
    assert any("40%" in w and "topdown_bbox_detr" in w for w in rec.warnings)
    # Too small a GPU: the same.
    assert recommend_pipeline(stats, gpu_memory_gb=1.0).recommended == "centroid"


def test_recommender_picks_detr_when_it_fits():
    pytest.importorskip("transformers")
    from sleap_nn.config_generator.recommender import (
        recommend_config,
        recommend_pipeline,
    )

    rec = recommend_pipeline(_touching_stats(touch_frequency=0.4), gpu_memory_gb=48)
    assert rec.recommended == "topdown_bbox_detr"
    assert rec.alternatives[0] == "centroid" and "40%" in rec.reason
    assert rec.requires_second_model and rec.second_model_type == "centered_instance"
    # Fits without touching animals too (multi-animal top-down data).
    calm = recommend_pipeline(_touching_stats(touch_frequency=0.0), gpu_memory_gb=48)
    assert calm.recommended == "topdown_bbox_detr" and "touch" not in calm.reason
    # auto() then applies the detector defaults of the pair.
    gen = ConfigGenerator.from_slp(FIXTURE_SLP).auto(gpu_memory_gb=48)
    assert gen._pipeline == "topdown_bbox_detr" and gen._output_stride == 16
    assert recommend_config(gen.stats).pipeline.recommended == "centroid"


def test_analyzer_box_stats():
    """Box-size percentiles (sqrt(w * h)) and the touching-frame rate."""
    from sleap_nn.config_generator.analyzer import analyze_slp

    stats = analyze_slp(FIXTURE_SLP)
    sizes = []
    for lf in sio.load_slp(FIXTURE_SLP):
        for inst in lf.user_instances:
            w, h = np.ptp(inst.numpy(), axis=0)
            sizes.append(np.sqrt(w * h))
    assert stats.median_bbox_size == pytest.approx(np.median(sizes))
    assert stats.small_bbox_size == pytest.approx(np.percentile(sizes, 5))
    assert stats.touch_frequency == 0.0  # the two fixture animals are apart
    assert "Touching frames" in str(stats)


# --------------------------------------------------------------------------- #
# Estimates and the resolution rule
# --------------------------------------------------------------------------- #
def test_resolution_rule_mouse_and_large_arena():
    """Small animals >= 3 tokens wide; a large arena is capped and flagged."""
    # sleap-mice_hc: 1280 x 1024 frames, padded boxes p5 99 px, median 193 px.
    res = recommend_detr_resolution(1024, 1280, 99, 193)
    assert (res.max_width, res.max_height) == (640, 512)
    assert res.small_tokens >= 3 and res.median_tokens >= 4 and not res.needs_tiling
    # mosaic-harvard: 3934 x 3490 frames, p5 87 px: needs ~2170 px, capped at 1280.
    big = recommend_detr_resolution(3490, 3934, 87, 169)
    assert big.needs_tiling and big.max_side == 1280 and big.small_tokens < 3
    assert "capped at 1280" in big.reason
    # Animals small even at native size: native, never upsampled, flagged.
    tiny = recommend_detr_resolution(384, 384, 30, 40)
    assert tiny.max_side == 384 and tiny.scale == 1.0 and "misses" in tiny.reason
    # Large animals: never below the floor.
    assert recommend_detr_resolution(1024, 1280, 600, 700).max_side == 384


def test_vit_param_estimate_matches_windowed_dinov2():
    """The ViT formula matches a real (meta-device) windowed DINOv2-S."""
    import torch

    from sleap_nn.architectures.detr import build_detr_backbone
    from sleap_nn.config.model_config import Dinov2WindowedConfig

    with torch.device("meta"):
        backbone = build_detr_backbone(
            "dinov2_windowed", OmegaConf.structured(Dinov2WindowedConfig())
        )
    actual = sum(p.numel() for p in backbone.parameters())
    # A learned 36 x 36 position embedding + CLS, no registers.
    estimate = estimate_vit_params(384, 12, prefix_tokens=1, pos_embed_tokens=1297)
    assert abs(estimate - actual) / actual < 0.002
    assert estimate_detr_params("rfdetr_m")["backbone"] == pytest.approx(
        actual, rel=0.002
    )


def test_detr_param_estimates():
    """RF-DETR's published sizes (91 COCO classes) and the DINOv3 presets."""
    published = {"rfdetr_n": 30.5e6, "rfdetr_s": 32.1e6, "rfdetr_m": 33.7e6}
    for preset, total in published.items():
        est = estimate_detr_params(preset, num_classes=91)["total"]
        assert est == pytest.approx(total, rel=0.02), preset
    small = estimate_detr_params("dinov3_s", num_queries=100)
    base = estimate_detr_params("dinov3_b", num_queries=100)
    assert small["backbone"] == pytest.approx(21.59e6, rel=0.002)
    assert base["backbone"] == pytest.approx(85.64e6, rel=0.002)
    assert small["total"] < base["total"]
    # Fewer queries: fewer learned query embeddings, nothing else.
    diff = estimate_detr_head_params(1536, num_queries=300) - estimate_detr_head_params(
        1536, num_queries=100
    )
    assert diff == 200 * 13 * (256 + 4)


def test_detr_memory_estimate():
    """Calibrated on M2 (RF-DETR-M, batch 4, 576 px: 4.7 GB measured)."""
    from sleap_nn.config_generator.analyzer import analyze_slp

    stats = analyze_slp(FIXTURE_SLP)
    m2 = estimate_detr_memory(stats, "rfdetr_m", 4, 576, 576, num_queries=300)
    assert m2.total_gpu_gb == pytest.approx(4.7, rel=0.1)
    mouse = estimate_detr_memory(stats, "dinov3_s", 8, 512, 640)
    vit_b = estimate_detr_memory(stats, "dinov3_b", 8, 512, 640)
    assert mouse.total_gpu_gb < vit_b.total_gpu_gb
    assert (
        mouse.params_count == estimate_detr_params("dinov3_s", num_queries=100)["total"]
    )
    assert mouse.gpu_status == "green" and "GPU Memory" in str(mouse)


# --------------------------------------------------------------------------- #
# TUI
# --------------------------------------------------------------------------- #
def test_tui_detr_selection_and_restore(tmp_path):
    """Picking bbox_detr sets its trainer; a pose pick restores the prior one.

    The configure step shows the DETR section, whose preset select feeds the
    generated config; a top-down DETR pair saves a detector + CI pair.
    """
    pytest.importorskip("textual")
    import asyncio

    from textual.widgets import Select, Static

    from sleap_nn.config.utils import get_model_type_from_cfg
    from sleap_nn.config_generator.tui.app import ConfigGeneratorApp

    async def run():
        app = ConfigGeneratorApp(FIXTURE_SLP)
        async with app.run_test(size=(160, 60)) as pilot:
            await app._show_step(2)
            await pilot.pause()
            screen = app.query_one("#model-screen")
            state = app._state
            before = (
                state._max_epochs,
                state._optimizer,
                state._early_stopping,
                state._output_stride,
                state._scheduler.type,
            )
            screen._select_type("bbox_detr")
            assert state.is_detr and state._max_epochs == 200
            cfg = OmegaConf.create(state.build_config())
            assert get_model_type_from_cfg(cfg) == "bbox_detr"
            assert cfg.trainer_config.optimizer_name == "AdamW"
            assert cfg.trainer_config.ema.enabled
            assert cfg.data_config.bbox_padding > 0
            assert cfg.model_config.head_configs.bbox_detr.detr.num_queries == 100
            assert state.model_params_estimate > 20e6
            assert state.memory_estimate().total_gpu_gb > 1

            # Another detector undoes the DETR trainer; a pose model the rest.
            screen._select_type("bbox_centernet")
            assert state._max_epochs == before[0] and state._bbox_padding == 0.0
            screen._select_type("bottomup")
            after = (
                state._max_epochs,
                state._optimizer,
                state._early_stopping,
                state._output_stride,
                state._scheduler.type,
            )
            assert after == before

            # Configure step of the standalone detector.
            screen._select_type("bbox_detr")
            await app._show_step(3)
            await pilot.pause()
            info = app.query_one("#detr-info-display", Static)
            assert "DINOv3 ViT-S/16" in str(info.render())
            app.query_one("#detr-preset-select", Select).value = "dinov3_b"
            await pilot.pause()
            cfg = OmegaConf.create(state.build_config())
            vit = cfg.model_config.backbone_config.pretrained
            assert vit.model_name == "timm/vit_base_patch16_dinov3.lvd1689m"

            # The top-down pair.
            await app._show_step(2)
            await pilot.pause()
            app.query_one("#model-screen")._select_type("topdown_bbox_detr")
            assert state.is_topdown_bbox and state.first_stage_name == "bbox_detr"
            det = OmegaConf.create(state.build_centroid_config())
            ci = OmegaConf.create(state.build_centered_instance_config())
            assert get_model_type_from_cfg(det) == "bbox_detr"
            assert get_model_type_from_cfg(ci) == "centered_instance"
            assert ci.data_config.bbox_padding == det.data_config.bbox_padding
            assert "ema" not in ci.trainer_config
            det_path, ci_path = state.save_dual(str(tmp_path / "labels"))
            assert det_path == str(tmp_path / "labels_bbox_detr.yaml")
            await app._show_step(3)
            await pilot.pause()
            assert app.query_one("#topdown-tabs")
            assert app.query_one("#detr-info-display", Static)
            await app._show_step(4)
            await pilot.pause()

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Sample configs
# --------------------------------------------------------------------------- #
SAMPLE_DIR = Path(__file__).resolve().parents[1] / "docs" / "sample_configs"


@pytest.mark.parametrize(
    "name",
    sorted(p.name for p in SAMPLE_DIR.glob("config_bbox_detr_*.yaml")),
)
def test_bbox_detr_sample_configs_verify(name):
    """Every bbox_detr sample config is a valid training config."""
    from sleap_nn.config.training_job_config import verify_training_cfg
    from sleap_nn.config.utils import get_model_type_from_cfg

    cfg = verify_training_cfg(OmegaConf.load(SAMPLE_DIR / name))
    assert get_model_type_from_cfg(cfg) == "bbox_detr"
    # The geometric section is off in every sample (keypoint boxes would inflate).
    assert cfg.data_config.augmentation_config.geometric is None


def test_mouse_sample_matches_the_rules():
    """The mouse sample's documented numbers follow the generator's rules."""
    cfg = OmegaConf.load(SAMPLE_DIR / "config_bbox_detr_dinov3_s_mouse.yaml")
    pre = cfg.data_config.preprocessing
    # sleap-mice_hc: 1280 x 1024, sqrt(w * h) p5 79 px / median 173 px, + 2 x 9 px.
    res = recommend_detr_resolution(1024, 1280, 79 + 18, 173 + 18)
    assert (pre.max_width, pre.max_height) == (res.max_width, res.max_height)
    assert cfg.data_config.bbox_padding == round(0.05 * 173)
    short = min(pre.max_height, pre.max_width)
    detr_aug = cfg.data_config.augmentation_config.detr
    assert list(detr_aug.shortest_sides) == [short]
    assert list(detr_aug.crop_range) == [round(short * 2 / 3), short]
    detr = cfg.model_config.head_configs.bbox_detr.detr
    assert detr.num_queries == 100 and detr.categories is None
    assert detr.dec_layers == DETR_PRESETS["dinov3_s"].dec_layers
