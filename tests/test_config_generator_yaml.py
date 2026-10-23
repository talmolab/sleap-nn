"""Snapshot + canonical-loader tests for ``ConfigGenerator`` YAML output.

These tests pin down the YAML schema the TUI/CLI auto-config emits, so any
change to the emit path is intentional. Goldens live in
``tests/assets/generated_configs/``.

The web app config picker (``docs/configuration/config-picker/app.html``)
emits the same schema; if these tests pass and the web app's
``generateConfigYaml`` continues to mirror the canonical
``TrainingJobConfig`` schema, both surfaces produce equivalent configs.
"""

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from sleap_nn.config_generator.generator import ConfigGenerator

PIPELINES = [
    "single_instance",
    "centroid",
    "centered_instance",
    "bottomup",
    "multi_class_bottomup",
    "multi_class_topdown",
]

FIXTURE_SLP = "tests/assets/datasets/minimal_instance.pkg.slp"
GOLDEN_DIR = Path("tests/assets/generated_configs")
PLACEHOLDER_PATH = "/path/to/labels.slp"


# ``centroid_only`` is a single-stage STANDALONE centroid pipeline whose head
# key remains the canonical ``centroid`` (so ``get_model_type_from_cfg`` returns
# 'centroid'). It is golden-tested but is NOT part of ``PIPELINES`` for the
# "head key == pipeline name" oneof test (its head key differs from its name).
# The bounding-box detectors (boxes derived from the poses). Their head key is the
# pipeline name, emitted next to the six null pose head keys.
BBOX_PIPELINES = ["bbox_centernet", "bbox_ssd"]
GOLDEN_PIPELINES = PIPELINES + ["centroid_only"] + BBOX_PIPELINES


def _build_config(pipeline: str) -> dict:
    """Build a config for ``pipeline`` and normalize the labels path."""
    gen = ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline(pipeline)
    cfg = gen.build()
    cfg.data_config.train_labels_path = [PLACEHOLDER_PATH]
    return yaml.safe_load(OmegaConf.to_yaml(cfg))


@pytest.mark.parametrize("pipeline", GOLDEN_PIPELINES)
def test_yaml_matches_golden(pipeline):
    """Generated YAML for each pipeline matches the checked-in golden.

    To intentionally update goldens, run::

        uv run python -c "
        from sleap_nn.config_generator.generator import ConfigGenerator
        from omegaconf import OmegaConf
        from pathlib import Path
        for p in ['single_instance', 'centroid', 'centroid_only',
                  'centered_instance', 'bottomup', 'multi_class_bottomup',
                  'multi_class_topdown', 'bbox_centernet', 'bbox_ssd']:
            gen = ConfigGenerator.from_slp(
                'tests/assets/datasets/minimal_instance.pkg.slp'
            ).auto().pipeline(p)
            cfg = gen.build()
            cfg.data_config.train_labels_path = ['/path/to/labels.slp']
            Path(f'tests/assets/generated_configs/{p}.yaml').write_text(
                OmegaConf.to_yaml(cfg)
            )
        "
    """
    actual = _build_config(pipeline)
    golden_path = GOLDEN_DIR / f"{pipeline}.yaml"
    expected = yaml.safe_load(golden_path.read_text())
    assert actual == expected


@pytest.mark.parametrize("pipeline", GOLDEN_PIPELINES)
def test_exactly_one_backbone_active(pipeline):
    """Generated YAML has exactly one non-null backbone (oneof contract)."""
    cfg = _build_config(pipeline)
    backbones = cfg["model_config"]["backbone_config"]
    active = [k for k, v in backbones.items() if v is not None]
    assert len(active) == 1, f"{pipeline}: expected 1 active backbone, got {active}"


@pytest.mark.parametrize("pipeline", PIPELINES + BBOX_PIPELINES)
def test_exactly_one_head_active(pipeline):
    """Generated YAML has exactly one non-null head matching the pipeline."""
    cfg = _build_config(pipeline)
    heads = cfg["model_config"]["head_configs"]
    active = [k for k, v in heads.items() if v is not None]
    assert active == [pipeline]


@pytest.mark.parametrize("pipeline", GOLDEN_PIPELINES)
def test_round_trips_through_model_trainer(pipeline):
    """Generated config is accepted by the canonical ModelTrainer.

    This is the strongest contract: any field name or shape error would
    surface here. Tests run lazily because ModelTrainer imports torch.
    """
    pytest.importorskip("torch")
    from sleap_nn.training.model_trainer import ModelTrainer

    gen = ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline(pipeline)
    cfg = gen.build()
    cfg.data_config.val_labels_path = []
    # Should not raise
    ModelTrainer(cfg)


def test_unet_emits_kernel_size():
    """UNet backbone must include kernel_size: 3 (canonical schema)."""
    cfg = _build_config("single_instance")
    assert cfg["model_config"]["backbone_config"]["unet"]["kernel_size"] == 3


def test_bottomup_emits_pafs_with_edges_and_part_names():
    """Bottomup head needs part_names on confmaps and edges on pafs."""
    cfg = _build_config("bottomup")
    head = cfg["model_config"]["head_configs"]["bottomup"]
    assert isinstance(head["confmaps"]["part_names"], list)
    assert len(head["confmaps"]["part_names"]) > 0
    assert "edges" in head["pafs"]
    assert isinstance(head["pafs"]["edges"], list)


def test_multi_class_bottomup_uses_class_maps_not_class_vectors():
    """multi_class_bottomup head uses class_maps (canonical schema)."""
    cfg = _build_config("multi_class_bottomup")
    mc = cfg["model_config"]["head_configs"]["multi_class_bottomup"]
    assert "class_maps" in mc
    assert "class_vectors" not in mc
    cm = mc["class_maps"]
    for required in ("classes", "sigma", "output_stride", "loss_weight"):
        assert required in cm, f"missing {required} in class_maps"


def test_multi_class_topdown_class_vectors_complete():
    """multi_class_topdown class_vectors has all required canonical fields."""
    cfg = _build_config("multi_class_topdown")
    cv = cfg["model_config"]["head_configs"]["multi_class_topdown"]["class_vectors"]
    for required in (
        "classes",
        "num_fc_layers",
        "num_fc_units",
        "global_pool",
        "output_stride",
        "loss_weight",
    ):
        assert required in cv, f"missing {required} in class_vectors"


def test_centered_instance_emits_part_names_and_loss_weight():
    """centered_instance head needs part_names + loss_weight."""
    cfg = _build_config("centered_instance")
    ci = cfg["model_config"]["head_configs"]["centered_instance"]["confmaps"]
    assert isinstance(ci["part_names"], list) and len(ci["part_names"]) > 0
    assert "loss_weight" in ci


@pytest.mark.parametrize("off", ["rotation", "scale"])
def test_disabled_geometric_aug_stays_off(off):
    """A disabled rotation/scale is switched off, not left to the schema default
    (+-15 deg / 0.9-1.1 at p=1.0) when another geometric augmentation is on."""
    from sleap_nn.config.training_job_config import verify_training_cfg

    gen = ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline("single_instance")
    if off == "rotation":
        gen.rotation(0.0, 0.0)
    else:
        gen.scale_augmentation(1.0, 1.0)
    geometric = verify_training_cfg(
        gen.build()
    ).data_config.augmentation_config.geometric
    assert getattr(geometric, f"{off}_p") == 0.0
    on = "scale" if off == "rotation" else "rotation"
    assert getattr(geometric, f"{on}_p") == 1.0


def test_lr_scheduler_has_all_four_keys():
    """lr_scheduler block has step_lr, reduce_lr_on_plateau, etc., one active."""
    cfg = _build_config("single_instance")
    lrs = cfg["trainer_config"]["lr_scheduler"]
    expected_keys = {
        "step_lr",
        "reduce_lr_on_plateau",
        "cosine_annealing_warmup",
        "linear_warmup_linear_decay",
    }
    assert set(lrs.keys()) == expected_keys
    active = [k for k, v in lrs.items() if v is not None]
    assert len(active) == 1


def test_augmentation_uses_canonical_field_names():
    """Augmentation config uses brightness_min/max/p, NOT brightness_limit."""
    cfg = _build_config("single_instance")
    aug = cfg["data_config"]["augmentation_config"]
    if aug is None:
        pytest.skip("augmentations disabled — nothing to check")
    geom = aug.get("geometric")
    if geom is not None:
        # Canonical names
        for k in geom:
            assert k in {
                "rotation_min",
                "rotation_max",
                "rotation_p",
                "scale_min",
                "scale_max",
                "scale_p",
                "translate_width",
                "translate_height",
                "translate_p",
                "affine_p",
            }
    intensity = aug.get("intensity")
    if intensity is not None:
        for k in intensity:
            assert k in {
                "brightness_min",
                "brightness_max",
                "brightness_p",
                "contrast_min",
                "contrast_max",
                "contrast_p",
            }
            assert "limit" not in k


# ──────────────────────────────────────────────────────────────────────────
# Standalone "centroid_only" pipeline (single-config centroid model)
# ──────────────────────────────────────────────────────────────────────────


def test_centroid_only_emits_single_centroid_head():
    """centroid_only emits ONE config with only the canonical ``centroid`` head.

    Head key is ``centroid`` (not ``centroid_only``) so ``get_model_type_from_cfg``
    returns 'centroid' and the inference flow auto-detects a centroid model.
    """
    cfg = _build_config("centroid_only")
    heads = cfg["model_config"]["head_configs"]
    active = [k for k, v in heads.items() if v is not None]
    assert active == ["centroid"]
    confmaps = heads["centroid"]["confmaps"]
    assert confmaps["sigma"] == 2.5
    assert confmaps["output_stride"] == 2


def test_centroid_only_full_res_no_crop():
    """centroid_only is full resolution (scale 1.0) with no crop_size."""
    cfg = _build_config("centroid_only")
    preproc = cfg["data_config"]["preprocessing"]
    assert preproc["scale"] == 1.0
    assert preproc["crop_size"] is None
    # Non-cropped preprocessing emits max_height/max_width (not a CI crop block).
    assert "max_height" in preproc
    assert "max_width" in preproc


def test_centroid_only_resolves_to_centroid_model_type():
    """The generated centroid_only config resolves to model_type 'centroid'."""
    from omegaconf import OmegaConf

    from sleap_nn.config.utils import get_model_type_from_cfg

    gen = ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline("centroid_only")
    cfg = gen.build()
    assert get_model_type_from_cfg(cfg) == "centroid"


def test_centroid_only_is_not_topdown_but_centroid_is():
    """is_topdown is False for centroid_only (single config) and True for centroid."""
    standalone = ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline("centroid_only")
    assert standalone.is_topdown is False

    paired = ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline("centroid")
    assert paired.is_topdown is True


def test_embedding_pipeline_rejected_with_pointer():
    """The pose-oriented generator rejects 'embedding' with a clear pointer (P2 #8).

    Rather than silently emitting a pose config with no embedding head, selecting the
    re-ID model type raises and points at the dedicated sample config.
    """
    gen = ConfigGenerator.from_slp(FIXTURE_SLP).auto()
    with pytest.raises(ValueError, match="embedding"):
        gen.pipeline("embedding")


def test_centroid_only_save_emits_single_file(tmp_path):
    """save() writes ONE file for centroid_only vs TWO for the paired centroid."""
    out = tmp_path / "cfg.yaml"

    gen = ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline("centroid_only")
    gen.save(str(out))
    assert out.exists()
    # No paired centroid/centered_instance split files.
    assert not (tmp_path / "cfg_centroid.yaml").exists()
    assert not (tmp_path / "cfg_centered_instance.yaml").exists()

    # By contrast, the paired top-down ``centroid`` pipeline dual-emits.
    out2 = tmp_path / "td.yaml"
    paired = ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline("centroid")
    paired.save(str(out2))
    assert (tmp_path / "td_centroid.yaml").exists()
    assert (tmp_path / "td_centered_instance.yaml").exists()
    assert not out2.exists()


# ──────────────────────────────────────────────────────────────────────────
# Recommender: single-node multi-instance -> centroid_only
# ──────────────────────────────────────────────────────────────────────────


def _make_stats(num_nodes: int, num_edges: int = 0) -> "DatasetStats":
    """Build a multi-instance DatasetStats with small animals (~10% of frame)."""
    from sleap_nn.config_generator.analyzer import DatasetStats

    node_names = [f"n{i}" for i in range(num_nodes)]
    return DatasetStats(
        slp_path="x.slp",
        num_labeled_frames=10,
        num_videos=1,
        max_height=1000,
        max_width=1000,
        num_channels=1,
        max_instances_per_frame=3,  # multi-instance
        avg_instances_per_frame=3.0,
        max_bbox_size=100.0,
        avg_bbox_size=100.0,  # ~10% of 1000 -> small animals
        avg_bbox_diagonal=140.0,
        num_nodes=num_nodes,
        num_edges=num_edges,
        node_names=node_names,
        edges=[],
        has_tracks=False,
        num_tracks=0,
        estimated_total_bytes=0,
    )


def test_recommender_single_node_multi_instance_picks_centroid_only():
    """A single-node multi-instance skeleton recommends 'centroid_only'."""
    from sleap_nn.config_generator.recommender import recommend_pipeline

    rec = recommend_pipeline(_make_stats(num_nodes=1))
    assert rec.recommended == "centroid_only"
    assert rec.requires_second_model is False
    assert rec.reason  # non-empty explanation


def test_recommender_multi_node_small_animals_still_centroid():
    """A multi-node small-animal skeleton still recommends the paired 'centroid'."""
    from sleap_nn.config_generator.recommender import recommend_pipeline

    rec = recommend_pipeline(_make_stats(num_nodes=5))
    assert rec.recommended == "centroid"
    assert rec.requires_second_model is True


def test_recommend_config_centroid_only_sigma_no_crop():
    """recommend_config for a single-node dataset uses tight sigma, no crop_size."""
    from sleap_nn.config_generator.recommender import recommend_config

    rec = recommend_config(_make_stats(num_nodes=1))
    assert rec.pipeline.recommended == "centroid_only"
    assert rec.sigma == 2.5
    assert rec.crop_size is None


# ──────────────────────────────────────────────────────────────────────────
# CLI: `config ... --pipeline centroid` -> single standalone config file
# ──────────────────────────────────────────────────────────────────────────


def test_cli_config_pipeline_centroid_writes_single_file(tmp_path):
    """`config --pipeline centroid` writes ONE config resolving to 'centroid'."""
    from click.testing import CliRunner
    from omegaconf import OmegaConf

    from sleap_nn.cli import config as config_cmd
    from sleap_nn.config.utils import get_model_type_from_cfg

    out = tmp_path / "out.yaml"
    runner = CliRunner()
    result = runner.invoke(
        config_cmd,
        [FIXTURE_SLP, "--auto", "--pipeline", "centroid", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output

    # Single standalone file — no paired split files.
    assert out.exists()
    assert not (tmp_path / "out_centroid.yaml").exists()
    assert not (tmp_path / "out_centered_instance.yaml").exists()

    cfg = OmegaConf.load(str(out))
    assert get_model_type_from_cfg(cfg) == "centroid"
    assert cfg.data_config.preprocessing.scale == 1.0
    assert cfg.data_config.preprocessing.crop_size is None


# ──────────────────────────────────────────────────────────────────────────
# Bounding-box detector pipelines
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pipeline", BBOX_PIPELINES)
def test_bbox_pipeline_defaults(pipeline):
    """Box detectors: half scale, derived boxes, no rotation, eval on (mAP ckpts)."""
    from sleap_nn.config.training_job_config import verify_training_cfg
    from sleap_nn.config.utils import get_model_type_from_cfg

    gen = ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline(pipeline)
    assert gen.is_bbox and not gen.is_topdown
    cfg = verify_training_cfg(gen.build())
    assert get_model_type_from_cfg(cfg) == pipeline
    data = cfg.data_config
    assert data.bbox_source == "auto" and data.bbox_padding == 0.0
    assert data.preprocessing.scale == 0.5 and data.preprocessing.crop_size is None
    geometric = data.augmentation_config.geometric
    assert geometric.rotation_p == 0.0 and geometric.scale_p == 1.0
    assert cfg.trainer_config.eval.enabled
    unet = cfg.model_config.backbone_config.unet
    if pipeline == "bbox_ssd":
        assert unet.output_stride == 8 and unet.max_stride >= 32
        assert cfg.model_config.head_configs.bbox_ssd.ssd.anchor_sizes is None
    else:
        assert unet.output_stride == 4
        assert cfg.model_config.head_configs.bbox_centernet.size.output_stride == 4


@pytest.mark.parametrize("pipeline", PIPELINES + ["centroid_only"])
def test_pose_pipelines_emit_no_bbox_fields(pipeline):
    """Pose configs keep their schema: no bbox data fields, no bbox head keys."""
    cfg = _build_config(pipeline)
    assert "bbox_source" not in cfg["data_config"]
    assert not set(BBOX_PIPELINES) & set(cfg["model_config"]["head_configs"])


def test_cli_config_pipeline_bbox_ssd_writes_single_file(tmp_path):
    """`config --auto --pipeline bbox_ssd` writes ONE config resolving to bbox_ssd."""
    from click.testing import CliRunner

    from sleap_nn.cli import config as config_cmd
    from sleap_nn.config.utils import get_model_type_from_cfg

    out = tmp_path / "out.yaml"
    result = CliRunner().invoke(
        config_cmd, [FIXTURE_SLP, "--auto", "--pipeline", "bbox_ssd", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists() and not (tmp_path / "out_centroid.yaml").exists()
    assert get_model_type_from_cfg(OmegaConf.load(str(out))) == "bbox_ssd"


def test_tui_bbox_selection_and_restore():
    """Picking a box detector in the TUI builds its config; picking a pose model
    again restores the rotation / eval / output stride the box defaults replaced."""
    pytest.importorskip("textual")
    import asyncio

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
                state._output_stride,
                state._augmentation.rotation_enabled,
                state._evaluation.enabled,
            )
            for pipeline in BBOX_PIPELINES:
                screen._select_type(pipeline)
                cfg = OmegaConf.create(state.build_config())
                assert get_model_type_from_cfg(cfg) == pipeline
                assert cfg.data_config.bbox_source == "auto"
                assert cfg.data_config.augmentation_config.geometric.rotation_p == 0.0
                assert cfg.trainer_config.eval.enabled
            assert state._max_stride >= 32  # SSD's deepest anchor level
            screen._select_type("bottomup")
            after = (
                state._output_stride,
                state._augmentation.rotation_enabled,
                state._evaluation.enabled,
            )
            assert after == before
            # The configure and export steps render for a box pipeline.
            await app._show_step(2)
            await pilot.pause()
            app.query_one("#model-screen")._select_type("bbox_centernet")
            await app._show_step(3)
            await pilot.pause()
            await app._show_step(4)
            await pilot.pause()

    asyncio.run(run())


# ──────────────────────────────────────────────────────────────────────────
# Top-down from boxes: a box detector + a box-scaled centered-instance model
# ──────────────────────────────────────────────────────────────────────────

TOPDOWN_BBOX_PIPELINES = ["topdown_bbox_centernet", "topdown_bbox_ssd"]
# The centered-instance config does not depend on the detector style.
TOPDOWN_BBOX_CI_GOLDEN = GOLDEN_DIR / "topdown_bbox_centered_instance.yaml"


def _normalized(cfg) -> dict:
    """``cfg`` as a plain dict with the labels path replaced by the placeholder."""
    cfg.data_config.train_labels_path = [PLACEHOLDER_PATH]
    return yaml.safe_load(OmegaConf.to_yaml(cfg))


@pytest.mark.parametrize("pipeline", TOPDOWN_BBOX_PIPELINES)
def test_topdown_bbox_pair_matches_goldens(pipeline):
    """Stage 1 is exactly the standalone detector config; stage 2 the box-crop CI.

    To intentionally update the CI golden, write
    ``build_centered_instance()`` of either pipeline (normalized like
    ``test_yaml_matches_golden``) to ``topdown_bbox_centered_instance.yaml``.
    """
    gen = ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline(pipeline)
    assert gen.is_topdown and gen.is_topdown_bbox and not gen.is_bbox
    detector = gen.detector_pipeline
    assert detector == pipeline.replace("topdown_", "")
    golden = yaml.safe_load((GOLDEN_DIR / f"{detector}.yaml").read_text())
    assert _normalized(gen.build_detector()) == golden
    assert _normalized(gen.build()) == golden  # build() is stage 1, as for topdown
    ci_golden = yaml.safe_load(TOPDOWN_BBOX_CI_GOLDEN.read_text())
    assert _normalized(gen.build_centered_instance()) == ci_golden


@pytest.mark.parametrize("pipeline", TOPDOWN_BBOX_PIPELINES)
def test_topdown_bbox_instance_config(pipeline):
    """The CI config cuts box-scaled crops from the detector's training boxes.

    Crops are auto-sized and centered on the box, and the CI keeps the pose
    augmentation that the detector drops.
    """
    from sleap_nn.config.training_job_config import verify_training_cfg
    from sleap_nn.config.utils import get_model_type_from_cfg

    gen = ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline(pipeline)
    detector = verify_training_cfg(gen.build_detector())
    ci = verify_training_cfg(gen.build_centered_instance())
    assert get_model_type_from_cfg(detector) == gen.detector_pipeline
    assert get_model_type_from_cfg(ci) == "centered_instance"

    pre = ci.data_config.preprocessing
    assert pre.crop_mode == "bbox" and pre.crop_bbox_aspect == "square"
    assert pre.crop_bbox_padding == 0.1 and pre.crop_bbox_min_size is None
    assert pre.crop_size is None and pre.scale == 1.0
    confmaps = ci.model_config.head_configs.centered_instance.confmaps
    assert confmaps.centroid_method == "bbox_center" and confmaps.anchor_part is None
    for key in ("bbox_source", "bbox_padding"):
        assert ci.data_config[key] == detector.data_config[key]

    # Rotation off / eval on are the detector's alone.
    assert detector.data_config.augmentation_config.geometric.rotation_p == 0.0
    assert ci.data_config.augmentation_config.geometric.rotation_p == 1.0
    assert detector.trainer_config.eval.enabled and not ci.trainer_config.eval.enabled


def test_topdown_bbox_instance_crop_sized_by_trainer():
    """The trainer sizes the generated config's box-scaled crops from the boxes."""
    pytest.importorskip("torch")
    from sleap_nn.training.model_trainer import ModelTrainer

    gen = ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline("topdown_bbox_ssd")
    cfg = gen.build_centered_instance()
    cfg.data_config.val_labels_path = []
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    pre = trainer.config.data_config.preprocessing
    assert pre.crop_size > 0 and pre.crop_size % 16 == 0  # rounded to max_stride
    assert pre.crop_bbox_min_size > 0
    ModelTrainer(gen.build_detector())  # The detector stage is accepted too.


def test_topdown_bbox_save_and_to_yaml(tmp_path):
    """save() writes the detector + CI pair, named by the detector; to_yaml() both."""
    gen = (
        ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline("topdown_bbox_centernet")
    )
    out = tmp_path / "td.yaml"
    gen.save(str(out))
    assert gen.topdown_paths(str(out)) == [
        ("detector", tmp_path / "td_bbox_centernet.yaml"),
        ("instance", tmp_path / "td_centered_instance.yaml"),
    ]
    assert OmegaConf.load(tmp_path / "td_bbox_centernet.yaml") == gen.build_detector()
    assert (
        OmegaConf.load(tmp_path / "td_centered_instance.yaml")
        == gen.build_centered_instance()
    )
    assert not out.exists() and not (tmp_path / "td_centroid.yaml").exists()

    text = gen.to_yaml()
    assert text.startswith("# === DETECTOR CONFIG ===")
    assert "# === CENTERED INSTANCE CONFIG ===" in text and "crop_mode: bbox" in text

    # The centroid top-down pair is named as before.
    paired = ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline("centroid")
    assert [stage for stage, _ in paired.topdown_paths(str(out))] == [
        "centroid",
        "instance",
    ]
    assert paired.to_yaml().startswith("# === CENTROID CONFIG ===")


def test_cli_config_pipeline_topdown_bbox_writes_pair(tmp_path):
    """`config --auto --pipeline topdown_bbox_ssd` writes the detector + CI pair."""
    from click.testing import CliRunner

    from sleap_nn.cli import config as config_cmd
    from sleap_nn.config.utils import get_model_type_from_cfg

    out = tmp_path / "out.yaml"
    result = CliRunner().invoke(
        config_cmd,
        [FIXTURE_SLP, "--auto", "--pipeline", "topdown_bbox_ssd", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    detector_path = tmp_path / "out_bbox_ssd.yaml"
    ci_path = tmp_path / "out_centered_instance.yaml"
    assert result.output.splitlines() == [
        f"Saved detector config to: {detector_path}",
        f"Saved instance config to: {ci_path}",
    ]
    assert not out.exists() and not (tmp_path / "out_centroid.yaml").exists()
    assert get_model_type_from_cfg(OmegaConf.load(detector_path)) == "bbox_ssd"
    ci = OmegaConf.load(ci_path)
    assert get_model_type_from_cfg(ci) == "centered_instance"
    assert ci.data_config.preprocessing.crop_mode == "bbox"


def test_cli_config_pipeline_topdown_messages_unchanged(tmp_path):
    """The centroid top-down pair prints the same two lines as before."""
    from click.testing import CliRunner

    from sleap_nn.cli import config as config_cmd

    out = tmp_path / "out.yaml"
    result = CliRunner().invoke(
        config_cmd, [FIXTURE_SLP, "--auto", "--pipeline", "topdown", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        f"Saved centroid config to: {tmp_path / 'out_centroid.yaml'}",
        f"Saved instance config to: {tmp_path / 'out_centered_instance.yaml'}",
    ]


def test_tui_topdown_bbox_selection(tmp_path):
    """The TUI treats a box top-down pick as a two-model pair.

    It builds and saves a detector config and a box-scaled CI config; a pose
    pick restores the settings the detector defaults replaced.
    """
    pytest.importorskip("textual")
    import asyncio

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
                state._output_stride,
                state._augmentation.rotation_enabled,
                state._evaluation.enabled,
            )
            for pipeline in TOPDOWN_BBOX_PIPELINES:
                screen._select_type(pipeline)
                assert state.is_topdown and state.is_topdown_bbox
                detector = pipeline.replace("topdown_", "")
                assert state.first_stage_name == detector
                det = OmegaConf.create(state.build_centroid_config())
                ci = OmegaConf.create(state.build_centered_instance_config())
                assert get_model_type_from_cfg(det) == detector
                assert det.data_config.augmentation_config.geometric.rotation_p == 0.0
                assert det.trainer_config.eval.enabled
                assert get_model_type_from_cfg(ci) == "centered_instance"
                pre = ci.data_config.preprocessing
                assert pre.crop_mode == "bbox" and pre.crop_size is None
                confmaps = ci.model_config.head_configs.centered_instance.confmaps
                assert confmaps.centroid_method == "bbox_center"
                assert confmaps.anchor_part is None
                assert ci.data_config.bbox_source == det.data_config.bbox_source
                assert ci.data_config.bbox_padding == det.data_config.bbox_padding
                # The CI keeps its own (pose) augmentation.
                assert ci.data_config.augmentation_config.geometric.rotation_p == 1.0
            assert state._max_stride >= 32  # SSD's deepest anchor level

            det_path, ci_path = state.save_dual(str(tmp_path / "labels"))
            assert det_path == str(tmp_path / "labels_bbox_ssd.yaml")
            assert ci_path == str(tmp_path / "labels_centered_instance.yaml")
            assert OmegaConf.load(ci_path).data_config.preprocessing.crop_mode == "bbox"

            screen._select_type("centroid")
            assert state.first_stage_name == "centroid"
            assert not state.is_topdown_bbox
            ci = OmegaConf.create(state.build_centered_instance_config())
            assert "crop_mode" not in ci.data_config.preprocessing
            assert (
                state._augmentation.rotation_enabled,
                state._evaluation.enabled,
            ) == (
                before[1],
                before[2],
            )

            # The configure and export steps render the pair.
            await app._show_step(2)
            await pilot.pause()
            app.query_one("#model-screen")._select_type("topdown_bbox_centernet")
            await app._show_step(3)
            await pilot.pause()
            assert app.query_one("#topdown-tabs")
            await app._show_step(4)
            await pilot.pause()

    asyncio.run(run())
