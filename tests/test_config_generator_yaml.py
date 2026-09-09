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
GOLDEN_PIPELINES = PIPELINES + ["centroid_only"]


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
                  'multi_class_topdown']:
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


@pytest.mark.parametrize("pipeline", PIPELINES)
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
