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


def _build_config(pipeline: str) -> dict:
    """Build a config for ``pipeline`` and normalize the labels path."""
    gen = ConfigGenerator.from_slp(FIXTURE_SLP).auto().pipeline(pipeline)
    cfg = gen.build()
    cfg.data_config.train_labels_path = [PLACEHOLDER_PATH]
    return yaml.safe_load(OmegaConf.to_yaml(cfg))


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_yaml_matches_golden(pipeline):
    """Generated YAML for each pipeline matches the checked-in golden.

    To intentionally update goldens, run::

        uv run python -c "
        from sleap_nn.config_generator.generator import ConfigGenerator
        from omegaconf import OmegaConf
        from pathlib import Path
        for p in ['single_instance', 'centroid', 'centered_instance',
                  'bottomup', 'multi_class_bottomup', 'multi_class_topdown']:
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


@pytest.mark.parametrize("pipeline", PIPELINES)
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


@pytest.mark.parametrize("pipeline", PIPELINES)
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
