"""Tests for the serializable configuration classes for specifying all data configuration parameters.

These configuration classes are intended to specify all
the parameters required to initialize the data config.
"""

import json
from pathlib import Path

import attrs
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from loguru import logger

from _pytest.logging import LogCaptureFixture

from sleap_nn.config.data_config import (
    DataConfig,
    PreprocessingConfig,
    AugmentationConfig,
    IntensityConfig,
    GeometricConfig,
    validate_proportion,
    validate_test_file_path,
    data_mapper,
)
from sleap_nn.config.training_job_config import TrainingJobConfig
from sleap_nn.data.augmentation import apply_geometric_augmentation


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


def test_data_config_initialization():
    """Test that DataConfig initializes correctly with default values."""
    config = DataConfig(train_labels_path="train.slp", val_labels_path="val.slp")
    assert config.provider == "LabelsReader"
    assert config.train_labels_path == "train.slp"
    assert config.val_labels_path == "val.slp"
    assert config.user_instances_only is True


def test_preprocessing_config_initialization(caplog):
    """Test PreprocessingConfig with valid values."""
    with pytest.raises(ValueError):
        config = PreprocessingConfig(max_height=256, max_width=256, scale=(0.5, 0.5))
    assert "PreprocessingConfig's scale" in caplog.text


def test_preprocessing_config_invalid_scale(caplog):
    """Test that PreprocessingConfig raises an error for invalid scale values."""
    with pytest.raises(ValueError):
        PreprocessingConfig(scale=-1.0)
    assert "PreprocessingConfig's scale" in caplog.text


def test_augmentation_config_initialization():
    """Test AugmentationConfig initialization with default values."""
    config = AugmentationConfig(intensity=IntensityConfig, geometric=GeometricConfig())
    assert config.intensity is not None
    assert config.geometric is not None


def test_intensity_config_validation(caplog):
    """Test validation rules in IntensityConfig."""
    with pytest.raises(ValueError):
        IntensityConfig(uniform_noise_min=-0.1)

    with pytest.raises(ValueError):
        IntensityConfig(uniform_noise_max=1.5)

    with pytest.raises(ValueError):
        IntensityConfig(uniform_noise_p=1.5)
    assert "uniform_noise_p" in caplog.text


def test_intensity_config_initialization():
    """Test IntensityConfig with valid values."""
    config = IntensityConfig(
        uniform_noise_min=0.1,
        uniform_noise_max=0.9,
        gaussian_noise_mean=0.0,
        gaussian_noise_std=1.0,
        contrast_p=0.5,
    )
    assert config.uniform_noise_min == 0.1
    assert config.uniform_noise_max == 0.9
    assert config.contrast_p == 0.5


def test_geometric_config_validation(caplog):
    """Test validation rules in GeometricConfig."""
    with pytest.raises(ValueError):
        GeometricConfig(affine_p=1.5)
    assert "affine_p" in caplog.text

    with pytest.raises(ValueError):
        GeometricConfig(erase_p=-0.5)
    assert "erase_p" in caplog.text


def test_geometric_config_initialization():
    """Test GeometricConfig with valid values."""
    config = GeometricConfig(
        rotation_max=30.0,
        rotation_min=-30.0,
        scale_min=0.8,
        scale_max=1.2,
    )
    assert config.rotation_max == 30.0
    assert config.rotation_min == -30.0
    assert config.scale_min == 0.8
    assert config.scale_max == 1.2


def test_geometric_config_flip_field():
    """flip_p defaults to disabled and is validated as a proportion."""
    config = GeometricConfig()
    assert config.flip_p == 0.0

    config = GeometricConfig(flip_p=1.0)
    assert config.flip_p == 1.0

    with pytest.raises(ValueError):
        GeometricConfig(flip_p=1.5)


def test_get_aug_config_flip_preset():
    """The 'flip' string preset enables flip_p in the geometric config."""
    from sleap_nn.config.get_config import get_aug_config

    aug = get_aug_config(geometric_aug="flip")
    assert aug.geometric.flip_p == 1.0


def test_validate_proportion(caplog):
    """Test the validate_proportion helper function."""
    with pytest.raises(ValueError):
        IntensityConfig(uniform_noise_p=1.1)
    assert "uniform_noise_p" in caplog.text

    with pytest.raises(ValueError):
        IntensityConfig(uniform_noise_p=-100)
    assert "uniform_noise_p" in caplog.text

    # Should pass
    validate_proportion(None, None, 0.5)


def test_data_mapper():
    """Test the data_mapper function with a sample legacy configuration."""
    legacy_config = {
        "data": {
            "labels": {
                "training_labels": "notMISSING",
                "validation_labels": "notMISSING",
            },
            "preprocessing": {
                "ensure_rgb": True,
                "target_height": 256,
                "target_width": 256,
                "input_scaling": 0.5,
            },
        },
        "optimization": {
            "augmentation_config": {
                "uniform_noise_min_val": 0.1,
                "uniform_noise_max_val": 0.9,
                "uniform_noise": 0.8,
                "gaussian_noise_mean": 0.0,
                "gaussian_noise_stddev": 1.0,
                "gaussian_noise": 0.7,
                "contrast_min_gamma": 0.6,
                "contrast_max_gamma": 1.8,
                "contrast": 0.9,
                "brightness_min_val": 0.8,
                "brightness_max_val": 1.2,
                "brightness": 0.6,
                "rotation_min_angle": -90.0,
                "rotation_max_angle": 90.0,
                "rotate": True,
                "scale_min": 0.8,
                "scale_max": 1.2,
                "scale": False,
            },
        },
    }

    config = data_mapper(legacy_config)

    # Test preprocessing config
    assert config.preprocessing.ensure_rgb is True
    assert config.preprocessing.ensure_grayscale is False
    assert config.preprocessing.max_height == 256
    assert config.preprocessing.max_width == 256
    assert config.preprocessing.scale == 0.5
    assert config.preprocessing.crop_size is None
    assert config.preprocessing.min_crop_size == 100

    # Test augmentation config
    assert config.use_augmentations_train is True
    assert config.augmentation_config is not None

    # Test intensity config
    intensity = config.augmentation_config.intensity
    assert intensity.uniform_noise_min == 0.1
    assert intensity.uniform_noise_max == 0.9
    assert intensity.uniform_noise_p == 0.8
    assert intensity.gaussian_noise_mean == 0.0
    assert intensity.gaussian_noise_std == 1.0
    assert intensity.gaussian_noise_p == 0.7
    assert intensity.contrast_min == 0.6
    assert intensity.contrast_max == 1.8
    assert intensity.contrast_p == 0.9
    assert intensity.brightness_min == 0.8
    assert intensity.brightness_max == 1.2
    assert intensity.brightness_p == 0.6

    # Test geometric config
    geometric = config.augmentation_config.geometric
    assert geometric.rotation_min == -90.0
    assert geometric.rotation_max == 90.0
    assert geometric.scale_min == 0.8
    assert geometric.scale_max == 1.2
    assert geometric.rotation_p == 1.0
    assert geometric.scale_p == 0.0
    assert geometric.affine_p == 0.0

    # Test skeletons
    assert config.skeletons == None


def test_data_mapper_negative_legacy_min_values():
    """Test that negative legacy min values are clamped instead of failing validation.

    Classic SLEAP's additive/imgaug augmentation could hold negative or inert
    placeholder values (e.g. `brightness_min_val: -10.0`) for fields that map into
    new `IntensityConfig` fields validated `>= 0` (multiplicative factors centered
    on 1.0). This must not raise, even when the corresponding augmentation is
    disabled. Regression test for #684.
    """
    legacy_config = {
        "data": {
            "labels": {
                "training_labels": "notMISSING",
                "validation_labels": "notMISSING",
            },
            "preprocessing": {
                "ensure_rgb": True,
                "target_height": 256,
                "target_width": 256,
                "input_scaling": 0.5,
            },
        },
        "optimization": {
            "augmentation_config": {
                "uniform_noise_min_val": -5.0,
                "uniform_noise": False,
                "contrast_min_gamma": -0.5,
                "contrast_max_gamma": -0.2,
                "contrast": False,
                "brightness_min_val": -10.0,
                "brightness_max_val": 1.2,
                "brightness": False,
            },
        },
    }

    config = data_mapper(legacy_config)

    intensity = config.augmentation_config.intensity
    assert intensity.uniform_noise_min == 0.0
    assert intensity.contrast_min == 0.0
    assert intensity.contrast_max == 0.0
    assert intensity.brightness_min == 0.0
    assert intensity.brightness_max == 1.2


def test_data_mapper_flip(caplog):
    """Test that legacy random_flip/flip_horizontal map to the new flip_p field."""
    base_legacy_config = {
        "data": {
            "labels": {
                "training_labels": "notMISSING",
                "validation_labels": "notMISSING",
            },
        },
        "optimization": {"augmentation_config": {}},
    }

    # random_flip disabled -> flip_p stays at default (0.0)
    config = data_mapper(base_legacy_config)
    assert config.augmentation_config.geometric.flip_p == 0.0

    # random_flip enabled + horizontal -> flip_p is set
    horizontal_config = {
        "data": base_legacy_config["data"],
        "optimization": {
            "augmentation_config": {"random_flip": True, "flip_horizontal": True}
        },
    }
    config = data_mapper(horizontal_config)
    assert config.augmentation_config.geometric.flip_p == 0.5

    # random_flip enabled + vertical -> unsupported, flip_p stays at default and warns
    vertical_config = {
        "data": base_legacy_config["data"],
        "optimization": {
            "augmentation_config": {"random_flip": True, "flip_horizontal": False}
        },
    }
    config = data_mapper(vertical_config)
    assert config.augmentation_config.geometric.flip_p == 0.0
    assert "vertical flip" in caplog.text


def _legacy_aug_config(augmentation_config):
    """Wrap a legacy `augmentation_config` dict in a minimal legacy config."""
    return {
        "data": {
            "labels": {
                "training_labels": "notMISSING",
                "validation_labels": "notMISSING",
            },
        },
        "optimization": {"augmentation_config": augmentation_config},
    }


@pytest.mark.parametrize(
    "rotate,scale,expected_rotation_p,expected_scale_p",
    [
        (False, False, 0.0, 0.0),
        (True, False, 1.0, 0.0),
        (False, True, 0.0, 1.0),
        (True, True, 1.0, 1.0),
    ],
)
def test_data_mapper_rotate_scale_flags(
    rotate, scale, expected_rotation_p, expected_scale_p
):
    """Legacy `rotate`/`scale` flags map to independent per-transform probabilities.

    Legacy SLEAP applies each enabled transform with p=1.0 and skips disabled ones.
    `GeometricConfig` defaults `rotation_p`/`scale_p` to 1.0 and the augmenter ignores
    `affine_p` whenever a per-transform probability is set, so the mapper must set
    them explicitly or a disabled transform is still applied.
    """
    config = data_mapper(
        _legacy_aug_config(
            {
                "rotate": rotate,
                "rotation_min_angle": -45.0,
                "rotation_max_angle": 45.0,
                "scale": scale,
                "scale_min": 0.5,
                "scale_max": 1.5,
            }
        )
    )
    geometric = config.augmentation_config.geometric
    assert geometric.rotation_p == expected_rotation_p
    assert geometric.scale_p == expected_scale_p
    assert geometric.affine_p == 0.0
    # Ranges are carried over regardless of the flags; they are inert when p=0.
    assert (geometric.rotation_min, geometric.rotation_max) == (-45.0, 45.0)
    assert (geometric.scale_min, geometric.scale_max) == (0.5, 1.5)


def test_data_mapper_rotate_scale_missing_keys():
    """No geometric keys at all means legacy defaults: rotation and scale off."""
    geometric = data_mapper(_legacy_aug_config({})).augmentation_config.geometric
    assert geometric.rotation_p == 0.0
    assert geometric.scale_p == 0.0
    assert geometric.affine_p == 0.0


@pytest.mark.parametrize("scale", [True, False, None])
def test_data_mapper_scale_without_range(scale):
    """A `scale` key without `scale_min`/`scale_max` falls back to defaults.

    Previously any non-None `scale` (including `False`) indexed `scale_min`
    directly and raised `KeyError`.
    """
    geometric = data_mapper(
        _legacy_aug_config({"scale": scale})
    ).augmentation_config.geometric
    assert geometric.scale_p == (1.0 if scale else 0.0)
    assert (geometric.scale_min, geometric.scale_max) == (0.9, 1.1)


def test_data_mapper_translate_dropped(caplog):
    """Legacy pixel translation cannot be converted, so it is dropped with a warning."""
    geometric = data_mapper(
        _legacy_aug_config({"translate": True, "translate_min": -5, "translate_max": 5})
    ).augmentation_config.geometric
    assert geometric.translate_width == 0.0
    assert geometric.translate_height == 0.0
    assert geometric.translate_p is None
    assert "translation" in caplog.text

    caplog.clear()
    data_mapper(_legacy_aug_config({"translate": False}))
    assert "translation" not in caplog.text


@pytest.mark.parametrize(
    "fixture",
    sorted(
        (Path(__file__).parents[1] / "assets" / "legacy_sleap_json_configs").glob(
            "*.json"
        )
    ),
    ids=lambda p: p.stem,
)
def test_legacy_json_fixtures_respect_disabled_geometric_aug(fixture):
    """Every bundled legacy config has rotate/scale off; loading must keep them off.

    These configs carry `rotation_min_angle=-180`/`rotation_max_angle=180`, so a
    mapping that ignores the flags trains with full random rotation.
    """
    with open(fixture) as f:
        legacy_aug = json.load(f)["optimization"]["augmentation_config"]
    config = TrainingJobConfig.load_sleap_config(str(fixture))
    geometric = config.data_config.augmentation_config.geometric
    assert geometric.rotation_p == (1.0 if legacy_aug["rotate"] else 0.0)
    assert geometric.scale_p == (1.0 if legacy_aug["scale"] else 0.0)
    assert geometric.affine_p == 0.0


@pytest.mark.parametrize(
    "rotate,scale,expect_moved", [(False, False, False), (True, False, True)]
)
def test_data_mapper_geometric_aug_runtime(rotate, scale, expect_moved):
    """End-to-end: mapped legacy config drives the augmenter as the flags say."""
    geometric = data_mapper(
        _legacy_aug_config(
            {
                "rotate": rotate,
                "rotation_min_angle": 30.0,
                "rotation_max_angle": 60.0,
                "scale": scale,
                "scale_min": 0.9,
                "scale_max": 1.1,
            }
        )
    ).augmentation_config.geometric
    image = torch.rand(1, 1, 64, 64)
    instances = torch.tensor([[[[10.0, 10.0], [50.0, 40.0]]]])
    np.random.seed(0)
    for _ in range(10):
        _, out = apply_geometric_augmentation(
            image.clone(), instances.clone(), **attrs.asdict(geometric)
        )
        assert (not torch.allclose(out, instances, atol=1e-3)) == expect_moved


def test_validate_test_file_path():
    """Test the validate_test_file_path validator function."""
    # Test with None (should pass)
    config = DataConfig(test_file_path=None)
    assert config.test_file_path is None

    # Test with string (should pass)
    config = DataConfig(test_file_path="test.slp")
    assert config.test_file_path == "test.slp"

    # Test with list of strings (should pass)
    config = DataConfig(test_file_path=["test1.slp", "test2.slp"])
    assert config.test_file_path == ["test1.slp", "test2.slp"]

    # Test with tuple of strings (should pass)
    config = DataConfig(test_file_path=("test1.slp", "test2.slp"))
    assert config.test_file_path == ("test1.slp", "test2.slp")


def test_validate_test_file_path_invalid(caplog):
    """Test that validate_test_file_path raises error for invalid types."""
    # Test with integer (should fail)
    with pytest.raises(ValueError):
        DataConfig(test_file_path=123)
    assert "test_file_path must be a string or list of strings" in caplog.text

    # Test with list containing non-strings (should fail)
    with pytest.raises(ValueError):
        DataConfig(test_file_path=["test.slp", 123])
    assert "test_file_path must be a string or list of strings" in caplog.text

    # Test with dict (should fail)
    with pytest.raises(ValueError):
        DataConfig(test_file_path={"path": "test.slp"})
    assert "test_file_path must be a string or list of strings" in caplog.text


def test_default_augmentation_enabled():
    """Test that augmentation is enabled by default with rotation and scale applied."""
    config = DataConfig()

    # Augmentation should be enabled by default
    assert config.use_augmentations_train is True

    # augmentation_config should not be None
    assert config.augmentation_config is not None

    # geometric config should be set
    assert config.augmentation_config.geometric is not None

    # rotation and scale should be applied with probability 1.0
    geometric = config.augmentation_config.geometric
    assert geometric.rotation_p == 1.0
    assert geometric.scale_p == 1.0

    # rotation range should be ±15 degrees
    assert geometric.rotation_min == -15.0
    assert geometric.rotation_max == 15.0

    # scale range should be 0.9-1.1
    assert geometric.scale_min == 0.9
    assert geometric.scale_max == 1.1


def test_geometric_config_default_probabilities():
    """Test that GeometricConfig has correct default probabilities."""
    config = GeometricConfig()

    # rotation and scale should always be applied by default
    assert config.rotation_p == 1.0
    assert config.scale_p == 1.0

    # translate should fall back to affine_p (None means use affine_p)
    assert config.translate_p is None

    # affine_p should be 0.0 (only used as fallback)
    assert config.affine_p == 0.0


def test_data_config_augmentation_unique_per_instance():
    """Test that each DataConfig instance gets its own augmentation_config."""
    config1 = DataConfig()
    config2 = DataConfig()

    # Each instance should have its own augmentation_config object
    assert config1.augmentation_config is not config2.augmentation_config
    assert (
        config1.augmentation_config.geometric
        is not config2.augmentation_config.geometric
    )

    # Modifying one should not affect the other
    config1.augmentation_config.geometric.rotation_p = 0.5
    assert config2.augmentation_config.geometric.rotation_p == 1.0
