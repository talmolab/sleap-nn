"""Tests for the serializable configuration classes for specifying all data configuration parameters.

These configuration classes are intended to specify all
the parameters required to initialize the data config.
"""

import pytest
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
                "rotation": True,
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

    # Test skeletons
    assert config.skeletons == None


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
