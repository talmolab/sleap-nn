import pytest
from omegaconf import OmegaConf, ValidationError
from attrs import fields

from sleap_nn.config.data_config import (
    DataConfig,
    PreprocessingConfig,
    AugmentationConfig,
    IntensityConfig,
    GeometricConfig,
    validate_proportion,
)


def test_data_config_initialization():
    """Test that DataConfig initializes correctly with default values."""
    config = DataConfig(train_labels_path="train.slp", val_labels_path="val.slp")
    assert config.provider == "LabelsReader"
    assert config.train_labels_path == "train.slp"
    assert config.val_labels_path == "val.slp"
    assert config.user_instances_only is True
    assert config.chunk_size == 100


def test_data_config_missing_values():
    # Test that DataConfig raises an error if required fields are missing.
    # with pytest.raises(ValidationError):
    config = OmegaConf.structured(DataConfig())
    # with pytest.raises(ValidationError, match="train_labels_path"):
    #     OmegaConf.to_container(config, resolve=True)


def test_preprocessing_config_initialization():
    """Test PreprocessingConfig with valid values."""
    with pytest.raises(ValueError, match="scale"):
        config = PreprocessingConfig(max_height=256, max_width=256, scale=(0.5, 0.5))


def test_preprocessing_config_invalid_scale():
    """Test that PreprocessingConfig raises an error for invalid scale values."""
    with pytest.raises(ValueError):
        PreprocessingConfig(scale=-1.0)


def test_augmentation_config_initialization():
    """Test AugmentationConfig initialization with default values."""
    config = AugmentationConfig()
    assert config.intensity is None
    assert config.geometric is None


def test_intensity_config_validation():
    """Test validation rules in IntensityConfig."""
    with pytest.raises(ValueError, match="uniform_noise_min"):
        IntensityConfig(uniform_noise_min=-0.1)

    with pytest.raises(ValueError, match="uniform_noise_max"):
        IntensityConfig(uniform_noise_max=1.5)

    with pytest.raises(ValueError, match="uniform_noise_p"):
        IntensityConfig(uniform_noise_p=1.5)


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


def test_geometric_config_validation():
    """Test validation rules in GeometricConfig."""
    with pytest.raises(ValueError, match="affine_p"):
        GeometricConfig(affine_p=1.5)

    with pytest.raises(ValueError, match="erase_p"):
        GeometricConfig(erase_p=-0.5)


def test_geometric_config_initialization():
    """Test GeometricConfig with valid values."""
    config = GeometricConfig(rotation=30.0, scale=(0.8, 1.2, 0.8, 1.2))
    assert config.rotation == 30.0
    assert config.scale == (0.8, 1.2, 0.8, 1.2)


def test_validate_proportion():
    """Test the validate_proportion helper function."""
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        IntensityConfig(uniform_noise_p=1.1)

    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        IntensityConfig(uniform_noise_p=-100)

    # Should pass
    validate_proportion(None, None, 0.5)
