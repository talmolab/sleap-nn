"""Tests for the Serializable configuration classes for specifying all training job parameters.

These configuration classes are intended to specify all the parameters required to run
a training job or perform inference from a serialized one.

They are explicitly not intended to implement any of the underlying functionality that
they parametrize. This serves two purposes:

    1. Parameter specification through simple attributes. These can be read/edited by a
        human, as well as easily be serialized/deserialized to/from simple dictionaries
        and YAML.

    2. Decoupling from the implementation. This makes it easier to design functional
        modules with attributes/parameters that contain objects that may not be easily
        serializable or may implement additional logic that relies on runtime
        information or other parameters.

In general, classes that implement the actual functionality related to these
configuration classes should provide a classmethod for instantiation from the
configuration class instances. This makes it easier to implement other logic not related
to the high level parameters at creation time.

Conveniently, this format also provides a single location where all user-facing
parameters are aggregated and documented for end users (as opposed to developers).
"""

import pytest
import os
import tempfile
from sleap_nn.config.training_job_config import TrainingJobConfig
from sleap_nn.config.model_config import ModelConfig
from sleap_nn.config.data_config import DataConfig
from sleap_nn.config.trainer_config import TrainerConfig, EarlyStoppingConfig
from sleap_nn.config.data_config import IntensityConfig
from omegaconf import DictConfig, OmegaConf, MissingMandatoryValue, ValidationError
from dataclasses import asdict
from loguru import logger
from _pytest.logging import LogCaptureFixture
import json
from omegaconf import MISSING
from pprint import pprint


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


def sample_config():
    """Provide a sample configuration for testing."""
    return {
        "name": "TestConfig",
        "description": "A sample configuration for testing.",
        "data_config": DataConfig(
            train_labels_path="example_train_path",
            val_labels_path="example_val_path",
            provider="default",
        ),
        "model_config": ModelConfig(
            init_weights="default",
            backbone_config="unet",
        ),
        "trainer_config": TrainerConfig(early_stopping=EarlyStoppingConfig()),
    }


def test_intensity_config_validation_logging(caplog):
    """Test IntensityConfig validation and logging."""
    with pytest.raises(ValueError):
        IntensityConfig(gaussian_noise_p=-0.1)  # This should trigger a validation error

    assert "gaussian_noise_p" in caplog.text


def test_to_sleap_nn_cfg():
    """Test serializing a TrainingJobConfig to YAML."""
    cfg = sample_config()
    config_dict = {
        "name": cfg["name"],
        "description": cfg["description"],
        "data_config": {
            "train_labels_path": cfg["data_config"].train_labels_path,
            "val_labels_path": cfg["data_config"].val_labels_path,
            "provider": cfg["data_config"].provider,
        },
        "model_config": {
            "init_weights": cfg["model_config"].init_weights,
        },
        "trainer_config": cfg["trainer_config"],  # Include full trainer config
    }
    yaml_data = OmegaConf.to_yaml(config_dict)
    parsed_yaml = OmegaConf.create(yaml_data)

    assert parsed_yaml.name == cfg["name"]
    assert parsed_yaml.description == cfg["description"]
    assert (
        parsed_yaml.data_config.train_labels_path
        == cfg["data_config"].train_labels_path
    )
    assert parsed_yaml.data_config.val_labels_path == cfg["data_config"].val_labels_path
    assert parsed_yaml.data_config.provider == cfg["data_config"].provider

    assert parsed_yaml.model_config.init_weights == cfg["model_config"].init_weights
    assert parsed_yaml.trainer_config == cfg["trainer_config"]


# def test_missing_attributes(sample_config):
#     """Test creating a TrainingJobConfig from a valid YAML string."""
#     config_dict = {
#         "name": sample_config["name"],
#         "description": sample_config["description"],
#         "data_config": {
#             "provider": sample_config["data_config"].provider,
#         },
#         "model_config": {
#             "init_weights": sample_config["model_config"].init_weights,
#         },
#         "trainer_config": {
#             "early_stopping": {
#                 "patience": sample_config["trainer_config"].early_stopping.patience,
#             },
#         },
#     }
#     yaml_data = OmegaConf.to_yaml(config_dict)

#     with pytest.raises(MissingMandatoryValue):
#         config = TrainingJobConfig.from_yaml(yaml_data)


def test_load_bottomup_multiclass_training_config_from_file(
    bottomup_multiclass_training_config_path,
):
    """Test the load_sleap_config method with a sample bottomup multiclass legacy
    configuration from a JSON file.
    """
    json_file_path = bottomup_multiclass_training_config_path

    config = TrainingJobConfig.load_sleap_config(json_file_path)

    # Assertions to check if the output matches expected values
    assert config.model_config.head_configs.multi_class_bottomup.class_maps.classes == [
        "female",
        "male",
    ]
    assert config.data_config.train_labels_path is None
    assert config.data_config.val_labels_path is None
    assert config.model_config.backbone_config.unet.filters == 8
    assert config.model_config.backbone_config.unet.max_stride == 16
    assert config.trainer_config.max_epochs == 200
    assert config.trainer_config.optimizer_name == "Adam"
    assert config.trainer_config.optimizer.lr == 0.0001
    assert config.trainer_config.trainer_devices is None  # Default value
    assert config.trainer_config.trainer_accelerator == "auto"  # Default value
    assert config.trainer_config.enable_progress_bar is True  # Default value
    assert config.trainer_config.train_data_loader.batch_size == 4  # From the JSON file
    assert (
        config.trainer_config.lr_scheduler.reduce_lr_on_plateau is not None
    )  # From the JSON file
    assert (
        config.trainer_config.early_stopping.stop_training_on_plateau is True
    )  # From the JSON file

    # Test by loading with JSON-formatted string
    with open(json_file_path, "r") as f:
        slp_cfg_json = json.load(f)

    config = TrainingJobConfig.load_sleap_config_from_json(slp_cfg_json)

    # Assertions to check if the output matches expected values
    assert config.data_config.train_labels_path is None
    assert config.data_config.val_labels_path is None
    assert config.model_config.backbone_config.unet.filters == 8
    assert config.model_config.backbone_config.unet.max_stride == 16
    assert config.trainer_config.max_epochs == 200
    assert config.trainer_config.optimizer_name == "Adam"
    assert config.trainer_config.optimizer.lr == 0.0001
    assert config.trainer_config.trainer_devices is None  # Default value
    assert config.trainer_config.trainer_device_indices is None  # Default value
    assert config.trainer_config.trainer_accelerator == "auto"  # Default value
    assert config.trainer_config.enable_progress_bar is True  # Default value
    assert config.trainer_config.train_data_loader.batch_size == 4  # From the JSON file
    assert (
        config.trainer_config.lr_scheduler.reduce_lr_on_plateau is not None
    )  # From the JSON file
    assert (
        config.trainer_config.early_stopping.stop_training_on_plateau is True
    )  # From the JSON file


def test_load_bottomup_training_config_from_file(bottomup_training_config_path):
    """Test the load_sleap_config function with a sample bottomup configuration from a JSON file."""
    json_file_path = bottomup_training_config_path

    # Load the configuration using the load_sleap_config method
    config = TrainingJobConfig.load_sleap_config(json_file_path)

    # Assertions to check if the output matches expected values
    assert config.data_config.train_labels_path is None
    assert config.data_config.val_labels_path is None
    assert config.model_config.backbone_config.unet.filters == 16
    assert config.model_config.backbone_config.unet.max_stride == 8
    assert config.model_config.head_configs.bottomup.confmaps.part_names == ["A", "B"]
    assert config.model_config.head_configs.bottomup.confmaps.sigma == 1.5
    assert config.model_config.head_configs.bottomup.pafs.sigma == 50


def test_load_centered_instance_training_config_from_file(
    centered_instance_training_config_path,
):
    """Test the load_sleap_config function with a sample centered instance configuration
    from a JSON file.
    """
    json_file_path = centered_instance_training_config_path

    # Load the configuration using the load_sleap_config method
    config = TrainingJobConfig.load_sleap_config(json_file_path)

    # Assertions to check if the output matches expected values
    assert config.data_config.train_labels_path is None
    assert config.data_config.val_labels_path is None
    assert config.model_config.head_configs.centered_instance.confmaps.part_names == [
        "A",
        "B",
    ]
    assert config.model_config.head_configs.centered_instance.confmaps.sigma == 1.5

    assert len(config.data_config.skeletons[0]["nodes"]) == 2
    assert len(config.data_config.skeletons[0]["edges"]) == 1
    assert config.data_config.skeletons[0]["name"] == "Skeleton-0"


def test_load_centered_instance_with_scaling_config_from_file(
    centered_instance_with_scaling_training_config_path,
):
    """Test loading centered instance with scaling configuration."""
    json_file_path = centered_instance_with_scaling_training_config_path

    config = TrainingJobConfig.load_sleap_config(json_file_path)

    # Test model backbone config
    assert config.model_config.backbone_config.unet.filters == 16
    assert config.model_config.backbone_config.unet.max_stride == 8
    assert config.model_config.backbone_config.unet.output_stride == 2

    # Test centered instance head config
    centered = config.model_config.head_configs.centered_instance
    assert centered.confmaps.part_names == ["A", "B"]
    assert centered.confmaps.sigma == 1.5
    assert centered.confmaps.output_stride == 2


def test_load_centroid_training_config_from_file(centroid_training_config_path):
    """Test loading centroid configuration."""
    json_file_path = centroid_training_config_path

    config = TrainingJobConfig.load_sleap_config(json_file_path)

    # Test model backbone config
    assert config.model_config.backbone_config.unet.filters == 16
    assert config.model_config.backbone_config.unet.max_stride == 8
    assert config.model_config.backbone_config.unet.output_stride == 4

    # Test centroid head config
    centroid = config.model_config.head_configs.centroid
    assert centroid.confmaps.sigma == 1.5
    assert centroid.confmaps.output_stride == 4
    assert centroid.confmaps.anchor_part is None


def test_load_single_instance_training_config_from_file(
    single_instance_training_config_path,
):
    """Test loading single instance configuration."""
    json_file_path = single_instance_training_config_path

    config = TrainingJobConfig.load_sleap_config(json_file_path)

    # Test model backbone config
    assert config.model_config.backbone_config.unet.filters == 8
    assert config.model_config.backbone_config.unet.max_stride == 4
    assert config.model_config.backbone_config.unet.output_stride == 4

    # Test single instance head config
    single = config.model_config.head_configs.single_instance
    assert single.confmaps.part_names == ["A", "B"]
    assert single.confmaps.sigma == 5.0
    assert single.confmaps.output_stride == 4
    assert len(config.data_config.skeletons[0]["nodes"]) == 2
    assert len(config.data_config.skeletons[0]["edges"]) == 1
    assert config.data_config.skeletons[0]["name"] == "Skeleton-0"


def test_load_topdown_training_config_from_file(topdown_training_config_path):
    """Test loading topdown configuration."""
    json_file_path = topdown_training_config_path

    config = TrainingJobConfig.load_sleap_config(json_file_path)

    # Test model backbone config
    assert config.model_config.backbone_config.unet.filters == 8
    assert config.model_config.backbone_config.unet.max_stride == 16
    assert config.model_config.backbone_config.unet.output_stride == 2

    assert not config.trainer_config.keep_viz
    assert not config.trainer_config.visualize_preds_during_training

    # Test topdown head config
    topdown = config.model_config.head_configs.multi_class_topdown
    assert topdown.confmaps.part_names == ["head", "thorax"]
    assert topdown.confmaps.sigma == 1.5
    assert topdown.confmaps.output_stride == 2
    assert topdown.confmaps.anchor_part == "thorax"
    assert topdown.class_vectors.classes == ["female", "male"]
    assert topdown.class_vectors.num_fc_layers == 3

    cfg = TrainingJobConfig()
    cfg.data_config.train_labels_path = ["test.slp"]
    cfg.data_config.val_labels_path = ["test.slp"]
    omegacfg = cfg.to_sleap_nn_cfg()
    assert isinstance(omegacfg, DictConfig)
    assert omegacfg.data_config.train_labels_path == ["test.slp"]
