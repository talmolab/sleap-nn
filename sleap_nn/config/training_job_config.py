"""Serializable configuration classes for specifying all training job parameters.

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

from attrs import define, asdict, field
from typing import Text, Optional
from omegaconf import DictConfig, OmegaConf
import sleap_nn
import json
from sleap_nn.config.data_config import DataConfig
from sleap_nn.config.data_config import data_mapper
from sleap_nn.config.model_config import ModelConfig
from sleap_nn.config.model_config import model_mapper
from sleap_nn.config.trainer_config import TrainerConfig
from sleap_nn.config.trainer_config import trainer_mapper
from sleap_nn.config.utils import get_output_strides_from_heads


@define
class TrainingJobConfig:
    """Configuration of a training job.

    Attributes:
        data_config: Configuration options related to the training data.
        model_config: Configuration options related to the model architecture.
        trainer_config: Configuration ooptions related to model training.
        outputs: Configuration options related to outputs during training.
        name: Optional name for this configuration profile.
        description: Optional description of the configuration.
        sleap_nn_version: Version of SLEAP that generated this configuration.
        filename: Path to this config file if it was loaded from disk.
    """

    data_config: DataConfig = field(factory=DataConfig)
    model_config: ModelConfig = field(factory=ModelConfig)
    trainer_config: TrainerConfig = field(factory=TrainerConfig)
    name: Optional[Text] = ""
    description: Optional[Text] = ""
    sleap_nn_version: Optional[Text] = sleap_nn.__version__
    filename: Optional[Text] = ""

    @classmethod
    def check_output_strides(
        cls, config: OmegaConf
    ) -> OmegaConf:  # TODO in model config
        """Check max_stride and output_stride in backbone_config with head_config."""
        output_strides = get_output_strides_from_heads(config.model_config.head_configs)
        # check which backbone architecture
        for k, v in config.model_config.backbone_config.items():
            if v is not None:
                backbone_type = k
                break
        if output_strides:
            config.model_config.backbone_config[f"{backbone_type}"]["output_stride"] = (
                min(output_strides)
            )
            if config.model_config.backbone_config[f"{backbone_type}"][
                "max_stride"
            ] < max(output_strides):
                config.model_config.backbone_config[f"{backbone_type}"][
                    "max_stride"
                ] = max(output_strides)
        return config

    def to_sleap_nn_cfg(self) -> DictConfig:
        """Convert the attrs class to OmegaConf object."""
        config = OmegaConf.structured(self)
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        return config

    @classmethod
    def load_sleap_config(cls, json_file_path: str) -> OmegaConf:
        """Load a SLEAP configuration from a JSON file and convert it to OmegaConf.

        Args:
            cls: The class to instantiate with the loaded configuration.
            json_file_path: Path to a JSON file containing the SLEAP configuration.

        Returns:
            An OmegaConf instance with the loaded configuration.
        """
        with open(json_file_path, "r") as f:
            old_config = json.load(f)

        return cls.load_sleap_config_from_json(old_config)

    @classmethod
    def load_sleap_config_from_json(cls, json_str: str) -> OmegaConf:
        """Load a SLEAP configuration from a JSON string and convert it to OmegaConf.

        Args:
            cls: The class to instantiate with the loaded configuration.
            json_str: JSON-formatted string containing the SLEAP configuration.

        Returns:
            An OmegaConf instance with the loaded configuration.
        """
        data_config = data_mapper(json_str)
        model_config = model_mapper(json_str)
        trainer_config = trainer_mapper(json_str)

        config = cls(
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config,
        )

        schema = OmegaConf.structured(config)
        config_omegaconf = OmegaConf.merge(schema, OmegaConf.create(asdict(config)))
        OmegaConf.to_container(config_omegaconf, resolve=True, throw_on_missing=True)

        return config_omegaconf


def verify_training_cfg(cfg: DictConfig) -> DictConfig:
    """Get sleap-nn training config from a DictConfig object."""
    schema = OmegaConf.structured(TrainingJobConfig())
    config = OmegaConf.merge(schema, cfg)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return config
