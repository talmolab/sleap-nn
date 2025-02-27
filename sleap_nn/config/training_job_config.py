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
from omegaconf import OmegaConf
import sleap_nn
from sleap_nn.config.data_config import DataConfig
from sleap_nn.config.model_config import ModelConfig
from sleap_nn.config.trainer_config import TrainerConfig
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
    def check_output_strides(cls, config: OmegaConf) -> OmegaConf:
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

    @classmethod
    def from_yaml(cls, yaml_data: Text) -> OmegaConf:
        """Create TrainingJobConfig from YAML-formatted string with schema validation.

        Arguments:
            yaml_data: YAML-formatted string that specifies the configurations.

        Returns:
            A OmegaConf instance parsed from the YAML text, validated against the schema.
        """
        schema = OmegaConf.structured(cls())
        config = OmegaConf.create(yaml_data)
        config = OmegaConf.merge(schema, config)
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        config = TrainingJobConfig.check_output_strides(config)
        return config

    @classmethod
    def load_yaml(cls, filename: Text) -> OmegaConf:
        """Load a training job configuration from a yaml file.

        Arguments:
            filename: Path to a training job configuration YAML file or a directory
                containing `"training_job.yaml"`.

        Returns:
          A OmegaConf instance parsed from the YAML file.
        """
        schema = OmegaConf.structured(cls())
        config = OmegaConf.load(filename)
        config = OmegaConf.merge(schema, config)
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        config = TrainingJobConfig.check_output_strides(config)
        return config

    def to_yaml(self, filename: Optional[Text] = None) -> None:
        """Serialize and optionally save the configuration to YAML format.

        Args:
            filename: Optional path to save the YAML file to. If not provided,
                     the configuration will only be converted to YAML format.
        """
        # Convert attrs objects to nested dictionaries
        config_dict = asdict(self)

        # Handle any special cases (like enums) that need manual conversion
        if config_dict.get("model", {}).get("backbone_type"):
            config_dict["model"]["backbone_type"] = self.model_config.backbone_type

        # Create OmegaConf object and save if filename provided
        conf = OmegaConf.create(config_dict)
        if filename is not None:
            OmegaConf.save(conf, filename)
        return


def load_config(filename: Text, load_training_config: bool = True) -> OmegaConf:
    """Load a training job configuration for a model run.

    Args:
        filename: Path to a YAML file or directory containing `training_job.yaml`.
        load_training_config: If `True` (the default), prefer `training_job.yaml` over
            `initial_config.yaml` if it is present in the same folder.

    Returns:
        The parsed `OmegaConf`.
    """
    return TrainingJobConfig.load_yaml(filename)
