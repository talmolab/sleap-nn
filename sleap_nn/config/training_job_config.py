"""Serializable configuration classes for specifying all training job parameters.

These configuration classes are intended to specify all the parameters required to run
a training job or perform inference from a serialized one.

They are explicitly not intended to implement any of the underlying functionality that
they parametrize. This serves two purposes:

    1. Parameter specification through simple attributes. These can be read/edited by a
        human, as well as easily be serialized/deserialized to/from simple dictionaries
        and JSON.

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

import os
import attrs
import cattr
import sleap
from sleap.nn.config.data_config import DataConfig
from sleap.nn.config.model_config import ModelConfig
from sleap.nn.config.trainer_config import TrainerConfig
import json
from jsmin import jsmin
from typing import Text, Dict, Any, Optional


@attrs
class TrainingJobConfig:
    """Configuration of a training job.

    Attributes:
        data: Configuration options related to the training data.
        model: Configuration options related to the model architecture.
        outputs: Configuration options related to outputs during training.
        name: Optional name for this configuration profile.
        description: Optional description of the configuration.
        sleap_version: Version of SLEAP that generated this configuration.
        filename: Path to this config file if it was loaded from disk.
    """

    data: DataConfig = attr.ib(factory=DataConfig)
    model: ModelConfig = attr.ib(factory=ModelConfig)
    outputs: OutputsConfig = attr.ib(factory=OutputsConfig)
    name: Optional[Text] = ""
    description: Optional[Text] = ""
    sleap_version: Optional[Text] = sleap.__version__
    filename: Optional[Text] = ""

    @classmethod
    def from_yaml(cls, yaml_data: Text) -> "TrainerConfig":
        """Create TrainerConfig from YAML-formatted string.

        Arguments:
            yaml_data: YAML-formatted string that specifies the configurations.

        Returns:
            A TrainerConfig instance parsed from the YAML text.
        """
        config = OmegaConf.create(yaml_data)
        return OmegaConf.to_object(config, cls)

    @classmethod
    def load_yaml(cls, filename: Text) -> "TrainerConfig":
        """Load a training job configuration from a yaml file.

        Arguments:
            filename: Path to a training job configuration YAML file or a directory
                containing `"training_job.yaml"`.

        Returns:
          A TrainerConfig instance parsed from the YAML file.
        """
        config = OmegaConf.load(filename)
        return OmegaConf.to_object(config, cls)

    def to_yaml(self) -> str:
        """Serialize the configuration into YAML-encoded string format.

        Returns:
            The YAML encoded string representation of the configuration.
        """
        config = self.to_dict()
        return OmegaConf.to_yaml(config)

    def save_yaml(self, filename: Text):
        """Save the configuration to a YAML file.

        Arguments:
            filename: Path to save the training job file to.
        """
        with open(filename, "w") as f:
            f.write(self.to_yaml())



def load_config(filename: Text, load_training_config: bool = True) -> TrainingJobConfig:
    """Load a training job configuration for a model run.

    Args:
        filename: Path to a JSON file or directory containing `training_job.json`.
        load_training_config: If `True` (the default), prefer `training_job.json` over
            `initial_config.json` if it is present in the same folder.

    Returns:
        The parsed `TrainingJobConfig`.
    """
    return TrainingJobConfig.load_json(
        filename, load_training_config=load_training_config
    )