@attr.s(auto_attribs=True)
class ModelConfig:
    """Configurations related to model architecture.

    Attributes:
        backbone: Configurations related to the main network architecture.
        heads: Configurations related to the output heads.
        base_checkpoint: Path to model folder for loading a checkpoint. Should contain the .h5 file
    """
