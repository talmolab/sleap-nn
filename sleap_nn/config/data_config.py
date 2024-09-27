@attr.s(auto_attribs=True)
class DataConfig:
    """Data configuration.

    labels: Configuration options related to user labels for training or testing.
    preprocessing: Configuration options related to data preprocessing.
    instance_cropping: Configuration options related to instance cropping for centroid
        and topdown models.
    """

    labels: LabelsConfig = attr.ib(factory=LabelsConfig)
    preprocessing: PreprocessingConfig = attr.ib(factory=PreprocessingConfig)
    instance_cropping: InstanceCroppingConfig = attr.ib(factory=InstanceCroppingConfig)
