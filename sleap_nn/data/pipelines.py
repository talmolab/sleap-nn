"""This module defines high level pipeline configurations from providers/transformers.

This allows for convenient ways to configure individual variants of common pipelines, as
well as to define training vs inference versions based on the same configurations.
"""
from typing import Tuple


class TopdownConfmapsPipeline:
    """Pipeline builder for instance-centered confidence map models.

    Attributes:
        data_config: Data-related configuration.
    """

    def __init__(self, data_config: DictConfig) -> None:
        """Initialize the data config."""
        self.data_config = data_config

    def make_training_pipeline(
        self, data_provider: IterDataPipe, filename: str
    ) -> IterDataPipe:
        """Create training pipeline with input data only.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.
            filename: A string path to the name of the `.slp` file.

        Returns:
            An `IterDataPipe` instance configured to produce input examples.
        """
        datapipe = data_provider.from_filename(filename=filename)
        datapipe = Normalizer(datapipe)

        datapipe = InstanceCentroidFinder(datapipe)
        datapipe = InstanceCropper(datapipe, self.data_config.preprocessing.crop_hw)

        if self.data_config.augmentation_config.random_crop.random_crop_p:
            datapipe = KorniaAugmenter(
                datapipe,
                random_crop_hw=self.data_config.augmentation_config.random_crop.random_crop_hw,
                random_crop_p=self.data_config.augmentation_config.random_crop.random_crop_p,
            )

        datapipe = KorniaAugmenter(
            datapipe, **dict(self.data_config.augmentation_config.augmentations)
        )

        datapipe = ConfidenceMapGenerator(
            datapipe,
            sigma=self.data_config.preprocessing.conf_map_gen.sigma,
            output_stride=self.data_config.preprocessing.conf_map_gen.output_stride,
        )
        datapipe = KeyFilter(datapipe, keep_keys=self.data_config.general.keep_keys)

        return datapipe
