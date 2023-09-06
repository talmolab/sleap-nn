"""This module defines high level pipeline configurations from providers/transformers.

This allows for convenient ways to configure individual variants of common pipelines, as
well as to define training vs inference versions based on the same configurations.
"""
from typing import Tuple

import torch
from omegaconf.dictconfig import DictConfig
from torch.utils.data.datapipes.datapipe import IterDataPipe

from sleap_nn.data.augmentation import KorniaAugmenter
from sleap_nn.data.confidence_maps import ConfidenceMapGenerator
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.instance_cropping import InstanceCropper
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.providers import LabelsReader


class SleapDataset(IterDataPipe):
    """Returns image and corresponding heatmap for the DataLoader.

    This class is to return the image and its corresponding confidence map
    to load the dataset with the DataLoader class

    Attributes:
    source_dp: The previous `DataPipe` with samples that contain an `instances` key.
    """

    def __init__(self, source_dp: IterDataPipe) -> None:
        """Initialize SleapDataset with the source `DataPipe."""
        self.dp = source_dp

    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a tuple with the cropped image and the heatmap."""
        for example in self.dp:
            if len(example["instance_image"].shape) == 4:
                example["instance_image"] = example["instance_image"].squeeze(dim=0)
            yield example["instance_image"], example["confidence_maps"]


class TopdownConfmapsPipeline:
    """Pipeline builder for instance-centered confidence map models.

    Attributes:
        data_config: Data-related configuration.
        optimization_config: Optimization-related configuration.
        instance_confmap_head: Instantiated head describing the output centered
            confidence maps tensor.
        offsets_head: Optional head describing the offset refinement maps.
    """

    def __init__(self, data_config: DictConfig) -> None:
        """Initialize the data config."""
        self.data_config = data_config

    def make_base_pipeline(
        self, data_provider: IterDataPipe, filename: str
    ) -> IterDataPipe:
        """Create base pipeline with input data only.

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

        datapipe = ConfidenceMapGenerator(
            datapipe,
            sigma=self.data_config.preprocessing.conf_map_gen.sigma,
            output_stride=self.data_config.preprocessing.conf_map_gen.output_stride,
        )
        datapipe = SleapDataset(datapipe)

        return datapipe

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
        datapipe = SleapDataset(datapipe)

        return datapipe
