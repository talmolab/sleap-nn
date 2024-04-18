"""This module defines high level pipeline configurations from providers/transformers.

This allows for convenient ways to configure individual variants of common pipelines, as
well as to define training vs inference versions based on the same configurations.
"""

from omegaconf.omegaconf import DictConfig
from sleap_nn.data.augmentation import KorniaAugmenter
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.instance_cropping import InstanceCropper
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.confidence_maps import (
    ConfidenceMapGenerator,
    MultiConfidenceMapGenerator,
)
from sleap_nn.data.general import KeyFilter
from torch.utils.data.datapipes.datapipe import IterDataPipe


class TopdownConfmapsPipeline:
    """Pipeline builder for instance-centered confidence map models.

    Attributes:
        data_config: Data-related configuration.
    """

    def __init__(self, data_config: DictConfig) -> None:
        """Initialize the data config."""
        self.data_config = data_config

    def make_training_pipeline(self, data_provider: IterDataPipe) -> IterDataPipe:
        """Create training pipeline with input data only.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.

        Returns:
            An `IterDataPipe` instance configured to produce input examples.
        """
        datapipe = data_provider
        datapipe = Normalizer(datapipe)

        if self.data_config.augmentation_config.use_augmentations:
            datapipe = KorniaAugmenter(
                datapipe,
                **dict(self.data_config.augmentation_config.augmentations.intensity),
                image_key="image",
                instance_key="instances",
            )

        datapipe = InstanceCentroidFinder(
            datapipe, anchor_ind=self.data_config.preprocessing.anchor_ind
        )
        datapipe = InstanceCropper(datapipe, self.data_config.preprocessing.crop_hw)

        if self.data_config.augmentation_config.random_crop.random_crop_p:
            datapipe = KorniaAugmenter(
                datapipe,
                random_crop_hw=self.data_config.augmentation_config.random_crop.random_crop_hw,
                random_crop_p=self.data_config.augmentation_config.random_crop.random_crop_p,
                image_key="instance_image",
                instance_key="instance",
            )

        if self.data_config.augmentation_config.use_augmentations:
            datapipe = KorniaAugmenter(
                datapipe,
                **dict(self.data_config.augmentation_config.augmentations.geometric),
                image_key="instance_image",
                instance_key="instance",
            )

        datapipe = ConfidenceMapGenerator(
            datapipe,
            sigma=self.data_config.preprocessing.conf_map_gen.sigma,
            output_stride=self.data_config.preprocessing.conf_map_gen.output_stride,
            image_key="instance_image",
            instance_key="instance",
        )
        datapipe = KeyFilter(
            datapipe,
            keep_keys=[
                "image",
                "video_idx",
                "frame_idx",
                "centroid",
                "instance",
                "instance_bbox",
                "instance_image",
                "confidence_maps",
                "num_instances",
                "centroids",
                "orig_size",
            ],
        )

        return datapipe


class SingleInstanceConfmapsPipeline:
    """Pipeline builder for single-instance confidence map models.

    Attributes:
        data_config: Data-related configuration.
    """

    def __init__(self, data_config: DictConfig) -> None:
        """Initialize the data config."""
        self.data_config = data_config

    def make_training_pipeline(self, data_provider: IterDataPipe) -> IterDataPipe:
        """Create training pipeline with input data only.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.

        Returns:
            An `IterDataPipe` instance configured to produce input examples.
        """
        datapipe = data_provider
        datapipe = Normalizer(datapipe)

        if self.data_config.augmentation_config.use_augmentations:
            datapipe = KorniaAugmenter(
                datapipe,
                **dict(self.data_config.augmentation_config.augmentations.intensity),
                **dict(self.data_config.augmentation_config.augmentations.geometric),
                image_key="image",
                instance_key="instances",
            )

        if self.data_config.augmentation_config.random_crop.random_crop_p:
            datapipe = KorniaAugmenter(
                datapipe,
                random_crop_hw=self.data_config.augmentation_config.random_crop.random_crop_hw,
                random_crop_p=self.data_config.augmentation_config.random_crop.random_crop_p,
                image_key="image",
                instance_key="instances",
            )

        datapipe = ConfidenceMapGenerator(
            datapipe,
            sigma=self.data_config.preprocessing.conf_map_gen.sigma,
            output_stride=self.data_config.preprocessing.conf_map_gen.output_stride,
            image_key="image",
            instance_key="instances",
        )
        datapipe = KeyFilter(
            datapipe,
            keep_keys=[
                "image",
                "video_idx",
                "frame_idx",
                "instances",
                "confidence_maps",
                "orig_size",
            ],
        )

        return datapipe


class CentroidConfmapsPipeline:
    """Pipeline builder for centroid confidence map models.

    Attributes:
        data_config: Data-related configuration.
    """

    def __init__(self, data_config: DictConfig) -> None:
        """Initialize the data config."""
        self.data_config = data_config

    def make_training_pipeline(self, data_provider: IterDataPipe) -> IterDataPipe:
        """Create training pipeline with input data only.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.

        Returns:
            An `IterDataPipe` instance configured to produce input examples.
        """
        datapipe = data_provider
        datapipe = Normalizer(datapipe)

        if self.data_config.augmentation_config.use_augmentations:
            datapipe = KorniaAugmenter(
                datapipe,
                **dict(self.data_config.augmentation_config.augmentations.intensity),
                image_key="image",
                instance_key="instances",
            )

        datapipe = InstanceCentroidFinder(
            datapipe, anchor_ind=self.data_config.preprocessing.anchor_ind
        )

        if self.data_config.augmentation_config.random_crop.random_crop_p:
            datapipe = KorniaAugmenter(
                datapipe,
                random_crop_hw=self.data_config.augmentation_config.random_crop.random_crop_hw,
                random_crop_p=self.data_config.augmentation_config.random_crop.random_crop_p,
                image_key="image",
                instance_key="centroids",
            )

        if self.data_config.augmentation_config.use_augmentations:
            datapipe = KorniaAugmenter(
                datapipe,
                **dict(self.data_config.augmentation_config.augmentations.geometric),
                image_key="image",
                instance_key="centroids",
            )

        datapipe = MultiConfidenceMapGenerator(
            datapipe,
            sigma=self.data_config.preprocessing.conf_map_gen.sigma,
            output_stride=self.data_config.preprocessing.conf_map_gen.output_stride,
            centroids=True,
        )
        datapipe = KeyFilter(
            datapipe,
            keep_keys=[
                "image",
                "instances",
                "video_idx",
                "frame_idx",
                "centroids",
                "centroids_confidence_maps",
                "orig_size",
                "num_instances",
            ],
        )

        return datapipe
