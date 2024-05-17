"""This module defines high level pipeline configurations from providers/transformers.

This allows for convenient ways to configure individual variants of common pipelines, as
well as to define training vs inference versions based on the same configurations.
"""

from omegaconf.omegaconf import DictConfig
import torch
from sleap_nn.data.augmentation import KorniaAugmenter
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.instance_cropping import InstanceCropper
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.resizing import Resizer, PadToStride, SizeMatcher
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
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.

    Note: If scale is provided for centered-instance model, the images are cropped out
    of original image according to given crop height and width and then the cropped
    images are scaled.
    """

    def __init__(self, data_config: DictConfig, max_stride: int) -> None:
        """Initialize the data config."""
        self.data_config = data_config
        self.max_stride = max_stride

    def make_training_pipeline(self, data_provider: IterDataPipe) -> IterDataPipe:
        """Create training pipeline with input data only.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.

        Returns:
            An `IterDataPipe` instance configured to produce input examples.
        """
        provider = data_provider
        datapipe = Normalizer(provider, self.data_config.is_rgb)
        datapipe = SizeMatcher(
            datapipe,
            max_height=self.data_config.max_height,
            max_width=self.data_config.max_width,
            provider=provider,
        )

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

        datapipe = InstanceCropper(
            datapipe,
            self.data_config.preprocessing.crop_hw,
        )

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

        datapipe = Resizer(
            datapipe,
            scale=self.data_config.scale,
            image_key="instance_image",
            instances_key="instance",
        )
        datapipe = PadToStride(
            datapipe, max_stride=self.max_stride, image_key="instance_image"
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
                "orig_size",
                "scale",
            ],
        )

        return datapipe


class SingleInstanceConfmapsPipeline:
    """Pipeline builder for single-instance confidence map models.

    Attributes:
        data_config: Data-related configuration.
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
    """

    def __init__(self, data_config: DictConfig, max_stride: int) -> None:
        """Initialize the data config."""
        self.data_config = data_config
        self.max_stride = max_stride

    def make_training_pipeline(self, data_provider: IterDataPipe) -> IterDataPipe:
        """Create training pipeline with input data only.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.

        Returns:
            An `IterDataPipe` instance configured to produce input examples.
        """
        provider = data_provider
        datapipe = Normalizer(provider, self.data_config.is_rgb)
        datapipe = SizeMatcher(
            datapipe,
            max_height=self.data_config.max_height,
            max_width=self.data_config.max_width,
            provider=provider,
        )

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

        datapipe = Resizer(datapipe, scale=self.data_config.scale)
        datapipe = PadToStride(datapipe, max_stride=self.max_stride)

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
                "scale",
            ],
        )

        return datapipe


class CentroidConfmapsPipeline:
    """Pipeline builder for centroid confidence map models.

    Attributes:
        data_config: Data-related configuration.
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
    """

    def __init__(self, data_config: DictConfig, max_stride: int) -> None:
        """Initialize the data config."""
        self.data_config = data_config
        self.max_stride = max_stride

    def make_training_pipeline(self, data_provider: IterDataPipe) -> IterDataPipe:
        """Create training pipeline with input data only.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.

        Returns:
            An `IterDataPipe` instance configured to produce input examples.
        """
        provider = data_provider
        keep_keys = [
            "image",
            "video_idx",
            "frame_idx",
            "centroids_confidence_maps",
            "orig_size",
            "num_instances",
            "scale",
        ]
        datapipe = Normalizer(provider, self.data_config.is_rgb)
        datapipe = SizeMatcher(
            datapipe,
            max_height=self.data_config.max_height,
            max_width=self.data_config.max_width,
            provider=provider,
        )

        if self.data_config.augmentation_config.use_augmentations:
            datapipe = KorniaAugmenter(
                datapipe,
                **dict(self.data_config.augmentation_config.augmentations.intensity),
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

        if self.data_config.augmentation_config.use_augmentations:
            datapipe = KorniaAugmenter(
                datapipe,
                **dict(self.data_config.augmentation_config.augmentations.geometric),
                image_key="image",
                instance_key="instances",
            )

        datapipe = Resizer(datapipe, scale=self.data_config.scale)
        datapipe = PadToStride(datapipe, max_stride=self.max_stride)
        datapipe = InstanceCentroidFinder(
            datapipe, anchor_ind=self.data_config.preprocessing.anchor_ind
        )

        datapipe = MultiConfidenceMapGenerator(
            datapipe,
            sigma=self.data_config.preprocessing.conf_map_gen.sigma,
            output_stride=self.data_config.preprocessing.conf_map_gen.output_stride,
            centroids=True,
        )

        datapipe = KeyFilter(datapipe, keep_keys=keep_keys)

        return datapipe
