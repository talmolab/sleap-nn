"""This module defines high level pipeline configurations from providers/transformers.

This allows for convenient ways to configure individual variants of common pipelines, as
well as to define training vs inference versions based on the same configurations.
"""

from typing import Optional, Tuple
from omegaconf.omegaconf import DictConfig
import torch
from torch.utils.data.datapipes.datapipe import IterDataPipe
from sleap_nn.data.augmentation import KorniaAugmenter
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.instance_cropping import InstanceCropper
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.resizing import Resizer, PadToStride, SizeMatcher
from sleap_nn.data.edge_maps import PartAffinityFieldsGenerator
from sleap_nn.data.confidence_maps import (
    ConfidenceMapGenerator,
    MultiConfidenceMapGenerator,
)
from sleap_nn.data.general import KeyFilter


class TopdownConfmapsPipeline:
    """Pipeline builder for instance-centered confidence map models.

    Attributes:
        data_config: Data-related configuration.
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        confmap_head: DictConfig object with all the keys in the `head_config` section.
        (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
        crop_hw: Height and width of the crop in pixels.

    Note: If scale is provided for centered-instance model, the images are cropped out
    of original image according to given crop height and width and then the cropped
    images are scaled.
    """

    def __init__(
        self,
        data_config: DictConfig,
        max_stride: int,
        confmap_head: DictConfig,
        crop_hw: Tuple[int],
    ) -> None:
        """Initialize the data config."""
        self.data_config = data_config
        self.max_stride = max_stride
        self.confmap_head = confmap_head
        self.crop_hw = crop_hw

    def make_training_pipeline(
        self, data_provider: IterDataPipe, use_augmentations: bool = False
    ) -> IterDataPipe:
        """Create training pipeline with input data only.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.
            use_augmentations: `True` if augmentations should be applied to the training
                pipeline, else `False`. Default: `False`.

        Returns:
            An `IterDataPipe` instance configured to produce input examples.
        """
        provider = data_provider
        datapipe = Normalizer(provider, self.data_config.preprocessing.is_rgb)
        datapipe = SizeMatcher(
            datapipe,
            max_height=self.data_config.preprocessing.max_height,
            max_width=self.data_config.preprocessing.max_width,
            provider=provider,
        )

        if use_augmentations and "intensity" in self.data_config.augmentation_config:
            datapipe = KorniaAugmenter(
                datapipe,
                **dict(self.data_config.augmentation_config.intensity),
                image_key="image",
                instance_key="instances",
            )

        datapipe = InstanceCentroidFinder(
            datapipe, anchor_ind=self.confmap_head.anchor_part
        )

        datapipe = InstanceCropper(datapipe, self.crop_hw)

        if use_augmentations and "geometric" in self.data_config.augmentation_config:
            datapipe = KorniaAugmenter(
                datapipe,
                **dict(self.data_config.augmentation_config.geometric),
                image_key="instance_image",
                instance_key="instance",
            )

        datapipe = Resizer(
            datapipe,
            scale=self.data_config.preprocessing.scale,
            image_key="instance_image",
            instances_key="instance",
        )
        datapipe = PadToStride(
            datapipe, max_stride=self.max_stride, image_key="instance_image"
        )

        datapipe = ConfidenceMapGenerator(
            datapipe,
            sigma=self.confmap_head.sigma,
            output_stride=self.confmap_head.output_stride,
            image_key="instance_image",
            instance_key="instance",
        )
        datapipe = KeyFilter(
            datapipe,
            keep_keys=[
                "video_idx",
                "frame_idx",
                "centroid",
                "instance",
                "instance_bbox",
                "instance_image",
                "confidence_maps",
                "num_instances",
                "orig_size",
            ],
        )

        return datapipe


class SingleInstanceConfmapsPipeline:
    """Pipeline builder for single-instance confidence map models.

    Attributes:
        data_config: Data-related configuration.
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        confmap_head: DictConfig object with all the keys in the `head_config` section.
        (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
    """

    def __init__(
        self, data_config: DictConfig, max_stride: int, confmap_head: DictConfig
    ) -> None:
        """Initialize the data config."""
        self.data_config = data_config
        self.max_stride = max_stride
        self.confmap_head = confmap_head

    def make_training_pipeline(
        self, data_provider: IterDataPipe, use_augmentations: bool = False
    ) -> IterDataPipe:
        """Create training pipeline with input data only.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.
            use_augmentations: `True` if augmentations should be applied to the training
                pipeline, else `False`. Default: `False`.

        Returns:
            An `IterDataPipe` instance configured to produce input examples.
        """
        provider = data_provider
        datapipe = Normalizer(provider, self.data_config.preprocessing.is_rgb)
        datapipe = SizeMatcher(
            datapipe,
            max_height=self.data_config.preprocessing.max_height,
            max_width=self.data_config.preprocessing.max_width,
            provider=provider,
        )

        if use_augmentations:
            if "intensity" in self.data_config.augmentation_config:
                datapipe = KorniaAugmenter(
                    datapipe,
                    **dict(self.data_config.augmentation_config.intensity),
                    image_key="image",
                    instance_key="instances",
                )
            if "geometric" in self.data_config.augmentation_config:
                datapipe = KorniaAugmenter(
                    datapipe,
                    **dict(self.data_config.augmentation_config.geometric),
                    image_key="image",
                    instance_key="instances",
                )

            if "random_crop" in self.data_config.augmentation_config:
                datapipe = KorniaAugmenter(
                    datapipe,
                    random_crop_height=self.data_config.augmentation_config.random_crop.crop_height,
                    random_crop_width=self.data_config.augmentation_config.random_crop.crop_width,
                    random_crop_p=self.data_config.augmentation_config.random_crop.random_crop_p,
                    image_key="image",
                    instance_key="instances",
                )

        datapipe = Resizer(datapipe, scale=self.data_config.preprocessing.scale)
        datapipe = PadToStride(datapipe, max_stride=self.max_stride)

        datapipe = ConfidenceMapGenerator(
            datapipe,
            sigma=self.confmap_head.sigma,
            output_stride=self.confmap_head.output_stride,
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
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        confmap_head: DictConfig object with all the keys in the `head_config` section.
        (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
    """

    def __init__(
        self, data_config: DictConfig, max_stride: int, confmap_head: DictConfig
    ) -> None:
        """Initialize the data config."""
        self.data_config = data_config
        self.max_stride = max_stride
        self.confmap_head = confmap_head

    def make_training_pipeline(
        self, data_provider: IterDataPipe, use_augmentations: bool = False
    ) -> IterDataPipe:
        """Create training pipeline with input data only.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.
            use_augmentations: `True` if augmentations should be applied to the training
                pipeline, else `False`. Default: `False`.

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
        ]
        datapipe = Normalizer(provider, self.data_config.preprocessing.is_rgb)
        datapipe = SizeMatcher(
            datapipe,
            max_height=self.data_config.preprocessing.max_height,
            max_width=self.data_config.preprocessing.max_width,
            provider=provider,
        )

        if use_augmentations:
            if "intensity" in self.data_config.augmentation_config:
                datapipe = KorniaAugmenter(
                    datapipe,
                    **dict(self.data_config.augmentation_config.intensity),
                    image_key="image",
                    instance_key="instances",
                )
            if "geometric" in self.data_config.augmentation_config:
                datapipe = KorniaAugmenter(
                    datapipe,
                    **dict(self.data_config.augmentation_config.geometric),
                    image_key="image",
                    instance_key="instances",
                )
            if "random_crop" in self.data_config.augmentation_config:
                datapipe = KorniaAugmenter(
                    datapipe,
                    random_crop_height=self.data_config.augmentation_config.random_crop.crop_height,
                    random_crop_width=self.data_config.augmentation_config.random_crop.crop_width,
                    random_crop_p=self.data_config.augmentation_config.random_crop.random_crop_p,
                    image_key="image",
                    instance_key="instances",
                )

        datapipe = Resizer(datapipe, scale=self.data_config.preprocessing.scale)
        datapipe = PadToStride(datapipe, max_stride=self.max_stride)
        datapipe = InstanceCentroidFinder(
            datapipe, anchor_ind=self.confmap_head.anchor_part
        )

        datapipe = MultiConfidenceMapGenerator(
            datapipe,
            sigma=self.confmap_head.sigma,
            output_stride=self.confmap_head.output_stride,
            centroids=True,
        )

        datapipe = KeyFilter(datapipe, keep_keys=keep_keys)

        return datapipe


class BottomUpPipeline:
    """Pipeline builder for (Bottom-up) confidence maps + part affinity fields models.

    Attributes:
        data_config: Data-related configuration.
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        confmap_head: DictConfig object with all the keys in the `head_config` section.
        (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
        pafs_head: DictConfig object with all the keys in the `head_config` section
        (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type )
        for PAFs.
    """

    def __init__(
        self,
        data_config: DictConfig,
        max_stride: int,
        confmap_head: DictConfig,
        pafs_head: DictConfig,
    ) -> None:
        """Initialize the data config."""
        self.data_config = data_config
        self.max_stride = max_stride
        self.confmap_head = confmap_head
        self.pafs_head = pafs_head

    def make_training_pipeline(
        self, data_provider: IterDataPipe, use_augmentations: bool = False
    ) -> IterDataPipe:
        """Create training pipeline with input data only.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.
            use_augmentations: `True` if augmentations should be applied to the training
                pipeline, else `False`. Default: `False`.

        Returns:
            An `IterDataPipe` instance configured to produce input examples.
        """
        provider = data_provider
        keep_keys = [
            "image",
            "video_idx",
            "frame_idx",
            "confidence_maps",
            "orig_size",
            "num_instances",
            "part_affinity_fields",
        ]
        datapipe = Normalizer(provider, self.data_config.preprocessing.is_rgb)
        datapipe = SizeMatcher(
            datapipe,
            max_height=self.data_config.preprocessing.max_height,
            max_width=self.data_config.preprocessing.max_width,
            provider=provider,
        )

        if use_augmentations:
            if "intensity" in self.data_config.augmentation_config:
                datapipe = KorniaAugmenter(
                    datapipe,
                    **dict(self.data_config.augmentation_config.intensity),
                    image_key="image",
                    instance_key="instances",
                )
            if "geometric" in self.data_config.augmentation_config:
                datapipe = KorniaAugmenter(
                    datapipe,
                    **dict(self.data_config.augmentation_config.geometric),
                    image_key="image",
                    instance_key="instances",
                )

            if "random_crop" in self.data_config.augmentation_config:
                datapipe = KorniaAugmenter(
                    datapipe,
                    random_crop_height=self.data_config.augmentation_config.random_crop.crop_height,
                    random_crop_width=self.data_config.augmentation_config.random_crop.crop_width,
                    random_crop_p=self.data_config.augmentation_config.random_crop.random_crop_p,
                    image_key="image",
                    instance_key="instances",
                )

        datapipe = Resizer(datapipe, scale=self.data_config.preprocessing.scale)
        datapipe = PadToStride(datapipe, max_stride=self.max_stride)

        datapipe = MultiConfidenceMapGenerator(
            datapipe,
            sigma=self.confmap_head.sigma,
            output_stride=self.confmap_head.output_stride,
            centroids=False,
        )

        datapipe = PartAffinityFieldsGenerator(
            datapipe,
            sigma=self.pafs_head.sigma,
            output_stride=self.pafs_head.output_stride,
            edge_inds=torch.Tensor(provider.edge_inds),
            flatten_channels=True,
        )

        datapipe = KeyFilter(datapipe, keep_keys=keep_keys)

        return datapipe
