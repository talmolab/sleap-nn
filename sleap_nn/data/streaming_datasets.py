"""Custom `litdata.StreamingDataset`s for different models."""

from kornia.geometry.transform import crop_and_resize
from omegaconf import DictConfig
from typing import List, Optional, Tuple
import litdata as ld
import torch
import torchvision.transforms as T
from sleap_nn.data.augmentation import (
    apply_geometric_augmentation,
    apply_intensity_augmentation,
)
from sleap_nn.data.confidence_maps import generate_confmaps, generate_multiconfmaps
from sleap_nn.data.edge_maps import generate_pafs
from sleap_nn.data.instance_cropping import make_centered_bboxes
from sleap_nn.data.normalization import apply_normalization
from sleap_nn.data.resizing import apply_pad_to_stride, apply_resizer


class BottomUpStreamingDataset(ld.StreamingDataset):
    """StreamingDataset for BottomUp pipeline.

    The `__getitem__()` applies augmentation, resizes and pads image (if needed for the
    given max_stride), and generates confidence maps and part affinity fields for every
    data sample stored in `.bin` files.

    Args:
        confmap_head: DictConfig object with all the keys in the `head_config` section.
            (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
        pafs_head: DictConfig object with all the keys in the `head_config` section
            (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type )
            for PAFs.
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        apply_aug: `True` if augmentations should be applied to the data pipeline,
            else `False`. Default: `False`.
        augmentation_config: Augmentation parameters. (`data_config.preprocessing.augmentation_config`
            section in the config file.)
    """

    def __init__(
        self,
        confmap_head: DictConfig,
        pafs_head: DictConfig,
        edge_inds: list,
        max_stride: int,
        apply_aug: bool = False,
        augmentation_config: DictConfig = None,
        *args,
        **kwargs,
    ):
        """Constructs a BottomUpStreamingDataset."""
        super().__init__(*args, **kwargs)

        self.confmap_head = confmap_head
        self.pafs_head = pafs_head
        self.edge_inds = edge_inds
        self.max_stride = max_stride
        self.apply_aug = apply_aug
        self.aug_config = augmentation_config

    def __getitem__(self, index):
        """Apply augmentation and generate confidence maps."""
        ex = super().__getitem__(index)
        transform = T.PILToTensor()
        ex["image"] = transform(ex["image"])
        ex["image"] = ex["image"].unsqueeze(dim=0).to(torch.float32)

        # Augmentation
        if self.apply_aug:
            if "intensity" in self.aug_config:
                ex["image"], ex["instances"] = apply_intensity_augmentation(
                    ex["image"], ex["instances"], **self.aug_config.intensity
                )

            if "geometric" in self.aug_config:
                ex["image"], ex["instances"] = apply_geometric_augmentation(
                    ex["image"], ex["instances"], **self.aug_config.geometric
                )

        ex["image"] = apply_normalization(ex["image"])

        # Pad the image (if needed) according max stride
        ex["image"] = apply_pad_to_stride(ex["image"], max_stride=self.max_stride)

        img_hw = ex["image"].shape[-2:]

        # Generate confidence maps
        confidence_maps = generate_multiconfmaps(
            ex["instances"],
            img_hw=img_hw,
            num_instances=ex["num_instances"],
            sigma=self.confmap_head.sigma,
            output_stride=self.confmap_head.output_stride,
            is_centroids=False,
        )

        # pafs
        pafs = generate_pafs(
            ex["instances"],
            img_hw=img_hw,
            sigma=self.pafs_head.sigma,
            output_stride=self.pafs_head.output_stride,
            edge_inds=torch.Tensor(self.edge_inds),
            flatten_channels=True,
        )

        ex["confidence_maps"] = confidence_maps
        ex["part_affinity_fields"] = pafs

        return ex


class CenteredInstanceStreamingDataset(ld.StreamingDataset):
    """StreamingDataset for CeneteredInstance pipeline.

    The `__getitem__()` applies augmentation, re-crops `instance_image` to original crop size,
    resizes and pads image (if needed for the given max_stride), and generates confidence maps
    for every data sample stored in `.bin` files.

    Args:
        confmap_head: DictConfig object with all the keys in the `head_config` section.
            (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
        crop_hw: Height and width of the crop in pixels.
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        apply_aug: `True` if augmentations should be applied to the data pipeline,
            else `False`. Default: `False`.
        augmentation_config: Augmentation parameters. (`data_config.preprocessing.augmentation_config`
            section in the config file.)
        input_scale: Resize factor applied to the image. Default: 1.0.
    """

    def __init__(
        self,
        confmap_head: DictConfig,
        crop_hw: Tuple[int],
        max_stride: int,
        apply_aug: bool = False,
        augmentation_config: DictConfig = None,
        input_scale: float = 1.0,
        *args,
        **kwargs,
    ):
        """Construct a CenteredInstanceStreamingDataset."""
        super().__init__(*args, **kwargs)

        self.confmap_head = confmap_head
        self.crop_hw = crop_hw
        self.max_stride = max_stride
        self.apply_aug = apply_aug
        self.aug_config = augmentation_config
        self.input_scale = input_scale
        # Re-crop to original crop size
        self.crop_hw = [int(x * self.input_scale) for x in self.crop_hw]

    def __getitem__(self, index):
        """Apply augmentation and generate confidence maps."""
        ex = super().__getitem__(index)
        transform = T.PILToTensor()
        ex["instance_image"] = transform(ex["instance_image"])
        ex["instance_image"] = ex["instance_image"].unsqueeze(dim=0).to(torch.float32)
        # Augmentation
        if self.apply_aug:
            if "intensity" in self.aug_config:
                ex["instance_image"], ex["instance"] = apply_intensity_augmentation(
                    ex["instance_image"], ex["instance"], **self.aug_config.intensity
                )

            if "geometric" in self.aug_config:
                ex["instance_image"], ex["instance"] = apply_geometric_augmentation(
                    ex["instance_image"], ex["instance"], **self.aug_config.geometric
                )
                                                                                                                          
        ex["instance_bbox"] = torch.unsqueeze(
            make_centered_bboxes(ex["centroid"][0], self.crop_hw[0], self.crop_hw[1]), 0
        )
        ex["instance_image"] = crop_and_resize(
            ex["instance_image"], boxes=ex["instance_bbox"], size=self.crop_hw
        )
        point = ex["instance_bbox"][0][0]
        center_instance = ex["instance"] - point
        centered_centroid = ex["centroid"] - point

        ex["instance"] = center_instance.unsqueeze(0)  # (n_samples=1, n_nodes, 2)
        ex["centroid"] = centered_centroid.unsqueeze(0)  # (n_samples=1, 2)

        ex["instance_image"] = apply_normalization(ex["instance_image"])

        # Pad the image (if needed) according max stride
        ex["instance_image"] = apply_pad_to_stride(
            ex["instance_image"], max_stride=self.max_stride
        )

        img_hw = ex["instance_image"].shape[-2:]

        # Generate confidence maps
        confidence_maps = generate_confmaps(
            ex["instance"],
            img_hw=img_hw,
            sigma=self.confmap_head.sigma,
            output_stride=self.confmap_head.output_stride,
        )

        ex["confidence_maps"] = confidence_maps

        return ex


class CentroidStreamingDataset(ld.StreamingDataset):
    """StreamingDataset for Centroid pipeline.

    The `__getitem__()` applies augmentation, resizes and pads image (if needed for the
    given max_stride), and generates confidence maps for every data sample stored in `.bin` files.

    Args:
        confmap_head: DictConfig object with all the keys in the `head_config` section.
            (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        apply_aug: `True` if augmentations should be applied to the data pipeline,
            else `False`. Default: `False`.
        augmentation_config: Augmentation parameters. (`data_config.preprocessing.augmentation_config`
            section in the config file.)
    """

    def __init__(
        self,
        confmap_head: DictConfig,
        max_stride: int,
        apply_aug: bool = False,
        augmentation_config: DictConfig = None,
        *args,
        **kwargs,
    ):
        """Construct a CentroidStreamingDataset."""
        super().__init__(*args, **kwargs)

        self.confmap_head = confmap_head
        self.max_stride = max_stride
        self.apply_aug = apply_aug
        self.aug_config = augmentation_config

    def __getitem__(self, index):
        """Apply augmentation and generate confidence maps."""
        ex = super().__getitem__(index)
        transform = T.PILToTensor()
        ex["image"] = transform(ex["image"])
        ex["image"] = ex["image"].unsqueeze(dim=0).to(torch.float32)

        # Augmentation
        if self.apply_aug:
            if "intensity" in self.aug_config:
                ex["image"], ex["centroids"] = apply_intensity_augmentation(
                    ex["image"], ex["centroids"], **self.aug_config.intensity
                )

            if "geometric" in self.aug_config:
                ex["image"], ex["centroids"] = apply_geometric_augmentation(
                    ex["image"], ex["centroids"], **self.aug_config.geometric
                )

        ex["image"] = apply_normalization(ex["image"])

        # Pad the image (if needed) according max stride
        ex["image"] = apply_pad_to_stride(ex["image"], max_stride=self.max_stride)

        img_hw = ex["image"].shape[-2:]

        # Generate confidence maps
        confidence_maps = generate_multiconfmaps(
            ex["centroids"],
            img_hw=img_hw,
            num_instances=ex["num_instances"],
            sigma=self.confmap_head.sigma,
            output_stride=self.confmap_head.output_stride,
            is_centroids=True,
        )

        ex["centroids_confidence_maps"] = confidence_maps

        return ex


class SingleInstanceStreamingDataset(ld.StreamingDataset):
    """StreamingDataset for SingleInstance pipeline.

    The `__getitem__()` applies augmentation, resizes and pads image (if needed for the
    given max_stride), and generates confidence maps for every data sample stored in `.bin` files.

    Args:
        confmap_head: DictConfig object with all the keys in the `head_config` section.
            (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        apply_aug: `True` if augmentations should be applied to the data pipeline,
            else `False`. Default: `False`.
        augmentation_config: Augmentation parameters. (`data_config.preprocessing.augmentation_config`
            section in the config file.)
    """

    def __init__(
        self,
        confmap_head: DictConfig,
        max_stride: int,
        apply_aug: bool = False,
        augmentation_config: DictConfig = None,
        *args,
        **kwargs,
    ):
        """Construct a SingleInstanceStreamingDataset."""
        super().__init__(*args, **kwargs)

        self.confmap_head = confmap_head
        self.max_stride = max_stride
        self.apply_aug = apply_aug
        self.aug_config = augmentation_config

    def __getitem__(self, index):
        """Apply augmentation and generate confidence maps."""
        ex = super().__getitem__(index)
        transform = T.PILToTensor()
        ex["image"] = transform(ex["image"])
        ex["image"] = ex["image"].unsqueeze(dim=0).to(torch.float32)

        # Augmentation
        if self.apply_aug:
            if "intensity" in self.aug_config:
                ex["image"], ex["instances"] = apply_intensity_augmentation(
                    ex["image"], ex["instances"], **self.aug_config.intensity
                )

            if "geometric" in self.aug_config:
                ex["image"], ex["instances"] = apply_geometric_augmentation(
                    ex["image"], ex["instances"], **self.aug_config.geometric
                )

        ex["image"] = apply_normalization(ex["image"])

        # Pad the image (if needed) according max stride
        ex["image"] = apply_pad_to_stride(ex["image"], max_stride=self.max_stride)

        img_hw = ex["image"].shape[-2:]

        # Generate confidence maps
        confidence_maps = generate_confmaps(
            ex["instances"],
            img_hw=img_hw,
            sigma=self.confmap_head.sigma,
            output_stride=self.confmap_head.output_stride,
        )

        ex["confidence_maps"] = confidence_maps

        return ex
