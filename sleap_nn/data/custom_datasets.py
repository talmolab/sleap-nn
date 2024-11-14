"""Custom `torch.utils.data.Dataset`s for different model types."""

from typing import Dict, Iterator, List, Optional, Tuple
from omegaconf import DictConfig
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import sleap_io as sio
from sleap_nn.data.instance_centroids import generate_centroids
from sleap_nn.data.instance_cropping import generate_crops
from sleap_nn.data.normalization import (
    apply_normalization,
    convert_to_grayscale,
    convert_to_rgb,
)
from sleap_nn.data.providers import get_max_instances, get_max_height_width, process_lf
from sleap_nn.data.resizing import apply_sizematcher, apply_resizer
from sleap_nn.data.augmentation import (
    apply_geometric_augmentation,
    apply_intensity_augmentation,
)
from sleap_nn.data.confidence_maps import generate_confmaps, generate_multiconfmaps
from sleap_nn.data.edge_maps import generate_pafs
from sleap_nn.data.normalization import apply_normalization
from sleap_nn.data.resizing import apply_pad_to_stride, apply_resizer


class BaseDataset(Dataset):
    """Base class for custom torch Datasets.

    Attributes:
        labels: Source `sio.Labels` object.
        data_config: Data-related configuration. (`data_config` section in the config file).
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        apply_aug: `True` if augmentations should be applied to the data pipeline,
            else `False`. Default: `False`.
        max_hw: Maximum height and width of images across the labels file. If `max_height` and
           `max_width` in the config is None, then `max_hw` is used (computed with
            `sleap_nn.data.providers.get_max_height_width`). Else the values in the config
            are used.
    """

    def __init__(
        self,
        labels: sio.Labels,
        data_config: DictConfig,
        max_stride: int,
        apply_aug: bool = False,
        max_hw: Tuple[Optional[int]] = (None, None),
    ) -> None:
        """Initialize class attributes."""
        super().__init__()
        self.labels = labels
        self.data_config = data_config
        self.max_stride = max_stride
        self.apply_aug = apply_aug
        self.max_hw = max_hw
        self.max_instances = get_max_instances(self.labels)

    def _get_video_idx(self, lf):
        """Return indsample of `lf.video` in `labels.videos`."""
        return self.labels.videos.index(lf.video)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, index) -> Dict:
        """Returns the sample dict for given index."""
        pass


class BottomUpDataset(BaseDataset):
    """Dataset class for bottom-up models.

    Attributes:
        labels: Source `sio.Labels` object.
        data_config: Data-related configuration. (`data_config` section in the config file).
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        apply_aug: `True` if augmentations should be applied to the data pipeline,
            else `False`. Default: `False`.
        max_hw: Maximum height and width of images across the labels file. If `max_height` and
           `max_width` in the config is None, then `max_hw` is used (computed with
            `sleap_nn.data.providers.get_max_height_width`). Else the values in the config
            are used.
        confmap_head_config: DictConfig object with all the keys in the `head_config` section.
            (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
        pafs_head_config: DictConfig object with all the keys in the `head_config` section
            (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type )
            for PAFs.
    """

    def __init__(
        self,
        labels: sio.Labels,
        data_config: DictConfig,
        confmap_head_config: DictConfig,
        pafs_head_config: DictConfig,
        max_stride: int,
        apply_aug: bool = False,
        max_hw: Tuple[Optional[int]] = (None, None),
    ) -> None:
        """Initialize class attributes."""
        super().__init__(
            labels=labels,
            data_config=data_config,
            max_stride=max_stride,
            apply_aug=apply_aug,
            max_hw=max_hw,
        )
        self.confmap_head_config = confmap_head_config
        self.pafs_head_config = pafs_head_config

        self.edge_inds = self.labels.skeletons[0].edge_inds

    def __len__(self) -> int:
        """Return number of `LabeledFrames` in the labels object."""
        return len(self.labels)

    def __getitem__(self, index) -> Dict:
        """Return dict with image, confmaps and pafs for given index."""
        lf = self.labels[index]
        video_idx = self._get_video_idx(lf)

        # get dict
        sample = process_lf(
            lf,
            video_idx=video_idx,
            max_instances=self.max_instances,
            user_instances_only=self.data_config.user_instances_only,
        )

        # apply normalization
        sample["image"] = apply_normalization(sample["image"])

        if self.data_config.preprocessing.is_rgb:
            sample["image"] = convert_to_rgb(sample["image"])
        else:
            sample["image"] = convert_to_grayscale(sample["image"])

        # size matcher
        sample["image"], eff_scale = apply_sizematcher(
            sample["image"],
            max_height=self.max_hw[0],
            max_width=self.max_hw[1],
        )
        sample["instances"] = sample["instances"] * eff_scale

        # apply augmentation
        if self.apply_aug:
            if "intensity" in self.data_config.augmentation_config:
                sample["image"], sample["instances"] = apply_intensity_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.data_config.augmentation_config.intensity,
                )

            if "geometric" in self.data_config.augmentation_config:
                sample["image"], sample["instances"] = apply_geometric_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.data_config.augmentation_config.geometric,
                )

        # resize image
        sample["image"], sample["instances"] = apply_resizer(
            sample["image"],
            sample["instances"],
            scale=self.data_config.preprocessing.scale,
        )

        # Pad the image (if needed) according max stride
        sample["image"] = apply_pad_to_stride(
            sample["image"], max_stride=self.max_stride
        )

        img_hw = sample["image"].shape[-2:]

        # Generate confidence maps
        confidence_maps = generate_multiconfmaps(
            sample["instances"],
            img_hw=img_hw,
            num_instances=sample["num_instances"],
            sigma=self.confmap_head_config.sigma,
            output_stride=self.confmap_head_config.output_stride,
            is_centroids=False,
        )

        # pafs
        pafs = generate_pafs(
            sample["instances"],
            img_hw=img_hw,
            sigma=self.pafs_head_config.sigma,
            output_stride=self.pafs_head_config.output_stride,
            edge_inds=torch.Tensor(self.edge_inds),
            flatten_channels=True,
        )

        sample["confidence_maps"] = confidence_maps
        sample["part_affinity_fields"] = pafs

        return sample


class CenteredInstanceDataset(BaseDataset):
    """Dataset class for instance-centered confidence map models.

    Attributes:
        labels: Source `sio.Labels` object.
        data_config: Data-related configuration. (`data_config` section in the config file).
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        apply_aug: `True` if augmentations should be applied to the data pipeline,
            else `False`. Default: `False`.
        max_hw: Maximum height and width of images across the labels file. If `max_height` and
           `max_width` in the config is None, then `max_hw` is used (computed with
            `sleap_nn.data.providers.get_max_height_width`). Else the values in the config
            are used.
        confmap_head_config: DictConfig object with all the keys in the `head_config` section.
        (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
        crop_hw: Height and width of the crop in pixels.

    Note: If scale is provided for centered-instance model, the images are cropped out
    of original image according to given crop height and width and then the cropped
    images are scaled.
    """

    def __init__(
        self,
        labels: sio.Labels,
        data_config: DictConfig,
        crop_hw: Tuple[int],
        confmap_head_config: DictConfig,
        max_stride: int,
        apply_aug: bool = False,
        max_hw: Tuple[Optional[int]] = (None, None),
    ) -> None:
        """Initialize class attributes."""
        super().__init__(
            labels=labels,
            data_config=data_config,
            max_stride=max_stride,
            apply_aug=apply_aug,
            max_hw=max_hw,
        )
        self.crop_hw = crop_hw
        self.confmap_head_config = confmap_head_config
        self.instance_idx_list = self._get_instance_idx_list()

    def _get_instance_idx_list(self) -> List[Tuple[int]]:
        """Return list of tuples with indices of labelled frames and instances."""
        instance_idx_list = []
        for lf_idx, lf in enumerate(self.labels):
            for inst_idx, _ in enumerate(lf.instances):
                instance_idx_list.append((lf_idx, inst_idx))
        return instance_idx_list

    def __len__(self) -> int:
        """Return number of instances in the labels object."""
        return len(self.instance_idx_list)

    def __getitem__(self, index) -> Dict:
        """Return dict with cropped image and confmaps of instance for given index."""
        lf_idx, inst_idx = self.instance_idx_list[index]

        lf = self.labels[lf_idx]
        video_idx = self._get_video_idx(lf)

        # get dict
        sample = process_lf(
            lf,
            video_idx=video_idx,
            max_instances=None,
            user_instances_only=self.data_config.user_instances_only,
        )

        sample["instances"] = sample["instances"][:, inst_idx]

        # apply normalization
        sample["image"] = apply_normalization(
            sample["image"]
        )  # TODO: get img with cache

        if self.data_config.preprocessing.is_rgb:
            sample["image"] = convert_to_rgb(sample["image"])
        else:
            sample["image"] = convert_to_grayscale(sample["image"])

        # size matcher
        sample["image"], eff_scale = apply_sizematcher(
            sample["image"],
            max_height=self.max_hw[0],
            max_width=self.max_hw[1],
        )
        sample["instances"] = sample["instances"] * eff_scale

        # apply augmentation
        if self.apply_aug:
            if "intensity" in self.data_config.augmentation_config:
                sample["image"], sample["instances"] = apply_intensity_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.data_config.augmentation_config.intensity,
                )

            if "geometric" in self.data_config.augmentation_config:
                sample["image"], sample["instances"] = apply_geometric_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.data_config.augmentation_config.geometric,
                )

        # get the centroids based on the anchor idx
        centroids = generate_centroids(
            sample["instances"], anchor_ind=self.confmap_head_config.anchor_part
        )

        instances, centroids = sample["instances"][0], centroids[0]  # (n_samples=1)
        instance, centroid = instances[0], centroids[0]  # (num_instances=1, ...)

        res = generate_crops(sample["image"], instance, centroid, self.crop_hw)

        res["frame_idx"] = sample["frame_idx"]
        res["video_idx"] = sample["video_idx"]
        res["num_instances"] = sample["num_instances"]
        res["orig_size"] = sample["orig_size"]

        # resize image
        res["instance_image"], res["instance"] = apply_resizer(
            res["instance_image"],
            res["instance"],
            scale=self.data_config.preprocessing.scale,
        )

        # Pad the image (if needed) according max stride
        res["instance_image"] = apply_pad_to_stride(
            res["instance_image"], max_stride=self.max_stride
        )

        img_hw = res["instance_image"].shape[-2:]

        # Generate confidence maps
        confidence_maps = generate_confmaps(
            res["instance"],
            img_hw=img_hw,
            sigma=self.confmap_head_config.sigma,
            output_stride=self.confmap_head_config.output_stride,
        )

        res["confidence_maps"] = confidence_maps

        return res


class CentroidDataset(BaseDataset):
    """Dataset class for centroid models.

    Attributes:
        labels: Source `sio.Labels` object.
        data_config: Data-related configuration. (`data_config` section in the config file).
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        apply_aug: `True` if augmentations should be applied to the data pipeline,
            else `False`. Default: `False`.
        max_hw: Maximum height and width of images across the labels file. If `max_height` and
           `max_width` in the config is None, then `max_hw` is used (computed with
            `sleap_nn.data.providers.get_max_height_width`). Else the values in the config
            are used.
        confmap_head_config: DictConfig object with all the keys in the `head_config` section.
        (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
    """

    def __init__(
        self,
        labels: sio.Labels,
        data_config: DictConfig,
        confmap_head_config: DictConfig,
        max_stride: int,
        apply_aug: bool = False,
        max_hw: Tuple[Optional[int]] = (None, None),
    ) -> None:
        """Initialize class attributes."""
        super().__init__(
            labels=labels,
            data_config=data_config,
            max_stride=max_stride,
            apply_aug=apply_aug,
            max_hw=max_hw,
        )
        self.confmap_head_config = confmap_head_config

    def __len__(self) -> int:
        """Return number of `LabeledFrames` in the labels object."""
        return len(self.labels)

    def __getitem__(self, index) -> Dict:
        """Return dict with image and confmaps for centroids for given index."""
        lf = self.labels[index]
        video_idx = self._get_video_idx(lf)

        # get dict
        sample = process_lf(
            lf,
            video_idx=video_idx,
            max_instances=self.max_instances,
            user_instances_only=self.data_config.user_instances_only,
        )

        # apply normalization
        sample["image"] = apply_normalization(sample["image"])

        if self.data_config.preprocessing.is_rgb:
            sample["image"] = convert_to_rgb(sample["image"])
        else:
            sample["image"] = convert_to_grayscale(sample["image"])

        # size matcher
        sample["image"], eff_scale = apply_sizematcher(
            sample["image"],
            max_height=self.max_hw[0],
            max_width=self.max_hw[1],
        )
        sample["instances"] = sample["instances"] * eff_scale

        # apply augmentation
        if self.apply_aug:
            if "intensity" in self.data_config.augmentation_config:
                sample["image"], sample["instances"] = apply_intensity_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.data_config.augmentation_config.intensity,
                )

            if "geometric" in self.data_config.augmentation_config:
                sample["image"], sample["instances"] = apply_geometric_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.data_config.augmentation_config.geometric,
                )

        # get the centroids based on the anchor idx
        centroids = generate_centroids(
            sample["instances"], anchor_ind=self.confmap_head_config.anchor_part
        )

        sample["centroids"] = centroids

        # resize image
        sample["image"], sample["centroids"] = apply_resizer(
            sample["image"],
            sample["centroids"],
            scale=self.data_config.preprocessing.scale,
        )

        # Pad the image (if needed) according max stride
        sample["image"] = apply_pad_to_stride(
            sample["image"], max_stride=self.max_stride
        )

        img_hw = sample["image"].shape[-2:]

        # Generate confidence maps
        confidence_maps = generate_multiconfmaps(
            sample["centroids"],
            img_hw=img_hw,
            num_instances=sample["num_instances"],
            sigma=self.confmap_head_config.sigma,
            output_stride=self.confmap_head_config.output_stride,
            is_centroids=True,
        )

        sample["centroids_confidence_maps"] = confidence_maps

        return sample


class SingleInstanceDataset(BaseDataset):
    """Dataset class for single-instance models.

    Attributes:
        labels: Source `sio.Labels` object.
        data_config: Data-related configuration. (`data_config` section in the config file).
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        apply_aug: `True` if augmentations should be applied to the data pipeline,
            else `False`. Default: `False`.
        max_hw: Maximum height and width of images across the labels file. If `max_height` and
           `max_width` in the config is None, then `max_hw` is used (computed with
            `sleap_nn.data.providers.get_max_height_width`). Else the values in the config
            are used.
        confmap_head_config: DictConfig object with all the keys in the `head_config` section.
        (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
    """

    def __init__(
        self,
        labels: sio.Labels,
        data_config: DictConfig,
        confmap_head_config: DictConfig,
        max_stride: int,
        apply_aug: bool = False,
        max_hw: Tuple[Optional[int]] = (None, None),
    ) -> None:
        """Initialize class attributes."""
        super().__init__(
            labels=labels,
            data_config=data_config,
            max_stride=max_stride,
            apply_aug=apply_aug,
            max_hw=max_hw,
        )
        self.confmap_head_config = confmap_head_config

    def __len__(self) -> int:
        """Return number of `LabeledFrames` in the labels object."""
        return len(self.labels)

    def __getitem__(self, index) -> Dict:
        """Return dict with image and confmaps for instance for given index."""
        lf = self.labels[index]
        video_idx = self._get_video_idx(lf)

        # get dict
        sample = process_lf(
            lf,
            video_idx=video_idx,
            max_instances=1,
            user_instances_only=self.data_config.user_instances_only,
        )

        # apply normalization
        sample["image"] = apply_normalization(sample["image"])

        if self.data_config.preprocessing.is_rgb:
            sample["image"] = convert_to_rgb(sample["image"])
        else:
            sample["image"] = convert_to_grayscale(sample["image"])

        # size matcher
        sample["image"], eff_scale = apply_sizematcher(
            sample["image"],
            max_height=self.max_hw[0],
            max_width=self.max_hw[1],
        )
        sample["instances"] = sample["instances"] * eff_scale

        # apply augmentation
        if self.apply_aug:
            if "intensity" in self.data_config.augmentation_config:
                sample["image"], sample["instances"] = apply_intensity_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.data_config.augmentation_config.intensity,
                )

            if "geometric" in self.data_config.augmentation_config:
                sample["image"], sample["instances"] = apply_geometric_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.data_config.augmentation_config.geometric,
                )

        # resize image
        sample["image"], sample["instances"] = apply_resizer(
            sample["image"],
            sample["instances"],
            scale=self.data_config.preprocessing.scale,
        )

        # Pad the image (if needed) according max stride
        sample["image"] = apply_pad_to_stride(
            sample["image"], max_stride=self.max_stride
        )

        img_hw = sample["image"].shape[-2:]

        # Generate confidence maps
        confidence_maps = generate_confmaps(
            sample["instances"],
            img_hw=img_hw,
            sigma=self.confmap_head_config.sigma,
            output_stride=self.confmap_head_config.output_stride,
        )

        sample["confidence_maps"] = confidence_maps

        return sample
