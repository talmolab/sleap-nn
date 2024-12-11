"""Custom `torch.utils.data.Dataset`s for different model types."""

from kornia.geometry.transform import crop_and_resize
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
from sleap_nn.data.instance_cropping import make_centered_bboxes
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

        self.cache = {}

    def _fill_cache(self):
        """Load all samples to cache."""
        for lf_idx, lf in enumerate(self.labels):
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

            self.cache[lf_idx] = sample.copy()

        for video in self.labels.videos:
            video.close()

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
        self._fill_cache()

    def __getitem__(self, index) -> Dict:
        """Return dict with image, confmaps and pafs for given index."""
        sample = self.cache[index]

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
        self.cache_lf = [None, None]
        self._fill_cache()

    def _fill_cache(self):
        """Load all samples to cache."""
        for idx, (lf_idx, inst_idx) in enumerate(self.instance_idx_list):
            lf = self.labels[lf_idx]
            video_idx = self._get_video_idx(lf)

            # Filter to user instances
            if self.data_config.user_instances_only:
                if lf.user_instances is not None and len(lf.user_instances) > 0:
                    lf.instances = lf.user_instances

            if lf_idx == self.cache_lf[0]:
                img = self.cache_lf[1]
            else:
                img = lf.image
                self.cache_lf = [lf_idx, img]

            image = np.transpose(img, (2, 0, 1))  # HWC -> CHW

            instances = []
            for inst in lf:
                if not inst.is_empty:
                    instances.append(inst.numpy())
            instances = np.stack(instances, axis=0)

            # Add singleton time dimension for single frames.
            image = np.expand_dims(image, axis=0)  # (n_samples=1, C, H, W)
            instances = np.expand_dims(
                instances, axis=0
            )  # (n_samples=1, num_instances, num_nodes, 2)

            instances = torch.from_numpy(instances.astype("float32"))
            image = torch.from_numpy(image)

            num_instances, _ = instances.shape[1:3]
            orig_img_height, orig_img_width = image.shape[-2:]

            instances = instances[:, inst_idx]

            # apply normalization
            image = apply_normalization(image)

            if self.data_config.preprocessing.is_rgb:
                image = convert_to_rgb(image)
            else:
                image = convert_to_grayscale(image)

            # size matcher
            image, eff_scale = apply_sizematcher(
                image,
                max_height=self.max_hw[0],
                max_width=self.max_hw[1],
            )
            instances = instances * eff_scale

            # get the centroids based on the anchor idx
            centroids = generate_centroids(
                instances, anchor_ind=self.confmap_head_config.anchor_part
            )

            instance, centroid = instances[0], centroids[0]  # (n_samples=1)

            crop_size = np.array(self.crop_hw) * np.sqrt(
                2
            )  # crop extra for rotation augmentation
            crop_size = crop_size.astype(np.int32).tolist()

            sample = generate_crops(image, instance, centroid, self.crop_hw)

            sample["frame_idx"] = torch.tensor(lf.frame_idx, dtype=torch.int32)
            sample["video_idx"] = torch.tensor(video_idx, dtype=torch.int32)
            sample["num_instances"] = num_instances
            sample["orig_size"] = torch.Tensor([orig_img_height, orig_img_width])

            self.cache[idx] = sample.copy()

        for video in self.labels.videos:
            video.close()

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
        sample = self.cache[index]

        # apply augmentation
        if self.apply_aug:
            if "intensity" in self.data_config.augmentation_config:
                sample["instance_image"], sample["instance"] = (
                    apply_intensity_augmentation(
                        sample["instance_image"],
                        sample["instance"],
                        **self.data_config.augmentation_config.intensity,
                    )
                )

            if "geometric" in self.data_config.augmentation_config:
                sample["instance_image"], sample["instance"] = (
                    apply_geometric_augmentation(
                        sample["instance_image"],
                        sample["instance"],
                        **self.data_config.augmentation_config.geometric,
                    )
                )

        # re-crop to original crop size
        sample["instance_bbox"] = torch.unsqueeze(
            make_centered_bboxes(
                sample["centroid"][0], self.crop_hw[0], self.crop_hw[1]
            ),
            0,
        )  # (n_samples=1, 4, 2)

        sample["instance_image"] = crop_and_resize(
            sample["instance_image"], boxes=sample["instance_bbox"], size=self.crop_hw
        )
        point = sample["instance_bbox"][0][0]
        center_instance = sample["instance"] - point
        centered_centroid = sample["centroid"] - point

        sample["instance"] = center_instance  # (n_samples=1, n_nodes, 2)
        sample["centroid"] = centered_centroid  # (n_samples=1, 2)

        # resize image
        sample["instance_image"], sample["instance"] = apply_resizer(
            sample["instance_image"],
            sample["instance"],
            scale=self.data_config.preprocessing.scale,
        )

        # Pad the image (if needed) according max stride
        sample["instance_image"] = apply_pad_to_stride(
            sample["instance_image"], max_stride=self.max_stride
        )

        img_hw = sample["instance_image"].shape[-2:]

        # Generate confidence maps
        confidence_maps = generate_confmaps(
            sample["instance"],
            img_hw=img_hw,
            sigma=self.confmap_head_config.sigma,
            output_stride=self.confmap_head_config.output_stride,
        )

        sample["confidence_maps"] = confidence_maps

        return sample


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
        self._fill_cache()

    def _fill_cache(self):
        """Load all samples to cache."""
        for lf_idx, lf in enumerate(self.labels):
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

            self.cache[lf_idx] = sample.copy()

        for video in self.labels.videos:
            video.close()

    def __getitem__(self, index) -> Dict:
        """Return dict with image and confmaps for centroids for given index."""
        sample = self.cache[index]

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
        self._fill_cache()

    def __getitem__(self, index) -> Dict:
        """Return dict with image and confmaps for instance for given index."""
        sample = self.cache[index]

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