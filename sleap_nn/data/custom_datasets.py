"""Custom `torch.utils.data.Dataset`s for different model types."""

from kornia.geometry.transform import crop_and_resize
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from omegaconf import DictConfig
import numpy as np
from PIL import Image
from loguru import logger
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import sleap_io as sio
from sleap_nn.data.instance_centroids import generate_centroids
from sleap_nn.data.instance_cropping import generate_crops
from sleap_nn.data.normalization import (
    apply_normalization,
    convert_to_grayscale,
    convert_to_rgb,
)
from sleap_nn.data.providers import get_max_instances, get_max_height_width, process_lf
from sleap_nn.data.resizing import apply_pad_to_stride, apply_sizematcher, apply_resizer
from sleap_nn.data.augmentation import (
    apply_geometric_augmentation,
    apply_intensity_augmentation,
)
from sleap_nn.data.confidence_maps import generate_confmaps, generate_multiconfmaps
from sleap_nn.data.edge_maps import generate_pafs
from sleap_nn.data.instance_cropping import make_centered_bboxes


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
        np_chunks: If `True`, `.npz` chunks are generated and samples are loaded from
            these chunks during training. Else, in-memory caching is used.
        np_chunks_path: Path to save the `.npz` chunks. If `None`, current working dir is used.
        use_existing_chunks: Use existing chunks in the `np_chunks_path`.
    """

    def __init__(
        self,
        labels: sio.Labels,
        data_config: DictConfig,
        max_stride: int,
        apply_aug: bool = False,
        max_hw: Tuple[Optional[int]] = (None, None),
        np_chunks: bool = False,
        np_chunks_path: Optional[str] = None,
        use_existing_chunks: bool = False,
    ) -> None:
        """Initialize class attributes."""
        super().__init__()
        self.labels = labels
        self.data_config = data_config
        self.curr_idx = 0
        self.max_stride = max_stride
        self.apply_aug = apply_aug
        self.max_hw = max_hw
        self.max_instances = get_max_instances(self.labels)
        self.np_chunks = np_chunks
        self.np_chunks_path = np_chunks_path
        self.use_existing_chunks = use_existing_chunks
        if self.np_chunks_path is None:
            self.np_chunks_path = "."
        path = (
            Path(self.np_chunks_path)
            if isinstance(self.np_chunks_path, str)
            else self.np_chunks_path
        )
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)

        self.transform_to_pil = T.ToPILImage()
        self.transform_pil_to_tensor = T.ToTensor()
        self.cache = {}

    def __next__(self):
        """Get the next sample from the dataset."""
        if self.curr_idx >= len(self):
            raise StopIteration

        sample = self.__getitem__(self.curr_idx)
        self.curr_idx += 1
        return sample

    def __iter__(self):
        """Returns an iterator."""
        return self

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

            if self.np_chunks:
                sample["image"] = self.transform_to_pil(sample["image"].squeeze(dim=0))
                for k, v in sample.items():
                    if k != "image" and isinstance(v, torch.Tensor):
                        sample[k] = v.numpy()
                f_name = f"{self.np_chunks_path}/sample_{lf_idx}.npz"
                np.savez_compressed(f_name, **sample)
                self.cache[lf_idx] = f_name

            else:
                self.cache[lf_idx] = sample.copy()

        for video in self.labels.videos:
            video.close()

    def _get_video_idx(self, lf):
        """Return indsample of `lf.video` in `labels.videos`."""
        return self.labels.videos.index(lf.video)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.cache)

    def __getitem__(self, index) -> Dict:
        """Returns the sample dict for given index."""
        message = "Subclasses must implement __getitem__"
        logger.error(message)
        raise NotImplementedError(message)


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
        np_chunks: If `True`, `.npz` chunks are generated and samples are loaded from
            these chunks during training. Else, in-memory caching is used.
        np_chunks_path: Path to save the `.npz` chunks. If `None`, current working dir is used.
        use_existing_chunks: Use existing chunks in the `np_chunks_path`.
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
        np_chunks: bool = False,
        np_chunks_path: Optional[str] = None,
        use_existing_chunks: bool = False,
    ) -> None:
        """Initialize class attributes."""
        super().__init__(
            labels=labels,
            data_config=data_config,
            max_stride=max_stride,
            apply_aug=apply_aug,
            max_hw=max_hw,
            np_chunks=np_chunks,
            np_chunks_path=np_chunks_path,
            use_existing_chunks=use_existing_chunks,
        )
        self.confmap_head_config = confmap_head_config
        self.pafs_head_config = pafs_head_config

        self.edge_inds = self.labels.skeletons[0].edge_inds
        if not self.use_existing_chunks:
            self._fill_cache()

    def __getitem__(self, index) -> Dict:
        """Return dict with image, confmaps and pafs for given index."""
        if self.np_chunks:
            ex = np.load(f"{self.np_chunks_path}/sample_{index}.npz")
            sample = {}
            for k, v in ex.items():
                if k != "image":
                    sample[k] = torch.from_numpy(v)
                else:
                    sample[k] = self.transform_pil_to_tensor(
                        Image.fromarray(ex["image"])
                    ).unsqueeze(dim=0)
        else:
            sample = self.cache[index].copy()

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
        np_chunks: If `True`, `.npz` chunks are generated and samples are loaded from
            these chunks during training. Else, in-memory caching is used.
        np_chunks_path: Path to save the `.npz` chunks. If `None`, current working dir is used.
        confmap_head_config: DictConfig object with all the keys in the `head_config` section.
        (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
        crop_hw: Height and width of the crop in pixels.
        use_existing_chunks: Use existing chunks in the `np_chunks_path`.

    Note: If scale is provided for centered-instance model, the images are cropped out
    from the scaled image with the given crop size.
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
        np_chunks: bool = False,
        np_chunks_path: Optional[str] = None,
        use_existing_chunks: bool = False,
    ) -> None:
        """Initialize class attributes."""
        super().__init__(
            labels=labels,
            data_config=data_config,
            max_stride=max_stride,
            apply_aug=apply_aug,
            max_hw=max_hw,
            np_chunks=np_chunks,
            np_chunks_path=np_chunks_path,
            use_existing_chunks=use_existing_chunks,
        )
        self.crop_hw = crop_hw
        self.confmap_head_config = confmap_head_config
        self.instance_idx_list = self._get_instance_idx_list()
        self.cache_lf = [None, None]
        if not self.use_existing_chunks:
            self._fill_cache()

    def _fill_cache(self):
        """Load all samples to cache."""
        for idx, (lf_idx, inst_idx) in enumerate(self.instance_idx_list):
            lf = self.labels[lf_idx]
            video_idx = self._get_video_idx(lf)

            if lf_idx == self.cache_lf[0]:
                img = self.cache_lf[1]
            else:
                img = lf.image
                self.cache_lf = [lf_idx, img]

            image = np.transpose(img, (2, 0, 1))  # HWC -> CHW

            instances = []
            for inst in lf:
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

            # resize image
            image, instances = apply_resizer(
                image,
                instances,
                scale=self.data_config.preprocessing.scale,
            )

            # get the centroids based on the anchor idx
            centroids = generate_centroids(
                instances, anchor_ind=self.confmap_head_config.anchor_part
            )

            instance, centroid = instances[0], centroids[0]  # (n_samples=1)

            crop_size = np.array(self.crop_hw) * np.sqrt(
                2
            )  # crop extra for rotation augmentation
            crop_size = crop_size.astype(np.int32).tolist()

            sample = generate_crops(image, instance, centroid, crop_size)

            sample["frame_idx"] = torch.tensor(lf.frame_idx, dtype=torch.int32)
            sample["video_idx"] = torch.tensor(video_idx, dtype=torch.int32)
            sample["num_instances"] = num_instances
            sample["orig_size"] = torch.Tensor([orig_img_height, orig_img_width])

            if self.np_chunks:
                sample["instance_image"] = self.transform_to_pil(
                    sample["instance_image"].squeeze(dim=0)
                )
                for k, v in sample.items():
                    if k != "instance_image" and isinstance(v, torch.Tensor):
                        sample[k] = v.numpy()
                f_name = f"{self.np_chunks_path}/sample_{idx}.npz"
                np.savez_compressed(f_name, **sample)
                self.cache[idx] = f_name

            else:
                self.cache[idx] = sample.copy()

        for video in self.labels.videos:
            video.close()

    def _get_instance_idx_list(self) -> List[Tuple[int]]:
        """Return list of tuples with indices of labelled frames and instances."""
        instance_idx_list = []
        for lf_idx, lf in enumerate(self.labels):
            # Filter to user instances
            if self.data_config.user_instances_only:
                if lf.user_instances is not None and len(lf.user_instances) > 0:
                    lf.instances = lf.user_instances
            for inst_idx, inst in enumerate(lf.instances):
                if not inst.is_empty:  # filter all NaN instances.
                    instance_idx_list.append((lf_idx, inst_idx))
        return instance_idx_list

    def __len__(self) -> int:
        """Return number of instances in the labels object."""
        return len(self.instance_idx_list)

    def __getitem__(self, index) -> Dict:
        """Return dict with cropped image and confmaps of instance for given index."""
        if self.np_chunks:
            ex = np.load(f"{self.np_chunks_path}/sample_{index}.npz")
            sample = {}
            for k, v in ex.items():
                if k != "instance_image":
                    sample[k] = torch.from_numpy(v)
                else:
                    sample[k] = self.transform_pil_to_tensor(
                        Image.fromarray(ex["instance_image"])
                    ).unsqueeze(dim=0)
        else:
            sample = self.cache[index].copy()

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
        np_chunks: If `True`, `.npz` chunks are generated and samples are loaded from
            these chunks during training. Else, in-memory caching is used.
        np_chunks_path: Path to save the `.npz` chunks. If `None`, current working dir is used.
        confmap_head_config: DictConfig object with all the keys in the `head_config` section.
        (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
        use_existing_chunks: Use existing chunks in the `np_chunks_path`.
    """

    def __init__(
        self,
        labels: sio.Labels,
        data_config: DictConfig,
        confmap_head_config: DictConfig,
        max_stride: int,
        apply_aug: bool = False,
        max_hw: Tuple[Optional[int]] = (None, None),
        np_chunks: bool = False,
        np_chunks_path: Optional[str] = None,
        use_existing_chunks: bool = False,
    ) -> None:
        """Initialize class attributes."""
        super().__init__(
            labels=labels,
            data_config=data_config,
            max_stride=max_stride,
            apply_aug=apply_aug,
            max_hw=max_hw,
            np_chunks=np_chunks,
            np_chunks_path=np_chunks_path,
            use_existing_chunks=use_existing_chunks,
        )
        self.confmap_head_config = confmap_head_config
        if not self.use_existing_chunks:
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

            # resize image
            sample["image"], sample["instances"] = apply_resizer(
                sample["image"],
                sample["instances"],
                scale=self.data_config.preprocessing.scale,
            )

            # get the centroids based on the anchor idx
            centroids = generate_centroids(
                sample["instances"], anchor_ind=self.confmap_head_config.anchor_part
            )

            sample["centroids"] = centroids

            # Pad the image (if needed) according max stride
            sample["image"] = apply_pad_to_stride(
                sample["image"], max_stride=self.max_stride
            )

            if self.np_chunks:
                sample["image"] = self.transform_to_pil(sample["image"].squeeze(dim=0))
                for k, v in sample.items():
                    if k != "image" and isinstance(v, torch.Tensor):
                        sample[k] = v.numpy()
                f_name = f"{self.np_chunks_path}/sample_{lf_idx}.npz"
                np.savez_compressed(f_name, **sample)
                self.cache[lf_idx] = f_name

            else:
                self.cache[lf_idx] = sample.copy()

        for video in self.labels.videos:
            video.close()

    def __getitem__(self, index) -> Dict:
        """Return dict with image and confmaps for centroids for given index."""
        if self.np_chunks:
            ex = np.load(f"{self.np_chunks_path}/sample_{index}.npz")
            sample = {}
            for k, v in ex.items():
                if k != "image":
                    sample[k] = torch.from_numpy(v)
                else:
                    sample[k] = self.transform_pil_to_tensor(
                        Image.fromarray(ex["image"])
                    ).unsqueeze(dim=0)
        else:
            sample = self.cache[index].copy()

        # apply augmentation
        if self.apply_aug:
            if "intensity" in self.data_config.augmentation_config:
                sample["image"], sample["centroids"] = apply_intensity_augmentation(
                    sample["image"],
                    sample["centroids"],
                    **self.data_config.augmentation_config.intensity,
                )

            if "geometric" in self.data_config.augmentation_config:
                sample["image"], sample["centroids"] = apply_geometric_augmentation(
                    sample["image"],
                    sample["centroids"],
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
        np_chunks: If `True`, `.npz` chunks are generated and samples are loaded from
            these chunks during training. Else, in-memory caching is used.
        np_chunks_path: Path to save the `.npz` chunks. If `None`, current working dir is used.
        confmap_head_config: DictConfig object with all the keys in the `head_config` section.
        (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
        use_existing_chunks: Use existing chunks in the `np_chunks_path`.
    """

    def __init__(
        self,
        labels: sio.Labels,
        data_config: DictConfig,
        confmap_head_config: DictConfig,
        max_stride: int,
        apply_aug: bool = False,
        max_hw: Tuple[Optional[int]] = (None, None),
        np_chunks: bool = False,
        np_chunks_path: Optional[str] = None,
        use_existing_chunks: bool = False,
    ) -> None:
        """Initialize class attributes."""
        super().__init__(
            labels=labels,
            data_config=data_config,
            max_stride=max_stride,
            apply_aug=apply_aug,
            max_hw=max_hw,
            np_chunks=np_chunks,
            np_chunks_path=np_chunks_path,
            use_existing_chunks=use_existing_chunks,
        )
        self.confmap_head_config = confmap_head_config
        if not self.use_existing_chunks:
            self._fill_cache()

    def __getitem__(self, index) -> Dict:
        """Return dict with image and confmaps for instance for given index."""
        if self.np_chunks:
            ex = np.load(f"{self.np_chunks_path}/sample_{index}.npz")
            sample = {}
            for k, v in ex.items():
                if k != "image":
                    sample[k] = torch.from_numpy(v)
                else:
                    sample[k] = self.transform_pil_to_tensor(
                        Image.fromarray(ex["image"])
                    ).unsqueeze(dim=0)
        else:
            sample = self.cache[index].copy()

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


class _RepeatSampler:
    """Sampler that cycles through the samples infintely.

    Source: Ultralytics

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes the sampler object."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


class CyclerDataLoader(DataLoader):
    """DataLoader that cycles through the dataset infinitely.

    Attributes:
        steps_per_epoch: Number of steps to be run in an epoch. If not provided, the
            length of the sampler is used (total_samples / batch_size)    .
    """

    def __init__(self, steps_per_epoch: Optional[int] = None, *args, **kwargs):
        """Initialize the object."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()
        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        """Returns the length of the dataloader."""
        if self.steps_per_epoch is not None:
            return int(self.steps_per_epoch)
        else:
            return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        while True:
            for _ in range(len(self)):
                yield next(self.iterator)

    def reset(self):
        """Reset iterator."""
        self.iterator = self._get_iterator()
