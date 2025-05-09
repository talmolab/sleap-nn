"""Custom `torch.utils.data.Dataset`s for different model types."""

from kornia.geometry.transform import crop_and_resize, crop_by_boxes
from itertools import cycle
from pathlib import Path
import torch.distributed as dist
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
from sleap_nn.data.resizing import (
    apply_pad_to_stride,
    apply_sizematcher,
    apply_resizer,
    apply_padding,
)
from sleap_nn.data.augmentation import (
    apply_geometric_augmentation,
    apply_intensity_augmentation,
)
from sleap_nn.data.confidence_maps import generate_confmaps, generate_multiconfmaps
from sleap_nn.data.edge_maps import generate_pafs
from sleap_nn.data.instance_cropping import make_centered_bboxes, get_fit_bbox
from sleap_nn.training.utils import is_distributed_initialized, get_dist_rank


class BaseDataset(Dataset):
    """Base class for custom torch Datasets.

    Attributes:
        labels: Source `sio.Labels` object.
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        user_instances_only: `True` if only user labeled instances should be used for training. If `False`,
            both user labeled and predicted instances would be used.
        is_rgb: True if the image has 3 channels (RGB image). If input has only one
            channel when this is set to `True`, then the images from single-channel
            is replicated along the channel axis. If input has three channels and this
            is set to False, then we convert the image to grayscale (single-channel)
            image.
        augmentation_config: DictConfig object with `intensity` and `geometric` keys
            according to structure `sleap_nn.config.data_config.AugmentationConfig`.
        scale: Factor to resize the image dimensions by, specified as a float. Default: 1.0.
        apply_aug: `True` if augmentations should be applied to the data pipeline,
            else `False`. Default: `False`.
        max_hw: Maximum height and width of images across the labels file. If `max_height` and
           `max_width` in the config is None, then `max_hw` is used (computed with
            `sleap_nn.data.providers.get_max_height_width`). Else the values in the config
            are used.
        cache_img: String to indicate which caching to use: `memory` or `disk`. If `None`,
            the images aren't cached and loaded from the `.slp` file on each access.
        cache_img_path: Path to save the `.jpg` files. If `None`, current working dir is used.
        use_existing_imgs: Use existing imgs/ chunks in the `cache_img_path`.
        rank: Indicates the rank of the process. Used during distributed training to ensure that image storage to
            disk occurs only once across all workers.
    """

    def __init__(
        self,
        labels: sio.Labels,
        max_stride: int,
        user_instances_only: bool = True,
        is_rgb: bool = False,
        augmentation_config: Optional[DictConfig] = None,
        scale: float = 1.0,
        apply_aug: bool = False,
        max_hw: Tuple[Optional[int]] = (None, None),
        cache_img: Optional[str] = None,
        cache_img_path: Optional[str] = None,
        use_existing_imgs: bool = False,
        rank: Optional[int] = None,
    ) -> None:
        """Initialize class attributes."""
        super().__init__()
        self.labels = labels
        self.user_instances_only = user_instances_only
        self.is_rgb = is_rgb
        self.augmentation_config = augmentation_config
        self.curr_idx = 0
        self.max_stride = max_stride
        self.scale = scale
        self.apply_aug = apply_aug
        self.max_hw = max_hw
        self.rank = rank
        self.max_instances = get_max_instances(self.labels) if self.labels else None
        self.lf_idx_list = self._get_lf_idx_list() if self.labels else None
        self.cache_img = cache_img
        self.cache_img_path = cache_img_path
        self.use_existing_imgs = use_existing_imgs
        if self.cache_img is not None and "disk" in self.cache_img:
            if self.cache_img_path is None:
                self.cache_img_path = "."
            path = (
                Path(self.cache_img_path)
                if isinstance(self.cache_img_path, str)
                else self.cache_img_path
            )
            if not path.is_dir():
                path.mkdir(parents=True, exist_ok=True)

        self.transform_to_pil = T.ToPILImage()
        self.transform_pil_to_tensor = T.ToTensor()
        self.cache = {}

        if self.cache_img is not None:
            if self.cache_img == "memory":
                self._fill_cache()
            elif self.cache_img == "disk" and not self.use_existing_imgs:
                if self.rank is None or self.rank == 0:
                    self._fill_cache()
                if is_distributed_initialized():
                    dist.barrier()

    def _get_lf_idx_list(self) -> List[Tuple[int]]:
        """Return list of indices of labelled frames."""
        lf_idx_list = []
        for lf_idx, lf in enumerate(self.labels):
            # Filter to user instances
            if self.user_instances_only:
                if lf.user_instances is not None and len(lf.user_instances) > 0:
                    lf.instances = lf.user_instances
            is_empty = True
            for _, inst in enumerate(lf.instances):
                if not inst.is_empty:  # filter all NaN instances.
                    is_empty = False
            if not is_empty:
                lf_idx_list.append((lf_idx))
        return lf_idx_list

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
        for lf_idx in self.lf_idx_list:
            img = self.labels[lf_idx].image
            if img.shape[-1] == 1:
                img = np.squeeze(img)
            if self.cache_img == "disk":
                f_name = f"{self.cache_img_path}/sample_{lf_idx}.jpg"
                Image.fromarray(img).save(f_name, format="JPEG")
            if self.cache_img == "memory":
                self.cache[lf_idx] = img

        for video in self.labels.videos:
            if video.is_open:
                video.close()

    def _get_video_idx(self, lf):
        """Return indsample of `lf.video` in `labels.videos`."""
        return self.labels.videos.index(lf.video)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.lf_idx_list)

    def __getitem__(self, index) -> Dict:
        """Returns the sample dict for given index."""
        message = "Subclasses must implement __getitem__"
        logger.error(message)
        raise NotImplementedError(message)


class BottomUpDataset(BaseDataset):
    """Dataset class for bottom-up models.

    Attributes:
        labels: Source `sio.Labels` object.
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        user_instances_only: `True` if only user labeled instances should be used for training. If `False`,
            both user labeled and predicted instances would be used.
        is_rgb: True if the image has 3 channels (RGB image). If input has only one
            channel when this is set to `True`, then the images from single-channel
            is replicated along the channel axis. If input has three channels and this
            is set to False, then we convert the image to grayscale (single-channel)
            image.
        augmentation_config: DictConfig object with `intensity` and `geometric` keys
            according to structure `sleap_nn.config.data_config.AugmentationConfig`.
        scale: Factor to resize the image dimensions by, specified as a float. Default: 1.0.
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
        cache_img: String to indicate which caching to use: `memory` or `disk`. If `None`,
            the images aren't cached and loaded from the `.slp` file on each access.
        cache_img_path: Path to save the `.jpg` files. If `None`, current working dir is used.
        use_existing_imgs: Use existing imgs/ chunks in the `cache_img_path`.
        rank: Indicates the rank of the process. Used during distributed training to ensure that image storage to
            disk occurs only once across all workers.
    """

    def __init__(
        self,
        labels: sio.Labels,
        confmap_head_config: DictConfig,
        pafs_head_config: DictConfig,
        max_stride: int,
        user_instances_only: bool = True,
        is_rgb: bool = False,
        augmentation_config: Optional[DictConfig] = None,
        scale: float = 1.0,
        apply_aug: bool = False,
        max_hw: Tuple[Optional[int]] = (None, None),
        cache_img: Optional[str] = None,
        cache_img_path: Optional[str] = None,
        use_existing_imgs: bool = False,
        rank: Optional[int] = None,
    ) -> None:
        """Initialize class attributes."""
        super().__init__(
            labels=labels,
            max_stride=max_stride,
            user_instances_only=user_instances_only,
            is_rgb=is_rgb,
            augmentation_config=augmentation_config,
            scale=scale,
            apply_aug=apply_aug,
            max_hw=max_hw,
            cache_img=cache_img,
            cache_img_path=cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )
        self.confmap_head_config = confmap_head_config
        self.pafs_head_config = pafs_head_config

        self.edge_inds = self.labels.skeletons[0].edge_inds

    def __getitem__(self, index) -> Dict:
        """Return dict with image, confmaps and pafs for given index."""
        lf_idx = self.lf_idx_list[index]

        lf = self.labels[lf_idx]

        # load the img
        if self.cache_img is not None:
            if self.cache_img == "disk":
                img = np.array(Image.open(f"{self.cache_img_path}/sample_{lf_idx}.jpg"))
            elif self.cache_img == "memory":
                img = self.cache[lf_idx].copy()

        else:  # load from slp file if not cached
            img = lf.image

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        video_idx = self._get_video_idx(lf)

        # get dict
        sample = process_lf(
            lf,
            video_idx=video_idx,
            max_instances=self.max_instances,
            user_instances_only=self.user_instances_only,
        )

        # apply normalization
        sample["image"] = apply_normalization(sample["image"])

        if self.is_rgb:
            sample["image"] = convert_to_rgb(sample["image"])
        else:
            sample["image"] = convert_to_grayscale(sample["image"])

        # size matcher
        sample["image"], eff_scale, (pad_w_l, pad_h_t) = apply_sizematcher(
            sample["image"],
            max_height=self.max_hw[0],
            max_width=self.max_hw[1],
        )
        sample["instances"] = sample["instances"] * eff_scale
        sample["instances"] = sample["instances"] + torch.Tensor((pad_w_l, pad_h_t))

        # resize image
        sample["image"], sample["instances"] = apply_resizer(
            sample["image"],
            sample["instances"],
            scale=self.scale,
        )

        # Pad the image (if needed) according max stride
        sample["image"], (pad_w_l, pad_h_t) = apply_pad_to_stride(
            sample["image"], max_stride=self.max_stride
        )
        sample["instances"] = sample["instances"] + torch.Tensor((pad_w_l, pad_h_t))

        # apply augmentation
        if self.apply_aug and self.augmentation_config is not None:
            if "intensity" in self.augmentation_config:
                sample["image"], sample["instances"] = apply_intensity_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.augmentation_config.intensity,
                )

            if "geometric" in self.augmentation_config:
                sample["image"], sample["instances"] = apply_geometric_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.augmentation_config.geometric,
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
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        user_instances_only: `True` if only user labeled instances should be used for training. If `False`,
            both user labeled and predicted instances would be used.
        is_rgb: True if the image has 3 channels (RGB image). If input has only one
            channel when this is set to `True`, then the images from single-channel
            is replicated along the channel axis. If input has three channels and this
            is set to False, then we convert the image to grayscale (single-channel)
            image.
        augmentation_config: DictConfig object with `intensity` and `geometric` keys
            according to structure `sleap_nn.config.data_config.AugmentationConfig`.
        scale: Factor to resize the image dimensions by, specified as a float. Default: 1.0.
        apply_aug: `True` if augmentations should be applied to the data pipeline,
            else `False`. Default: `False`.
        max_hw: Maximum height and width of images across the labels file. If `max_height` and
           `max_width` in the config is None, then `max_hw` is used (computed with
            `sleap_nn.data.providers.get_max_height_width`). Else the values in the config
            are used.
        cache_img: String to indicate which caching to use: `memory` or `disk`. If `None`,
            the images aren't cached and loaded from the `.slp` file on each access.
        cache_img_path: Path to save the `.jpg` files. If `None`, current working dir is used.
        use_existing_imgs: Use existing imgs/ chunks in the `cache_img_path`.
        crop_hw: Height and width of the crop in pixels.
        rank: Indicates the rank of the process. Used during distributed training to ensure that image storage to
            disk occurs only once across all workers.

    Note: If scale is provided for centered-instance model, the images are cropped out
    from the scaled image with the given crop size.
    """

    def __init__(
        self,
        labels: sio.Labels,
        crop_hw: Tuple[int],
        confmap_head_config: DictConfig,
        max_stride: int,
        user_instances_only: bool = True,
        is_rgb: bool = False,
        augmentation_config: Optional[DictConfig] = None,
        scale: float = 1.0,
        apply_aug: bool = False,
        max_hw: Tuple[Optional[int]] = (None, None),
        cache_img: Optional[str] = None,
        cache_img_path: Optional[str] = None,
        use_existing_imgs: bool = False,
        rank: Optional[int] = None,
    ) -> None:
        """Initialize class attributes."""
        super().__init__(
            labels=labels,
            max_stride=max_stride,
            user_instances_only=user_instances_only,
            is_rgb=is_rgb,
            augmentation_config=augmentation_config,
            scale=scale,
            apply_aug=apply_aug,
            max_hw=max_hw,
            cache_img=cache_img,
            cache_img_path=cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )
        self.crop_hw = crop_hw
        self.confmap_head_config = confmap_head_config
        self.instance_idx_list = self._get_instance_idx_list() if self.labels else None
        self.cache_lf = [None, None]

    def _get_instance_idx_list(self) -> List[Tuple[int]]:
        """Return list of tuples with indices of labelled frames and instances."""
        instance_idx_list = []
        for lf_idx, lf in enumerate(self.labels):
            # Filter to user instances
            if self.user_instances_only:
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
        lf_idx, inst_idx = self.instance_idx_list[index]
        lf = self.labels[lf_idx]

        if lf_idx == self.cache_lf[0]:
            img = self.cache_lf[1]
        else:
            # load the img
            if self.cache_img is not None:
                if self.cache_img == "disk":
                    img = np.array(
                        Image.open(f"{self.cache_img_path}/sample_{lf_idx}.jpg")
                    )
                elif self.cache_img == "memory":
                    img = self.cache[lf_idx].copy()

            else:  # load from slp file if not cached
                img = lf.image  # TODO: doesn't work when num_workers > 0

            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

            self.cache_lf = [lf_idx, img]

        video_idx = self._get_video_idx(lf)

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

        if self.is_rgb:
            image = convert_to_rgb(image)
        else:
            image = convert_to_grayscale(image)

        # size matcher
        image, eff_scale, (pad_w_l, pad_h_t) = apply_sizematcher(
            image,
            max_height=self.max_hw[0],
            max_width=self.max_hw[1],
        )
        instances = instances * eff_scale
        instances = instances + torch.Tensor((pad_w_l, pad_h_t))

        # resize image
        image, instances = apply_resizer(
            image,
            instances,
            scale=self.scale,
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

        # apply augmentation
        if self.apply_aug and self.augmentation_config is not None:
            if "intensity" in self.augmentation_config:
                sample["instance_image"], sample["instance"] = (
                    apply_intensity_augmentation(
                        sample["instance_image"],
                        sample["instance"],
                        **self.augmentation_config.intensity,
                    )
                )

            if "geometric" in self.augmentation_config:
                sample["instance_image"], sample["instance"] = (
                    apply_geometric_augmentation(
                        sample["instance_image"],
                        sample["instance"],
                        **self.augmentation_config.geometric,
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
        sample["instance_image"], (pad_w_l, pad_h_t) = apply_pad_to_stride(
            sample["instance_image"], max_stride=self.max_stride
        )
        sample["instance"] = sample["instance"] + torch.Tensor((pad_w_l, pad_h_t))

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


class CenteredInstanceDatasetFitBbox(CenteredInstanceDataset):
    def __init__(
        self,
        labels,
        max_crop_hw,
        confmap_head_config,
        max_stride,
        user_instances_only=True,
        is_rgb=False,
        augmentation_config=None,
        scale=1,
        apply_aug=False,
        max_hw=(None, None),
        cache_img=None,
        cache_img_path=None,
        use_existing_imgs=False,
        rank=None,
    ):
        super().__init__(
            labels,
            max_crop_hw,
            confmap_head_config,
            max_stride,
            user_instances_only,
            is_rgb,
            augmentation_config,
            scale,
            apply_aug,
            max_hw,
            cache_img,
            cache_img_path,
            use_existing_imgs,
            rank,
        )
        self.max_crop_h_w = max_crop_hw

    def __getitem__(self, index):
        """Return dict with cropped image and confmaps of instance for given index."""
        lf_idx, inst_idx = self.instance_idx_list[index]
        lf = self.labels[lf_idx]

        if lf_idx == self.cache_lf[0]:
            img = self.cache_lf[1]
        else:
            # load the img
            if self.cache_img is not None:
                if self.cache_img == "disk":
                    img = np.array(
                        Image.open(f"{self.cache_img_path}/sample_{lf_idx}.jpg")
                    )
                elif self.cache_img == "memory":
                    img = self.cache[lf_idx].copy()

            else:  # load from slp file if not cached
                img = lf.image  # TODO: doesn't work when num_workers > 0

            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

            self.cache_lf = [lf_idx, img]

        video_idx = self._get_video_idx(lf)

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

        if self.is_rgb:
            image = convert_to_rgb(image)
        else:
            image = convert_to_grayscale(image)

        # size matcher
        image, eff_scale, (pad_w_l, pad_h_t) = apply_sizematcher(
            image,
            max_height=self.max_hw[0],
            max_width=self.max_hw[1],
        )
        instances = instances * eff_scale
        instances = instances + torch.Tensor((pad_w_l, pad_h_t))

        # resize image
        image, instances = apply_resizer(
            image,
            instances,
            scale=self.scale,
        )

        # apply augmentation
        if self.apply_aug and self.augmentation_config is not None:
            if "intensity" in self.augmentation_config:
                image, instances = apply_intensity_augmentation(
                    image,
                    instances,
                    **self.augmentation_config.intensity,
                )

            if "geometric" in self.augmentation_config:
                image, instances = apply_geometric_augmentation(
                    image,
                    instances,
                    **self.augmentation_config.geometric,
                )

        instance = instances[0]  # (n_samples=1)

        bbox = get_fit_bbox(instance)  # bbox => (x_min, y_min, x_max, y_max)
        bbox[0] = bbox[0] - 16
        bbox[1] = bbox[1] - 16
        bbox[2] = bbox[2] + 16
        bbox[3] = bbox[3] + 16  # padding of 16 on all sides
        x_min, y_min, x_max, y_max = bbox
        crop_hw = (y_max - y_min, x_max - x_min)

        cropped_image = crop_by_boxes(
            image,
            src_box=torch.Tensor(
                [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            ).unsqueeze(dim=0),
            dst_box=torch.tensor(
                [
                    [
                        [0.0, 0.0],
                        [crop_hw[1], 0.0],
                        [crop_hw[1], crop_hw[0]],
                        [0.0, crop_hw[0]],
                    ]
                ]
            ),
        )
        instance = instance - bbox[:2]  # adjust for crops

        cropped_image_match_hw, eff_scale, pad_wh = apply_sizematcher(
            cropped_image, self.max_crop_h_w[0], self.max_crop_h_w[1]
        )  # resize and pad to max crfop  size
        instance = instance * eff_scale  # adjust keypoints acc to resizing/ padding
        instance = instance + torch.Tensor(pad_wh)
        instance = torch.unsqueeze(instance, dim=0)

        sample = {}
        sample["instance_image"] = cropped_image_match_hw
        sample["instance_bbox"] = bbox
        sample["instance"] = instance
        sample["frame_idx"] = torch.tensor(lf.frame_idx, dtype=torch.int32)
        sample["video_idx"] = torch.tensor(video_idx, dtype=torch.int32)
        sample["num_instances"] = num_instances
        sample["orig_size"] = torch.Tensor([orig_img_height, orig_img_width])
        sample["crop_hw"] = torch.Tensor([crop_hw])

        # Pad the image (if needed) according max stride
        sample["instance_image"], pad_wh = apply_pad_to_stride(
            sample["instance_image"], max_stride=self.max_stride
        )
        sample["instance"] = sample["instance"] + torch.Tensor(pad_wh)

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
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        user_instances_only: `True` if only user labeled instances should be used for training. If `False`,
            both user labeled and predicted instances would be used.
        is_rgb: True if the image has 3 channels (RGB image). If input has only one
            channel when this is set to `True`, then the images from single-channel
            is replicated along the channel axis. If input has three channels and this
            is set to False, then we convert the image to grayscale (single-channel)
            image.
        augmentation_config: DictConfig object with `intensity` and `geometric` keys
            according to structure `sleap_nn.config.data_config.AugmentationConfig`.
        scale: Factor to resize the image dimensions by, specified as a float. Default: 1.0.
        apply_aug: `True` if augmentations should be applied to the data pipeline,
            else `False`. Default: `False`.
        max_hw: Maximum height and width of images across the labels file. If `max_height` and
           `max_width` in the config is None, then `max_hw` is used (computed with
            `sleap_nn.data.providers.get_max_height_width`). Else the values in the config
            are used.
        cache_img: String to indicate which caching to use: `memory` or `disk`. If `None`,
            the images aren't cached and loaded from the `.slp` file on each access.
        cache_img_path: Path to save the `.jpg` files. If `None`, current working dir is used.
        use_existing_imgs: Use existing imgs/ chunks in the `cache_img_path`.
        confmap_head_config: DictConfig object with all the keys in the `head_config` section.
        (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
        rank: Indicates the rank of the process. Used during distributed training to ensure that image storage to
            disk occurs only once across all workers.
    """

    def __init__(
        self,
        labels: sio.Labels,
        confmap_head_config: DictConfig,
        max_stride: int,
        user_instances_only: bool = True,
        is_rgb: bool = False,
        augmentation_config: Optional[DictConfig] = None,
        scale: float = 1.0,
        apply_aug: bool = False,
        max_hw: Tuple[Optional[int]] = (None, None),
        cache_img: Optional[str] = None,
        cache_img_path: Optional[str] = None,
        use_existing_imgs: bool = False,
        rank: Optional[int] = None,
    ) -> None:
        """Initialize class attributes."""
        super().__init__(
            labels=labels,
            max_stride=max_stride,
            user_instances_only=user_instances_only,
            is_rgb=is_rgb,
            augmentation_config=augmentation_config,
            scale=scale,
            apply_aug=apply_aug,
            max_hw=max_hw,
            cache_img=cache_img,
            cache_img_path=cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )
        self.confmap_head_config = confmap_head_config

    def __getitem__(self, index) -> Dict:
        """Return dict with image and confmaps for centroids for given index."""
        lf_idx = self.lf_idx_list[index]

        lf = self.labels[lf_idx]

        # load the img
        if self.cache_img is not None:
            if self.cache_img == "disk":
                img = np.array(Image.open(f"{self.cache_img_path}/sample_{lf_idx}.jpg"))
            elif self.cache_img == "memory":
                img = self.cache[lf_idx].copy()

        else:  # load from slp file if not cached
            img = lf.image

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        video_idx = self._get_video_idx(lf)

        # get dict
        sample = process_lf(
            lf,
            video_idx=video_idx,
            max_instances=self.max_instances,
            user_instances_only=self.user_instances_only,
        )

        # apply normalization
        sample["image"] = apply_normalization(sample["image"])

        if self.is_rgb:
            sample["image"] = convert_to_rgb(sample["image"])
        else:
            sample["image"] = convert_to_grayscale(sample["image"])

        # size matcher
        sample["image"], eff_scale, (pad_w_l, pad_h_t) = apply_sizematcher(
            sample["image"],
            max_height=self.max_hw[0],
            max_width=self.max_hw[1],
        )
        sample["instances"] = sample["instances"] * eff_scale
        sample["instances"] = sample["instances"] + torch.Tensor((pad_w_l, pad_h_t))

        # resize image
        sample["image"], sample["instances"] = apply_resizer(
            sample["image"],
            sample["instances"],
            scale=self.scale,
        )

        # get the centroids based on the anchor idx
        centroids = generate_centroids(
            sample["instances"], anchor_ind=self.confmap_head_config.anchor_part
        )

        sample["centroids"] = centroids

        # Pad the image (if needed) according max stride
        sample["image"], (pad_w_l, pad_h_t) = apply_pad_to_stride(
            sample["image"], max_stride=self.max_stride
        )
        sample["instances"] = sample["instances"] + torch.Tensor((pad_w_l, pad_h_t))
        sample["centroids"] = sample["centroids"] + torch.Tensor((pad_w_l, pad_h_t))

        # apply augmentation
        if self.apply_aug and self.augmentation_config is not None:
            if "intensity" in self.augmentation_config:
                sample["image"], sample["centroids"] = apply_intensity_augmentation(
                    sample["image"],
                    sample["centroids"],
                    **self.augmentation_config.intensity,
                )

            if "geometric" in self.augmentation_config:
                sample["image"], sample["centroids"] = apply_geometric_augmentation(
                    sample["image"],
                    sample["centroids"],
                    **self.augmentation_config.geometric,
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
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        user_instances_only: `True` if only user labeled instances should be used for training. If `False`,
            both user labeled and predicted instances would be used.
        is_rgb: True if the image has 3 channels (RGB image). If input has only one
            channel when this is set to `True`, then the images from single-channel
            is replicated along the channel axis. If input has three channels and this
            is set to False, then we convert the image to grayscale (single-channel)
            image.
        augmentation_config: DictConfig object with `intensity` and `geometric` keys
            according to structure `sleap_nn.config.data_config.AugmentationConfig`.
        scale: Factor to resize the image dimensions by, specified as a float. Default: 1.0.
        apply_aug: `True` if augmentations should be applied to the data pipeline,
            else `False`. Default: `False`.
        max_hw: Maximum height and width of images across the labels file. If `max_height` and
           `max_width` in the config is None, then `max_hw` is used (computed with
            `sleap_nn.data.providers.get_max_height_width`). Else the values in the config
            are used.
        cache_img: String to indicate which caching to use: `memory` or `disk`. If `None`,
            the images aren't cached and loaded from the `.slp` file on each access.
        cache_img_path: Path to save the `.jpg` files. If `None`, current working dir is used.
        use_existing_imgs: Use existing imgs/ chunks in the `cache_img_path`.
        confmap_head_config: DictConfig object with all the keys in the `head_config` section.
        (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
        rank: Indicates the rank of the process. Used during distributed training to ensure that image storage to
            disk occurs only once across all workers.
    """

    def __init__(
        self,
        labels: sio.Labels,
        confmap_head_config: DictConfig,
        max_stride: int,
        user_instances_only: bool = True,
        is_rgb: bool = False,
        augmentation_config: Optional[DictConfig] = None,
        scale: float = 1.0,
        apply_aug: bool = False,
        max_hw: Tuple[Optional[int]] = (None, None),
        cache_img: Optional[str] = None,
        cache_img_path: Optional[str] = None,
        use_existing_imgs: bool = False,
        rank: Optional[int] = None,
    ) -> None:
        """Initialize class attributes."""
        super().__init__(
            labels=labels,
            max_stride=max_stride,
            user_instances_only=user_instances_only,
            is_rgb=is_rgb,
            augmentation_config=augmentation_config,
            scale=scale,
            apply_aug=apply_aug,
            max_hw=max_hw,
            cache_img=cache_img,
            cache_img_path=cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )
        self.confmap_head_config = confmap_head_config

    def __getitem__(self, index) -> Dict:
        """Return dict with image and confmaps for instance for given index."""
        lf_idx = self.lf_idx_list[index]

        lf = self.labels[lf_idx]

        # load the img
        if self.cache_img is not None:
            if self.cache_img == "disk":
                img = np.array(Image.open(f"{self.cache_img_path}/sample_{lf_idx}.jpg"))
            elif self.cache_img == "memory":
                img = self.cache[lf_idx].copy()

        else:  # load from slp file if not cached
            img = lf.image

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        video_idx = self._get_video_idx(lf)

        # get dict
        sample = process_lf(
            lf,
            video_idx=video_idx,
            max_instances=self.max_instances,
            user_instances_only=self.user_instances_only,
        )

        # apply normalization
        sample["image"] = apply_normalization(sample["image"])

        if self.is_rgb:
            sample["image"] = convert_to_rgb(sample["image"])
        else:
            sample["image"] = convert_to_grayscale(sample["image"])

        # size matcher
        sample["image"], eff_scale, (pad_w_l, pad_h_t) = apply_sizematcher(
            sample["image"],
            max_height=self.max_hw[0],
            max_width=self.max_hw[1],
        )
        sample["instances"] = sample["instances"] * eff_scale
        sample["instances"] = sample["instances"] + torch.Tensor((pad_w_l, pad_h_t))

        # resize image
        sample["image"], sample["instances"] = apply_resizer(
            sample["image"],
            sample["instances"],
            scale=self.scale,
        )

        # Pad the image (if needed) according max stride
        sample["image"], (pad_w_l, pad_h_t) = apply_pad_to_stride(
            sample["image"], max_stride=self.max_stride
        )
        sample["instances"] = sample["instances"] + torch.Tensor((pad_w_l, pad_h_t))

        # apply augmentation
        if self.apply_aug and self.augmentation_config is not None:
            if "intensity" in self.augmentation_config:
                sample["image"], sample["instances"] = apply_intensity_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.augmentation_config.intensity,
                )

            if "geometric" in self.augmentation_config:
                sample["image"], sample["instances"] = apply_geometric_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.augmentation_config.geometric,
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
