"""Custom `torch.utils.data.Dataset`s for different model types."""

from kornia.geometry.transform import crop_and_resize

# from concurrent.futures import ThreadPoolExecutor # TODO: implement parallel processing
# import concurrent.futures
# import os
from itertools import cycle
from pathlib import Path
import torch.distributed as dist
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from omegaconf import DictConfig, OmegaConf
import numpy as np
from PIL import Image
from loguru import logger
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.console import Console
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import sleap_io as sio
from sleap_nn.config.utils import get_backbone_type_from_cfg, get_model_type_from_cfg
from sleap_nn.data.identity import generate_class_maps, make_class_vectors
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
from sleap_nn.training.utils import is_distributed_initialized
from sleap_nn.config.get_config import get_aug_config


class BaseDataset(Dataset):
    """Base class for custom torch Datasets.

    Attributes:
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        user_instances_only: `True` if only user labeled instances should be used for training. If `False`,
            both user labeled and predicted instances would be used.
        ensure_rgb: (bool) True if the input image should have 3 channels (RGB image). If input has only one
        channel when this is set to `True`, then the images from single-channel
        is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. Default: `False`.
        ensure_grayscale: (bool) True if the input image should only have a single channel. If input has three channels (RGB) and this
        is set to True, then we convert the image to grayscale (single-channel)
        image. If the source image has only one channel and this is set to False, then we retain the single channel input. Default: `False`.
        intensity_aug: Intensity augmentation configuration. Can be:
            - String: One of ['uniform_noise', 'gaussian_noise', 'contrast', 'brightness']
            - List of strings: Multiple intensity augmentations from the allowed values
            - Dictionary: Custom intensity configuration
            - None: No intensity augmentation applied
        geometric_aug: Geometric augmentation configuration. Can be:
            - String: One of ['rotation', 'scale', 'translate', 'erase_scale', 'mixup']
            - List of strings: Multiple geometric augmentations from the allowed values
            - Dictionary: Custom geometric configuration
            - None: No geometric augmentation applied
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
        labels_list: List of `sio.Labels` objects. Used to store the labels in the cache. (only used if `cache_img` is `None`)
    """

    def __init__(
        self,
        labels: List[sio.Labels],
        max_stride: int,
        user_instances_only: bool = True,
        ensure_rgb: bool = False,
        ensure_grayscale: bool = False,
        intensity_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        geometric_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
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
        self.user_instances_only = user_instances_only
        self.ensure_rgb = ensure_rgb
        self.ensure_grayscale = ensure_grayscale

        # Handle intensity augmentation
        if intensity_aug is not None:
            if not isinstance(intensity_aug, DictConfig):
                intensity_aug = get_aug_config(intensity_aug=intensity_aug)
                config = OmegaConf.structured(intensity_aug)
                OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
                intensity_aug = DictConfig(config.intensity)
        self.intensity_aug = intensity_aug

        # Handle geometric augmentation
        if geometric_aug is not None:
            if not isinstance(geometric_aug, DictConfig):
                geometric_aug = get_aug_config(geometric_aug=geometric_aug)
                config = OmegaConf.structured(geometric_aug)
                OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
                geometric_aug = DictConfig(config.geometric)
        self.geometric_aug = geometric_aug
        self.curr_idx = 0
        self.max_stride = max_stride
        self.scale = scale
        self.apply_aug = apply_aug
        self.max_hw = max_hw
        self.rank = rank
        self.max_instances = 0
        for x in labels:
            max_instances = get_max_instances(x) if x else None

            if max_instances > self.max_instances:
                self.max_instances = max_instances

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

        self.lf_idx_list = self._get_lf_idx_list(labels)

        self.labels_list = None
        # this is to ensure that the labels are not passed to the multiprocessing pool when caching is enabled
        # (h5py objects can't be pickled error with num_workers > 0) in mac and windows
        if self.cache_img is None:
            self.labels_list = labels

        self.transform_to_pil = T.ToPILImage()
        self.transform_pil_to_tensor = T.ToTensor()
        self.cache = {}

        if self.cache_img is not None:
            if self.cache_img == "memory":
                self._fill_cache(labels)
            elif self.cache_img == "disk" and not self.use_existing_imgs:
                if self.rank is None or self.rank == -1 or self.rank == 0:
                    self._fill_cache(labels)
                # Synchronize all ranks after cache creation
                if is_distributed_initialized():
                    dist.barrier()

    def _get_lf_idx_list(self, labels: List[sio.Labels]) -> List[Tuple[int]]:
        """Return list of indices of labelled frames."""
        lf_idx_list = []
        for labels_idx, label in enumerate(labels):
            for lf_idx, lf in enumerate(label):
                # Filter to user instances
                if self.user_instances_only:
                    if lf.user_instances is not None and len(lf.user_instances) > 0:
                        lf.instances = lf.user_instances
                    else:
                        # Skip frames without user instances
                        continue
                is_empty = True
                for _, inst in enumerate(lf.instances):
                    if not inst.is_empty:  # filter all NaN instances.
                        is_empty = False
                if not is_empty:
                    video_idx = labels[labels_idx].videos.index(lf.video)
                    sample = {
                        "labels_idx": labels_idx,
                        "lf_idx": lf_idx,
                        "video_idx": video_idx,
                        "frame_idx": lf.frame_idx,
                        "instances": (
                            lf.instances if self.cache_img is not None else None
                        ),
                    }
                    lf_idx_list.append(sample)
                    # This is to ensure that the labels are not passed to the multiprocessing pool (h5py objects can't be pickled)
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

    def _fill_cache(self, labels: List[sio.Labels]):
        """Load all samples to cache."""
        # TODO: Implement parallel processing (using threads might cause error with MediaVideo backend)
        import os
        import sys

        total_samples = len(self.lf_idx_list)
        cache_type = "disk" if self.cache_img == "disk" else "memory"

        # Check for NO_COLOR env var or non-interactive terminal
        no_color = (
            os.environ.get("NO_COLOR") is not None
            or os.environ.get("FORCE_COLOR") == "0"
        )
        use_progress = sys.stdout.isatty() and not no_color

        def process_samples(progress=None, task=None):
            for sample in self.lf_idx_list:
                labels_idx = sample["labels_idx"]
                lf_idx = sample["lf_idx"]
                img = labels[labels_idx][lf_idx].image
                if img.shape[-1] == 1:
                    img = np.squeeze(img)
                if self.cache_img == "disk":
                    f_name = f"{self.cache_img_path}/sample_{labels_idx}_{lf_idx}.jpg"
                    Image.fromarray(img).save(f_name, format="JPEG")
                if self.cache_img == "memory":
                    self.cache[(labels_idx, lf_idx)] = img
                if progress is not None:
                    progress.update(task, advance=1)

        if use_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=Console(force_terminal=True),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"Caching images to {cache_type}", total=total_samples
                )
                process_samples(progress, task)
        else:
            logger.info(f"Caching {total_samples} images to {cache_type}...")
            process_samples()

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
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        user_instances_only: `True` if only user labeled instances should be used for training. If `False`,
            both user labeled and predicted instances would be used.
        ensure_rgb: (bool) True if the input image should have 3 channels (RGB image). If input has only one
        channel when this is set to `True`, then the images from single-channel
        is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. Default: `False`.
        ensure_grayscale: (bool) True if the input image should only have a single channel. If input has three channels (RGB) and this
        is set to True, then we convert the image to grayscale (single-channel)
        image. If the source image has only one channel and this is set to False, then we retain the single channel input. Default: `False`.
        intensity_aug: Intensity augmentation configuration. Can be:
            - String: One of ['uniform_noise', 'gaussian_noise', 'contrast', 'brightness']
            - List of strings: Multiple intensity augmentations from the allowed values
            - Dictionary: Custom intensity configuration
            - None: No intensity augmentation applied
        geometric_aug: Geometric augmentation configuration. Can be:
            - String: One of ['rotation', 'scale', 'translate', 'erase_scale', 'mixup']
            - List of strings: Multiple geometric augmentations from the allowed values
            - Dictionary: Custom geometric configuration
            - None: No geometric augmentation applied
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
        labels_list: List of `sio.Labels` objects. Used to store the labels in the cache. (only used if `cache_img` is `None`)
    """

    def __init__(
        self,
        labels: List[sio.Labels],
        confmap_head_config: DictConfig,
        pafs_head_config: DictConfig,
        max_stride: int,
        user_instances_only: bool = True,
        ensure_rgb: bool = False,
        ensure_grayscale: bool = False,
        intensity_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        geometric_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
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
            ensure_rgb=ensure_rgb,
            ensure_grayscale=ensure_grayscale,
            intensity_aug=intensity_aug,
            geometric_aug=geometric_aug,
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

        self.edge_inds = labels[0].skeletons[0].edge_inds

    def __getitem__(self, index) -> Dict:
        """Return dict with image, confmaps and pafs for given index."""
        sample = self.lf_idx_list[index]
        labels_idx = sample["labels_idx"]
        lf_idx = sample["lf_idx"]
        video_idx = sample["video_idx"]
        frame_idx = sample["frame_idx"]

        if self.cache_img is not None:
            instances = sample["instances"]
            if self.cache_img == "disk":
                img = np.array(
                    Image.open(
                        f"{self.cache_img_path}/sample_{labels_idx}_{lf_idx}.jpg"
                    )
                )
            elif self.cache_img == "memory":
                img = self.cache[(labels_idx, lf_idx)].copy()
        else:
            lf = self.labels_list[labels_idx][lf_idx]
            instances = lf.instances
            img = lf.image

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        # get dict
        sample = process_lf(
            instances_list=instances,
            img=img,
            frame_idx=frame_idx,
            video_idx=video_idx,
            max_instances=self.max_instances,
            user_instances_only=self.user_instances_only,
        )

        # apply normalization
        sample["image"] = apply_normalization(sample["image"])

        if self.ensure_rgb:
            sample["image"] = convert_to_rgb(sample["image"])
        elif self.ensure_grayscale:
            sample["image"] = convert_to_grayscale(sample["image"])

        # size matcher
        sample["image"], eff_scale = apply_sizematcher(
            sample["image"],
            max_height=self.max_hw[0],
            max_width=self.max_hw[1],
        )
        sample["instances"] = sample["instances"] * eff_scale
        sample["eff_scale"] = torch.tensor(eff_scale, dtype=torch.float32)

        # resize image
        sample["image"], sample["instances"] = apply_resizer(
            sample["image"],
            sample["instances"],
            scale=self.scale,
        )

        # Pad the image (if needed) according max stride
        sample["image"] = apply_pad_to_stride(
            sample["image"], max_stride=self.max_stride
        )

        # apply augmentation
        if self.apply_aug:
            if self.intensity_aug is not None:
                sample["image"], sample["instances"] = apply_intensity_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.intensity_aug,
                )

            if self.geometric_aug is not None:
                sample["image"], sample["instances"] = apply_geometric_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.geometric_aug,
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
        sample["labels_idx"] = labels_idx

        return sample


class BottomUpMultiClassDataset(BaseDataset):
    """Dataset class for bottom-up ID models.

    Attributes:
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        class_map_threshold: Minimum confidence map value below which map values will be
            replaced with zeros.
        user_instances_only: `True` if only user labeled instances should be used for training. If `False`,
            both user labeled and predicted instances would be used.
        ensure_rgb: (bool) True if the input image should have 3 channels (RGB image). If input has only one
        channel when this is set to `True`, then the images from single-channel
        is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. Default: `False`.
        ensure_grayscale: (bool) True if the input image should only have a single channel. If input has three channels (RGB) and this
        is set to True, then we convert the image to grayscale (single-channel)
        image. If the source image has only one channel and this is set to False, then we retain the single channel input. Default: `False`.
        intensity_aug: Intensity augmentation configuration. Can be:
            - String: One of ['uniform_noise', 'gaussian_noise', 'contrast', 'brightness']
            - List of strings: Multiple intensity augmentations from the allowed values
            - Dictionary: Custom intensity configuration
            - None: No intensity augmentation applied
        geometric_aug: Geometric augmentation configuration. Can be:
            - String: One of ['rotation', 'scale', 'translate', 'erase_scale', 'mixup']
            - List of strings: Multiple geometric augmentations from the allowed values
            - Dictionary: Custom geometric configuration
            - None: No geometric augmentation applied
        scale: Factor to resize the image dimensions by, specified as a float. Default: 1.0.
        apply_aug: `True` if augmentations should be applied to the data pipeline,
            else `False`. Default: `False`.
        max_hw: Maximum height and width of images across the labels file. If `max_height` and
           `max_width` in the config is None, then `max_hw` is used (computed with
            `sleap_nn.data.providers.get_max_height_width`). Else the values in the config
            are used.
        confmap_head_config: DictConfig object with all the keys in the `head_config` section.
            (required keys: `sigma`, `output_stride` and `anchor_part` depending on the model type ).
        class_maps_head_config: DictConfig object with all the keys in the `head_config` section
            (required keys: `sigma`, `output_stride` and `classes`)
            for class maps.
        cache_img: String to indicate which caching to use: `memory` or `disk`. If `None`,
            the images aren't cached and loaded from the `.slp` file on each access.
        cache_img_path: Path to save the `.jpg` files. If `None`, current working dir is used.
        use_existing_imgs: Use existing imgs/ chunks in the `cache_img_path`.
        rank: Indicates the rank of the process. Used during distributed training to ensure that image storage to
            disk occurs only once across all workers.
        labels_list: List of `sio.Labels` objects. Used to store the labels in the cache. (only used if `cache_img` is `None`)
    """

    def __init__(
        self,
        labels: List[sio.Labels],
        confmap_head_config: DictConfig,
        class_maps_head_config: DictConfig,
        max_stride: int,
        class_map_threshold: float = 0.2,
        user_instances_only: bool = True,
        ensure_rgb: bool = False,
        ensure_grayscale: bool = False,
        intensity_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        geometric_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
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
            ensure_rgb=ensure_rgb,
            ensure_grayscale=ensure_grayscale,
            intensity_aug=intensity_aug,
            geometric_aug=geometric_aug,
            scale=scale,
            apply_aug=apply_aug,
            max_hw=max_hw,
            cache_img=cache_img,
            cache_img_path=cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )
        self.confmap_head_config = confmap_head_config
        self.class_maps_head_config = class_maps_head_config

        self.class_names = self.class_maps_head_config.classes
        self.class_map_threshold = class_map_threshold

    def __getitem__(self, index) -> Dict:
        """Return dict with image, confmaps and pafs for given index."""
        sample = self.lf_idx_list[index]
        labels_idx = sample["labels_idx"]
        lf_idx = sample["lf_idx"]
        video_idx = sample["video_idx"]
        frame_idx = sample["frame_idx"]

        if self.cache_img is not None:
            instances = sample["instances"]
            if self.cache_img == "disk":
                img = np.array(
                    Image.open(
                        f"{self.cache_img_path}/sample_{labels_idx}_{lf_idx}.jpg"
                    )
                )
            elif self.cache_img == "memory":
                img = self.cache[(labels_idx, lf_idx)].copy()
        else:
            lf = self.labels_list[labels_idx][lf_idx]
            instances = lf.instances
            img = lf.image

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        # get dict
        sample = process_lf(
            instances_list=instances,
            img=img,
            frame_idx=frame_idx,
            video_idx=video_idx,
            max_instances=self.max_instances,
            user_instances_only=self.user_instances_only,
        )

        track_ids = torch.Tensor(
            [
                (
                    self.class_names.index(instances[idx].track.name)
                    if instances[idx].track is not None
                    else -1
                )
                for idx in range(sample["num_instances"])
            ]
        ).to(torch.int32)

        sample["num_tracks"] = torch.tensor(len(self.class_names), dtype=torch.int32)

        # apply normalization
        sample["image"] = apply_normalization(sample["image"])

        if self.ensure_rgb:
            sample["image"] = convert_to_rgb(sample["image"])
        elif self.ensure_grayscale:
            sample["image"] = convert_to_grayscale(sample["image"])

        # size matcher
        sample["image"], eff_scale = apply_sizematcher(
            sample["image"],
            max_height=self.max_hw[0],
            max_width=self.max_hw[1],
        )
        sample["instances"] = sample["instances"] * eff_scale
        sample["eff_scale"] = torch.tensor(eff_scale, dtype=torch.float32)

        # resize image
        sample["image"], sample["instances"] = apply_resizer(
            sample["image"],
            sample["instances"],
            scale=self.scale,
        )

        # Pad the image (if needed) according max stride
        sample["image"] = apply_pad_to_stride(
            sample["image"], max_stride=self.max_stride
        )

        # apply augmentation
        if self.apply_aug:
            if self.intensity_aug is not None:
                sample["image"], sample["instances"] = apply_intensity_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.intensity_aug,
                )

            if self.geometric_aug is not None:
                sample["image"], sample["instances"] = apply_geometric_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.geometric_aug,
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

        # class maps
        class_maps = generate_class_maps(
            instances=sample["instances"],
            img_hw=img_hw,
            num_instances=sample["num_instances"],
            class_inds=track_ids,
            num_tracks=sample["num_tracks"],
            class_map_threshold=self.class_map_threshold,
            sigma=self.class_maps_head_config.sigma,
            output_stride=self.class_maps_head_config.output_stride,
            is_centroids=False,
        )

        sample["confidence_maps"] = confidence_maps
        sample["class_maps"] = class_maps
        sample["labels_idx"] = labels_idx

        return sample


class CenteredInstanceDataset(BaseDataset):
    """Dataset class for instance-centered confidence map models.

    Attributes:
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        anchor_ind: Index of the node to use as the anchor point, based on its index in the
            ordered list of skeleton nodes.
        user_instances_only: `True` if only user labeled instances should be used for training. If `False`,
            both user labeled and predicted instances would be used.
        ensure_rgb: (bool) True if the input image should have 3 channels (RGB image). If input has only one
        channel when this is set to `True`, then the images from single-channel
        is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. Default: `False`.
        ensure_grayscale: (bool) True if the input image should only have a single channel. If input has three channels (RGB) and this
        is set to True, then we convert the image to grayscale (single-channel)
        image. If the source image has only one channel and this is set to False, then we retain the single channel input. Default: `False`.
        intensity_aug: Intensity augmentation configuration. Can be:
            - String: One of ['uniform_noise', 'gaussian_noise', 'contrast', 'brightness']
            - List of strings: Multiple intensity augmentations from the allowed values
            - Dictionary: Custom intensity configuration
            - None: No intensity augmentation applied
        geometric_aug: Geometric augmentation configuration. Can be:
            - String: One of ['rotation', 'scale', 'translate', 'erase_scale', 'mixup']
            - List of strings: Multiple geometric augmentations from the allowed values
            - Dictionary: Custom geometric configuration
            - None: No geometric augmentation applied
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
        crop_size: Crop size of each instance for centered-instance model. If `scale` is provided, then the cropped image will be resized according to `scale`.
        rank: Indicates the rank of the process. Used during distributed training to ensure that image storage to
            disk occurs only once across all workers.
        confmap_head_config: DictConfig object with all the keys in the `head_config` section.
            (required keys: `sigma`, `output_stride`, `part_names` and `anchor_part` depending on the model type ).
        labels_list: List of `sio.Labels` objects. Used to store the labels in the cache. (only used if `cache_img` is `None`)
    """

    def __init__(
        self,
        labels: List[sio.Labels],
        crop_size: int,
        confmap_head_config: DictConfig,
        max_stride: int,
        anchor_ind: Optional[int] = None,
        user_instances_only: bool = True,
        ensure_rgb: bool = False,
        ensure_grayscale: bool = False,
        intensity_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        geometric_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
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
            ensure_rgb=ensure_rgb,
            ensure_grayscale=ensure_grayscale,
            intensity_aug=intensity_aug,
            geometric_aug=geometric_aug,
            scale=scale,
            apply_aug=apply_aug,
            max_hw=max_hw,
            cache_img=cache_img,
            cache_img_path=cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )
        self.labels = None
        self.crop_size = crop_size
        self.anchor_ind = anchor_ind
        self.confmap_head_config = confmap_head_config
        self.instance_idx_list = self._get_instance_idx_list(labels)
        self.cache_lf = [None, None]

    def _get_instance_idx_list(self, labels: List[sio.Labels]) -> List[Tuple[int]]:
        """Return list of tuples with indices of labelled frames and instances."""
        instance_idx_list = []
        for labels_idx, label in enumerate(labels):
            for lf_idx, lf in enumerate(label):
                # Filter to user instances
                if self.user_instances_only:
                    if lf.user_instances is not None and len(lf.user_instances) > 0:
                        lf.instances = lf.user_instances
                    else:
                        # Skip frames without user instances
                        continue
                for inst_idx, inst in enumerate(lf.instances):
                    if not inst.is_empty:  # filter all NaN instances.
                        video_idx = labels[labels_idx].videos.index(lf.video)
                        sample = {
                            "labels_idx": labels_idx,
                            "lf_idx": lf_idx,
                            "inst_idx": inst_idx,
                            "video_idx": video_idx,
                            "instances": (
                                lf.instances if self.cache_img is not None else None
                            ),
                            "frame_idx": lf.frame_idx,
                        }
                        instance_idx_list.append(sample)
                        # This is to ensure that the labels are not passed to the multiprocessing pool (h5py objects can't be pickled)
        return instance_idx_list

    def __len__(self) -> int:
        """Return number of instances in the labels object."""
        return len(self.instance_idx_list)

    def __getitem__(self, index) -> Dict:
        """Return dict with cropped image and confmaps of instance for given index."""
        sample = self.instance_idx_list[index]
        labels_idx = sample["labels_idx"]
        lf_idx = sample["lf_idx"]
        inst_idx = sample["inst_idx"]
        video_idx = sample["video_idx"]
        lf_frame_idx = sample["frame_idx"]

        if self.cache_img is not None:
            instances_list = sample["instances"]
            if self.cache_img == "disk":
                img = np.array(
                    Image.open(
                        f"{self.cache_img_path}/sample_{labels_idx}_{lf_idx}.jpg"
                    )
                )
            elif self.cache_img == "memory":
                img = self.cache[(labels_idx, lf_idx)].copy()
        else:
            lf = self.labels_list[labels_idx][lf_idx]
            instances_list = lf.instances
            img = lf.image
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        image = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        instances = []
        for inst in instances_list:
            instances.append(
                inst.numpy()
            )  # no need to filter empty instances; handled while creating instance_idx_list
        instances = np.stack(instances, axis=0)

        # Add singleton time dimension for single frames.
        image = np.expand_dims(image, axis=0)  # (n_samples=1, C, H, W)
        instances = np.expand_dims(
            instances, axis=0
        )  # (n_samples=1, num_instances, num_nodes, 2)

        instances = torch.from_numpy(instances.astype("float32"))
        image = torch.from_numpy(image.copy())

        num_instances, _ = instances.shape[1:3]
        orig_img_height, orig_img_width = image.shape[-2:]

        instances = instances[:, inst_idx]

        # apply normalization
        image = apply_normalization(image)

        if self.ensure_rgb:
            image = convert_to_rgb(image)
        elif self.ensure_grayscale:
            image = convert_to_grayscale(image)

        # size matcher
        image, eff_scale = apply_sizematcher(
            image,
            max_height=self.max_hw[0],
            max_width=self.max_hw[1],
        )
        instances = instances * eff_scale

        # get the centroids based on the anchor idx
        centroids = generate_centroids(instances, anchor_ind=self.anchor_ind)

        instance, centroid = instances[0], centroids[0]  # (n_samples=1)

        crop_size = np.array([self.crop_size, self.crop_size]) * np.sqrt(
            2
        )  # crop extra for rotation augmentation
        crop_size = crop_size.astype(np.int32).tolist()

        sample = generate_crops(image, instance, centroid, crop_size)

        sample["frame_idx"] = torch.tensor(lf_frame_idx, dtype=torch.int32)
        sample["video_idx"] = torch.tensor(video_idx, dtype=torch.int32)
        sample["num_instances"] = num_instances
        sample["orig_size"] = torch.Tensor([orig_img_height, orig_img_width]).unsqueeze(
            0
        )
        sample["eff_scale"] = torch.tensor(eff_scale, dtype=torch.float32)

        # apply augmentation
        if self.apply_aug:
            if self.intensity_aug is not None:
                (
                    sample["instance_image"],
                    sample["instance"],
                ) = apply_intensity_augmentation(
                    sample["instance_image"],
                    sample["instance"],
                    **self.intensity_aug,
                )

            if self.geometric_aug is not None:
                (
                    sample["instance_image"],
                    sample["instance"],
                ) = apply_geometric_augmentation(
                    sample["instance_image"],
                    sample["instance"],
                    **self.geometric_aug,
                )

        # re-crop to original crop size
        sample["instance_bbox"] = torch.unsqueeze(
            make_centered_bboxes(sample["centroid"][0], self.crop_size, self.crop_size),
            0,
        )  # (n_samples=1, 4, 2)

        sample["instance_image"] = crop_and_resize(
            sample["instance_image"],
            boxes=sample["instance_bbox"],
            size=(self.crop_size, self.crop_size),
        )
        point = sample["instance_bbox"][0][0]
        center_instance = sample["instance"] - point
        centered_centroid = sample["centroid"] - point

        sample["instance"] = center_instance  # (n_samples=1, n_nodes, 2)
        sample["centroid"] = centered_centroid  # (n_samples=1, 2)

        # resize the cropped image
        sample["instance_image"], sample["instance"] = apply_resizer(
            sample["instance_image"],
            sample["instance"],
            scale=self.scale,
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
        sample["labels_idx"] = labels_idx

        return sample


class TopDownCenteredInstanceMultiClassDataset(CenteredInstanceDataset):
    """Dataset class for instance-centered confidence map ID models.

    Attributes:
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        anchor_ind: Index of the node to use as the anchor point, based on its index in the
            ordered list of skeleton nodes.
        user_instances_only: `True` if only user labeled instances should be used for training. If `False`,
            both user labeled and predicted instances would be used.
        ensure_rgb: (bool) True if the input image should have 3 channels (RGB image). If input has only one
        channel when this is set to `True`, then the images from single-channel
        is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. Default: `False`.
        ensure_grayscale: (bool) True if the input image should only have a single channel. If input has three channels (RGB) and this
        is set to True, then we convert the image to grayscale (single-channel)
        image. If the source image has only one channel and this is set to False, then we retain the single channel input. Default: `False`.
        intensity_aug: Intensity augmentation configuration. Can be:
            - String: One of ['uniform_noise', 'gaussian_noise', 'contrast', 'brightness']
            - List of strings: Multiple intensity augmentations from the allowed values
            - Dictionary: Custom intensity configuration
            - None: No intensity augmentation applied
        geometric_aug: Geometric augmentation configuration. Can be:
            - String: One of ['rotation', 'scale', 'translate', 'erase_scale', 'mixup']
            - List of strings: Multiple geometric augmentations from the allowed values
            - Dictionary: Custom geometric configuration
            - None: No geometric augmentation applied
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
        crop_size: Crop size of each instance for centered-instance model. If `scale` is provided, then the cropped image will be resized according to `scale`.
        rank: Indicates the rank of the process. Used during distributed training to ensure that image storage to
            disk occurs only once across all workers.
        confmap_head_config: DictConfig object with all the keys in the `head_config` section.
            (required keys: `sigma`, `output_stride`, `part_names` and `anchor_part` depending on the model type ).
        class_vectors_head_config: DictConfig object with all the keys in the `head_config` section.
            (required keys: `classes`, `num_fc_layers`, `num_fc_units`, `output_stride`, `loss_weight`).
        labels_list: List of `sio.Labels` objects. Used to store the labels in the cache. (only used if `cache_img` is `None`)
    """

    def __init__(
        self,
        labels: List[sio.Labels],
        crop_size: int,
        confmap_head_config: DictConfig,
        class_vectors_head_config: DictConfig,
        max_stride: int,
        anchor_ind: Optional[int] = None,
        user_instances_only: bool = True,
        ensure_rgb: bool = False,
        ensure_grayscale: bool = False,
        intensity_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        geometric_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
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
            crop_size=crop_size,
            confmap_head_config=confmap_head_config,
            max_stride=max_stride,
            anchor_ind=anchor_ind,
            user_instances_only=user_instances_only,
            ensure_rgb=ensure_rgb,
            ensure_grayscale=ensure_grayscale,
            intensity_aug=intensity_aug,
            geometric_aug=geometric_aug,
            scale=scale,
            apply_aug=apply_aug,
            max_hw=max_hw,
            cache_img=cache_img,
            cache_img_path=cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )
        self.class_vectors_head_config = class_vectors_head_config
        self.class_names = self.class_vectors_head_config.classes

    def __getitem__(self, index) -> Dict:
        """Return dict with cropped image and confmaps of instance for given index."""
        sample = self.instance_idx_list[index]
        labels_idx = sample["labels_idx"]
        lf_idx = sample["lf_idx"]
        inst_idx = sample["inst_idx"]
        video_idx = sample["video_idx"]
        lf_frame_idx = sample["frame_idx"]

        if self.cache_img is not None:
            instances_list = sample["instances"]
            if self.cache_img == "disk":
                img = np.array(
                    Image.open(
                        f"{self.cache_img_path}/sample_{labels_idx}_{lf_idx}.jpg"
                    )
                )
            elif self.cache_img == "memory":
                img = self.cache[(labels_idx, lf_idx)].copy()
        else:
            lf = self.labels_list[labels_idx][lf_idx]
            instances_list = lf.instances
            img = lf.image

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        image = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        instances = []
        for inst in instances_list:
            instances.append(
                inst.numpy()
            )  # no need to filter empty instance (handled while creating instance_idx)
        instances = np.stack(instances, axis=0)

        # Add singleton time dimension for single frames.
        image = np.expand_dims(image, axis=0)  # (n_samples=1, C, H, W)
        instances = np.expand_dims(
            instances, axis=0
        )  # (n_samples=1, num_instances, num_nodes, 2)

        instances = torch.from_numpy(instances.astype("float32"))
        image = torch.from_numpy(image.copy())

        num_instances, _ = instances.shape[1:3]
        orig_img_height, orig_img_width = image.shape[-2:]

        instances = instances[:, inst_idx]

        # apply normalization
        image = apply_normalization(image)

        if self.ensure_rgb:
            image = convert_to_rgb(image)
        elif self.ensure_grayscale:
            image = convert_to_grayscale(image)

        # size matcher
        image, eff_scale = apply_sizematcher(
            image,
            max_height=self.max_hw[0],
            max_width=self.max_hw[1],
        )
        instances = instances * eff_scale

        # get class vectors
        track_ids = torch.Tensor(
            [
                (
                    self.class_names.index(instances_list[idx].track.name)
                    if instances_list[idx].track is not None
                    else -1
                )
                for idx in range(num_instances)
            ]
        ).to(torch.int32)
        class_vectors = make_class_vectors(
            class_inds=track_ids,
            n_classes=torch.tensor(len(self.class_names), dtype=torch.int32),
        )

        # get the centroids based on the anchor idx
        centroids = generate_centroids(instances, anchor_ind=self.anchor_ind)

        instance, centroid = instances[0], centroids[0]  # (n_samples=1)

        crop_size = np.array([self.crop_size, self.crop_size]) * np.sqrt(
            2
        )  # crop extra for rotation augmentation
        crop_size = crop_size.astype(np.int32).tolist()

        sample = generate_crops(image, instance, centroid, crop_size)

        sample["frame_idx"] = torch.tensor(lf_frame_idx, dtype=torch.int32)
        sample["video_idx"] = torch.tensor(video_idx, dtype=torch.int32)
        sample["num_instances"] = num_instances
        sample["orig_size"] = torch.Tensor([orig_img_height, orig_img_width]).unsqueeze(
            0
        )
        sample["eff_scale"] = torch.tensor(eff_scale, dtype=torch.float32)

        # apply augmentation
        if self.apply_aug:
            if self.intensity_aug is not None:
                (
                    sample["instance_image"],
                    sample["instance"],
                ) = apply_intensity_augmentation(
                    sample["instance_image"],
                    sample["instance"],
                    **self.intensity_aug,
                )

            if self.geometric_aug is not None:
                (
                    sample["instance_image"],
                    sample["instance"],
                ) = apply_geometric_augmentation(
                    sample["instance_image"],
                    sample["instance"],
                    **self.geometric_aug,
                )

        # re-crop to original crop size
        sample["instance_bbox"] = torch.unsqueeze(
            make_centered_bboxes(sample["centroid"][0], self.crop_size, self.crop_size),
            0,
        )  # (n_samples=1, 4, 2)

        sample["instance_image"] = crop_and_resize(
            sample["instance_image"],
            boxes=sample["instance_bbox"],
            size=(self.crop_size, self.crop_size),
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
            scale=self.scale,
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

        sample["class_vectors"] = class_vectors[inst_idx].to(torch.float32)

        sample["confidence_maps"] = confidence_maps
        sample["labels_idx"] = labels_idx

        return sample


class CentroidDataset(BaseDataset):
    """Dataset class for centroid models.

    Attributes:
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        anchor_ind: Index of the node to use as the anchor point, based on its index in the
            ordered list of skeleton nodes.
        user_instances_only: `True` if only user labeled instances should be used for training. If `False`,
            both user labeled and predicted instances would be used.
        ensure_rgb: (bool) True if the input image should have 3 channels (RGB image). If input has only one
        channel when this is set to `True`, then the images from single-channel
        is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. Default: `False`.
        ensure_grayscale: (bool) True if the input image should only have a single channel. If input has three channels (RGB) and this
        is set to True, then we convert the image to grayscale (single-channel)
        image. If the source image has only one channel and this is set to False, then we retain the single channel input. Default: `False`.
        intensity_aug: Intensity augmentation configuration. Can be:
            - String: One of ['uniform_noise', 'gaussian_noise', 'contrast', 'brightness']
            - List of strings: Multiple intensity augmentations from the allowed values
            - Dictionary: Custom intensity configuration
            - None: No intensity augmentation applied
        geometric_aug: Geometric augmentation configuration. Can be:
            - String: One of ['rotation', 'scale', 'translate', 'erase_scale', 'mixup']
            - List of strings: Multiple geometric augmentations from the allowed values
            - Dictionary: Custom geometric configuration
            - None: No geometric augmentation applied
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
        labels_list: List of `sio.Labels` objects. Used to store the labels in the cache. (only used if `cache_img` is `None`)
    """

    def __init__(
        self,
        labels: List[sio.Labels],
        confmap_head_config: DictConfig,
        max_stride: int,
        anchor_ind: Optional[int] = None,
        user_instances_only: bool = True,
        ensure_rgb: bool = False,
        ensure_grayscale: bool = False,
        intensity_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        geometric_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
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
            ensure_rgb=ensure_rgb,
            ensure_grayscale=ensure_grayscale,
            intensity_aug=intensity_aug,
            geometric_aug=geometric_aug,
            scale=scale,
            apply_aug=apply_aug,
            max_hw=max_hw,
            cache_img=cache_img,
            cache_img_path=cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )
        self.anchor_ind = anchor_ind
        self.confmap_head_config = confmap_head_config

    def __getitem__(self, index) -> Dict:
        """Return dict with image and confmaps for centroids for given index."""
        sample = self.lf_idx_list[index]
        labels_idx = sample["labels_idx"]
        lf_idx = sample["lf_idx"]
        video_idx = sample["video_idx"]
        lf_frame_idx = sample["frame_idx"]

        if self.cache_img is not None:
            instances = sample["instances"]
            if self.cache_img == "disk":
                img = np.array(
                    Image.open(
                        f"{self.cache_img_path}/sample_{labels_idx}_{lf_idx}.jpg"
                    )
                )
            elif self.cache_img == "memory":
                img = self.cache[(labels_idx, lf_idx)].copy()
        else:
            lf = self.labels_list[labels_idx][lf_idx]
            instances = lf.instances
            img = lf.image
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        # get dict
        sample = process_lf(
            instances_list=instances,
            img=img,
            frame_idx=lf_frame_idx,
            video_idx=video_idx,
            max_instances=self.max_instances,
            user_instances_only=self.user_instances_only,
        )

        # apply normalization
        sample["image"] = apply_normalization(sample["image"])

        if self.ensure_rgb:
            sample["image"] = convert_to_rgb(sample["image"])
        elif self.ensure_grayscale:
            sample["image"] = convert_to_grayscale(sample["image"])

        # size matcher
        sample["image"], eff_scale = apply_sizematcher(
            sample["image"],
            max_height=self.max_hw[0],
            max_width=self.max_hw[1],
        )
        sample["instances"] = sample["instances"] * eff_scale
        sample["eff_scale"] = torch.tensor(eff_scale, dtype=torch.float32)

        # resize image
        sample["image"], sample["instances"] = apply_resizer(
            sample["image"],
            sample["instances"],
            scale=self.scale,
        )

        # get the centroids based on the anchor idx
        centroids = generate_centroids(sample["instances"], anchor_ind=self.anchor_ind)

        sample["centroids"] = centroids

        # Pad the image (if needed) according max stride
        sample["image"] = apply_pad_to_stride(
            sample["image"], max_stride=self.max_stride
        )

        # apply augmentation
        if self.apply_aug:
            if self.intensity_aug is not None:
                sample["image"], sample["centroids"] = apply_intensity_augmentation(
                    sample["image"],
                    sample["centroids"],
                    **self.intensity_aug,
                )

            if self.geometric_aug is not None:
                sample["image"], sample["centroids"] = apply_geometric_augmentation(
                    sample["image"],
                    sample["centroids"],
                    **self.geometric_aug,
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
        sample["labels_idx"] = labels_idx

        return sample


class SingleInstanceDataset(BaseDataset):
    """Dataset class for single-instance models.

    Attributes:
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.
        user_instances_only: `True` if only user labeled instances should be used for training. If `False`,
            both user labeled and predicted instances would be used.
        ensure_rgb: (bool) True if the input image should have 3 channels (RGB image). If input has only one
        channel when this is set to `True`, then the images from single-channel
        is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. Default: `False`.
        ensure_grayscale: (bool) True if the input image should only have a single channel. If input has three channels (RGB) and this
        is set to True, then we convert the image to grayscale (single-channel)
        image. If the source image has only one channel and this is set to False, then we retain the single channel input. Default: `False`.
        intensity_aug: Intensity augmentation configuration. Can be:
            - String: One of ['uniform_noise', 'gaussian_noise', 'contrast', 'brightness']
            - List of strings: Multiple intensity augmentations from the allowed values
            - Dictionary: Custom intensity configuration
            - None: No intensity augmentation applied
        geometric_aug: Geometric augmentation configuration. Can be:
            - String: One of ['rotation', 'scale', 'translate', 'erase_scale', 'mixup']
            - List of strings: Multiple geometric augmentations from the allowed values
            - Dictionary: Custom geometric configuration
            - None: No geometric augmentation applied
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
        (required keys: `sigma`, `output_stride` and `part_names` depending on the model type ).
        rank: Indicates the rank of the process. Used during distributed training to ensure that image storage to
            disk occurs only once across all workers.
        labels_list: List of `sio.Labels` objects. Used to store the labels in the cache. (only used if `cache_img` is `None`)
    """

    def __init__(
        self,
        labels: List[sio.Labels],
        confmap_head_config: DictConfig,
        max_stride: int,
        user_instances_only: bool = True,
        ensure_rgb: bool = False,
        ensure_grayscale: bool = False,
        intensity_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        geometric_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
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
            ensure_rgb=ensure_rgb,
            ensure_grayscale=ensure_grayscale,
            intensity_aug=intensity_aug,
            geometric_aug=geometric_aug,
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
        sample = self.lf_idx_list[index]
        labels_idx = sample["labels_idx"]
        lf_idx = sample["lf_idx"]
        video_idx = sample["video_idx"]
        lf_frame_idx = sample["frame_idx"]

        if self.cache_img is not None:
            instances = sample["instances"]
            if self.cache_img == "disk":
                img = np.array(
                    Image.open(
                        f"{self.cache_img_path}/sample_{labels_idx}_{lf_idx}.jpg"
                    )
                )
            elif self.cache_img == "memory":
                img = self.cache[(labels_idx, lf_idx)].copy()
        else:
            lf = self.labels_list[labels_idx][lf_idx]
            instances = lf.instances
            img = lf.image
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        # get dict
        sample = process_lf(
            instances_list=instances,
            img=img,
            frame_idx=lf_frame_idx,
            video_idx=video_idx,
            max_instances=self.max_instances,
            user_instances_only=self.user_instances_only,
        )

        # apply normalization
        sample["image"] = apply_normalization(sample["image"])

        if self.ensure_rgb:
            sample["image"] = convert_to_rgb(sample["image"])
        elif self.ensure_grayscale:
            sample["image"] = convert_to_grayscale(sample["image"])

        # size matcher
        sample["image"], eff_scale = apply_sizematcher(
            sample["image"],
            max_height=self.max_hw[0],
            max_width=self.max_hw[1],
        )
        sample["instances"] = sample["instances"] * eff_scale
        sample["eff_scale"] = torch.tensor(eff_scale, dtype=torch.float32)

        # resize image
        sample["image"], sample["instances"] = apply_resizer(
            sample["image"],
            sample["instances"],
            scale=self.scale,
        )

        # Pad the image (if needed) according max stride
        sample["image"] = apply_pad_to_stride(
            sample["image"], max_stride=self.max_stride
        )

        # apply augmentation
        if self.apply_aug:
            if self.intensity_aug is not None:
                sample["image"], sample["instances"] = apply_intensity_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.intensity_aug,
                )

            if self.geometric_aug is not None:
                sample["image"], sample["instances"] = apply_geometric_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.geometric_aug,
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
        sample["labels_idx"] = labels_idx

        return sample


class InfiniteDataLoader(DataLoader):
    """Dataloader that reuses workers for infinite iteration.

    This dataloader extends the PyTorch DataLoader to provide infinite recycling of workers, which improves efficiency
    for training loops that need to iterate through the dataset multiple times without recreating workers.

    Attributes:
        batch_sampler (_RepeatSampler): A sampler that repeats indefinitely.
        iterator (Iterator): The iterator from the parent DataLoader.
        len_dataloader (Optional[int]): Number of minibatches to be generated. If `None`, this is set to len(dataset)/batch_size.

    Methods:
        __len__: Return the length of the batch sampler's sampler.
        __iter__: Create a sampler that repeats indefinitely.
        __del__: Ensure workers are properly terminated.
        reset: Reset the iterator, useful when modifying dataset settings during training.

    Examples:
        Create an infinite dataloader for training
        >>> dataset = CenteredInstanceDataset(...)
        >>> dataloader = InfiniteDataLoader(dataset, batch_size=16, shuffle=True)
        >>> for batch in dataloader:  # Infinite iteration
        >>>     train_step(batch)

    Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/build.py
    """

    def __init__(self, len_dataloader: Optional[int] = None, *args: Any, **kwargs: Any):
        """Initialize the InfiniteDataLoader with the same arguments as DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()
        self.len_dataloader = len_dataloader

    def __len__(self) -> int:
        """Return the length of the batch sampler's sampler."""
        # set the len to required number of steps per epoch as Lightning Trainer
        # doesn't use the `__iter__` directly but instead uses the length to set
        # the number of steps per epoch. If this is just set to len(sampler), then
        # it only iterates through the samples in the dataset (and doesn't cycle through)
        # if the required steps per epoch is more than batches in dataset.
        return (
            self.len_dataloader
            if self.len_dataloader is not None
            else len(self.batch_sampler.sampler)
        )

    def __iter__(self) -> Iterator:
        """Create an iterator that yields indefinitely from the underlying iterator."""
        while True:
            yield next(self.iterator)

    def __del__(self):
        """Ensure that workers are properly terminated when the dataloader is deleted."""
        try:
            if not hasattr(self.iterator, "_workers"):
                return
            for w in self.iterator._workers:  # force terminate
                if w.is_alive():
                    w.terminate()
            self.iterator._shutdown_workers()  # cleanup
        except Exception:
            pass

    def reset(self):
        """Reset the iterator to allow modifications to the dataset during training."""
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """Sampler that repeats forever for infinite iteration.

    This sampler wraps another sampler and yields its contents indefinitely, allowing for infinite iteration
    over a dataset without recreating the sampler.

    Attributes:
        sampler (Dataset.sampler): The sampler to repeat.

    Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/build.py
    """

    def __init__(self, sampler: Any):
        """Initialize the _RepeatSampler with a sampler to repeat indefinitely."""
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        """Iterate over the sampler indefinitely, yielding its contents."""
        while True:
            yield from iter(self.sampler)


def get_train_val_datasets(
    train_labels: List[sio.Labels],
    val_labels: List[sio.Labels],
    config: DictConfig,
    rank: Optional[int] = None,
):
    """Return the train and val datasets.

    Args:
        train_labels: List of train labels.
        val_labels: List of val labels.
        config: Sleap-nn config.
        rank: Indicates the rank of the process. Used during distributed training to ensure that image storage to
            disk occurs only once across all workers.

    Returns:
        A tuple (train_dataset, val_dataset).
    """
    cache_imgs = (
        config.data_config.data_pipeline_fw.split("_")[-1]
        if "cache_img" in config.data_config.data_pipeline_fw
        else None
    )
    base_cache_img_path = config.data_config.cache_img_path
    train_cache_img_path, val_cache_img_path = None, None

    if cache_imgs == "disk":
        train_cache_img_path = Path(base_cache_img_path) / "train_imgs"
        val_cache_img_path = Path(base_cache_img_path) / "val_imgs"
    use_existing_imgs = config.data_config.use_existing_imgs

    model_type = get_model_type_from_cfg(config=config)
    backbone_type = get_backbone_type_from_cfg(config=config)

    if cache_imgs == "disk" and use_existing_imgs:
        if not (
            train_cache_img_path.exists()
            and train_cache_img_path.is_dir()
            and any(train_cache_img_path.glob("*.jpg"))
        ):
            message = f"There are no images in the path: {train_cache_img_path}"
            logger.error(message)
            raise Exception(message)

        if not (
            val_cache_img_path.exists()
            and val_cache_img_path.is_dir()
            and any(val_cache_img_path.glob("*.jpg"))
        ):
            message = f"There are no images in the path: {val_cache_img_path}"
            logger.error(message)
            raise Exception(message)

    if model_type == "bottomup":
        train_dataset = BottomUpDataset(
            labels=train_labels,
            confmap_head_config=config.model_config.head_configs.bottomup.confmaps,
            pafs_head_config=config.model_config.head_configs.bottomup.pafs,
            max_stride=config.model_config.backbone_config[f"{backbone_type}"][
                "max_stride"
            ],
            user_instances_only=config.data_config.user_instances_only,
            ensure_rgb=config.data_config.preprocessing.ensure_rgb,
            ensure_grayscale=config.data_config.preprocessing.ensure_grayscale,
            intensity_aug=(
                config.data_config.augmentation_config.intensity
                if config.data_config.augmentation_config is not None
                else None
            ),
            geometric_aug=(
                config.data_config.augmentation_config.geometric
                if config.data_config.augmentation_config is not None
                else None
            ),
            scale=config.data_config.preprocessing.scale,
            apply_aug=config.data_config.use_augmentations_train,
            max_hw=(
                config.data_config.preprocessing.max_height,
                config.data_config.preprocessing.max_width,
            ),
            cache_img=cache_imgs,
            cache_img_path=train_cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )
        val_dataset = BottomUpDataset(
            labels=val_labels,
            confmap_head_config=config.model_config.head_configs.bottomup.confmaps,
            pafs_head_config=config.model_config.head_configs.bottomup.pafs,
            max_stride=config.model_config.backbone_config[f"{backbone_type}"][
                "max_stride"
            ],
            user_instances_only=config.data_config.user_instances_only,
            ensure_rgb=config.data_config.preprocessing.ensure_rgb,
            ensure_grayscale=config.data_config.preprocessing.ensure_grayscale,
            intensity_aug=None,
            geometric_aug=None,
            scale=config.data_config.preprocessing.scale,
            apply_aug=False,
            max_hw=(
                config.data_config.preprocessing.max_height,
                config.data_config.preprocessing.max_width,
            ),
            cache_img=cache_imgs,
            cache_img_path=val_cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )

    elif model_type == "multi_class_bottomup":
        train_dataset = BottomUpMultiClassDataset(
            labels=train_labels,
            confmap_head_config=config.model_config.head_configs.multi_class_bottomup.confmaps,
            class_maps_head_config=config.model_config.head_configs.multi_class_bottomup.class_maps,
            max_stride=config.model_config.backbone_config[f"{backbone_type}"][
                "max_stride"
            ],
            user_instances_only=config.data_config.user_instances_only,
            ensure_rgb=config.data_config.preprocessing.ensure_rgb,
            ensure_grayscale=config.data_config.preprocessing.ensure_grayscale,
            intensity_aug=(
                config.data_config.augmentation_config.intensity
                if config.data_config.augmentation_config is not None
                else None
            ),
            geometric_aug=(
                config.data_config.augmentation_config.geometric
                if config.data_config.augmentation_config is not None
                else None
            ),
            scale=config.data_config.preprocessing.scale,
            apply_aug=config.data_config.use_augmentations_train,
            max_hw=(
                config.data_config.preprocessing.max_height,
                config.data_config.preprocessing.max_width,
            ),
            cache_img=cache_imgs,
            cache_img_path=train_cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )
        val_dataset = BottomUpMultiClassDataset(
            labels=val_labels,
            confmap_head_config=config.model_config.head_configs.multi_class_bottomup.confmaps,
            class_maps_head_config=config.model_config.head_configs.multi_class_bottomup.class_maps,
            max_stride=config.model_config.backbone_config[f"{backbone_type}"][
                "max_stride"
            ],
            user_instances_only=config.data_config.user_instances_only,
            ensure_rgb=config.data_config.preprocessing.ensure_rgb,
            ensure_grayscale=config.data_config.preprocessing.ensure_grayscale,
            intensity_aug=None,
            geometric_aug=None,
            scale=config.data_config.preprocessing.scale,
            apply_aug=False,
            max_hw=(
                config.data_config.preprocessing.max_height,
                config.data_config.preprocessing.max_width,
            ),
            cache_img=cache_imgs,
            cache_img_path=val_cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )

    elif model_type == "centered_instance":
        nodes = config.model_config.head_configs.centered_instance.confmaps.part_names
        anchor_part = (
            config.model_config.head_configs.centered_instance.confmaps.anchor_part
        )
        anchor_ind = nodes.index(anchor_part) if anchor_part is not None else None
        train_dataset = CenteredInstanceDataset(
            labels=train_labels,
            confmap_head_config=config.model_config.head_configs.centered_instance.confmaps,
            max_stride=config.model_config.backbone_config[f"{backbone_type}"][
                "max_stride"
            ],
            anchor_ind=anchor_ind,
            user_instances_only=config.data_config.user_instances_only,
            ensure_rgb=config.data_config.preprocessing.ensure_rgb,
            ensure_grayscale=config.data_config.preprocessing.ensure_grayscale,
            intensity_aug=(
                config.data_config.augmentation_config.intensity
                if config.data_config.augmentation_config is not None
                else None
            ),
            geometric_aug=(
                config.data_config.augmentation_config.geometric
                if config.data_config.augmentation_config is not None
                else None
            ),
            scale=config.data_config.preprocessing.scale,
            apply_aug=config.data_config.use_augmentations_train,
            crop_size=config.data_config.preprocessing.crop_size,
            max_hw=(
                config.data_config.preprocessing.max_height,
                config.data_config.preprocessing.max_width,
            ),
            cache_img=cache_imgs,
            cache_img_path=train_cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )
        val_dataset = CenteredInstanceDataset(
            labels=val_labels,
            confmap_head_config=config.model_config.head_configs.centered_instance.confmaps,
            max_stride=config.model_config.backbone_config[f"{backbone_type}"][
                "max_stride"
            ],
            anchor_ind=anchor_ind,
            user_instances_only=config.data_config.user_instances_only,
            ensure_rgb=config.data_config.preprocessing.ensure_rgb,
            ensure_grayscale=config.data_config.preprocessing.ensure_grayscale,
            intensity_aug=None,
            geometric_aug=None,
            scale=config.data_config.preprocessing.scale,
            apply_aug=False,
            crop_size=config.data_config.preprocessing.crop_size,
            max_hw=(
                config.data_config.preprocessing.max_height,
                config.data_config.preprocessing.max_width,
            ),
            cache_img=cache_imgs,
            cache_img_path=val_cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )

    elif model_type == "multi_class_topdown":
        nodes = config.model_config.head_configs.multi_class_topdown.confmaps.part_names
        anchor_part = (
            config.model_config.head_configs.multi_class_topdown.confmaps.anchor_part
        )
        anchor_ind = nodes.index(anchor_part) if anchor_part is not None else None
        train_dataset = TopDownCenteredInstanceMultiClassDataset(
            labels=train_labels,
            confmap_head_config=config.model_config.head_configs.multi_class_topdown.confmaps,
            class_vectors_head_config=config.model_config.head_configs.multi_class_topdown.class_vectors,
            max_stride=config.model_config.backbone_config[f"{backbone_type}"][
                "max_stride"
            ],
            anchor_ind=anchor_ind,
            user_instances_only=config.data_config.user_instances_only,
            ensure_rgb=config.data_config.preprocessing.ensure_rgb,
            ensure_grayscale=config.data_config.preprocessing.ensure_grayscale,
            intensity_aug=(
                config.data_config.augmentation_config.intensity
                if config.data_config.augmentation_config is not None
                else None
            ),
            geometric_aug=(
                config.data_config.augmentation_config.geometric
                if config.data_config.augmentation_config is not None
                else None
            ),
            scale=config.data_config.preprocessing.scale,
            apply_aug=config.data_config.use_augmentations_train,
            crop_size=config.data_config.preprocessing.crop_size,
            max_hw=(
                config.data_config.preprocessing.max_height,
                config.data_config.preprocessing.max_width,
            ),
            cache_img=cache_imgs,
            cache_img_path=train_cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )
        val_dataset = TopDownCenteredInstanceMultiClassDataset(
            labels=val_labels,
            confmap_head_config=config.model_config.head_configs.multi_class_topdown.confmaps,
            class_vectors_head_config=config.model_config.head_configs.multi_class_topdown.class_vectors,
            max_stride=config.model_config.backbone_config[f"{backbone_type}"][
                "max_stride"
            ],
            anchor_ind=anchor_ind,
            user_instances_only=config.data_config.user_instances_only,
            ensure_rgb=config.data_config.preprocessing.ensure_rgb,
            ensure_grayscale=config.data_config.preprocessing.ensure_grayscale,
            intensity_aug=None,
            geometric_aug=None,
            scale=config.data_config.preprocessing.scale,
            apply_aug=False,
            crop_size=config.data_config.preprocessing.crop_size,
            max_hw=(
                config.data_config.preprocessing.max_height,
                config.data_config.preprocessing.max_width,
            ),
            cache_img=cache_imgs,
            cache_img_path=val_cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )

    elif model_type == "centroid":
        nodes = [x["name"] for x in config.data_config.skeletons[0]["nodes"]]
        anchor_part = config.model_config.head_configs.centroid.confmaps.anchor_part
        anchor_ind = nodes.index(anchor_part) if anchor_part is not None else None
        train_dataset = CentroidDataset(
            labels=train_labels,
            confmap_head_config=config.model_config.head_configs.centroid.confmaps,
            max_stride=config.model_config.backbone_config[f"{backbone_type}"][
                "max_stride"
            ],
            anchor_ind=anchor_ind,
            user_instances_only=config.data_config.user_instances_only,
            ensure_rgb=config.data_config.preprocessing.ensure_rgb,
            ensure_grayscale=config.data_config.preprocessing.ensure_grayscale,
            intensity_aug=(
                config.data_config.augmentation_config.intensity
                if config.data_config.augmentation_config is not None
                else None
            ),
            geometric_aug=(
                config.data_config.augmentation_config.geometric
                if config.data_config.augmentation_config is not None
                else None
            ),
            scale=config.data_config.preprocessing.scale,
            apply_aug=config.data_config.use_augmentations_train,
            max_hw=(
                config.data_config.preprocessing.max_height,
                config.data_config.preprocessing.max_width,
            ),
            cache_img=cache_imgs,
            cache_img_path=train_cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )
        val_dataset = CentroidDataset(
            labels=val_labels,
            confmap_head_config=config.model_config.head_configs.centroid.confmaps,
            max_stride=config.model_config.backbone_config[f"{backbone_type}"][
                "max_stride"
            ],
            anchor_ind=anchor_ind,
            user_instances_only=config.data_config.user_instances_only,
            ensure_rgb=config.data_config.preprocessing.ensure_rgb,
            ensure_grayscale=config.data_config.preprocessing.ensure_grayscale,
            intensity_aug=None,
            geometric_aug=None,
            scale=config.data_config.preprocessing.scale,
            apply_aug=False,
            max_hw=(
                config.data_config.preprocessing.max_height,
                config.data_config.preprocessing.max_width,
            ),
            cache_img=cache_imgs,
            cache_img_path=val_cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )

    else:
        train_dataset = SingleInstanceDataset(
            labels=train_labels,
            confmap_head_config=config.model_config.head_configs.single_instance.confmaps,
            max_stride=config.model_config.backbone_config[f"{backbone_type}"][
                "max_stride"
            ],
            user_instances_only=config.data_config.user_instances_only,
            ensure_rgb=config.data_config.preprocessing.ensure_rgb,
            ensure_grayscale=config.data_config.preprocessing.ensure_grayscale,
            intensity_aug=(
                config.data_config.augmentation_config.intensity
                if config.data_config.augmentation_config is not None
                else None
            ),
            geometric_aug=(
                config.data_config.augmentation_config.geometric
                if config.data_config.augmentation_config is not None
                else None
            ),
            scale=config.data_config.preprocessing.scale,
            apply_aug=config.data_config.use_augmentations_train,
            max_hw=(
                config.data_config.preprocessing.max_height,
                config.data_config.preprocessing.max_width,
            ),
            cache_img=cache_imgs,
            cache_img_path=train_cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )
        val_dataset = SingleInstanceDataset(
            labels=val_labels,
            confmap_head_config=config.model_config.head_configs.single_instance.confmaps,
            max_stride=config.model_config.backbone_config[f"{backbone_type}"][
                "max_stride"
            ],
            user_instances_only=config.data_config.user_instances_only,
            ensure_rgb=config.data_config.preprocessing.ensure_rgb,
            ensure_grayscale=config.data_config.preprocessing.ensure_grayscale,
            intensity_aug=None,
            geometric_aug=None,
            scale=config.data_config.preprocessing.scale,
            apply_aug=False,
            max_hw=(
                config.data_config.preprocessing.max_height,
                config.data_config.preprocessing.max_width,
            ),
            cache_img=cache_imgs,
            cache_img_path=val_cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
        )

    # If using caching, close the videos to prevent `h5py objects can't be pickled error` when num_workers > 0.
    if "cache_img" in config.data_config.data_pipeline_fw:
        for train, val in zip(train_labels, val_labels):
            for video in train.videos:
                if video.is_open:
                    video.close()
            for video in val.videos:
                if video.is_open:
                    video.close()

    return train_dataset, val_dataset


def get_train_val_dataloaders(
    train_dataset: BaseDataset,
    val_dataset: BaseDataset,
    config: DictConfig,
    train_steps_per_epoch: Optional[int] = None,
    val_steps_per_epoch: Optional[int] = None,
    rank: Optional[int] = None,
    trainer_devices: int = 1,
):
    """Return the train and val dataloaders.

    Args:
        train_dataset: Train dataset-instance of one of the dataset classes [SingleInstanceDataset, CentroidDataset, CenteredInstanceDataset, BottomUpDataset, BottomUpMultiClassDataset, TopDownCenteredInstanceMultiClassDataset].
        val_dataset: Val dataset-instance of one of the dataset classes [SingleInstanceDataset, CentroidDataset, CenteredInstanceDataset, BottomUpDataset, BottomUpMultiClassDataset, TopDownCenteredInstanceMultiClassDataset].
        config: Sleap-nn config.
        train_steps_per_epoch: Number of minibatches (steps) to train for in an epoch. If set to `None`, this is set to the number of batches in the training data. **Note**: In a multi-gpu training setup, the effective steps during training would be the `trainer_steps_per_epoch` / `trainer_devices`.
        val_steps_per_epoch: Number of minibatches (steps) to run validation for in an epoch. If set to `None`, this is set to the number of batches in the val data.
        rank: Indicates the rank of the process. Used during distributed training to ensure that image storage to
            disk occurs only once across all workers.
        trainer_devices: Number of devices to use for training.

    Returns:
        A tuple (train_dataloader, val_dataloader).
    """
    pin_memory = (
        config.trainer_config.train_data_loader.pin_memory
        if "pin_memory" in config.trainer_config.train_data_loader
        and config.trainer_config.train_data_loader.pin_memory is not None
        else True
    )

    if train_steps_per_epoch is None:
        train_steps_per_epoch = config.trainer_config.train_steps_per_epoch
        if train_steps_per_epoch is None:
            train_steps_per_epoch = get_steps_per_epoch(
                dataset=train_dataset,
                batch_size=config.trainer_config.train_data_loader.batch_size,
            )

    if val_steps_per_epoch is None:
        val_steps_per_epoch = get_steps_per_epoch(
            dataset=val_dataset,
            batch_size=config.trainer_config.val_data_loader.batch_size,
        )

    train_sampler = (
        DistributedSampler(
            dataset=train_dataset,
            shuffle=config.trainer_config.train_data_loader.shuffle,
            rank=rank if rank is not None else 0,
            num_replicas=trainer_devices,
        )
        if trainer_devices > 1
        else None
    )

    train_data_loader = InfiniteDataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        len_dataloader=max(1, round(train_steps_per_epoch / trainer_devices)),
        shuffle=(
            config.trainer_config.train_data_loader.shuffle
            if train_sampler is None
            else None
        ),
        batch_size=config.trainer_config.train_data_loader.batch_size,
        num_workers=config.trainer_config.train_data_loader.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            True if config.trainer_config.train_data_loader.num_workers > 0 else None
        ),
        prefetch_factor=(
            config.trainer_config.train_data_loader.batch_size
            if config.trainer_config.train_data_loader.num_workers > 0
            else None
        ),
    )

    val_sampler = (
        DistributedSampler(
            dataset=val_dataset,
            shuffle=False,
            rank=rank if rank is not None else 0,
            num_replicas=trainer_devices,
        )
        if trainer_devices > 1
        else None
    )
    val_data_loader = InfiniteDataLoader(
        dataset=val_dataset,
        shuffle=False if val_sampler is None else None,
        sampler=val_sampler,
        len_dataloader=(
            max(1, round(val_steps_per_epoch / trainer_devices))
            if trainer_devices > 1
            else None
        ),
        batch_size=config.trainer_config.val_data_loader.batch_size,
        num_workers=config.trainer_config.val_data_loader.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            True if config.trainer_config.val_data_loader.num_workers > 0 else None
        ),
        prefetch_factor=(
            config.trainer_config.val_data_loader.batch_size
            if config.trainer_config.val_data_loader.num_workers > 0
            else None
        ),
    )

    return train_data_loader, val_data_loader


def get_steps_per_epoch(dataset: BaseDataset, batch_size: int):
    """Compute the number of steps (iterations) per epoch for the given dataset."""
    return (len(dataset) // batch_size) + (1 if (len(dataset) % batch_size) else 0)
