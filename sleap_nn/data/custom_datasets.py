"""Custom `torch.utils.data.Dataset`s for different model types."""

from sleap_nn.data.skia_augmentation import crop_and_resize_skia as crop_and_resize

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
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
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import sleap_io as sio
from sleap_nn.config.utils import get_backbone_type_from_cfg, get_model_type_from_cfg
from sleap_nn.data.identity import generate_class_maps, make_class_vectors
from sleap_nn.data.instance_centroids import generate_centroids
from sleap_nn.data.instance_cropping import generate_crops
from sleap_nn.data.normalization import (
    convert_to_grayscale,
    convert_to_rgb,
)
from sleap_nn.data.providers import (
    filter_oob_points,
    get_max_instances,
    get_max_height_width,
    process_lf,
    process_negative_lf,
)
from sleap_nn.data.resizing import apply_pad_to_stride, apply_sizematcher, apply_resizer
from sleap_nn.data.augmentation import (
    apply_geometric_augmentation,
    apply_intensity_augmentation,
)
from sleap_nn.data.utils import get_symmetric_inds
from sleap_nn.data.confidence_maps import generate_confmaps, generate_multiconfmaps
from sleap_nn.data.edge_maps import generate_pafs
from sleap_nn.data.segmentation_maps import (
    _compute_mask_centroids,
    generate_foreground_mask,
    generate_center_heatmap,
    generate_center_offsets,
)
from sleap_nn.data.instance_cropping import make_centered_bboxes
from sleap_nn.training.utils import is_distributed_initialized
from sleap_nn.config.get_config import get_aug_config

# Minimum number of samples to use parallel caching (overhead not worth it for smaller)
MIN_SAMPLES_FOR_PARALLEL_CACHING = 20


class ParallelCacheFiller:
    """Parallel implementation of image caching using thread-local video copies.

    This class uses ThreadPoolExecutor to parallelize I/O-bound operations when
    caching images to disk or memory. Each worker thread gets its own copy of
    video objects to ensure thread safety.

    Attributes:
        labels: List of sio.Labels objects containing the data.
        lf_idx_list: List of dictionaries with labeled frame indices.
        cache_type: Either "disk" or "memory".
        cache_path: Path to save cached images (for disk caching).
        num_workers: Number of worker threads.
    """

    def __init__(
        self,
        labels: List[sio.Labels],
        lf_idx_list: List[Dict],
        cache_type: str,
        cache_path: Optional[Path] = None,
        num_workers: int = 4,
    ):
        """Initialize the parallel cache filler.

        Args:
            labels: List of sio.Labels objects.
            lf_idx_list: List of sample dictionaries with frame indices.
            cache_type: Either "disk" or "memory".
            cache_path: Path for disk caching.
            num_workers: Number of worker threads.
        """
        self.labels = labels
        self.lf_idx_list = lf_idx_list
        self.cache_type = cache_type
        self.cache_path = cache_path
        self.num_workers = num_workers

        self.cache: Dict = {}
        self._cache_lock = threading.Lock()
        self._local = threading.local()
        self._video_info: Dict = {}

        # Prepare video copies for thread-local access
        self._prepare_video_copies()

    def _prepare_video_copies(self):
        """Close original videos and prepare for thread-local copies."""
        for label in self.labels:
            for video in label.videos:
                vid_id = id(video)
                if vid_id not in self._video_info:
                    # Store original state
                    original_open_backend = video.open_backend

                    # Close the video backend
                    video.close()
                    video.open_backend = False

                    self._video_info[vid_id] = {
                        "video": video,
                        "original_open_backend": original_open_backend,
                    }

    def _get_thread_local_video(self, video: sio.Video) -> sio.Video:
        """Get or create a thread-local video copy.

        Args:
            video: The original video object.

        Returns:
            A thread-local copy of the video that is safe to use.
        """
        vid_id = id(video)

        if not hasattr(self._local, "videos"):
            self._local.videos = {}

        if vid_id not in self._local.videos:
            # Create a thread-local copy
            video_copy = deepcopy(video)
            video_copy.open_backend = True
            self._local.videos[vid_id] = video_copy

        return self._local.videos[vid_id]

    def _process_sample(
        self, sample: Dict
    ) -> Tuple[int, int, Optional[np.ndarray], Optional[str]]:
        """Process a single sample (read image, optionally save/cache).

        Args:
            sample: Dictionary with labels_idx, lf_idx, etc.

        Returns:
            Tuple of (labels_idx, lf_idx, image_or_none, error_or_none).
        """
        labels_idx = sample["labels_idx"]
        lf_idx = sample["lf_idx"]

        try:
            if sample.get("is_negative", False):
                # Negative frames: read directly from video by index
                video_idx = sample["video_idx"]
                frame_idx = sample["frame_idx"]
                video = self._get_thread_local_video(
                    self.labels[labels_idx].videos[video_idx]
                )
                img = video[frame_idx]
            else:
                # Positive frames: read from labeled frame
                lf = self.labels[labels_idx][lf_idx]
                video = self._get_thread_local_video(lf.video)
                img = video[lf.frame_idx]

            if img.shape[-1] == 1:
                img = np.squeeze(img)

            if self.cache_type == "disk":
                f_name = self.cache_path / f"sample_{labels_idx}_{lf_idx}.jpg"
                Image.fromarray(img).save(str(f_name), format="JPEG")
                return labels_idx, lf_idx, None, None
            elif self.cache_type == "memory":
                return labels_idx, lf_idx, img, None

        except Exception as e:
            return labels_idx, lf_idx, None, f"{type(e).__name__}: {str(e)}"

    def fill_cache(
        self, progress_callback=None
    ) -> Tuple[Dict, List[Tuple[int, int, str]]]:
        """Fill the cache in parallel.

        Args:
            progress_callback: Optional callback(completed_count) for progress updates.

        Returns:
            Tuple of (cache_dict, list_of_errors).
        """
        errors = []
        completed = 0

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._process_sample, sample): sample
                for sample in self.lf_idx_list
            }

            for future in as_completed(futures):
                labels_idx, lf_idx, img, error = future.result()

                if error:
                    errors.append((labels_idx, lf_idx, error))
                elif self.cache_type == "memory" and img is not None:
                    with self._cache_lock:
                        self.cache[(labels_idx, lf_idx)] = img

                completed += 1
                if progress_callback:
                    progress_callback(completed)

        # Restore original video states
        self._restore_videos()

        return self.cache, errors

    def _restore_videos(self):
        """Restore original video states after caching is complete."""
        for vid_info in self._video_info.values():
            video = vid_info["video"]
            video.open_backend = vid_info["original_open_backend"]
            if video.open_backend:
                try:
                    video.open()
                except Exception:
                    pass


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
        parallel_caching: If True, use parallel processing for caching (faster for large datasets). Default: True.
        cache_workers: Number of worker threads for parallel caching. If 0, uses min(4, cpu_count). Default: 0.
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
        parallel_caching: bool = True,
        cache_workers: int = 0,
        use_negative_frames: bool = False,
    ) -> None:
        """Initialize class attributes."""
        super().__init__()
        self.user_instances_only = user_instances_only
        self.use_negative_frames = use_negative_frames
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

        # Store num_nodes for negative frame generation. Guard against mask-only
        # labels (e.g. instance segmentation) that may carry no skeleton.
        self.num_nodes = (
            len(labels[0].skeletons[0].nodes) if labels and labels[0].skeletons else 0
        )

        # Resolve symmetric node-index pairs once for flip augmentation. After
        # mirroring an image, left/right symmetric parts must be swapped to keep
        # labels correct. Read from the raw skeleton (version-proof; see
        # `get_symmetric_inds`).
        self.symmetric_inds = (
            get_symmetric_inds(labels[0].skeletons[0])
            if labels and labels[0].skeletons
            else []
        )
        # Warn about the silent correctness footgun: flipping a left/right
        # asymmetric skeleton without symmetries teaches the model wrong labels.
        flip_p = (
            self.geometric_aug.get("flip_p", 0.0)
            if self.geometric_aug is not None
            else 0.0
        )
        if self.apply_aug and flip_p and flip_p > 0 and not self.symmetric_inds:
            logger.warning(
                "Flip augmentation is enabled (flip_p > 0) but the skeleton has no "
                "symmetries. Flipping will not swap any nodes, which is only correct "
                "if the labeled animal is truly left/right symmetric. Add symmetry "
                "pairs to the skeleton to fix this."
            )

        self.cache_img = cache_img
        self.cache_img_path = cache_img_path
        self.use_existing_imgs = use_existing_imgs
        self.parallel_caching = parallel_caching
        self.cache_workers = cache_workers
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
                self._fill_cache(
                    labels,
                    parallel=self.parallel_caching,
                    num_workers=self.cache_workers,
                )
            elif self.cache_img == "disk" and not self.use_existing_imgs:
                if self.rank is None or self.rank == -1 or self.rank == 0:
                    self._fill_cache(
                        labels,
                        parallel=self.parallel_caching,
                        num_workers=self.cache_workers,
                    )
                # Synchronize all ranks after cache creation
                if is_distributed_initialized():
                    dist.barrier()

    def _get_lf_idx_list(self, labels: List[sio.Labels]) -> List[Tuple[int]]:
        """Return list of indices of labelled frames (and optionally negative frames).

        If ``self.use_negative_frames`` is True, all user-confirmed negative
        frames (``labels.negative_frames``) are appended to the sample list.
        """
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
                        "is_negative": False,
                        "instances": (
                            lf.instances if self.cache_img is not None else None
                        ),
                    }
                    lf_idx_list.append(sample)
                    # This is to ensure that the labels are not passed to the multiprocessing pool (h5py objects can't be pickled)

        # Add negative frames if requested
        if self.use_negative_frames and len(lf_idx_list) > 0:
            neg_samples = self._collect_negative_frames(labels)
            if neg_samples:
                n_positive = len(lf_idx_list)
                lf_idx_list.extend(neg_samples)
                logger.info(
                    f"Added {len(neg_samples)} negative samples "
                    f"to {n_positive} positive samples."
                )

        return lf_idx_list

    def _collect_negative_frames(
        self,
        labels: List[sio.Labels],
    ) -> List[Dict]:
        """Collect all user-confirmed negative frames from labels.

        Only frames explicitly marked by the user as negative are used
        (``LabeledFrame`` objects with ``is_negative=True``, accessed via
        ``labels.negative_frames``).  Unlabeled frames are **not** included
        because they may contain animals that simply haven't been annotated yet.

        Args:
            labels: List of sio.Labels objects.

        Returns:
            List of sample dicts with ``is_negative=True`` (one per unique
            negative frame).
        """
        neg_samples: List[Dict] = []

        for labels_idx, label in enumerate(labels):
            if not hasattr(label, "negative_frames"):
                continue
            for idx, lf in enumerate(label.negative_frames):
                video_idx = label.videos.index(lf.video)
                neg_samples.append(
                    {
                        "labels_idx": labels_idx,
                        "lf_idx": f"neg_{labels_idx}_{idx}",
                        "video_idx": video_idx,
                        "frame_idx": lf.frame_idx,
                        "is_negative": True,
                        "instances": None,
                    }
                )

        return neg_samples

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

    def _fill_cache(
        self,
        labels: List[sio.Labels],
        parallel: bool = True,
        num_workers: int = 0,
    ):
        """Load all samples to cache.

        Args:
            labels: List of sio.Labels objects containing the data.
            parallel: If True, use parallel processing for caching (faster for large
                datasets). Default: True.
            num_workers: Number of worker threads for parallel caching. If 0, uses
                min(4, cpu_count). Default: 0.
        """
        total_samples = len(self.lf_idx_list)
        cache_type = "disk" if self.cache_img == "disk" else "memory"

        # Check for NO_COLOR env var to disable progress bar
        no_color = (
            os.environ.get("NO_COLOR") is not None
            or os.environ.get("FORCE_COLOR") == "0"
        )
        use_progress = not no_color

        # Use parallel caching for larger datasets
        use_parallel = parallel and total_samples >= MIN_SAMPLES_FOR_PARALLEL_CACHING

        logger.info(f"Caching {total_samples} images to {cache_type}...")

        if use_parallel:
            self._fill_cache_parallel(
                labels, total_samples, cache_type, use_progress, num_workers
            )
        else:
            self._fill_cache_sequential(labels, total_samples, cache_type, use_progress)

        logger.info(f"Caching complete.")

    def _fill_cache_sequential(
        self,
        labels: List[sio.Labels],
        total_samples: int,
        cache_type: str,
        use_progress: bool,
    ):
        """Sequential implementation of cache filling.

        Args:
            labels: List of sio.Labels objects.
            total_samples: Total number of samples to cache.
            cache_type: Either "disk" or "memory".
            use_progress: Whether to show a progress bar.
        """

        def process_samples(progress=None, task=None):
            for sample in self.lf_idx_list:
                labels_idx = sample["labels_idx"]
                lf_idx = sample["lf_idx"]
                if sample.get("is_negative", False):
                    video_idx = sample["video_idx"]
                    frame_idx = sample["frame_idx"]
                    img = labels[labels_idx].videos[video_idx][frame_idx]
                else:
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
            process_samples()

    def _fill_cache_parallel(
        self,
        labels: List[sio.Labels],
        total_samples: int,
        cache_type: str,
        use_progress: bool,
        num_workers: int = 0,
    ):
        """Parallel implementation of cache filling using thread-local video copies.

        Args:
            labels: List of sio.Labels objects.
            total_samples: Total number of samples to cache.
            cache_type: Either "disk" or "memory".
            use_progress: Whether to show a progress bar.
            num_workers: Number of worker threads. If 0, uses min(4, cpu_count).
        """
        # Determine number of workers
        if num_workers <= 0:
            num_workers = min(4, os.cpu_count() or 1)

        cache_path = Path(self.cache_img_path) if self.cache_img_path else None

        filler = ParallelCacheFiller(
            labels=labels,
            lf_idx_list=self.lf_idx_list,
            cache_type=cache_type,
            cache_path=cache_path,
            num_workers=num_workers,
        )

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
                    f"Caching images to {cache_type} (parallel, {num_workers} workers)",
                    total=total_samples,
                )

                def progress_callback(completed):
                    progress.update(task, completed=completed)

                cache, errors = filler.fill_cache(progress_callback)
        else:
            logger.info(
                f"Caching {total_samples} images to {cache_type} "
                f"(parallel, {num_workers} workers)..."
            )
            cache, errors = filler.fill_cache()

        # Update instance cache
        if cache_type == "memory":
            self.cache.update(cache)

        # Log any errors
        if errors:
            logger.warning(
                f"Parallel caching completed with {len(errors)} errors. "
                f"First error: {errors[0]}"
            )

    def _apply_common_preprocessing(self, sample: Dict) -> Dict:
        """Apply common preprocessing steps shared across all dataset types.

        Handles: RGB/grayscale conversion, size matching, scaling, padding,
        and augmentation.

        Args:
            sample: Sample dict with at least ``image`` and ``instances`` keys.

        Returns:
            The sample dict with preprocessing applied in-place.
        """
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
                    symmetric_inds=self.symmetric_inds,
                    **self.geometric_aug,
                )

        return sample

    def _load_negative_sample(self, sample: Dict) -> Dict:
        """Load and preprocess a negative frame (no instances).

        Reads the image from the cache (if available) or video and returns a
        sample dict with all-NaN instances.  Downstream code will generate
        all-zero confidence maps from these NaN instances.

        Args:
            sample: Sample dict from lf_idx_list with ``is_negative=True``.

        Returns:
            Preprocessed sample dict.
        """
        labels_idx = sample["labels_idx"]
        video_idx = sample["video_idx"]
        frame_idx = sample["frame_idx"]
        lf_idx = sample["lf_idx"]

        if self.cache_img == "disk":
            img = np.array(
                Image.open(f"{self.cache_img_path}/sample_{labels_idx}_{lf_idx}.jpg")
            )
        elif self.cache_img == "memory":
            img = self.cache[(labels_idx, lf_idx)].copy()
        else:
            video = self.labels_list[labels_idx].videos[video_idx]
            img = video[frame_idx]
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        return process_negative_lf(
            img=img,
            frame_idx=frame_idx,
            video_idx=video_idx,
            max_instances=self.max_instances,
            num_nodes=self.num_nodes,
        )

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
        parallel_caching: bool = True,
        cache_workers: int = 0,
        use_negative_frames: bool = False,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
            use_negative_frames=use_negative_frames,
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

        if sample.get("is_negative", False):
            sample = self._load_negative_sample(sample)
        else:
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

        sample = self._apply_common_preprocessing(sample)

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
        if self.use_negative_frames:
            sample["is_negative"] = self.lf_idx_list[index].get("is_negative", False)

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
        parallel_caching: bool = True,
        cache_workers: int = 0,
        use_negative_frames: bool = False,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
            use_negative_frames=use_negative_frames,
        )
        self.confmap_head_config = confmap_head_config
        self.class_maps_head_config = class_maps_head_config

        self.class_names = self.class_maps_head_config.classes
        self.class_map_threshold = class_map_threshold

    def __getitem__(self, index) -> Dict:
        """Return dict with image, confmaps and class maps for given index."""
        sample = self.lf_idx_list[index]
        labels_idx = sample["labels_idx"]
        lf_idx = sample["lf_idx"]
        video_idx = sample["video_idx"]
        frame_idx = sample["frame_idx"]

        if sample.get("is_negative", False):
            sample = self._load_negative_sample(sample)
            track_ids = torch.zeros(0, dtype=torch.int32)
        else:
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

        sample = self._apply_common_preprocessing(sample)

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
        if self.use_negative_frames:
            sample["is_negative"] = self.lf_idx_list[index].get("is_negative", False)

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
        parallel_caching: bool = True,
        cache_workers: int = 0,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
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
                    symmetric_inds=self.symmetric_inds,
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

        # Drop keypoints pushed outside the crop by augmentation so their target
        # confidence map is empty rather than a partial blob at the crop edge.
        sample["instance"] = filter_oob_points(sample["instance"], img_hw[0], img_hw[1])

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


def _bbox_iou(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    """IoU of two ``(x0, y0, x1, y1)`` axis-aligned boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _match_mask_to_instance(
    inst: sio.Instance, lf_masks: List
) -> Optional["sio.SegmentationMask"]:
    """Find the segmentation mask belonging to ``inst`` on a labeled frame.

    Primary key is the one-way ``mask.instance`` identity link (set by sio and by
    the burned-in pseudo-label data). Falls back to the mask whose image-space
    bounding box best overlaps the instance's keypoint bounding box, so datasets
    without the explicit link still associate correctly.
    """
    if not lf_masks:
        return None
    for m in lf_masks:
        if getattr(m, "instance", None) is inst:
            return m
    # Fallback: max bbox-IoU between the keypoint bbox and each mask's bbox.
    pts = inst.numpy()
    vis = pts[~np.isnan(pts).any(axis=1)]
    if len(vis) == 0:
        return None
    inst_box = (
        float(vis[:, 0].min()),
        float(vis[:, 1].min()),
        float(vis[:, 0].max()),
        float(vis[:, 1].max()),
    )
    best, best_iou = None, 0.0
    for m in lf_masks:
        mx, my, mw, mh = m.bbox  # image-space (x, y, w, h)
        iou = _bbox_iou(inst_box, (mx, my, mx + mw, my + mh))
        if iou > best_iou:
            best, best_iou = m, iou
    return best


def _kp_bbox(inst: sio.Instance) -> Optional[Tuple[float, float, float, float]]:
    """Keypoint bounding box ``(x0, y0, x1, y1)`` of an instance, or None."""
    pts = inst.numpy()
    vis = pts[~np.isnan(pts).any(axis=1)]
    if len(vis) == 0:
        return None
    return (
        float(vis[:, 0].min()),
        float(vis[:, 1].min()),
        float(vis[:, 0].max()),
        float(vis[:, 1].max()),
    )


def _associate_masks(
    instances: List, lf_masks: List
) -> Dict[int, "sio.SegmentationMask"]:
    """One-to-one assign each instance at most one mask on a frame (no reuse).

    Identity links (``mask.instance``) are resolved first and their masks removed
    from the pool; remaining instances are matched to remaining masks by greedy
    descending bbox-IoU (> 0). This prevents two overlapping instances (the
    multi-animal case) from both grabbing the same mask, and prevents an unlinked
    instance from stealing a mask already claimed by another instance's link.
    """
    assigned: Dict[int, "sio.SegmentationMask"] = {}
    used: set = set()
    # 1) Identity links first.
    unlinked = []
    for i, inst in enumerate(instances):
        if inst.is_empty:
            continue
        linked = next(
            (
                m
                for m in lf_masks
                if id(m) not in used and getattr(m, "instance", None) is inst
            ),
            None,
        )
        if linked is not None:
            assigned[i] = linked
            used.add(id(linked))
        else:
            unlinked.append(i)
    # 2) Greedy max-IoU one-to-one for the rest over the remaining masks.
    triples = []
    for i in unlinked:
        ibox = _kp_bbox(instances[i])
        if ibox is None:
            continue
        for m in lf_masks:
            if id(m) in used:
                continue
            mx, my, mw, mh = m.bbox
            iou = _bbox_iou(ibox, (mx, my, mx + mw, my + mh))
            if iou > 0.0:
                triples.append((iou, i, m))
    triples.sort(key=lambda t: -t[0])
    for iou, i, m in triples:
        if i in assigned or id(m) in used:
            continue
        assigned[i] = m
        used.add(id(m))
    return assigned


class CenteredInstanceSegmentationDataset(CenteredInstanceDataset):
    """Dataset for top-down (crop-centered) instance segmentation (#622).

    Subclasses :class:`CenteredInstanceDataset`: reuses the centroid-crop
    pipeline but replaces the keypoint confidence-map GT with a single binary
    foreground mask of ONLY the centered instance (other instances' foreground
    inside the crop is background). The centered instance's full-frame mask is
    captured into the sample index at construction time (via the one-way
    ``mask.instance`` link, with a bbox-IoU fallback), decoded on access, and
    carried through the SAME size-matcher / crop / resize / pad operations as the
    image so it stays pixel-aligned with ``instance_image``; it is then
    downsampled to the segmentation head's output stride.

    Note:
        Augmentation: intensity aug is applied to the image; geometric aug
        (rotation/scale/translate/flip) co-transforms the centered-instance mask with
        the SAME affine matrix as the image+keypoints (nearest-neighbor, re-binarized)
        on the oversized ``sqrt(2)`` crop, which provides rotation headroom before the
        re-crop to ``crop_size``. Erase/mixup stay image-only. Train at ``scale=1.0``
        with crop dims divisible by ``max_stride`` (the mask resize/pad mirror the
        image but are not sub-pixel pad-aware).

    Attributes:
        seg_head_config: Configuration for the segmentation head (``output_stride``).
    """

    def __init__(
        self,
        labels: List[sio.Labels],
        crop_size: int,
        seg_head_config: DictConfig,
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
        parallel_caching: bool = True,
        cache_workers: int = 0,
    ) -> None:
        """Initialize class attributes."""
        self.seg_head_config = seg_head_config
        super().__init__(
            labels=labels,
            crop_size=crop_size,
            # The base class only uses confmap_head_config in its (overridden)
            # __getitem__; reuse the seg head config (it carries `output_stride`).
            confmap_head_config=seg_head_config,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
        )

    def _get_instance_idx_list(self, labels: List[sio.Labels]) -> List[Dict]:
        """Index per (frame, instance), capturing each instance's mask object.

        The associated ``sio.SegmentationMask`` (RLE-compact, picklable) is stored
        in each sample so ``__getitem__`` never needs a live ``Labels`` handle
        (correct under the memory/disk image caching paths). Instances with no
        associated mask are skipped.
        """
        instance_idx_list = []
        n_missing = 0
        for labels_idx, label in enumerate(labels):
            for lf_idx, lf in enumerate(label):
                if self.user_instances_only:
                    if lf.user_instances is not None and len(lf.user_instances) > 0:
                        lf.instances = lf.user_instances
                    else:
                        continue
                lf_masks = getattr(lf, "masks", None) or []
                # One-to-one mask<->instance assignment per frame (no mask reused
                # across overlapping instances).
                assigned = _associate_masks(lf.instances, lf_masks)
                for inst_idx, inst in enumerate(lf.instances):
                    if inst.is_empty:
                        continue
                    mask_obj = assigned.get(inst_idx)
                    if mask_obj is None:
                        n_missing += 1
                        continue
                    video_idx = labels[labels_idx].videos.index(lf.video)
                    instance_idx_list.append(
                        {
                            "labels_idx": labels_idx,
                            "lf_idx": lf_idx,
                            "inst_idx": inst_idx,
                            "video_idx": video_idx,
                            "instances": (
                                lf.instances if self.cache_img is not None else None
                            ),
                            "frame_idx": lf.frame_idx,
                            "mask_obj": mask_obj,
                        }
                    )
        if n_missing:
            logger.warning(
                f"CenteredInstanceSegmentationDataset: skipped {n_missing} "
                f"instance(s) with no associated segmentation mask."
            )
        return instance_idx_list

    def __getitem__(self, index) -> Dict:
        """Return dict with cropped image and the centered-instance mask GT."""
        from sleap_nn.inference.segmentation_convert import decode_mask_to_image_res

        meta = self.instance_idx_list[index]
        labels_idx = meta["labels_idx"]
        lf_idx = meta["lf_idx"]
        inst_idx = meta["inst_idx"]
        video_idx = meta["video_idx"]
        lf_frame_idx = meta["frame_idx"]

        if self.cache_img is not None:
            instances_list = meta["instances"]
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
        instances = np.stack([inst.numpy() for inst in instances_list], axis=0)
        image = np.expand_dims(image, axis=0)  # (1, C, H, W)
        instances = np.expand_dims(instances, axis=0)  # (1, num, nodes, 2)
        instances = torch.from_numpy(instances.astype("float32"))
        image = torch.from_numpy(image.copy())

        num_instances = instances.shape[1]
        orig_img_height, orig_img_width = image.shape[-2:]
        instances = instances[:, inst_idx]

        # Decode the centered instance's mask and place it on the EXACT full-frame
        # canvas so it shares the image's pixel grid. `decode_mask_to_image_res`
        # returns a partial-frame array for masks carrying a non-identity
        # scale/offset (e.g. a pseudo-label `.slp` written by a top-down seg
        # model); without this placement the mask desyncs from the (full-frame)
        # image under the shared size-matcher/crop and the GT is silently
        # corrupted. Clamp to frame bounds (decoded extent can be +/-1 px).
        mask_np = decode_mask_to_image_res(meta["mask_obj"])
        if mask_np.shape[:2] != (orig_img_height, orig_img_width):
            full = np.zeros((orig_img_height, orig_img_width), dtype=bool)
            h0 = min(mask_np.shape[0], orig_img_height)
            w0 = min(mask_np.shape[1], orig_img_width)
            full[:h0, :w0] = mask_np[:h0, :w0]
            mask_np = full
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_np, dtype=np.float32))[
            None, None
        ]

        if self.ensure_rgb:
            image = convert_to_rgb(image)
        elif self.ensure_grayscale:
            image = convert_to_grayscale(image)

        # Size matcher (apply the SAME geometry to image and mask).
        image, eff_scale = apply_sizematcher(
            image, max_height=self.max_hw[0], max_width=self.max_hw[1]
        )
        mask_t, _ = apply_sizematcher(
            mask_t, max_height=self.max_hw[0], max_width=self.max_hw[1]
        )
        instances = instances * eff_scale

        centroids = generate_centroids(instances, anchor_ind=self.anchor_ind)
        instance, centroid = instances[0], centroids[0]

        # Oversized (sqrt(2)) crop for rotation headroom — kept for parity with
        # CenteredInstanceDataset even though geometric aug is skipped here.
        crop_size_aug = (
            (np.array([self.crop_size, self.crop_size]) * np.sqrt(2))
            .astype(np.int32)
            .tolist()
        )
        sample = generate_crops(image, instance, centroid, crop_size_aug)
        # Carry the mask through the IDENTICAL crop bbox.
        mask_t = crop_and_resize(
            mask_t, boxes=sample["instance_bbox"], size=crop_size_aug
        )

        sample["frame_idx"] = torch.tensor(lf_frame_idx, dtype=torch.int32)
        sample["video_idx"] = torch.tensor(video_idx, dtype=torch.int32)
        sample["num_instances"] = num_instances
        sample["orig_size"] = torch.Tensor([orig_img_height, orig_img_width]).unsqueeze(
            0
        )
        sample["eff_scale"] = torch.tensor(eff_scale, dtype=torch.float32)

        # Intensity augmentation on the image only (mask is intensity-invariant).
        if self.apply_aug and self.intensity_aug is not None:
            (
                sample["instance_image"],
                sample["instance"],
            ) = apply_intensity_augmentation(
                sample["instance_image"], sample["instance"], **self.intensity_aug
            )
        # Geometric augmentation: co-transform the centered-instance mask with the
        # SAME flip/affine matrix as the image + keypoints (nearest-neighbor, then
        # re-binarized). The oversized sqrt(2) crop above provides the rotation
        # headroom; the re-crop below trims back to crop_size so no out-of-frame
        # corners reach the model.
        if self.apply_aug and self.geometric_aug is not None:
            (
                sample["instance_image"],
                sample["instance"],
                mask_t,
            ) = apply_geometric_augmentation(
                sample["instance_image"],
                sample["instance"],
                symmetric_inds=self.symmetric_inds,
                masks=mask_t,
                **self.geometric_aug,
            )

        # Re-crop to the exact crop size (same bbox for image and mask).
        sample["instance_bbox"] = torch.unsqueeze(
            make_centered_bboxes(sample["centroid"][0], self.crop_size, self.crop_size),
            0,
        )
        sample["instance_image"] = crop_and_resize(
            sample["instance_image"],
            boxes=sample["instance_bbox"],
            size=(self.crop_size, self.crop_size),
        )
        mask_t = crop_and_resize(
            mask_t, boxes=sample["instance_bbox"], size=(self.crop_size, self.crop_size)
        )
        point = sample["instance_bbox"][0][0]
        sample["instance"] = sample["instance"] - point
        sample["centroid"] = sample["centroid"] - point

        # Resize image + keypoints by scale; match the mask to the image size.
        sample["instance_image"], sample["instance"] = apply_resizer(
            sample["instance_image"], sample["instance"], scale=self.scale
        )
        tgt_hw = sample["instance_image"].shape[-2:]
        if tuple(mask_t.shape[-2:]) != tuple(tgt_hw):
            mask_t = F.interpolate(
                mask_t, size=(int(tgt_hw[0]), int(tgt_hw[1])), mode="area"
            )

        # Pad both to the model's max stride (bottom-right).
        sample["instance_image"] = apply_pad_to_stride(
            sample["instance_image"], max_stride=self.max_stride
        )
        mask_t = apply_pad_to_stride(mask_t, max_stride=self.max_stride)

        img_hw = sample["instance_image"].shape[-2:]
        mask_bool = mask_t[0, 0].numpy() > 0.5
        # Single-element list -> the centered instance's mask only (no union),
        # downsampled + thresholded to the seg head's output stride.
        sample["foreground_mask"] = generate_foreground_mask(
            [mask_bool],
            img_hw=img_hw,
            output_stride=self.seg_head_config.output_stride,
        )
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
        parallel_caching: bool = True,
        cache_workers: int = 0,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
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
                    symmetric_inds=self.symmetric_inds,
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

        # Drop keypoints pushed outside the crop by augmentation so their target
        # confidence map is empty rather than a partial blob at the crop edge.
        sample["instance"] = filter_oob_points(sample["instance"], img_hw[0], img_hw[1])

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
        parallel_caching: bool = True,
        cache_workers: int = 0,
        use_negative_frames: bool = False,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
            use_negative_frames=use_negative_frames,
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

        if sample.get("is_negative", False):
            sample = self._load_negative_sample(sample)
        else:
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
                    # Centroids are one point per instance (no node axis), so no
                    # symmetric swap applies — just mirror the coordinates.
                    symmetric_inds=[],
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
        if self.use_negative_frames:
            sample["is_negative"] = self.lf_idx_list[index].get("is_negative", False)

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
        parallel_caching: bool = True,
        cache_workers: int = 0,
        use_negative_frames: bool = False,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
            use_negative_frames=use_negative_frames,
        )
        self.confmap_head_config = confmap_head_config

    def __getitem__(self, index) -> Dict:
        """Return dict with image and confmaps for instance for given index."""
        sample = self.lf_idx_list[index]
        labels_idx = sample["labels_idx"]
        lf_idx = sample["lf_idx"]
        video_idx = sample["video_idx"]
        lf_frame_idx = sample["frame_idx"]

        if sample.get("is_negative", False):
            sample = self._load_negative_sample(sample)
        else:
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

        sample = self._apply_common_preprocessing(sample)

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
        if self.use_negative_frames:
            sample["is_negative"] = self.lf_idx_list[index].get("is_negative", False)

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


class BottomUpSegmentationDataset(BaseDataset):
    """Dataset class for bottom-up instance segmentation models.

    Loads per-instance segmentation masks from ``LabeledFrame.masks`` and
    generates ground truth tensors for a center-offset instance segmentation
    pipeline (foreground mask, instance-center heatmap, and per-pixel offsets).

    Masks are captured into the sample index at construction time (mirroring how
    keypoint instances are captured for caching), so ``__getitem__`` never needs
    a live ``Labels`` handle — this keeps it correct under the memory/disk image
    caching paths (where ``self.labels_list`` is ``None``).

    Note:
        Augmentation: intensity aug is applied to the image; geometric aug
        (rotation/scale/translate/flip) co-transforms every per-instance mask with the
        SAME affine matrix as the image (nearest-neighbor, re-binarized) at the
        preprocessed resolution, before center/offset targets are derived. Erase/mixup
        stay image-only. Mask resizing to the preprocessed image size handles scaling
        but is not pad-aware; for v1 train with ``scale=1.0`` and input dims divisible
        by ``max_stride`` (and prefer small rotation ranges, since a full-frame rotation
        can clip instances at the frame edge, as it does for bottom-up pose).

    Attributes:
        seg_head_config: Configuration for the segmentation head.
        center_head_config: Configuration for the instance center heatmap head.
        offset_head_config: Configuration for the center offset head.
    """

    def __init__(
        self,
        labels: List[sio.Labels],
        seg_head_config: DictConfig,
        center_head_config: DictConfig,
        offset_head_config: DictConfig,
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
        parallel_caching: bool = True,
        cache_workers: int = 0,
        use_negative_frames: bool = False,
    ) -> None:
        """Initialize class attributes."""
        self.seg_head_config = seg_head_config
        self.center_head_config = center_head_config
        self.offset_head_config = offset_head_config
        # Segmentation never uses negative frames (degenerate num_nodes/instances).
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
            use_negative_frames=False,
        )

    def _apply_common_preprocessing(self, sample: Dict) -> Dict:
        """Apply common preprocessing with geometric augmentation deferred.

        Geometric augmentation must co-transform the segmentation masks, but those
        masks are not present in ``sample`` here (they are loaded separately), so the
        base method would warp only the image. We disable geometric aug for the base
        call (intensity aug still applies) and re-apply it in ``__getitem__`` once the
        masks have been resized to the preprocessed resolution, co-transforming image
        and masks with the same matrix.

        Args:
            sample: Sample dict with at least ``image`` and ``instances`` keys.

        Returns:
            The sample dict with preprocessing applied in-place.
        """
        saved_geometric_aug = self.geometric_aug
        self.geometric_aug = None
        try:
            sample = super()._apply_common_preprocessing(sample)
        finally:
            self.geometric_aug = saved_geometric_aug
        return sample

    def _get_lf_idx_list(self, labels: List[sio.Labels]) -> List[Dict]:
        """Return samples for frames that have segmentation masks.

        Overrides the base class to index frames by their masks rather than
        keypoint instances. The decoded mask arrays are captured into each
        sample so ``__getitem__`` does not depend on a live ``Labels`` handle.
        """
        lf_idx_list = []
        for labels_idx, label in enumerate(labels):
            for lf_idx, lf in enumerate(label):
                lf_masks = getattr(lf, "masks", None)
                if not lf_masks:
                    continue
                # Scale-aware decode: masks written by the segmentation inference
                # layer are encoded at output-stride (non-identity scale); decode
                # them up to the IMAGE-pixel grid so self-training / pseudo-label
                # ``.slp`` files yield correctly-scaled targets (a stride-res
                # ``m.data`` would silently mis-scale the GT, since __getitem__'s
                # resize branch only fires on a preprocessing size change). Scale-1
                # GT masks take the zero-copy fast path.
                from sleap_nn.inference.segmentation_convert import (
                    decode_mask_to_image_res,
                )

                mask_arrays = [decode_mask_to_image_res(m) for m in lf_masks]
                if len(mask_arrays) == 0:
                    continue
                video_idx = label.videos.index(lf.video)
                sample = {
                    "labels_idx": labels_idx,
                    "lf_idx": lf_idx,
                    "video_idx": video_idx,
                    "frame_idx": lf.frame_idx,
                    "is_negative": False,
                    "instances": None,
                    "masks": mask_arrays,
                }
                lf_idx_list.append(sample)

        return lf_idx_list

    def __getitem__(self, index) -> Dict:
        """Return dict with image and segmentation GT for given index."""
        sample = self.lf_idx_list[index]
        labels_idx = sample["labels_idx"]
        lf_idx = sample["lf_idx"]
        video_idx = sample["video_idx"]
        frame_idx = sample["frame_idx"]

        # Load image
        if self.cache_img is not None:
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
            img = lf.image

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        image = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, axis=0)  # (1, C, H, W)
        image = torch.from_numpy(image.copy())

        # Dummy instances tensor (needed for preprocessing compatibility)
        instances = torch.zeros((1, 1, 1, 2), dtype=torch.float32)

        sample_dict = {
            "image": image,
            "instances": instances,
            "video_idx": torch.tensor(video_idx, dtype=torch.int32),
            "frame_idx": torch.tensor(frame_idx, dtype=torch.int32),
            "orig_size": torch.Tensor([image.shape[-2], image.shape[-1]]).unsqueeze(0),
            "num_instances": 0,
        }

        # Masks captured at index-build time (decoded bool arrays at orig res).
        mask_arrays = [np.asarray(m, dtype=bool) for m in sample["masks"]]

        # Record original image size before preprocessing
        orig_img_hw = (image.shape[-2], image.shape[-1])

        # Apply common preprocessing (RGB/grayscale, size matching, scaling, padding)
        sample_dict = self._apply_common_preprocessing(sample_dict)

        img_hw = sample_dict["image"].shape[-2:]

        # Resize masks to match preprocessed image dimensions if size changed.
        if img_hw != orig_img_hw and len(mask_arrays) > 0:
            target_h, target_w = img_hw
            resized_masks = []
            for m in mask_arrays:
                # Convert mask to float tensor: (1, 1, H, W)
                m_tensor = (
                    torch.from_numpy(m.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                )
                m_resized = F.interpolate(
                    m_tensor, size=(target_h, target_w), mode="area"
                )
                # Threshold back to binary
                resized_masks.append((m_resized.squeeze().numpy() > 0.5))
            mask_arrays = resized_masks

        # Geometric augmentation: co-transform the per-instance masks with the SAME
        # flip/affine matrix as the image (nearest-neighbor, re-binarized). Applied
        # here (post size-match / resize / pad) so the image and masks share a
        # resolution, and BEFORE centroid/heatmap/offset generation so those targets
        # are derived from the augmented masks. Bottom-up has no keypoints, so a dummy
        # instances tensor rides along; erase/mixup stay image-only.
        if self.apply_aug and self.geometric_aug is not None and len(mask_arrays) > 0:
            masks_t = torch.from_numpy(
                np.stack([m.astype(np.float32) for m in mask_arrays])
            ).unsqueeze(
                0
            )  # (1, K, H, W)
            (
                sample_dict["image"],
                _,
                masks_t,
            ) = apply_geometric_augmentation(
                sample_dict["image"],
                torch.zeros((1, 1, 1, 2), dtype=torch.float32),
                masks=masks_t,
                **self.geometric_aug,
            )
            mask_arrays = [masks_t[0, k].numpy() > 0.5 for k in range(masks_t.shape[1])]

        # Pre-compute mask centroids once for both center heatmap and offset heads
        centers = _compute_mask_centroids(mask_arrays) if len(mask_arrays) > 0 else []

        # Generate GT tensors
        foreground_mask = generate_foreground_mask(
            mask_arrays,
            img_hw=img_hw,
            output_stride=self.seg_head_config.output_stride,
        )

        center_heatmap = generate_center_heatmap(
            mask_arrays,
            img_hw=img_hw,
            output_stride=self.center_head_config.output_stride,
            sigma=self.center_head_config.sigma,
            centers=centers,
        )

        center_offsets, foreground_weight = generate_center_offsets(
            mask_arrays,
            img_hw=img_hw,
            output_stride=self.offset_head_config.output_stride,
            centers=centers,
        )

        sample_dict["foreground_mask"] = foreground_mask
        sample_dict["center_heatmap"] = center_heatmap
        sample_dict["center_offsets"] = center_offsets
        sample_dict["foreground_weight"] = foreground_weight
        sample_dict["labels_idx"] = labels_idx

        return sample_dict


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

    # Parallel caching configuration
    parallel_caching = getattr(config.data_config, "parallel_caching", True)
    cache_workers = getattr(config.data_config, "cache_workers", 0)

    use_negative_frames = getattr(config.data_config, "use_negative_frames", False)

    model_type = get_model_type_from_cfg(config=config)
    backbone_type = get_backbone_type_from_cfg(config=config)

    if use_negative_frames and model_type in (
        "centered_instance",
        "multi_class_topdown",
        "centered_instance_segmentation",
    ):
        logger.warning(
            f"use_negative_frames is enabled but model_type='{model_type}' "
            f"operates at instance-crop level and does not support frame-level "
            f"negatives. Negative frames will be disabled."
        )
        use_negative_frames = False

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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
            use_negative_frames=use_negative_frames,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
            use_negative_frames=use_negative_frames,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
            use_negative_frames=use_negative_frames,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
            use_negative_frames=use_negative_frames,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
            use_negative_frames=use_negative_frames,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
            use_negative_frames=use_negative_frames,
        )

    elif model_type == "bottomup_segmentation":
        seg_cfg = config.model_config.head_configs.bottomup_segmentation
        train_dataset = BottomUpSegmentationDataset(
            labels=train_labels,
            seg_head_config=seg_cfg.segmentation,
            center_head_config=seg_cfg.center,
            offset_head_config=seg_cfg.offsets,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
            use_negative_frames=False,
        )
        val_dataset = BottomUpSegmentationDataset(
            labels=val_labels,
            seg_head_config=seg_cfg.segmentation,
            center_head_config=seg_cfg.center,
            offset_head_config=seg_cfg.offsets,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
            use_negative_frames=False,
        )

    elif model_type == "centered_instance_segmentation":
        seg_cfg = config.model_config.head_configs.centered_instance_segmentation
        anchor_part = seg_cfg.segmentation.anchor_part
        if anchor_part is not None:
            nodes = [x["name"] for x in config.data_config.skeletons[0]["nodes"]]
            anchor_ind = nodes.index(anchor_part)
        else:
            anchor_ind = None
        train_dataset = CenteredInstanceSegmentationDataset(
            labels=train_labels,
            crop_size=config.data_config.preprocessing.crop_size,
            seg_head_config=seg_cfg.segmentation,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
        )
        val_dataset = CenteredInstanceSegmentationDataset(
            labels=val_labels,
            crop_size=config.data_config.preprocessing.crop_size,
            seg_head_config=seg_cfg.segmentation,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
            use_negative_frames=use_negative_frames,
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
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
            use_negative_frames=use_negative_frames,
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
