"""Custom `torch.utils.data.Dataset`s for different model types."""

from sleap_nn.data.skia_augmentation import crop_and_resize_skia as crop_and_resize

import math
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
from sleap_nn.data.tiling import (
    _FRAME_LRU_CAPACITY,
    _FrameLRU,
    FrameGroupedTileSampler,
    draw_tile_origin,
    extract_tile,
    frame_foreground_centers,
    generate_tile_grid,
    tile_sample_seed,
    tiling_worker_init_fn,
)
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
        tiling: Optional[Union[DictConfig, Any]] = None,
        output_stride: Optional[int] = None,
        base_seed: int = 0,
    ) -> None:
        """Initialize class attributes."""
        super().__init__()
        self.user_instances_only = user_instances_only
        self.use_negative_frames = use_negative_frames
        self.ensure_rgb = ensure_rgb
        self.ensure_grayscale = ensure_grayscale

        # --- Tiling awareness (Phase A) -------------------------------------
        # When a non-None, enabled tiling config is passed, the dataset switches
        # into tiled mode: the sizematcher step in `_apply_common_preprocessing`
        # is replaced by a per-tile slice (`extract_tile`) and subclasses emit
        # one (frame, tile-slot) sample per tile. When tiling is None/disabled
        # everything below stays inert and the dataset is byte-identical to
        # before (the regression guarantee).
        self.tiling = tiling
        self.tiling_enabled = tiling is not None and bool(
            getattr(tiling, "enabled", False)
        )
        self.base_seed = base_seed
        if self.tiling_enabled:
            self.tile_size = int(tiling.tile_size)
            self.overlap = int(tiling.overlap)
            # BaseDataset has no head; the head output stride is passed in by the
            # subclass so the tile grid can snap to the prediction grid.
            self.output_stride = int(output_stride) if output_stride is not None else 1
            self.tile_sampling = tiling.sampling
            self.min_overlap_fraction = float(tiling.min_overlap_fraction)
            self.tile_fg_fraction = float(tiling.tile_fg_fraction)
            self.center_jitter = float(tiling.center_jitter)
            self.min_visible_keypoints = int(tiling.min_visible_keypoints)
            self.samples_per_frame = int(tiling.samples_per_frame or 1)
            # Epoch counter in shared memory so persistent/spawned workers see the
            # main-process epoch updates (they never observe plain attribute writes).
            self._epoch = torch.zeros((), dtype=torch.long).share_memory_()

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

    def _frame_lru(self) -> _FrameLRU:
        """Return this process's decoded-frame LRU, building it lazily.

        The cache is stored keyed by ``os.getpid()`` so forked / persistent
        dataloader workers each get their own instance and never share (or
        pickle) decoded-frame tensors. It is excluded from pickling via
        ``__getstate__`` and rebuilt on first access in each process.
        """
        pid = os.getpid()
        store = self.__dict__.get("_frame_lru_store")
        if store is None or store.get("pid") != pid:
            store = {"pid": pid, "lru": _FrameLRU(_FRAME_LRU_CAPACITY)}
            self.__dict__["_frame_lru_store"] = store
        return store["lru"]

    def __getstate__(self):
        """Drop the per-process frame LRU so it is never pickled to workers."""
        state = self.__dict__.copy()
        state.pop("_frame_lru_store", None)
        return state

    def _frame_sized_hw(self, lf: sio.LabeledFrame) -> Tuple[int, int]:
        """Return a labeled frame's ``(H, W)`` after ``scale`` (sizematcher bypassed).

        Uses video metadata when available to avoid decoding the frame. Matches
        the ``int(dim * scale)`` truncation used by :func:`resize_image` so grid
        origins align with the sized frame produced by ``apply_resizer``.
        """
        shape = getattr(lf.video, "shape", None)
        if shape is not None and len(shape) >= 3:
            height, width = int(shape[1]), int(shape[2])
        else:
            img = lf.image
            height, width = int(img.shape[0]), int(img.shape[1])
        if self.scale != 1.0:
            height = int(height * self.scale)
            width = int(width * self.scale)
        return height, width

    def _get_tile_idx_list(self, labels: List[sio.Labels]) -> List[Dict]:
        """Return per-(frame, tile-slot) sample descriptors for tiled datasets.

        Mirrors ``_get_lf_idx_list`` (same user-instance filtering, empty-frame
        skip, and cache-aware ``instances`` storage), but explodes each frame
        into multiple tile descriptors: one per grid tile (``sampling="grid"``,
        pinned origins) or ``samples_per_frame`` runtime-drawn tiles
        (``sampling="foreground"``, ``tile_origin=None``). A frame's descriptors
        form a contiguous run (the block the sampler groups on).
        """
        tile_idx_list: List[Dict] = []
        for labels_idx, label in enumerate(labels):
            for lf_idx, lf in enumerate(label):
                if self.user_instances_only:
                    if lf.user_instances is not None and len(lf.user_instances) > 0:
                        lf.instances = lf.user_instances
                    else:
                        continue
                if all(inst.is_empty for inst in lf.instances):
                    continue
                video_idx = labels[labels_idx].videos.index(lf.video)

                if self.tile_sampling == "grid":
                    origins = generate_tile_grid(
                        self._frame_sized_hw(lf),
                        tile_size=self.tile_size,
                        overlap=self.overlap,
                        output_stride=self.output_stride,
                        max_stride=self.max_stride,
                        min_overlap_fraction=self.min_overlap_fraction,
                    )
                else:
                    origins = [None] * self.samples_per_frame

                for sample_k, origin in enumerate(origins):
                    tile_idx_list.append(
                        {
                            "labels_idx": labels_idx,
                            "lf_idx": lf_idx,
                            "video_idx": video_idx,
                            "frame_idx": lf.frame_idx,
                            "instances": (
                                lf.instances if self.cache_img is not None else None
                            ),
                            "sample_k": sample_k,
                            "tile_origin": origin,
                            "is_grid": self.tile_sampling == "grid",
                            "is_negative": False,
                        }
                    )

        if self.use_negative_frames:
            for neg in self._collect_negative_frames(labels):
                for sample_k in range(self.samples_per_frame):
                    d = dict(neg)
                    d.update(
                        sample_k=sample_k,
                        tile_origin=None,
                        is_grid=False,
                        is_negative=True,
                    )
                    tile_idx_list.append(d)

        return tile_idx_list

    @staticmethod
    def _build_frame_blocks(tile_idx_list: List[Dict]) -> List[List[int]]:
        """Group contiguous per-frame runs of ``tile_idx_list`` into index blocks."""
        blocks: List[List[int]] = []
        prev_key = object()
        for idx, d in enumerate(tile_idx_list):
            key = (d["labels_idx"], d["lf_idx"])
            if key != prev_key:
                blocks.append([])
                prev_key = key
            blocks[-1].append(idx)
        return blocks

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

        if not self.tiling_enabled:
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

            # Co-transform whole-frame segmentation masks (a (1, K, H, W) float
            # tensor the seg datasets place into the sample) through the IDENTICAL
            # size-match + scale as the image — the SAME helpers, so the mask
            # target stays pixel-aligned with the image by construction (bilinear
            # via ``tvf.resize``, matching the image). Non-seg samples have no
            # ``masks`` key, so this is a no-op for them.
            if sample.get("masks") is not None:
                sample["masks"], _ = apply_sizematcher(
                    sample["masks"],
                    max_height=self.max_hw[0],
                    max_width=self.max_hw[1],
                )
                sample["masks"], _ = apply_resizer(
                    sample["masks"], torch.zeros(1), scale=self.scale
                )
        else:
            # TILING: slice a tile IN PLACE OF the sizematcher (constant-zero pad
            # only). The incoming frame is already scaled + channel-coerced (via
            # `_to_sized_frame`), so `scale` is not re-applied here (that would
            # double-scale); tiles are extracted in the model's input space where
            # `tile_size` is divisible by the network strides. Geometric aug is
            # folded into `extract_tile` (halo path) when enabled.
            sample["image"], sample["instances"] = extract_tile(
                image=sample["image"],
                instances=sample["instances"],
                tile_origin=sample["tile_origin"],
                tile_size=self.tile_size,
                apply_geometric=(self.apply_aug and self.geometric_aug is not None),
                geometric_kwargs=(
                    dict(self.geometric_aug) if self.geometric_aug is not None else None
                ),
                symmetric_inds=self.symmetric_inds,
                rng_seed=sample.get("aug_seed"),
            )
            sample["eff_scale"] = torch.tensor(1.0, dtype=torch.float32)
            sample["tile_origin"] = torch.tensor(
                sample["tile_origin"], dtype=torch.int32
            )

        # Pad the image (if needed) according max stride. Per-tile under tiling;
        # a no-op when `tile_size % max_stride == 0` (guaranteed by write-back).
        sample["image"] = apply_pad_to_stride(
            sample["image"], max_stride=self.max_stride
        )
        # Pad segmentation masks to the SAME bottom-right stride multiple as the
        # image so the whole-frame target stays registered to the padded image.
        if sample.get("masks") is not None:
            sample["masks"] = apply_pad_to_stride(
                sample["masks"], max_stride=self.max_stride
            )

        # apply augmentation
        if self.apply_aug:
            if self.intensity_aug is not None:
                sample["image"], sample["instances"] = apply_intensity_augmentation(
                    sample["image"],
                    sample["instances"],
                    **self.intensity_aug,
                )

            # Under tiling, geometric augmentation is folded into `extract_tile`
            # (halo path); do NOT run it again here.
            if not self.tiling_enabled and self.geometric_aug is not None:
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
            # Antialiased bilinear, matching the image's ``apply_resizer`` /
            # ``tvf.resize`` above so the mask stays registered to the image
            # (standardized across every seg mask-geometry resize).
            mask_t = F.interpolate(
                mask_t,
                size=(int(tgt_hw[0]), int(tgt_hw[1])),
                mode="bilinear",
                align_corners=False,
                antialias=True,
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


def _global_identity_label(det, track_names_are_global: bool) -> Optional[str]:
    """The global (cross-video) identity label for one detection, or ``None``.

    A ``sio.Identity`` (``det.identity``, added by sleap-io#535) is the ground-truth
    global animal identity — the same object recognized across videos/sessions — so it
    is the canonical ``global_id`` grouping key when present (matched by ``name``, the
    sleap-io cross-file key). ``sio.Track`` is only a per-video tracklet, so a track
    name is used as the global label ONLY under the ``track_names_are_global`` promise
    (the pre-Identity convention that a track name denotes the same animal everywhere).

    Works for both pose ``Instance`` and ``SegmentationMask`` detections (both carry
    ``identity`` / ``track`` on sleap-io main).
    """
    identity = getattr(det, "identity", None)
    if identity is not None and getattr(identity, "name", None):
        return identity.name
    if track_names_are_global:
        track = getattr(det, "track", None)
        if track is not None and track.name is not None:
            return track.name
    return None


def resolve_embedding_class_names(
    labels: List[sio.Labels], track_names_are_global: bool = True
) -> List[str]:
    """Collect the sorted global-identity vocabulary (the ``global_id`` / eval grouping).

    The vocabulary is the set of GLOBAL animal identities — a real ``sio.Identity``
    name when a detection carries one, else its ``sio.Track`` name under the
    ``track_names_are_global`` promise (see :func:`_global_identity_label`). Resolving
    per detection (not per track) means an animal with both an ``Identity`` and a
    per-video ``Track`` contributes only its identity, so identity- and track-labelled
    data share one coherent vocabulary. Scanning BOTH train + val labels and sorting
    gives one consistent vocabulary shared by the train and val datasets.
    """
    names = set()
    for label in labels:
        for lf in label:
            dets = list(lf.instances) + list(getattr(lf, "masks", None) or [])
            for det in dets:
                lab = _global_identity_label(det, track_names_are_global)
                if lab is not None:
                    names.add(lab)
    return sorted(names)


def _mask_bbox_midpoint(mask_bool: np.ndarray) -> Tuple[float, float]:
    """Return the ``(cx, cy)`` midpoint of a boolean mask's bounding box.

    Robust to concave masks whose center-of-mass falls off the instance. Falls back
    to the image center when the mask is empty.
    """
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        h, w = mask_bool.shape[:2]
        return float(w) / 2.0, float(h) / 2.0
    cx = (float(xs.min()) + float(xs.max())) / 2.0
    cy = (float(ys.min()) + float(ys.max())) / 2.0
    return cx, cy


def derive_centroids_from_masks(
    labels: "sio.Labels", centering: str = "com"
) -> "sio.Labels":
    """Attach a single-node ``centroid`` pose derived from each segmentation mask.

    Mask-only data (e.g. instance segmentation) carries no keypoint skeleton, so the
    standard ``centroid`` training path (which reads ``lf.instances`` and reduces them
    via :func:`generate_centroids`) has nothing to train on. This synthesizes that input
    natively — without an offline preprocessing step — by adding, for every
    ``LabeledFrame.masks`` entry, a one-node ``"centroid"`` :class:`sio.Instance` at the
    mask's center-of-mass (``centering="com"``) or bounding-box midpoint
    (``centering="bbox"``), in full-image coordinates. The instance carries the mask's
    track and is back-linked via ``mask.instance``, so the rest of the pipeline (head
    config, dataset, eval, inference) sees a normal single-node centroid pose model.

    Mutates ``labels`` in place (sets ``lf.instances`` and ``labels.skeletons``) and
    returns it. A no-op (returns ``labels`` unchanged) when no frame has masks.

    Args:
        labels: The labels to transform (typically mask-only).
        centering: ``"com"`` (mask center-of-mass) or ``"bbox"`` (mask bounding-box
            midpoint, robust to concave masks whose COM lands off the instance).

    Returns:
        The same ``labels`` object, with synthesized centroid instances + skeleton.
    """
    if centering not in ("com", "bbox"):
        message = f"Unknown centering '{centering}'; choose 'com' or 'bbox'."
        logger.error(message)
        raise ValueError(message)
    # sleap-io #531 upstreamed mask -> centroid -> pose conversion. `center_of_mass`
    # matches the previous hand-rolled COM bit-for-bit at image resolution and handles
    # the mask's scale/offset natively (more correct for crop-centered masks);
    # `bbox_center` uses pixel-as-unit-area, so it sits +0.5px from the old
    # `(min+max)/2` bbox midpoint.
    method = "bbox_center" if centering == "bbox" else "center_of_mass"

    # Guard the footgun: this path overwrites `lf.instances` with the synthesized
    # single-node centroids, which would silently destroy real keypoint poses. It is
    # for mask-only data, so refuse (atomically, before any mutation) if any frame
    # carries BOTH masks and pre-existing instances rather than clobbering them.
    for lf in labels:
        if (getattr(lf, "masks", None)) and (getattr(lf, "instances", None)):
            message = (
                "derive_centroids_from_masks (data_config.centroids_from_masks) is for "
                "mask-only data, but a LabeledFrame carries BOTH masks and existing "
                "instances; synthesizing centroids from masks would overwrite the real "
                "pose instances. Disable centroids_from_masks for data that already has "
                "keypoint poses (the standard centroid path handles it directly)."
            )
            logger.error(message)
            raise ValueError(message)

    skeleton = sio.Skeleton(nodes=["centroid"], name="centroid")
    n_inst = 0
    saw_mask = False
    for lf in labels:
        masks = getattr(lf, "masks", None) or []
        if not masks:
            continue
        saw_mask = True
        instances = []
        for mask in masks:
            if mask.is_empty:
                # An empty/degenerate mask (filtered or thresholding artifact) has no
                # location; skip it rather than planting a spurious NaN / center-of-frame
                # centroid as a training target.
                continue
            # Full-image centroid pose via the upstream conversion; the mask's track /
            # identity propagate onto the derived single-node pose.
            inst = mask.to_centroid(method=method).to_pose(skeleton=skeleton)
            instances.append(inst)
            mask.instance = inst
            n_inst += 1
        lf.instances = instances

    if saw_mask:
        labels.skeletons = [skeleton]
        logger.info(
            f"Derived {n_inst} '{centering}' centroid instance(s) from masks "
            f"(single-node 'centroid' skeleton)."
        )
    return labels


class EmbeddingDataset(BaseDataset):
    """Dataset for the ``embedding`` (crop -> vector, re-ID) model type.

    One sample per tracked detection. Two detection modes are auto-detected:

    - ``mask``: a tracked ``lf.masks`` entry; the fixed-square crop is centered on the
      mask center-of-mass and carries the binary mask crop.
    - ``pose``: a tracked ``lf.instances`` keypoint detection; the crop is centered on
      the pose centroid (anchor node with a per-instance mean-of-visible-nodes
      fallback) and carries an all-ones mask.

    Returns the grayscale crop, a mask crop, and per-crop metadata (``video_idx``,
    ``frame_idx``, ``group_id``, ``global_group_id``, ``item_id``). When ``apply_aug``
    is set, two independently-augmented views (``instance_image`` / ``_view2``) are
    produced for the two-view contrastive loss using the standard config-driven
    augmentation; otherwise a single un-augmented view is returned (val / inference).

    The ``group_id`` keys the training groups: the global-identity index for
    ``global_id`` scope, or a per-``(labels, video, track)`` tracklet id for
    ``tracklet`` scope. ``global_group_id`` is always the global-identity index (the
    grouping used for evaluation). The global identity of a detection is its real
    ``sio.Identity`` name when present, else its ``sio.Track`` name under
    ``track_names_are_global`` (see :func:`_global_identity_label`).

    Attributes:
        crop_size: Side length of the square crop (should be divisible by max_stride).
        class_names: Ordered global-identity vocabulary (the ``global_id`` group space;
            ``sio.Identity`` names, or track names under ``track_names_are_global``).
        embedding_head_config: The head leaf config (carries ``output_stride`` and the
            optional pose ``anchor_part``).
        id_scope: Training-group key: ``global_id`` | ``tracklet`` | ``aug_view``.
        track_names_are_global: Treat a ``sio.Track`` name as a global animal identity
            for detections lacking a ``sio.Identity`` (the pre-Identity convention).
    """

    def __init__(
        self,
        labels: List[sio.Labels],
        crop_size: int,
        class_names: List[str],
        embedding_head_config: DictConfig,
        max_stride: int,
        id_scope: str = "global_id",
        track_names_are_global: bool = True,
        crop_centering: str = "auto",
        user_instances_only: bool = True,
        ensure_rgb: bool = False,
        ensure_grayscale: bool = True,
        intensity_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        geometric_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        apply_aug: bool = False,
        scale: float = 1.0,
        max_hw: Tuple[Optional[int]] = (None, None),
        cache_img: Optional[str] = None,
        cache_img_path: Optional[str] = None,
        use_existing_imgs: bool = False,
        rank: Optional[int] = None,
        parallel_caching: bool = True,
        cache_workers: int = 0,
    ) -> None:
        """Initialize class attributes."""
        self.crop_size = crop_size
        self.class_names = list(class_names)
        self.embedding_head_config = embedding_head_config
        # `group_id` keying: global identity (global_id) vs per-(video, track)
        # tracklet. `_tracklet_vocab` lazily assigns a dense id per distinct tracklet.
        self.id_scope = id_scope
        # Whether a `sio.Track` name doubles as a global animal identity for detections
        # without a real `sio.Identity` (the pre-Identity convention).
        self.track_names_are_global = bool(track_names_are_global)
        # Mask-mode crop center: `auto`/`mask_com` -> mask center-of-mass; `bbox` ->
        # mask bounding-box midpoint (robust to concave masks). Pose-mode centering is
        # driven by `anchor_part` regardless of this knob.
        if crop_centering not in ("auto", "mask_com", "bbox"):
            message = (
                f"Unknown crop_centering '{crop_centering}'; choose one of "
                f"auto|mask_com|bbox."
            )
            logger.error(message)
            raise ValueError(message)
        self.crop_centering = crop_centering
        # `scale` is NOT applied to embedding crops (the crop path sizes via the
        # centroid bbox + max_hw, then resizes to crop_size), but inference's
        # EmbeddingLayer DOES scale its preprocess — so a non-1.0 scale would make the
        # trained and inference crops disagree. Warn rather than silently diverge.
        if float(scale) != 1.0:
            logger.warning(
                f"data_config.preprocessing.scale={scale} is not applied to embedding "
                "crops (only crop_size sizing is), but inference scales its crops — "
                "leave scale=1.0 for the embedding model to keep train/inference crops "
                "consistent."
            )
        self._tracklet_vocab: Dict[tuple, int] = {}
        super().__init__(
            labels=labels,
            max_stride=max_stride,
            user_instances_only=user_instances_only,
            ensure_rgb=ensure_rgb,
            ensure_grayscale=ensure_grayscale,
            intensity_aug=intensity_aug,
            geometric_aug=geometric_aug,
            scale=scale,
            # Two-view contrastive aug now reuses the standard skia augmentation
            # (config-driven) per-crop in __getitem__ (CPU-side, like every other
            # dataset) instead of a bespoke GPU reimplementation in the LM.
            apply_aug=apply_aug,
            max_hw=max_hw,
            cache_img=cache_img,
            cache_img_path=cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
        )
        # Anchor node for pose-centroid centering (Stage 1). Resolved against the
        # skeleton; falls back per-instance to the mean of visible nodes (the topdown
        # convention via `generate_centroids`).
        self.anchor_part = OmegaConf.select(
            embedding_head_config, "anchor_part", default=None
        )
        self.anchor_ind = None
        if self.anchor_part is not None:
            for label in labels:
                if label.skeletons:
                    names = label.skeletons[0].node_names
                    if self.anchor_part in names:
                        self.anchor_ind = names.index(self.anchor_part)
                    break

        # Detection mode: tracked masks (crop on the mask center-of-mass) vs tracked
        # keypoint instances (crop on the pose centroid, no mask).
        self.detection_mode = self._detect_mode(labels)
        if self.detection_mode == "pose":
            self.mask_idx_list = self._get_instance_idx_list(labels)
        else:
            self.mask_idx_list = self._get_mask_idx_list(labels)
        # Per-crop arrays for the group-aware batch sampler.
        self.group_ids = np.array([m["group_id"] for m in self.mask_idx_list], np.int64)
        self.video_ids = np.array(
            [m["video_idx"] for m in self.mask_idx_list], np.int64
        )
        self.frame_ids = np.array(
            [m["frame_idx"] for m in self.mask_idx_list], np.int64
        )

    def _detect_mode(self, labels: List[sio.Labels]) -> str:
        """``"mask"`` if any frame has a tracked mask, else ``"pose"`` (keypoints)."""
        for label in labels:
            for lf in label:
                for m in getattr(lf, "masks", None) or []:
                    if getattr(m, "track", None) is not None:
                        return "mask"
        return "pose"

    def _group_keys(self, labels_idx, video_idx, track_name, global_label):
        """Return ``(group_id, global_group_id)`` for a detection.

        ``global_group_id`` is the detection's GLOBAL identity index (``sio.Identity``,
        or track name under ``track_names_are_global``) — the eval grouping; it falls
        back to ``group_id`` when the detection carries no global label (e.g. a bare
        tracklet under ``scope='tracklet'``).

        ``group_id`` is the TRAINING positive key: the global-identity index for
        ``global_id`` / ``aug_view`` scope, or a dense per-``(labels, video, track)``
        tracklet id for ``tracklet`` scope.
        """
        gid = (
            self.class_names.index(global_label)
            if global_label is not None and global_label in self.class_names
            else None
        )
        if self.id_scope == "tracklet":
            key = (labels_idx, video_idx, track_name)
            tid = self._tracklet_vocab.setdefault(key, len(self._tracklet_vocab))
            return tid, (gid if gid is not None else tid)
        return gid, gid

    def _is_member(self, det) -> bool:
        """Whether a detection is a training sample under the active scope.

        Side-effect free (does NOT assign tracklet ids), so it is safe to call in the
        frame-cache pass. ``tracklet`` needs a ``sio.Track`` (the per-video tracklet it
        groups on); ``global_id`` / ``aug_view`` need a global-identity label
        (``sio.Identity`` name, or a track name under ``track_names_are_global``)
        present in the shared vocabulary.
        """
        if self.id_scope == "tracklet":
            return getattr(det, "track", None) is not None
        global_label = _global_identity_label(det, self.track_names_are_global)
        return global_label is not None and global_label in self.class_names

    def _resolve_group(self, det, labels_idx, video_idx):
        """Return ``(group_id, global_group_id)`` for a detection, or ``None`` to skip."""
        if not self._is_member(det):
            return None
        track = getattr(det, "track", None)
        track_name = track.name if track is not None else None
        global_label = _global_identity_label(det, self.track_names_are_global)
        return self._group_keys(labels_idx, video_idx, track_name, global_label)

    def _get_instance_idx_list(self, labels: List[sio.Labels]) -> List[Dict]:
        """Index per tracked keypoint instance (pose mode).

        Centroid via the topdown :func:`generate_centroids` (anchor node with a
        per-instance fallback to the mean of visible nodes).
        """
        idx_list = []
        n_missing = 0
        for labels_idx, label in enumerate(labels):
            for lf_idx, lf in enumerate(label):
                for inst_idx, inst in enumerate(lf.instances):
                    video_idx = labels[labels_idx].videos.index(lf.video)
                    group = self._resolve_group(inst, labels_idx, video_idx)
                    if group is None:
                        n_missing += 1
                        continue
                    pts = torch.from_numpy(inst.numpy()).to(
                        torch.float32
                    )  # (n_nodes,2)
                    centroid = generate_centroids(
                        pts.unsqueeze(0), anchor_ind=self.anchor_ind
                    )[
                        0
                    ]  # (x, y) in original image coords
                    if torch.isnan(centroid).any():
                        continue
                    centroid = centroid.numpy().astype(np.float32)
                    group_id, global_group_id = group
                    idx_list.append(
                        {
                            "labels_idx": labels_idx,
                            "lf_idx": lf_idx,
                            "instance_idx": inst_idx,
                            "video_idx": video_idx,
                            "frame_idx": lf.frame_idx,
                            "centroid": centroid,
                            "group_id": group_id,
                            "global_group_id": global_group_id,
                        }
                    )
        if n_missing:
            logger.warning(
                f"EmbeddingDataset: skipped {n_missing} instance(s) with no group "
                f"under scope='{self.id_scope}' (no track for tracklet scope, or no "
                f"in-vocabulary global identity for global_id/aug_view)."
            )
        return idx_list

    def _get_lf_idx_list(self, labels: List[sio.Labels]) -> List[Dict]:
        """Index frames carrying >=1 tracked mask OR instance (so the image cache covers them).

        Mask-only (gerbil) data carries no user *instances* and pose (fly) data carries
        no *masks*, so the base ``_get_lf_idx_list`` (which filters on user instances)
        can leave the image cache empty. Index on either a tracked ``lf.masks`` or a
        tracked ``lf.instances``. Runs before ``detection_mode`` is set, so it is
        mode-agnostic.
        """
        lf_idx_list = []
        for labels_idx, label in enumerate(labels):
            for lf_idx, lf in enumerate(label):
                lf_masks = getattr(lf, "masks", None) or []
                has_tracked = any(self._is_member(m) for m in lf_masks) or any(
                    self._is_member(inst) for inst in lf.instances
                )
                if has_tracked:
                    video_idx = labels[labels_idx].videos.index(lf.video)
                    lf_idx_list.append(
                        {
                            "labels_idx": labels_idx,
                            "lf_idx": lf_idx,
                            "video_idx": video_idx,
                            "frame_idx": lf.frame_idx,
                            "is_negative": False,
                            "instances": None,
                        }
                    )
        return lf_idx_list

    def _get_mask_idx_list(self, labels: List[sio.Labels]) -> List[Dict]:
        """Index per mask, capturing the (picklable) mask object + its group id."""
        mask_idx_list = []
        n_missing = 0
        for labels_idx, label in enumerate(labels):
            for lf_idx, lf in enumerate(label):
                lf_masks = getattr(lf, "masks", None) or []
                for mask_idx, mask_obj in enumerate(lf_masks):
                    video_idx = labels[labels_idx].videos.index(lf.video)
                    group = self._resolve_group(mask_obj, labels_idx, video_idx)
                    if group is None:
                        n_missing += 1
                        continue
                    group_id, global_group_id = group
                    mask_idx_list.append(
                        {
                            "labels_idx": labels_idx,
                            "lf_idx": lf_idx,
                            "mask_idx": mask_idx,
                            "video_idx": video_idx,
                            "frame_idx": lf.frame_idx,
                            "mask_obj": mask_obj,
                            "group_id": group_id,
                            "global_group_id": global_group_id,
                        }
                    )
        if n_missing:
            logger.warning(
                f"EmbeddingDataset: skipped {n_missing} mask(s) with no group under "
                f"scope='{self.id_scope}' (no track for tracklet scope, or no "
                f"in-vocabulary global identity for global_id/aug_view)."
            )
        return mask_idx_list

    def __len__(self) -> int:
        """Return the number of mask crops."""
        return len(self.mask_idx_list)

    def __getitem__(self, index) -> Dict:
        """Return one grayscale crop + a mask crop + metadata.

        Mask mode: crop centered on the mask COM, carrying the binary mask crop.
        Pose mode: crop centered on the pose centroid (mean of visible keypoints),
        carrying an all-ones mask (no segmentation; train maskless via
        ``preprocessing.burn_in=false``).
        """
        meta = self.mask_idx_list[index]
        labels_idx, lf_idx = meta["labels_idx"], meta["lf_idx"]

        if self.cache_img is not None and self.cache_img == "memory":
            img = self.cache[(labels_idx, lf_idx)].copy()
        elif self.cache_img is not None and self.cache_img == "disk":
            img = np.array(
                Image.open(f"{self.cache_img_path}/sample_{labels_idx}_{lf_idx}.jpg")
            )
        else:
            img = self.labels_list[labels_idx][lf_idx].image
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        image = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)  # (1, C, H, W)
        image = torch.from_numpy(image.copy())

        if "centroid" in meta:  # pose mode
            instance_image, instance_mask = self._crop_pose(image, meta)
        else:  # mask mode
            instance_image, instance_mask = self._crop_mask(image, meta)
        return self._pack_sample(instance_image, instance_mask, meta, index)

    def _crop_mask(self, image, meta):
        """Crop centered on the mask COM; return ``(image_crop, mask_crop)``."""
        from sleap_nn.inference.segmentation_convert import decode_mask_to_image_res

        orig_h, orig_w = image.shape[-2:]
        mask_np = decode_mask_to_image_res(meta["mask_obj"])
        if mask_np.shape[:2] != (orig_h, orig_w):
            full = np.zeros((orig_h, orig_w), dtype=bool)
            h0 = min(mask_np.shape[0], orig_h)
            w0 = min(mask_np.shape[1], orig_w)
            full[:h0, :w0] = mask_np[:h0, :w0]
            mask_np = full
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_np, dtype=np.float32))[
            None, None
        ]

        if self.ensure_rgb:
            image = convert_to_rgb(image)
        elif self.ensure_grayscale:
            image = convert_to_grayscale(image)

        image, _ = apply_sizematcher(
            image, max_height=self.max_hw[0], max_width=self.max_hw[1]
        )
        mask_t, _ = apply_sizematcher(
            mask_t, max_height=self.max_hw[0], max_width=self.max_hw[1]
        )

        mask_bool = mask_t[0, 0].numpy() > 0.5
        if self.crop_centering == "bbox":
            cx, cy = _mask_bbox_midpoint(mask_bool)
        else:  # "auto" / "mask_com"
            cx, cy = _compute_mask_centroids([mask_bool])[0]
        bbox = make_centered_bboxes(
            torch.tensor([cx, cy], dtype=torch.float32),
            self.crop_size,
            self.crop_size,
        ).unsqueeze(0)
        instance_image = crop_and_resize(
            image, boxes=bbox, size=(self.crop_size, self.crop_size)
        )
        instance_mask = crop_and_resize(
            mask_t, boxes=bbox, size=(self.crop_size, self.crop_size)
        )
        return instance_image, instance_mask

    def _crop_pose(self, image, meta):
        """Crop centered on the pose centroid; return ``(image_crop, ones_mask)``."""
        if self.ensure_rgb:
            image = convert_to_rgb(image)
        elif self.ensure_grayscale:
            image = convert_to_grayscale(image)

        image, ratio = apply_sizematcher(
            image, max_height=self.max_hw[0], max_width=self.max_hw[1]
        )
        cx = float(meta["centroid"][0]) * ratio
        cy = float(meta["centroid"][1]) * ratio
        bbox = make_centered_bboxes(
            torch.tensor([cx, cy], dtype=torch.float32),
            self.crop_size,
            self.crop_size,
        ).unsqueeze(0)
        instance_image = crop_and_resize(
            image, boxes=bbox, size=(self.crop_size, self.crop_size)
        )
        # No segmentation in pose mode: an all-ones mask (burn-in is a no-op; train
        # maskless via preprocessing.burn_in=false).
        instance_mask = torch.ones(
            (1, 1, self.crop_size, self.crop_size), dtype=torch.float32
        )
        return instance_image, instance_mask

    def _apply_crop_aug(self, image, mask):
        """Apply the standard config-driven skia aug to a crop + its mask.

        Reuses sleap-nn's :func:`apply_intensity_augmentation` /
        :func:`apply_geometric_augmentation` (the same functions every other dataset
        uses) instead of a bespoke GPU reimplementation: the mask co-transforms under
        the SAME affine via the ``masks=`` arg, and a placeholder keypoint rides along
        (the crop has no keypoints) and is discarded. Returns ``(image, mask)``.
        """
        # (n_samples=1, n_inst=1, n_nodes=1, 2) center placeholder for the keypoint
        # co-transform the skia aug expects; its output is ignored.
        dummy = torch.full((1, 1, 1, 2), float(self.crop_size) / 2.0)
        if self.intensity_aug is not None:
            image, _ = apply_intensity_augmentation(image, dummy, **self.intensity_aug)
        if self.geometric_aug is not None:
            image, _, mask = apply_geometric_augmentation(
                image, dummy, masks=mask, symmetric_inds=None, **self.geometric_aug
            )
        return image, mask

    def _pack_sample(self, instance_image, instance_mask, meta, index):
        """Pack one crop into a sample dict.

        Training (``apply_aug=True``): emit TWO independently-augmented views
        (``instance_image``/``instance_image_view2`` + masks) for the two-view
        contrastive loss. Val / inference (``apply_aug=False``): emit one un-augmented
        view (``instance_image``) only.
        """
        sample = {
            "group_id": torch.tensor(meta["group_id"], dtype=torch.int64),
            "global_group_id": torch.tensor(
                meta.get("global_group_id", meta["group_id"]), dtype=torch.int64
            ),
            "video_idx": torch.tensor(meta["video_idx"], dtype=torch.int64),
            "frame_idx": torch.tensor(meta["frame_idx"], dtype=torch.int64),
            "item_id": torch.tensor(index, dtype=torch.int64),
            "labels_idx": meta["labels_idx"],
        }
        if self.apply_aug:
            img1, m1 = self._apply_crop_aug(instance_image, instance_mask)
            img2, m2 = self._apply_crop_aug(instance_image, instance_mask)
            sample["instance_image"] = img1.to(torch.float32)
            sample["instance_mask"] = (m1 > 0.5).to(torch.float32)
            sample["instance_image_view2"] = img2.to(torch.float32)
            sample["instance_mask_view2"] = (m2 > 0.5).to(torch.float32)
        else:
            sample["instance_image"] = instance_image.to(torch.float32)
            sample["instance_mask"] = (instance_mask > 0.5).to(torch.float32)
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

        # Drop keypoints pushed outside the image by augmentation so their target
        # confidence map is empty rather than a partial blob at the image edge.
        sample["instances"] = filter_oob_points(
            sample["instances"], img_hw[0], img_hw[1]
        )

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


class SingleInstanceTiledDataset(BaseDataset):
    """Single-instance dataset that emits fixed-size tiles instead of whole frames.

    Phase-A tiled training dataset. Each frame is decomposed into overlapping
    square tiles (foreground-aware random draws for training, a deterministic
    grid for validation). A frame is decoded/`process_lf`/channel-coerced/scaled
    once (cached in a per-worker LRU) and reused across all of its tiles; each
    tile is then sliced out (`extract_tile`, sizematcher bypassed), optionally
    geometrically augmented via the halo path, and its per-tile confidence maps
    are generated on tile-local coordinates.

    Emits one sample per ``(frame, tile-slot)``; ``__len__`` is the total number
    of tile slots. Returned samples match the ``SingleInstanceDataset`` key
    contract (plus an ``int32`` ``tile_origin`` of shape ``(2,)``), so the
    default collate applies with no custom ``collate_fn``.

    Single-instance keeps one pose per frame: on multi-instance labels a one-time
    warning is emitted (foreground sampling uses all keypoints and inference
    decodes a single global peak per node).
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
        tiling: Optional[Union[DictConfig, Any]] = None,
        base_seed: int = 0,
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
            tiling=tiling,
            output_stride=confmap_head_config.output_stride,
            base_seed=base_seed,
        )
        self.confmap_head_config = confmap_head_config

        # Per-(frame, tile-slot) descriptors + contiguous per-frame index blocks.
        self.tile_idx_list = self._get_tile_idx_list(labels)
        self.frame_blocks = self._build_frame_blocks(self.tile_idx_list)

        if self.max_instances > 1:
            logger.warning(
                "SingleInstanceTiledDataset received labels with more than one "
                "instance per frame. Single-instance models keep one pose per "
                "frame: all keypoints seed foreground tile sampling and inference "
                "decodes a single global peak per node."
            )

    def __len__(self) -> int:
        """Return the number of tile samples (frames x tiles-per-frame)."""
        return len(self.tile_idx_list)

    def _read_frame(self, d: Dict) -> Tuple[np.ndarray, List[sio.Instance]]:
        """Read a frame's raw image + instances (cache/disk/labels), restoring 2D->3D.

        Mirrors the cache/disk/labels-list read in ``SingleInstanceDataset``.
        """
        labels_idx = d["labels_idx"]
        lf_idx = d["lf_idx"]
        if self.cache_img is not None:
            instances = d["instances"]
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
        return img, instances

    def _to_sized_frame(self, frame: Dict) -> Dict:
        """Channel-coerce + `apply_resizer(scale)` so the LRU stores sized frames.

        Applies the ensure_rgb/grayscale + ``scale`` prefix once, up front, so the
        cached frame (and every tile sliced from it) is already in the model's
        input space. ``_apply_common_preprocessing`` therefore does not re-scale.
        """
        if self.ensure_rgb:
            frame["image"] = convert_to_rgb(frame["image"])
        elif self.ensure_grayscale:
            frame["image"] = convert_to_grayscale(frame["image"])
        frame["image"], frame["instances"] = apply_resizer(
            frame["image"], frame["instances"], scale=self.scale
        )
        return frame

    def __getitem__(self, index) -> Dict:
        """Return dict with image + confmaps for one tile of one frame."""
        d = self.tile_idx_list[index]
        labels_idx = d["labels_idx"]
        epoch = int(self._epoch)

        # Decode the full frame once per (labels_idx, lf_idx), via per-worker LRU.
        # The cached value is the fully sized frame dict (image, instances +
        # per-frame metadata); tiles clone from it so the cache stays pristine.
        frame = (
            self._frame_lru().get((labels_idx, d["lf_idx"]))
            if not d["is_negative"]
            else None
        )
        if frame is None:
            if d["is_negative"]:
                frame = self._load_negative_sample(d)
            else:
                img, instances = self._read_frame(d)
                frame = process_lf(
                    instances_list=instances,
                    img=img,
                    frame_idx=d["frame_idx"],
                    video_idx=d["video_idx"],
                    max_instances=self.max_instances,
                    user_instances_only=self.user_instances_only,
                )
            frame = self._to_sized_frame(frame)
            if not d["is_negative"]:
                self._frame_lru().put((labels_idx, d["lf_idx"]), frame)

        sample = {
            "image": frame["image"].clone(),
            "instances": frame["instances"].clone(),
            "video_idx": frame["video_idx"],
            "frame_idx": frame["frame_idx"],
            "orig_size": frame["orig_size"],
            "num_instances": frame["num_instances"],
        }

        # Resolve the tile origin: pinned for grid/val, drawn for train.
        if d["is_grid"]:
            sample["tile_origin"] = d["tile_origin"]
        else:
            rng = np.random.default_rng(
                tile_sample_seed(
                    self.base_seed,
                    epoch,
                    d["video_idx"],
                    d["frame_idx"],
                    d["sample_k"],
                )
            )
            centers = frame_foreground_centers(sample["instances"])
            sample["tile_origin"] = draw_tile_origin(
                centers,
                sample["image"].shape[-2:],
                self.tile_size,
                d["sample_k"],
                self.samples_per_frame,
                self.tile_fg_fraction,
                self.center_jitter,
                rng,
                pos_ratio=0.0 if d["is_negative"] else 1.0,
            )
            sample["aug_seed"] = tile_sample_seed(
                self.base_seed,
                epoch,
                d["video_idx"],
                d["frame_idx"],
                d["sample_k"],
                salt=1,
            )

        sample = self._apply_common_preprocessing(sample)

        img_hw = sample["image"].shape[-2:]

        # min_visible_keypoints: drop instances with too few in-tile keypoints
        # BEFORE OOB/confmap generation so a barely-clipped instance at a seam
        # does not seed a partial blob.
        inst = sample["instances"]
        inside = (
            (inst[..., 0] >= 0)
            & (inst[..., 0] < img_hw[1])
            & (inst[..., 1] >= 0)
            & (inst[..., 1] < img_hw[0])
        )
        keep = inside.sum(dim=-1) >= self.min_visible_keypoints
        inst[~keep] = torch.nan
        sample["instances"] = inst

        # NaN out remaining OOB keypoints at final tile resolution, then confmaps.
        sample["instances"] = filter_oob_points(
            sample["instances"], img_hw[0], img_hw[1]
        )
        sample["confidence_maps"] = generate_confmaps(
            sample["instances"],
            img_hw=img_hw,
            sigma=self.confmap_head_config.sigma,
            output_stride=self.confmap_head_config.output_stride,
        )

        sample["labels_idx"] = labels_idx
        if self.use_negative_frames:
            sample["is_negative"] = d["is_negative"]

        # Drop the transient aug seed so the batch collates uniformly.
        sample.pop("aug_seed", None)

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


def _masks_to_frame_canvas(
    mask_arrays: List[np.ndarray],
    orig_hw: Tuple[int, int],
) -> torch.Tensor:
    """Place decoded per-instance masks onto the full-frame image grid as one tensor.

    :func:`~sleap_nn.inference.segmentation_convert.decode_mask_to_image_res`
    returns a PARTIAL-frame array for any mask carrying a non-identity
    scale/offset (e.g. a top-down crop-centered mask in a pseudo-label ``.slp``),
    so the raw masks in a single frame can have DIFFERENT shapes and none may
    match the image. Each mask is zero-placed at its absolute top-left image
    coordinate (clamped to the frame; the decoded extent can be +/-1 px) so every
    mask shares the image's pixel grid *before* preprocessing — mirroring the
    single-mask placement in :class:`CenteredInstanceSegmentationDataset`.

    This is what lets the masks ride the image's size-match / scale / stride-pad
    chain in :meth:`BaseDataset._apply_common_preprocessing` and stay pixel-aligned
    with the image by construction (rather than a hand-maintained parallel resize).
    Uniform framing is also what makes the ``(1, K, H, W)`` stack well-defined —
    ragged per-instance shapes would otherwise raise in :func:`numpy.stack`.

    Args:
        mask_arrays: List of 2D boolean arrays at (possibly partial) image
            resolution, one per instance. May be empty.
        orig_hw: ``(height, width)`` of the loaded image (the target canvas).

    Returns:
        Float32 tensor of shape ``(1, K, *orig_hw)`` with each mask placed at its
        image-space origin (``K = 0`` is allowed and yields ``(1, 0, H, W)``).
    """
    h, w = int(orig_hw[0]), int(orig_hw[1])
    canvas = np.zeros((len(mask_arrays), h, w), dtype=np.float32)
    for k, m in enumerate(mask_arrays):
        mh, mw = m.shape[:2]
        h0, w0 = min(mh, h), min(mw, w)
        canvas[k, :h0, :w0] = m[:h0, :w0]
    return torch.from_numpy(canvas).unsqueeze(0)  # (1, K, H, W)


class GroupAwareBatchSampler(torch.utils.data.Sampler):
    """Group-aware batch sampler for contrastive embedding training.

    Its only job is to make the wanted positives/negatives co-occur in a batch. Modes:
      - ``pk``: P groups x K crops (the contrastive-standard sampler; guarantees K
        positives per group).
      - ``within_video``: pick ONE video, then P tracks x K crops from it — every
        in-batch pair is same-video so its relationship is KNOWN (the correct
        video-local sampler; cross-video pairs never co-occur). Falls back to PK when
        there is a single video.
      - ``random``: plain random batch (aug-view-only / self-supervised objectives).

    Yields lists of dataset indices. ``__len__`` = ``batches_per_epoch``.
    """

    def __init__(
        self,
        group_ids: np.ndarray,
        video_ids: np.ndarray,
        frame_ids: np.ndarray,
        kind: str = "pk",
        P: int = 8,
        K: int = 16,
        batches_per_epoch: Optional[int] = None,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        """Initialize the sampler from the dataset's per-crop arrays.

        Under multi-GPU (DDP) training, ``rank`` / ``world_size`` make each replica
        draw a DIFFERENT batch stream (the RNG is seeded per ``(seed, rank)``). With the
        per-replica ``batches_per_epoch``, the replicas process distinct batches whose
        gradients are all-reduced — a genuinely larger effective batch instead of every
        rank recomputing the identical batch. ``rank=0`` / ``world_size=1`` reproduces
        the single-GPU stream exactly.
        """
        self.group_ids = np.asarray(group_ids)
        self.video_ids = np.asarray(video_ids)
        self.frame_ids = np.asarray(frame_ids)
        self.kind = kind
        self.P = P
        self.K = K
        self.rank = int(rank)
        self.world_size = int(world_size)
        # Per-rank stream: offsetting the seed by the rank gives each replica an
        # independent batch sequence (SeedSequence decorrelates adjacent seeds), while
        # rank 0 reproduces the single-GPU stream (seed + 0 == seed) byte-for-byte.
        self.rng = np.random.default_rng(seed + self.rank)
        n = len(self.group_ids)
        self.all_idx = np.arange(n)

        self.uniq_groups = np.unique(self.group_ids)
        self.by_group = {g: self.all_idx[self.group_ids == g] for g in self.uniq_groups}

        self.uniq_videos = np.unique(self.video_ids)
        self.groups_in_video = {}
        self.by_video_group = {}
        for v in self.uniq_videos:
            vmask = self.video_ids == v
            vg = np.unique(self.group_ids[vmask])
            self.groups_in_video[v] = vg
            for g in vg:
                self.by_video_group[(v, g)] = self.all_idx[
                    vmask & (self.group_ids == g)
                ]
        elig = [v for v in self.uniq_videos if len(self.groups_in_video[v]) >= 2]
        self.elig_videos = np.array(elig) if elig else self.uniq_videos
        w = np.array([(self.video_ids == v).sum() for v in self.elig_videos], float)
        self.video_w = w / w.sum()

        self.batches_per_epoch = batches_per_epoch or max(
            1, int(np.ceil(n / (self.P * self.K)))
        )

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return self.batches_per_epoch

    def _pk_batch(self):
        P = min(self.P, len(self.uniq_groups))
        groups = self.rng.choice(self.uniq_groups, size=P, replace=False)
        batch = []
        for g in groups:
            pool = self.by_group[g]
            replace = len(pool) < self.K
            batch.extend(self.rng.choice(pool, size=self.K, replace=replace).tolist())
        return batch

    def _within_video_batch(self):
        v = self.elig_videos[self.rng.choice(len(self.elig_videos), p=self.video_w)]
        gs = self.groups_in_video[v]
        P = min(self.P, len(gs))
        groups = self.rng.choice(gs, size=P, replace=False)
        batch = []
        for g in groups:
            pool = self.by_video_group[(v, g)]
            replace = len(pool) < self.K
            batch.extend(self.rng.choice(pool, size=self.K, replace=replace).tolist())
        return batch

    def _random_batch(self):
        size = min(self.P * self.K, len(self.all_idx))
        return self.rng.choice(self.all_idx, size=size, replace=False).tolist()

    def __iter__(self) -> Iterator:
        """Yield ``batches_per_epoch`` lists of dataset indices."""
        for _ in range(self.batches_per_epoch):
            if self.kind == "pk":
                yield self._pk_batch()
            elif self.kind == "within_video":
                yield self._within_video_batch()
            elif self.kind == "random":
                yield self._random_batch()
            else:
                raise ValueError(f"Unknown sampler kind: {self.kind}")


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
        # Place them on the full-frame image grid as one float (1, K, H, W) tensor
        # so they ride the IDENTICAL size-match / scale / stride-pad chain as the
        # image inside ``_apply_common_preprocessing`` (kept as a float tensor
        # end-to-end and binarized ONCE just before target generation). This is
        # what keeps the whole-frame target registered to the padded image and
        # makes ragged/offset-carrying decoded masks well-defined (see
        # ``_masks_to_frame_canvas``).
        mask_arrays = [np.asarray(m, dtype=bool) for m in sample["masks"]]
        orig_img_hw = (image.shape[-2], image.shape[-1])
        sample_dict["masks"] = _masks_to_frame_canvas(mask_arrays, orig_img_hw)

        # Apply common preprocessing (RGB/grayscale, size matching, scaling, padding).
        # Co-transforms ``sample_dict["masks"]`` with the same geometry as the image.
        sample_dict = self._apply_common_preprocessing(sample_dict)

        img_hw = sample_dict["image"].shape[-2:]

        # Geometric augmentation: co-transform the per-instance masks with the SAME
        # flip/affine matrix as the image (nearest-neighbor). Applied here (post
        # size-match / resize / pad) so image and masks share a resolution, and
        # BEFORE centroid/heatmap/offset generation so those targets are derived
        # from the augmented masks. Bottom-up has no keypoints, so a dummy instances
        # tensor rides along; erase/mixup stay image-only.
        if (
            self.apply_aug
            and self.geometric_aug is not None
            and sample_dict["masks"].shape[1] > 0
        ):
            (
                sample_dict["image"],
                _,
                sample_dict["masks"],
            ) = apply_geometric_augmentation(
                sample_dict["image"],
                torch.zeros((1, 1, 1, 2), dtype=torch.float32),
                masks=sample_dict["masks"],
                **self.geometric_aug,
            )

        # Single re-binarization to bool arrays right before target generation.
        masks_t = sample_dict.pop("masks")
        mask_arrays = [masks_t[0, k].numpy() > 0.5 for k in range(masks_t.shape[1])]

        # Pre-compute mask centroids once for both center heatmap and offset heads
        centers = _compute_mask_centroids(mask_arrays) if len(mask_arrays) > 0 else []

        # Generate GT tensors
        foreground_mask = generate_foreground_mask(
            mask_arrays,
            img_hw=img_hw,
            output_stride=self.seg_head_config.output_stride,
            maxpool=bool(getattr(self.seg_head_config, "target_maxpool", False)),
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


class SemanticSegmentationDataset(BaseDataset):
    """Dataset class for whole-frame semantic (foreground/background) segmentation.

    Loads per-instance segmentation masks from ``LabeledFrame.masks`` and reduces
    them to a SINGLE binary foreground mask per frame (the union of every instance
    mask, area-downsampled to the segmentation head's output stride). There is NO
    instance grouping: unlike :class:`BottomUpSegmentationDataset`, no center
    heatmap, per-pixel offset field, or offset weight mask is generated — the model
    predicts one whole-frame foreground channel and decoding is a plain threshold.

    Masks are captured into the sample index at construction time (mirroring how
    keypoint instances are captured for caching), so ``__getitem__`` never needs
    a live ``Labels`` handle — this keeps it correct under the memory/disk image
    caching paths (where ``self.labels_list`` is ``None``).

    Note:
        Augmentation: intensity aug is applied to the image; geometric aug
        (rotation/scale/translate/flip) co-transforms every per-instance mask with the
        SAME affine matrix as the image (nearest-neighbor, re-binarized) at the
        preprocessed resolution, before the foreground mask is derived. Erase/mixup
        stay image-only. Mask resizing to the preprocessed image size handles scaling
        but is not pad-aware; for v1 train with ``scale=1.0`` and input dims divisible
        by ``max_stride`` (and prefer small rotation ranges, since a full-frame rotation
        can clip instances at the frame edge, as it does for bottom-up pose).

    Attributes:
        seg_head_config: Configuration for the segmentation head (``output_stride``).
    """

    def __init__(
        self,
        labels: List[sio.Labels],
        seg_head_config: DictConfig,
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
        """Return dict with image and whole-frame foreground mask for given index."""
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
        # Place them on the full-frame image grid as one float (1, K, H, W) tensor
        # so they ride the IDENTICAL size-match / scale / stride-pad chain as the
        # image inside ``_apply_common_preprocessing`` (kept as a float tensor
        # end-to-end and binarized ONCE just before target generation). This is
        # what keeps the whole-frame target registered to the padded image and
        # makes ragged/offset-carrying decoded masks well-defined (see
        # ``_masks_to_frame_canvas``).
        mask_arrays = [np.asarray(m, dtype=bool) for m in sample["masks"]]
        orig_img_hw = (image.shape[-2], image.shape[-1])
        sample_dict["masks"] = _masks_to_frame_canvas(mask_arrays, orig_img_hw)

        # Apply common preprocessing (RGB/grayscale, size matching, scaling, padding).
        # Co-transforms ``sample_dict["masks"]`` with the same geometry as the image.
        sample_dict = self._apply_common_preprocessing(sample_dict)

        img_hw = sample_dict["image"].shape[-2:]

        # Geometric augmentation: co-transform the per-instance masks with the SAME
        # flip/affine matrix as the image (nearest-neighbor). Applied here (post
        # size-match / resize / pad) so image and masks share a resolution, and
        # BEFORE the foreground mask is generated so it is derived from the augmented
        # masks. Semantic segmentation has no keypoints, so a dummy instances tensor
        # rides along; erase/mixup stay image-only.
        if (
            self.apply_aug
            and self.geometric_aug is not None
            and sample_dict["masks"].shape[1] > 0
        ):
            (
                sample_dict["image"],
                _,
                sample_dict["masks"],
            ) = apply_geometric_augmentation(
                sample_dict["image"],
                torch.zeros((1, 1, 1, 2), dtype=torch.float32),
                masks=sample_dict["masks"],
                **self.geometric_aug,
            )

        # Single re-binarization to bool arrays right before target generation.
        masks_t = sample_dict.pop("masks")
        mask_arrays = [masks_t[0, k].numpy() > 0.5 for k in range(masks_t.shape[1])]

        # Generate the single whole-frame foreground mask (union of all instance
        # masks, area-downsampled + re-binarized to the seg head's output stride).
        # No center/offset/weight targets: decoding is a plain threshold with no
        # instance grouping.
        foreground_mask = generate_foreground_mask(
            mask_arrays,
            img_hw=img_hw,
            output_stride=self.seg_head_config.output_stride,
            maxpool=bool(getattr(self.seg_head_config, "target_maxpool", False)),
        )

        sample_dict["foreground_mask"] = foreground_mask
        sample_dict["labels_idx"] = labels_idx

        return sample_dict


# Minimum number of tile-local foreground pixels for a mask to be "owned" by a tile.
# Guards against degenerate near-empty slivers (and all-zero background masks, whose
# `_compute_mask_centroids` fallback lands at the tile center) being kept as owners.
_MIN_OWNED_FG_PX = 4


class BottomUpSegmentationTiledDataset(BaseDataset):
    """Bottom-up segmentation dataset that emits fixed-size tiles (Phase C).

    Structural analogue of :class:`SingleInstanceTiledDataset` for the center-offset
    instance-segmentation pipeline. Each frame is decomposed into overlapping square
    tiles (foreground-aware random draws for training, a deterministic grid for
    validation). A frame is decoded / channel-coerced / scaled once (cached in a
    per-worker LRU together with its decoded per-instance masks) and reused across all
    of its tiles. Each tile is then cut out with a constant-zero pad; when geometric
    augmentation is enabled the tile is taken via a ``sqrt(2)`` halo so a rotation has
    valid context, co-transforming every per-instance mask with the SAME affine matrix
    as the image (nearest-neighbor, re-binarized). An ownership filter keeps only the
    masks whose (tile-local) centroid lands inside the tile so off-tile instances do
    not seed spurious center-offset regression, and the segmentation GT tensors
    (foreground mask, instance-center heatmap, per-pixel offsets) are generated on
    tile-local coordinates.

    Emits one sample per ``(frame, tile-slot)``; ``__len__`` is the total number of
    tile slots. Returned samples match the ``BottomUpSegmentationDataset`` key contract
    (plus an ``int32`` ``tile_origin`` of shape ``(2,)``), so the default collate and
    the ``BottomUpSegmentationLightningModule`` apply with no changes.

    Attributes:
        seg_head_config: Configuration for the segmentation (foreground) head.
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
        tiling: Optional[Union[DictConfig, Any]] = None,
        base_seed: int = 0,
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
            tiling=tiling,
            output_stride=seg_head_config.output_stride,
            base_seed=base_seed,
        )

        # Per-(frame, tile-slot) descriptors + contiguous per-frame index blocks.
        self.tile_idx_list = self._get_tile_idx_list(labels)
        self.frame_blocks = self._build_frame_blocks(self.tile_idx_list)

    def _get_lf_idx_list(self, labels: List[sio.Labels]) -> List[Dict]:
        """Return per-frame samples for frames that have segmentation masks.

        Mirrors ``BottomUpSegmentationDataset._get_lf_idx_list``: indexes frames by
        their masks (not keypoint instances) and captures the decoded mask arrays so
        ``__getitem__`` never needs a live ``Labels`` handle. This is what the base
        ``__init__`` stores as ``self.lf_idx_list`` (used for image caching);
        ``_get_tile_idx_list`` explodes it into per-tile descriptors.
        """
        from sleap_nn.inference.segmentation_convert import decode_mask_to_image_res

        lf_idx_list = []
        for labels_idx, label in enumerate(labels):
            for lf_idx, lf in enumerate(label):
                lf_masks = getattr(lf, "masks", None)
                if not lf_masks:
                    continue
                # Scale-aware decode up to the IMAGE-pixel grid (scale-1 GT masks
                # take the zero-copy fast path); see BottomUpSegmentationDataset.
                mask_arrays = [decode_mask_to_image_res(m) for m in lf_masks]
                if len(mask_arrays) == 0:
                    continue
                video_idx = label.videos.index(lf.video)
                lf_idx_list.append(
                    {
                        "labels_idx": labels_idx,
                        "lf_idx": lf_idx,
                        "video_idx": video_idx,
                        "frame_idx": lf.frame_idx,
                        "is_negative": False,
                        "instances": None,
                        "masks": mask_arrays,
                    }
                )
        return lf_idx_list

    def _get_tile_idx_list(self, labels: List[sio.Labels]) -> List[Dict]:
        """Return per-(frame, tile-slot) descriptors for the tiled seg dataset.

        Mirrors :meth:`BaseDataset._get_tile_idx_list` but keys off the mask-indexed
        per-frame list (``self.lf_idx_list``, built by :meth:`_get_lf_idx_list`) so the
        decoded ``mask_arrays`` ride along on every descriptor. Grid (val) pins one
        descriptor per :func:`generate_tile_grid` origin (sized-frame ``H, W`` taken
        from the decoded mask shape); foreground (train) emits ``samples_per_frame``
        slots with ``tile_origin=None`` (drawn at runtime). A frame's descriptors form
        a contiguous run (the block the sampler groups on).
        """
        tile_idx_list: List[Dict] = []
        for f in self.lf_idx_list:
            mask_arrays = f["masks"]
            # Masks are at image resolution; the sized (post-scale) frame H, W match
            # `apply_resizer`'s int(dim * scale) truncation used by `_frame_sized_hw`.
            mh, mw = mask_arrays[0].shape[:2]
            if self.scale != 1.0:
                sized_hw = (int(mh * self.scale), int(mw * self.scale))
            else:
                sized_hw = (int(mh), int(mw))

            if self.tile_sampling == "grid":
                origins = generate_tile_grid(
                    sized_hw,
                    tile_size=self.tile_size,
                    overlap=self.overlap,
                    output_stride=self.output_stride,
                    max_stride=self.max_stride,
                    min_overlap_fraction=self.min_overlap_fraction,
                )
            else:
                origins = [None] * self.samples_per_frame

            for sample_k, origin in enumerate(origins):
                tile_idx_list.append(
                    {
                        "labels_idx": f["labels_idx"],
                        "lf_idx": f["lf_idx"],
                        "video_idx": f["video_idx"],
                        "frame_idx": f["frame_idx"],
                        "masks": mask_arrays,
                        "sample_k": sample_k,
                        "tile_origin": origin,
                        "is_grid": self.tile_sampling == "grid",
                        "is_negative": False,
                    }
                )
        return tile_idx_list

    def __len__(self) -> int:
        """Return the number of tile samples (frames x tiles-per-frame)."""
        return len(self.tile_idx_list)

    def _read_image(self, d: Dict) -> np.ndarray:
        """Read a frame's raw HWC image (cache/disk/labels), restoring 2D -> 3D."""
        labels_idx = d["labels_idx"]
        lf_idx = d["lf_idx"]
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
        return img

    def _load_sized_frame(
        self, d: Dict
    ) -> Tuple[torch.Tensor, List[np.ndarray], tuple]:
        """Decode a frame once: channel-coerce + scale the image, size-match masks.

        Returns ``(image, mask_arrays, orig_hw)`` where ``image`` is a sized
        ``(1, C, H, W)`` tensor, ``mask_arrays`` are bool arrays at the sized image
        resolution, and ``orig_hw`` is the raw full-frame ``(H, W)`` before scaling.
        """
        img = self._read_image(d)  # HWC
        orig_hw = (int(img.shape[0]), int(img.shape[1]))

        image = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, axis=0)  # (1, C, H, W)
        image = torch.from_numpy(image.copy())

        if self.ensure_rgb:
            image = convert_to_rgb(image)
        elif self.ensure_grayscale:
            image = convert_to_grayscale(image)

        dummy = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
        image, _ = apply_resizer(image, dummy, scale=self.scale)

        mask_arrays = [np.asarray(m, dtype=bool) for m in d["masks"]]
        sized_hw = (image.shape[-2], image.shape[-1])
        if (sized_hw != orig_hw) and len(mask_arrays) > 0:
            # Resize masks to the sized image resolution with antialiased bilinear,
            # matching the image's ``apply_resizer`` / ``tvf.resize`` above
            # (standardized across every seg mask-geometry resize).
            target_h, target_w = sized_hw
            resized = []
            for m in mask_arrays:
                m_tensor = (
                    torch.from_numpy(m.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                )
                m_resized = F.interpolate(
                    m_tensor,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                resized.append(m_resized.squeeze().numpy() > 0.5)
            mask_arrays = resized

        return image, mask_arrays, orig_hw

    def _slice_halo(
        self,
        image: torch.Tensor,
        mask_arrays: List[np.ndarray],
        hy0: int,
        hx0: int,
        side: int,
    ) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """Slice a ``side x side`` window (top-left ``hy0, hx0``) with constant-zero pad.

        Slices both the image and every per-instance mask at the SAME offsets so they
        stay pixel-aligned. Out-of-bounds regions are zero.
        """
        _, C, H, W = image.shape
        ys, xs = max(0, hy0), max(0, hx0)
        ye, xe = min(H, hy0 + side), min(W, hx0 + side)
        win_img = image.new_zeros((1, C, side, side))
        win_masks = [np.zeros((side, side), dtype=bool) for _ in mask_arrays]
        if ye > ys and xe > xs:
            win_img[:, :, ys - hy0 : ye - hy0, xs - hx0 : xe - hx0] = image[
                :, :, ys:ye, xs:xe
            ]
            for k, m in enumerate(mask_arrays):
                win_masks[k][ys - hy0 : ye - hy0, xs - hx0 : xe - hx0] = m[ys:ye, xs:xe]
        return win_img, win_masks

    def __getitem__(self, index) -> Dict:
        """Return dict with image + segmentation GT for one tile of one frame."""
        d = self.tile_idx_list[index]
        labels_idx = d["labels_idx"]
        video_idx = d["video_idx"]
        frame_idx = d["frame_idx"]
        epoch = int(self._epoch)
        ts = self.tile_size

        # 1. Decode the full frame once per (labels_idx, lf_idx), via per-worker LRU.
        cached = self._frame_lru().get((labels_idx, d["lf_idx"]))
        if cached is None:
            cached = self._load_sized_frame(d)
            self._frame_lru().put((labels_idx, d["lf_idx"]), cached)
        image, mask_arrays, orig_hw = cached
        sized_hw = (image.shape[-2], image.shape[-1])

        # 2. Resolve the tile origin: pinned for grid/val, drawn for train.
        if d["is_grid"]:
            tile_origin = tuple(int(v) for v in d["tile_origin"])
            aug_seed = None
        else:
            rng = np.random.default_rng(
                tile_sample_seed(
                    self.base_seed, epoch, video_idx, frame_idx, d["sample_k"]
                )
            )
            cents = _compute_mask_centroids(mask_arrays)  # list of (x, y)
            centers = (
                torch.tensor(cents, dtype=torch.float32).reshape(-1, 2)
                if len(cents) > 0
                else torch.zeros((0, 2), dtype=torch.float32)
            )
            tile_origin = draw_tile_origin(
                centers,
                sized_hw,
                ts,
                d["sample_k"],
                self.samples_per_frame,
                self.tile_fg_fraction,
                self.center_jitter,
                rng,
            )
            aug_seed = tile_sample_seed(
                self.base_seed, epoch, video_idx, frame_idx, d["sample_k"], salt=1
            )

        y0, x0 = tile_origin
        apply_geo = self.apply_aug and self.geometric_aug is not None

        # 3. Cut the tile out of the frame (image + co-transformed masks). Under
        #    geometric aug, take a sqrt(2) halo centered on the tile center so the
        #    rotation has valid context, augment image + masks with the SAME matrix
        #    (nearest-neighbor, re-binarized), then trim the center tile back out.
        if apply_geo:
            halo = int(math.ceil(ts * math.sqrt(2)))
            hy0 = y0 - (halo - ts) // 2
            hx0 = x0 - (halo - ts) // 2
            halo_img, halo_masks = self._slice_halo(image, mask_arrays, hy0, hx0, halo)

            if len(halo_masks) > 0:
                halo_masks_t = torch.from_numpy(
                    np.stack([hm.astype(np.float32) for hm in halo_masks])
                ).unsqueeze(
                    0
                )  # (1, K, halo, halo)
                # The skia geometric backend samples its transform from the GLOBAL
                # numpy RNG (and torch); seed both so the halo path is reproducible.
                np.random.seed(aug_seed & 0xFFFFFFFF)
                torch.manual_seed(aug_seed)
                halo_img, _, halo_masks_t = apply_geometric_augmentation(
                    halo_img,
                    torch.zeros((1, 1, 1, 2), dtype=torch.float32),
                    masks=halo_masks_t,
                    **dict(self.geometric_aug),
                )
                halo_masks = [
                    halo_masks_t[0, k].numpy() > 0.5
                    for k in range(halo_masks_t.shape[1])
                ]

            # Trim the augmented halo back to `ts`, centered on the halo center:
            # crop_and_resize for the image (codebase convention), an equivalent
            # integer center-slice for the (axis-aligned) mask arrays.
            c = halo / 2.0
            bbox = make_centered_bboxes(
                torch.tensor([[c, c]], dtype=torch.float32), ts, ts
            )
            tile_image = crop_and_resize(halo_img, boxes=bbox, size=(ts, ts))
            off = (halo - ts) // 2
            tile_masks = [hm[off : off + ts, off : off + ts] for hm in halo_masks]
        else:
            # Fast path (no aug): direct slice + constant-zero pad, byte-identical.
            tile_image, tile_masks = self._slice_halo(image, mask_arrays, y0, x0, ts)

        # 4. Intensity aug (image only) + pad to stride (no-op when ts % max_stride==0).
        if self.apply_aug and self.intensity_aug is not None:
            tile_image, _ = apply_intensity_augmentation(
                tile_image,
                torch.zeros((1, 1, 1, 2), dtype=torch.float32),
                **self.intensity_aug,
            )
        tile_image = apply_pad_to_stride(tile_image, max_stride=self.max_stride)

        # 5. Ownership filter: keep only masks whose tile-local centroid lands inside
        #    [0, ts) x [0, ts) AND that have >= a few foreground px. Instances owned by
        #    a neighbor tile (centroid off-tile) are dropped so they do not seed
        #    off-tile center-offset regression.
        tile_centers = _compute_mask_centroids(tile_masks) if tile_masks else []
        owned_masks: List[np.ndarray] = []
        owned_centers: List[Tuple[float, float]] = []
        for m, (cx, cy) in zip(tile_masks, tile_centers):
            if int(m.sum()) < _MIN_OWNED_FG_PX:
                continue
            if 0.0 <= cx < ts and 0.0 <= cy < ts:
                owned_masks.append(m)
                owned_centers.append((cx, cy))

        # 6. Generate GT tensors on tile-local coordinates from the OWNED masks.
        img_hw = tile_image.shape[-2:]
        foreground_mask = generate_foreground_mask(
            owned_masks,
            img_hw=img_hw,
            output_stride=self.seg_head_config.output_stride,
            maxpool=bool(getattr(self.seg_head_config, "target_maxpool", False)),
        )
        center_heatmap = generate_center_heatmap(
            owned_masks,
            img_hw=img_hw,
            output_stride=self.center_head_config.output_stride,
            sigma=self.center_head_config.sigma,
            centers=owned_centers,
        )
        center_offsets, foreground_weight = generate_center_offsets(
            owned_masks,
            img_hw=img_hw,
            output_stride=self.offset_head_config.output_stride,
            centers=owned_centers,
        )

        return {
            "image": tile_image,
            "instances": torch.zeros((1, 1, 1, 2), dtype=torch.float32),
            "video_idx": torch.tensor(video_idx, dtype=torch.int32),
            "frame_idx": torch.tensor(frame_idx, dtype=torch.int32),
            "orig_size": torch.Tensor([orig_hw[0], orig_hw[1]]).unsqueeze(0),
            "num_instances": len(owned_masks),
            # Tiles are extracted in the model's input space (sizematcher bypassed),
            # so the effective scale is 1.0 (matches SingleInstanceTiledDataset).
            "eff_scale": torch.tensor(1.0, dtype=torch.float32),
            "foreground_mask": foreground_mask,
            "center_heatmap": center_heatmap,
            "center_offsets": center_offsets,
            "foreground_weight": foreground_weight,
            "labels_idx": labels_idx,
            "tile_origin": torch.tensor(tile_origin, dtype=torch.int32),
        }


class SemanticSegmentationTiledDataset(BaseDataset):
    """Whole-frame semantic (fg/bg) segmentation dataset emitting fixed-size tiles.

    Tiling analogue of :class:`SemanticSegmentationDataset` and a fg-only sibling of
    :class:`BottomUpSegmentationTiledDataset`: each frame is decomposed into
    overlapping square tiles (foreground-aware random draws for training, a
    deterministic grid for validation). A frame is decoded / channel-coerced / scaled
    once (cached in a per-worker LRU together with its decoded per-instance masks) and
    reused across all of its tiles. Each tile is then cut out with a constant-zero pad;
    when geometric augmentation is enabled the tile is taken via a ``sqrt(2)`` halo so a
    rotation has valid context, co-transforming every per-instance mask with the SAME
    affine matrix as the image (nearest-neighbor, re-binarized).

    Unlike the bottom-up center-offset pipeline there is NO instance grouping: the
    single ``foreground_mask`` GT is the union of ALL masks that touch the tile (no
    centroid-ownership filter, no center heatmap, no offsets, no foreground weight).

    Emits one sample per ``(frame, tile-slot)``; ``__len__`` is the total number of
    tile slots. Returned samples match the ``SemanticSegmentationDataset`` key contract
    (plus an ``int32`` ``tile_origin`` of shape ``(2,)``), so the default collate and
    the ``SemanticSegmentationLightningModule`` apply with no changes.

    Attributes:
        seg_head_config: Configuration for the segmentation (foreground) head.
    """

    def __init__(
        self,
        labels: List[sio.Labels],
        seg_head_config: DictConfig,
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
        tiling: Optional[Union[DictConfig, Any]] = None,
        base_seed: int = 0,
    ) -> None:
        """Initialize class attributes."""
        self.seg_head_config = seg_head_config
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
            tiling=tiling,
            output_stride=seg_head_config.output_stride,
            base_seed=base_seed,
        )

        # Per-(frame, tile-slot) descriptors + contiguous per-frame index blocks.
        self.tile_idx_list = self._get_tile_idx_list(labels)
        self.frame_blocks = self._build_frame_blocks(self.tile_idx_list)

    def _get_lf_idx_list(self, labels: List[sio.Labels]) -> List[Dict]:
        """Return per-frame samples for frames that have segmentation masks.

        Mirrors ``SemanticSegmentationDataset._get_lf_idx_list``: indexes frames by
        their masks (not keypoint instances) and captures the decoded mask arrays so
        ``__getitem__`` never needs a live ``Labels`` handle. This is what the base
        ``__init__`` stores as ``self.lf_idx_list`` (used for image caching);
        ``_get_tile_idx_list`` explodes it into per-tile descriptors.
        """
        from sleap_nn.inference.segmentation_convert import decode_mask_to_image_res

        lf_idx_list = []
        for labels_idx, label in enumerate(labels):
            for lf_idx, lf in enumerate(label):
                lf_masks = getattr(lf, "masks", None)
                if not lf_masks:
                    continue
                # Scale-aware decode up to the IMAGE-pixel grid (scale-1 GT masks
                # take the zero-copy fast path); see SemanticSegmentationDataset.
                mask_arrays = [decode_mask_to_image_res(m) for m in lf_masks]
                if len(mask_arrays) == 0:
                    continue
                video_idx = label.videos.index(lf.video)
                lf_idx_list.append(
                    {
                        "labels_idx": labels_idx,
                        "lf_idx": lf_idx,
                        "video_idx": video_idx,
                        "frame_idx": lf.frame_idx,
                        "is_negative": False,
                        "instances": None,
                        "masks": mask_arrays,
                    }
                )
        return lf_idx_list

    def _get_tile_idx_list(self, labels: List[sio.Labels]) -> List[Dict]:
        """Return per-(frame, tile-slot) descriptors for the tiled seg dataset.

        Mirrors :meth:`BaseDataset._get_tile_idx_list` but keys off the mask-indexed
        per-frame list (``self.lf_idx_list``, built by :meth:`_get_lf_idx_list`) so the
        decoded ``mask_arrays`` ride along on every descriptor. Grid (val) pins one
        descriptor per :func:`generate_tile_grid` origin (sized-frame ``H, W`` taken
        from the decoded mask shape); foreground (train) emits ``samples_per_frame``
        slots with ``tile_origin=None`` (drawn at runtime). A frame's descriptors form
        a contiguous run (the block the sampler groups on).
        """
        tile_idx_list: List[Dict] = []
        for f in self.lf_idx_list:
            mask_arrays = f["masks"]
            # Masks are at image resolution; the sized (post-scale) frame H, W match
            # `apply_resizer`'s int(dim * scale) truncation used by `_frame_sized_hw`.
            mh, mw = mask_arrays[0].shape[:2]
            if self.scale != 1.0:
                sized_hw = (int(mh * self.scale), int(mw * self.scale))
            else:
                sized_hw = (int(mh), int(mw))

            if self.tile_sampling == "grid":
                origins = generate_tile_grid(
                    sized_hw,
                    tile_size=self.tile_size,
                    overlap=self.overlap,
                    output_stride=self.output_stride,
                    max_stride=self.max_stride,
                    min_overlap_fraction=self.min_overlap_fraction,
                )
            else:
                origins = [None] * self.samples_per_frame

            for sample_k, origin in enumerate(origins):
                tile_idx_list.append(
                    {
                        "labels_idx": f["labels_idx"],
                        "lf_idx": f["lf_idx"],
                        "video_idx": f["video_idx"],
                        "frame_idx": f["frame_idx"],
                        "masks": mask_arrays,
                        "sample_k": sample_k,
                        "tile_origin": origin,
                        "is_grid": self.tile_sampling == "grid",
                        "is_negative": False,
                    }
                )
        return tile_idx_list

    def __len__(self) -> int:
        """Return the number of tile samples (frames x tiles-per-frame)."""
        return len(self.tile_idx_list)

    def _read_image(self, d: Dict) -> np.ndarray:
        """Read a frame's raw HWC image (cache/disk/labels), restoring 2D -> 3D."""
        labels_idx = d["labels_idx"]
        lf_idx = d["lf_idx"]
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
        return img

    def _load_sized_frame(
        self, d: Dict
    ) -> Tuple[torch.Tensor, List[np.ndarray], tuple]:
        """Decode a frame once: channel-coerce + scale the image, size-match masks.

        Returns ``(image, mask_arrays, orig_hw)`` where ``image`` is a sized
        ``(1, C, H, W)`` tensor, ``mask_arrays`` are bool arrays at the sized image
        resolution, and ``orig_hw`` is the raw full-frame ``(H, W)`` before scaling.
        """
        img = self._read_image(d)  # HWC
        orig_hw = (int(img.shape[0]), int(img.shape[1]))

        image = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, axis=0)  # (1, C, H, W)
        image = torch.from_numpy(image.copy())

        if self.ensure_rgb:
            image = convert_to_rgb(image)
        elif self.ensure_grayscale:
            image = convert_to_grayscale(image)

        dummy = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
        image, _ = apply_resizer(image, dummy, scale=self.scale)

        mask_arrays = [np.asarray(m, dtype=bool) for m in d["masks"]]
        sized_hw = (image.shape[-2], image.shape[-1])
        if (sized_hw != orig_hw) and len(mask_arrays) > 0:
            # Resize masks to the sized image resolution with antialiased bilinear,
            # matching the image's ``apply_resizer`` / ``tvf.resize`` above
            # (standardized across every seg mask-geometry resize).
            target_h, target_w = sized_hw
            resized = []
            for m in mask_arrays:
                m_tensor = (
                    torch.from_numpy(m.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                )
                m_resized = F.interpolate(
                    m_tensor,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                resized.append(m_resized.squeeze().numpy() > 0.5)
            mask_arrays = resized

        return image, mask_arrays, orig_hw

    def _slice_halo(
        self,
        image: torch.Tensor,
        mask_arrays: List[np.ndarray],
        hy0: int,
        hx0: int,
        side: int,
    ) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """Slice a ``side x side`` window (top-left ``hy0, hx0``) with constant-zero pad.

        Slices both the image and every per-instance mask at the SAME offsets so they
        stay pixel-aligned. Out-of-bounds regions are zero.
        """
        _, C, H, W = image.shape
        ys, xs = max(0, hy0), max(0, hx0)
        ye, xe = min(H, hy0 + side), min(W, hx0 + side)
        win_img = image.new_zeros((1, C, side, side))
        win_masks = [np.zeros((side, side), dtype=bool) for _ in mask_arrays]
        if ye > ys and xe > xs:
            win_img[:, :, ys - hy0 : ye - hy0, xs - hx0 : xe - hx0] = image[
                :, :, ys:ye, xs:xe
            ]
            for k, m in enumerate(mask_arrays):
                win_masks[k][ys - hy0 : ye - hy0, xs - hx0 : xe - hx0] = m[ys:ye, xs:xe]
        return win_img, win_masks

    def __getitem__(self, index) -> Dict:
        """Return dict with image + foreground GT for one tile of one frame."""
        d = self.tile_idx_list[index]
        labels_idx = d["labels_idx"]
        video_idx = d["video_idx"]
        frame_idx = d["frame_idx"]
        epoch = int(self._epoch)
        ts = self.tile_size

        # 1. Decode the full frame once per (labels_idx, lf_idx), via per-worker LRU.
        cached = self._frame_lru().get((labels_idx, d["lf_idx"]))
        if cached is None:
            cached = self._load_sized_frame(d)
            self._frame_lru().put((labels_idx, d["lf_idx"]), cached)
        image, mask_arrays, orig_hw = cached
        sized_hw = (image.shape[-2], image.shape[-1])

        # 2. Resolve the tile origin: pinned for grid/val, drawn for train. The
        #    foreground-aware draw is seeded from instance centroids (placement only;
        #    semantic GT does not use them), so tiles still land on foreground.
        if d["is_grid"]:
            tile_origin = tuple(int(v) for v in d["tile_origin"])
            aug_seed = None
        else:
            rng = np.random.default_rng(
                tile_sample_seed(
                    self.base_seed, epoch, video_idx, frame_idx, d["sample_k"]
                )
            )
            cents = _compute_mask_centroids(mask_arrays)  # list of (x, y)
            centers = (
                torch.tensor(cents, dtype=torch.float32).reshape(-1, 2)
                if len(cents) > 0
                else torch.zeros((0, 2), dtype=torch.float32)
            )
            tile_origin = draw_tile_origin(
                centers,
                sized_hw,
                ts,
                d["sample_k"],
                self.samples_per_frame,
                self.tile_fg_fraction,
                self.center_jitter,
                rng,
            )
            aug_seed = tile_sample_seed(
                self.base_seed, epoch, video_idx, frame_idx, d["sample_k"], salt=1
            )

        y0, x0 = tile_origin
        apply_geo = self.apply_aug and self.geometric_aug is not None

        # 3. Cut the tile out of the frame (image + co-transformed masks). Under
        #    geometric aug, take a sqrt(2) halo centered on the tile center so the
        #    rotation has valid context, augment image + masks with the SAME matrix
        #    (nearest-neighbor, re-binarized), then trim the center tile back out.
        if apply_geo:
            halo = int(math.ceil(ts * math.sqrt(2)))
            hy0 = y0 - (halo - ts) // 2
            hx0 = x0 - (halo - ts) // 2
            halo_img, halo_masks = self._slice_halo(image, mask_arrays, hy0, hx0, halo)

            if len(halo_masks) > 0:
                halo_masks_t = torch.from_numpy(
                    np.stack([hm.astype(np.float32) for hm in halo_masks])
                ).unsqueeze(
                    0
                )  # (1, K, halo, halo)
                # The skia geometric backend samples its transform from the GLOBAL
                # numpy RNG (and torch); seed both so the halo path is reproducible.
                np.random.seed(aug_seed & 0xFFFFFFFF)
                torch.manual_seed(aug_seed)
                halo_img, _, halo_masks_t = apply_geometric_augmentation(
                    halo_img,
                    torch.zeros((1, 1, 1, 2), dtype=torch.float32),
                    masks=halo_masks_t,
                    **dict(self.geometric_aug),
                )
                halo_masks = [
                    halo_masks_t[0, k].numpy() > 0.5
                    for k in range(halo_masks_t.shape[1])
                ]

            # Trim the augmented halo back to `ts`, centered on the halo center:
            # crop_and_resize for the image (codebase convention), an equivalent
            # integer center-slice for the (axis-aligned) mask arrays.
            c = halo / 2.0
            bbox = make_centered_bboxes(
                torch.tensor([[c, c]], dtype=torch.float32), ts, ts
            )
            tile_image = crop_and_resize(halo_img, boxes=bbox, size=(ts, ts))
            off = (halo - ts) // 2
            tile_masks = [hm[off : off + ts, off : off + ts] for hm in halo_masks]
        else:
            # Fast path (no aug): direct slice + constant-zero pad, byte-identical.
            tile_image, tile_masks = self._slice_halo(image, mask_arrays, y0, x0, ts)

        # 4. Intensity aug (image only) + pad to stride (no-op when ts % max_stride==0).
        if self.apply_aug and self.intensity_aug is not None:
            tile_image, _ = apply_intensity_augmentation(
                tile_image,
                torch.zeros((1, 1, 1, 2), dtype=torch.float32),
                **self.intensity_aug,
            )
        tile_image = apply_pad_to_stride(tile_image, max_stride=self.max_stride)

        # 5. Foreground GT: union of ALL masks touching the tile (NO ownership filter,
        #    NO center/offset heads). Empty (fully-zero) tile masks contribute nothing.
        img_hw = tile_image.shape[-2:]
        foreground_mask = generate_foreground_mask(
            tile_masks,
            img_hw=img_hw,
            output_stride=self.seg_head_config.output_stride,
            maxpool=bool(getattr(self.seg_head_config, "target_maxpool", False)),
        )
        # Count of masks with any foreground in this tile (logging/eval convenience).
        num_masks_in_tile = sum(1 for m in tile_masks if int(m.sum()) > 0)

        return {
            "image": tile_image,
            "instances": torch.zeros((1, 1, 1, 2), dtype=torch.float32),
            "video_idx": torch.tensor(video_idx, dtype=torch.int32),
            "frame_idx": torch.tensor(frame_idx, dtype=torch.int32),
            "orig_size": torch.Tensor([orig_hw[0], orig_hw[1]]).unsqueeze(0),
            "num_instances": num_masks_in_tile,
            # Tiles are extracted in the model's input space (sizematcher bypassed),
            # so the effective scale is 1.0 (matches SingleInstanceTiledDataset).
            "eff_scale": torch.tensor(1.0, dtype=torch.float32),
            "foreground_mask": foreground_mask,
            "labels_idx": labels_idx,
            "tile_origin": torch.tensor(tile_origin, dtype=torch.int32),
        }


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

    # Gate the lossy disk cache for the embedding (appearance / re-ID) model (SPEC §5.1):
    # `torch_dataset_cache_img_disk` writes source frames as JPEG, which silently
    # degrades an appearance model. The disk cache has no lossless format, so refuse it
    # for embedding and require the in-memory cache (or an uncached fw) instead.
    if model_type == "embedding" and cache_imgs == "disk":
        raise ValueError(
            "data_pipeline_fw='torch_dataset_cache_img_disk' is not supported for the "
            "`embedding` model type: the disk cache stores frames as JPEG, which "
            "silently degrades an appearance / re-ID model. Use "
            "data_pipeline_fw='torch_dataset_cache_img_memory' (lossless, recommended) "
            "or 'torch_dataset' (uncached)."
        )

    if use_negative_frames and model_type in (
        "centered_instance",
        "multi_class_topdown",
        "centered_instance_segmentation",
        "embedding",
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
        tiling = OmegaConf.select(
            config, "data_config.preprocessing.tiling", default=None
        )
        if tiling is not None and tiling.enabled:
            # Tiled bottom-up segmentation training. Train draws foreground-aware
            # tiles (aug on); val always uses a deterministic full-coverage grid
            # (no aug). Masks are co-transformed with the image under the halo path.
            base_seed = config.trainer_config.seed or 0
            train_dataset = BottomUpSegmentationTiledDataset(
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
                tiling=OmegaConf.merge(tiling, {"sampling": tiling.sampling}),
                base_seed=base_seed,
            )
            val_dataset = BottomUpSegmentationTiledDataset(
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
                tiling=OmegaConf.merge(tiling, {"sampling": "grid"}),
                base_seed=base_seed,
            )
        else:
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

    elif model_type == "semantic_segmentation":
        seg_cfg = config.model_config.head_configs.semantic_segmentation
        tiling = OmegaConf.select(
            config, "data_config.preprocessing.tiling", default=None
        )
        if tiling is not None and tiling.enabled:
            # Tiled whole-frame semantic segmentation. Train draws foreground-aware
            # tiles (aug on); val always uses a deterministic full-coverage grid (no
            # aug). Masks are co-transformed with the image under the halo path.
            base_seed = config.trainer_config.seed or 0
            train_dataset = SemanticSegmentationTiledDataset(
                labels=train_labels,
                seg_head_config=seg_cfg.segmentation,
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
                tiling=OmegaConf.merge(tiling, {"sampling": tiling.sampling}),
                base_seed=base_seed,
            )
            val_dataset = SemanticSegmentationTiledDataset(
                labels=val_labels,
                seg_head_config=seg_cfg.segmentation,
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
                tiling=OmegaConf.merge(tiling, {"sampling": "grid"}),
                base_seed=base_seed,
            )
        else:
            train_dataset = SemanticSegmentationDataset(
                labels=train_labels,
                seg_head_config=seg_cfg.segmentation,
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
            val_dataset = SemanticSegmentationDataset(
                labels=val_labels,
                seg_head_config=seg_cfg.segmentation,
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

    elif model_type == "embedding":
        emb_cfg = config.model_config.head_configs.embedding.embedding
        # Whether a per-video track name may stand in as a global animal identity for
        # detections without a real `sio.Identity` (the pre-Identity convention).
        track_names_are_global = bool(
            OmegaConf.select(
                config, "data_config.identity.track_names_are_global", default=False
            )
        )
        # One global-identity vocabulary shared by train + val (the group_id space):
        # `sio.Identity` names, else track names under `track_names_are_global`.
        class_names = resolve_embedding_class_names(
            train_labels + val_labels, track_names_are_global=track_names_are_global
        )
        # Training-group key (global identity vs per-video tracklet).
        id_scope = OmegaConf.select(
            emb_cfg, "objective.positives.scope", default="global_id"
        )
        max_stride = config.model_config.backbone_config[f"{backbone_type}"][
            "max_stride"
        ]
        crop_size = config.data_config.preprocessing.crop_size
        max_hw = (
            config.data_config.preprocessing.max_height,
            config.data_config.preprocessing.max_width,
        )
        # Grayscale is the embedding default (helps the cross-video gap), but the user
        # can opt into RGB via `preprocessing.ensure_rgb`. A 3ch ImageNet backbone
        # repeats the gray channel in Model.forward, so grayscale data still works with
        # convnext/swint. When neither flag is set, default to grayscale.
        emb_ensure_rgb = bool(config.data_config.preprocessing.ensure_rgb)
        emb_ensure_grayscale = (
            bool(config.data_config.preprocessing.ensure_grayscale)
            or not emb_ensure_rgb
        )
        crop_centering = OmegaConf.select(
            config, "data_config.preprocessing.crop_centering", default="auto"
        )
        train_dataset = EmbeddingDataset(
            labels=train_labels,
            crop_size=crop_size,
            class_names=class_names,
            embedding_head_config=emb_cfg,
            max_stride=max_stride,
            id_scope=id_scope,
            track_names_are_global=track_names_are_global,
            crop_centering=crop_centering,
            user_instances_only=config.data_config.user_instances_only,
            ensure_rgb=emb_ensure_rgb,
            ensure_grayscale=emb_ensure_grayscale,
            # Two contrastive views via the standard config-driven skia aug (per crop).
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
            apply_aug=config.data_config.use_augmentations_train,
            scale=config.data_config.preprocessing.scale,
            max_hw=max_hw,
            cache_img=cache_imgs,
            cache_img_path=train_cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
        )
        val_dataset = EmbeddingDataset(
            labels=val_labels,
            crop_size=crop_size,
            class_names=class_names,
            embedding_head_config=emb_cfg,
            max_stride=max_stride,
            id_scope=id_scope,
            track_names_are_global=track_names_are_global,
            crop_centering=crop_centering,
            user_instances_only=config.data_config.user_instances_only,
            ensure_rgb=emb_ensure_rgb,
            ensure_grayscale=emb_ensure_grayscale,
            scale=config.data_config.preprocessing.scale,
            max_hw=max_hw,
            cache_img=cache_imgs,
            cache_img_path=val_cache_img_path,
            use_existing_imgs=use_existing_imgs,
            rank=rank,
            parallel_caching=parallel_caching,
            cache_workers=cache_workers,
        )

        if len(train_dataset) == 0:
            message = (
                "The embedding train dataset is empty: no tracked detections whose "
                f"track name is in the resolved vocabulary ({len(class_names)} "
                "class(es)) were found. Check that the labels carry tracked "
                "masks/instances and that the track names match."
            )
            logger.error(message)
            raise ValueError(message)

    else:
        tiling = OmegaConf.select(
            config, "data_config.preprocessing.tiling", default=None
        )
        if tiling is not None and tiling.enabled:
            # Tiled single-instance training. Train draws foreground-aware tiles;
            # val always uses a deterministic full-coverage grid (no aug).
            base_seed = config.trainer_config.seed or 0
            train_dataset = SingleInstanceTiledDataset(
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
                tiling=OmegaConf.merge(tiling, {"sampling": tiling.sampling}),
                base_seed=base_seed,
            )
            val_dataset = SingleInstanceTiledDataset(
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
                tiling=OmegaConf.merge(tiling, {"sampling": "grid"}),
                base_seed=base_seed,
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

    # Embedding training uses a group-aware BATCH sampler (PK / within_video) so the
    # wanted positives/negatives co-occur in each batch. The batch_sampler is mutually
    # exclusive with batch_size/shuffle/sampler, so build a distinct loader here.
    if isinstance(train_dataset, EmbeddingDataset):
        sp = "model_config.head_configs.embedding.embedding.objective.sampler"
        batch_sampler = GroupAwareBatchSampler(
            group_ids=train_dataset.group_ids,
            video_ids=train_dataset.video_ids,
            frame_ids=train_dataset.frame_ids,
            kind=OmegaConf.select(config, f"{sp}.kind", default="pk"),
            P=OmegaConf.select(config, f"{sp}.groups_per_batch", default=8),
            K=OmegaConf.select(config, f"{sp}.samples_per_group", default=16),
            batches_per_epoch=max(1, round(train_steps_per_epoch / trainer_devices)),
            seed=OmegaConf.select(config, "trainer_config.seed", default=0) or 0,
            # DDP: each rank draws an INDEPENDENT (seed + rank) batch stream of the same
            # length over the full dataset — not a partition — so the all-reduced gradient
            # aggregates roughly world_size x P x K decorrelated crops per step.
            rank=rank if rank is not None else 0,
            world_size=trainer_devices,
        )
        # Use a plain DataLoader (NOT InfiniteDataLoader): the GroupAwareBatchSampler
        # is already epoch-bounded (yields `batches_per_epoch` batches per __iter__,
        # re-randomized each epoch via its rng), so the infinite-recycling wrapper is
        # unnecessary. Critically, InfiniteDataLoader eagerly creates its worker
        # iterator + wraps the sampler in an infinite _RepeatSampler, which deadlocks
        # under Lightning with num_workers>0; the plain loader iterates with workers
        # correctly (so heavy CPU/skia aug can be parallelized).
        train_nw = config.trainer_config.train_data_loader.num_workers
        val_nw = config.trainer_config.val_data_loader.num_workers
        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=batch_sampler,
            num_workers=train_nw,
            pin_memory=pin_memory,
            persistent_workers=train_nw > 0,
        )
        val_data_loader = DataLoader(
            dataset=val_dataset,
            shuffle=False,
            batch_size=config.trainer_config.val_data_loader.batch_size,
            num_workers=val_nw,
            pin_memory=pin_memory,
            persistent_workers=val_nw > 0,
        )
        return train_data_loader, val_data_loader

    tiling = OmegaConf.select(config, "data_config.preprocessing.tiling", default=None)
    tiling_enabled = tiling is not None and tiling.enabled

    # Under tiling, a frame-grouped block sampler REPLACES DistributedSampler (it
    # shards whole frame blocks so a frame's tiles stay together and DDP-disjoint),
    # and a per-worker RNG init de-correlates the halo augmentation streams.
    worker_init_fn = tiling_worker_init_fn if tiling_enabled else None

    if tiling_enabled:
        train_sampler = FrameGroupedTileSampler(
            train_dataset.frame_blocks,
            batch_size=config.trainer_config.train_data_loader.batch_size,
            shuffle=config.trainer_config.train_data_loader.shuffle,
            seed=config.trainer_config.seed or 0,
            num_replicas=trainer_devices,
            rank=(rank if rank is not None else 0),
        )
    else:
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
        worker_init_fn=worker_init_fn,
        persistent_workers=(
            True if config.trainer_config.train_data_loader.num_workers > 0 else None
        ),
        prefetch_factor=(
            config.trainer_config.train_data_loader.batch_size
            if config.trainer_config.train_data_loader.num_workers > 0
            else None
        ),
    )

    if tiling_enabled:
        val_sampler = FrameGroupedTileSampler(
            val_dataset.frame_blocks,
            batch_size=config.trainer_config.val_data_loader.batch_size,
            shuffle=False,
            seed=config.trainer_config.seed or 0,
            num_replicas=trainer_devices,
            rank=(rank if rank is not None else 0),
        )
    else:
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
        worker_init_fn=worker_init_fn,
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
