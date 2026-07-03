"""Shared tiling primitives for splitting large images into overlapping tiles.

This module holds the low-level building blocks used to tile large frames into
fixed-size square tiles for training and inference. :func:`generate_tile_grid`
computes the deterministic grid of snapped tile origins in input-pixel space
used at inference time. The remaining primitives support random tile sampling
during training:

* :func:`frame_foreground_centers` extracts valid keypoint (or centroid)
  locations to bias tiles toward foreground.
* :func:`draw_tile_origin` draws a single (possibly foreground-biased) tile
  origin for a frame.
* :func:`extract_tile` crops a tile (with an optional geometric-augmentation
  halo) and returns tile-local instance coordinates.
* :func:`tile_sample_seed`, :func:`tiling_worker_init_fn`, :class:`_FrameLRU`
  and :class:`FrameGroupedTileSampler` provide the deterministic seeding, frame
  caching and frame-grouped batching used by the tiling dataset.
"""

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler, get_worker_info

from sleap_nn.data.augmentation import apply_geometric_augmentation
from sleap_nn.data.skia_augmentation import crop_and_resize_skia as crop_and_resize
from sleap_nn.data.instance_cropping import make_centered_bboxes


def _axis_tile_origins(
    image_dim: int,
    tile_size: int,
    overlap: int,
    output_stride: int,
    max_stride: int,
    min_overlap_fraction: float,
) -> List[int]:
    """Compute snapped tile origins along a single axis.

    This helper operates on a single axis (either the height or the width) and
    returns the sorted, de-duplicated list of tile origins in input-pixel space.
    See :func:`generate_tile_grid` for the full description of the arguments and
    the snapping conventions.

    Args:
        image_dim: Size of the axis in pixels.
        tile_size: Side length of the (square) tile in pixels.
        overlap: Requested overlap between neighboring tiles in pixels.
        output_stride: Origins are snapped down to multiples of this value.
        max_stride: Coarse stride used to snap the step between tiles when it is
            compatible with ``output_stride``.
        min_overlap_fraction: Minimum overlap expressed as a fraction of
            ``tile_size``. If ``overlap`` is below this floor, the step is
            shrunk so the effective overlap meets the floor.

    Returns:
        A sorted list of unique tile origins (the top edge for the height axis
        or the left edge for the width axis). Each origin is a multiple of
        ``output_stride`` and lies within ``[0, image_dim - tile_size]`` when
        ``image_dim > tile_size``. When ``image_dim <= tile_size`` the single
        origin ``0`` is returned (the tile is zero-padded to size later).
    """
    # Frames no larger than a tile yield a single origin; the extractor pads.
    if image_dim <= tile_size:
        return [0]

    # Enforce a minimum overlap floor by (possibly) shrinking the step.
    eff_overlap = max(overlap, round(min_overlap_fraction * tile_size))
    step = tile_size - eff_overlap

    # Snap the step to a coarse multiple: prefer ``max_stride`` when it is large
    # enough and compatible with ``output_stride``, otherwise fall back to
    # ``output_stride`` so tiles land on network-friendly boundaries.
    if step >= max_stride and max_stride % output_stride == 0:
        snap_unit = max_stride
    else:
        snap_unit = output_stride
    step = (step // snap_unit) * snap_unit
    if step < output_stride:
        step = output_stride

    # Walk interior origins, snapping each down to a multiple of output_stride,
    # while the full tile still fits strictly inside the frame.
    origins: List[int] = []
    origin = 0
    while origin + tile_size < image_dim:
        origins.append((origin // output_stride) * output_stride)
        origin += step

    # Append the inward-snapped final origin so the far edge is fully covered
    # without overrunning the frame, de-duplicating if it repeats the last one.
    last_origin = ((image_dim - tile_size) // output_stride) * output_stride
    if not origins or origins[-1] != last_origin:
        origins.append(last_origin)

    return origins


def generate_tile_grid(
    image_hw: Tuple[int, int],
    tile_size: int,
    overlap: int,
    output_stride: int,
    max_stride: int = 1,
    min_overlap_fraction: float = 0.25,
) -> List[Tuple[int, int]]:
    """Compute snapped square-tile top-left origins covering an image.

    Origins are computed independently for the height and width axes and then
    combined via a Cartesian product in row-major (``y`` then ``x``) order. Each
    origin is snapped to a multiple of ``output_stride`` and constrained so that
    tiles never overrun the frame while still covering the far edges.

    Args:
        image_hw: Image size as a ``(height, width)`` tuple in pixels.
        tile_size: Side length of the (square) tile in pixels.
        overlap: Requested overlap between neighboring tiles in pixels. This is
            raised to ``round(min_overlap_fraction * tile_size)`` if it falls
            below that floor.
        output_stride: Tile origins are snapped down to multiples of this value.
            This is typically the output stride of the model so that tiles align
            to the network's prediction grid.
        max_stride: Coarse stride used to snap the step between neighboring
            tiles when it is at least as large as ``max_stride`` and evenly
            divisible into a multiple of ``output_stride``. Defaults to ``1``.
        min_overlap_fraction: Minimum overlap expressed as a fraction of
            ``tile_size``. Defaults to ``0.25``.

    Returns:
        A list of ``(y0, x0)`` tile origins (top-left corners) in input-pixel
        space, ordered row-major (all tiles for the first row of origins, then
        the next, and so on). At least one origin is always returned.

    Notes:
        The union of the resulting tiles covers the frame up to the last origin
        that can be placed on the ``output_stride`` grid without overrunning.
        When ``(image_dim - tile_size)`` is a multiple of ``output_stride`` (the
        typical case) this is the entire frame including the right/bottom edge.
    """
    y_origins = _axis_tile_origins(
        image_dim=image_hw[0],
        tile_size=tile_size,
        overlap=overlap,
        output_stride=output_stride,
        max_stride=max_stride,
        min_overlap_fraction=min_overlap_fraction,
    )
    x_origins = _axis_tile_origins(
        image_dim=image_hw[1],
        tile_size=tile_size,
        overlap=overlap,
        output_stride=output_stride,
        max_stride=max_stride,
        min_overlap_fraction=min_overlap_fraction,
    )
    return [(y0, x0) for y0 in y_origins for x0 in x_origins]


def frame_foreground_centers(
    instances: torch.Tensor, use_centroid: bool = False
) -> torch.Tensor:
    """Extract valid foreground keypoint (or centroid) locations for a frame.

    These locations are used by :func:`draw_tile_origin` to bias sampled tiles
    toward regions of the frame that actually contain animals.

    Args:
        instances: Sized-frame instance keypoints of shape ``(1, I, N, 2)`` in
            ``(x, y)`` order, where ``I`` is the number of instances and ``N``
            the number of nodes. Missing keypoints are encoded as ``NaN``.
        use_centroid: If ``True``, first reduce each instance to a single
            centroid by taking the ``NaN``-aware mean over the node axis before
            dropping invalid rows. If ``False`` (default), every valid keypoint
            is returned as its own candidate center.

    Returns:
        A ``(M, 2)`` tensor of ``(x, y)`` centers with all-``NaN``-containing
        rows removed. ``M`` is ``0`` when the frame has no valid keypoints.
        This function is pure and does not consume any random state.
    """
    frame = instances[0]  # (I, N, 2)
    if use_centroid:
        # NaN-aware mean over the node axis -> one center per instance.
        centers = torch.nanmean(frame, dim=1)  # (I, 2)
    else:
        centers = frame.reshape(-1, 2)  # (I * N, 2)

    valid = ~torch.isnan(centers).any(dim=-1)
    return centers[valid]


def draw_tile_origin(
    centers: torch.Tensor,
    frame_hw: Tuple[int, int],
    tile_size: int,
    sample_k: int,
    samples_per_frame: int,
    tile_fg_fraction: float,
    center_jitter: float,
    rng: np.random.Generator,
    pos_ratio: float = 1.0,
) -> Tuple[int, int]:
    """Draw a single (optionally foreground-biased) tile origin for a frame.

    The origin is returned *unclamped*; :func:`extract_tile` is responsible for
    handling origins that fall partially (or fully) outside the frame via
    zero-padding.

    Foreground slots are assigned deterministically by index: the last
    ``tile_fg_fraction`` of the ``samples_per_frame`` slots are foreground slots
    (``force_fg`` is ``True`` for them). Foreground slots draw a tile centered on
    a random valid center (plus jitter); all other slots draw a uniformly random
    origin. When there are no valid centers or ``pos_ratio`` is ``0.0``, every
    slot falls back to a uniformly random origin.

    Args:
        centers: ``(M, 2)`` tensor of candidate ``(x, y)`` centers (may be
            empty), as returned by :func:`frame_foreground_centers`.
        frame_hw: Frame size as a ``(height, width)`` tuple in pixels.
        tile_size: Side length of the square tile in pixels.
        sample_k: Index of this sample within the frame, in ``[0,
            samples_per_frame)``. Determines whether this is a foreground slot.
        samples_per_frame: Total number of tiles drawn per frame.
        tile_fg_fraction: Fraction of the per-frame slots that are foreground
            slots (the trailing slots by index).
        center_jitter: Jitter magnitude as a fraction of ``tile_size / 2``
            applied independently to each axis of a foreground draw.
        rng: A seeded :class:`numpy.random.Generator` used for all draws.
        pos_ratio: If ``0.0``, disables foreground sampling entirely (all slots
            uniform). Defaults to ``1.0``.

    Returns:
        The ``(y0, x0)`` top-left origin of the tile as Python ints. The origin
        is not clamped to the frame.
    """
    H, W = frame_hw
    M = centers.shape[0]

    # The trailing ``tile_fg_fraction`` of slots are foreground slots.
    force_fg = sample_k >= round(samples_per_frame * (1.0 - tile_fg_fraction))

    if M == 0 or pos_ratio == 0.0 or not force_fg:
        # Uniformly random origin (background / fallback).
        x0 = int(rng.integers(0, max(1, W - tile_size + 1)))
        y0 = int(rng.integers(0, max(1, H - tile_size + 1)))
        return y0, x0

    # Foreground draw: center a jittered tile on a random valid center.
    c = centers[rng.integers(M)]
    c_x = float(c[0])
    c_y = float(c[1])
    jitter_x = float(rng.uniform(-1, 1)) * center_jitter * (tile_size / 2.0)
    jitter_y = float(rng.uniform(-1, 1)) * center_jitter * (tile_size / 2.0)
    x0 = int(round(c_x - tile_size / 2.0 + jitter_x))
    y0 = int(round(c_y - tile_size / 2.0 + jitter_y))
    return y0, x0


def extract_tile(
    image: torch.Tensor,
    instances: torch.Tensor,
    tile_origin: Tuple[int, int],
    tile_size: int,
    *,
    apply_geometric: bool = False,
    geometric_kwargs: Optional[Dict] = None,
    symmetric_inds: Optional[Sequence[Tuple[int, int]]] = None,
    rng_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop a fixed-size tile from a frame and return tile-local instances.

    Two code paths are provided:

    * **Fast path** (``apply_geometric=False``, default): a pure tensor slice
      with constant-zero padding. The tile pixels are byte-identical to the
      source pixels (no resampling), even for origins that fall partially
      outside the frame.
    * **Halo path** (``apply_geometric=True``): a larger halo (side
      ``ceil(tile_size * sqrt(2))``) is extracted so that a subsequent geometric
      augmentation (e.g. rotation) has valid context, then the augmented halo is
      cropped back down to ``tile_size`` centered on the halo center.

    Args:
        image: Sized-frame image of shape ``(1, C, H, W)``.
        instances: Sized-frame instance keypoints of shape ``(1, I, N, 2)`` in
            ``(x, y)`` order (``NaN`` = missing).
        tile_origin: The ``(y0, x0)`` top-left origin of the tile. May be
            negative or extend past the frame; out-of-bounds pixels are zeroed.
        tile_size: Side length of the square tile in pixels.
        apply_geometric: If ``True``, use the halo path and apply geometric
            augmentation before cropping. Defaults to ``False``.
        geometric_kwargs: Extra keyword arguments forwarded to
            :func:`sleap_nn.data.augmentation.apply_geometric_augmentation`.
        symmetric_inds: Symmetric node-index pairs forwarded to the geometric
            augmentation (for left/right flips).
        rng_seed: If not ``None``, seeds ``torch`` before augmentation so the
            halo path is reproducible.

    Returns:
        A tuple ``(tile, tile_instances)`` where ``tile`` has shape
        ``(1, C, tile_size, tile_size)`` and ``tile_instances`` has shape
        ``(1, I, N, 2)`` in tile-local ``(x, y)`` coordinates.
    """
    y0, x0 = tile_origin
    _, C, H, W = image.shape

    if not apply_geometric:
        # Fast path: slice + constant-zero pad, byte-identical to the source.
        ys, xs = max(0, y0), max(0, x0)
        ye, xe = min(H, y0 + tile_size), min(W, x0 + tile_size)
        tile = image.new_zeros((1, C, tile_size, tile_size))
        if ye > ys and xe > xs:
            tile[:, :, ys - y0 : ye - y0, xs - x0 : xe - x0] = image[:, :, ys:ye, xs:xe]
        tile_inst = instances.clone()
        tile_inst[..., 0] -= x0
        tile_inst[..., 1] -= y0
        return tile, tile_inst

    # Halo path: extract a larger, tile-center-aligned halo (same slice + pad).
    halo = int(math.ceil(tile_size * math.sqrt(2)))
    hy0 = y0 - (halo - tile_size) // 2
    hx0 = x0 - (halo - tile_size) // 2

    ys, xs = max(0, hy0), max(0, hx0)
    ye, xe = min(H, hy0 + halo), min(W, hx0 + halo)
    halo_img = image.new_zeros((1, C, halo, halo))
    if ye > ys and xe > xs:
        halo_img[:, :, ys - hy0 : ye - hy0, xs - hx0 : xe - hx0] = image[
            :, :, ys:ye, xs:xe
        ]
    halo_inst = instances.clone()
    halo_inst[..., 0] -= hx0
    halo_inst[..., 1] -= hy0

    if rng_seed is not None:
        # The skia geometric-augmentation backend samples its affine transform from
        # the GLOBAL numpy RNG (not torch), so seed numpy to make the halo path
        # reproducible; also seed torch for any torch-based augmentation backend.
        np.random.seed(rng_seed & 0xFFFFFFFF)
        torch.manual_seed(rng_seed)
    aug = apply_geometric_augmentation(
        halo_img,
        halo_inst,
        symmetric_inds=symmetric_inds,
        **(geometric_kwargs or {}),
    )
    halo_img, halo_inst = aug[0], aug[1]

    # Trim the augmented halo back to ``tile_size``, centered on the halo center.
    c = halo / 2.0
    bbox = make_centered_bboxes(
        torch.tensor([[c, c]], dtype=torch.float32), tile_size, tile_size
    )  # (1, 4, 2)
    tile = crop_and_resize(halo_img, boxes=bbox, size=(tile_size, tile_size))
    tile_inst = halo_inst - bbox[0][0]
    return tile, tile_inst


def tile_sample_seed(
    base_seed: int,
    epoch: int,
    video_idx: int,
    frame_idx: int,
    sample_k: int,
    salt: int = 0,
) -> int:
    """Derive a deterministic per-sample seed for tile sampling.

    Uses :class:`numpy.random.SeedSequence` so that the returned seed is a
    stable, well-mixed function of all inputs (independent of worker/process).

    Args:
        base_seed: Global base seed for the run.
        epoch: Current training epoch.
        video_idx: Index of the source video.
        frame_idx: Index of the source frame within the video.
        sample_k: Index of the sample within the frame.
        salt: Extra disambiguating value (e.g. to derive independent streams).
            Defaults to ``0``.

    Returns:
        A 32-bit unsigned integer seed as a Python int.
    """
    ss = np.random.SeedSequence(
        [base_seed, epoch, video_idx, frame_idx, sample_k, salt]
    )
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def tiling_worker_init_fn(worker_id: int) -> None:
    """Seed ``numpy`` and ``torch`` per dataloader worker for tiling.

    Reads ``base_seed`` off the worker's dataset (defaulting to ``0``) and mixes
    it with the worker id so each worker gets an independent, reproducible
    stream.

    Args:
        worker_id: The dataloader worker id (as passed by PyTorch).
    """
    info = get_worker_info()
    base = getattr(info.dataset, "base_seed", 0)
    seed = (base * 2_654_435_761 + worker_id) & 0xFFFFFFFF
    np.random.seed(seed)
    torch.manual_seed(seed)


_FRAME_LRU_CAPACITY = 2


class _FrameLRU:
    """A tiny fixed-capacity LRU cache for decoded frames.

    Keeps the most recently used frames resident so that multiple tiles drawn
    from the same frame do not each re-decode it. Least-recently-used entries
    are evicted once the capacity is exceeded.
    """

    def __init__(self, capacity: int = _FRAME_LRU_CAPACITY):
        """Initialize the cache.

        Args:
            capacity: Maximum number of frames to retain. Defaults to
                :data:`_FRAME_LRU_CAPACITY`.
        """
        self.capacity = capacity
        self._d: "OrderedDict" = OrderedDict()

    def get(self, key):
        """Return the cached value for ``key`` and mark it most-recently-used.

        Args:
            key: Cache key.

        Returns:
            The cached value, or ``None`` if ``key`` is not present.
        """
        if key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key, value) -> None:
        """Insert or update ``key`` and evict LRU entries past capacity.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)


class FrameGroupedTileSampler(Sampler[int]):
    """Yield flat sample indices grouped into contiguous per-frame blocks.

    Each block holds the flat sample indices that belong to a single frame.
    Blocks are kept contiguous (so all tiles of a frame are consumed together,
    enabling frame caching), while the *order of blocks* can be shuffled. With
    ``block_align`` enabled, each block is padded (by repeating its own indices)
    up to a multiple of ``batch_size`` so batches never straddle frames. Blocks
    are sharded across DDP replicas as whole units.
    """

    def __init__(
        self,
        frame_blocks: List[List[int]],
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        block_align: bool = True,
        shuffle_within_block: bool = False,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        """Initialize the sampler.

        Args:
            frame_blocks: List of blocks, each a list of flat sample indices for
                one frame.
            batch_size: Batch size used for block-alignment padding.
            shuffle: If ``True``, shuffle the order of the blocks each epoch.
            seed: Base seed for shuffling. Defaults to ``0``.
            block_align: If ``True``, pad each block to a multiple of
                ``batch_size`` so batches never span frames. Defaults to
                ``True``.
            shuffle_within_block: If ``True``, also shuffle the indices within
                each block. Defaults to ``False``.
            num_replicas: Number of DDP replicas the blocks are sharded across.
                Defaults to ``1``.
            rank: This process's replica rank. Defaults to ``0``.
        """
        self.frame_blocks = frame_blocks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.block_align = block_align
        self.shuffle_within_block = shuffle_within_block
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch to reseed shuffling.

        Args:
            epoch: The epoch index.
        """
        self.epoch = epoch

    def _epoch_blocks(self) -> List[List[int]]:
        """Return this rank's blocks for the current epoch (post shuffle/shard).

        Returns:
            The list of blocks assigned to this replica after the (optional)
            epoch shuffle and rank sharding.
        """
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + self.epoch)
        blocks = list(self.frame_blocks)
        if self.shuffle:
            blocks = [blocks[i] for i in torch.randperm(len(blocks), generator=g)]
        return blocks[self.rank :: self.num_replicas]

    def _padded_len(self, n: int) -> int:
        """Return the block-aligned length of a block of size ``n``.

        Args:
            n: Number of indices in the block.

        Returns:
            ``n`` rounded up to a multiple of ``batch_size`` when
            ``block_align`` is set, else ``n``.
        """
        if self.block_align and self.batch_size and n % self.batch_size:
            n += self.batch_size - (n % self.batch_size)
        return n

    def __iter__(self):
        """Iterate flat sample indices, block by block.

        Yields:
            Flat sample indices with blocks kept contiguous, padded for block
            alignment, and (optionally) shuffled in order and within block.
        """
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + self.epoch)
        blocks = list(self.frame_blocks)
        if self.shuffle:
            blocks = [blocks[i] for i in torch.randperm(len(blocks), generator=g)]
        blocks = blocks[self.rank :: self.num_replicas]
        for block in blocks:
            b = list(block)
            if self.shuffle_within_block:
                b = [b[i] for i in torch.randperm(len(b), generator=g)]
            if self.block_align and self.batch_size and len(b) % self.batch_size:
                pad = self.batch_size - (len(b) % self.batch_size)
                b += [b[i % len(b)] for i in range(pad)]
            yield from b

    def __len__(self) -> int:
        """Return the number of indices yielded for the current epoch.

        Returns:
            The total padded length summed over this rank's blocks, consistent
            with :meth:`__iter__`.
        """
        return sum(self._padded_len(len(block)) for block in self._epoch_blocks())
