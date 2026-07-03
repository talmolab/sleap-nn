"""Tests for the Phase-2 training primitives in ``sleap_nn.data.tiling``."""

import math
from collections import Counter

import numpy as np
import torch

from sleap_nn.data.tiling import (
    FrameGroupedTileSampler,
    _FRAME_LRU_CAPACITY,
    _FrameLRU,
    draw_tile_origin,
    extract_tile,
    frame_foreground_centers,
    tile_sample_seed,
)

# ---------------------------------------------------------------------------
# frame_foreground_centers
# ---------------------------------------------------------------------------


def test_frame_foreground_centers_drops_nan_rows():
    """Keypoints with any NaN coordinate are dropped; valid ones are kept."""
    instances = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0], [float("nan"), 5.0]],  # instance 0
                [[6.0, 7.0], [float("nan"), float("nan")], [8.0, 9.0]],  # instance 1
            ]
        ]
    )  # (1, I=2, N=3, 2)

    centers = frame_foreground_centers(instances, use_centroid=False)

    # 6 keypoints total, 2 contain NaN -> 4 valid.
    assert centers.shape == (4, 2)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [6.0, 7.0], [8.0, 9.0]])
    assert torch.allclose(centers, expected)


def test_frame_foreground_centers_empty_on_all_nan():
    """An all-NaN frame yields an empty (0, 2) result."""
    instances = torch.full((1, 2, 3, 2), float("nan"))
    centers = frame_foreground_centers(instances)
    assert centers.shape == (0, 2)
    assert centers.numel() == 0


def test_frame_foreground_centers_centroid_mode():
    """Centroid mode nan-means over nodes and drops all-NaN instances."""
    instances = torch.tensor(
        [
            [
                # instance 0: mean of (0,0) and (2,4) = (1, 2), one node missing.
                [[0.0, 0.0], [2.0, 4.0], [float("nan"), float("nan")]],
                # instance 1: fully missing -> dropped.
                [
                    [float("nan"), float("nan")],
                    [float("nan"), float("nan")],
                    [float("nan"), float("nan")],
                ],
            ]
        ]
    )

    centers = frame_foreground_centers(instances, use_centroid=True)
    assert centers.shape == (1, 2)
    assert torch.allclose(centers, torch.tensor([[1.0, 2.0]]))


# ---------------------------------------------------------------------------
# draw_tile_origin
# ---------------------------------------------------------------------------


def test_draw_tile_origin_force_fg_slot_rule():
    """The trailing ``tile_fg_fraction`` of slots are foreground slots."""
    centers = torch.tensor([[50.0, 60.0]])
    frame_hw = (200, 200)
    tile_size = 40
    samples_per_frame = 10
    tile_fg_fraction = 0.4  # last 4 slots (k >= round(10 * 0.6) = 6) are fg
    # Foreground draw with zero jitter lands deterministically on the center.
    fg_origin = (
        int(round(60.0 - tile_size / 2.0)),
        int(round(50.0 - tile_size / 2.0)),
    )

    for k in range(samples_per_frame):
        rng = np.random.default_rng(0)
        y0, x0 = draw_tile_origin(
            centers,
            frame_hw,
            tile_size,
            sample_k=k,
            samples_per_frame=samples_per_frame,
            tile_fg_fraction=tile_fg_fraction,
            center_jitter=0.0,
            rng=rng,
        )
        if k >= 6:
            # Foreground slot: with zero jitter the origin is the center-tile.
            assert (y0, x0) == fg_origin
        else:
            # Background slot: uniform-random, essentially never equals fg.
            assert (y0, x0) != fg_origin


def test_draw_tile_origin_fg_draw_within_jitter_bound():
    """Foreground draws stay within the jitter bound of the chosen center."""
    centers = torch.tensor([[80.0, 120.0]])
    frame_hw = (400, 400)
    tile_size = 60
    center_jitter = 0.5
    bound = center_jitter * (tile_size / 2.0) + 0.5  # +0.5 for rounding
    cx0 = 80.0 - tile_size / 2.0
    cy0 = 120.0 - tile_size / 2.0

    for seed in range(50):
        rng = np.random.default_rng(seed)
        y0, x0 = draw_tile_origin(
            centers,
            frame_hw,
            tile_size,
            sample_k=9,  # a forced-fg slot
            samples_per_frame=10,
            tile_fg_fraction=1.0,
            center_jitter=center_jitter,
            rng=rng,
        )
        assert abs(x0 - cx0) <= bound
        assert abs(y0 - cy0) <= bound


def test_draw_tile_origin_uniform_branch_in_range():
    """Background (uniform) draws land within the valid origin range."""
    centers = torch.tensor([[10.0, 10.0]])
    H, W = 300, 500
    tile_size = 64
    for seed in range(50):
        rng = np.random.default_rng(seed)
        y0, x0 = draw_tile_origin(
            centers,
            (H, W),
            tile_size,
            sample_k=0,  # a background slot
            samples_per_frame=10,
            tile_fg_fraction=0.3,
            center_jitter=0.0,
            rng=rng,
        )
        assert 0 <= x0 <= W - tile_size
        assert 0 <= y0 <= H - tile_size


def test_draw_tile_origin_pos_ratio_zero_forces_uniform():
    """``pos_ratio=0`` disables foreground sampling even on a forced-fg slot."""
    centers = torch.tensor([[50.0, 50.0]])
    fg_origin = (int(round(50.0 - 20.0)), int(round(50.0 - 20.0)))
    seen_non_fg = False
    for seed in range(50):
        rng = np.random.default_rng(seed)
        origin = draw_tile_origin(
            centers,
            (200, 200),
            tile_size=40,
            sample_k=9,
            samples_per_frame=10,
            tile_fg_fraction=1.0,  # would be fg without pos_ratio override
            center_jitter=0.0,
            rng=rng,
            pos_ratio=0.0,
        )
        if origin != fg_origin:
            seen_non_fg = True
    assert seen_non_fg  # uniform sampling produces off-center origins


def test_draw_tile_origin_empty_centers_forces_uniform():
    """No valid centers (M == 0) forces a uniform draw and never crashes."""
    centers = torch.empty((0, 2))
    H, W = 128, 128
    tile_size = 32
    for seed in range(20):
        rng = np.random.default_rng(seed)
        y0, x0 = draw_tile_origin(
            centers,
            (H, W),
            tile_size,
            sample_k=9,
            samples_per_frame=10,
            tile_fg_fraction=1.0,
            center_jitter=0.5,
            rng=rng,
        )
        assert 0 <= x0 <= W - tile_size
        assert 0 <= y0 <= H - tile_size


def test_draw_tile_origin_deterministic_with_seeded_rng():
    """Same seed -> same origin; different seeds -> (generally) different."""
    centers = torch.tensor([[70.0, 90.0], [120.0, 40.0]])
    kwargs = dict(
        frame_hw=(256, 256),
        tile_size=48,
        sample_k=9,
        samples_per_frame=10,
        tile_fg_fraction=1.0,
        center_jitter=0.3,
    )
    a = draw_tile_origin(centers, rng=np.random.default_rng(123), **kwargs)
    b = draw_tile_origin(centers, rng=np.random.default_rng(123), **kwargs)
    assert a == b


# ---------------------------------------------------------------------------
# extract_tile
# ---------------------------------------------------------------------------


def _manual_slice_tile(image, y0, x0, tile_size):
    """Reference slice+zero-pad crop for the fast path."""
    _, C, H, W = image.shape
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y0 + tile_size), min(W, x0 + tile_size)
    tile = image.new_zeros((1, C, tile_size, tile_size))
    if ye > ys and xe > xs:
        tile[:, :, ys - y0 : ye - y0, xs - x0 : xe - x0] = image[:, :, ys:ye, xs:xe]
    return tile


def test_extract_tile_fast_path_byte_identical_interior():
    """Interior tiles match a manual slice exactly (no resampling)."""
    torch.manual_seed(0)
    image = torch.rand(1, 3, 100, 120)
    instances = torch.tensor([[[[30.0, 40.0], [50.0, 60.0]]]])  # (1, 1, 2, 2)
    origin = (20, 10)
    tile, tile_inst = extract_tile(image, instances, origin, tile_size=32)

    assert tile.shape == (1, 3, 32, 32)
    assert torch.equal(tile, _manual_slice_tile(image, 20, 10, 32))
    # Instances shifted into tile-local coords.
    expected_inst = instances.clone()
    expected_inst[..., 0] -= 10
    expected_inst[..., 1] -= 20
    assert torch.equal(tile_inst, expected_inst)


def test_extract_tile_fast_path_negative_and_overflow_origin():
    """Negative and overflow origins zero-pad and stay byte-identical."""
    torch.manual_seed(1)
    image = torch.rand(1, 1, 50, 50)
    instances = torch.tensor([[[[5.0, 5.0]]]])
    tile_size = 40

    for origin in [(-10, -8), (30, 35), (-5, 30), (48, -3)]:
        tile, tile_inst = extract_tile(image, instances, origin, tile_size)
        assert tile.shape == (1, 1, tile_size, tile_size)
        assert torch.equal(
            tile, _manual_slice_tile(image, origin[0], origin[1], tile_size)
        )
        # Instance shift is applied regardless of clipping.
        assert torch.equal(tile_inst[..., 0], instances[..., 0] - origin[1])
        assert torch.equal(tile_inst[..., 1], instances[..., 1] - origin[0])


def test_extract_tile_fast_path_fully_out_of_bounds_is_zero():
    """A tile fully outside the frame is all zeros (empty-slice guard)."""
    image = torch.rand(1, 3, 40, 40)
    instances = torch.tensor([[[[5.0, 5.0]]]])
    tile, _ = extract_tile(image, instances, (100, 100), tile_size=32)
    assert torch.equal(tile, torch.zeros_like(tile))


def test_extract_tile_halo_path_rotation_shapes():
    """Halo path with a rotation aug returns correct shapes and no crash."""
    torch.manual_seed(0)
    image = torch.rand(1, 3, 128, 128)
    instances = torch.tensor([[[[60.0, 70.0], [64.0, 64.0]]]])  # (1, 1, 2, 2)
    tile_size = 48
    tile, tile_inst = extract_tile(
        image,
        instances,
        (40, 40),
        tile_size,
        apply_geometric=True,
        geometric_kwargs={
            "rotation_min": 30.0,
            "rotation_max": 30.0,
            "rotation_p": 1.0,
        },
        rng_seed=7,
    )
    assert tile.shape == (1, 3, tile_size, tile_size)
    assert tile_inst.shape == instances.shape
    # Halo side used internally is ceil(tile_size * sqrt(2)).
    assert int(math.ceil(tile_size * math.sqrt(2))) == 68


def test_extract_tile_halo_path_reproducible_with_seed():
    """The halo path is reproducible given the same seeded RNG state.

    ``extract_tile`` seeds ``torch`` via ``rng_seed``, but the underlying
    geometric augmentation samples its transform from the global ``numpy`` RNG,
    so the caller seeds ``numpy`` too for full reproducibility.
    """
    image = torch.rand(1, 1, 96, 96)
    instances = torch.tensor([[[[48.0, 48.0]]]])
    common = dict(
        tile_size=40,
        apply_geometric=True,
        geometric_kwargs={
            "rotation_min": -20.0,
            "rotation_max": 20.0,
            "rotation_p": 1.0,
        },
        rng_seed=99,
    )
    np.random.seed(1234)
    t1, i1 = extract_tile(image, instances, (28, 28), **common)
    np.random.seed(1234)
    t2, i2 = extract_tile(image, instances, (28, 28), **common)
    assert torch.equal(t1, t2)
    assert torch.equal(i1, i2)


# ---------------------------------------------------------------------------
# tile_sample_seed
# ---------------------------------------------------------------------------


def test_tile_sample_seed_deterministic():
    """Identical inputs give identical seeds."""
    a = tile_sample_seed(1234, 2, 3, 4, 5)
    b = tile_sample_seed(1234, 2, 3, 4, 5)
    assert a == b
    assert isinstance(a, int)
    assert 0 <= a < 2**32


def test_tile_sample_seed_varies_per_field():
    """Changing any single field changes the derived seed."""
    base = tile_sample_seed(10, 0, 0, 0, 0, salt=0)
    variants = [
        tile_sample_seed(11, 0, 0, 0, 0, salt=0),
        tile_sample_seed(10, 1, 0, 0, 0, salt=0),
        tile_sample_seed(10, 0, 1, 0, 0, salt=0),
        tile_sample_seed(10, 0, 0, 1, 0, salt=0),
        tile_sample_seed(10, 0, 0, 0, 1, salt=0),
        tile_sample_seed(10, 0, 0, 0, 0, salt=1),
    ]
    for v in variants:
        assert v != base
    # All distinct from one another too.
    assert len(set(variants)) == len(variants)


# ---------------------------------------------------------------------------
# _FrameLRU
# ---------------------------------------------------------------------------


def test_frame_lru_default_capacity():
    """The module constant drives the default capacity."""
    assert _FRAME_LRU_CAPACITY == 2
    lru = _FrameLRU()
    assert lru.capacity == 2


def test_frame_lru_evicts_lru_at_capacity():
    """Inserting past capacity evicts the least-recently-used entry."""
    lru = _FrameLRU(capacity=2)
    lru.put("a", 1)
    lru.put("b", 2)
    lru.put("c", 3)  # evicts "a"
    assert lru.get("a") is None
    assert lru.get("b") == 2
    assert lru.get("c") == 3


def test_frame_lru_move_to_end_on_get():
    """A ``get`` refreshes recency so that key survives the next eviction."""
    lru = _FrameLRU(capacity=2)
    lru.put("a", 1)
    lru.put("b", 2)
    assert lru.get("a") == 1  # "a" now most-recently-used
    lru.put("c", 3)  # should evict "b", not "a"
    assert lru.get("b") is None
    assert lru.get("a") == 1
    assert lru.get("c") == 3


def test_frame_lru_miss_returns_none():
    """Missing keys return None without mutating the cache."""
    lru = _FrameLRU(capacity=2)
    assert lru.get("nope") is None


# ---------------------------------------------------------------------------
# FrameGroupedTileSampler
# ---------------------------------------------------------------------------


def _blocks():
    return [[0, 1, 2], [3, 4], [5, 6, 7, 8], [9]]


def test_sampler_every_index_appears_with_block_align_padding():
    """Every original index appears; block_align pads to batch multiples."""
    blocks = _blocks()
    sampler = FrameGroupedTileSampler(
        blocks, batch_size=2, shuffle=False, block_align=True
    )
    out = list(sampler)
    counts = Counter(out)
    # Every original index appears at least once.
    for block in blocks:
        for idx in block:
            assert counts[idx] >= 1
    # Total length matches the padded sum and __len__.
    expected_len = sum(
        len(b) + ((-len(b)) % 2) for b in blocks
    )  # each block padded to multiple of 2
    assert len(out) == expected_len
    assert len(sampler) == expected_len


def test_sampler_blocks_stay_contiguous_under_shuffle():
    """Shuffling permutes block order but keeps each block contiguous."""
    blocks = _blocks()
    # Disable padding so we can recover exact block boundaries.
    sampler = FrameGroupedTileSampler(
        blocks, batch_size=2, shuffle=True, seed=3, block_align=False
    )
    out = list(sampler)
    # Split the flat output back into runs matching some permutation of blocks.
    block_by_first = {b[0]: b for b in blocks}
    i = 0
    recovered = []
    while i < len(out):
        first = out[i]
        assert first in block_by_first, "block boundaries not contiguous"
        b = block_by_first[first]
        assert out[i : i + len(b)] == b
        recovered.append(b)
        i += len(b)
    # Every block recovered exactly once (order may differ).
    assert sorted(recovered) == sorted(blocks)


def test_sampler_len_matches_iterated_count():
    """__len__ equals the number of yielded indices for several configs."""
    blocks = _blocks()
    for shuffle in (False, True):
        for block_align in (False, True):
            sampler = FrameGroupedTileSampler(
                blocks,
                batch_size=3,
                shuffle=shuffle,
                seed=7,
                block_align=block_align,
            )
            assert len(sampler) == len(list(sampler))


def test_sampler_ddp_disjoint_and_covers_all_blocks():
    """DDP replicas get disjoint whole blocks whose union covers everything."""
    blocks = _blocks()
    r0 = FrameGroupedTileSampler(
        blocks,
        batch_size=2,
        shuffle=True,
        seed=5,
        block_align=False,
        num_replicas=2,
        rank=0,
    )
    r1 = FrameGroupedTileSampler(
        blocks,
        batch_size=2,
        shuffle=True,
        seed=5,
        block_align=False,
        num_replicas=2,
        rank=1,
    )
    out0 = list(r0)
    out1 = list(r1)

    # Disjoint across ranks.
    assert set(out0).isdisjoint(set(out1))
    # Union covers every index exactly once (no padding, whole-block sharding).
    assert sorted(out0 + out1) == sorted(idx for b in blocks for idx in b)
    # __len__ consistent per rank.
    assert len(r0) == len(out0)
    assert len(r1) == len(out1)


def test_sampler_set_epoch_changes_order():
    """set_epoch reseeds the shuffle so the emitted order changes."""
    blocks = [[i] for i in range(20)]  # singleton blocks -> pure order shuffle
    sampler = FrameGroupedTileSampler(
        blocks, batch_size=1, shuffle=True, seed=0, block_align=False
    )
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second
    # Same multiset of indices, just reordered.
    assert sorted(first) == sorted(second)
