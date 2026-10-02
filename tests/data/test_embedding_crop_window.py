"""Embedding crops read only the window they sample (emb-review F6).

`EmbeddingDataset.__getitem__` used to decode each mask to a full-frame array, build a
full-frame float tensor, size-match it and recompute the center of mass with
`np.nonzero`, and to hand the whole frame to Skia (an RGBA copy of it) for every crop:
134 ms per mask sample at 4K. The crop center is now computed once at index time, from
the run-length encoding, and a size-matched frame is cropped from its window only.

The crops must not change. `_reference_crops` below is the pre-F6 crop path, copied
verbatim, and every crop is compared to it byte for byte, over the geometry that
matters: crop centering, masks at output stride / with offsets / off the frame edge /
larger than the frame, pose centroids off the frame, grey and RGB frames, and size
matching (`max_hw`), which still takes the full-frame path.
"""

import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf
from PIL import Image

import sleap_nn.data.custom_datasets as custom_datasets
import sleap_nn.inference.segmentation_convert as segmentation_convert
from sleap_nn.data.custom_datasets import (
    EmbeddingDataset,
    _compute_mask_centroids,
    _mask_bbox_midpoint,
    get_train_val_datasets,
)
from sleap_nn.data.instance_cropping import iter_mask_extents, make_centered_bboxes
from sleap_nn.data.normalization import convert_to_grayscale, convert_to_rgb
from sleap_nn.data.resizing import apply_sizematcher
from sleap_nn.data.skia_augmentation import crop_and_resize_skia
from sleap_nn.inference.segmentation_convert import decode_mask_to_image_res

GREY_HW = (157, 211)
RGB_HW = (131, 187)
_SKEL = sio.Skeleton(nodes=["a", "b", "c"], name="s")
_HEAD = OmegaConf.create({"embedding_dim": 8, "output_stride": 16})


# ── the pre-F6 crop path, verbatim (full frame) ─────────────────────────────────


def _reference_crop_mask(ds, image, meta):
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

    if ds.ensure_rgb:
        image = convert_to_rgb(image)
    elif ds.ensure_grayscale:
        image = convert_to_grayscale(image)

    image, _ = apply_sizematcher(image, max_height=ds.max_hw[0], max_width=ds.max_hw[1])
    mask_t, _ = apply_sizematcher(
        mask_t, max_height=ds.max_hw[0], max_width=ds.max_hw[1]
    )

    mask_bool = mask_t[0, 0].numpy() > 0.5
    if ds.crop_centering == "bbox":
        cx, cy = _mask_bbox_midpoint(mask_bool)
    else:  # "auto" / "mask_com"
        cx, cy = _compute_mask_centroids([mask_bool])[0]
    bbox = make_centered_bboxes(
        torch.tensor([cx, cy], dtype=torch.float32),
        ds.crop_size,
        ds.crop_size,
    ).unsqueeze(0)
    instance_image = crop_and_resize_skia(
        image, boxes=bbox, size=(ds.crop_size, ds.crop_size)
    )
    instance_mask = crop_and_resize_skia(
        mask_t, boxes=bbox, size=(ds.crop_size, ds.crop_size)
    )
    return instance_image, instance_mask


def _reference_crop_pose(ds, image, meta):
    if ds.ensure_rgb:
        image = convert_to_rgb(image)
    elif ds.ensure_grayscale:
        image = convert_to_grayscale(image)

    image, ratio = apply_sizematcher(
        image, max_height=ds.max_hw[0], max_width=ds.max_hw[1]
    )
    cx = float(meta["centroid"][0]) * ratio
    cy = float(meta["centroid"][1]) * ratio
    bbox = make_centered_bboxes(
        torch.tensor([cx, cy], dtype=torch.float32),
        ds.crop_size,
        ds.crop_size,
    ).unsqueeze(0)
    instance_image = crop_and_resize_skia(
        image, boxes=bbox, size=(ds.crop_size, ds.crop_size)
    )
    instance_mask = torch.ones((1, 1, ds.crop_size, ds.crop_size), dtype=torch.float32)
    return instance_image, instance_mask


def _reference_crops(ds, labels, index):
    meta = ds.mask_idx_list[index]
    img = labels[meta["labels_idx"]][meta["lf_idx"]].image.copy()
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    image = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)  # (1, C, H, W)
    image = torch.from_numpy(image.copy())
    if "centroid" in meta:
        return _reference_crop_pose(ds, image, meta)
    return _reference_crop_mask(ds, image, meta)


# ── fixtures ────────────────────────────────────────────────────────────────────


def _frames(tmp_path, name, hw, n, rgb, seed):
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        shape = (*hw, 3) if rgb else hw
        # Smooth-ish content so bilinear sampling lands between distinct values.
        base = rng.integers(0, 256, shape).astype(np.float64)
        img = (0.5 * base + 0.5 * np.roll(base, 1, axis=1)).astype(np.uint8)
        path = tmp_path / f"{name}_{i}.png"
        Image.fromarray(img, mode="RGB" if rgb else "L").save(path)
        paths.append(path.as_posix())
    return sio.Video.from_filename(paths)


def _mask(arr, cls=sio.UserSegmentationMask, **kwargs):
    return cls.from_numpy(np.asarray(arr, bool), **kwargs)


def _grey_masks(frame_idx):
    """Masks covering the geometry the crop has to reproduce."""
    h, w = GREY_HW
    yy, xx = np.mgrid[:h, :w]
    shift = 7 * frame_idx
    masks = []
    # A ring: concave, its center of mass off the mask, COM != bbox midpoint.
    r = np.hypot(yy - 80, xx - 90 - shift)
    masks.append(_mask((r > 12) & (r < 30) & ~((xx > 90 + shift) & (yy > 80))))
    # Touching the right and bottom edges: the crop runs off the frame.
    edge = np.zeros((h, w), bool)
    edge[140 - frame_idx :, 190:] = True
    masks.append(_mask(edge))
    # Full-width rows: runs that wrap from one row to the next.
    band = np.zeros((h, w), bool)
    band[60:63, :] = True
    band[63, : 20 + shift] = True
    masks.append(_mask(band))
    # A top-down crop mask at output stride 2, with an offset that rounds half to
    # even and puts it partly off the left and bottom edges (the frame clips it).
    blob = np.zeros((40, 40), bool)
    blob[5:38, 3:30] = True
    blob[15:25, 10:20] = False
    masks.append(
        _mask(
            blob,
            cls=sio.PredictedSegmentationMask,
            score=0.9,
            scale=(0.5, 0.5),
            offset=(-12.5, 100.6 - shift),
        )
    )
    # Anisotropic stride and a small positive offset.
    aniso = np.zeros((20, 30), bool)
    aniso[2:15, 4:26] = True
    aniso[8:15, 10:18] = False
    masks.append(
        _mask(
            aniso,
            cls=sio.PredictedSegmentationMask,
            score=0.8,
            scale=(0.25, 0.5),
            offset=(5.0 + shift, 3.4),
        )
    )
    # Full resolution, half-pixel offset (rounds to even).
    small = np.zeros((30, 25), bool)
    small[4:27, 2:20] = True
    masks.append(_mask(small, offset=(40.5, 10.2 + shift)))
    # A grid larger than the frame with foreground past its bottom edge.
    big = np.zeros((h + 20, w + 30), bool)
    big[140:170, 30 + shift : 60 + shift] = True
    masks.append(_mask(big))
    # Foreground only outside the frame: the frame-center fallback.
    outside = np.zeros((h + 40, w + 40), bool)
    outside[h + 5 : h + 20, 10:40] = True
    masks.append(_mask(outside))
    return masks


def _rgb_masks(frame_idx):
    h, w = RGB_HW
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - 60, xx - 70 - 5 * frame_idx)
    edge = np.zeros((h, w), bool)
    edge[:20, :15] = True  # top-left corner
    return [_mask((r > 8) & (r < 25)), _mask(edge)]


def _poses(hw, frame_idx):
    h, w = hw
    centers = [
        (50.3 + frame_idx, 70.8),
        (2.0, 3.0),
        (-5.0, 60.0),
        (w - 1.5, h - 0.2),
        # Crops entirely off the frame: an empty window (on an RGB frame, whose
        # decoded array has a negative channel stride, this once crashed).
        (-150.0, 60.0),
        (w + 120.5, h + 99.0),
    ]
    return [
        sio.Instance.from_numpy(
            np.array([[cx - 6, cy], [cx + 6, cy + 1], [cx, cy - 5]], float),
            skeleton=_SKEL,
        )
        for cx, cy in centers
    ]


def _labels(video, hw, n, masks_fn):
    frames = [
        sio.LabeledFrame(
            video=video,
            frame_idx=fi,
            instances=_poses(hw, fi),
            masks=masks_fn(fi),
        )
        for fi in range(n)
    ]
    return sio.Labels(videos=[video], labeled_frames=frames, skeletons=[_SKEL])


@pytest.fixture
def grey_labels(tmp_path):
    video = _frames(tmp_path, "grey", GREY_HW, 3, rgb=False, seed=0)
    return _labels(video, GREY_HW, 3, _grey_masks)


@pytest.fixture
def rgb_labels(tmp_path):
    video = _frames(tmp_path, "rgb", RGB_HW, 2, rgb=True, seed=1)
    return _labels(video, RGB_HW, 2, _rgb_masks)


def _dataset(labels, mode, centering="auto", crop_size=48, rgb=False, **kwargs):
    return EmbeddingDataset(
        labels=labels,
        crop_size=crop_size,
        class_names=[],
        embedding_head_config=_HEAD,
        max_stride=16,
        id_scope="aug_view",
        crop_centering=centering,
        user_instances_only=False,
        ensure_rgb=rgb,
        ensure_grayscale=not rgb,
        detection_mode=mode,
        **{"cache_img": None, **kwargs},
    )


def _new_crops(ds, index):
    """The raw crops `__getitem__` packs (the real entry point)."""
    captured = {}
    pack = ds._pack_sample

    def capture(image, mask, meta, i):
        captured["crops"] = (image, mask)
        return pack(image, mask, meta, i)

    ds._pack_sample = capture
    try:
        sample = ds[index]
    finally:
        ds._pack_sample = pack
    return captured["crops"], sample


def _assert_same_crops(ds, labels):
    assert len(ds) > 0
    for index in range(len(ds)):
        (image, mask), sample = _new_crops(ds, index)
        ref_image, ref_mask = _reference_crops(ds, labels, index)
        assert image.dtype == ref_image.dtype and mask.dtype == ref_mask.dtype
        assert torch.equal(image, ref_image), f"image crop {index} differs"
        assert torch.equal(mask, ref_mask), f"mask crop {index} differs"
        assert torch.equal(sample["instance_image"], ref_image.to(torch.float32))
        assert torch.equal(sample["instance_mask"], (ref_mask > 0.5).to(torch.float32))


# ── crops are byte-identical (a guard: passes before and after F6) ──────────────

# Size matching: none; the grey frame's size (the RGB frames are resized, so they take
# the full-frame path); larger than every frame (every frame is resized).
MAX_HW = {"none": (None, None), "grey": GREY_HW, "larger": (170, 230)}


@pytest.mark.parametrize("max_hw", sorted(MAX_HW))
@pytest.mark.parametrize("rgb", [False, True], ids=["grey_out", "rgb_out"])
@pytest.mark.parametrize("crop_size", [48, 33])
@pytest.mark.parametrize("centering", ["auto", "mask_com", "bbox"])
def test_mask_crops_are_unchanged(
    grey_labels, rgb_labels, max_hw, rgb, crop_size, centering
):
    labels = [grey_labels, rgb_labels]
    ds = _dataset(
        labels,
        "mask",
        centering=centering,
        crop_size=crop_size,
        rgb=rgb,
        max_hw=MAX_HW[max_hw],
    )
    assert ds.detection_mode == "mask"
    assert len(ds) == 3 * 8 + 2 * 2
    _assert_same_crops(ds, labels)


@pytest.mark.parametrize("max_hw", sorted(MAX_HW))
@pytest.mark.parametrize("rgb", [False, True], ids=["grey_out", "rgb_out"])
@pytest.mark.parametrize("crop_size", [48, 33])
def test_pose_crops_are_unchanged(grey_labels, rgb_labels, max_hw, rgb, crop_size):
    labels = [grey_labels, rgb_labels]
    ds = _dataset(labels, "pose", crop_size=crop_size, rgb=rgb, max_hw=MAX_HW[max_hw])
    assert ds.detection_mode == "pose"
    assert len(ds) == (3 + 2) * 6
    _assert_same_crops(ds, labels)


def test_memory_cache_crops_are_unchanged_and_cache_untouched(grey_labels):
    """The windowed path reads the cached frame in place instead of copying it."""
    ds = _dataset([grey_labels], "mask", cache_img="memory", max_hw=GREY_HW)
    before = {key: frame.copy() for key, frame in ds.cache.items()}
    _assert_same_crops(ds, [grey_labels])
    for key, frame in ds.cache.items():
        assert np.array_equal(frame, before[key])


def test_crop_center_is_computed_at_index_time(grey_labels):
    """`mask_center` is the center the full-frame path computes on the decoded mask."""
    for centering in ("auto", "bbox"):
        ds = _dataset([grey_labels], "mask", centering=centering)
        for meta in ds.mask_idx_list:
            arr = decode_mask_to_image_res(meta["mask_obj"])
            if not arr.any():
                assert meta["mask_center"] is None
                continue
            expected = (
                _mask_bbox_midpoint(arr)
                if centering == "bbox"
                else _compute_mask_centroids([arr])[0]
            )
            assert meta["mask_center"] == expected
            assert meta["mask_image_hw"] == (
                int(np.flatnonzero(arr.any(axis=1))[-1]) + 1,
                int(np.flatnonzero(arr.any(axis=0))[-1]) + 1,
            )


def _reference_mask_extents(labels, crop_centering):
    """The pre-F6 `iter_mask_extents` loop body, verbatim (no size matching)."""
    for lf in labels:
        for mask in lf.masks:
            arr = decode_mask_to_image_res(mask)
            xs = np.flatnonzero(arr.any(axis=0))
            ys = np.flatnonzero(arr.any(axis=1))
            if xs.size == 0:
                continue
            x0, x1, y0, y1 = int(xs[0]), int(xs[-1]), int(ys[0]), int(ys[-1])
            if crop_centering == "bbox":
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            else:
                box = arr[y0 : y1 + 1, x0 : x1 + 1]
                col_sums = np.count_nonzero(box, axis=0)
                row_sums = np.count_nonzero(box, axis=1)
                n = float(col_sums.sum())
                cx = x0 + float(col_sums @ np.arange(col_sums.size)) / n
                cy = y0 + float(row_sums @ np.arange(row_sums.size)) / n
            reach = max(cx - x0, x1 - cx, cy - y0, y1 - cy)
            yield float(2.0 * reach + 1.0), float(max(x1 - x0, y1 - y0) + 1.0)


@pytest.mark.parametrize("centering", ["auto", "bbox"])
def test_mask_extents_are_unchanged(grey_labels, centering):
    """`iter_mask_extents` (auto crop sizing) reads the same geometry from the
    run-length encoding, to the last bit, so `find_mask_crop_size` cannot move."""
    got = list(iter_mask_extents(grey_labels, crop_centering=centering))
    expected = list(_reference_mask_extents(grey_labels, centering))
    assert len(got) == 3 * 8
    assert got == expected


def test_out_of_window_scratch_contents_are_never_read(
    grey_labels, rgb_labels, monkeypatch
):
    """The windowed crop writes its window into a reused frame-sized RGBA buffer and
    leaves the rest stale. Filling that buffer with random garbage (alpha too) before
    every crop changes no byte: Skia never samples outside the window."""
    import sleap_nn.data.skia_augmentation as skia_augmentation

    rng = np.random.default_rng(0)
    scratch = skia_augmentation._frame_rgba_scratch

    def poisoned(height, width):
        buffer = scratch(height, width)
        buffer[...] = rng.integers(0, 256, buffer.shape, dtype=np.uint8)
        return buffer

    monkeypatch.setattr(skia_augmentation, "_frame_rgba_scratch", poisoned)
    labels = [grey_labels, rgb_labels]
    for mode in ("mask", "pose"):
        for rgb in (False, True):
            _assert_same_crops(_dataset(labels, mode, crop_size=33, rgb=rgb), labels)


def test_crops_are_unchanged_across_threads(grey_labels, rgb_labels):
    """Each thread crops into its own scratch buffer."""
    from concurrent.futures import ThreadPoolExecutor

    labels = [grey_labels, rgb_labels]
    ds = _dataset(labels, "mask", crop_size=48)
    expected = [_reference_crops(ds, labels, i) for i in range(len(ds))]
    order = list(range(len(ds))) * 4
    np.random.default_rng(1).shuffle(order)
    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(ds.__getitem__, order))
    for index, sample in zip(order, got):
        ref_image, ref_mask = expected[index]
        assert torch.equal(sample["instance_image"], ref_image.to(torch.float32))
        assert torch.equal(sample["instance_mask"], (ref_mask > 0.5).float())


# ── the perf fix: no full-frame work per sample (fails before F6) ───────────────


@pytest.fixture
def full_frame_work(monkeypatch):
    """Count full-frame mask decodes and full-frame crops."""
    counts = {"mask_decode": 0, "full_frame_crop": 0}
    data = sio.SegmentationMask.data

    def counting_data(mask):
        counts["mask_decode"] += 1
        return data.fget(mask)

    def counting_crop(*args, **kwargs):
        counts["full_frame_crop"] += 1
        return crop_and_resize_skia(*args, **kwargs)

    monkeypatch.setattr(sio.SegmentationMask, "data", property(counting_data))
    monkeypatch.setattr(custom_datasets, "crop_and_resize", counting_crop)
    decode = segmentation_convert.decode_mask_to_image_res

    def counting_decode(mask):
        counts["mask_decode"] += 1
        return decode(mask)

    monkeypatch.setattr(
        segmentation_convert, "decode_mask_to_image_res", counting_decode
    )
    return counts


def _mask_training_config(video_path, tmp_path):
    """A mask-mode embedding training config on a real 384 x 384 video."""
    from tests.training.test_embedding_train_e2e import _config

    video = sio.load_video(video_path.as_posix())
    identities = [sio.Identity(name=f"animal_{i}") for i in range(2)]
    frames = []
    for fi in range(6):
        masks = []
        for identity, (x0, y0) in zip(identities, [(60 + fi, 100), (220, 180 + fi)]):
            arr = np.zeros((384, 384), bool)
            arr[y0 : y0 + 50, x0 : x0 + 40] = True
            arr[y0 + 10 : y0 + 30, x0 + 10 : x0 + 20] = False
            mask = sio.UserSegmentationMask.from_numpy(arr)
            mask.identity = identity
            masks.append(mask)
        frames.append(
            sio.LabeledFrame(video=video, frame_idx=fi, instances=[], masks=masks)
        )
    labels = sio.Labels(labeled_frames=frames, videos=[video], skeletons=[])
    path = tmp_path / "masks.slp"
    labels.save(path.as_posix())
    return _config(
        path,
        tmp_path,
        "f6",
        **{"data_config.identity.track_names_are_global": False},
    )


def test_training_samples_do_no_full_frame_work(
    centered_instance_video, tmp_path, full_frame_work
):
    """Every mask sample of the datasets training builds decoded its mask to a full
    frame and cropped the full frame (twice); now none does."""
    from sleap_nn.training.model_trainer import ModelTrainer

    trainer = ModelTrainer.get_model_trainer_from_config(
        _mask_training_config(centered_instance_video, tmp_path)
    )
    train_ds, val_ds = get_train_val_datasets(
        train_labels=trainer.train_labels,
        val_labels=trainer.val_labels,
        config=trainer.config,
    )
    assert train_ds.detection_mode == "mask" and len(train_ds) == 12

    full_frame_work.update(mask_decode=0, full_frame_crop=0)
    for index in range(len(train_ds)):
        sample = train_ds[index]
        assert sample["instance_mask"].sum() > 0
    for index in range(len(val_ds)):
        val_ds[index]
    assert full_frame_work == {"mask_decode": 0, "full_frame_crop": 0}


def test_resized_frames_still_take_the_full_frame_path(grey_labels, full_frame_work):
    """A frame the size matcher resizes is cropped whole, as before (the exception)."""
    ds = _dataset([grey_labels], "mask", max_hw=MAX_HW["larger"])
    full_frame_work.update(mask_decode=0, full_frame_crop=0)
    ds[0]
    assert full_frame_work["full_frame_crop"] == 2
    assert full_frame_work["mask_decode"] >= 1


def test_non_finite_pose_centroids_are_skipped(tmp_path):
    """A detection whose centroid is inf (or overflows float32) is skipped like a NaN
    one; computing its crop window used to raise ``OverflowError`` on every sample."""
    video = _frames(tmp_path, "inf", (40, 48), 1, rgb=False, seed=0)
    skeleton = sio.Skeleton(nodes=["a"], name="s")
    xs = (np.inf, 1e39, -np.inf, 20.0)
    instances = [
        sio.Instance.from_numpy(np.array([[x, 5.0]]), skeleton=skeleton) for x in xs
    ]
    labels = sio.Labels(
        videos=[video],
        labeled_frames=[
            sio.LabeledFrame(video=video, frame_idx=0, instances=instances)
        ],
        skeletons=[skeleton],
    )

    ds = _dataset([labels], "pose", crop_size=32, include_untracked=True)

    assert len(ds) == 1
    assert ds[0]["instance_image"].shape[-2:] == (32, 32)
