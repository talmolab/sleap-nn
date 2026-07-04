"""Byte-identical golden regressions for the tiling data path (spec §2.13).

Two guarantees are pinned here:

1. ``extract_tile(apply_geometric=False)`` is a pure tensor slice with
   constant-zero padding: it is byte-identical to a hand-computed
   ``frame[..., y0:y0+t, x0:x0+t]`` slice (including the zero-pad region), and
   returns tile-local instances equal to ``frame_instances - [x0, y0]``.
2. The ``not tiling_enabled`` branch of ``_apply_common_preprocessing`` on a
   plain ``SingleInstanceDataset`` sample is unchanged (equals the explicit
   sizematcher -> resizer -> pad pipeline).
"""

import copy

import numpy as np
import sleap_io as sio
import torch

from sleap_nn.data.custom_datasets import SingleInstanceDataset
from sleap_nn.data.normalization import convert_to_grayscale, convert_to_rgb
from sleap_nn.data.providers import process_lf
from sleap_nn.data.resizing import (
    apply_pad_to_stride,
    apply_resizer,
    apply_sizematcher,
)
from sleap_nn.data.tiling import extract_tile


def _manual_slice(image, y0, x0, t):
    """Hand-computed zero-padded tile slice (the reference for extract_tile)."""
    _, C, H, W = image.shape
    tile = image.new_zeros((1, C, t, t))
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y0 + t), min(W, x0 + t)
    if ye > ys and xe > xs:
        tile[:, :, ys - y0 : ye - y0, xs - x0 : xe - x0] = image[:, :, ys:ye, xs:xe]
    return tile


def test_extract_tile_interior_is_exact_slice():
    """A fully-interior tile equals the raw frame slice, byte-for-byte."""
    torch.manual_seed(0)
    image = torch.randint(0, 255, (1, 3, 64, 80), dtype=torch.uint8)
    instances = torch.tensor([[[[10.0, 12.0], [30.0, 40.0]]]])  # (1, 1, 2, 2)
    t, y0, x0 = 32, 8, 16

    tile, tile_inst = extract_tile(image, instances, (y0, x0), t)

    assert tile.shape == (1, 3, t, t)
    # Interior tile: byte-identical to the direct slice.
    np.testing.assert_array_equal(
        tile.numpy(), image[:, :, y0 : y0 + t, x0 : x0 + t].numpy()
    )
    # Tile-local instances are the frame instances minus the top-left origin.
    expected = instances.clone()
    expected[..., 0] -= x0
    expected[..., 1] -= y0
    np.testing.assert_array_equal(tile_inst.numpy(), expected.numpy())


def test_extract_tile_partial_zero_pad_byte_identical():
    """A tile hanging off the bottom/right (and negative origin) zero-pads exactly."""
    torch.manual_seed(1)
    image = torch.randint(0, 255, (1, 1, 40, 40), dtype=torch.uint8)
    instances = torch.tensor([[[[5.0, 6.0], [35.0, 38.0]]]])

    for y0, x0 in [(24, 24), (-8, 10), (10, -8), (-8, -8)]:
        t = 32
        tile, tile_inst = extract_tile(image, instances, (y0, x0), t)
        np.testing.assert_array_equal(
            tile.numpy(), _manual_slice(image, y0, x0, t).numpy()
        )

        # Padded region is exactly zero.
        _, _, H, W = image.shape
        ys, xs = max(0, y0), max(0, x0)
        ye, xe = min(H, y0 + t), min(W, x0 + t)
        mask = torch.ones((t, t), dtype=torch.bool)
        mask[ys - y0 : ye - y0, xs - x0 : xe - x0] = False
        assert torch.all(tile[0, :, mask] == 0)

        expected = instances.clone()
        expected[..., 0] -= x0
        expected[..., 1] -= y0
        np.testing.assert_array_equal(tile_inst.numpy(), expected.numpy())


def test_non_tiled_common_preprocessing_unchanged(minimal_instance):
    """The `not tiling_enabled` preprocessing path equals the explicit pipeline."""
    labels = sio.load_slp(minimal_instance)
    for lf in labels:
        lf.instances = [lf.instances[0]]

    confmap = {"part_names": ["A", "B"], "sigma": 1.5, "output_stride": 2}
    ds = SingleInstanceDataset(
        labels=[labels],
        confmap_head_config=confmap,
        max_stride=8,
        scale=0.5,
        apply_aug=False,
        max_hw=(384, 384),
    )
    # Tiling must be inert on a plain SingleInstanceDataset.
    assert ds.tiling_enabled is False

    lf = labels[0]
    img = lf.image
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    raw = process_lf(
        instances_list=lf.instances,
        img=img,
        frame_idx=0,
        video_idx=0,
        max_instances=1,
        user_instances_only=True,
    )

    got = ds._apply_common_preprocessing(copy.deepcopy(raw))

    # Reference: explicit convert -> sizematcher -> resizer -> pad.
    ref = copy.deepcopy(raw)
    ref_img = ref["image"]
    if ds.ensure_rgb:
        ref_img = convert_to_rgb(ref_img)
    elif ds.ensure_grayscale:
        ref_img = convert_to_grayscale(ref_img)
    ref_img, eff_scale = apply_sizematcher(ref_img, max_height=384, max_width=384)
    ref_inst = ref["instances"] * eff_scale
    ref_img, ref_inst = apply_resizer(ref_img, ref_inst, scale=0.5)
    ref_img = apply_pad_to_stride(ref_img, max_stride=8)

    np.testing.assert_array_equal(got["image"].numpy(), ref_img.numpy())
    np.testing.assert_array_equal(got["instances"].numpy(), ref_inst.numpy())
    assert float(got["eff_scale"]) == float(eff_scale)
    # The tiling-only key is absent on the non-tiling path.
    assert "tile_origin" not in got
