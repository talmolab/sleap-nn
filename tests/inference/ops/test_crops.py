"""Regression tests for ``crop_bboxes`` top-left rounding (parity with legacy).

PR #530 audit found that the refactored ``crop_bboxes`` floored the raw bbox
top-left with ``.long()`` (truncate-toward-zero), whereas legacy mapped the
top-left via ``trunc(top_left + dim // 2) - dim // 2`` over a half-padded
unfold grid. The two agree for integer-aligned bboxes (the common path) but
diverge by one pixel when the bbox top-left is NEGATIVE and fractional — an
instance overhanging the top/left image edge — which silently shifts the crop
fed to the centered-instance model in top-down inference.

These tests pin the legacy semantics with an independent NumPy reference.
"""

from __future__ import annotations

import numpy as np
import torch

from sleap_nn.inference.ops.crops import crop_bboxes, make_centered_bboxes


def _ramp_image(h: int, w: int, c: int = 1) -> torch.Tensor:
    """A (1, c, h, w) image whose pixel value encodes its (y, x) position."""
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    val = yy * 1000.0 + xx  # unique per (y, x); decodes back to position
    return val.view(1, 1, h, w).expand(1, c, h, w).contiguous()


def _legacy_reference_crop(
    image: torch.Tensor, bbox: torch.Tensor, ch: int, cw: int
) -> np.ndarray:
    """Independent reference: legacy ``trunc(tl + half) - half`` extraction.

    Returns a (c, ch, cw) crop with zeros for out-of-image samples.
    """
    img = image[0].numpy()  # (c, H, W)
    c, H, W = img.shape
    half_w, half_h = cw // 2, ch // 2
    tlx = int(np.trunc(float(bbox[0, 0]) + half_w)) - half_w
    tly = int(np.trunc(float(bbox[0, 1]) + half_h)) - half_h
    out = np.zeros((c, ch, cw), dtype=img.dtype)
    for j in range(ch):
        for i in range(cw):
            yy, xx = tly + j, tlx + i
            if 0 <= yy < H and 0 <= xx < W:
                out[:, j, i] = img[:, yy, xx]
    return out


def test_crop_bboxes_matches_legacy_reference_including_border():
    """Crops around interior AND top/left-border sub-pixel centroids match legacy."""
    h = w = 40
    ch = cw = 6
    image = _ramp_image(h, w)
    # Centroids: interior, top/left border (negative fractional top-left), and
    # an integer-aligned one (the common/integral-refinement path).
    centroids = torch.tensor(
        [[20.3, 18.7], [2.4, 2.4], [0.6, 3.0], [10.0, 10.0], [1.4, 1.4]],
        dtype=torch.float32,
    )
    bboxes = make_centered_bboxes(centroids, ch, cw)
    sample_inds = torch.zeros(centroids.shape[0], dtype=torch.long)
    crops = crop_bboxes(image, bboxes, sample_inds).numpy()  # (n, c, ch, cw)

    for k in range(centroids.shape[0]):
        ref = _legacy_reference_crop(image, bboxes[k], ch, cw)
        np.testing.assert_array_equal(
            crops[k],
            ref,
            err_msg=f"crop {k} (centroid {centroids[k].tolist()}) "
            "diverged from legacy trunc(tl+half)-half extraction",
        )


def test_crop_bboxes_integer_aligned_unchanged():
    """Integer peaks (the integral-refinement call site) crop bit-exactly."""
    h = w = 32
    ch = cw = 5
    image = _ramp_image(h, w)
    centroids = torch.tensor([[10.0, 12.0], [4.0, 4.0]], dtype=torch.float32)
    bboxes = make_centered_bboxes(centroids, ch, cw)
    sample_inds = torch.zeros(centroids.shape[0], dtype=torch.long)
    crops = crop_bboxes(image, bboxes, sample_inds).numpy()
    for k in range(centroids.shape[0]):
        ref = _legacy_reference_crop(image, bboxes[k], ch, cw)
        np.testing.assert_array_equal(crops[k], ref)


def test_crop_bboxes_border_case_differs_from_naive_trunc():
    """Guard: the fix actually changes behavior for the negative-fractional case.

    (If someone reverts to plain ``.long()`` this test fails, catching the
    regression rather than silently passing.)
    """
    h = w = 20
    ch = cw = 6
    image = _ramp_image(h, w)
    centroid = torch.tensor([[2.4, 2.4]], dtype=torch.float32)
    bbox = make_centered_bboxes(centroid, ch, cw)
    # bbox top-left should be negative + fractional for this to be meaningful.
    assert float(bbox[0, 0, 0]) < 0 and float(bbox[0, 0, 0]) % 1 != 0
    crop = crop_bboxes(image, bbox, torch.zeros(1, dtype=torch.long))[0].numpy()
    legacy = _legacy_reference_crop(image, bbox[0], ch, cw)
    naive_tl_x = int(np.trunc(float(bbox[0, 0, 0])))  # plain .long()
    legacy_tl_x = int(np.trunc(float(bbox[0, 0, 0]) + cw // 2)) - cw // 2
    assert naive_tl_x != legacy_tl_x  # the two strategies genuinely differ here
    np.testing.assert_array_equal(crop, legacy)


def test_crop_bboxes_far_out_of_bounds_is_zero_filled():
    """Far/extreme out-of-bounds crops are zero-filled (pins current behavior, #584).

    The near-edge negative-fractional case is covered above; this pins the
    far-OOB clamp so a future change to the shared crop op is caught.
    """
    h = w = 20
    ch = cw = 6
    image = _ramp_image(h, w)
    # Centroids far outside the image on both sides.
    centroids = torch.tensor(
        [[1000.0, 1000.0], [-1000.0, -1000.0]], dtype=torch.float32
    )
    bboxes = make_centered_bboxes(centroids, ch, cw)
    sample_inds = torch.zeros(centroids.shape[0], dtype=torch.long)
    crops = crop_bboxes(image, bboxes, sample_inds).numpy()
    assert crops.shape[0] == 2
    # Entirely outside -> all-zero (no real pixels sampled).
    np.testing.assert_array_equal(crops, np.zeros_like(crops))


def test_crop_bboxes_size_survives_float32_rounding():
    """Crops are always the requested size, whatever the centroid (#778).

    `make_centered_bboxes` corners are float32; for some centroids their span
    comes out just under `size - 1` (190.99998 for 192 px), which truncation
    turned into a crop 1 px short for the whole batch.
    """
    image = torch.zeros(1, 1, 256, 320)
    boxes = make_centered_bboxes(
        torch.tensor([[221.54798889160156, 77.36744689941406]]), 192, 192
    )
    assert (boxes[0, 1, 0] - boxes[0, 0, 0]).item() < 191  # the float32 shortfall
    assert crop_bboxes(image, boxes, torch.tensor([0])).shape[-2:] == (192, 192)

    torch.manual_seed(0)
    image = _ramp_image(256, 320)
    centroids = torch.rand(500, 2) * torch.tensor([300.0, 240.0])
    for size in (96, 97, 192, 320):
        boxes = make_centered_bboxes(centroids, size, size)
        for i in range(len(centroids)):
            crop = crop_bboxes(image, boxes[i : i + 1], torch.tensor([0]))
            assert crop.shape[-2:] == (size, size)
        # Passing the known size gives the same crops, without deriving it.
        inds = torch.zeros(len(centroids), dtype=torch.long)
        derived = torch.cat(
            [crop_bboxes(image, boxes[i : i + 1], inds[:1]) for i in range(10)]
        )
        given = crop_bboxes(image, boxes[:10], inds[:10], crop_size=(size, size))
        assert torch.equal(derived, given)
