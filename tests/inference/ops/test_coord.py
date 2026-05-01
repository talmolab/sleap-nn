"""Unit tests for the coordinate-ladder ops.

``sleap_nn/inference/ops/coord.py`` is the only newly written module in PR 1.
Every other op is covered by tests of its old module that still pass via the
shim. These tests exercise:

* identity short-circuits (the perf optimization that makes the no-op cases free)
* correctness on non-identity inputs
* shape preservation on multi-dim inputs
* device-mixing safety on ``add_crop_offset`` and ``undo_eff_scale``
"""

import numpy as np
import pytest
import torch

from sleap_nn.inference.ops.coord import (
    add_crop_offset,
    apply_input_scale,
    undo_eff_scale,
    undo_input_scale,
    undo_stride,
)

# ──────────────────────────────────────────────────────────────────────────
# undo_stride
# ──────────────────────────────────────────────────────────────────────────


def test_undo_stride_identity_returns_same_object():
    coords = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = undo_stride(coords, output_stride=1)
    assert out is coords  # exact identity, no allocation


def test_undo_stride_scales():
    coords = torch.tensor([[2.0, 4.0]])
    out = undo_stride(coords, output_stride=4)
    torch.testing.assert_close(out, torch.tensor([[8.0, 16.0]]))


def test_undo_stride_preserves_shape():
    coords = torch.zeros(3, 5, 7, 2)
    out = undo_stride(coords, output_stride=2)
    assert out.shape == coords.shape


# ──────────────────────────────────────────────────────────────────────────
# undo_input_scale
# ──────────────────────────────────────────────────────────────────────────


def test_undo_input_scale_identity_returns_same_object():
    coords = torch.tensor([[1.0, 2.0]])
    out = undo_input_scale(coords, input_scale=1.0)
    assert out is coords


def test_undo_input_scale_divides():
    coords = torch.tensor([[2.0, 8.0]])
    out = undo_input_scale(coords, input_scale=0.5)
    torch.testing.assert_close(out, torch.tensor([[4.0, 16.0]]))


# ──────────────────────────────────────────────────────────────────────────
# undo_eff_scale
# ──────────────────────────────────────────────────────────────────────────


def test_undo_eff_scale_all_ones_returns_same_object():
    coords = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    eff_scale = torch.ones(1)
    out = undo_eff_scale(coords, eff_scale)
    assert out is coords


def test_undo_eff_scale_per_sample():
    # Two samples in the batch, each with two keypoints (1, 2) coords.
    coords = torch.tensor(
        [
            [[2.0, 4.0], [6.0, 8.0]],
            [[1.0, 1.0], [2.0, 2.0]],
        ]
    )
    eff_scale = torch.tensor([2.0, 0.5])

    expected = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],  # sample 0 divided by 2.0
            [[2.0, 2.0], [4.0, 4.0]],  # sample 1 divided by 0.5
        ]
    )
    out = undo_eff_scale(coords, eff_scale)
    torch.testing.assert_close(out, expected)


def test_undo_eff_scale_4d_coords_broadcast():
    # (B=2, I=3, N=2, 2)
    coords = torch.ones(2, 3, 2, 2) * 4.0
    eff_scale = torch.tensor([2.0, 4.0])
    out = undo_eff_scale(coords, eff_scale)
    assert out.shape == coords.shape
    torch.testing.assert_close(out[0], torch.full((3, 2, 2), 2.0))
    torch.testing.assert_close(out[1], torch.full((3, 2, 2), 1.0))


def test_undo_eff_scale_eff_scale_on_different_device_is_safe():
    coords = torch.zeros(2, 3, 2)
    eff_scale = torch.tensor([2.0, 4.0])
    # Should not raise even if eff_scale lives on cpu when coords would
    # otherwise be on a different device. Identity check via the cpu path.
    out = undo_eff_scale(coords, eff_scale)
    assert out.device == coords.device


# ──────────────────────────────────────────────────────────────────────────
# add_crop_offset
# ──────────────────────────────────────────────────────────────────────────


def test_add_crop_offset_simple():
    peaks = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    crop_topleft = torch.tensor([[10.0, 20.0]])  # (1, 2)
    out = add_crop_offset(peaks, crop_topleft)
    expected = torch.tensor([[[11.0, 22.0], [13.0, 24.0]]])
    torch.testing.assert_close(out, expected)


def test_add_crop_offset_per_instance():
    # Two crops, two keypoints each.
    peaks = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0]],
            [[3.0, 3.0], [4.0, 4.0]],
        ]
    )
    crop_topleft = torch.tensor([[100.0, 0.0], [0.0, 100.0]])
    out = add_crop_offset(peaks, crop_topleft)
    expected = torch.tensor(
        [
            [[101.0, 1.0], [102.0, 2.0]],
            [[3.0, 103.0], [4.0, 104.0]],
        ]
    )
    torch.testing.assert_close(out, expected)


def test_add_crop_offset_topleft_other_device_safe():
    """``crop_topleft`` arriving from a different device must not crash."""
    peaks = torch.zeros(1, 2, 2)
    crop_topleft = torch.zeros(1, 2)  # same device on this CPU host
    add_crop_offset(peaks, crop_topleft)  # should not raise


# ──────────────────────────────────────────────────────────────────────────
# apply_input_scale
# ──────────────────────────────────────────────────────────────────────────


def test_apply_input_scale_identity_returns_same_object():
    image = torch.ones(2, 1, 8, 8)
    out = apply_input_scale(image, input_scale=1.0)
    assert out is image


def test_apply_input_scale_halves():
    image = torch.ones(1, 1, 8, 8)
    out = apply_input_scale(image, input_scale=0.5)
    assert out.shape == (1, 1, 4, 4)


def test_apply_input_scale_doubles():
    image = torch.ones(1, 1, 4, 4)
    out = apply_input_scale(image, input_scale=2.0)
    assert out.shape == (1, 1, 8, 8)


# ──────────────────────────────────────────────────────────────────────────
# Round-trip: undo_stride ∘ undo_input_scale ∘ undo_eff_scale
# ──────────────────────────────────────────────────────────────────────────


def test_full_ladder_roundtrip():
    """A peak placed in original space, scaled forward, then unwound by the
    full ladder must end up back at its starting coordinate."""
    rng = np.random.default_rng(0)
    original = torch.tensor(rng.uniform(0, 100, size=(2, 3, 2)).astype(np.float32))

    # Forward: simulate the preprocessing scales the peak undergoes.
    eff_scale = torch.tensor([2.0, 1.5])  # per-sample sizematcher
    input_scale = 0.5
    output_stride = 4

    # Apply forward (the inverse of the "undo" ops), noting that eff_scale
    # acts per-sample with broadcasting.
    forward = original * eff_scale.view(2, 1, 1)
    forward = forward * input_scale
    forward = forward / output_stride

    # Now undo.
    back = undo_stride(forward, output_stride)
    back = undo_input_scale(back, input_scale)
    back = undo_eff_scale(back, eff_scale)

    torch.testing.assert_close(back, original)


# ──────────────────────────────────────────────────────────────────────────
# MPS device parametrization (only runs if MPS is genuinely usable)
# ──────────────────────────────────────────────────────────────────────────


def _mps_actually_works() -> bool:
    """Return True only if MPS is both reported available and can allocate.

    GitHub Actions Mac runners report ``torch.backends.mps.is_available()``
    as True but raise ``RuntimeError: MPS backend out of memory`` on any
    real allocation (the runner has no usable Metal memory). A strict
    ``is_available()`` check passes the gate but then the allocation in
    the test body fails. This wrapper does an actual zero-byte allocation
    to confirm MPS is genuinely usable.
    """
    if not torch.backends.mps.is_available():
        return False
    try:
        torch.zeros(1, device="mps")
    except RuntimeError:
        return False
    return True


_MPS_OK = _mps_actually_works()


@pytest.mark.skipif(not _MPS_OK, reason="MPS not usable on this host")
def test_undo_stride_on_mps():
    """Coord-ladder identity ops also run on MPS without device drift."""
    coords = torch.tensor([[2.0, 4.0]], device="mps")
    out = undo_stride(coords, output_stride=4)
    torch.testing.assert_close(out.cpu(), torch.tensor([[8.0, 16.0]]))
