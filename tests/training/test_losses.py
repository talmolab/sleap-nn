"""Tests for `sleap_nn.training.losses`."""

import math

import numpy as np
import pytest
import torch

from sleap_nn.training.losses import compute_centroid_focal_loss


def _numpy_reference(y_preds, y, alpha=2.0, beta=4.0, pos_threshold=0.5, eps=1e-4):
    """Independent numpy re-implementation of the penalty-reduced focal loss."""
    y_preds = np.clip(y_preds, eps, 1.0 - eps)
    pos_mask = (y >= pos_threshold).astype(y_preds.dtype)
    neg_mask = 1.0 - pos_mask

    pos_loss = -((1.0 - y_preds) ** alpha) * np.log(y_preds) * pos_mask
    neg_loss = (
        -((1.0 - y) ** beta) * (y_preds**alpha) * np.log(1.0 - y_preds) * neg_mask
    )
    return pos_loss + neg_loss


def test_matches_numpy_reference():
    """Elementwise loss matches an independent numpy implementation."""
    rng = np.random.default_rng(0)
    y_np = rng.random((2, 1, 8, 8)).astype(np.float32)
    y_preds_np = rng.random((2, 1, 8, 8)).astype(np.float32)

    y = torch.from_numpy(y_np)
    y_preds = torch.from_numpy(y_preds_np)

    expected = _numpy_reference(y_preds_np, y_np)
    actual = compute_centroid_focal_loss(y_preds, y, reduction="none").numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    expected_mean = expected.mean()
    actual_mean = compute_centroid_focal_loss(y_preds, y).item()
    assert abs(actual_mean - expected_mean) < 1e-5


def test_confident_correct_predictions_give_low_loss():
    """Near-perfect predictions (pred matches target class) give a small loss."""
    y = torch.zeros(1, 1, 4, 4)
    y[0, 0, 2, 2] = 1.0  # single positive pixel

    y_preds = torch.full((1, 1, 4, 4), 1e-4)  # confidently background everywhere
    y_preds[0, 0, 2, 2] = 1.0 - 1e-4  # confidently positive at the peak

    loss = compute_centroid_focal_loss(y_preds, y)
    assert loss.item() < 0.01


def test_confidently_wrong_predictions_give_high_loss():
    """Predictions inverted relative to the target give a large loss."""
    y = torch.zeros(1, 1, 4, 4)
    y[0, 0, 2, 2] = 1.0

    y_preds = torch.full((1, 1, 4, 4), 1.0 - 1e-4)  # confidently positive everywhere
    y_preds[0, 0, 2, 2] = 1e-4  # confidently background at the true peak

    loss = compute_centroid_focal_loss(y_preds, y)
    assert loss.item() > 1.0


def test_pos_threshold_boundary():
    """Positive/negative branch assignment follows `y >= pos_threshold`, not `y == 1`."""
    y_preds = torch.full((1, 1, 1, 3), 0.5)

    # Values straddling the default 0.5 threshold.
    y = torch.tensor([[[[0.49, 0.5, 0.51]]]])
    elementwise = compute_centroid_focal_loss(y_preds, y, reduction="none")

    expected = _numpy_reference(y_preds.numpy(), y.numpy(), pos_threshold=0.5)
    np.testing.assert_allclose(elementwise.numpy(), expected, rtol=1e-5, atol=1e-6)

    # 0.5 and 0.51 should be treated as positive (same branch), 0.49 as negative.
    pos_at_threshold = elementwise[0, 0, 0, 1].item()
    pos_above_threshold = elementwise[0, 0, 0, 2].item()
    neg_below_threshold = elementwise[0, 0, 0, 0].item()
    assert pos_at_threshold != neg_below_threshold

    # Sanity: an explicit custom threshold flips the classification of 0.5 itself.
    elementwise_custom = compute_centroid_focal_loss(
        y_preds, y, pos_threshold=0.6, reduction="none"
    )
    expected_custom = _numpy_reference(y_preds.numpy(), y.numpy(), pos_threshold=0.6)
    np.testing.assert_allclose(
        elementwise_custom.numpy(), expected_custom, rtol=1e-5, atol=1e-6
    )


def test_reduction_none_matches_mean():
    """`reduction="none"` returns the unreduced tensor whose mean equals `reduction="mean"`."""
    rng = np.random.default_rng(1)
    y = torch.from_numpy(rng.random((2, 1, 5, 5)).astype(np.float32))
    y_preds = torch.from_numpy(rng.random((2, 1, 5, 5)).astype(np.float32))

    elementwise = compute_centroid_focal_loss(y_preds, y, reduction="none")
    reduced = compute_centroid_focal_loss(y_preds, y, reduction="mean")

    assert elementwise.shape == y_preds.shape
    assert abs(elementwise.mean().item() - reduced.item()) < 1e-6


def test_gradient_flows_and_is_finite():
    """Loss supports backpropagation with finite gradients."""
    y = torch.zeros(1, 1, 8, 8)
    y[0, 0, 4, 4] = 1.0
    y_preds = torch.full((1, 1, 8, 8), 0.5, requires_grad=True)

    loss = compute_centroid_focal_loss(y_preds, y)
    loss.backward()

    assert y_preds.grad is not None
    assert torch.isfinite(y_preds.grad).all()


def test_extreme_predictions_do_not_produce_nan_or_inf():
    """Predictions at the very edge of (0, 1) stay numerically stable via clamping."""
    y = torch.tensor([[[[0.0, 1.0]]]])
    y_preds = torch.tensor([[[[0.0, 1.0]]]], requires_grad=True)

    loss = compute_centroid_focal_loss(y_preds, y)
    assert torch.isfinite(loss).all()

    loss.backward()
    assert torch.isfinite(y_preds.grad).all()


def test_alpha_zero_removes_easy_example_downweighting():
    """With alpha=0, the focal `(1 - p)^alpha` / `p^alpha` modulation term is inert."""
    y = torch.zeros(1, 1, 1, 2)
    y[0, 0, 0, 0] = 1.0  # positive pixel
    y_preds = torch.tensor([[[[0.9, 0.1]]]])

    loss = compute_centroid_focal_loss(y_preds, y, alpha=0.0, beta=4.0)

    # alpha=0 leaves: pos -> -log(p); neg -> -(1-y)^beta * log(1-p).
    expected = -math.log(0.9) + -((1.0 - 0.0) ** 4.0) * math.log(1.0 - 0.1)
    expected /= 2.0
    assert abs(loss.item() - expected) < 1e-5


def test_higher_alpha_downweights_confident_correct_pixel_more():
    """Increasing alpha shrinks the loss contribution of an already-confident positive pixel."""
    y = torch.ones(1, 1, 1, 1)
    y_preds = torch.tensor([[[[0.99]]]])

    loss_low_alpha = compute_centroid_focal_loss(y_preds, y, alpha=0.5, beta=4.0)
    loss_high_alpha = compute_centroid_focal_loss(y_preds, y, alpha=4.0, beta=4.0)

    assert loss_high_alpha.item() < loss_low_alpha.item()


@pytest.mark.parametrize("shape", [(1, 1, 1, 1), (3, 1, 12, 12), (2, 2, 6, 6)])
def test_output_shape_and_dtype(shape):
    """Function accepts arbitrary shapes and preserves scalar/elementwise output conventions."""
    y = torch.rand(*shape)
    y_preds = torch.rand(*shape)

    scalar = compute_centroid_focal_loss(y_preds, y)
    assert scalar.ndim == 0

    elementwise = compute_centroid_focal_loss(y_preds, y, reduction="none")
    assert elementwise.shape == shape
