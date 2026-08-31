"""Custom loss functions."""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_ohkm_loss(
    y_gt: torch.Tensor,
    y_pr: torch.Tensor,
    hard_to_easy_ratio: float = 2.0,
    min_hard_keypoints: int = 2,
    max_hard_keypoints: Optional[int] = None,
    loss_scale: float = 5.0,
) -> torch.Tensor:
    """Compute the online hard keypoint mining loss."""
    if max_hard_keypoints is None:
        max_hard_keypoints = -1
    # Compute elementwise squared difference.
    loss = (y_pr - y_gt) ** 2

    # Store initial shape for normalization.
    batch_shape = loss.shape

    # Reduce over everything but channels axis.
    l = torch.sum(loss, dim=(0, 2, 3))

    # Compute the loss for the "easy" keypoint.
    best_loss = torch.min(l)

    # Find the number of hard keypoints.
    is_hard_keypoint = (l / best_loss) >= hard_to_easy_ratio
    n_hard_keypoints = torch.sum(is_hard_keypoint.to(torch.int32))

    # Work out the actual final number of keypoints to consider as hard.
    if max_hard_keypoints < 0:
        max_hard_keypoints = l.shape[0]
    else:
        max_hard_keypoints = min(
            max_hard_keypoints,
            l.shape[0],
        )
    k = min(
        max(
            n_hard_keypoints,
            min_hard_keypoints,
        ),
        max_hard_keypoints,
    )

    # Pull out the top hard values.
    k_vals, k_inds = torch.topk(l, k=k, largest=True, sorted=False)

    # Apply weights.
    k_loss = k_vals * loss_scale

    # Reduce over all channels.
    n_elements = batch_shape[0] * batch_shape[2] * batch_shape[3] * k
    k_loss = torch.sum(k_loss) / n_elements

    return k_loss


def compute_bce_dice_loss(
    y_pred: torch.Tensor,
    y_gt: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    smooth: float = 1.0,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    """Compute combined Binary Cross-Entropy and Dice loss for segmentation.

    Args:
        y_pred: Predicted logits (before sigmoid). Shape: (B, 1, H, W).
        y_gt: Ground truth binary mask. Shape: (B, 1, H, W).
        bce_weight: Weight for the BCE component.
        dice_weight: Weight for the Dice component.
        smooth: Smoothing factor for Dice loss to avoid division by zero.
        pos_weight: Optional positive-class weight for the BCE term (passed to
            ``binary_cross_entropy_with_logits``). For thin/rare foreground (e.g.
            plant roots at <1% of pixels), a value >1 up-weights the foreground so
            the head does not hedge below 0.5 on faint thin structures. ``None``
            (default) leaves BCE unweighted — byte-for-byte the previous behavior.

    Returns:
        Scalar loss tensor.
    """
    pw = (
        None
        if pos_weight is None
        else torch.as_tensor(pos_weight, dtype=y_pred.dtype, device=y_pred.device)
    )
    bce_loss = F.binary_cross_entropy_with_logits(
        y_pred, y_gt, reduction="mean", pos_weight=pw
    )

    # Dice loss (apply sigmoid to convert logits to probabilities)
    y_pred_sig = torch.sigmoid(y_pred)
    intersection = (y_pred_sig * y_gt).sum(dim=(2, 3))
    union = y_pred_sig.sum(dim=(2, 3)) + y_gt.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice.mean()

    return bce_weight * bce_loss + dice_weight * dice_loss


def compute_masked_smooth_l1(
    y_pred: torch.Tensor,
    y_gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute smooth L1 loss only on masked (foreground) pixels.

    Args:
        y_pred: Predicted offset field. Shape: (B, 2, H, W).
        y_gt: Ground truth offset field. Shape: (B, 2, H, W).
        mask: Binary mask indicating valid pixels. Shape: (B, 1, H, W).

    Returns:
        Scalar loss tensor. Returns 0 if no foreground pixels.
    """
    # Expand mask to match offset channels
    mask_expanded = mask.expand_as(y_pred)  # (B, 2, H, W)

    n_valid = mask_expanded.sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    loss = F.smooth_l1_loss(
        y_pred * mask_expanded, y_gt * mask_expanded, reduction="sum"
    )
    return loss / n_valid


def compute_centroid_focal_loss(
    y_preds: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    pos_threshold: float = 0.5,
    eps: float = 1e-4,
    reduction: str = "mean",
) -> torch.Tensor:
    """CenterNet/CornerNet-style penalty-reduced pixelwise focal loss.

    Adapted for sleap-nn's continuous (sub-pixel) Gaussian confmap targets --
    see `sleap_nn.data.confidence_maps.make_confmaps`, which evaluates the
    Gaussian at the true keypoint location on a fixed grid, so the peak
    pixel's target value is generally close to but not exactly 1.0 (unlike
    the original CenterNet formulation, which snaps the peak to an exact
    grid cell with target 1.0). Positive pixels are therefore defined as
    `y >= pos_threshold` rather than `y == 1`.

    For positive pixels: ``-(1 - y_preds)^alpha * log(y_preds)``.
    For the rest: ``-(1 - y)^beta * y_preds^alpha * log(1 - y_preds)``.

    Requires `y_preds` to be a calibrated `(0, 1)` probability (e.g. from a
    head with a sigmoid output activation -- see
    `CentroidConfmapsHead.use_sigmoid_activation`), not raw/unbounded
    regression output.

    Args:
        y_preds: Predicted confidence maps, expected in `(0, 1)` (values
            outside this range are clamped). Any shape.
        y: Ground-truth confidence maps (continuous Gaussian targets), same
            shape as `y_preds`.
        alpha: Focal exponent applied to both branches -- down-weights
            already-confident (easy) pixels on both the positive and
            negative side. Default 2.0 (standard CenterNet value).
        beta: Penalty-reduction exponent for negative pixels near a true
            peak (`y` close to but below `pos_threshold` counts less against
            the model). Default 4.0 (standard CenterNet value).
        pos_threshold: Minimum target value for a pixel to count as
            "positive" (near a true peak). Default 0.5 (~1.18 sigma from the
            true peak).
        eps: Clamp margin to keep `log` finite.
        reduction: ``"mean"`` (default, same convention as plain
            ``nn.MSELoss()``) or ``"none"`` to return the elementwise loss
            tensor unreduced (same shape as ``y_preds``) -- used by
            ``CentroidLightningModule._compute_loss`` to take a per-sample
            mean before applying the existing negative-frame weighting on
            top.

    Returns:
        Mean-reduced scalar loss tensor, or the elementwise loss tensor if
        ``reduction="none"``.
    """
    y_preds = torch.clamp(y_preds, eps, 1.0 - eps)
    pos_mask = (y >= pos_threshold).to(y_preds.dtype)
    neg_mask = 1.0 - pos_mask

    pos_loss = -torch.pow(1.0 - y_preds, alpha) * torch.log(y_preds) * pos_mask
    neg_loss = (
        -torch.pow(1.0 - y, beta)
        * torch.pow(y_preds, alpha)
        * torch.log(1.0 - y_preds)
        * neg_mask
    )

    elementwise = pos_loss + neg_loss
    return elementwise if reduction == "none" else elementwise.mean()
