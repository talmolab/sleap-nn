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


# ---------------------------------------------------------------------------
# Contrastive losses for the `embedding` model type.
#
# The OBJECTIVE = positives x negatives x loss (SPEC §4). Sampling builds a batch;
# `build_contrastive_masks` turns each item's (video, frame, group, item_id) into a
# positive mask + a negative-eligibility mask; the loss consumes
# `(embeddings, pos_mask, neg_mask)`. So swapping supcon/infonce/triplet is purely the
# loss; swapping the sampling regime is purely the masks.
# ---------------------------------------------------------------------------
def build_contrastive_masks(
    item_id: torch.Tensor,
    video: torch.Tensor,
    frame: torch.Tensor,
    group: torch.Tensor,
    positives_scope: str = "global_id",
    negatives_sources=("same_frame", "in_batch"),
    exclude_same_track: bool = True,
    restrict_same_video: bool = False,
):
    """Build the positive + negative-eligibility masks for a contrastive batch.

    Args:
        item_id: (B,) id of the ORIGINAL crop; augmented views share an item_id.
        video: (B,) video id (for same_frame + restrict_same_video).
        frame: (B,) frame id within a video (negative = -1 if unknown).
        group: (B,) group key — identity (global_id) or (video, track) (tracklet).
        positives_scope: ``aug_view`` | ``tracklet`` | ``global_id``.
        negatives_sources: subset of ``{same_frame, in_batch}``.
        exclude_same_track: drop same-group pairs from negatives.
        restrict_same_video: restrict negatives to same-video pairs. Required for
            video-local (tracklet) ids: cross-video pairs are UNKNOWN and must never
            be used as negatives (they go in neither mask).

    Returns:
        ``(pos_mask, neg_mask)``, both bool ``(B, B)`` with a False diagonal.
    """
    B = item_id.shape[0]
    eye = torch.eye(B, device=item_id.device, dtype=torch.bool)

    same_item = item_id[:, None] == item_id[None, :]
    same_group = group[:, None] == group[None, :]
    same_video = video[:, None] == video[None, :]
    same_frame = same_video & (frame[:, None] == frame[None, :]) & (frame[:, None] >= 0)

    # Positives: augmented views always; tracklet/global_id add same_group.
    pos = same_item.clone()
    if positives_scope in ("tracklet", "global_id"):
        pos = pos | same_group
    pos = pos & ~eye

    # Negative eligibility (a KNOWN-different pair, not merely "not known-positive").
    neg = torch.zeros_like(pos)
    if "in_batch" in negatives_sources:
        neg = neg | ~eye
    if "same_frame" in negatives_sources:
        neg = neg | same_frame
    neg = neg & ~pos & ~eye
    if exclude_same_track:
        neg = neg & ~same_group
    if restrict_same_video:
        neg = neg & same_video
    return pos, neg


def supcon_loss(z, pos_mask, neg_mask, temperature: float = 0.1):
    """Supervised contrastive loss (Khosla et al.) over masks. `z` is L2-normalized."""
    B = z.shape[0]
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    sim = (z @ z.T) / temperature
    sim = sim - sim.max(1, keepdim=True).values.detach()  # numerical stability
    denom = (pos_mask | neg_mask) & ~eye
    exp = torch.exp(sim) * denom.float()
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    pos = pos_mask & ~eye
    pos_cnt = pos.sum(1)
    valid = pos_cnt > 0
    if not valid.any():
        return (sim * 0).sum()
    mean_log_prob_pos = (pos.float() * log_prob).sum(1) / pos_cnt.clamp(min=1)
    return -mean_log_prob_pos[valid].mean()


def infonce_loss(z, pos_mask, neg_mask, temperature: float = 0.1):
    """NT-Xent / InfoNCE: log of summed-positive over summed-contrast."""
    B = z.shape[0]
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    sim = (z @ z.T) / temperature
    sim = sim - sim.max(1, keepdim=True).values.detach()
    denom = (pos_mask | neg_mask) & ~eye
    pos = pos_mask & ~eye
    exp = torch.exp(sim)
    denom_sum = (exp * denom.float()).sum(1)
    pos_sum = (exp * pos.float()).sum(1)
    valid = pos.sum(1) > 0
    if not valid.any():
        return (sim * 0).sum()
    loss = -torch.log((pos_sum + 1e-12) / (denom_sum + 1e-12))
    return loss[valid].mean()


def triplet_loss(z, pos_mask, neg_mask, margin: float = 0.2):
    """Batch-hard triplet on cosine distance with a margin. `z` is L2-normalized."""
    B = z.shape[0]
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    dist = 1.0 - (z @ z.T)
    pos = pos_mask & ~eye
    neg = neg_mask & ~eye
    big = dist.max().detach() + 1.0
    hardest_pos = torch.where(pos, dist, torch.zeros_like(dist)).max(1).values
    hardest_neg = (
        torch.where(neg, dist, torch.full_like(dist, float(big))).min(1).values
    )
    valid = (pos.sum(1) > 0) & (neg.sum(1) > 0)
    if not valid.any():
        return (dist * 0).sum()
    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss[valid].mean()


_CONTRASTIVE_LOSSES = {
    "supcon": supcon_loss,
    "infonce": infonce_loss,
    "triplet": triplet_loss,
}


def get_contrastive_loss(name: str):
    """Return the contrastive loss function by name (supcon|infonce|triplet)."""
    if name not in _CONTRASTIVE_LOSSES:
        raise ValueError(
            f"Unknown contrastive loss '{name}'; choose one of "
            f"{list(_CONTRASTIVE_LOSSES)}."
        )
    return _CONTRASTIVE_LOSSES[name]
