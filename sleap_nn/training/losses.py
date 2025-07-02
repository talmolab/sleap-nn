"""Custom loss functions."""

import torch
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
