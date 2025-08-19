"""Utilities for models that learn identity.

These functions implement the inference logic for classifying peaks using class maps or
classification vectors.
"""

import numpy as np
from typing import Tuple
import torch
from scipy.optimize import linear_sum_assignment


def group_class_peaks(
    peak_class_probs: torch.Tensor,
    peak_sample_inds: torch.Tensor,
    peak_channel_inds: torch.Tensor,
    n_samples: int,
    n_channels: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group local peaks using class probabilities, matching peaks to classes using the Hungarian algorithm, per (sample, channel) pair."""
    peak_inds_list = []
    class_inds_list = []

    for sample in range(n_samples):
        for channel in range(n_channels):
            # Mask to find peaks belonging to this (sample, channel) pair
            mask = (peak_sample_inds == sample) & (peak_channel_inds == channel)
            if not torch.any(mask):
                continue

            # Extract probabilities for current group
            probs = peak_class_probs[mask]  # (n_peaks_sc, n_classes)
            if probs.numel() == 0:
                continue

            # Run Hungarian algorithm (note: maximize => minimize negative cost)
            cost = -probs.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)

            # Get original indices in peak_class_probs
            masked_indices = (
                torch.nonzero(mask, as_tuple=False)
                .squeeze(1)
                .to(peak_sample_inds.device)
            )
            peak_inds_sc = masked_indices[row_ind]
            class_inds_sc = torch.tensor(col_ind, dtype=torch.int64).to(
                peak_sample_inds.device
            )

            peak_inds_list.append(peak_inds_sc)
            class_inds_list.append(class_inds_sc)

    if not peak_inds_list:
        return (
            torch.empty(0, dtype=torch.int64).to(peak_sample_inds.device),
            torch.empty(0, dtype=torch.int64).to(peak_sample_inds.device),
        )

    peak_inds = torch.cat(peak_inds_list, dim=0).to(peak_sample_inds.device)
    class_inds = torch.cat(class_inds_list, dim=0).to(peak_sample_inds.device)

    # Filter to keep only best class per peak
    matched_probs = peak_class_probs[peak_inds, class_inds]
    best_probs = peak_class_probs[peak_inds].max(dim=1).values
    is_best = matched_probs == best_probs

    return peak_inds[is_best], class_inds[is_best]


def classify_peaks_from_maps(
    class_maps: torch.Tensor,
    peak_points: torch.Tensor,
    peak_vals: torch.Tensor,
    peak_sample_inds: torch.Tensor,
    peak_channel_inds: torch.Tensor,
    n_channels: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify and group local peaks by their class map probability.

    Args:
        class_maps: Class maps with shape `(n_samples, n_classes, height, width, )`.
        peak_points: Local peak coordinates of shape `(n_peaks,)`. These should be in the same scale as the class maps.
        peak_vals: Confidence map value with shape `(n_peaks,)`.
        peak_sample_inds: Sample index for each peak with shape `(n_peaks,)`.
        peak_channel_inds: Channel index for each peak with shape `(n_peaks,)`.
        n_channels: Integer number of channels (nodes) the instances should have.

    Returns:
        A tuple of `(points, point_vals, class_probs)` containing the grouped peaks.

        `points`: Predicted instances `(n_samples, n_classes, n_peaks, 2)`. Missing points will be denoted by
            NaNs.

        `point_vals`: The confidence map values for each point with shape `(n_samples, n_classes, n_peaks)`.

        `class_probs`: Classification probabilities for matched points with shape `(n_samples, n_classes, n_peaks)`.

    See also: group_class_peaks
    """
    # Build subscripts and pull out class probabilities for each peak from class maps.
    n_samples, n_instances, h, w = class_maps.shape
    peak_sample_inds = peak_sample_inds.to(torch.int32)
    peak_channel_inds = peak_channel_inds.to(torch.int32)

    subs = torch.cat(
        [
            peak_sample_inds.view(-1, 1),
            torch.round(torch.flip(peak_points, dims=[1])).to(torch.int32),
        ],
        dim=1,
    )
    subs[:, 1] = subs[:, 1].clamp(0, h - 1)
    subs[:, 2] = subs[:, 2].clamp(0, w - 1)

    peak_class_probs = class_maps[subs[:, 0], :, subs[:, 1], subs[:, 2]]

    # Classify the peaks.
    peak_inds, class_inds = group_class_peaks(
        peak_class_probs, peak_sample_inds, peak_channel_inds, n_samples, n_channels
    )

    # Assign the results to fixed size tensors.
    subs = torch.stack(
        [peak_sample_inds[peak_inds], class_inds, peak_channel_inds[peak_inds]], dim=1
    )

    points = torch.full(
        (n_samples, n_instances, n_channels, 2), float("nan"), device=class_maps.device
    )
    point_vals = torch.full(
        (n_samples, n_instances, n_channels), float("nan"), device=class_maps.device
    )
    class_probs = torch.full(
        (n_samples, n_instances, n_channels), float("nan"), device=class_maps.device
    )

    points[subs[:, 0], subs[:, 1], subs[:, 2]] = peak_points[peak_inds]
    point_vals[subs[:, 0], subs[:, 1], subs[:, 2]] = peak_vals[peak_inds]

    gather_inds = torch.stack([peak_inds, class_inds], dim=1)
    gathered_class_probs = peak_class_probs[gather_inds[:, 0], gather_inds[:, 1]]

    class_probs[subs[:, 0], subs[:, 1], subs[:, 2]] = gathered_class_probs

    return points, point_vals, class_probs


def get_class_inds_from_vectors(peak_class_probs: torch.Tensor):
    """Get class indices from the probability scores.

    Args:
        peak_class_probs: (n_samples, n_classes) softmax output for each sample

    Returns:
        class_inds: (n_samples,) class index assigned to each sample
        class_probs: (n_samples,) the probability of the assigned class
    """
    n_samples, n_classes = peak_class_probs.shape

    # Run Hungarian matching on negative probabilities (maximize total confidence)
    row_inds, col_inds = linear_sum_assignment(-peak_class_probs.cpu().numpy())

    # Initialize result tensors
    class_inds = torch.full((n_samples,), -1, dtype=torch.int64)
    class_probs = torch.full((n_samples,), float("nan"))

    # Assign class IDs and probabilities to samples
    for sample_idx, class_idx in zip(row_inds, col_inds):
        class_inds[sample_idx] = class_idx
        class_probs[sample_idx] = peak_class_probs[sample_idx, class_idx]

    return class_inds, class_probs
