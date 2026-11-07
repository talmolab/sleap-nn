"""Tests for `sleap_nn.inference.ops.identity` peak grouping by class maps."""

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from sleap_nn.inference.ops.identity import classify_peaks_from_maps, group_class_peaks

DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")
if torch.cuda.is_available():
    DEVICES.append("cuda")


def _loop_group_class_peaks(
    peak_class_probs, peak_sample_inds, peak_channel_inds, n_samples, n_channels
):
    """The per-(sample, channel) loop that `group_class_peaks` used before."""
    peak_inds_list = []
    class_inds_list = []
    for sample in range(n_samples):
        for channel in range(n_channels):
            mask = (peak_sample_inds == sample) & (peak_channel_inds == channel)
            if not torch.any(mask):
                continue
            probs = peak_class_probs[mask]
            if probs.numel() == 0:
                continue
            cost = -probs.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            masked_indices = (
                torch.nonzero(mask, as_tuple=False)
                .squeeze(1)
                .to(peak_sample_inds.device)
            )
            peak_inds_list.append(masked_indices[row_ind])
            class_inds_list.append(
                torch.tensor(col_ind, dtype=torch.int64).to(peak_sample_inds.device)
            )
    if not peak_inds_list:
        return (
            torch.empty(0, dtype=torch.int64).to(peak_sample_inds.device),
            torch.empty(0, dtype=torch.int64).to(peak_sample_inds.device),
        )
    peak_inds = torch.cat(peak_inds_list, dim=0).to(peak_sample_inds.device)
    class_inds = torch.cat(class_inds_list, dim=0).to(peak_sample_inds.device)
    matched_probs = peak_class_probs[peak_inds, class_inds]
    best_probs = peak_class_probs[peak_inds].max(dim=1).values
    is_best = matched_probs == best_probs
    return peak_inds[is_best], class_inds[is_best]


def _random_class_peaks(rng, B, n_channels, n_classes, ties=False, shuffle=True):
    n = int(rng.integers(0, 8 * B * n_channels // 2 + 1))
    sample_inds = np.sort(rng.integers(0, B, n))
    channel_inds = rng.integers(0, n_channels, n)
    # With ``shuffle``, nodes are mixed within a sample, as `find_local_peaks`
    # gives them; without, peaks are grouped by (sample, node).
    if not shuffle:
        order = np.lexsort((channel_inds, sample_inds))
        sample_inds, channel_inds = sample_inds[order], channel_inds[order]
    logits = rng.normal(size=(n, n_classes)) * 3
    if ties:  # coarse values, so equal probabilities are common
        logits = np.round(logits)
    probs = torch.softmax(torch.from_numpy(logits).float(), dim=1)
    return (
        probs,
        torch.from_numpy(sample_inds).to(torch.int32),
        torch.from_numpy(channel_inds).to(torch.int32),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ties", [False, True])
def test_group_class_peaks_matches_loop(device, ties):
    rng = np.random.default_rng(0)
    for trial in range(150):
        B = int(rng.integers(1, 6))
        n_channels = int(rng.integers(1, 6))
        n_classes = int(rng.integers(1, 6))
        probs, s, c = _random_class_peaks(
            rng, B, n_channels, n_classes, ties=ties, shuffle=bool(trial % 2)
        )
        if trial % 7 == 0:
            probs = probs.half()
        args = (probs.to(device), s.to(device), c.to(device), B, n_channels)
        old = _loop_group_class_peaks(*args)
        new = group_class_peaks(*args)
        for o, n in zip(old, new):
            assert n.dtype == o.dtype == torch.int64
            assert n.device == o.device
            assert torch.equal(o, n)


@pytest.mark.parametrize("device", DEVICES)
def test_group_class_peaks_empty(device):
    probs = torch.zeros((0, 3), device=device)
    inds = torch.zeros(0, dtype=torch.int32, device=device)
    peak_inds, class_inds = group_class_peaks(probs, inds, inds, 4, 2)
    assert peak_inds.shape == (0,) and class_inds.shape == (0,)
    assert peak_inds.device == probs.device and peak_inds.dtype == torch.int64


@pytest.mark.parametrize("device", DEVICES)
def test_classify_peaks_from_maps_matches_loop(device, monkeypatch):
    """The whole classification gives the same instances as with the loop."""
    import sleap_nn.inference.ops.identity as identity

    rng = np.random.default_rng(1)
    for _ in range(30):
        B, n_channels, n_classes, h, w = 3, 4, 3, 16, 20
        class_maps = torch.softmax(
            torch.from_numpy(rng.normal(size=(B, n_classes, h, w)) * 3).float(), 1
        ).to(device)
        n = int(rng.integers(0, 30))
        s = torch.from_numpy(np.sort(rng.integers(0, B, n))).to(torch.int32)
        c = torch.from_numpy(rng.integers(0, n_channels, n)).to(torch.int32)
        pts = torch.from_numpy(rng.uniform(-1, [w, h], (n, 2))).float()
        vals = torch.from_numpy(rng.uniform(0.2, 1, n)).float()
        args = (class_maps, pts.to(device), vals.to(device), s.to(device), c.to(device))
        new = classify_peaks_from_maps(*args, n_channels=n_channels)
        monkeypatch.setattr(identity, "group_class_peaks", _loop_group_class_peaks)
        old = classify_peaks_from_maps(*args, n_channels=n_channels)
        monkeypatch.undo()
        for o, nw in zip(old, new):
            assert torch.equal(o.isnan(), nw.isnan())
            assert torch.equal(o.nan_to_num(), nw.nan_to_num())
