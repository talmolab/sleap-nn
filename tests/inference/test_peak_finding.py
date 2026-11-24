import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, label, maximum_filter

from sleap_nn.inference.ops.peaks import (
    morphological_dilation,
    morphological_dilation_split,
)
from sleap_nn.inference.peak_finding import (
    crop_bboxes,
    integral_regression,
    find_global_peaks,
    find_global_peaks_rough,
    find_local_peaks,
    find_local_peaks_rough,
)
from sleap_nn.data.instance_cropping import make_centered_bboxes


def test_crop_bboxes(minimal_bboxes, minimal_cms):
    cms = torch.load(minimal_cms).unsqueeze(0)
    bboxes = torch.load(minimal_bboxes)

    samples = cms.size(0)
    channels = cms.size(1)
    cms = torch.reshape(
        cms,
        [samples * channels, 1, cms.size(2), cms.size(3)],
    )
    valid_idx = torch.arange(0, 13)

    cm_crops = crop_bboxes(cms, bboxes, valid_idx)

    assert cm_crops.shape == (13, 1, 5, 5)
    assert cm_crops.dtype == torch.float32


def test_crop_bboxes_edge_cases():
    """Test crop_bboxes with edge cases like peaks near image boundaries."""
    # Create a test image with peaks at various positions including edges
    img = torch.zeros(1, 1, 20, 20)

    # Peak at center
    img[0, 0, 10, 10] = 1.0

    # Peak at corner (0, 0)
    img[0, 0, 0, 0] = 0.8

    # Peak at edge
    img[0, 0, 0, 10] = 0.9

    # Create bboxes for these peaks
    points = torch.tensor(
        [
            [10.0, 10.0],  # center
            [0.0, 0.0],  # corner
            [10.0, 0.0],  # edge
        ]
    )
    bboxes = make_centered_bboxes(points, box_height=5, box_width=5)
    sample_inds = torch.tensor([0, 0, 0])

    crops = crop_bboxes(img, bboxes, sample_inds)

    assert crops.shape == (3, 1, 5, 5)

    # Center crop should have the peak at center
    assert crops[0, 0, 2, 2] == 1.0

    # Corner crop should have the peak at center (with zero padding)
    assert crops[1, 0, 2, 2] == 0.8

    # Edge crop should have the peak at center
    assert crops[2, 0, 2, 2] == 0.9


def test_crop_bboxes_empty():
    """Test crop_bboxes with empty bboxes."""
    img = torch.zeros(1, 1, 20, 20)
    bboxes = torch.empty(0, 4, 2)
    sample_inds = torch.empty(0, dtype=torch.long)

    crops = crop_bboxes(img, bboxes, sample_inds)

    # Should return empty tensor
    assert crops.shape[0] == 0
    assert crops.shape[1] == 1  # Preserves channel dimension


def test_crop_bboxes_multiple_samples():
    """Test crop_bboxes with multiple samples."""
    # Create 3 samples with different peak locations
    imgs = torch.zeros(3, 1, 20, 20)
    imgs[0, 0, 5, 5] = 1.0
    imgs[1, 0, 10, 10] = 2.0
    imgs[2, 0, 15, 15] = 3.0

    points = torch.tensor(
        [
            [5.0, 5.0],
            [10.0, 10.0],
            [15.0, 15.0],
        ]
    )
    bboxes = make_centered_bboxes(points, box_height=5, box_width=5)
    sample_inds = torch.tensor([0, 1, 2])

    crops = crop_bboxes(imgs, bboxes, sample_inds)

    assert crops.shape == (3, 1, 5, 5)
    assert crops[0, 0, 2, 2] == 1.0
    assert crops[1, 0, 2, 2] == 2.0
    assert crops[2, 0, 2, 2] == 3.0


def test_integral_regression(minimal_bboxes, minimal_cms):
    cms = torch.load(minimal_cms).unsqueeze(0)
    bboxes = torch.load(minimal_bboxes)

    samples = cms.size(0)
    channels = cms.size(1)
    cms = torch.reshape(
        cms,
        [samples * channels, 1, cms.size(2), cms.size(3)],
    )
    valid_idx = torch.arange(0, 13)

    cm_crops = crop_bboxes(cms, bboxes, valid_idx)

    crop_size = 5
    gv = torch.arange(crop_size, dtype=torch.float32) - ((crop_size - 1) / 2)
    dx_hat, dy_hat = integral_regression(cm_crops, xv=gv, yv=gv)

    assert dx_hat.shape == dy_hat.shape == (13, 1)
    assert dx_hat.dtype == dy_hat.dtype == torch.float32


def test_find_global_peaks_rough(minimal_cms):
    cms = torch.load(minimal_cms).unsqueeze(0)

    gt_rough_peaks = torch.Tensor(
        [
            [
                [27.0, 23.0],
                [40.0, 40.0],
                [49.0, 55.0],
                [54.0, 63.0],
                [56.0, 60.0],
                [18.0, 32.0],
                [29.0, 12.0],
                [17.0, 44.0],
                [44.0, 20.0],
                [36.0, 70.0],
                [0.0, 0.0],
                [25.0, 30.0],
                [34.0, 24.0],
            ]
        ]
    )
    gt_peak_vals = torch.Tensor(
        [
            [
                0.9163541793823242,
                0.9957404136657715,
                0.929328203201294,
                0.9020472168922424,
                0.8870090246200562,
                0.8547359108924866,
                0.8420282602310181,
                0.86271071434021,
                0.863940954208374,
                0.8226016163825989,
                1.0,
                0.9693551063537598,
                0.8798434734344482,
            ]
        ]
    )

    rough_peaks, peak_vals = find_global_peaks_rough(cms, threshold=0.1)

    assert rough_peaks.shape == (1, 13, 2)
    assert peak_vals.shape == (1, 13)
    assert rough_peaks.dtype == peak_vals.dtype == torch.float32
    assert torch.equal(gt_rough_peaks, rough_peaks)
    assert torch.equal(gt_peak_vals, peak_vals)


def test_find_global_peaks(minimal_cms):
    cms = torch.load(minimal_cms).unsqueeze(0)  # (1, 13, 80, 80)

    rough_peaks, peak_vals = find_global_peaks(cms, threshold=0.2)

    gt_rough_peaks = torch.Tensor(
        [
            [
                [27.0, 23.0],
                [40.0, 40.0],
                [49.0, 55.0],
                [54.0, 63.0],
                [56.0, 60.0],
                [18.0, 32.0],
                [29.0, 12.0],
                [17.0, 44.0],
                [44.0, 20.0],
                [36.0, 70.0],
                [0.0, 0.0],
                [25.0, 30.0],
                [34.0, 24.0],
            ]
        ]
    )
    gt_peak_vals = torch.Tensor(
        [
            [
                0.9163541793823242,
                0.9957404136657715,
                0.929328203201294,
                0.9020472168922424,
                0.8870090246200562,
                0.8547359108924866,
                0.8420282602310181,
                0.86271071434021,
                0.863940954208374,
                0.8226016163825989,
                1.0,
                0.9693551063537598,
                0.8798434734344482,
            ]
        ]
    )

    assert rough_peaks.shape == (1, 13, 2)
    assert peak_vals.shape == (1, 13)
    assert rough_peaks.dtype == peak_vals.dtype == torch.float32
    assert torch.equal(gt_rough_peaks, rough_peaks)
    assert torch.equal(gt_peak_vals, peak_vals)

    # Unknown refinement names used to fall through to grid peaks silently.
    with pytest.raises(ValueError, match="Unknown peak refinement"):
        find_global_peaks(cms, refinement="invalid_input", threshold=0.2)

    gt_refined_peaks = torch.Tensor(
        [
            [
                [27.2498, 22.8141],
                [39.9390, 40.0320],
                [48.7837, 54.8141],
                [53.8752, 63.3142],
                [56.1249, 60.3423],
                [18.2802, 31.6910],
                [29.0320, 12.4346],
                [17.2178, 43.6591],
                [44.3712, 19.8446],
                [35.6288, 69.7198],
                [0.3252, 0.3252],
                [24.8141, 30.0000],
                [34.0625, 23.6288],
            ]
        ]
    )
    gt_peak_vals = torch.Tensor(
        [
            [
                0.9164,
                0.9957,
                0.9293,
                0.9020,
                0.8870,
                0.8547,
                0.8420,
                0.8627,
                0.8639,
                0.8226,
                1.0000,
                0.9694,
                0.8798,
            ]
        ]
    )

    refined_peaks, peak_vals = find_global_peaks(
        cms, refinement="integral", threshold=0.2
    )

    torch.testing.assert_close(gt_refined_peaks, refined_peaks, atol=0.001, rtol=0.0)
    torch.testing.assert_close(gt_peak_vals, peak_vals, atol=0.001, rtol=0.0)


def test_find_local_peaks_rough(minimal_cms):
    cms = torch.load(minimal_cms).unsqueeze(0)  # (1, 13, 80, 80)

    (
        peak_points,
        peak_vals,
        peak_sample_inds,
        peak_channel_inds,
    ) = find_local_peaks_rough(cms)

    gt_peak_points = torch.Tensor(
        [
            [0.0, 0.0],
            [29.0, 12.0],
            [44.0, 20.0],
            [27.0, 23.0],
            [34.0, 24.0],
            [25.0, 30.0],
            [18.0, 32.0],
            [40.0, 40.0],
            [17.0, 44.0],
            [49.0, 55.0],
            [56.0, 60.0],
            [54.0, 63.0],
            [36.0, 70.0],
        ]
    )

    gt_peak_vals = torch.Tensor(
        [
            1.0,
            0.8420282602310181,
            0.863940954208374,
            0.9163541793823242,
            0.8798434734344482,
            0.9693551063537598,
            0.8547359108924866,
            0.9957404136657715,
            0.86271071434021,
            0.929328203201294,
            0.8870090246200562,
            0.9020472168922424,
            0.8226016163825989,
        ]
    )

    gt_peak_sample_inds = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    gt_peak_channel_inds = torch.Tensor([10, 6, 8, 0, 12, 11, 5, 1, 7, 2, 4, 3, 9])

    assert peak_points.shape == (13, 2)
    assert peak_vals.shape == peak_sample_inds.shape == peak_channel_inds.shape == (13,)
    assert torch.equal(gt_peak_vals, peak_vals)
    assert torch.equal(gt_peak_points, peak_points)
    assert torch.equal(gt_peak_sample_inds, peak_sample_inds)
    assert torch.equal(gt_peak_channel_inds, peak_channel_inds)


def test_find_local_peaks(minimal_cms):
    cms = torch.load(minimal_cms).unsqueeze(0)  # (1, 13, 80, 80)

    peak_points, peak_vals, peak_sample_inds, peak_channel_inds = find_local_peaks(cms)

    gt_peak_points = torch.Tensor(
        [
            [0.0, 0.0],
            [29.0, 12.0],
            [44.0, 20.0],
            [27.0, 23.0],
            [34.0, 24.0],
            [25.0, 30.0],
            [18.0, 32.0],
            [40.0, 40.0],
            [17.0, 44.0],
            [49.0, 55.0],
            [56.0, 60.0],
            [54.0, 63.0],
            [36.0, 70.0],
        ]
    )

    gt_peak_vals = torch.Tensor(
        [
            1.0,
            0.8420282602310181,
            0.863940954208374,
            0.9163541793823242,
            0.8798434734344482,
            0.9693551063537598,
            0.8547359108924866,
            0.9957404136657715,
            0.86271071434021,
            0.929328203201294,
            0.8870090246200562,
            0.9020472168922424,
            0.8226016163825989,
        ]
    )

    gt_peak_sample_inds = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    gt_peak_channel_inds = torch.Tensor([10, 6, 8, 0, 12, 11, 5, 1, 7, 2, 4, 3, 9])

    assert peak_points.shape == (13, 2)
    assert peak_vals.shape == peak_sample_inds.shape == peak_channel_inds.shape == (13,)
    assert torch.equal(gt_peak_vals, peak_vals)
    assert torch.equal(gt_peak_points, peak_points)
    assert torch.equal(gt_peak_sample_inds, peak_sample_inds)
    assert torch.equal(gt_peak_channel_inds, peak_channel_inds)

    peak_points, peak_vals, peak_sample_inds, peak_channel_inds = find_local_peaks(
        cms, refinement="invalid_input"
    )

    assert peak_points.shape == (13, 2)
    assert peak_vals.shape == peak_sample_inds.shape == peak_channel_inds.shape == (13,)
    assert torch.equal(gt_peak_vals, peak_vals)
    assert torch.equal(gt_peak_points, peak_points)
    assert torch.equal(gt_peak_sample_inds, peak_sample_inds)
    assert torch.equal(gt_peak_channel_inds, peak_channel_inds)

    peak_points, peak_vals, peak_sample_inds, peak_channel_inds = find_local_peaks(
        cms, refinement="integral"
    )

    gt_peak_points = torch.Tensor(
        [
            [0.32524189352989197, 0.32524189352989197],
            [29.032001495361328, 12.43461799621582],
            [44.371177673339844, 19.84455680847168],
            [27.249767303466797, 22.814102172851562],
            [34.06249237060547, 23.628822326660156],
            [24.81409454345703, 29.999998092651367],
            [18.28015899658203, 31.690954208374023],
            [39.939002990722656, 40.0319938659668],
            [17.217844009399414, 43.659122467041016],
            [48.78366470336914, 54.814109802246094],
            [56.12494659423828, 60.34233856201172],
            [53.875205993652344, 63.31416702270508],
            [35.628822326660156, 69.71983337402344],
        ]
    )

    gt_peak_vals = torch.Tensor(
        [
            1.0,
            0.8420282602310181,
            0.863940954208374,
            0.9163541793823242,
            0.8798434734344482,
            0.9693551063537598,
            0.8547359108924866,
            0.9957404136657715,
            0.86271071434021,
            0.929328203201294,
            0.8870090246200562,
            0.9020472168922424,
            0.8226016163825989,
        ]
    )

    gt_peak_sample_inds = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    gt_peak_channel_inds = torch.Tensor([10, 6, 8, 0, 12, 11, 5, 1, 7, 2, 4, 3, 9])

    assert peak_points.shape == (13, 2)
    assert peak_vals.shape == peak_sample_inds.shape == peak_channel_inds.shape == (13,)
    torch.testing.assert_close(gt_peak_points, peak_points, atol=0.001, rtol=0.0)
    torch.testing.assert_close(gt_peak_vals, peak_vals, atol=0.001, rtol=0.0)
    assert torch.equal(gt_peak_sample_inds, peak_sample_inds)
    assert torch.equal(gt_peak_channel_inds, peak_channel_inds)


def _gaussian(height, width, cx, cy, sigma=2.5):
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    return torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))


def _peak_list(cms, threshold=0.2):
    """``find_local_peaks_rough`` output as ``[(sample, channel, x, y), ...]``."""
    points, _vals, samples, channels = find_local_peaks_rough(cms, threshold)
    return [
        (int(s), int(c), round(x, 4), round(y, 4))
        for (x, y), s, c in zip(points.tolist(), samples, channels)
    ]


def _strict_peak_mask(cms, threshold=0.2):
    """The previous rule: strictly greater than all 8 neighbors."""
    height, width = cms.shape[-2:]
    max_img = morphological_dilation(cms.reshape(-1, 1, height, width), None)
    return (cms > max_img.reshape(cms.shape)) & (cms > threshold)


def _reference_peaks(cms, threshold=0.2):
    """Brute-force peaks of ``(samples, channels, H, W)`` maps, sorted.

    A peak is a plateau (8-connected equal pixels) above ``threshold`` that
    fits in a 2x2 block and whose surrounding pixels are all lower, at the
    mean of its pixel coordinates.
    """
    peaks = []
    for s, sample in enumerate(cms.float().cpu().numpy()):
        for c, img in enumerate(sample):
            # Every pixel of a peak's plateau is >= its neighbors.
            is_max = maximum_filter(img, size=3, mode="constant", cval=-np.inf)
            for v in np.unique(img[is_max == img]):
                if not v > threshold:
                    continue
                plateaus, n = label(img == v, structure=np.ones((3, 3)))
                for i in range(1, n + 1):
                    plateau = plateaus == i
                    ys, xs = np.nonzero(plateau)
                    ring = binary_dilation(plateau, structure=np.ones((3, 3)))
                    ring &= ~plateau
                    if np.ptp(ys) <= 1 and np.ptp(xs) <= 1 and (img[ring] < v).all():
                        peaks.append(
                            (
                                s,
                                c,
                                round(float(xs.mean()), 4),
                                round(float(ys.mean()), 4),
                            )
                        )
    return sorted(peaks)


def test_morphological_dilation_split_matches_dilation():
    torch.manual_seed(0)
    img = torch.randn(3, 1, 9, 11)
    before, after = morphological_dilation_split(img)
    assert before.shape == after.shape == img.shape
    padded = F.pad(img, (1, 1, 1, 1), value=float("-inf"))
    shifts = [
        padded[..., 1 + dy : 10 + dy, 1 + dx : 12 + dx]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dy or dx
    ]
    assert torch.equal(torch.stack(shifts[:4]).amax(0), before)
    assert torch.equal(torch.stack(shifts[4:]).amax(0), after)
    assert torch.equal(torch.stack(shifts).amax(0), morphological_dilation(img, None))
    # The first pixel has no earlier neighbors; the last has no later ones.
    assert torch.all(before[..., 0, 0] == float("-inf"))
    assert torch.all(after[..., -1, -1] == float("-inf"))


@pytest.mark.parametrize("dx, dy", [(1, 0), (0, 1), (1, 1), (-1, 1)])
def test_find_local_peaks_rough_two_equal_top_pixels(dx, dy):
    """Two equal top pixels (E, S, SE or SW of each other) give one peak.

    The previous strict rule found none (#781). The peak is at their midpoint.
    """
    cms = _gaussian(20, 24, 10, 8)[None, None] * 0.5
    a, b = (10, 8), (10 + dx, 8 + dy)
    for x, y in (a, b):
        cms[0, 0, y, x] = 0.9
    assert not _strict_peak_mask(cms).any()
    assert _peak_list(cms) == [(0, 0, 10 + dx / 2, 8 + dy / 2)]


def test_find_local_peaks_rough_flat_top():
    # A Gaussian centered between two pixels has two equal top pixels; one
    # centered between four has a 2x2 flat top. Each gives one peak at the
    # Gaussian's center.
    cms = _gaussian(20, 24, 10.5, 8)[None, None]
    assert cms[0, 0, 8, 10] == cms[0, 0, 8, 11]
    assert _peak_list(cms) == [(0, 0, 10.5, 8)]
    cms = _gaussian(20, 24, 10.5, 8.5)[None, None]
    assert _peak_list(cms) == [(0, 0, 10.5, 8.5)]


@pytest.mark.parametrize(
    "tied, peak",
    [
        ([(4, 4), (5, 4), (4, 5)], (13 / 3, 13 / 3)),  # an L (3 of a 2x2)
        ([(5, 4), (4, 5), (5, 5)], (14 / 3, 14 / 3)),  # the other way up
        ([(4, 4), (5, 4), (4, 5), (5, 5)], (4.5, 4.5)),  # 2x2
        ([(4, 4), (5, 4), (6, 4)], None),  # a row of 3
        ([(4, 4), (5, 5), (6, 6)], None),  # a diagonal of 3
        ([(5, 3), (4, 4), (3, 4)], None),  # an L 3 pixels wide
        ([(x, y) for x in (4, 5, 6) for y in (4, 5)], None),  # 2x3
        # a flat top that widens by 2 pixels per side
        ([(4, 3), (5, 3)] + [(x, y) for x in range(2, 8) for y in (4, 5)], None),
    ],
)
def test_find_local_peaks_rough_flat_top_shapes(tied, peak):
    """A flat top that fits in a 2x2 block is one peak at its mean; else none."""
    cms = torch.full((1, 1, 9, 10), 0.3)
    for x, y in tied:
        cms[0, 0, y, x] = 0.9
    want = [] if peak is None else [(0, 0) + tuple(round(v, 4) for v in peak)]
    assert _peak_list(cms) == want


def test_find_local_peaks_rough_rejects_tie_next_to_higher():
    # Two equal pixels, one of them next to a higher pixel: a shoulder.
    cms = torch.zeros(1, 1, 7, 9)
    cms[0, 0, 3, 2] = cms[0, 0, 3, 3] = 0.6
    cms[0, 0, 3, 4] = 0.9
    assert _peak_list(cms) == [(0, 0, 4, 3)]
    # ... and the same pair with the higher pixel 2 away from both is a peak.
    cms[0, 0, 3, 4] = 0.0
    cms[0, 0, 3, 5] = 0.9
    assert _peak_list(cms) == [(0, 0, 2.5, 3), (0, 0, 5, 3)]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs CUDA"
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("levels", [2, 3, 5, 9, 33])
def test_find_local_peaks_rough_matches_plateau_reference(device, dtype, levels):
    # Coarsely quantized random maps are full of flat tops of every shape.
    torch.manual_seed(levels)
    cms = (torch.rand(8, 2, 13, 17) * (levels - 1)).round() / (levels - 1)
    cms = cms.to(device, dtype)
    assert sorted(_peak_list(cms)) == _reference_peaks(cms)
    _points, vals, samples, channels = find_local_peaks_rough(cms)
    assert vals.dtype == dtype and samples.dtype == channels.dtype == torch.int32


def test_find_local_peaks_rough_ties_at_edges():
    cms = torch.zeros(1, 1, 6, 6)
    cms[0, 0, 0, 0] = 0.9  # a corner: off-image neighbors don't count
    cms[0, 0, 5, 4] = cms[0, 0, 5, 5] = 0.6  # tied on the last row
    cms[0, 0, 2, 3] = 0.2  # not above the threshold
    assert _peak_list(cms) == [(0, 0, 0, 0), (0, 0, 4.5, 5)]
    points, vals, samples, channels = find_local_peaks_rough(torch.zeros(1, 1, 6, 6))
    assert points.shape == (0, 2) and vals.shape == (0,)
    assert samples.dtype == channels.dtype == torch.int32


def test_find_local_peaks_rough_unchanged_without_ties():
    # Random fp32 maps have no two equal neighbors: the same peaks as the
    # previous strict rule.
    torch.manual_seed(0)
    cms = torch.rand(4, 3, 40, 50)
    mask = _strict_peak_mask(cms)
    assert mask.sum() > 100
    subs = torch.stack(torch.where(mask.permute(0, 2, 3, 1)), axis=-1)
    points, vals, samples, channels = find_local_peaks_rough(cms)
    assert torch.equal(points, subs[:, [2, 1]].to(torch.float32))
    assert torch.equal(vals, cms[subs[:, 0], subs[:, 3], subs[:, 1], subs[:, 2]])
    assert torch.equal(samples, subs[:, 0].to(torch.int32))
    assert torch.equal(channels, subs[:, 3].to(torch.int32))


def test_find_local_peaks_integral_refinement_of_flat_top():
    # Refinement crops around the pixel a fractional flat-top peak floors to
    # and adds its offset to that pixel, not to the fractional peak: the
    # refined point is between that pixel and the Gaussian's center (the 5x5
    # patch pulls it toward the pixel), not past the center.
    for cx, cy in [(10.5, 8), (10, 8.5), (10.5, 8.5)]:
        cms = _gaussian(20, 24, cx, cy)[None, None]
        rough, *_ = find_local_peaks_rough(cms)
        assert rough.tolist() == [[cx, cy]]
        (x, y), *_ = find_local_peaks(cms, refinement="integral")[0].tolist()
        assert 10 <= x <= cx and 8 <= y <= cy
        assert (x > 10) == (cx > 10) and (y > 8) == (cy > 8)


def test_find_local_peaks_rough_fp16_ties():
    """In fp16, the top pixels of a Gaussian sometimes round to one value.

    Every peak the strict rule finds is still found, and the extra ones are
    the flat tops.
    """
    torch.manual_seed(1)
    height, width = 64, 80
    # 200 maps of 5 Gaussians each, at random positions and heights.
    cx = torch.rand(200, 5, 1, 1) * (width - 10.0) + 5
    cy = torch.rand(200, 5, 1, 1) * (height - 10.0) + 5
    amp = 0.6 + 0.4 * torch.rand(200, 5, 1, 1)
    cms = (amp * _gaussian(height, width, cx, cy)).amax(1, keepdim=True).half()

    found = set(_peak_list(cms))
    strict = {
        (int(s), int(c), int(x), int(y))
        for s, y, x, c in torch.stack(
            torch.where(_strict_peak_mask(cms).permute(0, 2, 3, 1)), axis=-1
        ).tolist()
    }
    assert strict <= found
    assert found - strict  # fp16 ties do happen
    assert sorted(found) == _reference_peaks(cms)
