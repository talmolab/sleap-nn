import pytest
import torch

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

    rough_peaks, peak_vals = find_global_peaks(
        cms, refinement="invalid_input", threshold=0.2
    )

    assert rough_peaks.shape == (1, 13, 2)
    assert peak_vals.shape == (1, 13)
    assert rough_peaks.dtype == peak_vals.dtype == torch.float32
    assert torch.equal(gt_rough_peaks, rough_peaks)
    assert torch.equal(gt_peak_vals, peak_vals)

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
        (int(s), int(c), int(x), int(y))
        for (x, y), s, c in zip(points.tolist(), samples, channels)
    ]


def _strict_peak_mask(cms, threshold=0.2):
    """The previous rule: strictly greater than all 8 neighbors."""
    height, width = cms.shape[-2:]
    max_img = morphological_dilation(cms.reshape(-1, 1, height, width), None)
    return (cms > max_img.reshape(cms.shape)) & (cms > threshold)


def test_morphological_dilation_split_matches_dilation():
    torch.manual_seed(0)
    img = torch.randn(3, 1, 9, 11)
    before, after = morphological_dilation_split(img)
    assert before.shape == after.shape == img.shape
    assert torch.equal(torch.maximum(before, after), morphological_dilation(img, None))
    # The first pixel has no earlier neighbors; the last has no later ones.
    assert torch.all(before[..., 0, 0] == float("-inf"))
    assert torch.all(after[..., -1, -1] == float("-inf"))


@pytest.mark.parametrize("dx, dy", [(1, 0), (0, 1), (1, 1), (-1, 1)])
def test_find_local_peaks_rough_two_equal_top_pixels(dx, dy):
    """Two equal top pixels (E, S, SE or SW of each other) give one peak.

    The previous strict rule found none (#781). The kept pixel is the first of
    the two in raster order.
    """
    cms = _gaussian(20, 24, 10, 8)[None, None] * 0.5
    a, b = (10, 8), (10 + dx, 8 + dy)
    for x, y in (a, b):
        cms[0, 0, y, x] = 0.9
    first = min(a, b, key=lambda p: (p[1], p[0]))
    assert not _strict_peak_mask(cms).any()
    assert _peak_list(cms) == [(0, 0) + first]


def test_find_local_peaks_rough_flat_top():
    # A Gaussian centered between two pixels has two equal top pixels; one
    # centered between four has a 2x2 flat top. Each gives one peak.
    cms = _gaussian(20, 24, 10.5, 8)[None, None]
    assert cms[0, 0, 8, 10] == cms[0, 0, 8, 11]
    assert _peak_list(cms) == [(0, 0, 10, 8)]
    cms = _gaussian(20, 24, 10.5, 8.5)[None, None]
    assert _peak_list(cms) == [(0, 0, 10, 8)]


def test_find_local_peaks_rough_ties_at_edges():
    cms = torch.zeros(1, 1, 6, 6)
    cms[0, 0, 0, 0] = 0.9  # a corner: off-image neighbors don't count
    cms[0, 0, 5, 4] = cms[0, 0, 5, 5] = 0.6  # tied on the last row
    cms[0, 0, 2, 3] = 0.2  # not above the threshold
    assert _peak_list(cms) == [(0, 0, 0, 0), (0, 0, 4, 5)]
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


def test_find_local_peaks_rough_fp16_ties():
    """In fp16, the top two pixels of a Gaussian sometimes round to one value.

    Every peak the strict rule finds is still found; the extra ones are
    exactly at tied tops, one per tie.
    """
    torch.manual_seed(1)
    height, width = 64, 80
    cms = torch.zeros(200, 1, height, width)
    for i in range(cms.shape[0]):
        for _ in range(5):
            cx, cy = torch.rand(2) * torch.tensor([width - 10.0, height - 10.0]) + 5
            peak = _gaussian(height, width, float(cx), float(cy))
            peak = peak * (0.6 + 0.4 * float(torch.rand(1)))
            cms[i, 0] = torch.maximum(cms[i, 0], peak)
    cms = cms.half()

    found = set(_peak_list(cms))
    strict = {
        (int(s), int(c), int(x), int(y))
        for s, y, x, c in torch.stack(
            torch.where(_strict_peak_mask(cms).permute(0, 2, 3, 1)), axis=-1
        ).tolist()
    }
    assert strict <= found
    extra = found - strict
    assert extra  # fp16 ties do happen
    for s, c, x, y in extra:
        top = cms[s, c, y, x]
        around = cms[s, c, max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2]
        assert (around == top).sum() >= 2 and (around <= top).all()
