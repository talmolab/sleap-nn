import torch

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

    (peak_points, peak_vals, peak_sample_inds, peak_channel_inds) = find_local_peaks(
        cms
    )

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

    (peak_points, peak_vals, peak_sample_inds, peak_channel_inds) = find_local_peaks(
        cms, refinement="invalid_input"
    )

    assert peak_points.shape == (13, 2)
    assert peak_vals.shape == peak_sample_inds.shape == peak_channel_inds.shape == (13,)
    assert torch.equal(gt_peak_vals, peak_vals)
    assert torch.equal(gt_peak_points, peak_points)
    assert torch.equal(gt_peak_sample_inds, peak_sample_inds)
    assert torch.equal(gt_peak_channel_inds, peak_channel_inds)

    (peak_points, peak_vals, peak_sample_inds, peak_channel_inds) = find_local_peaks(
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
