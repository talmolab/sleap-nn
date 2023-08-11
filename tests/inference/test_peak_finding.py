import torch

from sleap_nn.inference.peak_finding import (
    crop_bboxes,
    integral_regression,
    find_global_peaks,
    find_global_peaks_rough,
)


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

    gt_dx_hat = torch.Tensor(
        [
            [0.24976766109466553],
            [-0.06099589914083481],
            [-0.216335266828537],
            [-0.12479443103075027],
            [0.12494532763957977],
            [0.28015944361686707],
            [0.03200167417526245],
            [0.21784470975399017],
            [0.3711766004562378],
            [-0.37117865681648254],
            [0.32524189352989197],
            [-0.18590612709522247],
            [0.06249351054430008],
        ]
    )

    gt_dy_hat = torch.Tensor(
        [
            [-0.1858985275030136],
            [0.031994160264730453],
            [-0.18588940799236298],
            [0.3141670227050781],
            [0.3423368036746979],
            [-0.3090454936027527],
            [0.43461763858795166],
            [-0.3408771753311157],
            [-0.155443474650383],
            [-0.28016677498817444],
            [0.32524189352989197],
            [-2.0254956325516105e-06],
            [-0.37117743492126465],
        ]
    )

    assert dx_hat.shape == dy_hat.shape == (13, 1)
    assert dx_hat.dtype == dy_hat.dtype == torch.float32
    assert torch.equal(gt_dx_hat, dx_hat)
    assert torch.equal(gt_dy_hat, dy_hat)


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
