"""Tests for sub-pixel global peak refinement (DARK, quarter offset) and flip fusion."""

import pytest
import torch

from sleap_nn.data.confidence_maps import make_confmaps
from sleap_nn.data.utils import make_grid_vectors
from sleap_nn.inference.ops.peaks import (
    dark_refinement,
    find_global_peaks,
    find_global_peaks_rough,
    flip_idx_from_symmetries,
    fuse_flip_confmaps,
    resolve_refinement,
)

# sleap-nn renders a target sampled at input pixel ``i * stride`` with an input-px
# std of ``sigma * stride`` (``generate_confmaps``), so SIGMA is in confmap pixels.
STRIDE, SIGMA, H_IN, W_IN = 4, 2.0, 256, 192


def _render(points, stride=STRIDE, sigma=SIGMA, h=H_IN, w=W_IN):
    """Training targets for ``(N, 2)`` input-px points -> ``(1, N, h/s, w/s)``."""
    xv, yv = make_grid_vectors(h, w, stride)
    return make_confmaps(points[None], xv, yv, sigma * stride)


def _points(n=200, seed=0, margin=40.0):
    g = torch.Generator().manual_seed(seed)
    return torch.stack(
        [
            margin + torch.rand(n, generator=g) * (W_IN - 2 * margin),
            margin + torch.rand(n, generator=g) * (H_IN - 2 * margin),
        ],
        dim=-1,
    )


def _err(cms, pts, **kw):
    """Per-point decode error in input pixels."""
    peaks, _ = find_global_peaks(cms, threshold=0.0, **kw)
    return (peaks[0] * STRIDE - pts).norm(dim=-1)


@pytest.mark.parametrize("blur", [0.0, 1.0, SIGMA])
def test_dark_is_exact_on_gaussian_targets(blur):
    pts = _points()
    err = _err(_render(pts), pts, refinement="dark", dark_blur_sigma=blur)
    assert err.max() < 1e-2


def test_refinement_ordering_on_clean_targets():
    """dark < local < integral(5x5) < argmax; integral truncates a sigma=2 blob."""
    pts = _points()
    cms = _render(pts)
    e_argmax = _err(cms, pts).mean()
    e_int = _err(cms, pts, refinement="integral").mean()
    e_local = _err(cms, pts, refinement="local").mean()
    e_dark = _err(cms, pts, refinement="dark", dark_blur_sigma=SIGMA).mean()
    assert e_dark < 1e-3 < e_local < e_int < e_argmax


def test_dark_local_blur_matches_full_map_blur():
    """Blurring a patch around the peak equals blurring the whole map."""
    g = torch.Generator().manual_seed(1)
    pts = _points(20, seed=1)
    cms = _render(pts) + 0.05 * torch.rand(_render(pts).shape, generator=g)
    rough, _ = find_global_peaks_rough(cms, threshold=0.0)

    # Reference: blur the whole map (replicate padding), then Newton step w/o blur.
    radius = int(round(3 * SIGMA))
    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / SIGMA) ** 2)
    k = k / k.sum()
    s, c, h, w = cms.shape
    x = torch.nn.functional.pad(
        cms.reshape(s * c, 1, h, w), (radius,) * 4, mode="replicate"
    )
    x = torch.nn.functional.conv2d(x, k.view(1, 1, 1, -1))
    x = torch.nn.functional.conv2d(x, k.view(1, 1, -1, 1)).reshape(s, c, h, w)
    expected = dark_refinement(x, rough, blur_sigma=0.0)

    actual = dark_refinement(cms, rough, blur_sigma=SIGMA)
    assert torch.allclose(actual, expected, atol=1e-4)


def test_dark_half_precision_and_flat_top():
    """fp16 maps decode in fp32; a 2x2 flat top resolves to its center."""
    pts = _points(50)
    err = _err(_render(pts).half(), pts, refinement="dark", dark_blur_sigma=SIGMA)
    assert err.max() < 0.05

    # A blob centered between pixels: its top 2x2 pixels are equal, and the rough
    # (argmax) peak is the top-left one.
    xv, yv = make_grid_vectors(16, 16, 1)
    cms = make_confmaps(torch.tensor([[[5.5, 7.5]]]), xv, yv, 2.0)
    rough, _ = find_global_peaks_rough(cms, threshold=0.0)
    assert rough[0, 0].tolist() == [5.0, 7.0]
    peaks, _ = find_global_peaks(
        cms, threshold=0.0, refinement="dark", dark_blur_sigma=1.0
    )
    assert torch.allclose(peaks[0, 0], torch.tensor([5.5, 7.5]), atol=1e-3)


def test_local_refinement_is_quarter_step():
    pts = torch.tensor([[30.6 * STRIDE, 30.2 * STRIDE]])
    peaks, _ = find_global_peaks(_render(pts), threshold=0.0, refinement="local")
    assert torch.allclose(peaks[0, 0], torch.tensor([31.0 - 0.25, 30.25]))


@pytest.mark.parametrize("refinement", ["local", "dark"])
def test_refinement_keeps_nan_for_missing_peaks(refinement):
    cms = _render(torch.tensor([[100.0, 100.0], [100.0, 100.0]]))
    cms[0, 1] = 0.0
    peaks, vals = find_global_peaks(cms, threshold=0.2, refinement=refinement)
    assert torch.isnan(peaks[0, 1]).all()
    assert torch.isfinite(peaks[0, 0]).all()
    assert vals[0, 1] == 0


def test_dark_at_border_is_finite_and_bounded():
    cms = _render(torch.tensor([[0.0, 0.0], [W_IN - 1.0, H_IN - 1.0]]))
    rough, _ = find_global_peaks_rough(cms, threshold=0.0)
    peaks, _ = find_global_peaks(cms, threshold=0.0, refinement="dark")
    assert torch.isfinite(peaks).all()
    assert ((peaks - rough).abs() <= 1.0).all()


def test_find_global_peaks_refinement_names():
    cms = _render(torch.tensor([[101.0, 99.0]]))
    rough, _ = find_global_peaks(cms, threshold=0.0)
    none, _ = find_global_peaks(cms, threshold=0.0, refinement="none")
    assert torch.equal(rough, none)
    for bad in ("auto", "Dark", "bogus"):
        with pytest.raises(ValueError, match="Unknown peak refinement"):
            find_global_peaks(cms, threshold=0.0, refinement=bad)


@pytest.mark.parametrize(
    "requested, kwargs, expected",
    [
        ("auto", {}, "integral"),
        ("auto", {"auto": "dark"}, "dark"),
        ("none", {"auto": "dark"}, None),
        (None, {}, None),
        ("dark", {}, "dark"),
        ("local", {}, "local"),
        ("integral", {"auto": "dark"}, "integral"),
        # Centroid stage: dark/local fall back to integral.
        ("dark", {"peaks": "centroid"}, "integral"),
        ("local", {"peaks": "centroid"}, "integral"),
        ("auto", {"peaks": "centroid"}, "integral"),
        ("none", {"peaks": "centroid"}, None),
        # Bottom-up keeps integral / none.
        ("auto", {"peaks": "local"}, "integral"),
        ("integral", {"peaks": "local"}, "integral"),
    ],
)
def test_resolve_refinement(requested, kwargs, expected):
    assert resolve_refinement(requested, **kwargs) == expected


@pytest.mark.parametrize("requested", ["dark", "local"])
def test_resolve_refinement_rejects_global_only_for_bottomup(requested):
    with pytest.raises(ValueError, match="bottom-up"):
        resolve_refinement(requested, peaks="local")


def test_resolve_refinement_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown peak refinement"):
        resolve_refinement("subpixel")


def _flip_points(pts, w=W_IN):
    out = pts.clone()
    out[:, 0] = (w - 1) - out[:, 0]
    return out


def test_flip_fusion_shift_correction():
    pts = _points()
    cms, cms_f = _render(pts), _render(_flip_points(pts))
    fused = fuse_flip_confmaps(cms, cms_f, output_stride=STRIDE)
    err = _err(fused, pts, refinement="dark", dark_blur_sigma=SIGMA)
    assert err.max() < 0.05

    # Without the shift (stride 1 assumed) the decoded x is biased by
    # -(stride - 1) / 2 input pixels.
    unshifted = 0.5 * (cms + torch.flip(cms_f, dims=[-1]))
    peaks, _ = find_global_peaks(unshifted, threshold=0.0, refinement="dark")
    bias = (peaks[0, :, 0] * STRIDE - pts[:, 0]).mean()
    assert abs(bias + (STRIDE - 1) / 2) < 0.05


def test_flip_fusion_stride_one_needs_no_shift():
    xv, yv = make_grid_vectors(64, 48, 1)
    p = torch.tensor([[20.3, 30.7]])
    c = make_confmaps(p[None], xv, yv, 2.0)
    c_f = make_confmaps(_flip_points(p, w=48)[None], xv, yv, 2.0)
    assert torch.allclose(fuse_flip_confmaps(c, c_f, output_stride=1), 0.5 * (c + c))
    peaks, _ = find_global_peaks(
        fuse_flip_confmaps(c, c_f, output_stride=1), threshold=0.0, refinement="dark"
    )
    assert torch.allclose(peaks[0, 0], p[0], atol=1e-3)


def test_flip_fusion_uses_input_width():
    """An input wider than ``Wc * stride`` shifts the mirrored grid further."""
    w_in, stride = 195, 4  # confmap width 48 covers x = 0..188; 6 px of slack
    pts = _points(50, margin=50.0)
    xv, yv = make_grid_vectors(H_IN, w_in, stride)
    xv = xv[: W_IN // stride]  # what a model with floor division would output
    cms = make_confmaps(pts[None], xv, yv, SIGMA * stride)
    cms_f = make_confmaps(_flip_points(pts, w=w_in)[None], xv, yv, SIGMA * stride)
    fused = fuse_flip_confmaps(cms, cms_f, output_stride=stride, input_width=w_in)
    err = _err(fused, pts, refinement="dark", dark_blur_sigma=SIGMA)
    assert err.max() < 0.1
    # The default (input_width = Wc * stride) is wrong for this input.
    wrong = fuse_flip_confmaps(cms, cms_f, output_stride=stride)
    assert _err(wrong, pts, refinement="dark", dark_blur_sigma=SIGMA).mean() > 1.0


def test_flip_fusion_swaps_channels():
    """Left/right channels predicted on the mirror are swapped back."""
    pts = torch.tensor([[60.3, 100.0], [130.6, 120.0]])  # node 0 left, node 1 right
    cms = _render(pts)
    # On the mirrored image the network reports the (mirrored) left node in
    # channel 0 even though it now sits where the right node was.
    cms_f = _render(_flip_points(pts)[[1, 0]])
    fused = fuse_flip_confmaps(
        cms, cms_f, flip_idx=torch.tensor([1, 0]), output_stride=STRIDE
    )
    err = _err(fused, pts, refinement="dark", dark_blur_sigma=SIGMA)
    assert err.max() < 0.05


def test_flip_idx_from_symmetries():
    assert flip_idx_from_symmetries(5, [(1, 3)]).tolist() == [0, 3, 2, 1, 4]
    assert flip_idx_from_symmetries(5, [(1, 3), (3, 1)]).tolist() == [0, 3, 2, 1, 4]
    assert flip_idx_from_symmetries(3, []).tolist() == [0, 1, 2]
    assert flip_idx_from_symmetries(3, None).tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="at most one symmetric partner"):
        flip_idx_from_symmetries(4, [(0, 1), (0, 2)])
    with pytest.raises(ValueError, match="out of range"):
        flip_idx_from_symmetries(2, [(0, 2)])
