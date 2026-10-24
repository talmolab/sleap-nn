"""Flat-top (tied) peaks in the export wrappers' peak finding (#781).

The export wrappers have their own fixed-shape peak finding
(``BaseExportWrapper._local_peaks``). These tests check that it gives the same
peaks as the PyTorch path (``find_local_peaks_rough``), in eager mode,
through onnxruntime and (on a GPU) in an fp16 TensorRT engine, on maps with ties (fp32 and fp16-quantized), and that it
is bit-identical to the old strict rule when there are no ties.
"""

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sleap_nn.export.wrappers import BottomUpONNXWrapper, CentroidONNXWrapper
from sleap_nn.export.wrappers.base import BaseExportWrapper
from sleap_nn.inference.ops.peaks import (
    find_local_peaks_rough,
    morphological_dilation_split,
)

from .conftest import requires_onnxruntime, requires_tensorrt

THRESHOLD = 0.2

# (name, top pixels as (row, col) offsets, number of peaks the top gives)
FLAT_TOPS = [
    ("pair E", [(0, 0), (0, 1)], 1),
    ("pair S", [(0, 0), (1, 0)], 1),
    ("pair SE", [(0, 0), (1, 1)], 1),
    ("pair SW", [(0, 1), (1, 0)], 1),
    ("L", [(0, 0), (0, 1), (1, 0)], 1),
    ("L flipped", [(0, 1), (1, 0), (1, 1)], 1),
    ("L down", [(0, 0), (1, 0), (1, 1)], 1),
    ("2x2", [(0, 0), (0, 1), (1, 0), (1, 1)], 1),
    ("row of 3", [(0, 0), (0, 1), (0, 2)], 0),
    ("diagonal of 3", [(0, 0), (1, 1), (2, 2)], 0),
    ("2x3", [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], 0),
]


def _flat_top_map(top, value=0.9, origin=(5, 6), size=16):
    """A ``(1, 1, size, size)`` map with a flat top over a lower slope."""
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    r0, c0 = origin
    cms = 0.5 * torch.exp(-((yy - r0) ** 2 + (xx - c0) ** 2) / 8)
    for dr, dc in top:
        cms[r0 + dr, c0 + dc] = value
    return cms[None, None]


def _flat_top_batch():
    """``(1, n_tops, 16, 16)``: one flat top per channel, plus a shoulder."""
    maps = [_flat_top_map(top) for _, top, _ in FLAT_TOPS]
    # A tied pair next to a higher pixel: no peak there, only the higher one.
    shoulder = _flat_top_map([(0, 0), (0, 1)])
    shoulder[0, 0, 5, 8] = 0.95
    maps.append(shoulder)
    return torch.cat(maps, dim=1)


def _gaussian_maps(n_samples, n_channels, size=48, n_peaks=5, sigma=2.5, seed=0):
    """Random sums of Gaussians, centered off the pixel grid."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    centers = torch.rand(n_samples, n_channels, n_peaks, 2, generator=g) * size
    d2 = (xx - centers[..., 0, None, None]) ** 2 + (
        yy - centers[..., 1, None, None]
    ) ** 2
    return torch.exp(-d2 / (2 * sigma**2)).amax(dim=2)


def _reference_peaks(cms):
    """Sorted ``(sample, channel, x, y, value)`` from the PyTorch path."""
    pts, vals, s, c = find_local_peaks_rough(cms, threshold=THRESHOLD)
    return sorted(
        (int(si), int(ci), float(p[0]), float(p[1]), float(v))
        for p, v, si, ci in zip(pts, vals, s, c)
    )


def _wrapper_peaks(peaks, values, valid):
    """Sorted valid ``(sample, channel, x, y, value)`` from per-node top-K."""
    peaks, values, valid = (np.asarray(a) for a in (peaks, values, valid))
    return sorted(
        (int(s), int(c), float(peaks[s, c, j, 0]), float(peaks[s, c, j, 1]))
        + (float(values[s, c, j]),)
        for s, c, j in zip(*np.nonzero(valid))
    )


def _strict_topk_peaks(confmaps, k, peak_threshold):
    """The export wrappers' peak finding before #781 (strict ``>`` rule)."""
    batch_size, n_nodes, height, width = confmaps.shape
    p = F.pad(confmaps, [1, 1, 1, 1], mode="constant", value=float("-inf"))
    shifts = [
        p[:, :, 1 + dr : 1 + dr + height, 1 + dc : 1 + dc + width]
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr or dc
    ]
    neighbor_max = torch.stack(shifts).amax(0)
    is_peak = (confmaps > neighbor_max) & (confmaps > peak_threshold)
    flat = confmaps.reshape(batch_size, n_nodes, height * width)
    masked = torch.where(
        is_peak.reshape(batch_size, n_nodes, height * width),
        flat,
        torch.full_like(flat, -1e9),
    )
    values, indices = torch.topk(masked, k=k, dim=2)
    peaks = torch.stack([(indices % width).float(), (indices // width).float()], -1)
    return peaks, values, values > peak_threshold


class _PeaksModule(nn.Module):
    """Exports the per-node peak finding alone, on float confmaps."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, confmaps):
        peaks, values, valid = BaseExportWrapper._find_topk_peaks_per_node(
            confmaps, self.k, THRESHOLD
        )
        return {"peaks": peaks, "values": values, "valid": valid}


class _FixedMaps(nn.Module):
    """A stand-in backbone that returns fixed maps (kept input-dependent)."""

    def __init__(self, **maps):
        super().__init__()
        self.names = list(maps)
        for name, value in maps.items():
            self.register_buffer(name, value)

    def forward(self, x):
        zero = x.mean() * 0
        return {name: getattr(self, name) + zero for name in self.names}


def _run_onnx(module, inputs, tmp_path, input_dtype=torch.float32, trace_shape=None):
    """Export ``module`` (traced on random input) and run it in onnxruntime."""
    import onnxruntime as ort

    from sleap_nn.export.exporters import export_to_onnx

    path = tmp_path / "model.onnx"
    if trace_shape is None:
        trace_shape = tuple(inputs.shape)
    export_to_onnx(module, path, input_shape=trace_shape, input_dtype=input_dtype)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    outputs = session.run(None, {session.get_inputs()[0].name: inputs.numpy()})
    return dict(zip(names, outputs))


class TestFlatTopsEager:
    """``BaseExportWrapper`` peak finding in PyTorch eager mode."""

    @pytest.mark.parametrize("name,top,n_expected", FLAT_TOPS)
    def test_flat_top_shapes(self, name, top, n_expected):
        """Each flat top gives one peak at its mean if it fits in a 2x2."""
        cms = _flat_top_map(top)
        peaks, values, valid = BaseExportWrapper._find_topk_peaks(cms, k=4)
        assert int(valid.sum()) == n_expected
        if n_expected:
            r0, c0 = 5, 6
            mean_y = r0 + sum(r for r, _ in top) / len(top)
            mean_x = c0 + sum(c for _, c in top) / len(top)
            got = peaks[0, 0]
            assert got[0].item() == pytest.approx(mean_x, abs=1e-6)
            assert got[1].item() == pytest.approx(mean_y, abs=1e-6)
            assert values[0, 0].item() == pytest.approx(0.9)

    def test_shoulder_gives_no_peak(self):
        """A tied pair next to a higher pixel is not a peak."""
        cms = _flat_top_map([(0, 0), (0, 1)])
        cms[0, 0, 5, 8] = 0.95
        peaks, values, valid = BaseExportWrapper._find_topk_peaks(cms, k=4)
        assert int(valid.sum()) == 1
        assert peaks[0, 0].tolist() == [8.0, 5.0]

    def test_matches_pytorch_path_on_flat_tops(self):
        """Every flat top (and the shoulder) matches ``find_local_peaks_rough``."""
        cms = _flat_top_batch()
        out = BaseExportWrapper._find_topk_peaks_per_node(cms, 8, THRESHOLD)
        assert _wrapper_peaks(*out) == _reference_peaks(cms)

    @pytest.mark.parametrize("levels", [2, 3, 5, 9, 17, 33])
    def test_matches_pytorch_path_on_quantized_maps(self, levels):
        """Coarsely quantized random maps (many flat tops of all shapes)."""
        g = torch.Generator().manual_seed(levels)
        cms = (torch.rand(3, 4, 20, 21, generator=g) * levels).floor() / levels
        out = BaseExportWrapper._find_topk_peaks_per_node(cms, 20 * 21, THRESHOLD)
        assert _wrapper_peaks(*out) == _reference_peaks(cms)

    def test_matches_pytorch_path_on_fp16_maps(self):
        """fp16 Gaussians: same peaks as the PyTorch path run in fp16."""
        cms16 = _gaussian_maps(8, 4).half()
        ref = _reference_peaks(cms16)
        # The rounding must create flat tops for this test to mean anything.
        assert any(x % 1 or y % 1 for _, _, x, y, _ in ref)
        out = BaseExportWrapper._find_topk_peaks_per_node(cms16.float(), 10, THRESHOLD)
        assert _wrapper_peaks(*out) == ref

    def test_untied_maps_bit_identical_to_strict_rule(self):
        """Without ties, outputs (incl. invalid slots) equal the old rule's."""
        g = torch.Generator().manual_seed(0)
        cms = torch.rand(4, 3, 32, 33, generator=g)
        for new, old in zip(
            BaseExportWrapper._find_topk_peaks_per_node(cms, 50, THRESHOLD),
            _strict_topk_peaks(cms, 50, THRESHOLD),
        ):
            assert new.dtype == old.dtype
            assert torch.equal(new, old)
        single = cms[:, :1]
        for new, old in zip(
            BaseExportWrapper._find_topk_peaks(single, 50, THRESHOLD),
            _strict_topk_peaks(single, 50, THRESHOLD),
        ):
            assert torch.equal(new, old[:, 0])

        gauss = _gaussian_maps(4, 3)  # fp32: no ties
        for new, old in zip(
            BaseExportWrapper._find_topk_peaks_per_node(gauss, 10, THRESHOLD),
            _strict_topk_peaks(gauss, 10, THRESHOLD),
        ):
            assert torch.equal(new, old)


@requires_onnxruntime
class TestFlatTopsONNX:
    """The exported graph (traced on tie-free input) in onnxruntime."""

    def test_flat_tops_fp32(self, tmp_path):
        """fp32 flat tops: one peak each, at the PyTorch path's point."""
        cms = _flat_top_batch()
        out = _run_onnx(_PeaksModule(k=8), cms, tmp_path)
        assert _wrapper_peaks(out["peaks"], out["values"], out["valid"]) == (
            _reference_peaks(cms)
        )

    def test_quantized_maps_fp32(self, tmp_path):
        """Coarsely quantized random maps match the PyTorch path."""
        g = torch.Generator().manual_seed(1)
        cms = (torch.rand(3, 4, 20, 21, generator=g) * 5).floor() / 5
        out = _run_onnx(_PeaksModule(k=20 * 21), cms, tmp_path)
        assert _wrapper_peaks(out["peaks"], out["values"], out["valid"]) == (
            _reference_peaks(cms)
        )

    def test_dynamic_shape(self, tmp_path):
        """A graph traced at one batch/H/W gives the same peaks at another."""
        g = torch.Generator().manual_seed(2)
        cms = (torch.rand(3, 4, 20, 21, generator=g) * 5).floor() / 5
        out = _run_onnx(_PeaksModule(k=50), cms, tmp_path, trace_shape=(1, 4, 24, 17))
        assert _wrapper_peaks(out["peaks"], out["values"], out["valid"]) == (
            _reference_peaks(cms)
        )

    def test_fp16_maps(self, tmp_path):
        """fp16-rounded Gaussians match the PyTorch path run in fp16."""
        cms16 = _gaussian_maps(8, 4).half()
        ref = _reference_peaks(cms16)
        assert any(x % 1 or y % 1 for _, _, x, y, _ in ref)
        out = _run_onnx(_PeaksModule(k=10), cms16.float(), tmp_path)
        assert _wrapper_peaks(out["peaks"], out["values"], out["valid"]) == ref

    def test_untied_matches_eager(self, tmp_path):
        """Without ties, onnxruntime equals the old strict rule exactly.

        Invalid slots hold filler pixels whose order ``TopK`` leaves to the
        backend (as before this change), so only valid points are compared.
        """
        cms = _gaussian_maps(2, 3, seed=3)
        out = _run_onnx(_PeaksModule(k=10), cms, tmp_path)
        peaks, values, valid = _strict_topk_peaks(cms, 10, THRESHOLD)
        np.testing.assert_array_equal(out["valid"], valid.numpy())
        np.testing.assert_array_equal(out["values"], values.numpy())
        np.testing.assert_array_equal(out["peaks"][out["valid"]], peaks[valid].numpy())

    @pytest.mark.parametrize("name", ["pair E", "L", "2x2"])
    def test_centroid_wrapper(self, name, tmp_path):
        """An exported centroid model finds a flat top once, at its mean."""
        top = {n: t for n, t, _ in FLAT_TOPS}[name]
        cms = _flat_top_map(top)
        stride = 2
        wrapper = CentroidONNXWrapper(
            _FixedMaps(CentroidConfmapsHead=cms),
            max_instances=4,
            output_stride=stride,
            peak_threshold=THRESHOLD,
        )
        image = torch.zeros(1, 1, 32, 32, dtype=torch.uint8)
        out = _run_onnx(wrapper, image, tmp_path, input_dtype=torch.uint8)

        ((_, _, x, y, v),) = _reference_peaks(cms)
        valid = out["instance_valid"][0]
        assert valid.sum() == 1
        expected = np.array([x, y], dtype=np.float32) * stride
        np.testing.assert_array_equal(out["centroids"][0][valid][0], expected)
        assert out["centroid_vals"][0][valid][0] == np.float32(v)

    def test_bottomup_wrapper(self, tmp_path):
        """An exported bottom-up model finds each flat top once, at its mean."""
        cms = _flat_top_batch()
        n_nodes = cms.shape[1]
        stride = 4
        pafs = torch.zeros(1, 2 * (n_nodes - 1), 8, 8)
        wrapper = BottomUpONNXWrapper(
            _FixedMaps(MultiInstanceConfmapsHead=cms, PartAffinityFieldsHead=pafs),
            skeleton_edges=[(i, i + 1) for i in range(n_nodes - 1)],
            n_nodes=n_nodes,
            max_peaks_per_node=4,
            cms_output_stride=stride,
            pafs_output_stride=8,
            peak_threshold=THRESHOLD,
        )
        image = torch.zeros(1, 1, 64, 64, dtype=torch.uint8)
        out = _run_onnx(wrapper, image, tmp_path, input_dtype=torch.uint8)

        got = _wrapper_peaks(out["peaks"] / stride, out["peak_vals"], out["peak_mask"])
        assert got == _reference_peaks(cms)
        n_per_node = out["peak_mask"][0].sum(-1).tolist()
        assert n_per_node == [n for _, _, n in FLAT_TOPS] + [1]


class _BlurBackbone(nn.Module):
    """A blur "network" wide enough that TensorRT picks fp16 kernels for it.

    Its confidence maps are smooth blobs; computed in fp16, their top pixels
    often round to the same value.
    """

    def __init__(self, channels=32):
        super().__init__()
        r = torch.arange(-6, 7, dtype=torch.float32)
        k = torch.exp(-(r**2) / (2 * 3.0**2))
        k2 = k[:, None] * k[None, :]
        self.blur = nn.Conv2d(1, channels, 13, padding=6, bias=False)
        self.blur.weight.data[:] = k2 / k2.sum()
        self.mix = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.mix.weight.data.zero_()
        idx = torch.arange(channels)
        self.mix.weight.data[idx, idx, 1, 1] = 1.0
        self.head = nn.Conv2d(channels, 1, 1, bias=False)
        self.head.weight.data[:] = 1.0 / channels

    def forward(self, x):
        return {"CentroidConfmapsHead": self.head(self.mix(self.blur(x)))}


def _blob_frames(n_frames, size=96, sigma=4.0, seed=1):
    """uint8 frames with 5 well-separated blobs at random sub-pixel centers."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    frames = []
    for _ in range(n_frames):
        cx = torch.tensor([15.0, 45, 75, 30, 60]) + torch.rand(5, generator=g) * 6
        cy = torch.tensor([20.0, 20, 20, 65, 65]) + torch.rand(5, generator=g) * 6
        d2 = (xx - cx[:, None, None]) ** 2 + (yy - cy[:, None, None]) ** 2
        img = torch.exp(-d2 / (2 * sigma**2)).amax(0) * 255
        frames.append(img.round().to(torch.uint8))
    return torch.stack(frames)[:, None]


@requires_tensorrt
def test_fp16_tensorrt_centroid_engine_finds_tied_peaks(tmp_path):
    """An fp16 TensorRT centroid engine finds every blob exactly once.

    With the old strict rule, this engine lost ~3% of the centroids (82 of
    2560 on an RTX A6000, TensorRT 10.15) to fp16 ties.
    """
    from sleap_nn.export.exporters.tensorrt_exporter import export_to_tensorrt
    from sleap_nn.inference.layers.backends.tensorrt_backend import (
        TensorRTBackend,
    )

    images = _blob_frames(256)
    # The fp16 maps have ties, so the test exercises the flat-top rule.
    with torch.no_grad():
        backbone = _BlurBackbone().cuda().half()
        cms16 = backbone(images.cuda().half() / 255)["CentroidConfmapsHead"]
    neighbor_max = torch.maximum(*morphological_dilation_split(cms16))
    strict = (cms16 > neighbor_max) & (cms16 > THRESHOLD)
    assert int(strict.sum()) < 5 * len(images)

    wrapper = CentroidONNXWrapper(
        _BlurBackbone(), max_instances=8, output_stride=1, peak_threshold=THRESHOLD
    ).cuda()
    shape = tuple(images.shape)
    engine = export_to_tensorrt(
        wrapper,
        tmp_path / "centroid.trt",
        input_shape=shape,
        precision="fp16",
        min_shape=shape,
        opt_shape=shape,
        max_shape=shape,
        verbose=False,
    )
    out = TensorRTBackend(str(engine))(images.cuda())
    assert out["instance_valid"].sum(1).tolist() == [5] * len(images)
