"""Exported graphs of models with an input scale resize as checkpoint inference does.

Checkpoint inference resizes each frame (or top-down crop) with ``resize_image``:
``torchvision.transforms.functional.resize`` (bilinear, antialiased) to
``(int(H * scale), int(W * scale))`` of that frame. The centroid, centered-instance,
bottom-up, multi-class and top-down wrappers used to resize without antialiasing, to
a size fixed when the graph was traced. Each graph is exported at one size and run
under ONNX Runtime at that size and at another one, against the same wrapper run in
PyTorch with every resize it makes replaced by ``resize_image`` at the stage's scale
(on the float image: the graph normalizes before resizing).
"""

import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from torch import nn

from sleap_nn.data.resizing import resize_image
from sleap_nn.export.wrappers import (
    BottomUpMultiClassONNXWrapper,
    BottomUpONNXWrapper,
    CenteredInstanceONNXWrapper,
    CentroidONNXWrapper,
    SingleInstanceONNXWrapper,
    TopDownMultiClassCombinedONNXWrapper,
    TopDownMultiClassONNXWrapper,
    TopDownONNXWrapper,
)

FRAME = (128, 160)  # traced frame size
OTHER_FRAME = (97, 203)  # odd sizes
CROP = (32, 32)  # traced crop size of the crop-input graphs
OTHER_CROP = (47, 47)
BUNDLE_CROP = (40, 36)  # crop size of the bundles


class _Heads(nn.Module):
    """Fixed linear heads: 1x1 convs of the image, average-pooled to each stride.

    ``heads`` maps a head name to ``(channels, stride)``; a stride of ``None`` gives
    a vector (the spatial mean), as a class-vector head.
    """

    def __init__(self, heads, seed=0):
        super().__init__()
        self.heads = dict(heads)
        g = torch.Generator().manual_seed(seed)
        self.convs = nn.ModuleList()
        for c, _ in self.heads.values():
            conv = nn.Conv2d(1, c, 1)
            with torch.no_grad():
                conv.weight.copy_(0.5 + torch.rand(c, 1, 1, 1, generator=g))
                conv.bias.zero_()
            self.convs.append(conv)

    def forward(self, x):
        x = x.mean(1, keepdim=True)
        out = {}
        for (name, (_, stride)), conv in zip(self.heads.items(), self.convs):
            y = conv(x)
            out[name] = y.mean((-2, -1)) if stride is None else F.avg_pool2d(y, stride)
        return out


def _blobs(hw, n=2, seed=0):
    """uint8 ``(n, 1, H, W)`` frames with three Gaussian blobs each."""
    rng = np.random.default_rng(seed)
    h, w = hw
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    frames = []
    for _ in range(n):
        img = np.zeros((h, w))
        for _ in range(3):
            cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
            img += rng.uniform(0.5, 1.0) * np.exp(
                -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rng.uniform(3, 6) ** 2)
            )
        frames.append(np.clip(img * 230, 0, 255).astype(np.uint8))
    return torch.from_numpy(np.stack(frames)[:, None])


def _centroid_heads(seed=0):
    return _Heads({"CentroidConfmapsHead": (1, 2)}, seed)


def _instance_heads(seed=1):
    return _Heads({"CenteredInstanceConfmapsHead": (3, 2)}, seed)


def _instance_class_heads(seed=1):
    return _Heads(
        {"CenteredInstanceConfmapsHead": (3, 2), "ClassVectorsHead": (2, None)}, seed
    )


def _single_instance(s):
    return SingleInstanceONNXWrapper(
        _Heads({"SingleInstanceConfmapsHead": (3, 2)}),
        output_stride=2,
        input_scale=s,
        peak_threshold=0.05,
    )


def _centroid(s):
    return CentroidONNXWrapper(
        _centroid_heads(), max_instances=4, output_stride=2, input_scale=s
    )


def _centered_instance(s):
    return CenteredInstanceONNXWrapper(
        _instance_heads(), output_stride=2, input_scale=s, peak_threshold=0.05
    )


def _bottomup(s):
    return BottomUpONNXWrapper(
        _Heads({"MultiInstanceConfmapsHead": (2, 2), "PartAffinityFieldsHead": (2, 4)}),
        skeleton_edges=[(0, 1)],
        n_nodes=2,
        max_peaks_per_node=4,
        cms_output_stride=2,
        pafs_output_stride=4,
        input_scale=s,
    )


def _multiclass_bottomup(s):
    return BottomUpMultiClassONNXWrapper(
        _Heads({"MultiInstanceConfmapsHead": (2, 2), "ClassMapsHead": (2, 4)}),
        n_nodes=2,
        n_classes=2,
        max_peaks_per_node=4,
        cms_output_stride=2,
        class_maps_output_stride=4,
        input_scale=s,
    )


def _multiclass_centered_instance(s):
    return TopDownMultiClassONNXWrapper(
        _instance_class_heads(), output_stride=2, input_scale=s, peak_threshold=0.05
    )


def _topdown(sc, si):
    return TopDownONNXWrapper(
        _centroid_heads(),
        _instance_heads(),
        max_instances=4,
        crop_size=BUNDLE_CROP,
        centroid_output_stride=2,
        instance_output_stride=2,
        centroid_input_scale=sc,
        instance_input_scale=si,
        n_nodes=3,
        instance_peak_threshold=0.05,
    )


def _multiclass_topdown(sc, si):
    return TopDownMultiClassCombinedONNXWrapper(
        _centroid_heads(),
        _instance_class_heads(),
        max_instances=4,
        crop_size=BUNDLE_CROP,
        centroid_output_stride=2,
        instance_output_stride=2,
        centroid_input_scale=sc,
        instance_input_scale=si,
        n_nodes=3,
        n_classes=2,
        instance_peak_threshold=0.05,
    )


_CENTROID_KEYS = ("centroids", "centroid_vals", "instance_valid")

#: id -> (wrapper factory, (frame-stage scale, crop-stage scale), takes crops,
#: keys compared at another size: None for all). The bundles' crops are cut with
#: the traced frame size (a separate limit of exported top-down graphs), so at
#: another frame size only their centroid stage is compared.
CASES = {}
for _s in (0.5, 0.75):
    CASES.update(
        {
            f"centroid-{_s}": (lambda s=_s: _centroid(s), (_s, None), False, None),
            f"centered_instance-{_s}": (
                lambda s=_s: _centered_instance(s),
                (None, _s),
                True,
                None,
            ),
            f"bottomup-{_s}": (lambda s=_s: _bottomup(s), (_s, None), False, None),
            f"multiclass_bottomup-{_s}": (
                lambda s=_s: _multiclass_bottomup(s),
                (_s, None),
                False,
                None,
            ),
            f"multiclass_centered_instance-{_s}": (
                lambda s=_s: _multiclass_centered_instance(s),
                (None, _s),
                True,
                None,
            ),
        }
    )
for _sc, _si in ((0.5, 0.5), (0.75, 0.6), (0.5, 1.0), (1.0, 0.5)):
    CASES[f"topdown-{_sc}-{_si}"] = (
        lambda a=_sc, b=_si: _topdown(a, b),
        (_sc, _si),
        False,
        _CENTROID_KEYS,
    )
    CASES[f"multiclass_topdown-{_sc}-{_si}"] = (
        lambda a=_sc, b=_si: _multiclass_topdown(a, b),
        (_sc, _si),
        False,
        _CENTROID_KEYS,
    )


class _CheckpointResizeF:
    """``torch.nn.functional`` with ``interpolate`` doing the checkpoint path's resize.

    Replaces the wrappers' module-level ``F`` before this change, whose only
    ``F.interpolate`` calls were the input-scale resizes.
    """

    def __getattr__(self, name):
        return getattr(F, name)

    @staticmethod
    def interpolate(image, size, **_kwargs):
        return tvf.resize(image, list(size), antialias=True)


def _reference(wrapper, image, scales, monkeypatch):
    """The wrapper in PyTorch, every resize replaced by the checkpoint path's.

    ``scales`` is ``(frame-stage scale, crop-stage scale)``: frames (and the
    crop-input wrappers' crops) go through ``antialiased_resize``, the bundles' crops
    through ``antialiased_resize_to``; each is replaced by ``resize_image`` at its
    stage's scale.
    """
    frame_scale, crop_scale = scales
    input_scale = frame_scale if frame_scale is not None else crop_scale

    def frame_resize(x, *_rule):
        return resize_image(x, input_scale)

    def fixed_resize(x, _in_hw, _out_hw):
        return resize_image(x, crop_scale)

    with monkeypatch.context() as m:
        mod = sys.modules[type(wrapper).__module__]
        m.setattr(mod, "antialiased_resize", frame_resize, raising=False)
        m.setattr(mod, "antialiased_resize_to", fixed_resize, raising=False)
        m.setattr(mod, "F", _CheckpointResizeF(), raising=False)
        with torch.no_grad():
            return {k: v.numpy() for k, v in wrapper(image.clone()).items()}


@pytest.mark.parametrize("case", list(CASES))
def test_graph_resizes_as_checkpoint_inference(case, tmp_path, monkeypatch):
    """ONNX Runtime output == checkpoint-resize reference, traced and other size."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from sleap_nn.export.exporters.onnx_exporter import export_to_onnx

    factory, scales, crops, other_keys = CASES[case]
    wrapper = factory().eval()
    traced, other = (CROP, OTHER_CROP) if crops else (FRAME, OTHER_FRAME)
    with torch.no_grad():
        out_names = list(wrapper(_blobs(traced)).keys())
    path = export_to_onnx(
        wrapper,
        tmp_path / "model.onnx",
        input_shape=(2, 1, *traced),
        input_names=["image"],
        output_names=out_names,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    for hw, keys in [(traced, out_names), (other, other_keys or out_names)]:
        image = _blobs(hw, seed=hw[0])
        got = dict(zip(out_names, session.run(None, {"image": image.numpy()})))
        want = _reference(wrapper, image, scales, monkeypatch)
        valid = want.get("instance_valid", want.get("peak_mask"))
        assert valid is None or valid.any(), "the test frames must give peaks"
        for key in keys:
            g, r = got[key], want[key]
            assert g.shape == r.shape, (hw, key)
            if r.dtype == bool:
                np.testing.assert_array_equal(g, r, err_msg=f"{hw} {key}")
                continue
            if valid is not None and r.shape[: valid.ndim] == valid.shape:
                # Empty slots hold arbitrary filler: compare the valid ones.
                g, r = g[valid], r[valid]
            # Float rounding only: coordinates are scaled by stride / scale.
            np.testing.assert_allclose(g, r, atol=1e-3, rtol=0, err_msg=f"{hw} {key}")


def _refuse(*_args, **_kwargs):
    raise AssertionError("a stage with input scale 1.0 resized")


#: Scale-1 wrappers of every type: none may call a resize helper.
SCALE_ONE = {
    "single_instance": (lambda: _single_instance(1.0), FRAME),
    "centroid": (lambda: _centroid(1.0), FRAME),
    "centered_instance": (lambda: _centered_instance(1.0), CROP),
    "bottomup": (lambda: _bottomup(1.0), FRAME),
    "multiclass_bottomup": (lambda: _multiclass_bottomup(1.0), FRAME),
    "multiclass_centered_instance": (
        lambda: _multiclass_centered_instance(1.0),
        CROP,
    ),
    "topdown": (lambda: _topdown(1.0, 1.0), FRAME),
    "multiclass_topdown": (lambda: _multiclass_topdown(1.0, 1.0), FRAME),
}


@pytest.mark.parametrize("name", list(SCALE_ONE))
def test_scale_one_wrappers_do_not_resize(name, monkeypatch):
    """At input scale 1.0 no wrapper calls a resize helper or ``F.interpolate``."""
    factory, hw = SCALE_ONE[name]
    wrapper = factory().eval()
    mod = sys.modules[type(wrapper).__module__]
    for helper in ("antialiased_resize", "antialiased_resize_to"):
        monkeypatch.setattr(mod, helper, _refuse, raising=False)
    monkeypatch.setattr(F, "interpolate", _refuse)
    with torch.no_grad():
        wrapper(_blobs(hw))


@pytest.mark.parametrize(
    "factory", [_topdown, _multiclass_topdown], ids=["topdown", "multiclass_topdown"]
)
@pytest.mark.parametrize("stage_at_one", ["centroid", "instance"])
def test_bundle_stage_at_scale_one_does_not_resize(factory, stage_at_one, monkeypatch):
    """A bundle whose centroid (or instance) stage is at 1.0 resizes only the other."""
    wrapper = factory(1.0, 0.5) if stage_at_one == "centroid" else factory(0.5, 1.0)
    mod = sys.modules[type(wrapper).__module__]
    helper = (
        "antialiased_resize" if stage_at_one == "centroid" else "antialiased_resize_to"
    )
    monkeypatch.setattr(mod, helper, _refuse)
    with torch.no_grad():
        wrapper.eval()(_blobs(FRAME))


def test_inexact_scale_warns_once_per_wrapper():
    """A scale no size rule reproduces warns once per frame-resizing wrapper."""
    from loguru import logger

    seen = []
    sink = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        CentroidONNXWrapper(nn.Identity(), input_scale=0.7)
        TopDownONNXWrapper(
            nn.Identity(),
            nn.Identity(),
            centroid_input_scale=0.7,
            instance_input_scale=0.7,
        )
        CentroidONNXWrapper(nn.Identity(), input_scale=0.5)
    finally:
        logger.remove(sink)
    # The bundle's instance stage resizes crops of a fixed size: no size rule.
    assert len(seen) == 2 and all("one pixel" in s for s in seen)
