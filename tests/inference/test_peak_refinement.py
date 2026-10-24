"""DARK-by-default top-down decoding and flip-test through layers, loaders and CLI.

The ops themselves are tested in ``tests/inference/ops/test_peak_refinement.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio
import torch
from _pytest.logging import LogCaptureFixture
from click.testing import CliRunner
from loguru import logger

from sleap_nn.data.confidence_maps import make_confmaps
from sleap_nn.data.utils import make_grid_vectors
from sleap_nn.inference.layers.centered_instance import CenteredInstanceLayer
from sleap_nn.inference.layers.configs import PostprocessConfig
from sleap_nn.inference.layers.single_instance import SingleInstanceLayer
from sleap_nn.inference.loaders import _flip_test_idx
from sleap_nn.inference.ops.peaks import flip_idx_from_symmetries
from sleap_nn.inference.predictor import Predictor
from sleap_nn.inference.topdown import FindInstancePeaks

STRIDE, SIGMA = 4, 2.0
CKPTS = Path(__file__).resolve().parents[1] / "assets" / "model_ckpts"
CENTROID = CKPTS / "minimal_instance_centroid"
CENTERED = CKPTS / "minimal_instance_centered_instance"
BOTTOMUP = CKPTS / "minimal_instance_bottomup"
SINGLE = CKPTS / "minimal_instance_single_instance"
SLP = Path(__file__).resolve().parents[1] / "assets" / "datasets"
SLP = SLP / "minimal_instance.pkg.slp"


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """Route loguru records into pytest's ``caplog`` (project convention)."""
    handler_id = logger.add(caplog.handler, format="{message}", level=0)
    yield caplog
    logger.remove(handler_id)


class _MirrorModel:
    """Stub backend: confmap ch0 = image left half, ch1 = right half (stride 4).

    The input image holds Gaussian blobs, so sampling it at ``i * stride`` gives
    exactly sleap-nn's confmap target. Under a horizontal flip the left node's
    blob lands in the right-half channel, which ``flip_idx`` must undo.
    """

    device = "cpu"
    does_baked_postproc = False

    def warmup(self, input_shape):
        return None

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1]).float()[:, :1]
        cm = x[..., ::STRIDE, ::STRIDE]
        w = cm.shape[-1]
        left = torch.zeros_like(cm)
        left[..., : w // 2] = 1
        return {"output": torch.cat([cm * left, cm * (1 - left)], dim=1)}


def _two_node_image(h=96, w=96):
    """Image of a bright left node 0 and a dimmer right node 1 (input px)."""
    pts = torch.tensor([[30.3, 41.7], [70.6, 50.2]])
    xv, yv = make_grid_vectors(h, w, 1)
    # Node 1 is dimmer so a missing left/right swap visibly picks the wrong blob.
    amp = torch.tensor([1.0, 0.6]).view(1, 2, 1, 1)
    img = (make_confmaps(pts[None], xv, yv, SIGMA * STRIDE) * amp).amax(
        dim=1, keepdim=True
    )
    return img, pts


def _centered_layer(**kwargs):
    postprocess = kwargs.pop("postprocess", {})
    return CenteredInstanceLayer(
        backend=_MirrorModel(),
        output_stride=STRIDE,
        postprocess_config=PostprocessConfig(
            peak_threshold=0.1, dark_blur_sigma=SIGMA, **postprocess
        ),
        **kwargs,
    )


def test_centered_instance_layer_defaults_to_dark():
    img, pts = _two_node_image()
    crops = img.expand(2, 1, *img.shape[-2:]).contiguous()
    assert PostprocessConfig().refinement == "auto"
    dark = _centered_layer().predict(crops).pred_keypoints[:, 0]
    integral = (
        _centered_layer(postprocess={"refinement": "integral"})
        .predict(crops)
        .pred_keypoints[:, 0]
    )
    # (the half-image channel masks truncate the blobs a little)
    assert (dark - pts).norm(dim=-1).max() < 0.15
    assert (integral - pts).norm(dim=-1).max() > 0.3


def test_centered_instance_layer_flip_test():
    img, pts = _two_node_image()
    crops = img.expand(2, 1, *img.shape[-2:]).contiguous()
    flipped = _centered_layer(
        flip_test=True, flip_idx=flip_idx_from_symmetries(2, [(0, 1)])
    )
    out = flipped.predict(crops).pred_keypoints[:, 0]
    assert (out - pts).norm(dim=-1).max() < 0.15
    # Without swapping left/right channels the mirrored pass mixes up the nodes.
    wrong = _centered_layer(flip_test=True).predict(crops).pred_keypoints[:, 0]
    assert (wrong - pts).norm(dim=-1).max() > 5


def test_centered_instance_layer_flip_test_with_stride_padding():
    """A crop padded to ``max_stride`` mirrors about the padded width."""
    img, pts = _two_node_image(h=90, w=90)  # padded to 96 by max_stride=32
    crops = img.expand(1, 1, 90, 90).contiguous()
    layer = _centered_layer(
        max_stride=32, flip_test=True, flip_idx=flip_idx_from_symmetries(2, [(0, 1)])
    )
    out = layer.predict(crops).pred_keypoints[:, 0]
    assert (out - pts).norm(dim=-1).max() < 0.2


def test_single_instance_layer_auto_is_integral():
    img, _ = _two_node_image()

    class _Backend(_MirrorModel):
        def __call__(self, x):
            return {"output": super().__call__(x)["output"]}

    def run(refinement):
        layer = SingleInstanceLayer(
            backend=_Backend(),
            output_stride=STRIDE,
            postprocess_config=PostprocessConfig(
                peak_threshold=0.1, refinement=refinement
            ),
        )
        return layer.predict(img).pred_keypoints

    assert torch.equal(run("auto"), run("integral"))
    assert not torch.equal(run("auto"), run("dark"))


def test_find_instance_peaks_flip_test_and_dark():
    """``FindInstancePeaks`` (in-training eval path) runs DARK + flip-test."""
    img, pts = _two_node_image()
    h, w = img.shape[-2:]
    stub = _MirrorModel()
    layer = FindInstancePeaks(
        torch_model=lambda x: stub(x)["output"],
        output_stride=STRIDE,
        peak_threshold=0.1,
        refinement="dark",
        dark_blur_sigma=SIGMA,
        flip_test=True,
        flip_idx=flip_idx_from_symmetries(2, [(0, 1)]),
    )
    out = layer(
        {
            "instance_image": img[:, None].expand(2, 1, 1, h, w).contiguous(),
            "eff_scale": torch.ones(2),
            "instance_bbox": torch.zeros(2, 1, 4, 2),
        }
    )
    assert (out["pred_instance_peaks"] - pts).norm(dim=-1).max() < 0.15


def _skeleton(symmetries):
    nodes = ["head", "left_ear", "right_ear", "left_paw", "right_paw"]
    skel = sio.Skeleton(nodes=list(nodes))
    for a, b in symmetries:
        skel.add_symmetry(a, b)
    return skel


def test_flip_test_idx_follows_part_names():
    skel = _skeleton([("left_ear", "right_ear"), ("left_paw", "right_paw")])
    assert _flip_test_idx(skel, None).tolist() == [0, 2, 1, 4, 3]
    # A head that predicts a subset (and a different order): pairs with a node
    # it does not predict are dropped.
    idx = _flip_test_idx(skel, ["right_ear", "head", "left_ear", "left_paw"])
    assert idx.tolist() == [2, 1, 0, 3]


def test_flip_test_idx_warns_without_symmetries(caplog):
    idx = _flip_test_idx(_skeleton([]), None)
    assert idx.tolist() == [0, 1, 2, 3, 4]
    assert "no symmetries" in caplog.text


def _require(*paths):
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"missing fixture {p}")


def test_topdown_predictor_defaults_and_opt_out():
    """Top-down keypoints decode with DARK by default; 'integral' opts out."""
    _require(CENTROID, CENTERED, SLP)
    pred = Predictor.from_model_paths([str(CENTROID), str(CENTERED)], device="cpu")
    inst = pred.layer.centered_instance_layer.postprocess_config
    assert inst.refinement == "dark"
    assert inst.dark_blur_sigma == 1.5  # the head's training sigma
    assert pred.layer.centroid_layer.postprocess_config.refinement == "integral"
    assert pred.layer.centered_instance_layer.flip_test is False

    def keypoints(**kwargs):
        labels = pred.predict(str(SLP), make_labels=True, **kwargs)
        return np.stack([i.numpy() for lf in labels for i in lf.instances])

    dark = keypoints()
    integral = keypoints(integral_refinement="integral")
    assert dark.shape == integral.shape
    assert not np.allclose(dark, integral, equal_nan=True)
    # Same peaks, different sub-pixel offsets (output stride 2).
    finite = np.isfinite(dark) & np.isfinite(integral)
    assert np.abs(dark - integral)[finite].max() < 2.0
    # The per-call override resolves "auto" per stage, like the default.
    assert np.allclose(keypoints(integral_refinement="auto"), dark, equal_nan=True)
    # Explicit dark refines keypoints; the centroid stage falls back to integral.
    assert np.allclose(keypoints(integral_refinement="dark"), dark, equal_nan=True)


def test_topdown_predictor_flip_test(caplog):
    _require(CENTROID, CENTERED, SLP)
    pred = Predictor.from_model_paths(
        [str(CENTROID), str(CENTERED)], device="cpu", flip_test=True
    )
    layer = pred.layer.centered_instance_layer
    assert layer.flip_test is True
    # The fixture skeleton has no symmetries: plain mirror, with a warning.
    assert layer.flip_idx.tolist() == [0, 1]
    assert "no symmetries" in caplog.text
    labels = pred.predict(str(SLP), make_labels=True)
    assert sum(len(lf.instances) for lf in labels) > 0


def test_flip_test_requires_centered_instance():
    _require(SINGLE)
    with pytest.raises(ValueError, match="flip_test is implemented"):
        Predictor.from_model_paths([str(SINGLE)], device="cpu", flip_test=True)


def test_bottomup_rejects_global_only_refinement():
    _require(BOTTOMUP)
    with pytest.raises(ValueError, match="bottom-up"):
        Predictor.from_model_paths(
            [str(BOTTOMUP)], device="cpu", integral_refinement="dark"
        )
    pred = Predictor.from_model_paths([str(BOTTOMUP)], device="cpu")
    assert pred.layer.postprocess_config.refinement == "integral"
    with pytest.raises(ValueError, match="bottom-up"):
        pred.predict(str(SLP), integral_refinement="local")


def test_unknown_refinement_is_rejected_before_loading():
    _require(SINGLE)
    with pytest.raises(ValueError, match="Unknown peak refinement"):
        Predictor.from_model_paths(
            [str(SINGLE)], device="cpu", integral_refinement="subpixel"
        )


def test_single_instance_explicit_dark_uses_training_sigma():
    _require(SINGLE)
    pred = Predictor.from_model_paths(
        [str(SINGLE)], device="cpu", integral_refinement="dark"
    )
    cfg = pred.layer.postprocess_config
    assert cfg.refinement == "dark"
    assert cfg.dark_blur_sigma == 5.0  # the head's training sigma


def test_run_predict_refuses_flip_test_for_export_dir(tmp_path):
    from sleap_nn.inference.run import predict

    with pytest.raises(ValueError, match="not available for exported"):
        predict(str(SLP), export_dir=str(tmp_path), flip_test=True)


def test_cli_flip_test_refused_for_export_dir(tmp_path):
    from sleap_nn.cli import cli

    (tmp_path / "model.onnx").write_bytes(b"")
    result = CliRunner().invoke(
        cli, ["predict", "-i", str(SLP), "-m", str(tmp_path), "--flip_test"]
    )
    assert result.exit_code != 0
    assert "--flip_test is not supported for exported" in result.output


def test_cli_integral_refinement_choices():
    from sleap_nn.cli import cli

    result = CliRunner().invoke(
        cli,
        ["predict", "-i", str(SLP), "-m", str(SINGLE), "--integral_refinement", "x"],
    )
    assert result.exit_code != 0
    assert "auto" in result.output and "dark" in result.output
