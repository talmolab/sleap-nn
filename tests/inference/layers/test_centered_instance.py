"""Tests for ``CenteredInstanceLayer``.

Coverage:

1. **Parity vs legacy ``FindInstancePeaks``** — model path matches the
   legacy Lightning module's keypoint output bit-exactly.
2. **GT path** — ``use_gt_peaks=True`` matches each centroid to its
   nearest GT instance and returns those keypoints. Compared against
   the legacy ``FindInstancePeaksGroundTruth`` matching.
3. **GT path requires centroids + instances** — calling without either
   raises ``ValueError``.
4. **Direct numpy/torch crop input** — works on raw-numpy crops.
5. **``return_confmaps``** — opt-in populates ``Outputs.pred_confmaps``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sleap_nn.inference.layers.backends import TorchBackend
from sleap_nn.inference.layers.centered_instance import CenteredInstanceLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.outputs import Outputs

CKPT_ROOT = Path(__file__).resolve().parents[3] / "tests" / "assets" / "model_ckpts"
CENTROID_CKPT = CKPT_ROOT / "minimal_instance_centroid"
CENTERED_CKPT = CKPT_ROOT / "minimal_instance_centered_instance"

PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-5

NEUTRAL_PREPROCESS = OmegaConf.create(
    {
        "ensure_rgb": None,
        "ensure_grayscale": None,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }
)


def _build_predictor():
    """Build the topdown predictor and pull its initialized inference_model."""
    from sleap_nn.inference.predictors import Predictor

    predictor = Predictor.from_model_paths(
        [str(CENTROID_CKPT), str(CENTERED_CKPT)],
        device="cpu",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=NEUTRAL_PREPROCESS,
    )
    predictor._initialize_inference_model()
    return predictor


def _build_layer(predictor) -> CenteredInstanceLayer:
    """Build a ``CenteredInstanceLayer`` around the legacy ``FindInstancePeaks``."""
    legacy = predictor.inference_model.instance_peaks
    return CenteredInstanceLayer(
        backend=TorchBackend(model=legacy.torch_model, device="cpu"),
        output_stride=legacy.output_stride,
        max_stride=legacy.max_stride,
        preprocess_config=PreprocessConfig(scale=legacy.input_scale),
        postprocess_config=PostprocessConfig(
            peak_threshold=legacy.peak_threshold,
            refinement=legacy.refinement or "none",
            integral_patch_size=legacy.integral_patch_size,
        ),
    )


# Per-layer "parity vs legacy FindInstancePeaks.forward" was removed:
# end-to-end byte-for-byte parity is captured at the top of the stack
# via the PR 0 goldens.


# ─────────────────────────────────────────────────────────────────────────
# 1. GT path: matches centroid → nearest GT instance
# ─────────────────────────────────────────────────────────────────────────


def test_use_gt_peaks_matches_nearest_instance():
    """The GT path returns the GT instance whose nearest keypoint is
    closest to the centroid."""
    # Two batch slots, one centroid each, two candidate GT instances per slot.
    centroids = torch.tensor([[[10.0, 10.0]], [[50.0, 50.0]]])  # (B=2, max_inst=1, 2)
    instances = torch.tensor(
        [
            [
                [[1.0, 1.0], [2.0, 2.0]],  # GT instance 0 — close to (10,10)? no, far
                [[10.0, 10.0], [11.0, 11.0]],  # GT instance 1 — close
            ],
            [
                [[50.0, 50.0], [51.0, 51.0]],  # GT 0 — close
                [[1.0, 1.0], [2.0, 2.0]],  # GT 1 — far
            ],
        ]
    )

    layer = CenteredInstanceLayer(
        backend=TorchBackend(model=torch.nn.Identity(), device="cpu"),
        output_stride=1,
        use_gt_peaks=True,
    )
    out = layer.predict(crops=None, centroids=centroids, instances=instances)
    # B=0 → matched instance 1 (the close one)
    assert torch.allclose(out.pred_keypoints[0, 0], instances[0, 1])
    # B=1 → matched instance 0 (the close one)
    assert torch.allclose(out.pred_keypoints[1, 0], instances[1, 0])


def test_use_gt_peaks_propagates_nan_centroids():
    """A NaN centroid should produce NaN matched keypoints (no spurious
    match against a far-away GT instance)."""
    centroids = torch.tensor([[[float("nan"), float("nan")]]])  # (1, 1, 2)
    instances = torch.tensor([[[[1.0, 1.0], [2.0, 2.0]]]])  # (1, 1, 2, 2)

    layer = CenteredInstanceLayer(
        backend=TorchBackend(model=torch.nn.Identity(), device="cpu"),
        output_stride=1,
        use_gt_peaks=True,
    )
    out = layer.predict(crops=None, centroids=centroids, instances=instances)
    assert torch.isnan(out.pred_keypoints).all()
    assert torch.isnan(out.pred_peak_values).all()


# ─────────────────────────────────────────────────────────────────────────
# 3. GT path requires centroids + instances
# ─────────────────────────────────────────────────────────────────────────


def test_gt_path_requires_centroids_and_instances():
    layer = CenteredInstanceLayer(
        backend=TorchBackend(model=torch.nn.Identity(), device="cpu"),
        output_stride=1,
        use_gt_peaks=True,
    )
    with pytest.raises(ValueError, match="centroids.*instances"):
        layer.predict(crops=None)
    with pytest.raises(ValueError, match="centroids.*instances"):
        layer.predict(crops=None, centroids=torch.zeros(1, 1, 2))


# ─────────────────────────────────────────────────────────────────────────
# 4. Direct numpy crops accepted by model path
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not CENTERED_CKPT.exists(), reason="centered-instance checkpoint not present"
)
def test_predict_accepts_numpy_crops():
    layer = _build_layer(_build_predictor())
    crops = np.zeros((2, 1, 96, 96), dtype=np.float32)
    out = layer.predict(crops)
    assert isinstance(out, Outputs)
    # (B, I=1, N, 2)
    assert out.pred_keypoints.ndim == 4
    assert out.pred_keypoints.shape[0] == 2
    assert out.pred_keypoints.shape[1] == 1


# ─────────────────────────────────────────────────────────────────────────
# 5. return_confmaps
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not CENTERED_CKPT.exists(), reason="centered-instance checkpoint not present"
)
def test_return_confmaps_opt_in():
    predictor = _build_predictor()
    legacy = predictor.inference_model.instance_peaks
    layer = CenteredInstanceLayer(
        backend=TorchBackend(model=legacy.torch_model, device="cpu"),
        output_stride=legacy.output_stride,
        max_stride=legacy.max_stride,
        postprocess_config=PostprocessConfig(
            peak_threshold=legacy.peak_threshold,
            refinement=legacy.refinement or "none",
            return_confmaps=True,
        ),
    )
    out = layer.predict(np.zeros((1, 1, 96, 96), dtype=np.float32))
    assert out.pred_confmaps is not None
    assert out.pred_confmaps.ndim == 4
