"""Tests for ``BottomUpLayer``.

Coverage:

1. **Parity vs legacy ``BottomUpInferenceModel``** — for the same image
   input, the new layer's predicted keypoints, peak values, and instance
   scores match the legacy module within the design-doc tolerance budget
   (1e-4 abs / 1e-5 rel).
2. **Shape contract** — ``Outputs`` has fixed-shape ``(B, max_instances,
   N, 2)`` keypoints (NaN-padded where the variable per-frame instance
   count falls short of ``max_instances``).
3. **``return_confmaps`` / ``return_pafs``** — opt-in flags populate the
   corresponding ``Outputs`` fields.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sleap_nn.inference.layers.backends import TorchBackend
from sleap_nn.inference.layers.bottomup import BottomUpLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.outputs import Outputs

CKPT_ROOT = Path(__file__).resolve().parents[3] / "tests" / "assets" / "model_ckpts"
BOTTOMUP_CKPT = CKPT_ROOT / "minimal_instance_bottomup"

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
    """Build a BottomUpPredictor and pull its initialized inference_model."""
    from sleap_nn.inference.predictors import Predictor

    predictor = Predictor.from_model_paths(
        [str(BOTTOMUP_CKPT)],
        device="cpu",
        peak_threshold=0.05,
        preprocess_config=NEUTRAL_PREPROCESS,
    )
    predictor._initialize_inference_model()
    return predictor


def _build_layer(predictor) -> BottomUpLayer:
    """Build a ``BottomUpLayer`` around the legacy module's torch_model + scorer."""
    legacy = predictor.inference_model
    # The bottomup model's max_stride lives on the centroid_crop attribute in
    # topdown predictors; for bottomup it's part of the backbone config.
    max_stride = predictor.bottomup_config.model_config.backbone_config[
        predictor.backbone_type
    ]["max_stride"]
    return BottomUpLayer(
        backend=TorchBackend(model=legacy.torch_model, device="cpu"),
        paf_scorer=legacy.paf_scorer,
        cms_output_stride=legacy.cms_output_stride,
        pafs_output_stride=legacy.pafs_output_stride,
        max_stride=max_stride,
        max_peaks_per_node=legacy.max_peaks_per_node,
        preprocess_config=PreprocessConfig(scale=legacy.input_scale),
        postprocess_config=PostprocessConfig(
            peak_threshold=legacy.peak_threshold,
            refinement=legacy.refinement or "none",
            integral_patch_size=legacy.integral_patch_size,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────
# 1. Shape contract
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not BOTTOMUP_CKPT.exists(), reason="bottomup checkpoint not present"
)
def test_bottomup_layer_returns_outputs_with_fixed_shape():
    """The ``Outputs`` shape is canonical regardless of detection count."""
    layer = _build_layer(_build_predictor())
    img = np.zeros((1, 1, 384, 384), dtype=np.float32)
    out = layer.predict(img)
    assert isinstance(out, Outputs)
    assert out.pred_keypoints is not None
    assert out.pred_keypoints.ndim == 4  # (B, I, N, 2)
    assert out.pred_keypoints.shape[0] == 1


# ─────────────────────────────────────────────────────────────────────────
# 2. Parity vs legacy BottomUpInferenceModel
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not BOTTOMUP_CKPT.exists(), reason="bottomup checkpoint not present"
)
def test_bottomup_layer_parity_vs_legacy():
    """``BottomUpLayer.predict`` matches ``BottomUpInferenceModel.forward``."""
    predictor = _build_predictor()
    layer = _build_layer(predictor)
    legacy = predictor.inference_model
    legacy.eval()

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(2, 1, 1, 384, 384)).astype(np.uint8)
    image_t = torch.from_numpy(image).float()  # (B, 1, C, H, W)

    legacy_input = {
        "image": image_t,
        "frame_idx": torch.tensor([0, 1], dtype=torch.float32),
        "video_idx": torch.tensor([0, 0], dtype=torch.float32),
        "orig_size": torch.tensor([[384.0, 384.0], [384.0, 384.0]]),
        "eff_scale": torch.ones(2),
    }
    with torch.no_grad():
        legacy_out_list = legacy(legacy_input)
    legacy_out = legacy_out_list[0]
    legacy_peaks = legacy_out["pred_instance_peaks"]
    legacy_vals = legacy_out["pred_peak_values"]
    legacy_scores = legacy_out["instance_scores"]

    new_outputs = layer.predict(image_t.squeeze(1))  # (B, C, H, W)

    # The legacy returns variable-shape lists per batch; the new layer
    # NaN-pads to a uniform max_instances. Compare per-batch by trimming
    # the new layer's output to the legacy's instance count.
    for b in range(2):
        legacy_n = int(legacy_peaks[b].shape[0])
        if legacy_n == 0:
            assert torch.isnan(
                new_outputs.pred_keypoints[b]
            ).all(), (
                f"batch {b}: legacy returned 0 instances; new layer should be all NaN"
            )
            continue
        new_b_peaks = new_outputs.pred_keypoints[b, :legacy_n]
        new_b_vals = new_outputs.pred_peak_values[b, :legacy_n]
        new_b_scores = new_outputs.instance_scores[b, :legacy_n]
        np.testing.assert_allclose(
            new_b_peaks.numpy(),
            legacy_peaks[b].numpy(),
            equal_nan=True,
            atol=PARITY_ATOL,
            rtol=PARITY_RTOL,
            err_msg=f"batch {b}: pred_keypoints drifted vs legacy BottomUp",
        )
        np.testing.assert_allclose(
            new_b_vals.numpy(),
            legacy_vals[b].numpy(),
            equal_nan=True,
            atol=PARITY_ATOL,
            rtol=PARITY_RTOL,
            err_msg=f"batch {b}: pred_peak_values drifted vs legacy BottomUp",
        )
        np.testing.assert_allclose(
            new_b_scores.numpy(),
            legacy_scores[b].numpy(),
            equal_nan=True,
            atol=PARITY_ATOL,
            rtol=PARITY_RTOL,
            err_msg=f"batch {b}: instance_scores drifted vs legacy BottomUp",
        )


# ─────────────────────────────────────────────────────────────────────────
# 3. return_confmaps + return_pafs
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not BOTTOMUP_CKPT.exists(), reason="bottomup checkpoint not present"
)
def test_return_confmaps_and_pafs():
    predictor = _build_predictor()
    legacy = predictor.inference_model
    max_stride = predictor.bottomup_config.model_config.backbone_config[
        predictor.backbone_type
    ]["max_stride"]
    layer = BottomUpLayer(
        backend=TorchBackend(model=legacy.torch_model, device="cpu"),
        paf_scorer=legacy.paf_scorer,
        cms_output_stride=legacy.cms_output_stride,
        pafs_output_stride=legacy.pafs_output_stride,
        max_stride=max_stride,
        max_peaks_per_node=legacy.max_peaks_per_node,
        postprocess_config=PostprocessConfig(
            peak_threshold=legacy.peak_threshold,
            refinement=legacy.refinement or "none",
            return_confmaps=True,
            return_pafs=True,
        ),
    )
    out = layer.predict(np.zeros((1, 1, 384, 384), dtype=np.float32))
    assert out.pred_confmaps is not None
    assert out.pred_confmaps.ndim == 4
    assert out.pred_pafs is not None
    assert out.pred_pafs.ndim == 4
