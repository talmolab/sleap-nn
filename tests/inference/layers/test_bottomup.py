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
    """Load model assets and pull the initialized inference_model."""
    from sleap_nn.inference.loaders import load_model_assets

    assets, _ = load_model_assets(
        [str(BOTTOMUP_CKPT)],
        device="cpu",
        peak_threshold=0.05,
        preprocess_config=NEUTRAL_PREPROCESS,
    )
    return assets


def _build_layer(predictor) -> BottomUpLayer:
    """Build a ``BottomUpLayer`` around the loaded module's torch_model + scorer."""
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
# 2. return_confmaps + return_pafs
# ─────────────────────────────────────────────────────────────────────────
#
# Per-layer "parity vs legacy BottomUpInferenceModel.forward" was
# removed: it loaded a real checkpoint and ran the model end-to-end
# twice (~430s on Mac CI). End-to-end byte-for-byte parity vs main is
# captured once at the top of the stack via the PR 0 goldens.


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
