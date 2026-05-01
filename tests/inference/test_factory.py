"""Tests for :func:`sleap_nn.inference.factory.from_model_paths`.

The factory wraps the legacy ``inference.predictors.Predictor`` loader
and re-emits a new ``Predictor`` with the appropriate layer composition.

Coverage:

1. Each of the 6 supported model-type combinations builds a new
   ``Predictor`` whose layer is the expected type.
2. ``Predictor.from_model_paths`` (classmethod) and the free
   ``factory.from_model_paths`` produce equivalent objects.
3. Each layer-type's ``predict()`` returns a structurally well-formed
   ``Outputs`` on a synthetic image (smoke test — full per-type parity
   vs the legacy ``InferenceModel.forward`` is already covered in
   ``tests/inference/layers/test_*.py``).
4. The factory raises ``ValueError`` on an unsupported combination
   (e.g., two centroid models).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sleap_nn.inference.factory import from_model_paths
from sleap_nn.inference.layers.bottomup import BottomUpLayer
from sleap_nn.inference.layers.bottomup_multiclass import BottomUpMultiClassLayer
from sleap_nn.inference.layers.single_instance import SingleInstanceLayer
from sleap_nn.inference.layers.topdown import TopDownLayer
from sleap_nn.inference.layers.topdown_multiclass import TopDownMultiClassLayer
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.predictor import Predictor

CKPT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "model_ckpts"
SINGLE_CKPT = CKPT_ROOT / "minimal_instance_single_instance"
BOTTOMUP_CKPT = CKPT_ROOT / "minimal_instance_bottomup"
MULTICLASS_BU_CKPT = CKPT_ROOT / "minimal_instance_multiclass_bottomup"
CENTROID_CKPT = CKPT_ROOT / "minimal_instance_centroid"
CENTERED_CKPT = CKPT_ROOT / "minimal_instance_centered_instance"
MULTICLASS_TD_CKPT = CKPT_ROOT / "minimal_instance_multiclass_centered_instance"


# ─────────────────────────────────────────────────────────────────────────
# 1. Layer-type dispatch — one test per supported combination
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not SINGLE_CKPT.exists(), reason="single-instance ckpt absent")
def test_factory_builds_single_instance_layer():
    """``[single_instance]`` → ``SingleInstanceLayer``."""
    predictor = from_model_paths([str(SINGLE_CKPT)], device="cpu")
    assert isinstance(predictor, Predictor)
    assert isinstance(predictor.layer, SingleInstanceLayer)


@pytest.mark.skipif(not BOTTOMUP_CKPT.exists(), reason="bottomup ckpt absent")
def test_factory_builds_bottomup_layer():
    """``[bottomup]`` → ``BottomUpLayer``."""
    predictor = from_model_paths([str(BOTTOMUP_CKPT)], device="cpu")
    assert isinstance(predictor.layer, BottomUpLayer)


@pytest.mark.skipif(
    not MULTICLASS_BU_CKPT.exists(), reason="multiclass-bottomup ckpt absent"
)
def test_factory_builds_bottomup_multiclass_layer():
    """``[multi_class_bottomup]`` → ``BottomUpMultiClassLayer``."""
    predictor = from_model_paths([str(MULTICLASS_BU_CKPT)], device="cpu")
    assert isinstance(predictor.layer, BottomUpMultiClassLayer)


@pytest.mark.skipif(
    not (CENTROID_CKPT.exists() and CENTERED_CKPT.exists()),
    reason="topdown ckpts absent",
)
def test_factory_builds_topdown_layer():
    """``[centroid, centered_instance]`` → ``TopDownLayer``."""
    predictor = from_model_paths(
        [str(CENTROID_CKPT), str(CENTERED_CKPT)],
        device="cpu",
        peak_threshold=0.03,
        max_instances=6,
    )
    assert isinstance(predictor.layer, TopDownLayer)


@pytest.mark.skipif(
    not (CENTROID_CKPT.exists() and MULTICLASS_TD_CKPT.exists()),
    reason="topdown-multiclass ckpts absent",
)
def test_factory_builds_topdown_multiclass_layer():
    """``[centroid, multi_class_topdown]`` → ``TopDownMultiClassLayer``."""
    predictor = from_model_paths(
        [str(CENTROID_CKPT), str(MULTICLASS_TD_CKPT)],
        device="cpu",
        peak_threshold=0.03,
        max_instances=6,
    )
    assert isinstance(predictor.layer, TopDownMultiClassLayer)


# ─────────────────────────────────────────────────────────────────────────
# 2. Classmethod equivalence
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not SINGLE_CKPT.exists(), reason="single-instance ckpt absent")
def test_classmethod_matches_factory_function():
    """``Predictor.from_model_paths(...)`` builds the same kind of object."""
    via_classmethod = Predictor.from_model_paths([str(SINGLE_CKPT)], device="cpu")
    via_function = from_model_paths([str(SINGLE_CKPT)], device="cpu")
    assert type(via_classmethod) is type(via_function)
    assert type(via_classmethod.layer) is type(via_function.layer)


# ─────────────────────────────────────────────────────────────────────────
# 3. End-to-end smoke: layer.predict produces valid Outputs
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not SINGLE_CKPT.exists(), reason="single-instance ckpt absent")
def test_factory_single_instance_predict_smoke():
    """Built ``Predictor.layer.predict`` returns a structurally valid ``Outputs``."""
    predictor = from_model_paths([str(SINGLE_CKPT)], device="cpu")
    img = np.zeros((1, 1, 384, 384), dtype=np.float32)
    out = predictor.layer.predict(img)
    assert isinstance(out, Outputs)
    assert out.pred_keypoints is not None
    assert out.pred_keypoints.ndim == 4  # (B, I, N, 2)
    assert out.pred_keypoints.shape[0] == 1


@pytest.mark.skipif(not BOTTOMUP_CKPT.exists(), reason="bottomup ckpt absent")
def test_factory_bottomup_predict_smoke():
    """Bottomup factory builds a layer that returns valid ``Outputs``."""
    predictor = from_model_paths([str(BOTTOMUP_CKPT)], device="cpu")
    img = np.zeros((1, 1, 384, 384), dtype=np.float32)
    out = predictor.layer.predict(img)
    assert isinstance(out, Outputs)
    assert out.pred_keypoints.ndim == 4


@pytest.mark.skipif(
    not (CENTROID_CKPT.exists() and CENTERED_CKPT.exists()),
    reason="topdown ckpts absent",
)
def test_factory_topdown_predict_smoke():
    """TopDown factory builds a layer that returns valid ``Outputs``."""
    predictor = from_model_paths(
        [str(CENTROID_CKPT), str(CENTERED_CKPT)],
        device="cpu",
        peak_threshold=0.03,
        max_instances=6,
    )
    img = np.zeros((1, 1, 384, 384), dtype=np.float32)
    out = predictor.layer.predict(img)
    assert isinstance(out, Outputs)
    assert out.pred_keypoints.ndim == 4
    # TopDown produces pred_centroids alongside pred_keypoints.
    assert out.pred_centroids is not None


# ─────────────────────────────────────────────────────────────────────────
# 4. Error path
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not CENTROID_CKPT.exists(), reason="centroid ckpt absent")
def test_factory_rejects_unsupported_combination():
    """Two centroid models is not a supported pipeline → ``ValueError``."""
    # The legacy loader will accept two-centroid input but the factory's
    # _select_layer dispatch refuses it. Capture either failure point.
    with pytest.raises((ValueError, RuntimeError)):
        from_model_paths(
            [str(CENTROID_CKPT), str(CENTROID_CKPT)],
            device="cpu",
        )


# ─────────────────────────────────────────────────────────────────────────
# 5. Parity vs legacy Predictor.from_model_paths on the same image
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not SINGLE_CKPT.exists(), reason="single-instance ckpt absent")
def test_factory_parity_vs_legacy_single_instance():
    """New ``Predictor.layer.predict`` matches legacy ``inference_model.forward``."""
    from omegaconf import OmegaConf

    from sleap_nn.inference.predictors import Predictor as LegacyPredictor

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(2, 1, 1, 384, 384)).astype(np.uint8)
    image_t = torch.from_numpy(image).float()  # (B, 1, C, H, W)

    # Legacy path
    legacy = LegacyPredictor.from_model_paths(
        [str(SINGLE_CKPT)],
        device="cpu",
        peak_threshold=0.2,
        preprocess_config=OmegaConf.create(
            {
                "ensure_rgb": None,
                "ensure_grayscale": None,
                "crop_size": None,
                "max_width": None,
                "max_height": None,
                "scale": None,
            }
        ),
    )
    legacy._initialize_inference_model()
    legacy_inf = legacy.inference_model
    legacy_inf.eval()
    legacy_input = {
        "image": image_t,
        "frame_idx": torch.tensor([0, 1], dtype=torch.float32),
        "video_idx": torch.tensor([0, 0], dtype=torch.float32),
        "orig_size": torch.tensor([[384.0, 384.0], [384.0, 384.0]]),
        "eff_scale": torch.ones(2),
    }
    with torch.no_grad():
        legacy_out_list = legacy_inf(legacy_input)
    legacy_out = legacy_out_list[0]
    legacy_peaks = legacy_out["pred_instance_peaks"]

    # New factory path
    predictor = from_model_paths([str(SINGLE_CKPT)], device="cpu", peak_threshold=0.2)
    new_outputs = predictor.layer.predict(image_t.squeeze(1))  # (B, C, H, W)
    new_peaks = new_outputs.pred_keypoints

    # Single-instance: new_peaks is (B, 1, N, 2); legacy is (B, N, 2).
    # Squeeze the I=1 dim for shape match.
    np.testing.assert_allclose(
        new_peaks.squeeze(1).numpy(),
        legacy_peaks.numpy(),
        equal_nan=True,
        atol=1e-4,
        rtol=1e-5,
    )
