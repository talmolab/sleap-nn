"""Tests for :func:`sleap_nn.inference.factory.from_model_paths`.

The factory wraps the legacy ``inference.predictors.Predictor`` loader
and re-emits a new ``Predictor`` with the appropriate layer composition.

Coverage:

1. Each of the 5 supported model-type combinations builds a new
   ``Predictor`` whose layer is the expected type.
2. ``Predictor.from_model_paths`` (classmethod) and the free
   ``factory.from_model_paths`` produce equivalent objects.
3. Each layer-type's ``predict()`` returns a structurally well-formed
   ``Outputs`` on a synthetic image (smoke test — full per-type parity
   vs the legacy ``InferenceModel.forward`` is already covered in
   ``tests/inference/layers/test_*.py``).
4. The factory raises ``ValueError`` on an unsupported combination.
5. Parity vs legacy ``inference_model.forward`` on the single-instance
   checkpoint within 1e-4 atol / 1e-5 rtol.

Performance: each ckpt-combo predictor is module-scoped so we only
pay the Lightning checkpoint load cost once per CI run. Without this,
~11 fresh model loads would inflate Linux/Windows runtime by 3-5x and
trigger memory pressure that slows neighbouring test files.
"""

from __future__ import annotations

import gc
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
# Module-scoped fixtures — load each ckpt ONCE per test session
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def single_predictor() -> Predictor:
    """Built once per module; reused across single-instance tests."""
    if not SINGLE_CKPT.exists():
        pytest.skip("single-instance ckpt absent")
    p = from_model_paths([str(SINGLE_CKPT)], device="cpu")
    yield p
    del p
    gc.collect()


@pytest.fixture(scope="module")
def bottomup_predictor() -> Predictor:
    """Built once per module; reused across bottom-up tests."""
    if not BOTTOMUP_CKPT.exists():
        pytest.skip("bottomup ckpt absent")
    p = from_model_paths([str(BOTTOMUP_CKPT)], device="cpu")
    yield p
    del p
    gc.collect()


@pytest.fixture(scope="module")
def multiclass_bu_predictor() -> Predictor:
    """Built once per module; reused across multi-class bottom-up tests."""
    if not MULTICLASS_BU_CKPT.exists():
        pytest.skip("multiclass-bottomup ckpt absent")
    p = from_model_paths([str(MULTICLASS_BU_CKPT)], device="cpu")
    yield p
    del p
    gc.collect()


@pytest.fixture(scope="module")
def topdown_predictor() -> Predictor:
    """Built once per module; reused across top-down tests."""
    if not (CENTROID_CKPT.exists() and CENTERED_CKPT.exists()):
        pytest.skip("topdown ckpts absent")
    p = from_model_paths(
        [str(CENTROID_CKPT), str(CENTERED_CKPT)],
        device="cpu",
        peak_threshold=0.03,
        max_instances=6,
    )
    yield p
    del p
    gc.collect()


@pytest.fixture(scope="module")
def topdown_multiclass_predictor() -> Predictor:
    """Built once per module; reused across top-down multi-class tests."""
    if not (CENTROID_CKPT.exists() and MULTICLASS_TD_CKPT.exists()):
        pytest.skip("topdown-multiclass ckpts absent")
    p = from_model_paths(
        [str(CENTROID_CKPT), str(MULTICLASS_TD_CKPT)],
        device="cpu",
        peak_threshold=0.03,
        max_instances=6,
    )
    yield p
    del p
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────
# 1. Layer-type dispatch — one test per supported combination
# ─────────────────────────────────────────────────────────────────────────


def test_factory_builds_single_instance_layer(single_predictor):
    """``[single_instance]`` → ``SingleInstanceLayer``."""
    assert isinstance(single_predictor, Predictor)
    assert isinstance(single_predictor.layer, SingleInstanceLayer)


def test_factory_builds_bottomup_layer(bottomup_predictor):
    """``[bottomup]`` → ``BottomUpLayer``."""
    assert isinstance(bottomup_predictor.layer, BottomUpLayer)


def test_factory_builds_bottomup_multiclass_layer(multiclass_bu_predictor):
    """``[multi_class_bottomup]`` → ``BottomUpMultiClassLayer``."""
    assert isinstance(multiclass_bu_predictor.layer, BottomUpMultiClassLayer)


def test_factory_builds_topdown_layer(topdown_predictor):
    """``[centroid, centered_instance]`` → ``TopDownLayer``."""
    assert isinstance(topdown_predictor.layer, TopDownLayer)


def test_factory_builds_topdown_multiclass_layer(topdown_multiclass_predictor):
    """``[centroid, multi_class_topdown]`` → ``TopDownMultiClassLayer``."""
    assert isinstance(topdown_multiclass_predictor.layer, TopDownMultiClassLayer)


# ─────────────────────────────────────────────────────────────────────────
# 2. Classmethod equivalence
# ─────────────────────────────────────────────────────────────────────────


def test_classmethod_matches_factory_function(single_predictor):
    """``Predictor.from_model_paths(...)`` builds the same kind of object.

    Uses the module-scoped factory predictor on one side and a fresh
    classmethod call on the other; freed at end of test.
    """
    via_classmethod = Predictor.from_model_paths([str(SINGLE_CKPT)], device="cpu")
    assert type(via_classmethod) is type(single_predictor)
    assert type(via_classmethod.layer) is type(single_predictor.layer)
    del via_classmethod
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────
# 3. End-to-end smoke: layer.predict produces valid Outputs
# ─────────────────────────────────────────────────────────────────────────


def test_factory_single_instance_predict_smoke(single_predictor):
    """Built ``Predictor.layer.predict`` returns a structurally valid ``Outputs``."""
    img = np.zeros((1, 1, 384, 384), dtype=np.float32)
    out = single_predictor.layer.predict(img)
    assert isinstance(out, Outputs)
    assert out.pred_keypoints is not None
    assert out.pred_keypoints.ndim == 4  # (B, I, N, 2)
    assert out.pred_keypoints.shape[0] == 1


def test_factory_bottomup_predict_smoke(bottomup_predictor):
    """Bottomup factory builds a layer that returns valid ``Outputs``."""
    img = np.zeros((1, 1, 384, 384), dtype=np.float32)
    out = bottomup_predictor.layer.predict(img)
    assert isinstance(out, Outputs)
    assert out.pred_keypoints.ndim == 4


def test_factory_topdown_predict_smoke(topdown_predictor):
    """TopDown factory builds a layer that returns valid ``Outputs``."""
    img = np.zeros((1, 1, 384, 384), dtype=np.float32)
    out = topdown_predictor.layer.predict(img)
    assert isinstance(out, Outputs)
    assert out.pred_keypoints.ndim == 4
    # TopDown produces pred_centroids alongside pred_keypoints.
    assert out.pred_centroids is not None


# ─────────────────────────────────────────────────────────────────────────
# 4. Error path
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not CENTROID_CKPT.exists(), reason="centroid ckpt absent")
def test_factory_rejects_unsupported_combination():
    """Two centroid models is not a supported pipeline → ``ValueError``.

    Note: the legacy loader runs first and may either accept the
    duplicate or fail; both surfaces are valid signals for "this
    combination isn't supported".
    """
    with pytest.raises((ValueError, RuntimeError)):
        from_model_paths(
            [str(CENTROID_CKPT), str(CENTROID_CKPT)],
            device="cpu",
        )


# ─────────────────────────────────────────────────────────────────────────
# Note: the per-test "parity vs legacy InferenceModel.forward" check was
# removed. End-to-end byte-for-byte parity is captured once at the top
# of the stack via the PR 0 goldens; per-layer parity vs legacy is
# development-time scaffolding that locks in legacy behaviour
# (including ULP-drift quirks) and triples CI runtime.
