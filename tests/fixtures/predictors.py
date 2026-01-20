"""Module-scoped predictor fixtures for faster testing.

These fixtures load predictors once per test module to avoid repeated
model loading overhead. Each model load takes ~0.5-1s, so reusing
predictors across tests provides significant speedup.
"""

import pytest
from omegaconf import OmegaConf


@pytest.fixture(scope="module")
def topdown_predictor_module(
    minimal_instance_centroid_ckpt, minimal_instance_centered_instance_ckpt
):
    """Pre-loaded TopDown predictor (module-scoped for reuse across tests)."""
    from sleap_nn.inference.predictors import TopDownPredictor

    predictor = TopDownPredictor.from_trained_models(
        centroid_ckpt_path=str(minimal_instance_centroid_ckpt),
        confmap_ckpt_path=str(minimal_instance_centered_instance_ckpt),
        device="cpu",
        peak_threshold=0.0,
        integral_refinement=None,
    )
    return predictor


@pytest.fixture(scope="module")
def centered_instance_predictor_module(minimal_instance_centered_instance_ckpt):
    """Pre-loaded centered instance predictor (module-scoped)."""
    from sleap_nn.inference.predictors import TopDownPredictor

    predictor = TopDownPredictor.from_trained_models(
        centroid_ckpt_path=None,
        confmap_ckpt_path=str(minimal_instance_centered_instance_ckpt),
        device="cpu",
        peak_threshold=0.0,
        integral_refinement=None,
    )
    return predictor


@pytest.fixture(scope="module")
def bottomup_predictor_module(minimal_instance_bottomup_ckpt):
    """Pre-loaded BottomUp predictor (module-scoped for reuse across tests)."""
    from sleap_nn.inference.predictors import BottomUpPredictor

    predictor = BottomUpPredictor.from_trained_models(
        ckpt_path=str(minimal_instance_bottomup_ckpt),
        device="cpu",
        peak_threshold=0.05,
        integral_refinement=None,
    )
    return predictor


@pytest.fixture(scope="module")
def multiclass_topdown_predictor_module(
    minimal_instance_centroid_ckpt, minimal_instance_multi_class_topdown_ckpt
):
    """Pre-loaded multiclass TopDown predictor (module-scoped)."""
    from sleap_nn.inference.predictors import TopDownMultiClassPredictor

    predictor = TopDownMultiClassPredictor.from_trained_models(
        centroid_ckpt_path=str(minimal_instance_centroid_ckpt),
        confmap_ckpt_path=str(minimal_instance_multi_class_topdown_ckpt),
        device="cpu",
        peak_threshold=0.0,
        integral_refinement=None,
    )
    return predictor


@pytest.fixture(scope="module")
def multiclass_bottomup_predictor_module(minimal_instance_multi_class_bottomup_ckpt):
    """Pre-loaded multiclass BottomUp predictor (module-scoped)."""
    from sleap_nn.inference.predictors import BottomUpMultiClassPredictor

    predictor = BottomUpMultiClassPredictor.from_trained_models(
        ckpt_path=str(minimal_instance_multi_class_bottomup_ckpt),
        device="cpu",
        peak_threshold=0.05,
        integral_refinement=None,
    )
    return predictor
