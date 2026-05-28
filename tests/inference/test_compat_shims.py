"""Tests for the legacy ``sleap_nn.inference.predictors`` compatibility shim.

PR 23 of #508. The legacy ``Predictor.from_model_paths`` dispatcher and each
of the 5 ``*Predictor.from_trained_models`` classmethods emit a
:class:`DeprecationWarning` pointing callers at
:meth:`sleap_nn.inference.Predictor.from_model_paths` (the new factory).

The factory itself still uses these classes as its checkpoint loader, so it
wraps delegation calls in :func:`legacy_predictor_internal_use` to stay
silent. These tests pin both behaviors.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from sleap_nn.inference.predictors import (
    BottomUpMultiClassPredictor,
    BottomUpPredictor,
    Predictor as LegacyPredictor,
    SingleInstancePredictor,
    TopDownMultiClassPredictor,
    TopDownPredictor,
    legacy_predictor_internal_use,
)

CKPT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "model_ckpts"
SINGLE_CKPT = CKPT_ROOT / "minimal_instance_single_instance"


def _capture_deprecation_warnings(callable_, *args, **kwargs):
    """Run ``callable_`` and return DeprecationWarnings raised, swallowing errors."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            callable_(*args, **kwargs)
        except Exception:
            # Bogus paths trigger downstream errors; the warning fires first.
            pass
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_predictor_from_model_paths_emits_deprecation_warning():
    """The abstract dispatcher emits ``DeprecationWarning`` at call entry."""
    deps = _capture_deprecation_warnings(
        LegacyPredictor.from_model_paths,
        model_paths=["/nonexistent/path"],
    )
    assert deps, "expected DeprecationWarning from Predictor.from_model_paths"
    msg = str(deps[0].message)
    assert "Predictor.from_model_paths" in msg
    assert "sleap_nn.inference" in msg
    assert "Predictor.from_model_paths" in msg
    assert "removed in a future release" in msg


@pytest.mark.parametrize(
    "predictor_cls,kwargs",
    [
        (SingleInstancePredictor, {"confmap_ckpt_path": "/nonexistent/path"}),
        (
            TopDownPredictor,
            {"centroid_ckpt_path": None, "confmap_ckpt_path": "/nonexistent/path"},
        ),
        (BottomUpPredictor, {"bottomup_ckpt_path": "/nonexistent/path"}),
        (
            BottomUpMultiClassPredictor,
            {"bottomup_ckpt_path": "/nonexistent/path"},
        ),
        (
            TopDownMultiClassPredictor,
            {"centroid_ckpt_path": None, "confmap_ckpt_path": "/nonexistent/path"},
        ),
    ],
    ids=[
        "single_instance",
        "topdown",
        "bottomup",
        "bottomup_multiclass",
        "topdown_multiclass",
    ],
)
def test_concrete_predictor_from_trained_models_emits_deprecation_warning(
    predictor_cls, kwargs
):
    """Each of the 5 concrete predictors emits ``DeprecationWarning``."""
    deps = _capture_deprecation_warnings(predictor_cls.from_trained_models, **kwargs)
    assert deps, f"expected DeprecationWarning from {predictor_cls.__name__}"
    msg = str(deps[0].message)
    assert predictor_cls.__name__ in msg
    assert "from_trained_models" in msg


def test_legacy_predictor_internal_use_suppresses_warning():
    """The context manager silences the DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with legacy_predictor_internal_use():
            try:
                LegacyPredictor.from_model_paths(model_paths=["/nonexistent/path"])
            except Exception:
                pass
    deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deps, (
        f"legacy_predictor_internal_use should suppress DeprecationWarning, "
        f"but got: {[str(w.message) for w in deps]}"
    )


def test_legacy_predictor_internal_use_restores_state():
    """The context manager nests cleanly and restores state on exit."""
    from sleap_nn.inference.predictors import _LEGACY_INTERNAL_USE

    assert getattr(_LEGACY_INTERNAL_USE, "active", False) is False
    with legacy_predictor_internal_use():
        assert _LEGACY_INTERNAL_USE.active is True
        with legacy_predictor_internal_use():
            assert _LEGACY_INTERNAL_USE.active is True
        assert _LEGACY_INTERNAL_USE.active is True
    assert getattr(_LEGACY_INTERNAL_USE, "active", False) is False


@pytest.mark.skipif(
    not SINGLE_CKPT.exists(), reason="single_instance ckpt fixture not present"
)
def test_factory_from_model_paths_does_not_emit_legacy_deprecation():
    """Factory delegation must not leak the legacy module's DeprecationWarning."""
    from sleap_nn.inference.predictor import Predictor

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Predictor.from_model_paths(model_paths=[str(SINGLE_CKPT)], device="cpu")

    leaked = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "sleap_nn.inference.predictors" in str(w.message)
    ]
    assert not leaked, (
        f"factory leaked legacy deprecation warning: "
        f"{[str(w.message) for w in leaked]}"
    )
