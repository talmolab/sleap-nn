"""Top-level public API surface: ``sleap_nn.{predict, Predictor, load_models}``."""

from unittest.mock import patch

import pytest

import sleap_nn


def test_top_level_names_are_discoverable():
    """The high-level entry points are advertised in ``__all__`` and ``dir()``."""
    for name in ("predict", "Predictor", "load_models"):
        assert name in sleap_nn.__all__
        assert name in dir(sleap_nn)


def test_predict_and_predictor_resolve_to_inference():
    """``sleap_nn.predict`` / ``Predictor`` are the inference function / class."""
    from sleap_nn import inference

    assert sleap_nn.predict is inference.predict
    assert sleap_nn.Predictor is inference.Predictor
    assert callable(sleap_nn.predict)


def test_predict_stays_callable_after_legacy_import():
    """Regression: importing the legacy module must not shadow ``sleap_nn.predict``.

    Before ``sleap_nn/predict.py`` was renamed to ``legacy_predict.py``,
    importing ``sleap_nn.train`` (which pulls in the old ``sleap_nn.predict``
    module) rebound ``sleap_nn.predict`` to the *module*, breaking
    ``sleap_nn.predict(...)``.
    """
    import sleap_nn.train  # noqa: F401  (imports sleap_nn.legacy_predict)

    assert callable(sleap_nn.predict)
    from sleap_nn import inference

    assert sleap_nn.predict is inference.predict


def test_load_models_wraps_from_model_paths():
    """``sleap_nn.load_models`` forwards to ``Predictor.from_model_paths``."""
    sentinel = object()
    with patch(
        "sleap_nn.inference.Predictor.from_model_paths",
        return_value=sentinel,
    ) as mock_fmp:
        result = sleap_nn.load_models(["/m1", "/m2"], device="cuda", batch_size=8)
    assert result is sentinel
    mock_fmp.assert_called_once_with(["/m1", "/m2"], device="cuda", batch_size=8)


def test_unknown_attribute_raises():
    """``__getattr__`` still raises ``AttributeError`` for unknown names."""
    with pytest.raises(AttributeError):
        sleap_nn.totally_made_up_name  # noqa: B018
