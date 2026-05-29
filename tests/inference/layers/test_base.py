"""Tests for the ``InferenceLayer`` ABC contract + image-coercion helper."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from sleap_nn.inference.layers.backends import TorchBackend
from sleap_nn.inference.layers.base import InferenceLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.preprocess_info import PreprocInfo

# ─────────────────────────────────────────────────────────────────────────
# A trivial concrete layer for testing the ABC machinery
# ─────────────────────────────────────────────────────────────────────────


class _NoopModel(nn.Module):
    def forward(self, x):  # noqa: D401
        """Identity forward — return the input tensor unchanged."""
        return x.float()


class _IdentityLayer(InferenceLayer):
    """Concrete layer used only by ABC tests — passes input through."""

    def preprocess(self, image):
        x = self._to_4d_float_tensor(image)
        info = PreprocInfo(
            original_size=tuple(x.shape[2:]),
            processed_size=tuple(x.shape[2:]),
            eff_scale=torch.ones(x.shape[0]),
        )
        return x, info

    def postprocess(self, raw_out, info):
        return Outputs(
            preprocess_info=info,
            pred_keypoints=torch.zeros(1, 1, 1, 2),
        )


def _new_identity_layer():
    return _IdentityLayer(
        backend=TorchBackend(model=_NoopModel(), device="cpu"),
        preprocess_config=PreprocessConfig(),
        postprocess_config=PostprocessConfig(),
        output_stride=1,
    )


# ─────────────────────────────────────────────────────────────────────────
# 1. ABC enforcement
# ─────────────────────────────────────────────────────────────────────────


def test_inference_layer_cannot_be_instantiated_directly():
    """``InferenceLayer`` is an ABC — direct instantiation must fail."""
    with pytest.raises(TypeError):
        InferenceLayer(  # type: ignore[abstract]
            backend=TorchBackend(model=_NoopModel(), device="cpu"),
            preprocess_config=PreprocessConfig(),
            postprocess_config=PostprocessConfig(),
            output_stride=1,
        )


def test_layer_rejects_non_backend():
    """Construction validates the backend satisfies the protocol."""
    with pytest.raises(TypeError, match="ModelBackend"):
        _IdentityLayer(  # type: ignore[arg-type]
            backend="not a backend",
            preprocess_config=PreprocessConfig(),
            postprocess_config=PostprocessConfig(),
            output_stride=1,
        )


# ─────────────────────────────────────────────────────────────────────────
# 2. predict() and __call__() agreement
# ─────────────────────────────────────────────────────────────────────────


def test_predict_returns_outputs_instance():
    layer = _new_identity_layer()
    out = layer.predict(np.zeros((4, 4), dtype=np.uint8))
    assert isinstance(out, Outputs)


def test_call_is_alias_for_predict():
    layer = _new_identity_layer()
    a = layer.predict(np.zeros((4, 4), dtype=np.uint8))
    b = layer(np.zeros((4, 4), dtype=np.uint8))
    assert isinstance(a, Outputs) and isinstance(b, Outputs)


# ─────────────────────────────────────────────────────────────────────────
# 3. _to_4d_float_tensor — every accepted input shape
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "in_shape,expected_shape",
    [
        ((8, 8), (1, 1, 8, 8)),  # (H, W) grayscale
        ((8, 8, 3), (1, 3, 8, 8)),  # (H, W, C) channel-last
        ((3, 8, 8), (1, 3, 8, 8)),  # (C, H, W) channel-first single
        ((2, 8, 8, 3), (2, 3, 8, 8)),  # (B, H, W, C)
        ((2, 3, 8, 8), (2, 3, 8, 8)),  # (B, C, H, W)
    ],
)
def test_to_4d_float_tensor_shapes(in_shape, expected_shape):
    arr = np.zeros(in_shape, dtype=np.uint8)
    out = InferenceLayer._to_4d_float_tensor(arr)
    assert out.shape == expected_shape
    assert out.dtype == torch.float32


def test_to_4d_float_tensor_accepts_torch():
    t = torch.zeros((8, 8), dtype=torch.uint8)
    out = InferenceLayer._to_4d_float_tensor(t)
    assert out.shape == (1, 1, 8, 8)
    assert out.dtype == torch.float32


def test_to_4d_float_tensor_rejects_other_types():
    with pytest.raises(TypeError, match="np.ndarray or torch.Tensor"):
        InferenceLayer._to_4d_float_tensor("not an image")  # type: ignore[arg-type]


def test_to_4d_float_tensor_rejects_weird_ranks():
    with pytest.raises(ValueError, match="unexpected image rank"):
        InferenceLayer._to_4d_float_tensor(np.zeros((1, 1, 1, 1, 1)))


# ─────────────────────────────────────────────────────────────────────────
# 4. Warmup
# ─────────────────────────────────────────────────────────────────────────


def test_warmup_uses_default_shape():
    layer = _new_identity_layer()
    # CPU warmup is a no-op in TorchBackend, so this just exercises the path.
    layer.warmup()


def test_warmup_accepts_explicit_shape():
    layer = _new_identity_layer()
    layer.warmup((1, 1, 32, 32))
