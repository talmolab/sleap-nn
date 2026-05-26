"""Tests for the new ``Predictor`` orchestrator + ``Provider`` protocol.

Coverage:

1. ``NumpyProvider`` yields the right batch shapes / metadata.
2. ``Predictor.predict()`` runs end-to-end on a synthetic layer.
3. ``Predictor.predict_streaming()`` yields one ``Outputs`` per batch.
4. ``Predictor`` propagates the filter pipeline (default no-op +
   non-trivial ``FilterConfig``).
5. Frame / video indices from the provider land on the resulting
   ``Outputs`` (so downstream label conversion sees them).
6. ``make_labels=True`` requires ``skeleton`` (clear ``ValueError``).
7. ``Provider`` protocol — ``isinstance(numpy_provider, Provider)``
   returns ``True``.
8. Source dispatch: ``predict`` accepts ``sio.Video``, ``Provider``, etc.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from sleap_nn.inference.filters import FilterConfig
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.predictor import Predictor
from sleap_nn.inference.providers import Batch, NumpyProvider, Provider


class _StubLayer:
    """Minimal layer-shaped object: returns a fixed ``Outputs`` per call."""

    def predict(self, image, **kwargs) -> Outputs:
        """Return predictable keypoints derived from input shape."""
        b = image.shape[0]
        return Outputs(
            pred_keypoints=torch.zeros(b, 1, 4, 2),
            pred_peak_values=torch.ones(b, 1, 4),
            instance_scores=torch.ones(b, 1) * 0.9,
        )


# ─────────────────────────────────────────────────────────────────────────
# 1. NumpyProvider basics
# ─────────────────────────────────────────────────────────────────────────


def test_numpy_provider_yields_expected_batches():
    """N=10 images, batch_size=4 → 3 batches of (4, 4, 2)."""
    images = np.zeros((10, 1, 8, 8), dtype=np.uint8)
    provider = NumpyProvider(images=images, batch_size=4)
    assert len(provider) == 3
    batches = list(provider)
    assert len(batches) == 3
    assert batches[0].images.shape == (4, 1, 8, 8)
    assert batches[1].images.shape == (4, 1, 8, 8)
    assert batches[2].images.shape == (2, 1, 8, 8)
    assert np.array_equal(batches[0].frame_indices, [0, 1, 2, 3])
    assert np.array_equal(batches[2].frame_indices, [8, 9])


def test_numpy_provider_satisfies_provider_protocol():
    """``isinstance(provider, Provider)`` confirms the structural type."""
    provider = NumpyProvider(images=np.zeros((1, 1, 4, 4), dtype=np.uint8))
    assert isinstance(provider, Provider)


# ─────────────────────────────────────────────────────────────────────────
# 2. Predictor.predict end-to-end
# ─────────────────────────────────────────────────────────────────────────


def test_predictor_predict_returns_outputs_list():
    """``make_labels=False`` returns a list of ``Outputs``, one per batch."""
    images = np.zeros((6, 1, 8, 8), dtype=np.float32)
    provider = NumpyProvider(images=images, batch_size=2)
    predictor = Predictor(layer=_StubLayer())
    outputs_list = predictor.predict(provider, make_labels=False)
    assert isinstance(outputs_list, list)
    assert len(outputs_list) == 3
    assert all(isinstance(o, Outputs) for o in outputs_list)


# ─────────────────────────────────────────────────────────────────────────
# 3. predict_streaming yields one Outputs per batch
# ─────────────────────────────────────────────────────────────────────────


def test_predict_streaming_yields_outputs():
    """``predict_streaming`` is a generator that yields ``Outputs``."""
    images = np.zeros((5, 1, 8, 8), dtype=np.float32)
    provider = NumpyProvider(images=images, batch_size=2)
    predictor = Predictor(layer=_StubLayer())
    iter_ = predictor.predict_streaming(provider)
    first = next(iter_)
    assert isinstance(first, Outputs)
    rest = list(iter_)
    assert len(rest) == 2  # 5 frames / batch=2 → 3 batches total


# ─────────────────────────────────────────────────────────────────────────
# 4. FilterPipeline propagation
# ─────────────────────────────────────────────────────────────────────────


def test_predictor_applies_filter_config():
    """A non-trivial ``FilterConfig`` filters the ``Outputs`` per batch."""
    images = np.zeros((2, 1, 8, 8), dtype=np.float32)
    provider = NumpyProvider(images=images, batch_size=2)
    predictor = Predictor(
        layer=_StubLayer(),
        filter_config=FilterConfig(min_instance_score=0.95),
    )
    out = predictor.predict(provider, make_labels=False)[0]
    assert torch.isnan(out.pred_keypoints).all()


def test_default_filter_is_noop():
    """Default ``FilterConfig`` doesn't touch the ``Outputs``."""
    images = np.zeros((2, 1, 8, 8), dtype=np.float32)
    provider = NumpyProvider(images=images, batch_size=2)
    predictor = Predictor(layer=_StubLayer())
    out = predictor.predict(provider, make_labels=False)[0]
    assert not torch.isnan(out.pred_keypoints).any()


# ─────────────────────────────────────────────────────────────────────────
# 5. Frame / video indices land on Outputs
# ─────────────────────────────────────────────────────────────────────────


def test_metadata_propagates_from_provider():
    """Per-batch ``frame_indices`` / ``video_indices`` end up on ``Outputs``."""
    images = np.zeros((4, 1, 8, 8), dtype=np.float32)
    provider = NumpyProvider(
        images=images,
        batch_size=2,
        frame_indices=np.array([10, 11, 12, 13], dtype=np.int64),
        video_indices=np.array([0, 0, 1, 1], dtype=np.int64),
    )
    predictor = Predictor(layer=_StubLayer())
    out_list = predictor.predict(provider, make_labels=False)
    assert torch.equal(out_list[0].frame_indices, torch.tensor([10, 11]))
    assert torch.equal(out_list[1].video_indices, torch.tensor([1, 1]))


# ─────────────────────────────────────────────────────────────────────────
# 6. make_labels requires skeleton
# ─────────────────────────────────────────────────────────────────────────


def test_make_labels_without_skeleton_raises():
    images = np.zeros((1, 1, 8, 8), dtype=np.float32)
    provider = NumpyProvider(images=images, batch_size=1)
    predictor = Predictor(layer=_StubLayer())
    with pytest.raises(ValueError, match="skeleton"):
        predictor.predict(provider, make_labels=True)


def test_make_labels_returns_sio_labels():
    """``make_labels=True`` with a skeleton returns an ``sio.Labels``."""
    import sleap_io as sio

    skel = sio.Skeleton(nodes=[sio.Node(name=f"n{i}") for i in range(4)])
    images = np.zeros((2, 1, 8, 8), dtype=np.float32)
    provider = NumpyProvider(images=images, batch_size=2)
    predictor = Predictor(layer=_StubLayer())
    labels = predictor.predict(provider, make_labels=True, skeleton=skel)
    assert isinstance(labels, sio.Labels)


def test_make_labels_uses_predictor_skeleton():
    """``make_labels=True`` without explicit skeleton uses ``self.skeleton``."""
    import sleap_io as sio

    skel = sio.Skeleton(nodes=[sio.Node(name=f"n{i}") for i in range(4)])
    images = np.zeros((2, 1, 8, 8), dtype=np.float32)
    provider = NumpyProvider(images=images, batch_size=2)
    predictor = Predictor(layer=_StubLayer(), skeleton=skel)
    labels = predictor.predict(provider)
    assert isinstance(labels, sio.Labels)
    assert labels.skeletons[0] is skel


# ─────────────────────────────────────────────────────────────────────────
# 8. batch_size stored on Predictor
# ─────────────────────────────────────────────────────────────────────────


def test_batch_size_stored_on_predictor():
    predictor = Predictor(layer=_StubLayer(), batch_size=8)
    assert predictor.batch_size == 8


def test_batch_size_default():
    predictor = Predictor(layer=_StubLayer())
    assert predictor.batch_size == 4
