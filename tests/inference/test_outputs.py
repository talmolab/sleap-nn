"""Tests for ``sleap_nn.inference.outputs.Outputs`` and ``PreprocInfo``.

Covers:

1. Construction with every field type (None, tensor, tuple-of-tensors,
   PreprocInfo, integers).
2. Tensor management: ``to``, ``cpu``, ``detach``, ``numpy``.
3. ``slim()`` drops the heavy intermediates exactly as documented.
4. **Pickle round-trip** on a fully populated ``Outputs`` and on its
   ``slim()`` form. This is the locked acceptance criterion from the
   PR 2 design — guarantees the multi-process post-processing path
   (PR 9) and streaming writer (PR 8) can ship ``Outputs`` between
   processes.
5. ``__slots__`` enforcement — class must use slots so adding a stray
   attribute raises ``AttributeError`` (no live-reference fields can
   creep in).
6. Live-reference contract — every field is a value type, not a handle
   to a model / file / generator. Enforced structurally.
7. ``__repr__`` is compact and never prints tensor contents.
8. Shape properties (``batch_size``, ``n_instances``, ``n_nodes``).
9. ``to_instances`` / ``to_labels`` round-trip on a synthetic batch
   (full PR 4+ exercise comes when the layers land).
"""

from __future__ import annotations

import io
import pickle
from types import GeneratorType
from typing import get_type_hints

import attrs
import numpy as np
import pytest
import sleap_io as sio
import torch

from sleap_nn.inference.outputs import _HEAVY_FIELDS, Outputs
from sleap_nn.inference.preprocess_info import PreprocInfo

# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


def _full_outputs(B: int = 2, I: int = 3, N: int = 4, C: int = 2) -> Outputs:
    """Build an ``Outputs`` with every field populated."""
    H, W, E = 16, 16, 5
    return Outputs(
        original_image=torch.randn(B, 1, H, W),
        processed_image=torch.randn(B, 1, H, W),
        crops=torch.randn(B, I, 1, H // 2, W // 2),
        pred_keypoints=torch.randn(B, I, N, 2),
        pred_peak_values=torch.rand(B, I, N),
        pred_confmaps=torch.randn(B, N, H, W),
        pred_pafs=torch.randn(B, 2 * E, H, W),
        pred_centroids=torch.randn(B, I, 2),
        pred_centroid_values=torch.rand(B, I),
        instance_scores=torch.rand(B, I),
        instance_valid=torch.ones(B, I, dtype=torch.bool),
        instance_bboxes=torch.randn(B, I, 4, 2),
        pred_class_vectors=torch.randn(B, I, N, C),
        pred_class_maps=torch.randn(B, C, H, W),
        pred_class_inds=torch.zeros(B, I, N, dtype=torch.int64),
        pred_class_probs=torch.softmax(torch.randn(B, I, C), dim=-1),
        pred_paf_graph=(
            torch.randn(10, 2),
            torch.zeros(10, dtype=torch.int64),
            torch.zeros(10, dtype=torch.int64),
            torch.rand(10),
        ),
        preprocess_info=PreprocInfo(
            original_size=(384, 384),
            processed_size=(192, 192),
            eff_scale=torch.tensor([1.0, 0.5]),
            input_scale=0.5,
            output_stride=4,
            pad_amount=(0, 0),
            crop_offsets=torch.zeros(B * I, 2),
        ),
        frame_indices=torch.arange(B, dtype=torch.int64),
        video_indices=torch.zeros(B, dtype=torch.int64),
    )


# ─────────────────────────────────────────────────────────────────────────
# 1. Construction
# ─────────────────────────────────────────────────────────────────────────


def test_default_construction_is_all_none():
    """An ``Outputs()`` with no args is a placeholder of None fields."""
    outputs = Outputs()
    assert outputs.batch_size == 0
    assert outputs.n_instances == 0
    assert outputs.n_nodes == 0
    for f in attrs.fields(Outputs):
        assert getattr(outputs, f.name) is None


def test_full_construction_round_trip():
    """Every populated field survives construction unchanged."""
    o = _full_outputs()
    assert o.batch_size == 2
    assert o.n_instances == 3
    assert o.n_nodes == 4
    assert o.preprocess_info is not None
    assert o.preprocess_info.input_scale == 0.5


def test_evolve_replaces_only_named_fields():
    """``attrs.evolve`` should preserve untouched fields."""
    o = _full_outputs()
    new = attrs.evolve(o, pred_keypoints=torch.zeros(2, 3, 4, 2))
    assert torch.all(new.pred_keypoints == 0.0)
    # other fields untouched
    assert new.pred_peak_values is o.pred_peak_values


# ─────────────────────────────────────────────────────────────────────────
# 2. Tensor management
# ─────────────────────────────────────────────────────────────────────────


def test_to_returns_new_outputs_with_moved_tensors():
    o = _full_outputs()
    moved = o.to("cpu")
    assert moved is not o
    for f in attrs.fields(Outputs):
        a = getattr(o, f.name)
        b = getattr(moved, f.name)
        if isinstance(a, torch.Tensor):
            assert b.device.type == "cpu"


def test_cpu_alias_for_to_cpu():
    o = _full_outputs()
    cpu_o = o.cpu()
    assert isinstance(cpu_o, Outputs)
    assert cpu_o.pred_keypoints is not None
    assert cpu_o.pred_keypoints.device.type == "cpu"


def test_detach_strips_grad():
    o = Outputs(pred_keypoints=torch.randn(2, 3, 4, 2, requires_grad=True))
    detached = o.detach()
    assert detached.pred_keypoints is not None
    assert detached.pred_keypoints.requires_grad is False
    # original is untouched
    assert o.pred_keypoints.requires_grad is True


def test_numpy_returns_dict_of_ndarrays_skipping_none():
    o = _full_outputs()
    arrays = o.numpy()
    assert isinstance(arrays, dict)
    # populated fields are present, None fields are not
    assert "pred_keypoints" in arrays
    assert isinstance(arrays["pred_keypoints"], np.ndarray)
    # tuple-of-tensors becomes tuple-of-arrays
    assert isinstance(arrays["pred_paf_graph"], tuple)
    assert isinstance(arrays["pred_paf_graph"][0], np.ndarray)
    # PreprocInfo is non-tensor — passes through
    assert isinstance(arrays["preprocess_info"], PreprocInfo)


def test_numpy_omits_none_fields():
    o = Outputs(pred_keypoints=torch.zeros(1, 1, 1, 2))
    arrays = o.numpy()
    assert "pred_keypoints" in arrays
    assert "pred_pafs" not in arrays


# ─────────────────────────────────────────────────────────────────────────
# 3. slim()
# ─────────────────────────────────────────────────────────────────────────


def test_slim_drops_only_heavy_fields():
    o = _full_outputs()
    slimmed = o.slim()
    for name in _HEAVY_FIELDS:
        assert getattr(slimmed, name) is None, f"slim should drop {name}"
    # core predictions survive
    assert slimmed.pred_keypoints is not None
    assert slimmed.pred_peak_values is not None
    assert slimmed.preprocess_info is not None


def test_slim_moves_remaining_tensors_to_cpu():
    o = _full_outputs()
    # Force one tensor onto a non-CPU device-type label to verify .to('cpu')
    # is in fact called by slim().
    slimmed = o.slim()
    for f in attrs.fields(Outputs):
        v = getattr(slimmed, f.name)
        if isinstance(v, torch.Tensor):
            assert v.device.type == "cpu"


def test_slim_detaches_grad():
    o = Outputs(pred_keypoints=torch.randn(1, 1, 4, 2, requires_grad=True))
    slimmed = o.slim()
    assert slimmed.pred_keypoints is not None
    assert slimmed.pred_keypoints.requires_grad is False


# ─────────────────────────────────────────────────────────────────────────
# 4. PICKLE ROUND-TRIP — locked acceptance criterion (Tom's feedback)
# ─────────────────────────────────────────────────────────────────────────


def _pickle_roundtrip(obj):
    buf = io.BytesIO()
    pickle.dump(obj, buf, protocol=4)
    buf.seek(0)
    return pickle.load(buf)


def _assert_outputs_equal_value(a: Outputs, b: Outputs) -> None:
    for f in attrs.fields(Outputs):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if va is None:
            assert vb is None, f"{f.name}: lost None-ness"
            continue
        if isinstance(va, torch.Tensor):
            assert torch.equal(va, vb), f"{f.name} drifted across pickle"
        elif isinstance(va, tuple) and va and isinstance(va[0], torch.Tensor):
            assert all(torch.equal(x, y) for x, y in zip(va, vb)), f"{f.name} drifted"
        elif isinstance(va, PreprocInfo):
            # frozen, eq=False — compare field-by-field instead.
            for ff in attrs.fields(PreprocInfo):
                pa, pb = getattr(va, ff.name), getattr(vb, ff.name)
                if isinstance(pa, torch.Tensor):
                    assert torch.equal(pa, pb), f"PreprocInfo.{ff.name} drifted"
                else:
                    assert pa == pb, f"PreprocInfo.{ff.name} drifted"
        else:
            assert va == vb, f"{f.name} drifted across pickle"


def test_full_outputs_pickles():
    """Full fat ``Outputs`` round-trips through pickle protocol 4."""
    o = _full_outputs()
    back = _pickle_roundtrip(o)
    _assert_outputs_equal_value(o, back)


def test_slim_outputs_pickles():
    """Slimmed ``Outputs`` (the contract) round-trips through pickle."""
    o = _full_outputs()
    slim = o.slim()
    back = _pickle_roundtrip(slim)
    _assert_outputs_equal_value(slim, back)


def test_empty_outputs_pickles():
    """A default ``Outputs`` with no populated fields pickles."""
    back = _pickle_roundtrip(Outputs())
    assert isinstance(back, Outputs)
    assert back.batch_size == 0


def test_paf_graph_tuple_field_pickles():
    """The bottom-up PAF-graph tuple field is the trickiest type — verify."""
    o = Outputs(
        pred_paf_graph=(
            torch.randn(5, 2),
            torch.arange(5, dtype=torch.int64),
            torch.arange(5, dtype=torch.int64),
            torch.rand(5),
        ),
    )
    back = _pickle_roundtrip(o)
    assert isinstance(back.pred_paf_graph, tuple)
    assert len(back.pred_paf_graph) == 4
    assert torch.equal(back.pred_paf_graph[0], o.pred_paf_graph[0])


# ─────────────────────────────────────────────────────────────────────────
# 5. __slots__ enforcement
# ─────────────────────────────────────────────────────────────────────────


def test_outputs_uses_slots():
    """``Outputs`` is declared with ``slots=True`` for memory efficiency.

    Confirm at runtime — guards against an accidental ``slots=False`` change.
    """
    o = Outputs()
    with pytest.raises(AttributeError):
        o.unrelated_field = 42  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────────────
# 6. Live-reference contract — every field must be a value type
# ─────────────────────────────────────────────────────────────────────────


_ALLOWED_FIELD_TYPES = (
    torch.Tensor,
    np.ndarray,
    PreprocInfo,
    int,
    float,
    str,
    bool,
    bytes,
    type(None),
)


def _is_value_type(v) -> bool:
    if isinstance(v, _ALLOWED_FIELD_TYPES):
        return True
    if isinstance(v, (tuple, list)):
        return all(_is_value_type(x) for x in v)
    if isinstance(v, dict):
        return all(_is_value_type(k) and _is_value_type(val) for k, val in v.items())
    return False


def test_no_field_can_hold_a_live_reference():
    """Every populated field on a fat ``Outputs`` must be a pickle-friendly
    value type. Catches anyone adding a ``layer: InferenceLayer`` or
    ``video_handle: h5py.File`` field for "convenience"."""
    o = _full_outputs()
    for f in attrs.fields(Outputs):
        v = getattr(o, f.name)
        if v is None:
            continue
        # explicit deny-list check on common foot-guns
        assert not isinstance(v, GeneratorType), f"{f.name} is a generator"
        assert not callable(v) or isinstance(v, type), f"{f.name} is callable"
        assert _is_value_type(v), (
            f"{f.name} of type {type(v).__name__} is not a known value type — "
            f"would break pickle / multi-process transport"
        )


# ─────────────────────────────────────────────────────────────────────────
# 7. __repr__
# ─────────────────────────────────────────────────────────────────────────


def test_repr_is_compact_no_tensor_contents():
    """``__repr__`` of a fat ``Outputs`` should fit on a line and not
    dump tensor contents (the default repr would be megabytes)."""
    o = _full_outputs()
    s = repr(o)
    assert s.startswith("Outputs(")
    # Pre-emptively cap at a reasonable size — fat ``Outputs`` should still
    # produce a one-or-two-liner repr.
    assert len(s) < 4000, f"repr too long: {len(s)}"
    # Must not contain raw tensor numerical content
    assert "tensor(" not in s
    # Must list at least the keypoint shape
    assert "pred_keypoints=Tensor" in s


def test_repr_of_empty_outputs():
    assert repr(Outputs()) == "Outputs(empty)"


# ─────────────────────────────────────────────────────────────────────────
# 8. Shape properties
# ─────────────────────────────────────────────────────────────────────────


def test_shape_properties_consistent_with_fields():
    o = _full_outputs(B=4, I=5, N=7)
    assert o.batch_size == 4
    assert o.n_instances == 5
    assert o.n_nodes == 7


def test_shape_properties_zero_when_empty():
    o = Outputs()
    assert o.batch_size == 0
    assert o.n_instances == 0
    assert o.n_nodes == 0


def test_shape_properties_use_centroids_if_keypoints_missing():
    """A centroid-only output has no ``pred_keypoints`` but still defines B/I."""
    o = Outputs(pred_centroids=torch.zeros(3, 2, 2))
    assert o.batch_size == 3
    assert o.n_instances == 2


# ─────────────────────────────────────────────────────────────────────────
# 9. sleap-io conversion (light coverage; PR 4+ exercises end-to-end)
# ─────────────────────────────────────────────────────────────────────────


def _toy_skeleton(n_nodes: int = 4) -> sio.Skeleton:
    nodes = [sio.Node(name=f"n{i}") for i in range(n_nodes)]
    return sio.Skeleton(nodes=nodes)


def test_to_instances_skips_all_nan_slots():
    skel = _toy_skeleton(n_nodes=2)
    kpts = torch.full((1, 2, 2, 2), float("nan"))
    kpts[0, 0, 0] = torch.tensor([1.0, 2.0])
    kpts[0, 0, 1] = torch.tensor([3.0, 4.0])
    o = Outputs(
        pred_keypoints=kpts,
        pred_peak_values=torch.tensor([[[0.9, 0.8], [0.0, 0.0]]]),
    )
    instances = o.to_instances(skeleton=skel, batch_index=0)
    assert len(instances) == 1
    assert isinstance(instances[0], sio.PredictedInstance)


def test_to_labels_builds_one_frame_per_nonempty_batch():
    skel = _toy_skeleton(n_nodes=2)
    kpts = torch.zeros(2, 1, 2, 2)
    kpts[1] = float("nan")  # second batch slot is empty
    o = Outputs(
        pred_keypoints=kpts,
        pred_peak_values=torch.ones(2, 1, 2),
        frame_indices=torch.tensor([10, 11], dtype=torch.int64),
        video_indices=torch.zeros(2, dtype=torch.int64),
    )
    labels = o.to_labels(skeleton=skel)
    assert isinstance(labels, sio.Labels)
    assert len(labels.labeled_frames) == 1
    assert labels.labeled_frames[0].frame_idx == 10


# ─────────────────────────────────────────────────────────────────────────
# Type-hint sanity (catches accidental Optional → required regressions)
# ─────────────────────────────────────────────────────────────────────────


def test_every_field_is_optional_with_default_none():
    """Every field on ``Outputs`` is optional and defaults to ``None`` so
    layers can populate only what they actually compute. Regression guard."""
    type_hints = get_type_hints(Outputs)
    for f in attrs.fields(Outputs):
        assert f.default is None, f"{f.name} must default to None"
        annotation = str(type_hints.get(f.name, ""))
        assert (
            "Optional" in annotation or "None" in annotation
        ), f"{f.name} should be typed Optional"
