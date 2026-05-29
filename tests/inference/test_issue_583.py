"""Regression tests for issue #583 (CLI/streaming/feature follow-ups).

Library-level coverage (CLI-level tests live in tests/cli/test_infer_command.py):

* Predict-time ``return_*`` intermediate-tensor overrides via
  ``Predictor._postprocess_overrides``.
* Streaming pool primitives (``PafGroupingPool.__len__`` / ``drain_one``) used
  by the bounded pipelined-streaming path.
* Streaming writer accumulate-until-finalize contract + provenance persistence.
* The Rich progress-callback factory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sleap_io as sio
import torch

from sleap_nn.inference.layers.configs import PostprocessConfig
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.predictor import Predictor
from sleap_nn.inference.writer import IncrementalLabelsWriter

CKPT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "model_ckpts"
DATA_ROOT = Path(__file__).resolve().parents[1] / "assets" / "datasets"
MC_TOPDOWN_CKPTS = [
    CKPT_ROOT / "minimal_instance_centroid",
    CKPT_ROOT / "minimal_instance_multiclass_centered_instance",
]
MC_TOPDOWN_VIDEO = DATA_ROOT / "centered_pair_small.mp4"

# ─────────────────────────────────────────────────────────────────────────
# Predict-time return_* overrides (gap)
# ─────────────────────────────────────────────────────────────────────────


class _PostprocStub:
    """Minimal layer with a ``postprocess_config`` (so it is an override target)."""

    def __init__(self):
        self.postprocess_config = PostprocessConfig()


def test_postprocess_overrides_thread_return_flags():
    """Each return_* flag flips the layer's PostprocessConfig and restores (#583)."""
    for name in (
        "return_pafs",
        "return_paf_graph",
        "return_class_maps",
        "return_class_vectors",
    ):
        predictor = Predictor(layer=_PostprocStub())
        assert getattr(predictor.layer.postprocess_config, name) is False
        with predictor._postprocess_overrides(**{name: True}):
            assert getattr(predictor.layer.postprocess_config, name) is True
        # Restored on exit.
        assert getattr(predictor.layer.postprocess_config, name) is False


def test_postprocess_overrides_no_args_is_noop():
    """With no overrides the config object is untouched (has_any short-circuit)."""
    predictor = Predictor(layer=_PostprocStub())
    before = predictor.layer.postprocess_config
    with predictor._postprocess_overrides():
        pass
    assert predictor.layer.postprocess_config is before


@pytest.mark.skipif(
    not (all(p.exists() for p in MC_TOPDOWN_CKPTS) and MC_TOPDOWN_VIDEO.exists()),
    reason="multiclass top-down checkpoints / video not present",
)
def test_return_class_vectors_propagates_through_topdown_multiclass():
    """return_class_vectors reaches Outputs for the composed multi-class top-down.

    The flag flips the config but was dropped at the TopDownLayer composition;
    _run_stage_2 now scatters the per-crop class vectors (#583 review).
    """
    from sleap_nn.inference.providers import VideoProvider

    predictor = Predictor.from_model_paths(
        [str(p) for p in MC_TOPDOWN_CKPTS],
        device="cpu",
        peak_threshold=0.03,
        max_instances=6,
    )
    outs_on = list(
        predictor.predict_streaming(
            VideoProvider(video=str(MC_TOPDOWN_VIDEO), batch_size=2, frames=[0, 1]),
            return_class_vectors=True,
        )
    )
    assert any(o.pred_class_vectors is not None for o in outs_on)
    # Default (no override) leaves it unset.
    outs_off = list(
        predictor.predict_streaming(
            VideoProvider(video=str(MC_TOPDOWN_VIDEO), batch_size=2, frames=[0, 1])
        )
    )
    assert all(o.pred_class_vectors is None for o in outs_off)


# ─────────────────────────────────────────────────────────────────────────
# Streaming pool primitives (bounded pipelined streaming)
# ─────────────────────────────────────────────────────────────────────────


def test_paf_grouping_pool_len_and_drain_one_fifo():
    """``__len__`` reflects pending depth; ``drain_one`` pops FIFO; None when empty."""
    from concurrent.futures import Future

    from sleap_nn.inference.streaming import GroupingParams, PafGroupingPool

    pool = PafGroupingPool(
        n_workers=1, grouping_params=GroupingParams(paf_scorer_kwargs={})
    )
    # Seed the pending queue with already-resolved futures (no executor needed):
    # drain_one only pops the oldest and reads .result().
    for i in range(3):
        fut: Future = Future()
        fut.set_result(f"out{i}")
        pool._pending.append((i, fut))

    assert len(pool) == 3
    assert pool.drain_one() == (0, "out0")
    assert pool.drain_one() == (1, "out1")
    assert len(pool) == 1
    assert pool.drain_one() == (2, "out2")
    assert len(pool) == 0
    assert pool.drain_one() is None


# ─────────────────────────────────────────────────────────────────────────
# Streaming writer: accumulate-until-finalize + provenance persistence
# ─────────────────────────────────────────────────────────────────────────


def _one_instance_outputs(frame_idx: int) -> Outputs:
    return Outputs(
        pred_keypoints=torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
        pred_peak_values=torch.tensor([[[0.9, 0.9]]]),
        frame_indices=torch.tensor([frame_idx]),
        video_indices=torch.tensor([0]),
    )


def test_incremental_writer_accumulates_until_finalize(tmp_path):
    """flush != disk-write today: the .slp appears only at close (#583 docs)."""
    skel = sio.Skeleton(nodes=[sio.Node(name="n0"), sio.Node(name="n1")])
    out_path = tmp_path / "out.slp"
    writer = IncrementalLabelsWriter(
        path=str(out_path), skeleton=skel, write_interval=1
    )
    with writer:
        for i in range(3):
            writer.write(_one_instance_outputs(i))
            # No final file mid-stream (write happens once at finalize).
            assert not out_path.exists()
        assert writer.frame_count == 3
    assert out_path.exists()
    assert len(sio.load_slp(str(out_path)).labeled_frames) == 3


def test_incremental_writer_persists_provenance(tmp_path):
    """A provenance dict set on the writer round-trips into the saved .slp (#583)."""
    skel = sio.Skeleton(nodes=[sio.Node(name="n0"), sio.Node(name="n1")])
    out_path = tmp_path / "out.slp"
    writer = IncrementalLabelsWriter(
        path=str(out_path),
        skeleton=skel,
        provenance={"sleap_nn_version": "x.y.z", "model_paths": ["/m"]},
    )
    with writer:
        writer.write(_one_instance_outputs(0))
    labels = sio.load_slp(str(out_path))
    assert labels.provenance.get("sleap_nn_version") == "x.y.z"
    assert labels.provenance.get("model_paths") == ["/m"]


# ─────────────────────────────────────────────────────────────────────────
# Rich progress callback factory
# ─────────────────────────────────────────────────────────────────────────


def test_rich_progress_callback_returns_callable_and_progress():
    """The factory yields a (callback, progress) pair; callback drives cleanly."""
    from sleap_nn.cli import _rich_progress_callback

    cb, progress = _rich_progress_callback()
    try:
        assert callable(cb)
        # Unknown total (length-less provider) must not raise.
        cb(1, -1)
        cb(2, -1)
        # Known total path.
        cb(1, 4)
        cb(4, 4)
    finally:
        progress.stop()


# ─────────────────────────────────────────────────────────────────────────
# --video_index scoping helper (composes with suggestions + frames; #583 review)
# ─────────────────────────────────────────────────────────────────────────


def test_scope_labels_to_video_carries_suggestions_frames_provenance():
    """The scoping helper keeps the video's suggestions, applies --frames, and
    preserves source provenance (review bug 66c)."""
    from sleap_nn.cli import _scope_labels_to_video

    skel = sio.Skeleton(nodes=["a", "b"])
    v0, v1 = sio.Video(filename="a.mp4"), sio.Video(filename="b.mp4")
    suggestions = [
        sio.SuggestionFrame(video=v1, frame_idx=2),
        sio.SuggestionFrame(video=v0, frame_idx=0),
        sio.SuggestionFrame(video=v1, frame_idx=9),
    ]
    labels = sio.Labels(
        videos=[v0, v1],
        skeletons=[skel],
        labeled_frames=[],
        suggestions=suggestions,
        provenance={"filename": "/orig.slp"},
    )
    scoped, target = _scope_labels_to_video(labels, 1, frames=[2])
    assert target is v1
    assert list(scoped.videos) == [v1]
    # v1/frame2 kept; v1/frame9 dropped by --frames; v0 suggestion excluded.
    assert [s.frame_idx for s in scoped.suggestions] == [2]
    assert scoped.provenance.get("filename") == "/orig.slp"


def test_scope_labels_to_video_out_of_range_raises():
    """An out-of-range video_index raises a click UsageError."""
    import click

    from sleap_nn.cli import _scope_labels_to_video

    skel = sio.Skeleton(nodes=["a", "b"])
    labels = sio.Labels(
        videos=[sio.Video(filename="a.mp4")], skeletons=[skel], labeled_frames=[]
    )
    with __import__("pytest").raises(click.UsageError):
        _scope_labels_to_video(labels, 5)
