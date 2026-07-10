"""WF2: embed (including untracked) detections + track by appearance, in one run.

Covers the second tracker-integration workflow: ``sleap-nn predict --model_paths
<embedding_model> --data_path <detections.slp> --tracking``. The embedding model
embeds EVERY detection (tracked or not — ``EmbeddingDataset(include_untracked=True)``),
attaches the vectors, and :func:`apply_tracking` assigns ``sio.Track``s by cosine
similarity. ``--save_embeddings`` decides whether the vectors persist in the tracked
``.slp`` (default ``none`` = tracks only; ``slp`` keeps the vectors).

Reuses the tiny random-weights embedding model + crop config from
``test_embedding_persistence`` (the canonical embedding inference fixtures).
"""

import os

import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.data.custom_datasets import EmbeddingDataset
from sleap_nn.inference.embedding import predict_embeddings_to_slp
from sleap_nn.inference.tracking import TrackerConfig

# Reuse the canonical embedding fixtures/helpers (model dir, video writer, config).
from tests.inference.test_embedding_persistence import (  # noqa: F401
    _DIM,
    _build_embedding_config,
    _write_video,
    embedding_model_dir,
)

# ── fixtures: UNTRACKED pose / mask .slp (the WF2 input) ─────────────────────────


@pytest.fixture
def untracked_pose_slp(tmp_path):
    """A pose ``.slp`` with NO tracks: 4 frames x 2 instances (positions swap)."""
    vid = _write_video(tmp_path / "vid.mp4", n=4)
    skel = sio.Skeleton(nodes=["a", "b"])
    lfs = []
    for fi in range(4):
        xa, xb = (16.0, 44.0) if fi % 2 == 0 else (44.0, 16.0)
        insts = [
            sio.PredictedInstance.from_numpy(
                np.array([[x, 20.0], [x + 6, 28.0]]), skeleton=skel, score=0.9
            )
            for x in (xa, xb)
        ]
        lfs.append(sio.LabeledFrame(video=vid, frame_idx=fi, instances=insts))
    labels = sio.Labels(labeled_frames=lfs, videos=[vid], skeletons=[skel])  # no tracks
    sp = str(tmp_path / "untracked_pose.slp")
    sio.save_slp(labels, sp, embed=False)
    return sp


@pytest.fixture
def untracked_mask_slp(tmp_path):
    """A mask-only ``.slp`` with NO tracks: 4 frames x 2 disks."""
    vid = _write_video(tmp_path / "mvid.mp4", n=4)
    yy, xx = np.ogrid[:64, :64]
    lfs = []
    for fi in range(4):
        cy_a, cy_b = (20, 44) if fi % 2 == 0 else (44, 20)
        masks = []
        for cy in (cy_a, cy_b):
            disk = ((yy - cy) ** 2 + (xx - 30) ** 2) <= 9**2
            masks.append(sio.PredictedSegmentationMask.from_numpy(disk, score=0.9))
        lfs.append(sio.LabeledFrame(video=vid, frame_idx=fi, masks=masks))
    labels = sio.Labels(labeled_frames=lfs, videos=[vid])  # no tracks
    sp = str(tmp_path / "untracked_mask.slp")
    sio.save_slp(labels, sp, embed=False)
    return sp


# ── EmbeddingDataset: include_untracked enumeration ──────────────────────────────


def _emb_head(cfg):
    return cfg.model_config.head_configs.embedding.embedding


def test_dataset_skips_untracked_by_default(untracked_pose_slp):
    """Default (training/offline) enumeration drops untracked detections."""
    labels = sio.load_slp(untracked_pose_slp)
    cfg = _build_embedding_config()
    ds = EmbeddingDataset(
        labels=[labels],
        crop_size=32,
        class_names=[],  # no tracks -> empty vocabulary
        embedding_head_config=_emb_head(cfg),
        max_stride=16,
        cache_img=None,
    )
    assert len(ds) == 0


def test_dataset_include_untracked_enumerates_all_pose(untracked_pose_slp):
    labels = sio.load_slp(untracked_pose_slp)
    cfg = _build_embedding_config()
    ds = EmbeddingDataset(
        labels=[labels],
        crop_size=32,
        class_names=[],
        embedding_head_config=_emb_head(cfg),
        max_stride=16,
        include_untracked=True,
        cache_img=None,
    )
    assert ds.detection_mode == "pose"
    assert len(ds) == 8  # 4 frames x 2 instances
    # group_id is a placeholder (0) for untracked detections.
    assert all(m["group_id"] == 0 for m in ds.mask_idx_list)


def test_dataset_include_untracked_detects_mask_mode(untracked_mask_slp):
    labels = sio.load_slp(untracked_mask_slp)
    cfg = _build_embedding_config()
    ds = EmbeddingDataset(
        labels=[labels],
        crop_size=32,
        class_names=[],
        embedding_head_config=_emb_head(cfg),
        max_stride=16,
        include_untracked=True,
        cache_img=None,
    )
    assert ds.detection_mode == "mask"  # not misread as pose
    assert len(ds) == 8


# ── predict_embeddings_to_slp + tracker_config (the WF2 entry point) ──────────────


def _trk_cfg(**kw):
    kw.setdefault("features", "embeddings")
    kw.setdefault("features_explicit", True)
    kw.setdefault("scoring_method", "cosine_sim")
    kw.setdefault("scoring_method_explicit", True)
    kw.setdefault("candidates_method", "local_queues")
    return TrackerConfig(**kw)


def test_wf2_pose_untracked_gets_tracked(
    embedding_model_dir, untracked_pose_slp, tmp_path
):
    """Untracked pose .slp -> all detections embedded + tracked into a .slp."""
    out = predict_embeddings_to_slp(
        [embedding_model_dir],
        untracked_pose_slp,
        output_path=str(tmp_path / "tracked.slp"),
        device="cpu",
        batch_size=4,
        save_embeddings="none",
        tracker_config=_trk_cfg(),
    )
    assert out == str(tmp_path / "tracked.slp")
    assert os.path.exists(out)
    tracked = sio.load_slp(out)
    insts = [i for lf in tracked.labeled_frames for i in lf.instances]
    assert len(insts) == 8
    assert all(i.track is not None for i in insts)
    # save_embeddings="none" -> vectors stripped from the tracked .slp.
    assert all(i.identity_embedding is None for i in insts)


def test_wf2_save_embeddings_persists_vectors(
    embedding_model_dir, untracked_pose_slp, tmp_path
):
    out = predict_embeddings_to_slp(
        [embedding_model_dir],
        untracked_pose_slp,
        output_path=str(tmp_path / "tracked.slp"),
        device="cpu",
        batch_size=4,
        save_embeddings="slp",
        tracker_config=_trk_cfg(),
    )
    tracked = sio.load_slp(out)
    insts = [i for lf in tracked.labeled_frames for i in lf.instances]
    assert all(i.track is not None for i in insts)
    # save_embeddings="slp" -> the appearance vectors persist alongside tracks.
    assert all(i.identity_embedding is not None for i in insts)
    assert all(np.asarray(i.identity_embedding.vector).shape == (_DIM,) for i in insts)


def test_wf2_mask_untracked_gets_tracked(
    embedding_model_dir, untracked_mask_slp, tmp_path
):
    out = predict_embeddings_to_slp(
        [embedding_model_dir],
        untracked_mask_slp,
        output_path=str(tmp_path / "tracked_mask.slp"),
        device="cpu",
        batch_size=4,
        save_embeddings="slp",
        tracker_config=_trk_cfg(),
    )
    tracked = sio.load_slp(out)
    masks = [m for lf in tracked.labeled_frames for m in lf.masks]
    assert len(masks) == 8
    assert all(m.track is not None for m in masks)
    assert all(
        m.identity_embedding is not None for m in masks
    )  # slp -> vectors persist


def test_wf2_default_output_path(embedding_model_dir, untracked_pose_slp):
    """No output_path -> defaults to ``<data_path>.tracked.slp``."""
    out = predict_embeddings_to_slp(
        [embedding_model_dir],
        untracked_pose_slp,
        device="cpu",
        batch_size=4,
        tracker_config=_trk_cfg(),
    )
    assert out == f"{untracked_pose_slp}.tracked.slp"
    assert os.path.exists(out)


# ── CLI routing: --tracking on an embedding model ────────────────────────────────


def _impl_kwargs(**over):
    base = dict(
        model_paths=["m"],
        frames=None,
        data_path="d.slp",
        device="cpu",
        batch_size=4,
        peak_threshold=None,
        save_embeddings="none",
        tracking=False,
    )
    base.update(over)
    return base


def _patch_embed_writer(monkeypatch):
    """Patch the embedding writer + model-type detection; capture threaded kwargs."""
    import sleap_nn.cli as cli
    import sleap_nn.inference.embedding as emb_mod

    captured = {}

    def _fake(**kwargs):
        captured.update(kwargs)
        return "RET"

    monkeypatch.setattr(emb_mod, "predict_embeddings_to_slp", _fake)
    # The single fake model path is the embedding model (no detection stack -> no fused
    # detect pass), so _run_embeddings calls the (patched) writer directly.
    monkeypatch.setattr(cli, "_is_embedding_model", lambda *_: True)
    return captured


def test_cli_embedding_tracking_threads_tracker_config(monkeypatch):
    """``predict --tracking`` on an embedding model builds + threads a TrackerConfig."""
    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_has_embedding_model", lambda *_: True)
    captured = _patch_embed_writer(monkeypatch)
    out = cli._run_inference_impl(**_impl_kwargs(tracking=True))
    assert out == "RET"
    cfg = captured["tracker_config"]
    assert cfg is not None
    # Defaults to appearance/cosine for an embedding model.
    assert cfg.features == "embeddings"
    assert cfg.scoring_method == "cosine_sim"


def test_cli_embedding_tracking_no_save_embeddings_required(monkeypatch):
    """``--tracking`` lifts the save_embeddings requirement (tracks-only output)."""
    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_has_embedding_model", lambda *_: True)
    _patch_embed_writer(monkeypatch)
    # Default OFF + no --save_embeddings would normally raise; --tracking makes it OK.
    cli._run_inference_impl(**_impl_kwargs(tracking=True))


def test_cli_embedding_no_tracking_requires_save_embeddings(monkeypatch):
    """Without --tracking, the OFF guard fires (need --save_embeddings slp)."""
    import click

    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_has_embedding_model", lambda *_: True)
    _patch_embed_writer(monkeypatch)
    with pytest.raises(click.UsageError):
        cli._run_inference_impl(**_impl_kwargs(tracking=False))


def test_cli_embedding_tracking_respects_explicit_features(monkeypatch):
    """An explicit --features/--scoring_method is not overridden by the defaults."""
    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_has_embedding_model", lambda *_: True)
    captured = _patch_embed_writer(monkeypatch)
    cli._run_inference_impl(
        **_impl_kwargs(
            tracking=True, features="embeddings", scoring_method="euclidean_dist"
        )
    )
    assert captured["tracker_config"].scoring_method == "euclidean_dist"


@pytest.mark.parametrize(
    "flag", [{"stream_to_file": "out.slp"}, {"mask_backend": "sam2"}]
)
def test_cli_embedding_rejects_stream_and_mask_backend(monkeypatch, flag):
    """--stream-to-file / --mask_backend are unsupported for embedding models
    (review finding [7]) — they must error, not be silently ignored."""
    import click

    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_has_embedding_model", lambda *_: True)
    _patch_embed_writer(monkeypatch)
    with pytest.raises(click.UsageError):
        cli._run_inference_impl(**_impl_kwargs(save_embeddings="slp", **flag))


# ── WF3: fused detect -> embed (+track) ──────────────────────────────────────────


def _patch_fused(monkeypatch, emb_dir, detections_slp):
    """Classify ``emb_dir`` as the embedding model and stub the detection pass.

    The fused path runs the detection stack to ``det_kwargs["output_path"]``; the stub
    drops a known UNTRACKED ``.slp`` there so the (real) embedding step has fresh,
    track-less detections to embed — exactly what a centroid/CI detector produces.
    """
    import shutil

    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_is_embedding_model", lambda m: m == emb_dir)

    def _fake_detect(det_kwargs, paf_workers=0):
        shutil.copy(detections_slp, det_kwargs["output_path"])

    monkeypatch.setattr(cli, "_run_in_memory_new_flow", _fake_detect)


def _fused_kwargs(emb_dir, **over):
    base = dict(
        model_paths=["det_dir", emb_dir],  # detection stack + embedding model
        frames=None,
        data_path="video.mp4",  # a raw video -> triggers the fused detect pass
        device="cpu",
        batch_size=4,
        peak_threshold=None,
        save_embeddings="none",
        tracking=False,
    )
    base.update(over)
    return base


def test_cli_fused_detect_embed_track(
    monkeypatch, embedding_model_dir, untracked_pose_slp, tmp_path
):
    """WF3: detection dirs + embedding dir -> detect (stubbed) -> embed + track a .slp."""
    import sleap_nn.cli as cli

    _patch_fused(monkeypatch, embedding_model_dir, untracked_pose_slp)
    out = cli._run_inference_impl(
        **_fused_kwargs(
            embedding_model_dir,
            tracking=True,
            max_tracks=6,
            output_path=str(tmp_path / "fused.slp"),
        )
    )
    assert out == str(tmp_path / "fused.slp")
    tracked = sio.load_slp(out)
    insts = [i for lf in tracked.labeled_frames for i in lf.instances]
    # 4 frames x 2 untracked detections -> all embedded + tracked.
    assert len(insts) == 8
    assert all(i.track is not None for i in insts)


def test_cli_fused_no_tracking_persists_vectors(
    monkeypatch, embedding_model_dir, untracked_pose_slp, tmp_path
):
    """WF3 without --tracking: fresh (untracked) detections are embedded + persisted.

    Regression guard: the fused detections carry no tracks, so embedding must run in
    include-untracked mode even when not tracking (else it raises "No tracked
    detections found to embed").
    """
    import sleap_nn.cli as cli

    _patch_fused(monkeypatch, embedding_model_dir, untracked_pose_slp)
    out = cli._run_inference_impl(
        **_fused_kwargs(
            embedding_model_dir,
            tracking=False,
            save_embeddings="slp",
            output_path=str(tmp_path / "fused_emb.slp"),
        )
    )
    embedded = sio.load_slp(out)
    insts = [i for lf in embedded.labeled_frames for i in lf.instances]
    assert len(insts) == 8
    assert all(i.identity_embedding is not None for i in insts)
