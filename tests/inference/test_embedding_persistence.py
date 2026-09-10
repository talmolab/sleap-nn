"""Embedding persistence into ``.slp`` via the sleap-io ``Embedding`` data model.

``sleap_nn.inference.embedding.predict_embeddings_to_slp`` takes a ``save_embeddings``
option (``"none"`` / ``"slp"``). With ``"slp"`` it attaches each emitted crop's
appearance vector to its **object-exact source detection** (the ``sio.Instance`` for
pose data, the ``sio.SegmentationMask`` for mask data) via ``identity_embedding`` and
writes a ``.slp``.

These tests assert, on a tiny (random-weights) embedding model:

1. ``"none"`` without tracking has nothing to persist -> raises (the source ``.slp`` is
   not written / not mutated on disk),
2. ``"slp"`` (pose) -> each ``Instance`` carries ``identity_embedding`` of dim D,
   ``(video, frame, track)`` integrity preserved,
   and it survives ``save_slp`` -> ``load_slp``,
3. ``"slp"`` (pose) -> every detection's vector is written,
4. the pathway stores vectors ONLY — it does not fabricate ``sio.Identity`` from
   GT track names (existing identities pass through; tracks are preserved),
5. mask path: a mask-only ``.slp`` round-trips its ``SegmentationMask`` embeddings
   into the ``.slp`` (mask-modality persistence, ``owner_type=3``, landed in
   sleap-io#527).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import sleap_io as sio

_DIM = 16
_MAX_STRIDE = 16
_CROP = 32


def _build_embedding_config(in_channels: int = 1):
    """A minimal, loadable ``embedding`` training config (mirrors the export fixture)."""
    backbone_leaf = {
        "in_channels": in_channels,
        "kernel_size": 3,
        "filters": 8,
        "filters_rate": 1.5,
        "max_stride": _MAX_STRIDE,
        "stem_stride": None,
        "middle_block": True,
        "up_interpolate": True,
        "stacks": 1,
        "convs_per_block": 2,
        "output_stride": 2,
    }
    objective = {
        "positives": {"scope": "global_id"},
        "negatives": {
            "sources": ["in_batch"],
            "exclude_same_track": True,
            "restrict_same_video": False,
        },
        "loss": {"name": "supcon", "temperature": 0.1, "margin": 0.2},
        "sampler": {"kind": "pk", "groups_per_batch": 2, "samples_per_group": 4},
        "use_projection": True,
        "projection_dim": _DIM,
    }
    return OmegaConf.create(
        {
            "data_config": {
                "preprocessing": {
                    "scale": 1.0,
                    "crop_size": _CROP,
                    "max_height": 64,
                    "max_width": 64,
                    "ensure_rgb": in_channels == 3,
                    "ensure_grayscale": in_channels == 1,
                },
                "skeletons": None,
            },
            "model_config": {
                "backbone_type": "unet",
                "backbone_config": {"unet": backbone_leaf},
                "head_configs": {
                    "embedding": {
                        "embedding": {
                            "embedding_dim": _DIM,
                            "num_fc_layers": 1,
                            "num_fc_units": 32,
                            "pool": "gem",
                            "normalize": True,
                            "output_stride": _MAX_STRIDE,
                            "loss_weight": 1.0,
                            "freeze_backbone": False,
                            "objective": objective,
                        }
                    },
                },
                "pretrained_backbone_weights": None,
                "pretrained_head_weights": None,
                "init_weights": "xavier",
            },
            "trainer_config": {
                "lr_scheduler": None,
                "optimizer_name": "Adam",
                "optimizer": {"lr": 1e-3, "amsgrad": False},
                "online_hard_keypoint_mining": {
                    "online_mining": False,
                    "hard_to_easy_ratio": 2.0,
                    "min_hard_keypoints": 2,
                    "max_hard_keypoints": None,
                    "loss_scale": 5.0,
                },
            },
        }
    )


@pytest.fixture(scope="module")
def embedding_model_dir(tmp_path_factory):
    """A tiny random-weights embedding model directory (``best.ckpt`` + config)."""
    from sleap_nn.training.lightning_modules import EmbeddingLightningModule

    cfg = _build_embedding_config()
    module = EmbeddingLightningModule(
        model_type="embedding",
        backbone_type="unet",
        backbone_config=cfg.model_config.backbone_config,
        head_configs=cfg.model_config.head_configs,
        init_weights="xavier",
    ).eval()

    model_dir = tmp_path_factory.mktemp("embedding_model")
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": {},
            "pytorch-lightning_version": "2.0.0",
            "epoch": 0,
            "global_step": 0,
        },
        model_dir / "best.ckpt",
    )
    OmegaConf.save(cfg, model_dir / "training_config.yaml")
    return str(model_dir)


def _write_video(path, n=3, h=64, w=64):
    import imageio.v3 as iio

    frames = np.random.default_rng(0).integers(0, 255, size=(n, h, w), dtype=np.uint8)
    iio.imwrite(path, frames, fps=5)
    return sio.load_video(str(path))


@pytest.fixture
def pose_slp(tmp_path):
    """A synthetic pose ``.slp``: 3 frames x 2 tracked 2-node instances."""
    vid = _write_video(tmp_path / "vid.mp4")
    skel = sio.Skeleton(nodes=["a", "b"])
    t0, t1 = sio.Track("animal0"), sio.Track("animal1")
    lfs = []
    for fi in range(3):
        i0 = sio.Instance.from_numpy(
            np.array([[18.0 + fi, 20.0], [26.0 + fi, 28.0]]), skeleton=skel, track=t0
        )
        i1 = sio.Instance.from_numpy(
            np.array([[40.0 + fi, 42.0], [46.0 + fi, 48.0]]), skeleton=skel, track=t1
        )
        lfs.append(sio.LabeledFrame(video=vid, frame_idx=fi, instances=[i0, i1]))
    labels = sio.Labels(
        labeled_frames=lfs, videos=[vid], skeletons=[skel], tracks=[t0, t1]
    )
    sp = str(tmp_path / "pose.slp")
    sio.save_slp(labels, sp, embed=False)
    return sp


@pytest.fixture
def mask_slp(tmp_path):
    """A synthetic mask-only ``.slp``: 3 frames x 2 tracked disk masks."""
    vid = _write_video(tmp_path / "mvid.mp4")
    t0, t1 = sio.Track("animal0"), sio.Track("animal1")
    yy, xx = np.ogrid[:64, :64]
    lfs = []
    for fi in range(3):
        masks = []
        for (cy, cx), tr in (((20 + fi, 20), t0), ((44 + fi, 44), t1)):
            disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= 9**2
            m = sio.PredictedSegmentationMask.from_numpy(disk, score=0.9)
            m.track = tr
            masks.append(m)
        lfs.append(sio.LabeledFrame(video=vid, frame_idx=fi, masks=masks))
    labels = sio.Labels(labeled_frames=lfs, videos=[vid], tracks=[t0, t1])
    sp = str(tmp_path / "mask.slp")
    sio.save_slp(labels, sp, embed=False)
    return sp


def _run(model_dir, data_path, tmp_path, save_embeddings):
    from sleap_nn.inference.embedding import predict_embeddings_to_slp

    slp_out = str(tmp_path / "out.embeddings.slp")
    res = predict_embeddings_to_slp(
        [model_dir],
        data_path,
        output_path=slp_out,
        device="cpu",
        batch_size=4,
        save_embeddings=save_embeddings,
    )
    return res, slp_out


# ── (1) "none" without tracking: nothing to persist -> raises ──────────────────


@pytest.mark.parametrize("off", [None, "none"])
def test_none_without_tracking_raises(embedding_model_dir, pose_slp, tmp_path, off):
    """``none`` (no tracking) has nothing to persist: raises, source untouched."""
    with pytest.raises(ValueError):
        _run(embedding_model_dir, pose_slp, tmp_path, off)
    # No .slp is written.
    assert not os.path.exists(str(tmp_path / "out.embeddings.slp"))
    # Source .slp is untouched: no embeddings on reload.
    src = sio.load_slp(pose_slp)
    for lf in src.labeled_frames:
        for inst in lf.instances:
            assert inst.identity_embedding is None


# ── (2) slp: pose embeddings persist with dim + integrity ──────────────────────


def test_slp_writes_slp_pose(embedding_model_dir, pose_slp, tmp_path):
    """``slp``: pose vectors persist (dim) + (video,frame,track) integrity."""
    res, slp_out = _run(embedding_model_dir, pose_slp, tmp_path, "slp")
    assert res == slp_out
    assert os.path.exists(slp_out)

    reloaded = sio.load_slp(slp_out)
    # (video, frame, track) integrity: same frames, 2 instances each, same tracks.
    assert len(reloaded.labeled_frames) == 3
    for lf in reloaded.labeled_frames:
        assert len(lf.instances) == 2
        names = {inst.track.name for inst in lf.instances}
        assert names == {"animal0", "animal1"}
        for inst in lf.instances:
            emb = inst.identity_embedding
            assert emb is not None, "instance missing identity_embedding"
            assert np.asarray(emb.vector).shape == (_DIM,)

    # Round-trip: re-save -> re-load preserves the vectors.
    rt = str(tmp_path / "rt.slp")
    sio.save_slp(reloaded, rt, embed=False, save_embedding_vectors=True)
    rt_labels = sio.load_slp(rt)
    before = {
        (lf.frame_idx, inst.track.name): np.asarray(inst.identity_embedding.vector)
        for lf in reloaded.labeled_frames
        for inst in lf.instances
    }
    after = {
        (lf.frame_idx, inst.track.name): np.asarray(inst.identity_embedding.vector)
        for lf in rt_labels.labeled_frames
        for inst in lf.instances
    }
    assert before.keys() == after.keys()
    for k in before:
        np.testing.assert_allclose(before[k], after[k], rtol=0, atol=1e-6)


# ── (3) slp: every detection's vector is written ───────────────────────────────


def test_slp_writes_all_vectors(embedding_model_dir, pose_slp, tmp_path):
    """``slp``: returns the .slp path; every detection (3 frames x 2) gets a vector."""
    res, slp_out = _run(embedding_model_dir, pose_slp, tmp_path, "slp")
    assert res == slp_out
    assert os.path.exists(slp_out)
    reloaded = sio.load_slp(slp_out)
    n = sum(
        1
        for lf in reloaded.labeled_frames
        for inst in lf.instances
        if inst.identity_embedding is not None
    )
    assert n == 6


# ── (4) embeddings only: no identity fabricated from tracks ───────────────────


def test_no_identity_minted_from_track_pose(embedding_model_dir, pose_slp, tmp_path):
    """The embedding pathway stores vectors only — it does NOT mint identities.

    A track/class name is not a global animal identity, so the embedding writer
    must never fabricate ``sio.Identity`` objects from GT track names (that's the
    re-ID resolve loop's job, deferred). The source tracks are preserved; the
    identity slots stay empty and the identity catalog is not populated.
    """
    _, slp_out = _run(embedding_model_dir, pose_slp, tmp_path, "slp")
    reloaded = sio.load_slp(slp_out)

    assert not (getattr(reloaded, "identities", None) or [])  # no catalog minted
    for lf in reloaded.labeled_frames:
        for inst in lf.instances:
            assert inst.identity_embedding is not None  # vector attached
            assert inst.identity is None  # no fabricated identity
            assert inst.identity_score is None
            assert inst.track is not None  # GT track preserved


# ── (5) mask path: SegmentationMask embeddings persist (#527) ─────────────────


def test_mask_path_persists_embeddings(embedding_model_dir, mask_slp, tmp_path):
    """Mask path: SegmentationMask embeddings persist (no fabricated identity).

    Mask-modality embedding persistence (``owner_type=3``) landed in sleap-io#527,
    so mask detections now round-trip their re-ID vector into the ``.slp`` — but,
    like the pose path, no ``sio.Identity`` is minted from the GT track.
    """
    res, slp_out = _run(embedding_model_dir, mask_slp, tmp_path, "slp")
    assert res == slp_out
    assert os.path.exists(slp_out)
    reloaded = sio.load_slp(slp_out)
    masks = [m for lf in reloaded.labeled_frames for m in lf.masks]
    assert len(masks) == 6
    # Every mask carries its identity_embedding (dim _DIM).
    assert all(m.identity_embedding is not None for m in masks)
    assert {m.identity_embedding.dim for m in masks} == {_DIM}
    # No identity fabricated from the GT track.
    assert all(m.identity is None for m in masks)
    assert not (getattr(reloaded, "identities", None) or [])


# ── CLI routing: --save_embeddings threads through / guards correctly ─────────


def _impl_kwargs(**over):
    base = dict(
        model_paths=["m"],
        frames=None,
        data_path="d.slp",
        device="cpu",
        batch_size=4,
        peak_threshold=None,
        save_embeddings="none",
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


def test_cli_threads_save_embeddings(monkeypatch):
    """``predict --save_embeddings slp`` threads the mode through (no path needed)."""
    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_has_embedding_model", lambda *_: True)
    captured = _patch_embed_writer(monkeypatch)
    out = cli._run_inference_impl(**_impl_kwargs(save_embeddings="slp"))
    assert out == "RET"
    assert captured["save_embeddings"] == "slp"
    assert captured["output_path"] is None  # defaults derived in the writer


def test_cli_output_path_threads(monkeypatch):
    """``-o/--output_path`` is threaded to the writer as the output .slp."""
    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_has_embedding_model", lambda *_: True)
    captured = _patch_embed_writer(monkeypatch)
    cli._run_inference_impl(**_impl_kwargs(save_embeddings="slp", output_path="o.slp"))
    assert captured["output_path"] == "o.slp"


def test_cli_embedding_model_off_requires_save_embeddings(monkeypatch):
    """Embedding model + default OFF + no tracking requires --save_embeddings slp."""
    import click

    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_has_embedding_model", lambda *_: True)
    with pytest.raises(click.UsageError):
        cli._run_inference_impl(**_impl_kwargs())  # off + no tracking


def test_cli_save_embeddings_rejected_for_non_embedding(monkeypatch):
    """``--save_embeddings`` needs a model that computes vectors, or a .slp input.

    A video input carries no appearance vectors and there is no embedding model to
    compute them, so the flag can only be a mistake -> UsageError.
    """
    import click

    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_has_embedding_model", lambda *_: False)
    with pytest.raises(click.UsageError, match="already carries them"):
        cli._run_inference_impl(
            **_impl_kwargs(save_embeddings="slp", data_path="video.mp4")
        )


def test_cli_save_embeddings_allowed_on_slp_input(monkeypatch):
    """``--save_embeddings`` IS allowed on a .slp input with no embedding model.

    The input may already carry vectors (an earlier ``--save_embeddings slp`` run),
    which is what the guide's retrack-at-several-weights sweep chains off. Asserts
    the choice reaches the save layer as ``save_embedding_vectors``.
    """
    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_has_embedding_model", lambda *_: False)
    seen = {}

    def _fake_flow(kwargs, paf_workers=0):
        seen["save_embedding_vectors"] = kwargs.get("save_embedding_vectors")
        return None

    monkeypatch.setattr(cli, "_run_in_memory_new_flow", _fake_flow)
    cli._run_inference_impl(**_impl_kwargs(save_embeddings="slp", data_path="dets.slp"))
    assert seen["save_embedding_vectors"] is True


def test_cli_save_embeddings_default_preserves_existing_vectors(monkeypatch):
    """Leaving ``--save_embeddings`` unset resolves to "preserve if present".

    sleap-io's own ``save_embedding_vectors`` default is ``False``, so before this a
    plain ``predict``/retrack of an embedded .slp silently dropped every vector and
    its output could not be retracked by appearance again.
    """
    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_has_embedding_model", lambda *_: False)
    seen = {}

    def _fake_flow(kwargs, paf_workers=0):
        seen["save_embedding_vectors"] = kwargs.get("save_embedding_vectors")
        return None

    monkeypatch.setattr(cli, "_run_in_memory_new_flow", _fake_flow)
    cli._run_inference_impl(**_impl_kwargs(data_path="dets.slp"))
    assert seen["save_embedding_vectors"] is None


# ── The output must be readable: don't restore an absent source video ─────────


def test_pkg_slp_input_output_keeps_the_package(embedding_model_dir, tmp_path):
    """A `.pkg.slp` input must not yield an output pointing at the source video.

    sleap-io's `save_slp` defaults `restore_original_videos=True`, which every
    other writer in the repo overrides (`save_predictions`;
    `--restore_source_videos` defaults False, "the pre-embedding source video is
    often unavailable"). Here it was left at the default, so `lf.image` on the
    output raised FileNotFoundError for the standard SLEAP training-package input.
    """
    from sleap_nn.inference.embedding import predict_embeddings_to_slp

    src = "tests/assets/datasets/minimal_instance.pkg.slp"
    out = predict_embeddings_to_slp(
        [embedding_model_dir],
        src,
        output_path=str(tmp_path / "pkg.embeddings.slp"),
        device="cpu",
        batch_size=2,
        save_embeddings="slp",
        include_untracked=True,
    )
    labels = sio.load_slp(out)
    assert os.path.basename(str(labels.videos[0].filename)).endswith(".pkg.slp")
    assert labels.labeled_frames[0].image is not None  # pixels still readable


def test_restore_source_videos_true_is_still_available(embedding_model_dir, tmp_path):
    """The flag is threaded, not hard-coded: True restores the source reference."""
    from sleap_nn.inference.embedding import predict_embeddings_to_slp

    out = predict_embeddings_to_slp(
        [embedding_model_dir],
        "tests/assets/datasets/minimal_instance.pkg.slp",
        output_path=str(tmp_path / "restored.slp"),
        device="cpu",
        batch_size=2,
        save_embeddings="slp",
        include_untracked=True,
        restore_source_videos=True,
    )
    labels = sio.load_slp(out)
    assert not str(labels.videos[0].filename).endswith(".pkg.slp")


# ── Burn-in divergence must not be silent (review finding [9]) ────────────────


def test_burn_in_pose_mode_warns(embedding_model_dir, pose_slp, tmp_path, caplog):
    """A mask-trained (`burn_in=True`) model embedding MASKLESS crops diverges.

    Pose mode hands the module an all-ones mask, so `_standardize` degrades to a
    whole-crop standardize with the background left in. Every fused
    `detect -> embed` run is pose mode, so this is the common case. `origin/main`
    warned; the rewritten kernel dropped the warning entirely.
    """
    import logging

    from loguru import logger as _loguru

    from unittest.mock import patch

    from sleap_nn.inference.embedding import predict_embeddings_to_slp
    from sleap_nn.inference.predictor import Predictor

    real_from_model_paths = Predictor.from_model_paths.__func__

    def _burn_in_predictor(cls, *args, **kwargs):
        predictor = real_from_model_paths(cls, *args, **kwargs)
        predictor.layer.embedding_module.burn_in = True
        return predictor

    handler_id = _loguru.add(lambda m: logging.getLogger("loguru").warning(m))
    try:
        with patch.object(
            Predictor, "from_model_paths", classmethod(_burn_in_predictor)
        ):
            with caplog.at_level(logging.WARNING, logger="loguru"):
                predict_embeddings_to_slp(
                    [embedding_model_dir],
                    pose_slp,
                    output_path=str(tmp_path / "burn_in.slp"),
                    device="cpu",
                    batch_size=2,
                    save_embeddings="slp",
                )
        assert "trained with mask burn-in" in caplog.text
    finally:
        _loguru.remove(handler_id)


def test_no_burn_in_warning_when_off(embedding_model_dir, pose_slp, tmp_path, caplog):
    """The warning must not fire for the ordinary (burn_in=False) model."""
    import logging

    from loguru import logger as _loguru

    from sleap_nn.inference.embedding import predict_embeddings_to_slp

    handler_id = _loguru.add(lambda m: logging.getLogger("loguru").warning(m))
    try:
        with caplog.at_level(logging.WARNING, logger="loguru"):
            predict_embeddings_to_slp(
                [embedding_model_dir],
                pose_slp,
                output_path=str(tmp_path / "plain.slp"),
                device="cpu",
                batch_size=2,
                save_embeddings="slp",
            )
        assert "mask burn-in" not in caplog.text
    finally:
        _loguru.remove(handler_id)


# ── Clear diagnostics instead of opaque HDF5 / training-flavored errors ───────


def test_video_data_path_is_rejected_clearly(embedding_model_dir, tmp_path):
    """A lone embedding model embeds EXISTING detections, so its input is a .slp.

    A video path hit `sio.load_slp` unconditionally and died with h5py's
    "file signature not found" OSError.
    """
    from sleap_nn.inference.embedding import predict_embeddings_to_slp

    video = tmp_path / "clip.mp4"
    _write_video(video)
    with pytest.raises(ValueError, match="must be a .slp of detections"):
        predict_embeddings_to_slp(
            [embedding_model_dir], str(video), device="cpu", save_embeddings="slp"
        )


def test_no_detections_says_so(embedding_model_dir, tmp_path):
    """An empty (detector-found-nothing) input must not surface the training-flavored
    "none of the labeled frames contain user-labeled data" from BaseDataset."""
    from sleap_nn.inference.embedding import predict_embeddings_to_slp

    vid = _write_video(tmp_path / "empty.mp4")
    labels = sio.Labels(
        videos=[vid],
        labeled_frames=[sio.LabeledFrame(video=vid, frame_idx=i) for i in range(3)],
    )
    empty = str(tmp_path / "empty.slp")
    sio.save_slp(labels, empty, embed=False)
    with pytest.raises(ValueError, match="No detections to embed"):
        predict_embeddings_to_slp(
            [embedding_model_dir],
            empty,
            device="cpu",
            save_embeddings="slp",
            include_untracked=True,
        )


# ── Retracking must not destroy the vectors it tracked on (finding [5]) ───────


@pytest.fixture
def embedded_pose_slp(tmp_path):
    """A pose ``.slp`` whose instances already carry appearance vectors."""
    vid = _write_video(tmp_path / "evid.mp4")
    skel = sio.Skeleton(nodes=["a", "b"])
    lfs = []
    for fi in range(3):
        insts = []
        for x, vec in ((18.0, (1.0, 0.0)), (44.0, (0.0, 1.0))):
            inst = sio.PredictedInstance.from_numpy(
                np.array([[x + fi, 20.0], [x + fi + 6, 28.0]]),
                skeleton=skel,
                score=0.9,
                point_scores=np.ones(2),
            )
            inst.identity_embedding = sio.Embedding(np.asarray(vec, np.float32))
            insts.append(inst)
        lfs.append(sio.LabeledFrame(video=vid, frame_idx=fi, instances=insts))
    labels = sio.Labels(labeled_frames=lfs, videos=[vid], skeletons=[skel])
    sp = str(tmp_path / "embedded.slp")
    sio.save_slp(labels, sp, embed=False, save_embedding_vectors=True)
    return sp


def test_save_predictions_preserves_existing_vectors(embedded_pose_slp, tmp_path):
    """`save_predictions` defaults to "preserve if present".

    sleap-io's own `save_embedding_vectors` default is False, so the WF1 retrack
    route (`labels.save(...)` via `save_predictions`) silently dropped every
    vector -- and retracking its output then raised "no detection ... carries an
    appearance embedding".
    """
    from sleap_nn.inference.run import save_predictions

    labels = sio.load_slp(embedded_pose_slp)
    out = str(tmp_path / "resaved.slp")
    save_predictions(labels, out)
    reloaded = sio.load_slp(out)
    insts = [i for lf in reloaded.labeled_frames for i in lf.instances]
    assert len(insts) == 6
    assert all(i.identity_embedding is not None for i in insts)


def test_save_predictions_can_strip_vectors_explicitly(embedded_pose_slp, tmp_path):
    """`--save_embeddings none` on the command line still means "strip"."""
    from sleap_nn.inference.run import save_predictions

    labels = sio.load_slp(embedded_pose_slp)
    out = str(tmp_path / "stripped.slp")
    save_predictions(labels, out, save_embedding_vectors=False)
    reloaded = sio.load_slp(out)
    insts = [i for lf in reloaded.labeled_frames for i in lf.instances]
    assert insts and all(i.identity_embedding is None for i in insts)


def test_wf1_retrack_output_can_be_retracked(embedded_pose_slp, tmp_path):
    """End-to-end: the guide's weight sweep chains off its own outputs.

    Retrack an embedded .slp by appearance, then retrack the RESULT again. The
    second call used to raise because the first had stripped the vectors.
    """
    from click.testing import CliRunner

    from sleap_nn.cli import cli

    first = str(tmp_path / "tracked1.slp")
    res = CliRunner().invoke(
        cli,
        [
            "predict",
            "-i",
            embedded_pose_slp,
            "-t",
            "--features",
            "embeddings",
            "-o",
            first,
        ],
    )
    assert res.exit_code == 0, res.output
    insts = [i for lf in sio.load_slp(first).labeled_frames for i in lf.instances]
    assert insts and all(i.track is not None for i in insts)
    assert all(i.identity_embedding is not None for i in insts)

    second = str(tmp_path / "tracked2.slp")
    res = CliRunner().invoke(
        cli,
        ["predict", "-i", first, "-t", "--features", "embeddings", "-o", second],
    )
    assert res.exit_code == 0, res.output
