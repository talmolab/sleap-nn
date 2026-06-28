"""A1: optional embedding persistence into ``.slp`` (default OFF).

``sleap_nn.inference.embedding.predict_embeddings_to_h5`` gains a
``save_embeddings`` option (``None`` / ``"none"`` / ``"slp"`` / ``"both"``,
default OFF). When enabled it attaches each emitted crop's appearance vector to
its **object-exact source detection** (the ``sio.Instance`` for pose data, the
``sio.SegmentationMask`` for mask data) via ``set_embedding`` and writes a
``.slp`` (in addition to / instead of today's sidecar ``.h5``).

These tests assert, on a tiny (random-weights) embedding model:

1. default OFF -> output unchanged: ``.h5`` only, NO ``.slp`` written, the source
   ``.slp`` is untouched (no embeddings on reload),
2. ``"both"`` (pose) -> ``.h5`` AND ``.slp`` written; each ``Instance`` carries
   ``embedding`` of dim D, the ``normalized`` flag + ``source`` set, ``(video,
   frame, track)`` integrity preserved, and it survives ``save_slp`` ->
   ``load_slp``,
3. ``"slp"`` (pose) -> only the ``.slp`` is written (no ``.h5``),
4. the source detection's GT track is promoted to a canonical ``sio.Identity``
   (``identity_score=None``) registered on ``labels.identities`` (pose-only),
5. mask path: a mask-only ``.slp`` with ``save_embeddings`` on does NOT crash
   (mask embeddings warn + drop until sleap-io#525); the ``.h5`` is still written.
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
    from sleap_nn.inference.embedding import predict_embeddings_to_h5

    out_h5 = str(tmp_path / "out.embeddings.h5")
    res = predict_embeddings_to_h5(
        [model_dir],
        data_path,
        output_path=out_h5,
        device="cpu",
        batch_size=4,
        save_embeddings=save_embeddings,
    )
    slp_out = out_h5[:-3] + ".slp"
    return res, out_h5, slp_out


# ── (1) default OFF: byte-identical to today (h5 only, no .slp) ────────────────


@pytest.mark.parametrize("off", [None, "none"])
def test_default_off_writes_only_h5(embedding_model_dir, pose_slp, tmp_path, off):
    """Default OFF: .h5 only, no .slp written, source .slp untouched."""
    res, out_h5, slp_out = _run(embedding_model_dir, pose_slp, tmp_path, off)
    assert res == out_h5
    assert os.path.exists(out_h5)
    # No .slp is written when OFF.
    assert not os.path.exists(slp_out)
    # Source .slp is untouched: no embeddings on reload.
    src = sio.load_slp(pose_slp)
    for lf in src.labeled_frames:
        for inst in lf.instances:
            assert not inst.embeddings


# ── (2) both: pose embeddings persist with dim/flag/source + integrity ────────


def test_both_writes_h5_and_slp_pose(embedding_model_dir, pose_slp, tmp_path):
    """``both``: pose vectors persist (dim/flag/source) + (video,frame,track) integrity."""
    res, out_h5, slp_out = _run(embedding_model_dir, pose_slp, tmp_path, "both")
    assert res == out_h5
    assert os.path.exists(out_h5) and os.path.exists(slp_out)

    model_id = os.path.basename(embedding_model_dir.rstrip("/"))
    reloaded = sio.load_slp(slp_out)
    # (video, frame, track) integrity: same frames, 2 instances each, same tracks.
    assert len(reloaded.labeled_frames) == 3
    for lf in reloaded.labeled_frames:
        assert len(lf.instances) == 2
        names = {inst.track.name for inst in lf.instances}
        assert names == {"animal0", "animal1"}
        for inst in lf.instances:
            emb = inst.embeddings.get("reid")
            assert emb is not None, "instance missing reid embedding"
            assert np.asarray(emb.vector).shape == (_DIM,)
            assert emb.normalized is True  # config normalize=True
            assert emb.source == model_id

    # Round-trip: re-save -> re-load preserves the vectors.
    rt = str(tmp_path / "rt.slp")
    sio.save_slp(reloaded, rt, embed=False)
    rt_labels = sio.load_slp(rt)
    before = {
        (lf.frame_idx, inst.track.name): np.asarray(inst.embeddings["reid"].vector)
        for lf in reloaded.labeled_frames
        for inst in lf.instances
    }
    after = {
        (lf.frame_idx, inst.track.name): np.asarray(inst.embeddings["reid"].vector)
        for lf in rt_labels.labeled_frames
        for inst in lf.instances
    }
    assert before.keys() == after.keys()
    for k in before:
        np.testing.assert_allclose(before[k], after[k], rtol=0, atol=1e-6)


# ── (3) slp-only: no .h5 produced ─────────────────────────────────────────────


def test_slp_only_writes_no_h5(embedding_model_dir, pose_slp, tmp_path):
    """``slp``: only the .slp is written (no .h5); returns the .slp path."""
    res, out_h5, slp_out = _run(embedding_model_dir, pose_slp, tmp_path, "slp")
    assert res == slp_out
    assert os.path.exists(slp_out)
    assert not os.path.exists(out_h5)
    reloaded = sio.load_slp(slp_out)
    n = sum(
        1
        for lf in reloaded.labeled_frames
        for inst in lf.instances
        if inst.embeddings.get("reid") is not None
    )
    assert n == 6


# ── (4) GT track -> canonical identity (pose-only) ────────────────────────────


def test_identity_from_gt_track_pose(embedding_model_dir, pose_slp, tmp_path):
    """GT track -> canonical sio.Identity (score None), registered + round-tripped."""
    _, _, slp_out = _run(embedding_model_dir, pose_slp, tmp_path, "both")
    reloaded = sio.load_slp(slp_out)

    assert {i.name for i in reloaded.identities} == {"animal0", "animal1"}
    by_name = {i.name: i for i in reloaded.identities}
    for lf in reloaded.labeled_frames:
        for inst in lf.instances:
            assert inst.identity is not None
            assert isinstance(inst.identity, sio.Identity)
            assert inst.identity.name == inst.track.name
            # GT identity: no score.
            assert inst.identity_score is None
            # Same canonical object per name (deduped + registered).
            assert inst.identity.uuid == by_name[inst.identity.name].uuid


# ── (5) mask path: no crash; .h5 still written (vectors warn + drop) ──────────


def test_mask_path_no_crash_h5_written(embedding_model_dir, mask_slp, tmp_path):
    """Mask path: no crash with save_embeddings on; .h5 still carries the vectors."""
    # Must not raise even though save_slp drops mask (owner_type=3) embeddings.
    res, out_h5, slp_out = _run(embedding_model_dir, mask_slp, tmp_path, "both")
    assert res == out_h5
    # The .h5 still carries the vectors.
    assert os.path.exists(out_h5)
    import h5py

    with h5py.File(out_h5, "r") as h:
        assert h["embeddings"].shape == (6, _DIM)
    # The .slp is written and reloads cleanly (mask embeddings dropped, no crash).
    assert os.path.exists(slp_out)
    reloaded = sio.load_slp(slp_out)
    assert sum(len(lf.masks) for lf in reloaded.labeled_frames) == 6


# ── CLI routing: --save_embeddings threads through / guards correctly ─────────


def _impl_kwargs(**over):
    base = dict(
        model_paths=["m"],
        frames=None,
        data_path="d.slp",
        device="cpu",
        batch_size=4,
        peak_threshold=None,
        embeddings_path=None,
        save_embeddings="none",
    )
    base.update(over)
    return base


def _patch_embed_writer(monkeypatch):
    """Capture the kwargs threaded into ``predict_embeddings_to_h5``."""
    import sleap_nn.inference.embedding as emb_mod

    captured = {}

    def _fake(**kwargs):
        captured.update(kwargs)
        return "RET"

    monkeypatch.setattr(emb_mod, "predict_embeddings_to_h5", _fake)
    return captured


def test_cli_threads_save_embeddings(monkeypatch):
    """``predict --save_embeddings slp`` threads the mode through (no path needed)."""
    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_has_embedding_model", lambda *_: True)
    captured = _patch_embed_writer(monkeypatch)
    out = cli._run_inference_impl(**_impl_kwargs(save_embeddings="slp"))
    assert out == "RET"
    assert captured["save_embeddings"] == "slp"
    assert captured["output_path"] is None  # no --embeddings_path needed for slp


def test_cli_both_routes_with_embeddings_path(monkeypatch):
    """``--save_embeddings both`` with an --embeddings_path threads both through."""
    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_has_embedding_model", lambda *_: True)
    captured = _patch_embed_writer(monkeypatch)
    cli._run_inference_impl(
        **_impl_kwargs(save_embeddings="both", embeddings_path="o.h5")
    )
    assert captured["save_embeddings"] == "both"
    assert captured["output_path"] == "o.h5"


def test_cli_embedding_model_off_requires_path(monkeypatch):
    """Embedding model + default OFF + no path still requires --embeddings_path."""
    import click

    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_has_embedding_model", lambda *_: True)
    with pytest.raises(click.UsageError):
        cli._run_inference_impl(**_impl_kwargs())  # off + no embeddings_path


def test_cli_save_embeddings_rejected_for_non_embedding(monkeypatch):
    """``--save_embeddings`` on a non-embedding model is a UsageError."""
    import click

    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_has_embedding_model", lambda *_: False)
    with pytest.raises(click.UsageError):
        cli._run_inference_impl(**_impl_kwargs(save_embeddings="slp"))


def test_cli_embeddings_path_rejected_for_non_embedding(monkeypatch):
    """``--embeddings_path`` on a non-embedding model is a UsageError (unchanged)."""
    import click

    import sleap_nn.cli as cli

    monkeypatch.setattr(cli, "_has_embedding_model", lambda *_: False)
    with pytest.raises(click.UsageError):
        cli._run_inference_impl(**_impl_kwargs(embeddings_path="o.h5"))
