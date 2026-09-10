"""The embedding (re-ID) routes must honor the predict-pipeline hardening.

#745 adds three routes -- lone embed, lone embed + track (WF2), and fused
detect->embed->track (WF3) -- that bypass ``run.predict``'s packaging. The
July-August hardening (#711-#717, #728, #732) established four behaviors on
``predict`` that a new output path can silently lose, so each is pinned here:

1. **Empty-frame retention** (#714, #717) -- a frame with no detections must survive
   into the output rather than being dropped.
2. **Provenance** (#728) -- the output must record model paths, the RESOLVED crop
   geometry, versions and timings; the tracking routes must also record the tracker
   parameters, as the retrack route does.
3. **``--input_scale`` / geometry overrides** (#716) -- honored on the fused route's
   detection stage, and REJECTED on the lone route (crop geometry has to come from
   the trained config or the model embeds crops it never saw).
4. **CLI frame filters** (#712, #732) -- likewise forwarded to the fused detection
   stage and rejected on the lone route.

Finding: (2) was absent entirely -- both embedding saves produced a ``.slp`` whose
provenance held only sleap-io's ``filename``.
"""

from __future__ import annotations

import numpy as np
import pytest

import sleap_io as sio

from sleap_nn.inference.tracking import TrackerConfig

# Reuse the canonical embedding inference fixtures.
from tests.inference.test_embedding_persistence import (  # noqa: F401
    _build_embedding_config,
    _write_video,
    embedding_model_dir,
)

SLP = "tests/assets/datasets/minimal_instance.pkg.slp"
CENTROID_CKPT = "tests/assets/model_ckpts/minimal_instance_centroid"


@pytest.fixture
def slp_with_empty_frame(tmp_path):
    """A tracked ``.slp`` whose middle frame carries NO detections."""
    vid = _write_video(tmp_path / "gap.mp4", n=4)
    skel = sio.Skeleton(nodes=["a", "b"])
    t0, t1 = sio.Track("a0"), sio.Track("a1")
    lfs = []
    for fi in range(4):
        if fi == 2:
            lfs.append(sio.LabeledFrame(video=vid, frame_idx=fi, instances=[]))
            continue
        insts = [
            sio.PredictedInstance.from_numpy(
                np.array([[x + fi, 20.0], [x + fi + 6, 28.0]]),
                skeleton=skel,
                score=0.9,
                point_scores=np.ones(2),
                track=t,
            )
            for x, t in ((18.0, t0), (44.0, t1))
        ]
        lfs.append(sio.LabeledFrame(video=vid, frame_idx=fi, instances=insts))
    path = str(tmp_path / "gap.slp")
    sio.save_slp(
        sio.Labels(labeled_frames=lfs, videos=[vid], skeletons=[skel], tracks=[t0, t1]),
        path,
        embed=False,
    )
    return path


def _embed(model_dir, src, out, **kw):
    from sleap_nn.inference.embedding import predict_embeddings_to_slp

    return predict_embeddings_to_slp(
        [model_dir],
        src,
        output_path=out,
        device="cpu",
        batch_size=4,
        save_embeddings="slp",
        **kw,
    )


# ── 1. empty-frame retention (#714, #717) ────────────────────────────────────


@pytest.mark.parametrize("tracking", [False, True])
def test_empty_frames_survive_the_embedding_routes(
    embedding_model_dir, slp_with_empty_frame, tmp_path, tracking
):
    """4 frames in (one empty) -> 4 frames out, on both lone routes."""
    kw = {}
    if tracking:
        kw["tracker_config"] = TrackerConfig(
            features="embeddings",
            features_explicit=True,
            scoring_method="cosine_sim",
            scoring_method_explicit=True,
        )
    out = _embed(
        embedding_model_dir,
        slp_with_empty_frame,
        str(tmp_path / f"out_{tracking}.slp"),
        **kw,
    )
    labels = sio.load_slp(out)
    assert len(labels.labeled_frames) == 4, "a frame was dropped"
    assert any(
        len(lf.instances) == 0 for lf in labels.labeled_frames
    ), "the empty frame was dropped"
    assert sorted(lf.frame_idx for lf in labels.labeled_frames) == [0, 1, 2, 3]


# ── 2. provenance (#728) ─────────────────────────────────────────────────────


def test_lone_embed_records_provenance(
    embedding_model_dir, slp_with_empty_frame, tmp_path
):
    """The output must carry the lineage `predict` records, not just `filename`."""
    out = _embed(embedding_model_dir, slp_with_empty_frame, str(tmp_path / "prov.slp"))
    prov = sio.load_slp(out).provenance or {}

    assert prov.get("model_type") == "embedding"
    assert prov.get("model_paths") == [str(embedding_model_dir)]
    assert prov.get("sleap_nn_version") and prov.get("sleap_io_version")
    assert prov.get("inference_start_timestamp") and prov.get("inference_end_timestamp")
    assert prov.get("source_file", "").endswith(".slp")
    # The input's own provenance is preserved rather than overwritten.
    assert "input_provenance" in prov
    # #728's specific ask: the RESOLVED crop geometry is recorded. It comes from the
    # trained config on this route (the CLI rejects overriding it), so recording it
    # is the only way to know what the crops were.
    cfg = prov.get("inference_config") or {}
    assert cfg.get("crop_size") is not None
    # The EFFECTIVE geometry, with `embed_labels`' own defaults resolved -- a config
    # that omits `crop_centering` still cropped somehow, so recording nothing would
    # be the same blind spot #728 fixed.
    assert cfg.get("scale") == 1.0
    assert cfg.get("crop_centering") == "auto"
    for key in ("max_height", "max_width"):
        assert key in cfg, key
    assert cfg.get("embeddings_attached") == 6
    assert cfg.get("embedding_dim") == 16
    # No tracker ran, so no tracking parameters are claimed.
    assert not prov.get("tracking_config")


def test_embed_and_track_records_tracking_provenance(
    embedding_model_dir, slp_with_empty_frame, tmp_path
):
    """WF2 must also record the tracker parameters, as the retrack route does."""
    out = _embed(
        embedding_model_dir,
        slp_with_empty_frame,
        str(tmp_path / "prov_tracked.slp"),
        tracker_config=TrackerConfig(
            features="embeddings",
            features_explicit=True,
            scoring_method="cosine_sim",
            scoring_method_explicit=True,
            appearance_weight=0.0,
        ),
    )
    prov = sio.load_slp(out).provenance or {}
    tracking = prov.get("tracking_config") or {}
    assert tracking.get("features") == "embeddings"
    assert tracking.get("scoring_method") == "cosine_sim"
    assert "appearance_weight" in tracking, "the blend weight is not recorded"


def test_provenance_never_fails_the_run(
    embedding_model_dir, slp_with_empty_frame, tmp_path, monkeypatch
):
    """Provenance is metadata: a failure building it must not lose the output."""
    import sleap_nn.inference.embedding as emb

    def _boom(*a, **k):
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(emb, "_load_training_config", _boom, raising=False)
    monkeypatch.setattr(
        "sleap_nn.inference.loaders._load_training_config", _boom, raising=False
    )
    # The embedding pass itself needs the config, so this asserts the narrower
    # promise: the geometry lookup is guarded, not that the whole run survives.
    prov = emb._embedding_provenance(
        sio.load_slp(slp_with_empty_frame),
        embedding_model_dir,
        slp_with_empty_frame,
        start_time=__import__("datetime").datetime.now(),
        device="cpu",
        batch_size=4,
        save_embeddings="slp",
        n_attached=6,
        embedding_dim=16,
    )
    assert prov["model_type"] == "embedding"
    assert "crop_size" not in (prov.get("inference_config") or {})


# ── 3 & 4. geometry / filter overrides reach the FUSED detection stage ───────


@pytest.mark.parametrize(
    "flag,value,key",
    [
        ("--input_scale", "0.5", "input_scale"),
        ("--max_height", "512", "max_height"),
        ("--only_predicted_frames", None, "only_predicted_frames"),
        ("--frames", "0-2", "frames"),
    ],
)
def test_fused_route_forwards_overrides_to_the_detection_stage(
    embedding_model_dir, monkeypatch, flag, value, key
):
    """The lone route rejects these (see test_embeddings_flags); the FUSED route must
    honor them, because there they apply to real pixel inference."""
    from click.testing import CliRunner

    import sleap_nn.cli as cli_mod
    from sleap_nn.cli import cli

    seen = {}

    def _fake_detect(det_kwargs, paf_workers=0, save_output=True):
        seen.update(det_kwargs)
        raise RuntimeError("stop after the detection stage")

    monkeypatch.setattr(cli_mod, "_run_in_memory_new_flow", _fake_detect)
    args = [
        "predict",
        "-m",
        CENTROID_CKPT,
        "-m",
        embedding_model_dir,
        "-i",
        SLP,
        "--save_embeddings",
        "slp",
    ]
    args += [flag] if value is None else [flag, value]
    CliRunner().invoke(cli, args)

    assert seen, "the detection stage never ran"
    assert seen.get(key) not in (
        None,
        False,
        (),
    ), f"{flag} did not reach the fused detection stage: {seen.get(key)!r}"
