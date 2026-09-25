"""The embedding route's persistence + CLI behavior, through `sleap-nn predict`.

Embedding-stack review, PR E (FINDINGS #12, #13 + the Tier-3 inference/CLI items).
Every test drives the real command via ``CliRunner`` on a tiny random-weights
embedding model, and each one failed on ``8d3ef25c``:

* E1 -- ``--save_embeddings slp`` without ``-t`` embedded only TRACKED detections:
  an untracked file died with a raw ``ValueError``, a partially-tracked one got
  vectors on 2 of 8 detections.
* E2 -- vectors already on the input were never cleared, so a detection this pass
  skips kept a previous model's vector (mixed-model file), or ``save_slp`` crashed on
  the dimension mismatch after the whole pass.
* E3 -- a remote ``.slp`` input: default output ``f"{url}.embeddings.slp"``,
  ``--headers`` accepted but not forwarded, signed URLs rejected as "not a .slp".
* E4 -- ``-o`` naming the input ``.pkg.slp`` destroyed its stored frames (also on the
  generic retrack route).
* E5 -- the fused route silently ignored ``--output_format`` / ``--embed``, and every
  ``--video_index`` run defaulted to one output path.
* E6 -- ``class_output`` validated only after the inference pass; two embedding
  models silently used the first; ``-t --features keypoints`` computed vectors and
  threw them away; lone-route user errors were tracebacks.
* E7 -- one full-frame decode per crop.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio
from click.testing import CliRunner
from omegaconf import OmegaConf

from sleap_nn.cli import cli
from tests.inference.test_embedding_persistence import (  # noqa: F401
    _DIM,
    _write_video,
    embedding_model_dir,
)

CENTROID_CKPT = "tests/assets/model_ckpts/minimal_instance_centroid"
MULTICLASS_BU_CKPT = "tests/assets/model_ckpts/minimal_instance_multiclass_bottomup"
PKG_SLP = "tests/assets/datasets/minimal_instance.pkg.slp"

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _flat(result) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output (rich-click wraps by width)."""
    text = result.output + (str(result.exception) if result.exception else "")
    return " ".join(_ANSI.sub("", text).replace("│", " ").split())


def _predict(*args):
    return CliRunner().invoke(cli, ["predict", *map(str, args)])


def _vectors(path):
    """``[(frame_idx, carrier, vector-or-None)]`` for every detection in ``path``."""
    out = []
    for lf in sio.load_slp(str(path)).labeled_frames:
        for inst in lf.instances:
            emb = inst.identity_embedding
            out.append((lf.frame_idx, "pose", None if emb is None else emb.vector))
        for m in getattr(lf, "masks", None) or []:
            emb = m.identity_embedding
            out.append((lf.frame_idx, "mask", None if emb is None else emb.vector))
    return out


def _pose_labels(video, n_frames, tracked=lambda fi, k: False, nan=False):
    """``n_frames`` frames x 2 predicted instances (+1 all-NaN pose if ``nan``)."""
    skel = sio.Skeleton(nodes=["a", "b"])
    track = sio.Track("a0")
    lfs = []
    for fi in range(n_frames):
        insts = []
        for k, x in enumerate((16.0, 44.0)):
            inst = sio.PredictedInstance.from_numpy(
                np.array([[x, 20.0], [x + 6, 28.0]]), skeleton=skel, score=0.9
            )
            if tracked(fi, k):
                inst.track = track
            insts.append(inst)
        if nan:
            insts.append(
                sio.PredictedInstance.from_numpy(
                    np.full((2, 2), np.nan), skeleton=skel, score=0.9, track=track
                )
            )
        lfs.append(sio.LabeledFrame(video=video, frame_idx=fi, instances=insts))
    return sio.Labels(
        labeled_frames=lfs, videos=[video], skeletons=[skel], tracks=[track]
    )


@pytest.fixture
def untracked_slp(tmp_path):
    video = _write_video(tmp_path / "u.mp4", n=4)
    path = tmp_path / "untracked.slp"
    sio.save_slp(_pose_labels(video, 4), str(path), embed=False)
    return path


@pytest.fixture
def tracked_slp(tmp_path):
    """Every detection tracked, so the E1 fix does not affect what these tests see."""
    video = _write_video(tmp_path / "t.mp4", n=4)
    labels = _pose_labels(video, 4, tracked=lambda fi, k: True)
    path = tmp_path / "tracked.slp"
    sio.save_slp(labels, str(path), embed=False)
    return path


@pytest.fixture
def partial_slp(tmp_path):
    """8 detections, only 2 tracked (instance 0 of frames 0-1)."""
    video = _write_video(tmp_path / "p.mp4", n=4)
    labels = _pose_labels(video, 4, tracked=lambda fi, k: fi < 2 and k == 0)
    path = tmp_path / "partial.slp"
    sio.save_slp(labels, str(path), embed=False)
    return path


# ── E1: every detection is embedded, tracked or not (FINDINGS #12) ─────────────


def test_untracked_input_gets_a_vector_on_every_detection(
    embedding_model_dir, untracked_slp, tmp_path
):
    out = tmp_path / "out.slp"
    result = _predict(
        "-m", embedding_model_dir, "-i", untracked_slp, "--save_embeddings", "slp",
        "-o", out, "--device", "cpu",
    )  # fmt: skip

    assert result.exit_code == 0, _flat(result)
    vectors = _vectors(out)
    assert len(vectors) == 8
    assert all(v is not None and v.shape == (_DIM,) for _, _, v in vectors)


def test_partially_tracked_input_gets_a_vector_on_every_detection(
    embedding_model_dir, partial_slp, tmp_path
):
    """The reviewer's partial file got 2 of 8 vectors; its WF1 retrack then split 2
    animals into 7 tracks, with no warning."""
    out = tmp_path / "out.slp"
    result = _predict(
        "-m", embedding_model_dir, "-i", partial_slp, "--save_embeddings", "slp",
        "-o", out, "--device", "cpu",
    )  # fmt: skip

    assert result.exit_code == 0, _flat(result)
    vectors = _vectors(out)
    assert len(vectors) == 8
    assert sum(v is not None for _, _, v in vectors) == 8
    # The tracks the input had pass through untouched.
    labels = sio.load_slp(str(out))
    assert sum(i.track is not None for lf in labels for i in lf.instances) == 2


# ── E2: vectors already on the input never survive (FINDINGS #13) ──────────────


def _stale_slp(tmp_path, dim):
    """Tracked + untracked + all-NaN poses, plus one mask (the minority carrier),
    every detection carrying a stale all-ones vector of ``dim``."""
    video = _write_video(tmp_path / "s.mp4", n=3)
    labels = _pose_labels(video, 3, tracked=lambda fi, k: k == 0, nan=True)
    yy, xx = np.ogrid[:64, :64]
    mask = sio.PredictedSegmentationMask.from_numpy(
        ((yy - 32) ** 2 + (xx - 32) ** 2) <= 81, score=0.9
    )
    labels.labeled_frames[0].masks = [mask]
    for lf in labels.labeled_frames:
        for det in list(lf.instances) + list(lf.masks or []):
            det.identity_embedding = sio.Embedding(np.ones(dim, np.float32))
    path = tmp_path / f"stale{dim}.slp"
    sio.save_slp(labels, str(path), embed=False, save_embedding_vectors=True)
    return path


@pytest.mark.parametrize("stale_dim", [_DIM, 8], ids=["same-dim", "other-dim"])
def test_stale_vectors_are_replaced_or_cleared(
    embedding_model_dir, tmp_path, stale_dim
):
    """Same dim: the file silently mixed two models' vectors. Other dim: `save_slp`
    raised `inconsistent dimensions` after the whole embedding pass."""
    src = _stale_slp(tmp_path, stale_dim)
    out = tmp_path / "out.slp"
    result = _predict(
        "-m", embedding_model_dir, "-i", src, "--save_embeddings", "slp",
        "-o", out, "--device", "cpu",
    )  # fmt: skip

    assert result.exit_code == 0, _flat(result)
    stale = np.ones(stale_dim, np.float32)
    vectors = _vectors(out)
    assert len(vectors) == 3 * 3 + 1
    for _, _, vec in vectors:
        assert vec is None or not (
            vec.shape == stale.shape and np.allclose(vec, stale)
        ), "a stale vector survived"
    embedded = [v for _, _, v in vectors if v is not None]
    # The 6 finite poses are embedded; the all-NaN poses and the mask (not the
    # carrier this pose-dominant file is embedded on) are left without a vector.
    assert len(embedded) == 6
    assert all(v.shape == (_DIM,) for v in embedded)


# ── E3: a remote .slp input (Tier 3) ───────────────────────────────────────────


class _RecordingHandler(SimpleHTTPRequestHandler):
    seen_headers: list = []

    def do_GET(self):  # noqa: N802 -- http.server API
        type(self).seen_headers.append(dict(self.headers))
        super().do_GET()

    def do_HEAD(self):  # noqa: N802 -- http.server API
        type(self).seen_headers.append(dict(self.headers))
        super().do_HEAD()

    def log_message(self, format, *args):  # noqa: A002 -- http.server API
        pass


@pytest.fixture
def http_dir(tmp_path):
    """Serve a directory over HTTP; yields ``(root, base_url, seen_headers)``."""
    pytest.importorskip("fsspec")
    pytest.importorskip("aiohttp")
    root = tmp_path / "served"
    root.mkdir()
    _RecordingHandler.seen_headers = []
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(_RecordingHandler, directory=str(root))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield root, f"http://127.0.0.1:{server.server_address[1]}", (
            _RecordingHandler.seen_headers
        )
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("query", ["", "?sig=abc"], ids=["plain", "signed"])
def test_remote_slp_lone_route(
    embedding_model_dir, tracked_slp, http_dir, tmp_path, monkeypatch, query
):
    """Default output lands in the cwd under the URL's file name; --headers reach
    the server; a signed URL is still recognized as a .slp."""
    root, base, seen = http_dir
    shutil.copy(tracked_slp, root / "untracked.slp")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    result = _predict(
        "-m", embedding_model_dir, "-i", f"{base}/untracked.slp{query}",
        "--save_embeddings", "slp", "--device", "cpu",
        "--headers", json.dumps({"X-Emb-Review": "e3"}),
    )  # fmt: skip

    assert result.exit_code == 0, _flat(result)
    out = work / "untracked.slp.embeddings.slp"
    assert out.exists(), sorted(p.name for p in work.iterdir())
    assert len(_vectors(out)) == 8
    assert any(h.get("X-Emb-Review") == "e3" for h in seen), seen


def test_lone_route_rejects_a_video_input_as_a_usage_error(
    embedding_model_dir, tmp_path
):
    """A usage error (exit 2, message), not a raw ValueError traceback."""
    video = tmp_path / "clip.mp4"
    _write_video(video)
    result = _predict(
        "-m", embedding_model_dir, "-i", video, "--save_embeddings", "slp",
        "--device", "cpu",
    )  # fmt: skip

    assert result.exit_code == 2, _flat(result)
    assert "must be a .slp of detections" in _flat(result)


# ── E4: -o naming the input .pkg.slp would destroy its frames (Tier 3) ─────────


def _assert_frames_intact(path):
    labels = sio.load_slp(str(path))
    assert labels.labeled_frames[0].image is not None


def test_output_over_the_input_pkg_slp_is_refused(embedding_model_dir, tmp_path):
    pkg = tmp_path / "in.pkg.slp"
    shutil.copy(PKG_SLP, pkg)
    result = _predict(
        "-m", embedding_model_dir, "-i", pkg, "--save_embeddings", "slp",
        "-o", pkg, "--device", "cpu",
    )  # fmt: skip

    assert result.exit_code == 2, _flat(result)
    assert "Refusing to write" in _flat(result)
    _assert_frames_intact(pkg)


def test_retrack_output_over_the_input_pkg_slp_is_refused(tmp_path):
    """The generic `save_predictions` had the same exposure (retrack route shown);
    with `--embed true` the input was deleted outright."""
    pkg = tmp_path / "in.pkg.slp"
    shutil.copy(PKG_SLP, pkg)
    result = _predict("-i", pkg, "-t", "-o", pkg)

    assert result.exit_code != 0
    assert "Refusing to write" in _flat(result)
    _assert_frames_intact(pkg)


# ── E5: the fused route's output options (Tier 3) ──────────────────────────────


@pytest.mark.parametrize(
    "flag,value", [("--output_format", "analysis_h5"), ("--embed", "true")]
)
def test_fused_route_rejects_output_shaping_flags(
    embedding_model_dir, tmp_path, monkeypatch, flag, value
):
    """It wrote an SLP-schema file named `out.h5`, frames not embedded."""
    import sleap_nn.cli as cli_mod

    monkeypatch.setattr(
        cli_mod, "_run_in_memory_new_flow", lambda *a, **k: pytest.fail("ran")
    )
    result = _predict(
        "-m", CENTROID_CKPT, "-m", embedding_model_dir, "-i", PKG_SLP,
        "--save_embeddings", "slp", flag, value, "-o", tmp_path / "out.h5",
    )  # fmt: skip

    assert result.exit_code == 2, _flat(result)
    assert "does not support" in _flat(result)
    assert flag in _flat(result)
    assert not (tmp_path / "out.h5").exists()


@pytest.fixture
def two_video_slp(tmp_path):
    """Two videos of the minimal_instance frame, one user-labeled frame each."""
    import imageio.v3 as iio

    src = sio.load_slp(PKG_SLP)
    img = src[0].image[..., 0]
    videos, lfs = [], []
    for k in range(2):
        path = tmp_path / f"clip{k}.mp4"
        iio.imwrite(path, np.stack([img, img]), fps=5)
        video = sio.load_video(str(path))
        videos.append(video)
        insts = [
            sio.Instance.from_numpy(i.numpy(), skeleton=src.skeletons[0])
            for i in src[0].instances
        ]
        lfs.append(sio.LabeledFrame(video=video, frame_idx=0, instances=insts))
    path = tmp_path / "project.slp"
    sio.save_slp(
        sio.Labels(labeled_frames=lfs, videos=videos, skeletons=src.skeletons),
        str(path),
        embed=False,
    )
    return path


def test_fused_video_index_runs_do_not_share_a_default_path(
    embedding_model_dir, two_video_slp
):
    """Both `--video_index` runs defaulted to `<input>.embeddings.slp`, so the
    second overwrote the first."""
    for k in (0, 1):
        result = _predict(
            "-m", CENTROID_CKPT, "-m", embedding_model_dir, "-i", two_video_slp,
            "--video_index", k, "--save_embeddings", "slp", "--device", "cpu",
        )  # fmt: skip
        assert result.exit_code == 0, _flat(result)

    outs = {k: two_video_slp.parent / f"project.clip{k}.embeddings.slp" for k in (0, 1)}
    for k, out in outs.items():
        assert out.exists(), sorted(p.name for p in two_video_slp.parent.iterdir())
        labels = sio.load_slp(str(out))
        assert (
            Path(labels.videos[0].filename).resolve()
            == (two_video_slp.parent / f"clip{k}.mp4").resolve()
        )
        assert all(v is not None for _, _, v in _vectors(out))


# ── E6: errors surface before the work, as usage errors (Tier 3) ───────────────


def test_bad_class_output_fails_before_inference(tmp_path, monkeypatch):
    """`class_output` was validated in `to_labels`, after `Predicting... 100%`."""
    from sleap_nn.inference.layers.base import InferenceLayer

    model = tmp_path / "mc_bu"
    shutil.copytree(MULTICLASS_BU_CKPT, model)
    cfg = OmegaConf.load(model / "training_config.yaml")
    cfg.model_config.head_configs.multi_class_bottomup.class_maps.class_output = (
        "category"
    )
    OmegaConf.save(cfg, model / "training_config.yaml")

    forwards = []
    real_predict = InferenceLayer.predict
    monkeypatch.setattr(
        InferenceLayer,
        "predict",
        lambda self, *a, **k: forwards.append(1) or real_predict(self, *a, **k),
    )
    result = _predict(
        "-m", model, "-i", PKG_SLP, "-o", tmp_path / "out.slp", "--device", "cpu"
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, NotImplementedError), _flat(result)
    assert "class_output='category'" in _flat(result)
    assert not forwards, "the inference pass ran before class_output was checked"


def test_two_embedding_models_are_rejected(embedding_model_dir, tracked_slp):
    result = _predict(
        "-m", embedding_model_dir, "-m", embedding_model_dir, "-i", tracked_slp,
        "--save_embeddings", "slp", "--device", "cpu",
    )  # fmt: skip

    assert result.exit_code == 2, _flat(result)
    assert "pass exactly one" in _flat(result)


def test_geometry_only_tracking_that_discards_the_vectors_is_rejected(
    embedding_model_dir, untracked_slp, monkeypatch
):
    """It logged "tracking by appearance" while tracking by OKS, and stripped the
    vectors it had just computed."""
    import sleap_nn.inference.embedding as emb_mod

    monkeypatch.setattr(
        emb_mod, "embed_labels", lambda *a, **k: pytest.fail("embedded anyway")
    )
    result = _predict(
        "-m", embedding_model_dir, "-i", untracked_slp, "-t",
        "--features", "keypoints", "--device", "cpu",
    )  # fmt: skip

    assert result.exit_code == 2, _flat(result)
    assert "computed and then discarded" in _flat(result)


def test_geometry_tracking_that_keeps_the_vectors_is_allowed(
    embedding_model_dir, untracked_slp, tmp_path
):
    """ "Persist vectors, track by geometry" is a real combination."""
    out = tmp_path / "out.slp"
    result = _predict(
        "-m", embedding_model_dir, "-i", untracked_slp, "-t",
        "--features", "keypoints", "--save_embeddings", "slp", "-o", out,
        "--device", "cpu",
    )  # fmt: skip

    assert result.exit_code == 0, _flat(result)
    labels = sio.load_slp(str(out))
    assert all(i.track is not None for lf in labels for i in lf.instances)
    assert all(v is not None for _, _, v in _vectors(out))


# ── E7: one decode per frame, identical vectors (Tier 3) ───────────────────────


def test_each_frame_is_decoded_once_and_vectors_are_unchanged(
    embedding_model_dir, tmp_path, monkeypatch
):
    from sleap_nn.data.custom_datasets import EmbeddingDataset

    video = _write_video(tmp_path / "d.mp4", n=5)
    skel = sio.Skeleton(nodes=["a", "b"])
    tracks = [sio.Track(f"animal{k}") for k in range(4)]
    lfs = [
        sio.LabeledFrame(
            video=video,
            frame_idx=fi,
            instances=[
                sio.PredictedInstance.from_numpy(
                    np.array([[x, 20.0], [x + 4, 28.0]]),
                    skeleton=skel,
                    score=0.9,
                    track=track,
                )
                for x, track in zip((10.0, 22.0, 34.0, 46.0), tracks)
            ],
        )
        for fi in range(5)
    ]
    src = tmp_path / "dense.slp"
    sio.save_slp(
        sio.Labels(labeled_frames=lfs, videos=[video], skeletons=[skel], tracks=tracks),
        str(src),
        embed=False,
    )

    decodes = []
    real_getitem = sio.Video.__getitem__
    monkeypatch.setattr(
        sio.Video,
        "__getitem__",
        lambda self, idx: decodes.append(idx) or real_getitem(self, idx),
    )

    def run(out):
        result = _predict(
            "-m", embedding_model_dir, "-i", src, "--save_embeddings", "slp",
            "-o", out, "--device", "cpu",
        )  # fmt: skip
        assert result.exit_code == 0, _flat(result)
        return [v for _, _, v in _vectors(out)]

    cached = run(tmp_path / "cached.slp")
    assert len(decodes) == 5, f"{len(decodes)} decodes for 5 frames x 4 detections"

    # Reference: the per-crop decode the dataset did before.
    monkeypatch.setattr(
        EmbeddingDataset,
        "_read_frame",
        lambda self, li, fi: self.labels_list[li][fi].image,
    )
    uncached = run(tmp_path / "uncached.slp")
    assert len(cached) == len(uncached) == 20
    for a, b in zip(cached, uncached):
        np.testing.assert_array_equal(a, b)


def test_crop_batch_defaults_to_the_embedding_stage_size(
    embedding_model_dir, untracked_slp, monkeypatch
):
    """`-b` defaults to 4 frames for detection; the embedder got 4 crops/pass."""
    import sleap_nn.inference.embedding as emb_mod

    seen = []
    monkeypatch.setattr(
        emb_mod,
        "predict_embeddings_to_slp",
        lambda **k: seen.append(k["batch_size"]) or "out.slp",
    )
    _predict("-m", embedding_model_dir, "-i", untracked_slp, "--save_embeddings", "slp")
    _predict(
        "-m", embedding_model_dir, "-i", untracked_slp, "--save_embeddings", "slp",
        "-b", "2",
    )  # fmt: skip

    assert seen == [64, 2]


# ── E8: `scale` is not applied to embedding crops, on either side ──────────────


def test_config_scale_does_not_change_inference_vectors(
    embedding_model_dir, tracked_slp, tmp_path
):
    """Training's `EmbeddingDataset` never applies `scale`; neither does inference
    (it reuses that dataset). The comments claiming inference DOES scale were
    stale, and the warning fired on every inference for `scale != 1`."""
    half = tmp_path / "emb_scale_half"
    shutil.copytree(embedding_model_dir, half)
    cfg = OmegaConf.load(half / "training_config.yaml")
    cfg.data_config.preprocessing.scale = 0.5
    OmegaConf.save(cfg, half / "training_config.yaml")

    outs = {}
    for name, model in (("one", embedding_model_dir), ("half", half)):
        outs[name] = tmp_path / f"{name}.slp"
        result = _predict(
            "-m", model, "-i", tracked_slp, "--save_embeddings", "slp",
            "-o", outs[name], "--device", "cpu",
        )  # fmt: skip
        assert result.exit_code == 0, _flat(result)
        # The "inference scales its crops" warning fired on every inference.
        assert "preprocessing.scale" not in _flat(result)

    for (_, _, a), (_, _, b) in zip(_vectors(outs["one"]), _vectors(outs["half"])):
        np.testing.assert_array_equal(a, b)
    prov = sio.load_slp(str(outs["half"])).provenance
    assert (prov.get("inference_config") or {}).get("scale") == 1.0


# ── E6: API-only corners (no CLI path reaches these) ───────────────────────────


def test_labels_without_a_data_path_require_an_output_path(
    embedding_model_dir, tracked_slp, tmp_path, monkeypatch
):
    """The default was `f"{data_path}.embeddings.slp"`, i.e. `None.embeddings.slp`
    written into the cwd."""
    from sleap_nn.inference.embedding import predict_embeddings_to_slp

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    with pytest.raises(ValueError, match="output_path is required"):
        predict_embeddings_to_slp(
            [embedding_model_dir],
            labels=sio.load_slp(str(tracked_slp)),
            device="cpu",
            save_embeddings="slp",
        )
    assert not list(work.iterdir())


def test_include_untracked_names_each_detection_by_its_own_track(
    embedding_model_dir, partial_slp
):
    """`group_id` is a placeholder 0 under `include_untracked`, so every returned
    name was `class_names[0]`; the docstring promises "" for untracked."""
    from sleap_nn.inference.embedding import embed_labels

    _, names, n_attached, _ = embed_labels(
        embedding_model_dir,
        sio.load_slp(str(partial_slp)),
        device="cpu",
        include_untracked=True,
    )

    assert n_attached == 8
    assert sorted(names) == [""] * 6 + ["a0"] * 2
