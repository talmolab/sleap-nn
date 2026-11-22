"""The ``.slp`` save options and the Python saves that take them.

The options are ``slp_version``, ``image_format``, ``image_quality``,
``compression`` and ``unlabeled_as``.

Unset options must pass nothing new to sleap-io, so every save is unchanged by
default; set options must reach the save.
"""

from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import sleap_io as sio

import sleap_nn.inference.save_options as so
from sleap_nn.inference.save_options import slp_save_kwargs, validate_save_options

ALL_SET = dict(
    slp_version="v2",
    image_format="png",
    image_quality=3,
    compression="gzip",
    unlabeled_as="absent",
)


def _params_with(*names):
    """A ``save_slp`` parameter set holding the base names plus ``names``."""
    base = {"labels", "filename", "embed", "restore_original_videos"}
    return lambda: frozenset(base | set(names))


def _predicted_labels(video=None, unlabeled=False):
    """One frame with one two-node predicted instance."""
    sk = sio.Skeleton(["a", "b"])
    video = video or sio.Video(filename="fake.mp4", open_backend=False)
    if unlabeled:  # node "b" not predicted: no location, marked not labeled
        inst = sio.PredictedInstance.from_numpy(
            np.array([[10.0, 20.0], [np.nan, np.nan]]),
            skeleton=sk,
            point_scores=np.array([0.9, np.nan]),
            score=0.9,
            labeled=np.array([True, False]),
        )
    else:
        inst = sio.PredictedInstance.from_numpy(
            np.array([[10.0, 20.0], [30.0, 40.0]]),
            skeleton=sk,
            point_scores=np.array([0.9, 0.8]),
            score=0.85,
        )
    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
    return sio.Labels([lf], videos=[video], skeletons=[sk])


def _format_id(path) -> float:
    with h5py.File(path, "r") as f:
        return float(f["metadata"].attrs["format_id"])


# ── the helper ───────────────────────────────────────────────────────────────


def test_no_options_passes_nothing():
    assert slp_save_kwargs() == {}
    assert slp_save_kwargs(embeds_frames=False) == {}


def test_bridge_without_slp_version_parameter(monkeypatch):
    """A sleap-io with only ``legacy_format``: v2/v3 map to it, auto to nothing."""
    monkeypatch.setattr(so, "_save_slp_parameters", _params_with("legacy_format"))
    assert slp_save_kwargs(slp_version="v2") == {"legacy_format": True}
    assert slp_save_kwargs(slp_version="v3") == {"legacy_format": False}
    assert slp_save_kwargs(slp_version="auto") == {}
    assert slp_save_kwargs(slp_version="V2") == {"legacy_format": True}


def test_bridge_with_slp_version_parameter(monkeypatch):
    """A sleap-io that takes ``slp_version`` gets it straight through."""
    monkeypatch.setattr(
        so, "_save_slp_parameters", _params_with("slp_version", "legacy_format")
    )
    for v in ("auto", "v2", "v3"):
        assert slp_save_kwargs(slp_version=v) == {"slp_version": v}


def test_installed_sleap_io_bridge():
    """The installed sleap-io: whichever of the two it has, v2 asks for 2.x."""
    kw = slp_save_kwargs(slp_version="v2")
    assert kw in ({"slp_version": "v2"}, {"legacy_format": True})


def test_every_option_reaches_the_kwargs(monkeypatch):
    monkeypatch.setattr(
        so,
        "_save_slp_parameters",
        _params_with(
            "legacy_format",
            "image_format",
            "image_quality",
            "compression",
            "unlabeled_as",
        ),
    )
    assert slp_save_kwargs(**ALL_SET) == {
        "legacy_format": True,
        "image_format": "png",
        "image_quality": 3,
        "compression": "gzip",
        "unlabeled_as": "absent",
    }


@pytest.mark.parametrize("name", ["compression", "unlabeled_as", "image_format"])
def test_option_the_installed_sleap_io_lacks_is_refused(monkeypatch, name):
    monkeypatch.setattr(so, "_save_slp_parameters", _params_with("legacy_format"))
    value = {"compression": "lzf", "unlabeled_as": "absent", "image_format": "png"}
    with pytest.raises(ValueError, match=name):
        slp_save_kwargs(**{name: value[name]})


def test_slp_version_without_either_parameter_is_refused(monkeypatch):
    monkeypatch.setattr(so, "_save_slp_parameters", _params_with())
    with pytest.raises(ValueError, match="slp_version"):
        slp_save_kwargs(slp_version="v2")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(slp_version="v4"), "slp_version"),
        (dict(image_format="tiff"), "image_format"),
        (dict(compression="zstd"), "compression"),
        (dict(unlabeled_as="drop"), "unlabeled_as"),
        (dict(image_quality=50), "image_quality needs"),
        (dict(image_format="hdf5", image_quality=5), "image_quality needs"),
        (dict(image_format="png", image_quality=10), "0-9"),
        (dict(image_format="jpg", image_quality=101), "0-100"),
        (dict(image_format="jpg", image_quality=-1), "0-100"),
        (dict(image_format="jpg", image_quality="high"), "integer"),
    ],
)
def test_validation_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        validate_save_options(**kwargs)
    with pytest.raises(ValueError, match=match):
        slp_save_kwargs(**kwargs)


def test_validation_normalizes():
    out = validate_save_options(
        slp_version="V3", image_format="JPEG", image_quality="90", compression="LZF"
    )
    assert out == {
        "slp_version": "v3",
        "image_format": "jpeg",
        "image_quality": 90,
        "compression": "lzf",
        "unlabeled_as": None,
    }


def test_image_options_without_embedding_warn():
    with pytest.warns(UserWarning, match="only to embedded frames"):
        slp_save_kwargs(image_format="png", embeds_frames=False)
    with pytest.warns(UserWarning, match="image_format and image_quality"):
        slp_save_kwargs(image_format="jpg", image_quality=80, embeds_frames=False)


def test_image_options_with_embedding_do_not_warn(recwarn):
    slp_save_kwargs(image_format="png", embeds_frames=True)
    slp_save_kwargs(compression="gzip", embeds_frames=False)
    assert not [w for w in recwarn if "embedded frames" in str(w.message)]


# ── save_predictions ─────────────────────────────────────────────────────────


def test_save_predictions_defaults_unchanged(tmp_path):
    from sleap_nn.inference.run import save_predictions

    with patch.object(sio.Labels, "save") as save:
        save_predictions(_predicted_labels(), tmp_path / "out.slp")
    assert save.call_args.kwargs == {
        "embed": False,
        "restore_original_videos": False,
        "save_embedding_vectors": False,
    }


def test_save_predictions_forwards_options(tmp_path, monkeypatch):
    from sleap_nn.inference.run import save_predictions

    monkeypatch.setattr(
        so,
        "_save_slp_parameters",
        _params_with("slp_version", "compression", "unlabeled_as"),
    )
    with patch.object(sio.Labels, "save") as save:
        save_predictions(
            _predicted_labels(),
            tmp_path / "out.slp",
            slp_version="v2",
            compression="lzf",
            unlabeled_as="absent",
        )
    kw = save.call_args.kwargs
    assert kw["slp_version"] == "v2"
    assert kw["compression"] == "lzf"
    assert kw["unlabeled_as"] == "absent"


def test_save_predictions_image_format_needs_embedding(tmp_path):
    from sleap_nn.inference.run import save_predictions

    with patch.object(sio.Labels, "save") as save:
        with pytest.warns(UserWarning, match="only to embedded frames"):
            save_predictions(
                _predicted_labels(), tmp_path / "o.slp", image_format="jpg"
            )
        save_predictions(
            _predicted_labels(),
            tmp_path / "o2.slp",
            embed=True,
            image_format="jpg",
            image_quality=80,
        )
    kw = save.call_args.kwargs
    assert kw["embed"] is True
    assert kw["image_format"] == "jpg" and kw["image_quality"] == 80


def test_save_predictions_invalid_option_writes_nothing(tmp_path):
    from sleap_nn.inference.run import save_predictions

    with pytest.raises(ValueError, match="compression"):
        save_predictions(_predicted_labels(), tmp_path / "o.slp", compression="zip")
    assert not (tmp_path / "o.slp").exists()


def test_save_predictions_options_without_slp_warn(tmp_path):
    from sleap_nn.inference.run import save_predictions

    with patch("sleap_nn.inference.run.save_analysis_h5_files", return_value=[]):
        with pytest.warns(UserWarning, match="only to .slp output"):
            save_predictions(
                _predicted_labels(),
                tmp_path / "o.slp",
                output_format="analysis_h5",
                slp_version="v2",
            )


def test_round_trip_v2_and_default(tmp_path):
    """A real save: ``v2`` writes the 2.x format id, the default writes 3.0."""
    from sleap_nn.inference.run import save_predictions

    v2, default = tmp_path / "v2.slp", tmp_path / "default.slp"
    save_predictions(_predicted_labels(), v2, slp_version="v2")
    save_predictions(_predicted_labels(), default)
    assert _format_id(v2) < 3.0
    assert _format_id(default) >= 3.0
    back = sio.load_slp(str(v2), open_videos=False)
    np.testing.assert_allclose(back[0].instances[0].numpy(), [[10, 20], [30, 40]])

    gz = tmp_path / "gz.slp"
    save_predictions(_predicted_labels(), gz, compression="gzip")
    assert len(sio.load_slp(str(gz), open_videos=False)) == 1


def test_round_trip_unlabeled_points(tmp_path):
    """``v2`` refuses unlabeled points by default; ``absent`` writes them hidden."""
    from sleap_nn.inference.run import save_predictions

    out = tmp_path / "u.slp"
    with pytest.raises(ValueError):
        save_predictions(_predicted_labels(unlabeled=True), out, slp_version="v2")
    assert not out.exists()
    with pytest.warns(UserWarning):
        save_predictions(
            _predicted_labels(unlabeled=True),
            out,
            slp_version="v2",
            unlabeled_as="absent",
        )
    assert _format_id(out) < 3.0
    pts = sio.load_slp(str(out), open_videos=False)[0].instances[0].points
    assert not pts["visible"][1] and np.isnan(pts["xy"][1]).all()
    assert pts["visible"][0]


# ── predict() ────────────────────────────────────────────────────────────────


def test_predict_refuses_bad_option_before_loading_models():
    from sleap_nn.inference.run import predict

    with pytest.raises(ValueError, match="slp_version"):
        predict("missing.mp4", model_paths=["/nope"], slp_version="v9")


def test_predict_saves_with_options(tmp_path, monkeypatch):
    """predict() hands the options to save_predictions; unset ones stay None."""
    import sleap_nn.inference.run as run

    fake_predictor = MagicMock()
    fake_predictor.predict.return_value = _predicted_labels()
    monkeypatch.setattr(run, "_build_predictor", lambda **kw: (fake_predictor, None))
    seen = {}
    monkeypatch.setattr(run, "save_predictions", lambda *a, **kw: seen.update(kw))

    run.predict(
        "video.mp4",
        model_paths=["m"],
        device="cpu",
        output_path=str(tmp_path / "o.slp"),
    )
    assert all(seen[k] is None for k in so.SAVE_OPTION_NAMES)

    run.predict(
        "video.mp4",
        model_paths=["m"],
        device="cpu",
        output_path=str(tmp_path / "o.slp"),
        slp_version="v2",
        compression="gzip",
    )
    assert seen["slp_version"] == "v2" and seen["compression"] == "gzip"


def test_predict_sam_route_forwards_options(monkeypatch):
    from sleap_nn.inference.run import predict

    captured = {}

    def fake_run_sam_segmentation(source, mask_backend, **kwargs):
        captured.update(kwargs)
        return sio.Labels()

    monkeypatch.setattr(
        "sleap_nn.inference.sam.run_sam_segmentation", fake_run_sam_segmentation
    )
    predict(sio.Labels(), mask_backend="sam", device="cpu", slp_version="v2")
    assert captured["slp_version"] == "v2"
    assert captured["compression"] is None


# ── SAM writer ───────────────────────────────────────────────────────────────


class _NoMaskBackend:
    pred_iou_min = 0.5

    def masks(self, image, prompts):
        return [], []


@pytest.mark.parametrize("opts", [{}, {"slp_version": "v2", "compression": "gzip"}])
def test_run_sam_segmentation_save(minimal_instance, tmp_path, monkeypatch, opts):
    from sleap_nn.inference.sam import run_sam_segmentation
    from sleap_nn.inference.sam.mask_layer import SamSegmentationLayer

    monkeypatch.setattr(SamSegmentationLayer, "masks_for_frame", lambda *a: [])
    monkeypatch.setattr(
        so, "_save_slp_parameters", _params_with("slp_version", "compression")
    )
    with patch.object(sio.Labels, "save") as save:
        run_sam_segmentation(
            sio.load_slp(str(minimal_instance)),
            "sam",
            backend=_NoMaskBackend(),
            output_path=str(tmp_path / "m.slp"),
            **opts,
        )
    expected = {"embed": False, "restore_original_videos": False, **opts}
    assert save.call_args.kwargs == expected


# ── stream-to-file writer ────────────────────────────────────────────────────


def _writer_save_kwargs(tmp_path, save_options):
    from sleap_nn.inference.writer import IncrementalLabelsWriter

    def _touch(path, **kwargs):
        open(path, "wb").close()

    writer = IncrementalLabelsWriter(
        path=str(tmp_path / "s.slp"),
        skeleton=sio.Skeleton(["a"]),
        save_options=save_options,
    )
    with patch.object(sio.Labels, "save", side_effect=_touch) as save:
        writer.close()
    return save.call_args.kwargs


def test_writer_defaults_unchanged(tmp_path):
    assert _writer_save_kwargs(tmp_path, None) == {
        "format": "slp",
        "restore_original_videos": False,
    }


def test_writer_forwards_options(tmp_path, monkeypatch):
    monkeypatch.setattr(
        so, "_save_slp_parameters", _params_with("slp_version", "unlabeled_as")
    )
    kw = _writer_save_kwargs(tmp_path, {"slp_version": "v2", "unlabeled_as": "absent"})
    assert kw["slp_version"] == "v2" and kw["unlabeled_as"] == "absent"


def test_predict_to_file_passes_options_to_the_writer(tmp_path, monkeypatch):
    """``Predictor.predict_to_file`` hands its options to the writer it builds."""
    from sleap_nn.inference import writer as writer_mod
    from sleap_nn.inference.predictor import Predictor

    built = {}

    class _Writer:
        frame_count = 0

        def __init__(self, **kwargs):
            built.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr(writer_mod, "IncrementalLabelsWriter", _Writer)
    pred = MagicMock()
    pred.skeleton = sio.Skeleton(["a"])
    pred._is_embedding_layer.return_value = False
    pred._make_provider.return_value = (MagicMock(), None)
    pred.predict_streaming.return_value = iter(())
    Predictor.predict_to_file(
        pred, "video.mp4", str(tmp_path / "s.slp"), slp_version="v2"
    )
    assert built["save_options"]["slp_version"] == "v2"
    assert built["save_options"]["compression"] is None

    with pytest.raises(ValueError, match="compression"):
        Predictor.predict_to_file(pred, "video.mp4", "x.slp", compression="zip")


# ── PoseStream.save ──────────────────────────────────────────────────────────


def test_pose_stream_save(monkeypatch):
    from sleap_nn.inference.stream import PoseStream

    fake = MagicMock()
    PoseStream.save(fake, "a.slp")
    assert fake.labels.return_value.save.call_args.kwargs == {"embed": False}

    monkeypatch.setattr(so, "_save_slp_parameters", _params_with("compression"))
    PoseStream.save(fake, "a.slp", compression="gzip")
    assert fake.labels.return_value.save.call_args.kwargs == {
        "embed": False,
        "compression": "gzip",
    }


# ── embedding writer ─────────────────────────────────────────────────────────


def test_embedding_writer_forwards_options(minimal_embedding_model_dir, tmp_path):
    from sleap_nn.inference.embedding import predict_embeddings_to_slp

    src = "tests/assets/datasets/minimal_instance.pkg.slp"
    calls = []
    real_save = sio.save_slp

    def _spy(labels, path, **kwargs):
        calls.append(kwargs)
        return real_save(labels, path, **kwargs)

    # autospec keeps save_slp's signature, which the option bridge inspects.
    with patch(
        "sleap_nn.inference.embedding.sio.save_slp", autospec=True, side_effect=_spy
    ):
        predict_embeddings_to_slp(
            [minimal_embedding_model_dir],
            src,
            output_path=str(tmp_path / "default.slp"),
            device="cpu",
            batch_size=2,
            save_embeddings="slp",
        )
        out = predict_embeddings_to_slp(
            [minimal_embedding_model_dir],
            src,
            output_path=str(tmp_path / "v3gz.slp"),
            device="cpu",
            batch_size=2,
            save_embeddings="slp",
            slp_version="v3",
            compression="gzip",
        )
    assert calls[0] == {
        "embed": False,
        "restore_original_videos": False,
        "save_embedding_vectors": True,
    }
    assert calls[1]["compression"] == "gzip"
    assert _format_id(out) >= 3.0


def test_embedding_writer_refuses_bad_option_first(tmp_path):
    from sleap_nn.inference.embedding import predict_embeddings_to_slp

    with pytest.raises(ValueError, match="unlabeled_as"):
        predict_embeddings_to_slp(["/nope"], "x.slp", unlabeled_as="drop")
