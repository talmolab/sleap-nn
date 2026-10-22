"""`sleap-nn predict` flags of the modern MOT trackers (`--tracker` and friends).

Drives the real CLI wiring with the inference / retrack entry points patched, and
asserts on the `TrackerConfig` they receive, or on the UsageError raised before
anything runs.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from omegaconf import OmegaConf

from sleap_nn.cli import cli

CENTROID_CKPT = "tests/assets/model_ckpts/minimal_instance_centroid"
SLP = "tests/assets/datasets/minimal_instance.pkg.slp"

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _flat(result) -> str:
    """Output with ANSI codes stripped and rich-click's panel wrapping collapsed."""
    return " ".join(_ANSI.sub("", result.output).replace("│", " ").split())


def _predict_video(*extra):
    """`predict` on a (fake) video with a (fake) model; returns (result, config)."""
    with patch(
        "sleap_nn.inference.run.predict", return_value=MagicMock()
    ) as mock_predict:
        result = CliRunner().invoke(
            cli,
            [
                "predict",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                *extra,
            ],
        )
    cfg = (
        mock_predict.call_args[1].get("tracker_config") if mock_predict.called else None
    )
    return result, cfg


def _retrack(*extra, tmp_path):
    """Tracking-only retrack of a (fake) `.slp`; returns (result, config)."""
    fake = MagicMock()
    fake.videos, fake.skeletons, fake.labeled_frames = [], [], []
    with (
        patch("sleap_io.load_slp", return_value=fake),
        patch(
            "sleap_nn.inference.predictor.Predictor.retrack", return_value=MagicMock()
        ) as mock_retrack,
    ):
        result = CliRunner().invoke(
            cli,
            [
                "predict",
                "--data_path",
                "/fake/path.slp",
                "--tracking",
                "--output_path",
                str(tmp_path / "out.slp"),
                *extra,
            ],
        )
    cfg = mock_retrack.call_args[0][1] if mock_retrack.called else None
    return result, cfg


def test_every_mot_flag_maps_to_tracker_config():
    result, cfg = _predict_video(
        "--tracking",
        "--tracker",
        "sort",
        "--track_buffer",
        "7",
        "--min_hits",
        "2",
        "--track_high_threshold",
        "0.6",
        "--track_low_threshold",
        "0.2",
        "--new_track_threshold",
        "0.7",
        "--min_match_similarity",
        "0.25",
        "--link_tracklets",
        "--mot_param",
        "preset=paper",
        "--mot_param",
        "gallery_size=50",
        "--scoring_method",
        "oks",
        "--oks_stddev",
        "0.05",
    )
    assert result.exit_code == 0, result.output
    assert cfg.tracker == "sort" and cfg.is_mot
    assert (cfg.track_buffer, cfg.min_hits) == (7, 2)
    assert (cfg.track_high_threshold, cfg.track_low_threshold) == (0.6, 0.2)
    assert (cfg.new_track_threshold, cfg.min_match_similarity) == (0.7, 0.25)
    assert cfg.link_tracklets is True
    assert cfg.mot_params == {"preset": "paper", "gallery_size": 50}
    assert cfg.scoring_method == "oks" and cfg.scoring_method_explicit
    assert cfg.oks_stddev == 0.05
    assert cfg.appearance_threshold is None


def test_mot_does_not_derive_classic_defaults():
    """No max_tracks / target from --max_instances, no forced local_queues."""
    result, cfg = _predict_video(
        "--tracking", "--tracker", "sort", "--max_instances", "3"
    )
    assert result.exit_code == 0, result.output
    assert cfg.max_tracks is None and cfg.tracking_target_instance_count is None
    assert cfg.candidates_method == "fixed_window"
    assert not cfg.candidates_method_explicit
    assert not cfg.features_explicit and not cfg.scoring_method_explicit

    result, cfg = _predict_video("--tracking", "--tracker", "sort", "--max_tracks", "2")
    assert result.exit_code == 0, result.output
    assert cfg.max_tracks == 2 and cfg.candidates_method == "fixed_window"


@pytest.mark.parametrize("value,expected", [("auto", "auto"), ("0.4", 0.4)])
def test_appearance_threshold_flag(tmp_path, value, expected):
    result, cfg = _retrack(
        "--tracker", "botsort", "--appearance_threshold", value, tmp_path=tmp_path
    )
    assert result.exit_code == 0, result.output
    assert cfg.tracker == "botsort" and cfg.appearance_threshold == expected


def test_appearance_threshold_out_of_range_is_a_usage_error(tmp_path):
    result, cfg = _retrack(
        "--tracker", "botsort", "--appearance_threshold", "2", tmp_path=tmp_path
    )
    assert result.exit_code == 2 and cfg is None
    assert "must be 'auto' or a cosine" in _flat(result)


@pytest.mark.parametrize(
    "param,match",
    [("bogus=1", "Unknown MOT parameter"), ("gallery_size", "KEY=VALUE")],
)
def test_bad_mot_param_is_a_usage_error(tmp_path, param, match):
    result, cfg = _retrack("--tracker", "sort", "--mot_param", param, tmp_path=tmp_path)
    assert result.exit_code == 2 and cfg is None
    assert match in _flat(result)


def test_out_of_range_mot_option_is_a_usage_error():
    result, cfg = _predict_video(
        "--tracking", "--tracker", "sort", "--track_high_threshold", "1.5"
    )
    assert result.exit_code == 2 and cfg is None
    assert "track_high_threshold must be in [0, 1]" in _flat(result)


@pytest.mark.parametrize("tracker", ["deepsort", "strongsort", "botsort", "deepocsort"])
def test_appearance_tracker_on_a_video_needs_an_embedding_model(tracker):
    result, cfg = _predict_video("--tracking", "--tracker", tracker)
    assert result.exit_code == 2 and cfg is None
    assert "no `embedding` model is among --model_paths" in _flat(result)


def test_appearance_tracker_retrack_of_a_slp_is_allowed(tmp_path):
    """A .slp may already carry vectors; apply_tracking checks them."""
    result, cfg = _retrack("--tracker", "deepsort", tmp_path=tmp_path)
    assert result.exit_code == 0, result.output
    assert cfg.tracker == "deepsort"


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--track_buffer", "5"),
        ("--min_hits", "2"),
        ("--track_high_threshold", "0.5"),
        ("--track_low_threshold", "0.1"),
        ("--new_track_threshold", "0.5"),
        ("--min_match_similarity", "0.3"),
        ("--appearance_threshold", "auto"),
        ("--mot_param", "gallery_size=5"),
    ],
)
def test_mot_flags_with_the_classic_tracker_are_rejected(flag, value):
    result, cfg = _predict_video("--tracking", flag, value)
    assert result.exit_code == 2 and cfg is None
    assert "only applies to the MOT trackers" in _flat(result)


def test_link_tracklets_works_with_the_classic_tracker():
    result, cfg = _predict_video("--tracking", "--link_tracklets")
    assert result.exit_code == 0, result.output
    assert cfg.tracker == "classic" and cfg.link_tracklets is True


def test_classic_link_tracklets_takes_the_linking_mot_params():
    """With --link_tracklets, the classic tracker's identity linking reads
    --mot_param time_base / body_size; any other key, or either one without
    --link_tracklets, is refused."""
    result, cfg = _predict_video(
        "--tracking",
        "--link_tracklets",
        "--mot_param",
        "body_size=40",
        "--mot_param",
        "time_base=steps",
    )
    assert result.exit_code == 0, result.output
    assert cfg.tracker == "classic"
    assert cfg.mot_params == {"body_size": 40.0, "time_base": "steps"}

    result, cfg = _predict_video("--tracking", "--mot_param", "body_size=40")
    assert result.exit_code == 2 and cfg is None
    assert "only applies to the MOT trackers" in _flat(result)
    assert "--mot_param time_base / --mot_param body_size" in _flat(result)
    result, cfg = _predict_video(
        "--tracking", "--link_tracklets", "--mot_param", "gallery_size=5"
    )
    assert result.exit_code == 2 and cfg is None
    result, cfg = _predict_video(
        "--tracking", "--link_tracklets", "--mot_param", "body_size=-3"
    )
    assert result.exit_code == 2 and cfg is None
    assert "body_size" in _flat(result)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--tracking_window_size", "9"),
        ("--candidates_method", "local_queues"),
        ("--use_kalman", None),
        ("--appearance_weight", "0.3"),
        ("--post_connect_single_breaks", None),
        ("--tracking_clean_instance_count", "2"),
        ("--tracking_pre_cull_to_target", "2"),
        ("--scoring_reduction", "max"),
    ],
)
def test_classic_only_flags_with_a_mot_tracker_are_rejected(flag, value):
    extra = [flag] if value is None else [flag, value]
    result, cfg = _predict_video("--tracking", "--tracker", "sort", *extra)
    assert result.exit_code == 2 and cfg is None
    flat = _flat(result)
    assert flag in flat and "of the classic tracker" in flat


def test_features_embeddings_with_a_mot_tracker_is_rejected():
    result, cfg = _predict_video(
        "--tracking", "--tracker", "sort", "--features", "embeddings"
    )
    assert result.exit_code == 2 and cfg is None
    assert "appearance tracker" in _flat(result)


@pytest.mark.parametrize("extra", [("--tracker", "sort"), ("--link_tracklets",)])
def test_tracker_flags_without_tracking_are_rejected(extra):
    result, _ = _predict_video(*extra)
    assert result.exit_code == 2
    assert "only applies with --tracking" in _flat(result)


def test_tracker_option_keys_include_the_mot_options():
    from sleap_nn.cli import _tracker_option_keys, predict

    derived = _tracker_option_keys()
    mot = {
        "tracker",
        "track_buffer",
        "min_hits",
        "track_high_threshold",
        "track_low_threshold",
        "new_track_threshold",
        "min_match_similarity",
        "appearance_threshold",
        "link_tracklets",
        "mot_param",
    }
    assert mot <= derived
    assert derived <= {p.name for p in predict.params}


def test_tracker_choices_match_tracker_names():
    from sleap_nn.cli import _TRACKER_CHOICES
    from sleap_nn.inference.tracking import TRACKER_NAMES

    assert _TRACKER_CHOICES == TRACKER_NAMES


# ── Embedding route ────────────────────────────────────────────────────────


@pytest.fixture
def embedding_model_dir(tmp_path):
    """A model dir the embedding route accepts (only its training config)."""
    dest = tmp_path / "embedding_model"
    dest.mkdir()
    cfg = OmegaConf.load(f"{CENTROID_CKPT}/training_config.yaml")
    heads = {key: None for key in cfg.model_config.head_configs}
    heads["embedding"] = {"embedding": {"embedding_dim": 16, "output_stride": 16}}
    cfg.model_config.head_configs = OmegaConf.create(heads)
    OmegaConf.save(cfg, dest / "training_config.yaml")
    return dest.as_posix()


def test_embedding_route_does_not_inject_classic_appearance_mode(
    embedding_model_dir, monkeypatch
):
    """With a MOT tracker the route must not inject `features="embeddings"`."""
    import sleap_nn.cli as cli_mod

    seen = {}
    monkeypatch.setattr(
        cli_mod,
        "_run_embeddings",
        lambda kwargs, save_embeddings, tracker_config, paf_workers: seen.update(
            cfg=tracker_config
        ),
    )
    result = CliRunner().invoke(
        cli,
        ["predict", "-m", embedding_model_dir, "-i", SLP, "-t", "--tracker", "botsort"],
    )
    assert result.exit_code == 0, result.output
    cfg = seen["cfg"]
    assert cfg.tracker == "botsort" and cfg.features == "keypoints"
    assert not cfg.features_explicit and not cfg.scoring_method_explicit


def test_embedding_route_mot_flags_are_honored(embedding_model_dir, monkeypatch):
    """The lone-embedding route's allowlist derives the MOT flags too."""
    import sleap_nn.cli as cli_mod

    seen = {}
    monkeypatch.setattr(
        cli_mod,
        "_run_embeddings",
        lambda kwargs, save_embeddings, tracker_config, paf_workers: seen.update(
            cfg=tracker_config
        ),
    )
    result = CliRunner().invoke(
        cli,
        [
            "predict",
            "-m",
            embedding_model_dir,
            "-i",
            SLP,
            "-t",
            "--tracker",
            "botsort",
            "--appearance_threshold",
            "0.3",
            "--track_buffer",
            "40",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["cfg"].appearance_threshold == 0.3 and seen["cfg"].track_buffer == 40


def test_embedding_route_geometry_only_mot_tracker_is_rejected(embedding_model_dir):
    result = CliRunner().invoke(
        cli,
        ["predict", "-m", embedding_model_dir, "-i", SLP, "-t", "--tracker", "sort"],
    )
    assert result.exit_code != 0
    flat = _flat(result)
    assert "--tracker sort tracks by geometry only" in flat
    assert "--tracker botsort" in flat and "--link_tracklets" in flat


def test_embedding_route_geometry_mot_tracker_with_linking_is_accepted(
    embedding_model_dir, monkeypatch
):
    """Identity linking reads the vectors, so the route is not pointless."""
    ran = {}
    monkeypatch.setattr(
        "sleap_nn.inference.embedding.predict_embeddings_to_slp",
        lambda **kw: ran.update(cfg=kw["tracker_config"]) or "out.slp",
    )
    result = CliRunner().invoke(
        cli,
        [
            "predict",
            "-m",
            embedding_model_dir,
            "-i",
            SLP,
            "-t",
            "--tracker",
            "sort",
            "--link_tracklets",
        ],
    )
    assert result.exit_code == 0, result.output
    assert ran["cfg"].tracker == "sort" and ran["cfg"].link_tracklets


# ── Review round 2 regressions ─────────────────────────────────────────────


@pytest.mark.parametrize("value", ["0", "-0.1", "nan", "inf"])
def test_eval_tracking_rejects_a_bad_oks_stddev(value):
    """#24: 0 / negative crash or silently miss everything; nan / inf too (a
    `FloatRange` alone lets them through). Refused before any file is read."""
    with patch("sleap_nn.evaluation.run_identity_evaluation") as run:
        result = CliRunner().invoke(
            cli,
            ["eval-tracking", "-g", "gt.slp", "-p", "pred.slp", "--oks_stddev", value],
        )
    assert result.exit_code == 2, result.output
    assert "--oks_stddev" in _flat(result) and not run.called


def test_eval_tracking_accepts_a_positive_oks_stddev():
    with patch("sleap_nn.evaluation.run_identity_evaluation") as run:
        result = CliRunner().invoke(
            cli,
            ["eval-tracking", "-g", "gt.slp", "-p", "pred.slp", "--oks_stddev", "0.05"],
        )
    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs["oks_stddev"] == 0.05


@pytest.mark.parametrize(
    "extra",
    [
        ("--tracker", "sort", "--features", "masks"),
        ("--features", "masks", "--scoring_method", "mask_iou"),  # classic
    ],
    ids=["sort", "classic"],
)
def test_features_masks_on_a_pose_stack_is_refused_before_inference(extra):
    """#32: a pose model's predictions never hold masks; refused up front instead of
    after the whole inference."""
    with patch("sleap_nn.inference.run.predict") as predict:
        result = CliRunner().invoke(
            cli,
            [
                "predict",
                "-i",
                "/fake/path.mp4",
                "-m",
                CENTROID_CKPT,
                "--tracking",
                *extra,
            ],
        )
    assert result.exit_code == 2 and not predict.called, result.output
    flat = _flat(result)
    assert "emits pose detections only" in flat and "no frame holds a mask" in flat


def test_pose_features_on_a_pose_stack_gets_to_inference():
    with patch("sleap_nn.inference.run.predict", return_value=MagicMock()) as predict:
        result = CliRunner().invoke(
            cli,
            [
                "predict",
                "-i",
                "/fake/path.mp4",
                "-m",
                CENTROID_CKPT,
                "--tracking",
                "--tracker",
                "sort",
                "--features",
                "keypoints",
            ],
        )
    assert result.exit_code == 0, result.output
    assert predict.called


def test_oks_stddev_without_oks_scoring_is_rejected():
    """#35: the MOT `auto` similarity never reads --oks_stddev."""
    result, cfg = _predict_video(
        "--tracking", "--tracker", "sort", "--oks_stddev", "0.05"
    )
    assert result.exit_code == 2 and cfg is None
    assert "only applies to OKS scoring" in _flat(result)


def test_new_mot_params_are_mot_param_keys():
    """`normalize_scores` / `body_size` are exposed through --mot_param."""
    result, cfg = _predict_video(
        "--tracking",
        "--tracker",
        "sort",
        "--mot_param",
        "normalize_scores=false",
        "--mot_param",
        "body_size=40",
    )
    assert result.exit_code == 0, result.output
    assert cfg.mot_params == {"normalize_scores": False, "body_size": 40.0}


def test_embedding_route_mot_param_linking_is_accepted(
    embedding_model_dir, monkeypatch
):
    """#29: linking requested through --mot_param reads the vectors too."""
    ran = {}
    monkeypatch.setattr(
        "sleap_nn.inference.embedding.predict_embeddings_to_slp",
        lambda **kw: ran.update(cfg=kw["tracker_config"]) or "out.slp",
    )
    result = CliRunner().invoke(
        cli,
        [
            "predict",
            "-m",
            embedding_model_dir,
            "-i",
            SLP,
            "-t",
            "--tracker",
            "sort",
            "--mot_param",
            "max_tracks=2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert ran["cfg"].runs_linking


def _tracked_slp(path):
    """A 2-animal `.slp` tracked by bytetrack (its provenance holds the record)."""
    import numpy as np
    import sleap_io as sio

    from sleap_nn.inference.tracking import TrackerConfig, apply_tracking

    skeleton = sio.Skeleton(nodes=["head", "tail"])
    video = sio.Video(filename=str(path.parent / "a.mp4"))

    def pose(x):
        return sio.PredictedInstance.from_numpy(
            points_data=np.array([[x, 50.0], [x + 30.0, 60.0]], dtype=np.float32),
            skeleton=skeleton,
            score=0.9,
            point_scores=np.full(2, 0.9, dtype=np.float32),
        )

    labels = sio.Labels(
        videos=[video],
        skeletons=[skeleton],
        labeled_frames=[
            sio.LabeledFrame(
                video=video, frame_idx=i, instances=[pose(2.0 * i), pose(200 + 2.0 * i)]
            )
            for i in range(12)
        ],
    )
    apply_tracking(labels, TrackerConfig(tracker="bytetrack"))
    sio.save_slp(labels, str(path))
    return str(path)


@pytest.mark.parametrize(
    "extra,top", [(("--tracker", "sort"), "sort"), ((), None)], ids=["sort", "classic"]
)
def test_retrack_keeps_the_input_tracking_record_as_lineage(tmp_path, extra, top):
    """#31: the input's own `tracking` record stays in `input_provenance` (it was
    lost when the input was tracked in place before its provenance was copied),
    and the output's top-level record is this run's (none for plain classic)."""
    import sleap_io as sio

    src = _tracked_slp(tmp_path / "in.slp")
    out = tmp_path / "out.slp"
    result = CliRunner().invoke(
        cli, ["predict", "-i", src, "--tracking", "-o", str(out), *extra]
    )
    assert result.exit_code == 0, result.output
    prov = sio.load_slp(str(out)).provenance
    assert (prov.get("tracking") or {}).get("tracker") == top
    assert prov["input_provenance"]["tracking"]["tracker"] == "bytetrack"
