"""The embedding (re-ID) route must reject flags it cannot honor.

`--save_embeddings` / `--tracking` on an embedding model embed every detection
already present in `--data_path`, so frame scoping has nowhere to go, and crop
geometry has to come from the trained config or the model embeds crops it was
never fitted on. Silently ignoring those flags is the class of surprise #732
fixed for `predict`.

The rejection is scoped to the LONE-embedding route: when a detection stack is
passed alongside the embedding model (the fused detect->embed path), the
detection stage receives the full option set and those same flags are honored.
"""

import pytest
from click.testing import CliRunner
from omegaconf import OmegaConf

from sleap_nn.cli import cli

CENTROID_CKPT = "tests/assets/model_ckpts/minimal_instance_centroid"
SLP = "tests/assets/datasets/minimal_instance.pkg.slp"


@pytest.fixture
def embedding_model_dir(tmp_path):
    """A model dir the embedding route accepts.

    Only `training_config.yaml` is needed: the route is selected on the saved
    model type, and these tests assert on flag validation, which happens before
    any checkpoint is loaded.
    """
    dest = tmp_path / "embedding_model"
    dest.mkdir()
    cfg = OmegaConf.load(f"{CENTROID_CKPT}/training_config.yaml")
    heads = {key: None for key in cfg.model_config.head_configs}
    heads["embedding"] = {"embedding": {"embedding_dim": 16, "output_stride": 16}}
    cfg.model_config.head_configs = OmegaConf.create(heads)
    OmegaConf.save(cfg, dest / "training_config.yaml")
    return dest.as_posix()


def _invoke(model_dirs, *extra):
    args = ["predict"]
    for d in model_dirs:
        args += ["-m", d]
    args += ["-i", SLP, "--save_embeddings", "slp", *extra]
    return CliRunner().invoke(cli, args)


@pytest.mark.parametrize(
    "flag,value,reason",
    [
        ("--frames", "0-3", "frame scoping"),
        ("--video_index", "0", "frame scoping"),
        ("--only_labeled_frames", None, "frame scoping"),
        ("--max_height", "512", "crop geometry"),
        ("--max_width", "512", "crop geometry"),
        ("--crop_size", "64", "crop geometry"),
        ("--input_scale", "0.5", "crop geometry"),
    ],
)
def test_unsupported_flags_are_rejected(embedding_model_dir, flag, value, reason):
    """Each unusable flag fails loudly, naming itself and why."""
    extra = [flag] if value is None else [flag, value]
    result = _invoke([embedding_model_dir], *extra)

    assert result.exit_code != 0
    assert "does not support" in result.output
    assert flag.lstrip("-").replace("_", "") in result.output.replace("_", "").replace(
        "-", ""
    )


def test_the_rejection_explains_the_fused_alternative(embedding_model_dir):
    """The error points at the path that DOES honor these flags."""
    result = _invoke([embedding_model_dir], "--frames", "0-3")

    assert "detection model alongside" in result.output


def test_fused_mode_accepts_the_same_flags(embedding_model_dir):
    """With a detection stack, frame scoping applies to the detection stage.

    The run fails later (this fixture carries no weights), but it must not fail
    with the unsupported-flag error.
    """
    result = _invoke([embedding_model_dir, CENTROID_CKPT], "--frames", "0-3")

    assert "does not support" not in result.output


def test_supported_flags_get_past_validation(embedding_model_dir, tmp_path):
    """Device, batch size and the output path are honored, not rejected."""
    result = _invoke(
        [embedding_model_dir],
        "--device",
        "cpu",
        "--batch_size",
        "2",
        "-o",
        (tmp_path / "out.slp").as_posix(),
    )

    assert "does not support" not in result.output


def test_tracking_flags_are_not_rejected(embedding_model_dir):
    """`--tracking` is the point of the route; its knobs must pass through."""
    result = _invoke(
        [embedding_model_dir],
        "-t",
        "--tracking_window_size",
        "5",
    )

    assert "does not support" not in result.output


# ── The honored set is COMPUTED, not hand-listed (review finding [12]) ─────────


@pytest.mark.parametrize(
    "flag,value",
    [
        # Same "frame scoping" rationale as the four originally listed.
        ("--only_predicted_frames", None),
        ("--exclude_user_labeled", None),
        ("--video_dataset", "images"),
        # Channels/centering come from the checkpoint, like crop geometry.
        ("--ensure_rgb", None),
        ("--ensure_grayscale", None),
        ("--anchor_part", "head"),
        # The route detects nothing and writes a bare .slp.
        ("--peak_threshold", "0.1"),
        ("--output_format", "analysis_h5"),
        ("--embed", "true"),
    ],
)
def test_more_unhonored_flags_are_rejected(embedding_model_dir, flag, value):
    """The old hand-written denylist named 8 of ~107 options, so these all passed
    validation and did nothing. The allowlist is derived instead."""
    extra = [flag] if value is None else [flag, value]
    result = _invoke([embedding_model_dir], *extra)

    assert result.exit_code != 0, result.output
    assert "does not support" in result.output


def test_tracking_flags_without_tracking_are_rejected(embedding_model_dir):
    """A `--tracking_*` knob is only consumed inside `if tracking:`, so passing it
    without `-t` did nothing at all. (The deleted `test_tracking_flag_is_rejected`
    covered this class.)"""
    result = _invoke([embedding_model_dir], "--tracking_window_size", "9")

    assert result.exit_code != 0, result.output
    assert "does not support" in result.output
    assert "require --tracking" in result.output


def test_derived_tracker_options_are_all_real_predict_options():
    """The probe reads `_build_tracker_config`; a typo there would silently widen
    the allowlist. Pin every derived and hand-listed key to a real option."""
    from sleap_nn.cli import (
        _EMBEDDING_ROUTE_HONORED_OPTIONS,
        _tracker_option_keys,
        predict,
    )

    real = {p.name for p in predict.params}
    derived = _tracker_option_keys()
    assert "appearance_weight" in derived
    assert "tracking_window_size" in derived
    # Picked up with no edit here when `--euclidean_scale` was added, which is the
    # whole point of deriving rather than listing.
    assert "euclidean_scale" in derived
    assert not (derived | _EMBEDDING_ROUTE_HONORED_OPTIONS) - real


# ── The appearance blend must not be handed appearance-only defaults ───────────


def test_appearance_weight_does_not_inject_embedding_defaults(
    embedding_model_dir, monkeypatch
):
    """The guide's own fused command must reach the tracker as a BLEND.

    `features`/`scoring_method` were injected whenever unset, without looking at
    `appearance_weight` -- so `-t --appearance_weight 0.3` became
    `features='embeddings'`, which the blend rejects, *after* the detection stack
    and embedding pass had run. And `--features keypoints --appearance_weight 0.3`
    still got `cosine_sim` injected, scoring the geometric 70% by the cosine of
    ravel'd pixel coordinates.
    """
    import sleap_nn.cli as cli_mod

    seen = {}
    monkeypatch.setattr(cli_mod, "_is_embedding_model", lambda m: True)
    monkeypatch.setattr(
        cli_mod,
        "_run_embeddings",
        lambda kwargs, save_embeddings, tracker_config, paf_workers: seen.update(
            features=tracker_config.features,
            scoring_method=tracker_config.scoring_method,
            weight=tracker_config.appearance_weight,
        ),
    )
    cli_mod._run_inference_impl(
        model_paths=[embedding_model_dir],
        data_path=SLP,
        frames=None,
        tracking=True,
        appearance_weight=0.3,
    )
    assert seen == {"features": "keypoints", "scoring_method": "oks", "weight": 0.3}


def test_no_appearance_weight_still_injects_embedding_defaults(
    embedding_model_dir, monkeypatch
):
    """Appearance-ONLY tracking keeps its defaults; only the blend opts out."""
    import sleap_nn.cli as cli_mod

    seen = {}
    monkeypatch.setattr(cli_mod, "_is_embedding_model", lambda m: True)
    monkeypatch.setattr(
        cli_mod,
        "_run_embeddings",
        lambda kwargs, save_embeddings, tracker_config, paf_workers: seen.update(
            features=tracker_config.features,
            scoring_method=tracker_config.scoring_method,
        ),
    )
    cli_mod._run_inference_impl(
        model_paths=[embedding_model_dir], data_path=SLP, frames=None, tracking=True
    )
    assert seen == {"features": "embeddings", "scoring_method": "cosine_sim"}


def test_incoherent_blend_is_rejected_before_inference(embedding_model_dir):
    """`--features embeddings --appearance_weight 0.3` must fail at the CLI edge.

    `apply_tracking` is the LAST step of the route, so validating only there meant
    the detection stack and the embedding pass both ran first.
    """
    result = _invoke(
        [embedding_model_dir],
        "-t",
        "--features",
        "embeddings",
        "--appearance_weight",
        "0.3",
    )

    assert result.exit_code != 0
    assert "already appearance-only" in result.output


def test_fused_centroid_only_blend_needs_a_scale_before_inference(
    embedding_model_dir, monkeypatch
):
    """A centroid-only detection stack resolves to `euclidean_dist`, so the blend
    needs `--euclidean_scale` -- and it must be said BEFORE the detection stack
    runs, not after minutes of inference."""
    import sleap_nn.cli as cli_mod

    monkeypatch.setattr(
        cli_mod, "_run_in_memory_new_flow", lambda *a, **k: pytest.fail("ran inference")
    )
    result = _invoke(
        [embedding_model_dir, CENTROID_CKPT], "-t", "--appearance_weight", "0.3"
    )

    assert result.exit_code != 0
    assert "requires euclidean_scale" in result.output


def test_fused_centroid_only_blend_accepted_with_a_scale(
    embedding_model_dir, monkeypatch
):
    """...and with the scale it gets past validation into inference."""
    import sleap_nn.cli as cli_mod

    ran = {}

    def _fake_detect(det_kwargs, paf_workers=0, save_output=True):
        ran["yes"] = True
        raise RuntimeError("stop after validation")

    monkeypatch.setattr(cli_mod, "_run_in_memory_new_flow", _fake_detect)
    result = _invoke(
        [embedding_model_dir, CENTROID_CKPT],
        "-t",
        "--appearance_weight",
        "0.3",
        "--euclidean_scale",
        "25",
    )

    assert ran, f"validation rejected the run: {result.output}"


# ── The fused route's default output path (review finding [11]) ───────────────


def test_default_output_path_handles_remote_urls():
    """`Path(src).with_suffix("")` collapsed `scheme://` into `scheme:/`.

    URLs are a documented `--data_path` input, and the fused branch dispatches
    before any URL handling -- so a remote run completed detection, embedding and
    tracking and only THEN failed on `sio.save_slp` with "Unable to synchronously
    create file" (h5py will not create parent dirs), with the temp detections
    already deleted.
    """
    from sleap_nn.cli import _default_embedding_output_path

    got = _default_embedding_output_path(
        "https://host/videos/v.mp4", is_url=True, tracking=True
    )
    assert got == "v.mp4.tracked.slp"
    assert "https:/" not in got


def test_default_output_path_matches_the_lone_route_convention():
    """One naming rule for both routes.

    The three routes had three conventions for `/data/run/dets.slp`:
    `dets.tracked.slp` (fused), `dets.slp.tracked.slp` (lone -- what the guide
    documents) and `dets.slp.slp` (predict). The fused one now follows the lone
    route's, which is what `predict_embeddings_to_slp` produces by default.
    """
    from sleap_nn.cli import _default_embedding_output_path

    assert (
        _default_embedding_output_path("/data/run/dets.slp", False, tracking=True)
        == "/data/run/dets.slp.tracked.slp"
    )
    assert (
        _default_embedding_output_path("/data/run/dets.slp", False, tracking=False)
        == "/data/run/dets.slp.embeddings.slp"
    )
    assert (
        _default_embedding_output_path("/data/clip.mp4", False, tracking=True)
        == "/data/clip.mp4.tracked.slp"
    )


def test_fused_detection_stage_is_not_told_to_write(embedding_model_dir, monkeypatch):
    """The detections are handed over IN MEMORY.

    Writing them to a temp .slp coupled the fused route to `--output_format`:
    `save_predictions` writes the .slp only `if "slp" in formats`, so
    `--output_format analysis_h5` ran the whole detection stage, wrote only an
    analysis h5 into the tmpdir, and then died in `sio.load_slp` with
    FileNotFoundError. Nothing rejected it (`_reject_unsupported_embedding_options`
    returns immediately in fused mode).
    """
    import sleap_nn.cli as cli_mod

    seen = {}

    def _fake_detect(det_kwargs, paf_workers=0, save_output=True):
        seen.update(det_kwargs)
        seen["_save_output"] = save_output
        raise RuntimeError("stop after the detection stage")

    monkeypatch.setattr(cli_mod, "_run_in_memory_new_flow", _fake_detect)
    result = _invoke(
        [embedding_model_dir, CENTROID_CKPT], "--output_format", "analysis_h5"
    )

    assert seen, "the detection stage never ran"
    assert seen["_save_output"] is False, "the detection stage still writes a file"
    assert seen["tracking"] is False
    # The user's --output_format is no longer forwarded to the throwaway stage.
    assert result.exit_code != 0  # our RuntimeError
