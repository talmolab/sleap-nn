"""`--embeddings_path` must reject flags it cannot honor.

The route forwards six options and silently dropped every other flag the shared
inference option set accepts — including frame scoping, tracking, and crop
geometry overrides. Crop geometry in particular must come from the saved training
config: an override would embed crops the model was never fitted on.
"""

import pytest
from click.testing import CliRunner
from omegaconf import OmegaConf

from sleap_nn.cli import cli

CENTROID_CKPT = "tests/assets/model_ckpts/minimal_instance_centroid"
SLP = "tests/assets/datasets/minimal_instance.pkg.slp"


@pytest.fixture
def embedding_model_dir(tmp_path):
    """A model dir the `embedding` route accepts.

    Only `training_config.yaml` is needed: `--embeddings_path` is gated on the
    saved model type, and these tests assert on flag validation, which happens
    before any checkpoint is loaded.
    """
    dest = tmp_path / "embedding_model"
    dest.mkdir()
    cfg = OmegaConf.load(f"{CENTROID_CKPT}/training_config.yaml")
    heads = {key: None for key in cfg.model_config.head_configs}
    heads["embedding"] = {"embedding": {"embedding_dim": 16, "output_stride": 16}}
    cfg.model_config.head_configs = OmegaConf.create(heads)
    OmegaConf.save(cfg, dest / "training_config.yaml")
    return dest.as_posix()


def _invoke(embedding_model_dir, out_path, *extra):
    return CliRunner().invoke(
        cli,
        [
            "predict",
            "-m",
            embedding_model_dir,
            "-i",
            SLP,
            "--embeddings_path",
            out_path,
            *extra,
        ],
    )


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--max_height", "512"),
        ("--max_width", "512"),
        ("--crop_size", "64"),
        ("--input_scale", "0.5"),
        ("--video_index", "0"),
        ("--frames", "0-3"),
        ("--only_labeled_frames", None),
    ],
)
def test_unsupported_flags_are_rejected(tmp_path, embedding_model_dir, flag, value):
    """Each ignored flag fails loudly, naming itself."""
    extra = [flag] if value is None else [flag, value]
    result = _invoke(embedding_model_dir, (tmp_path / "emb.h5").as_posix(), *extra)

    assert result.exit_code != 0
    assert "does not support" in result.output
    assert flag.lstrip("-").replace("_", "") in result.output.replace("_", "").replace(
        "-", ""
    )


def test_tracking_flag_is_rejected(tmp_path, embedding_model_dir):
    """`-t` would silently do nothing on this route."""
    result = _invoke(embedding_model_dir, (tmp_path / "emb.h5").as_posix(), "-t")

    assert result.exit_code != 0
    assert "does not support" in result.output
    assert "tracking" in result.output


def test_forwarded_flags_get_past_validation(tmp_path, embedding_model_dir):
    """The six forwarded options must still be accepted.

    The run fails later (this fixture has no weights), but it must not fail with
    the unsupported-flag error.
    """
    result = _invoke(
        embedding_model_dir,
        (tmp_path / "emb.h5").as_posix(),
        "--device",
        "cpu",
        "--batch_size",
        "2",
        "--peak_threshold",
        "0.2",
    )

    assert "does not support" not in result.output


def test_data_path_is_still_required(tmp_path, embedding_model_dir):
    """A missing input is reported before the flag check.

    `--data_path` is a required click option, so click rejects it first; the
    `_run_embeddings` guard with the same intent is for programmatic callers.
    """
    result = CliRunner().invoke(
        cli,
        [
            "predict",
            "-m",
            embedding_model_dir,
            "--embeddings_path",
            (tmp_path / "emb.h5").as_posix(),
            "--crop_size",
            "64",
        ],
    )

    assert result.exit_code != 0
    assert "data_path" in result.output
    assert "does not support" not in result.output
