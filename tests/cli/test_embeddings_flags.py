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
