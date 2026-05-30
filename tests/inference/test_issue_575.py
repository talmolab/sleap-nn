"""Regression tests for issue #575 — accept .ckpt / training_config paths.

``model_paths`` historically had to be a model *directory* (holding
``training_config.{yaml,json}`` + ``best.ckpt``). The inference / tracking entry
points (``Predictor.from_model_paths``, the new ``sleap-nn infer`` flow, and the
legacy ``sleap-nn track`` / ``run_inference`` pipeline) now also accept a path to
a model's ``best.ckpt`` or ``training_config.{yaml,json}`` file; every form
resolves to the model directory, which is loaded via ``best.ckpt``.

These tests cover the pure-path normalizer (:func:`resolve_model_dir`) plus
parity between the directory form and the file forms through the public
``Predictor.from_model_paths`` API and the legacy ``run_inference`` guard.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from loguru import logger
from omegaconf import OmegaConf

from sleap_nn.config.utils import resolve_model_dir
from sleap_nn.inference.predictor import Predictor

CKPT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "model_ckpts"
LEGACY_ROOT = Path(__file__).resolve().parents[1] / "assets" / "legacy_models"

SINGLE_DIR = CKPT_ROOT / "minimal_instance_single_instance"
CENTROID_DIR = CKPT_ROOT / "minimal_instance_centroid"
CENTERED_DIR = CKPT_ROOT / "minimal_instance_centered_instance"
LEGACY_SINGLE_DIR = LEGACY_ROOT / "minimal_robot.UNet.single_instance"

_PREPROCESS = {
    "ensure_rgb": None,
    "ensure_grayscale": None,
    "crop_size": None,
    "max_width": None,
    "max_height": None,
    "scale": None,
}


def _build(paths):
    """Build a Predictor from one or more model paths (dir or file form)."""
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    return Predictor.from_model_paths(
        [str(p) for p in paths],
        peak_threshold=0.3,
        preprocess_config=OmegaConf.create(_PREPROCESS),
    )


# ─────────────────────────────────────────────────────────────────────────
# resolve_model_dir — pure path normalization
# ─────────────────────────────────────────────────────────────────────────


def test_resolve_dir_unchanged():
    """A model directory resolves to itself (POSIX-normalized)."""
    assert resolve_model_dir(SINGLE_DIR) == SINGLE_DIR.as_posix()


def test_resolve_dir_trailing_slash():
    """A trailing slash is normalized away, not treated as a missing path."""
    assert resolve_model_dir(SINGLE_DIR.as_posix() + "/") == SINGLE_DIR.as_posix()


def test_resolve_best_ckpt():
    """A best.ckpt path resolves to its containing directory."""
    assert resolve_model_dir(SINGLE_DIR / "best.ckpt") == SINGLE_DIR.as_posix()


def test_resolve_training_config_yaml():
    """A training_config.yaml path resolves to its containing directory."""
    assert (
        resolve_model_dir(SINGLE_DIR / "training_config.yaml") == SINGLE_DIR.as_posix()
    )


def test_resolve_training_config_json():
    """A legacy training_config.json path resolves to its directory (kept a dir
    so the legacy ``load_legacy_model(model_dir=...)`` branch still gets a dir)."""
    assert (
        resolve_model_dir(LEGACY_SINGLE_DIR / "training_config.json")
        == LEGACY_SINGLE_DIR.as_posix()
    )


def test_resolve_other_config_file_resolves_to_parent():
    """Any config-like file inside a model dir resolves to that dir; the dir's
    contents are validated downstream, not by the resolver."""
    assert (
        resolve_model_dir(SINGLE_DIR / "initial_config.yaml") == SINGLE_DIR.as_posix()
    )


def test_resolve_nonbest_ckpt_warns(tmp_path):
    """Pointing at a non-best .ckpt still resolves to the dir but warns that
    best.ckpt will be loaded instead."""
    d = tmp_path / "model"
    d.mkdir()
    shutil.copy(SINGLE_DIR / "training_config.yaml", d / "training_config.yaml")
    shutil.copy(SINGLE_DIR / "best.ckpt", d / "last.ckpt")

    msgs: list[str] = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        out = resolve_model_dir(d / "last.ckpt")
    finally:
        logger.remove(sink)

    assert out == d.as_posix()
    assert any("best.ckpt" in m and "last.ckpt" in m for m in msgs)


def test_resolve_best_ckpt_does_not_warn(tmp_path):
    """A plain best.ckpt path resolves silently (no spurious warning)."""
    msgs: list[str] = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        resolve_model_dir(SINGLE_DIR / "best.ckpt")
    finally:
        logger.remove(sink)
    assert not any("best.ckpt" in m for m in msgs)


def test_resolve_nonexistent_raises():
    """A path that does not exist raises FileNotFoundError with guidance."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_model_dir(CKPT_ROOT / "no_such_model_dir")


def test_resolve_unrecognized_file_raises():
    """A real file that is neither a config nor a .ckpt raises FileNotFoundError."""
    slp = SINGLE_DIR / "labels_val_gt_0.slp"
    assert slp.exists()
    with pytest.raises(FileNotFoundError, match="not a recognized model file"):
        resolve_model_dir(slp)


# ─────────────────────────────────────────────────────────────────────────
# Predictor.from_model_paths — parity across path forms
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("form", ["best.ckpt", "training_config.yaml"])
def test_single_instance_file_form_parity(form):
    """Building from best.ckpt / training_config.yaml matches the dir form."""
    ref = _build(SINGLE_DIR)
    alt = _build(SINGLE_DIR / form)
    assert type(alt.layer).__name__ == type(ref.layer).__name__
    assert alt.skeleton.node_names == ref.skeleton.node_names


def test_topdown_mixed_forms_parity():
    """A top-down pair with *mixed* forms (one .ckpt, one config) builds the same
    predictor as passing both directories."""
    ref = _build([CENTROID_DIR, CENTERED_DIR])
    alt = _build([CENTROID_DIR / "best.ckpt", CENTERED_DIR / "training_config.yaml"])
    assert type(alt.layer).__name__ == type(ref.layer).__name__
    assert alt.skeleton.node_names == ref.skeleton.node_names


def test_single_instance_prediction_parity(small_robot_minimal_video):
    """End-to-end: predictions from the best.ckpt form are identical to the dir
    form (same checkpoint, same resolved directory)."""
    frames = list(range(4))

    def run(model_path):
        predictor = _build(model_path)
        return predictor.predict(
            small_robot_minimal_video.as_posix(), frames=frames, make_labels=True
        )

    ref = run(SINGLE_DIR)
    alt = run(SINGLE_DIR / "best.ckpt")

    assert len(alt) == len(ref) == len(frames)
    for lf_ref, lf_alt in zip(ref, alt):
        assert len(lf_alt.instances) == len(lf_ref.instances)
        for inst_ref, inst_alt in zip(lf_ref.instances, lf_alt.instances):
            np.testing.assert_array_equal(inst_alt.numpy(), inst_ref.numpy())


# ─────────────────────────────────────────────────────────────────────────
# Legacy run_inference centroid guard — must see through a .ckpt path
# ─────────────────────────────────────────────────────────────────────────


def test_run_inference_centroid_guard_accepts_ckpt_path():
    """The lone-centroid guard in ``run_inference`` resolves a centroid best.ckpt
    path to its dir and still detects the centroid-only case (raising the
    redirect error rather than silently failing open)."""
    from sleap_nn.predict import run_inference

    with pytest.raises(ValueError, match="Centroid-only inference is not supported"):
        run_inference(
            data_path="nonexistent_video.mp4",
            model_paths=[str(CENTROID_DIR / "best.ckpt")],
            tracking=False,
            make_labels=True,
        )
