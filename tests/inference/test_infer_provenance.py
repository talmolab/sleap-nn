"""Regression: the new Predictor actually WRITES provenance to saved Labels.

PR #530 gap: ``build_inference_provenance`` existed but was never called by
``run.predict`` / ``Predictor.predict`` / the CLI, so saved ``.slp`` files
carried no inference lineage. ``test_provenance.py`` only tested the builder
helpers in isolation.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from sleap_nn.inference.predictor import Predictor

ASSETS = Path(__file__).resolve().parents[1] / "assets"
SLP = ASSETS / "datasets" / "minimal_instance.pkg.slp"
SINGLE = ASSETS / "model_ckpts" / "minimal_instance_single_instance"
CENTROID = ASSETS / "model_ckpts" / "minimal_instance_centroid"
CENTERED_INSTANCE = ASSETS / "model_ckpts" / "minimal_instance_centered_instance"


@pytest.mark.skipif(not (SLP.exists() and SINGLE.exists()), reason="missing fixtures")
def test_predict_attaches_provenance():
    pred = Predictor.from_model_paths([str(SINGLE)], device="cpu", batch_size=4)
    labels = pred.predict(str(SLP), make_labels=True)
    prov = labels.provenance
    assert prov, "predicted Labels carry no provenance"
    assert "sleap_nn_version" in prov
    assert prov.get("model_paths"), "model_paths not recorded in provenance"
    assert prov.get("model_type"), "model_type not recorded"
    assert prov.get("device") == "cpu"
    assert "inference_config" in prov
    # Timestamps + runtime present.
    assert "inference_start_timestamp" in prov
    assert "inference_runtime_seconds" in prov


@pytest.mark.skipif(not (SLP.exists() and SINGLE.exists()), reason="missing fixtures")
def test_predict_provenance_records_scale_for_single_stage_model():
    """`inference_config` must record the scale actually used, not just exist.

    Complements `test_predict_attaches_provenance` (which only checks
    `inference_config` is present) by checking its content.
    """
    pred = Predictor.from_model_paths([str(SINGLE)], device="cpu", batch_size=4)
    labels = pred.predict(str(SLP), make_labels=True)
    assert labels.provenance["inference_config"].get("scale") == pytest.approx(0.5)


def _copy_ckpt_with_scale(src: Path, dst: Path, scale: float) -> Path:
    """Copy a checkpoint dir to *dst*, overriding ``data_config.preprocessing.scale``."""
    shutil.copytree(src, dst)
    cfg = OmegaConf.load(str(dst / "training_config.yaml"))
    cfg.data_config.preprocessing.scale = scale
    OmegaConf.save(cfg, str(dst / "training_config.yaml"))
    return dst


@pytest.mark.skipif(
    not (SLP.exists() and CENTROID.exists() and CENTERED_INSTANCE.exists()),
    reason="missing fixtures",
)
def test_predict_provenance_records_distinct_per_stage_scale_for_topdown(tmp_path):
    """Provenance gap fix: topdown's two stages must be recorded distinctly.

    `predict`'s provenance previously never recorded `scale`/`crop_size` at
    all (unlike the legacy `track` pipeline, which did). Now that it does,
    verify it records the TWO stages' scales distinctly rather than one
    shared value -- the same distinction the actual inference math needed
    fixing for (topdown centroid/confmap scale-sharing bug, #725). Uses a
    deliberately mismatched pair so a regression that collapsed both stages
    to one value would be caught here too.
    """
    centroid_dir = _copy_ckpt_with_scale(CENTROID, tmp_path / "centroid", scale=0.5)
    centered_dir = _copy_ckpt_with_scale(
        CENTERED_INSTANCE, tmp_path / "centered_instance", scale=1.0
    )
    pred = Predictor.from_model_paths(
        [str(centroid_dir), str(centered_dir)], device="cpu", batch_size=4
    )
    labels = pred.predict(str(SLP), make_labels=True)
    cfg = labels.provenance["inference_config"]
    assert cfg.get("centroid_scale") == pytest.approx(0.5)
    assert cfg.get("instance_scale") == pytest.approx(1.0)
    assert "crop_size" in cfg
