"""Regression: the new Predictor actually WRITES provenance to saved Labels.

PR #530 gap: ``build_inference_provenance`` existed but was never called by
``run.predict`` / ``Predictor.predict`` / the CLI, so saved ``.slp`` files
carried no inference lineage. ``test_provenance.py`` only tested the builder
helpers in isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sleap_nn.inference.predictor import Predictor

ASSETS = Path(__file__).resolve().parents[1] / "assets"
SLP = ASSETS / "datasets" / "minimal_instance.pkg.slp"
SINGLE = ASSETS / "model_ckpts" / "minimal_instance_single_instance"


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
