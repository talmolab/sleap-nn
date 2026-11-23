"""Export folders on the export-directory route and their runtime choice.

The folders are built by hand (metadata, a copied training config, empty graph
files); no ONNX runtime or GPU is needed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

ASSETS = Path(__file__).resolve().parents[1] / "assets"
CENTERED = ASSETS / "model_ckpts" / "minimal_instance_centered_instance"
SLP = ASSETS / "datasets" / "minimal_instance.pkg.slp"


def _folder(tmp_path, name, metadata, *, ckpt=False, graphs=("model.onnx",)):
    d = tmp_path / name
    d.mkdir()
    shutil.copy(CENTERED / "training_config.yaml", d / "training_config.yaml")
    if ckpt:
        shutil.copy(CENTERED / "best.ckpt", d / "best.ckpt")
    (d / "export_metadata.json").write_text(json.dumps(metadata))
    for g in graphs:
        (d / g).write_bytes(b"")
    return d


NETWORK_ONLY = {"model_type": "centered_instance", "network_only": True}


def test_export_dir_route_refuses_a_network_only_export(tmp_path):
    """A network-only export is a model path, not an export directory."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.run import predict

    alone = _folder(tmp_path, "net", NETWORK_ONLY)
    both = _folder(tmp_path, "both", NETWORK_ONLY, ckpt=True)
    for d in (alone, both):
        with pytest.raises(ValueError, match="model path"):
            Predictor.from_export_dir(d, device="cpu")
        with pytest.raises(ValueError, match="network-only export"):
            predict(str(SLP), export_dir=str(d), device="cpu")


def test_box_detector_exports_stay_on_the_export_dir_route(tmp_path):
    """Detector exports are network-only by type and keep their route."""
    from sleap_nn.inference.predictor import _refuse_network_only_export_dir

    for meta in ({"model_type": "bbox_centernet"}, {"model_type": "centroid"}):
        d = _folder(tmp_path, meta["model_type"], meta)
        _refuse_network_only_export_dir(d, d / "export_metadata.json")
