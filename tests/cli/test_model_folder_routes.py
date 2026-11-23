"""Which route ``sleap-nn predict -m`` takes for each kind of model folder.

The folders are built by hand: a copied training config, a copied ``best.ckpt``
where the case needs one, an ``export_metadata.json`` and an empty graph file.
``run.predict`` is replaced, so only the route the command line picks is checked.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from sleap_nn.cli import cli

ASSETS = Path(__file__).resolve().parents[1] / "assets"
CENTERED = ASSETS / "model_ckpts" / "minimal_instance_centered_instance"
SLP = ASSETS / "datasets" / "minimal_instance.pkg.slp"


def _folder(tmp_path, name, *, ckpt=False, metadata=None, graphs=("model.onnx",)):
    d = tmp_path / name
    d.mkdir()
    shutil.copy(CENTERED / "training_config.yaml", d / "training_config.yaml")
    if ckpt:
        shutil.copy(CENTERED / "best.ckpt", d / "best.ckpt")
    if metadata is not None:
        (d / "export_metadata.json").write_text(json.dumps(metadata))
    for g in graphs:
        (d / g).write_bytes(b"")
    return d


NETWORK_ONLY = {"model_type": "centered_instance", "network_only": True}
FUSED = {"model_type": "centered_instance"}


def _route(*args):
    """Run ``predict`` with ``args`` and return what reached ``run.predict``."""
    with patch("sleap_nn.inference.run.predict", return_value=MagicMock()) as mock:
        result = CliRunner().invoke(cli, ["predict", "--data_path", str(SLP), *args])
    assert result.exit_code == 0, result.output
    return mock.call_args.kwargs


@pytest.fixture
def both_network_only(tmp_path):
    """``best.ckpt`` with a ``--network-only`` export written next to it."""
    return _folder(tmp_path, "both", ckpt=True, metadata=NETWORK_ONLY)


def test_lone_folder_with_checkpoint_and_network_only_export_runs_the_checkpoint(
    both_network_only,
):
    """It takes the checkpoint route; the exported-model route cannot run the graph."""
    kwargs = _route("-m", str(both_network_only))
    assert kwargs["model_paths"] == [str(both_network_only)]
    assert kwargs.get("export_dir") is None


def test_boxes_and_part_model_agree_with_that_route(both_network_only, tmp_path):
    """``--boxes`` and ``--part_model`` reach the checkpoint route, not a refusal."""
    kwargs = _route("-m", str(both_network_only), "--boxes", "gt")
    assert kwargs["model_paths"] == [str(both_network_only)]
    assert kwargs["boxes"] == "gt"

    rule = tmp_path / "rule.json"
    rule.write_text("{}")
    kwargs = _route(
        "-m",
        str(both_network_only),
        "--part_model",
        str(CENTERED),
        "--part_crop_rule",
        str(rule),
    )
    assert kwargs["model_paths"] == [str(both_network_only)]
    assert kwargs["part_model_paths"] == [str(CENTERED)]


def test_routes_that_already_worked_are_unchanged(tmp_path):
    """Fused exports (with or without best.ckpt) and lone network-only exports."""
    lone_net = _folder(tmp_path, "net", metadata=NETWORK_ONLY)
    assert _route("-m", str(lone_net))["model_paths"] == [str(lone_net)]
    for d in (
        _folder(tmp_path, "fused", metadata=FUSED),
        _folder(tmp_path, "both_fused", ckpt=True, metadata=FUSED),
    ):
        kwargs = _route("-m", str(d))
        assert kwargs["export_dir"] == str(d) and kwargs["model_paths"] is None
    assert _route("-m", str(CENTERED))["model_paths"] == [str(CENTERED)]


def test_a_graph_file_without_export_metadata_is_not_an_export(tmp_path):
    """A checkpoint directory with a stray model.onnx loads its checkpoint."""
    stray = _folder(tmp_path, "stray", ckpt=True)
    kwargs = _route("-m", str(stray))
    assert kwargs["model_paths"] == [str(stray)]
    assert kwargs.get("export_dir") is None
