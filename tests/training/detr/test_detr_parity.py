"""Parity of the DETR matcher and criterion with HF transformers.

HF's ``RfDetrForObjectDetection`` trains with the LW-DETR loss
(``LwDetrHungarianMatcher`` + ``LwDetrImageLoss``). The fixture test runs everywhere
against ``tests/assets/detr/hf_lw_detr_parity.pt`` (written by
``make_hf_parity_fixture.py`` with transformers 5.20.2); the live test runs HF itself
when the installed transformers has RF-DETR.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from sleap_nn.training.detr import SetCriterion

HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parents[1] / "assets" / "detr" / "hf_lw_detr_parity.pt"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "make_hf_parity_fixture", HERE / "make_hf_parity_fixture.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_case(case, expected):
    """Same indices per layer and loss terms within 1e-4 as HF."""
    groups = case["group_detr"]
    criterion = SetCriterion(case["num_classes"], group_detr=groups)
    criterion.train(case["training"])
    outputs, targets = case["outputs"], case["targets"]

    layers = {"": outputs, "_enc": outputs["enc_outputs"]}
    layers.update({f"_{i}": a for i, a in enumerate(outputs["aux_outputs"])})
    assert set(layers) == set(expected["indices"])
    active_groups = groups if case["training"] else 1
    for suffix, layer in layers.items():
        ours = criterion.matcher(layer, targets, group_detr=active_groups)
        theirs = expected["indices"][suffix]
        for (rows, cols), (hf_rows, hf_cols) in zip(ours, theirs, strict=True):
            torch.testing.assert_close(rows, hf_rows, rtol=0, atol=0)
            torch.testing.assert_close(cols, hf_cols, rtol=0, atol=0)

    losses = criterion(outputs, targets)
    assert set(losses) == set(expected["losses"])
    for key, value in expected["losses"].items():
        assert losses[key].item() == pytest.approx(value, rel=1e-4, abs=1e-4), key


def _case_id(case):
    mode = "train" if case["training"] else "eval"
    sizes = "-".join(str(len(t["labels"])) for t in case["targets"])
    return f"{mode}-g{case['group_detr']}-n{sizes}"


_FIXTURE_CASES = torch.load(FIXTURE, weights_only=True)["cases"]


@pytest.mark.parametrize("case", _FIXTURE_CASES, ids=_case_id)
def test_matches_hf_fixture(case):
    """Indices and losses equal those HF produced for the checked-in inputs."""
    _check_case(case, case["expected"])


@pytest.mark.parametrize("seed", [10, 11])
@pytest.mark.parametrize("training", [True, False])
def test_matches_hf_live(seed, training):
    """Indices and losses equal HF's on fresh random inputs."""
    pytest.importorskip("transformers.models.rf_detr")
    pytest.importorskip("transformers.loss.loss_lw_detr")
    generator = _load_generator()
    case = generator.make_case(
        seed=seed,
        sizes=[4, 0, 7, 1],
        group_size=10,
        groups=4,
        aux_layers=3,
        num_classes=5,
        training=training,
    )
    _check_case(case, generator.run_hf(case))
