"""Parity of ``BBoxDetrHead`` with HF transformers' ``RfDetrForObjectDetection``.

The fixture (``tests/assets/detr/hf_rf_detr_model_parity.pt``, written by
``make_hf_model_parity_fixture.py`` with transformers 5.20.2) holds a tiny RF-DETR
with random weights, fixed backbone features and HF's outputs. HF's state dict
loads into ``BBoxDetrHead`` by stripping the ``model.backbone.`` / ``model.``
prefixes; then every output must match to 1e-4.
"""

from pathlib import Path

import pytest
import torch

from sleap_nn.architectures.detr import BBoxDetrHead

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "detr"
    / "hf_rf_detr_model_parity.pt"
)
ATOL = 1e-4


def hf_to_sleap_nn_key(key: str) -> str:
    """Rename an HF ``RfDetrForObjectDetection`` key (minus its backbone)."""
    for prefix in ("model.backbone.", "model."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _head_from_fixture(fixture) -> BBoxDetrHead:
    dims = fixture["dims"]
    head = BBoxDetrHead(
        in_channels=fixture["backbone_width"] * fixture["num_taps"],
        num_classes=fixture["num_labels"] - 1,
        hidden_dim=dims["d_model"],
        num_queries=dims["num_queries"],
        group_detr=dims["group_detr"],
        dec_layers=dims["decoder_layers"],
        sa_nheads=dims["decoder_self_attention_heads"],
        ca_nheads=dims["decoder_cross_attention_heads"],
        dec_n_points=dims["decoder_n_points"],
        dim_feedforward=dims["decoder_ffn_dim"],
        dropout=dims["dropout"],
        c2f_num_blocks=dims["c2f_num_blocks"],
        c2f_expansion=dims["hidden_expansion"],
    )
    state_dict = {hf_to_sleap_nn_key(k): v for k, v in fixture["state_dict"].items()}
    head.load_state_dict(state_dict, strict=True)
    return head


@pytest.fixture(scope="module")
def fixture():
    return torch.load(FIXTURE, weights_only=False)


def test_state_dict_keys_match_hf_by_prefix_rename(fixture):
    """Every HF decoder/projector/head tensor has a same-shaped native tensor."""
    head = _head_from_fixture(fixture)
    ours = {k: tuple(v.shape) for k, v in head.state_dict().items()}
    theirs = {
        hf_to_sleap_nn_key(k): tuple(v.shape) for k, v in fixture["state_dict"].items()
    }
    assert ours == theirs


@pytest.mark.parametrize("index", [0, 1, 2], ids=["eval", "train", "invalid-edges"])
def test_outputs_match_hf(fixture, index):
    case = fixture["cases"][index]
    head = _head_from_fixture(fixture)
    head.train(case["case"]["training"])
    with torch.no_grad():
        out = head(case["features"])

    groups = fixture["dims"]["group_detr"] if case["case"]["training"] else 1
    assert out["pred_logits"].shape[1] == groups * fixture["dims"]["num_queries"]

    def close(actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=ATOL)

    close(out["pred_logits"], case["pred_logits"])
    close(out["pred_boxes"], case["pred_boxes"])
    close(out["enc_outputs"]["pred_logits"], case["enc_logits"])
    close(out["enc_outputs"]["pred_boxes"], case["enc_boxes"])
    num_layers = case["layer_logits"].shape[0]
    assert len(out["aux_outputs"]) == num_layers - 1
    for i, aux in enumerate(out["aux_outputs"]):
        close(aux["pred_logits"], case["layer_logits"][i])
        close(aux["pred_boxes"], case["layer_boxes"][i])
