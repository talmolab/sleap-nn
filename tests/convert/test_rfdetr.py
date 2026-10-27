"""Tests of the RF-DETR checkpoint importer (``sleap_nn.convert.rfdetr``).

The fixture ``tests/assets/detr/rfdetr_tiny/`` (written by
``make_rfdetr_fixture.py`` with transformers 5.20.2) is a tiny
``RfDetrForObjectDetection`` with random weights, saved with ``save_pretrained``
(so with the original ``rfdetr`` parameter names, as on the Hub), plus HF's
outputs. Importing it and running the sleap-nn model must reproduce every output
(final, per decoder layer, encoder proposals) to 1e-4.

``test_import_real_small_matches_hf`` downloads ``Roboflow/rf-detr-small``; it
runs only with ``SLEAP_NN_TEST_HF_WEIGHTS=1``.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio
import torch
import torchvision.transforms.v2.functional as tvf
from click.testing import CliRunner
from make_rfdetr_fixture import synthetic_image
from omegaconf import OmegaConf
from PIL import Image

from sleap_nn.architectures.model import Model
from sleap_nn.convert.rfdetr import (
    PAD_CLASS_LOGIT,
    convert_rfdetr_state_dict,
    import_rfdetr,
    rfdetr_categories,
    rfdetr_config_to_sleap_nn,
    rfdetr_key_to_sleap_nn,
)
from sleap_nn.inference.bbox import decode_set

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "detr"
TINY = ASSETS / "rfdetr_tiny"
ATOL = 1e-4


def _load_model(model_dir: Path) -> Model:
    cfg = OmegaConf.load(model_dir / "training_config.yaml")
    model = Model(
        backbone_type="dinov2_windowed",
        backbone_config=cfg.model_config.backbone_config.dinov2_windowed,
        head_configs=cfg.model_config.head_configs.bbox_detr,
        model_type="bbox_detr",
    )
    ckpt = torch.load(model_dir / "best.ckpt", weights_only=False)
    state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    return model


@pytest.fixture(scope="module")
def tiny_import(tmp_path_factory):
    out = tmp_path_factory.mktemp("rfdetr") / "tiny"
    import_rfdetr(TINY, out)
    return out


def _close(actual, expected, num_logits=None):
    if num_logits is not None:
        actual = actual[..., :num_logits]
    torch.testing.assert_close(actual, expected, rtol=0, atol=ATOL)


@pytest.mark.parametrize(
    "index", [0, 1, 2], ids=["eval", "eval-interpolated-pos", "train"]
)
def test_imported_model_matches_hf(tiny_import, index):
    case = torch.load(TINY / "hf_outputs.pt", weights_only=False)["cases"][index]
    model = _load_model(tiny_import).train(case["case"]["training"])
    with torch.no_grad():
        out = model(case["image"].float() / 255)
    n = case["pred_logits"].shape[-1]

    _close(out["pred_logits"], case["pred_logits"], n)
    _close(out["pred_boxes"], case["pred_boxes"])
    _close(out["enc_outputs"]["pred_logits"], case["enc_logits"], n)
    _close(out["enc_outputs"]["pred_boxes"], case["enc_boxes"])
    layers = [*out["aux_outputs"], out]
    assert len(layers) == case["layer_logits"].shape[0]
    for i, layer in enumerate(layers):
        _close(layer["pred_logits"], case["layer_logits"][i], n)
        _close(layer["pred_boxes"], case["layer_boxes"][i])
    # The appended logit is never a detection.
    assert (out["pred_logits"][..., n:] == PAD_CLASS_LOGIT).all()


def test_imported_config(tiny_import):
    cfg = OmegaConf.load(tiny_import / "training_config.yaml")
    hf = json.loads((TINY / "config.json").read_text())
    pre = cfg.data_config.preprocessing
    assert pre.resize_mode == "squash"
    assert (pre.max_height, pre.max_width) == (64, 64)
    assert pre.ensure_rgb
    bb = cfg.model_config.backbone_config.dinov2_windowed
    assert (bb.hidden_size, bb.num_layers, bb.pos_embed_grid) == (16, 4, 4)
    assert list(bb.out_indices) == [2, 4]
    assert list(bb.window_block_indexes) == [0, 1, 3]
    detr = cfg.model_config.head_configs.bbox_detr.detr
    assert len(detr.categories) == len(hf["id2label"]) == 5
    assert (detr.dec_layers, detr.num_queries, detr.group_detr) == (2, 12, 3)
    assert cfg.model_config.init_weights == "default"
    assert "Imported from RF-DETR checkpoint" in cfg.description


def test_import_refuses_to_overwrite(tiny_import):
    with pytest.raises(FileExistsError):
        import_rfdetr(TINY, tiny_import)
    import_rfdetr(TINY, tiny_import, overwrite=True)


@pytest.mark.parametrize(
    "key,expected",
    [
        # Original rfdetr names (Hub checkpoints).
        (
            "backbone.0.encoder.encoder.encoder.layer.3.attention.attention.query.weight",
            "backbone.encoder.layer.3.attention.q_proj.weight",
        ),
        (
            "backbone.0.encoder.encoder.encoder.layer.0.attention.output.dense.bias",
            "backbone.encoder.layer.0.attention.o_proj.bias",
        ),
        (
            "backbone.0.encoder.encoder.layernorm.weight",
            "backbone.layernorm.weight",
        ),
        (
            "backbone.0.projector.stages.0.0.m.2.cv1.bn.bias",
            "detr.projector.projector_layer.bottlenecks.2.conv1.norm.bias",
        ),
        (
            "backbone.0.projector.stages.0.1.weight",
            "detr.projector.layer_norm.weight",
        ),
        ("refpoint_embed.weight", "detr.reference_point_embed.weight"),
        (
            "transformer.decoder.layers.1.norm1.weight",
            "detr.decoder.layers.1.self_attn_layer_norm.weight",
        ),
        (
            "transformer.decoder.layers.1.linear2.bias",
            "detr.decoder.layers.1.mlp.fc2.bias",
        ),
        ("transformer.decoder.norm.bias", "detr.decoder.layernorm.bias"),
        (
            "transformer.enc_out_class_embed.4.weight",
            "detr.enc_out_class_embed.4.weight",
        ),
        ("class_embed.bias", "detr.class_embed.bias"),
        # HF names (`save_pretrained` of a model loaded in HF).
        (
            "model.backbone.backbone.encoder.layer.3.attention.q_proj.weight",
            "backbone.encoder.layer.3.attention.q_proj.weight",
        ),
        (
            "model.backbone.projector.layer_norm.weight",
            "detr.projector.layer_norm.weight",
        ),
        (
            "model.decoder.layers.0.self_attn.k_proj.bias",
            "detr.decoder.layers.0.self_attn.k_proj.bias",
        ),
        ("bbox_embed.layers.2.weight", "detr.bbox_embed.layers.2.weight"),
    ],
)
def test_key_renames(key, expected):
    assert rfdetr_key_to_sleap_nn(key) == expected


def test_in_proj_is_split_and_class_heads_padded():
    w = torch.arange(12.0).reshape(6, 2)
    b = torch.arange(6.0)
    out = convert_rfdetr_state_dict(
        {
            "transformer.decoder.layers.0.self_attn.in_proj_weight": w,
            "transformer.decoder.layers.0.self_attn.in_proj_bias": b,
            "class_embed.weight": torch.ones(3, 2),
            "class_embed.bias": torch.zeros(3),
            "transformer.enc_out_class_embed.0.bias": torch.zeros(3),
        }
    )
    prefix = "detr.decoder.layers.0.self_attn."
    assert torch.equal(out[prefix + "q_proj.weight"], w[:2])
    assert torch.equal(out[prefix + "k_proj.weight"], w[2:4])
    assert torch.equal(out[prefix + "v_proj.bias"], b[4:])
    assert out["detr.class_embed.weight"].shape == (4, 2)
    assert (out["detr.class_embed.weight"][3] == 0).all()
    assert out["detr.class_embed.bias"][3] == PAD_CLASS_LOGIT
    assert out["detr.enc_out_class_embed.0.bias"][3] == PAD_CLASS_LOGIT

    with pytest.raises(ValueError, match="detection checkpoints"):
        convert_rfdetr_state_dict({"segmentation_head.x.weight": w})


def test_categories_keep_coco_ids():
    names = rfdetr_categories(
        {"id2label": {"0": "N/A", "1": "person", "2": "bicycle", "3": "N/A"}}
    )
    assert names == ["N/A_0", "person", "bicycle", "N/A_3"]


def test_unsupported_configs_are_refused():
    hf = json.loads((TINY / "config.json").read_text())
    for change in (
        {"architectures": ["RfDetrForInstanceSegmentation"]},
        {"projector_scale_factors": [2.0, 0.5]},
    ):
        with pytest.raises(ValueError, match="Unsupported RF-DETR checkpoint"):
            rfdetr_config_to_sleap_nn({**hf, **change})


def test_cli_import(tmp_path):
    from sleap_nn.cli import cli

    out = tmp_path / "cli"
    result = CliRunner().invoke(cli, ["import-rfdetr", TINY.as_posix(), "-o", out])
    assert result.exit_code == 0, result.output
    assert (out / "best.ckpt").exists() and (out / "training_config.yaml").exists()
    result = CliRunner().invoke(cli, ["import-rfdetr", TINY.as_posix(), "-o", out])
    assert result.exit_code != 0 and "--overwrite" in result.output


def test_predict_squashes_and_maps_boxes_back(tiny_import, tmp_path):
    """`sleap-nn predict` squashes each frame and undoes the scale per axis."""
    from sleap_nn.inference.predictor import Predictor

    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (96, 160, 3), dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "frame.png")
    video = sio.Video.from_filename((tmp_path / "frame.png").as_posix())
    labels = sio.Labels(
        labeled_frames=[sio.LabeledFrame(video=video, frame_idx=0)], videos=[video]
    )
    labels_path = tmp_path / "frame.slp"
    labels.save(labels_path.as_posix())

    pred = Predictor.from_model_paths(
        [tiny_import.as_posix()], device="cpu", peak_threshold=0.0, max_instances=20
    )
    predicted = pred.predict(labels_path.as_posix())
    boxes = sorted(
        ([b.x1, b.y1, b.x2, b.y2], b.score) for b in predicted.labeled_frames[0].bboxes
    )
    # num_select (= the 12 queries) (query, class) pairs, all above threshold 0.
    assert len(boxes) == 12

    # The same by hand: squash 96x160 -> 64x64, run, decode, unscale per axis.
    model = _load_model(tiny_import).eval()
    x = tvf.resize(torch.from_numpy(image).permute(2, 0, 1)[None], [64, 64])
    with torch.no_grad():
        out = model(x.float() / 255)
    xyxy, scores, _ = decode_set(
        out["pred_logits"], out["pred_boxes"], (64, 64), 0.0, 12, 20
    )
    xyxy = xyxy[0] / torch.tensor([64 / 160, 64 / 96, 64 / 160, 64 / 96])
    expected = sorted(zip(xyxy.tolist(), scores[0].tolist()))
    for (box, score), (exp_box, exp_score) in zip(boxes, expected):
        np.testing.assert_allclose(box, exp_box, atol=1e-3)
        assert score == pytest.approx(exp_score, abs=1e-5)


requires_hf_weights = pytest.mark.skipif(
    os.environ.get("SLEAP_NN_TEST_HF_WEIGHTS") != "1",
    reason="Downloads Roboflow/rf-detr-small; set SLEAP_NN_TEST_HF_WEIGHTS=1.",
)


@requires_hf_weights
def test_import_real_small_matches_hf(tmp_path):
    """The real RF-DETR-S matches HF's float64 outputs (first queries)."""
    ref = torch.load(ASSETS / "rfdetr_small_reference.pt", weights_only=False)
    out_dir = import_rfdetr("small", tmp_path / "small")
    model = _load_model(out_dir).eval().double()
    with torch.no_grad():
        out = model(synthetic_image(ref["side"]).double())
    k = ref["keep"]
    n = ref["pred_logits"].shape[-1]
    layers = [*out["aux_outputs"], out]

    def close(actual, expected):
        torch.testing.assert_close(
            actual[..., :k, :n].float(), expected, rtol=0, atol=ATOL
        )

    close(out["pred_logits"], ref["pred_logits"])
    close(out["pred_boxes"], ref["pred_boxes"])
    close(out["enc_outputs"]["pred_logits"], ref["enc_logits"])
    close(out["enc_outputs"]["pred_boxes"], ref["enc_boxes"])
    for i, layer in enumerate(layers):
        close(layer["pred_logits"], ref["layer_logits"][i])
        close(layer["pred_boxes"], ref["layer_boxes"][i])
    shutil.rmtree(out_dir)
