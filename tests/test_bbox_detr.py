"""Tests for the ``bbox_detr`` model type's config, set decoder and squash resize.

The model modules are tested in ``tests/architectures/detr`` and the set losses in
``tests/training/detr``.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sleap_nn.config.get_config import get_head_configs
from sleap_nn.config.model_config import BBOX_MODEL_TYPES
from sleap_nn.config.training_job_config import TrainingJobConfig, verify_training_cfg
from sleap_nn.config.utils import get_model_type_from_cfg
from sleap_nn.inference.bbox import BBoxDecoder, decode_set


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_bbox_detr_head_config_defaults():
    """The preset carries RF-DETR's defaults and is a registered bbox model type."""
    assert "bbox_detr" in BBOX_MODEL_TYPES
    leaf = get_head_configs("bbox_detr").bbox_detr.detr
    assert (leaf.hidden_dim, leaf.num_queries, leaf.group_detr) == (256, 300, 13)
    assert (leaf.dec_layers, leaf.sa_nheads, leaf.ca_nheads) == (3, 8, 16)
    assert (leaf.dec_n_points, leaf.dim_feedforward, leaf.output_stride) == (
        2,
        2048,
        16,
    )
    assert (leaf.cls_loss_coef, leaf.bbox_loss_coef, leaf.giou_loss_coef) == (1, 5, 2)
    assert (leaf.set_cost_class, leaf.set_cost_bbox, leaf.set_cost_giou) == (2, 5, 2)

    hc = get_head_configs({"bbox_detr": {"detr": {"num_queries": 100}}})
    assert hc.bbox_detr.detr.num_queries == 100
    cfg = OmegaConf.create({"model_config": {"head_configs": OmegaConf.structured(hc)}})
    assert get_model_type_from_cfg(cfg) == "bbox_detr"


def test_bbox_detr_config_yaml_round_trip(tmp_path):
    """A full training config with a bbox_detr head survives YAML save/load."""
    from sleap_nn.config.model_config import BackboneConfig, ModelConfig, UNetConfig

    config = OmegaConf.structured(
        TrainingJobConfig(
            model_config=ModelConfig(
                backbone_config=BackboneConfig(unet=UNetConfig()),
                head_configs=get_head_configs(
                    {
                        "bbox_detr": {
                            "detr": {
                                "categories": ["person"],
                                "dec_layers": 4,
                                "group_detr": 7,
                                "focal_alpha": 0.3,
                            }
                        }
                    }
                ),
            )
        )
    )
    config.data_config.preprocessing.update(
        max_height=576, max_width=576, resize_mode="squash"
    )
    path = tmp_path / "config.yaml"
    OmegaConf.save(config, path)
    loaded = verify_training_cfg(OmegaConf.load(path))
    leaf = loaded.model_config.head_configs.bbox_detr.detr
    assert list(leaf.categories) == ["person"]
    assert (leaf.dec_layers, leaf.group_detr, leaf.focal_alpha) == (4, 7, 0.3)
    assert leaf.num_select == 300
    assert loaded.data_config.preprocessing.resize_mode == "squash"
    assert get_model_type_from_cfg(loaded) == "bbox_detr"


def test_resize_mode_validation():
    from sleap_nn.config.data_config import PreprocessingConfig
    from sleap_nn.inference.layers.configs import PreprocessConfig

    assert PreprocessingConfig().resize_mode == "letterbox"
    with pytest.raises(ValueError, match="letterbox"):
        PreprocessingConfig(resize_mode="stretch")
    with pytest.raises(ValueError, match="max_height"):
        PreprocessingConfig(max_height=576, resize_mode="squash")
    with pytest.raises(ValueError, match="max_height"):
        PreprocessConfig(max_width=576, resize_mode="squash")


@pytest.mark.parametrize("model_type", ["bbox_detr", "bbox_ssd"])
def test_squash_only_for_bbox_detr(model_type):
    from sleap_nn.config.utils import check_output_strides

    cfg = OmegaConf.create(
        {
            "model_config": {
                "backbone_config": {"unet": {"max_stride": 16, "output_stride": 1}},
                "head_configs": OmegaConf.structured(get_head_configs(model_type)),
            },
            "data_config": {"preprocessing": {"resize_mode": "squash"}},
        }
    )
    if model_type == "bbox_detr":
        cfg = check_output_strides(cfg)
        assert cfg.model_config.backbone_config.unet.output_stride == 16
    else:
        with pytest.raises(ValueError, match="only supported for bbox_detr"):
            check_output_strides(cfg)


def test_bbox_detr_training_not_available_yet():
    from sleap_nn.training.lightning_modules import LightningModel

    cfg = OmegaConf.create(
        {
            "model_config": {
                "backbone_config": {"unet": {}},
                "head_configs": OmegaConf.structured(get_head_configs("bbox_detr")),
            }
        }
    )
    with pytest.raises(NotImplementedError, match="not trained yet"):
        LightningModel.get_lightning_model_from_config(cfg)


# --------------------------------------------------------------------------- #
# Set decoder
# --------------------------------------------------------------------------- #
def _logits(probs):
    """Logits with the given sigmoid probabilities (an extra last column added)."""
    probs = torch.as_tensor(probs, dtype=torch.float32)
    logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
    extra = torch.full(logits.shape[:-1] + (1,), 10.0)  # never selected
    return torch.cat([logits, extra], dim=-1)


def test_decode_set_top_k_over_queries_and_classes():
    # 1 image, 3 queries, 2 real classes (+ the extra logit, which is ignored).
    logits = _logits([[[0.9, 0.1], [0.2, 0.8], [0.05, 0.6]]])
    boxes = torch.tensor(
        [[[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1], [0.9, 0.9, 0.4, 0.4]]]
    )
    out_boxes, scores, classes = decode_set(
        logits,
        boxes,
        img_hw=(100, 200),
        score_threshold=0.15,
        num_select=4,
        max_detections=6,
    )
    assert out_boxes.shape == (1, 6, 4)
    # Top-4 pairs: (q0, c0) .9, (q1, c1) .8, (q2, c1) .6, (q1, c0) .2; all >= .15.
    torch.testing.assert_close(
        scores[0, :4], torch.tensor([0.9, 0.8, 0.6, 0.2]), atol=1e-5, rtol=0
    )
    assert classes[0].tolist() == [0, 1, 1, 0, -1, -1]
    # cxcywh normalized -> xyxy input pixels (W = 200, H = 100).
    torch.testing.assert_close(out_boxes[0, 0], torch.tensor([80.0, 30.0, 120.0, 70.0]))
    torch.testing.assert_close(out_boxes[0, 1], torch.tensor([40.0, 20.0, 60.0, 30.0]))
    torch.testing.assert_close(out_boxes[0, 3], out_boxes[0, 1])  # same query
    # Clipped to the input: x2 = 220 -> 199, y2 = 110 -> 99.
    torch.testing.assert_close(
        out_boxes[0, 2], torch.tensor([140.0, 70.0, 199.0, 99.0])
    )
    assert torch.isnan(out_boxes[0, 4:]).all() and torch.isnan(scores[0, 4:]).all()

    # The threshold drops low pairs; num_select caps candidates before it.
    _, scores, _ = decode_set(logits, boxes, (100, 200), score_threshold=0.5)
    assert torch.isfinite(scores[0]).sum() == 3
    _, scores, _ = decode_set(logits, boxes, (100, 200), 0.0, num_select=2)
    assert torch.isfinite(scores[0]).sum() == 2


def test_bbox_decoder_set_style_from_config():
    hc = OmegaConf.structured(
        get_head_configs(
            {"bbox_detr": {"detr": {"categories": ["a", "b"], "num_select": 2}}}
        )
    )
    dec = BBoxDecoder.from_head_configs("bbox_detr", hc, score_threshold=0.0)
    assert dec.iou_threshold is None  # no NMS by default
    assert dec.categories == ["a", "b"] and dec.num_select == 2
    logits = _logits([[[0.9, 0.1], [0.2, 0.8], [0.05, 0.6]]])
    boxes = torch.rand(1, 3, 4) * 0.2 + 0.4
    out_boxes, scores, classes = dec(
        {"pred_logits": logits, "pred_boxes": boxes}, img_hw=(64, 64)
    )
    assert torch.isfinite(scores[0]).sum() == 2
    assert classes[0, :2].tolist() == [0, 1]


# --------------------------------------------------------------------------- #
# Squash resize
# --------------------------------------------------------------------------- #
def test_squash_resize_round_trip():
    """Boxes mapped into a squashed image and back land where they started."""
    from sleap_nn.data.resizing import apply_squash_resize
    from sleap_nn.inference.ops.coord import undo_eff_scale

    image = torch.rand(1, 120, 300)
    squashed, (sx, sy) = apply_squash_resize(image, 64, 64)
    assert squashed.shape == (1, 64, 64)
    assert (sx, sy) == (64 / 300, 64 / 120)
    boxes = torch.tensor([[[30.0, 12.0, 150.0, 90.0], [0.0, 0.0, 300.0, 120.0]]])
    in_model = boxes * torch.tensor([sx, sy, sx, sy])
    torch.testing.assert_close(in_model[0, 1], torch.tensor([0.0, 0.0, 64.0, 64.0]))
    corners = in_model.reshape(1, -1, 2)
    back = undo_eff_scale(corners, torch.tensor([[sx, sy]])).reshape(boxes.shape)
    torch.testing.assert_close(back, boxes)
    # Same-size frames pass through untouched.
    same, scale = apply_squash_resize(squashed, 64, 64)
    assert same is squashed and scale == (1.0, 1.0)


class _FixedBoxes(torch.nn.Module):
    """A detector stand-in: one box covering the middle half of the input."""

    def __init__(self):
        super().__init__()
        self.input_hw = None

    def forward(self, x):
        self.input_hw = tuple(x.shape[-2:])
        b = x.shape[0]
        logits = torch.full((b, 2, 2), -10.0)
        logits[:, 0, 0] = 10.0
        boxes = torch.tensor([0.5, 0.5, 0.5, 0.5]).expand(b, 2, 4)
        return {"pred_logits": logits, "pred_boxes": boxes}


@pytest.mark.parametrize("resize_mode", ["squash", "letterbox"])
def test_bbox_layer_squash_resize(resize_mode):
    """Squash maps model boxes back per axis; letterbox keeps today's behaviour."""
    from sleap_nn.inference.layers.backends import TorchBackend
    from sleap_nn.inference.layers.bbox import BBoxLayer
    from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig

    model = _FixedBoxes()
    layer = BBoxLayer(
        backend=TorchBackend(model=model, device="cpu"),
        decoder=BBoxDecoder("bbox_detr", ["a"], score_threshold=0.5),
        output_stride=16,
        max_stride=16,
        preprocess_config=PreprocessConfig(
            max_height=64, max_width=64, resize_mode=resize_mode
        ),
        postprocess_config=PostprocessConfig(peak_threshold=0.5, max_instances=5),
    )
    out = layer.predict(torch.zeros(1, 1, 120, 300))
    assert model.input_hw == (64, 64)
    box = out.pred_bboxes[0, 0]
    assert torch.isfinite(out.pred_bbox_scores[0]).sum() == 1
    if resize_mode == "squash":
        # The middle half of the 64x64 input is the middle half of the frame.
        torch.testing.assert_close(box, torch.tensor([75.0, 30.0, 225.0, 90.0]))
    else:
        # Letterbox: 300x120 -> 64x25.6 (scale 64/300), padded below; the box
        # [16, 16, 48, 48] maps back by 300/64 and is clipped to the frame.
        s = 300 / 64
        torch.testing.assert_close(
            box, torch.tensor([16 * s, 16 * s, 48 * s, 119.0]), rtol=0, atol=1e-3
        )


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_detr_model():
    """A small ``Model(bbox_detr)`` with fully random weights (no zero-inits)."""
    from sleap_nn.architectures.detr import DETR_BACKBONES, TinyDetrBackbone
    from sleap_nn.architectures.model import Model

    DETR_BACKBONES["unet"] = lambda cfg: TinyDetrBackbone(
        in_channels=cfg.in_channels, out_channels=16, stride=16
    )
    try:
        hc = OmegaConf.structured(get_head_configs("bbox_detr"))
        hc.bbox_detr.detr.update(
            categories=["a", "b"],
            hidden_dim=32,
            num_queries=10,
            group_detr=2,
            dec_layers=2,
            sa_nheads=2,
            ca_nheads=4,
            dim_feedforward=64,
            num_select=20,
        )
        torch.manual_seed(0)
        model = Model(
            backbone_type="unet",
            backbone_config=OmegaConf.create({"in_channels": 1, "output_stride": 16}),
            head_configs=hc.bbox_detr,
            model_type="bbox_detr",
        ).eval()
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0, 0.1)
        yield model, hc
    finally:
        del DETR_BACKBONES["unet"]


def test_bbox_detr_onnx_export(tiny_detr_model, tmp_path):
    """The exported graph matches PyTorch, also at sizes other than the trace."""
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from sleap_nn.export.exporters.onnx_exporter import export_to_onnx
    from sleap_nn.export.wrappers.bbox import BBoxONNXWrapper

    model, _ = tiny_detr_model
    wrapper = BBoxONNXWrapper(model, model_type="bbox_detr").eval()
    path = export_to_onnx(
        wrapper,
        tmp_path / "detr.onnx",
        input_shape=(1, 1, 96, 80),
        numerical_check=True,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [o.name for o in session.get_outputs()] == ["pred_logits", "pred_boxes"]
    for shape in [(1, 1, 96, 80), (3, 1, 96, 80), (2, 1, 128, 192)]:
        x = torch.randint(0, 256, shape, dtype=torch.uint8)
        with torch.no_grad():
            expected = wrapper(x)
        logits, boxes = session.run(None, {"image": x.numpy()})
        assert logits.shape == (shape[0], 10, 3)
        np.testing.assert_allclose(
            logits, expected["pred_logits"].numpy(), atol=1e-4, rtol=0
        )
        np.testing.assert_allclose(
            boxes, expected["pred_boxes"].numpy(), atol=1e-4, rtol=0
        )


@pytest.mark.parametrize("resize_mode", ["squash", "letterbox"])
def test_exported_bbox_detr_layer_matches_native(tiny_detr_model, resize_mode):
    """ExportedBBoxLayer + the export wrapper give the native layer's boxes."""
    from sleap_nn.export.wrappers.bbox import BBoxONNXWrapper
    from sleap_nn.inference.layers.backends import TorchBackend
    from sleap_nn.inference.layers.bbox import BBoxLayer, ExportedBBoxLayer
    from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig

    model, hc = tiny_detr_model

    class _Native(torch.nn.Module):
        """The checkpoint path's forward: drop n_samples, scale to [0, 1]."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x.squeeze(1) / 255.0)
            return {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}

    decoder = BBoxDecoder.from_head_configs("bbox_detr", hc, score_threshold=0.0)
    kwargs = dict(
        decoder=decoder,
        output_stride=16,
        max_stride=16,
        preprocess_config=PreprocessConfig(
            max_height=96, max_width=128, resize_mode=resize_mode
        ),
        postprocess_config=PostprocessConfig(peak_threshold=0.0, max_instances=20),
    )
    frame = torch.randint(0, 256, (1, 1, 150, 230), dtype=torch.uint8)
    native = BBoxLayer(
        backend=TorchBackend(model=_Native(model), device="cpu"), **kwargs
    ).predict(frame.float())
    wrapper = BBoxONNXWrapper(model, model_type="bbox_detr").eval()
    exported = ExportedBBoxLayer(
        backend=TorchBackend(model=wrapper, device="cpu"), **kwargs
    ).predict(frame.numpy())
    valid = torch.isfinite(native.pred_bbox_scores[0])
    assert valid.sum() > 0
    # The exported path rounds the resized frame to uint8 (the graph's input), so
    # allow sub-pixel differences.
    torch.testing.assert_close(
        exported.pred_bbox_scores[0][valid],
        native.pred_bbox_scores[0][valid],
        atol=1e-4,
        rtol=0,
    )
    torch.testing.assert_close(
        exported.pred_bboxes[0][valid], native.pred_bboxes[0][valid], atol=1e-3, rtol=0
    )
    boxes = native.pred_bboxes[0][valid]
    assert boxes[:, [0, 2]].max() <= 229 and boxes[:, [1, 3]].max() <= 149
