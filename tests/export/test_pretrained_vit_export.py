"""ONNX export of plain-ViT ("Case C") pretrained backbones.

A plain ViT feeding a spatial head interpolates its positional embeddings to the
token grid inside the graph. DINOv2-with-registers does that with an
*antialiased* bicubic resize (``aten::_upsample_bicubic2d_aa``), which the
default TorchScript exporter cannot lower at opset 17. These tests pin that the
export still yields a graph that matches PyTorch under onnxruntime at a batch
size and an input size other than the ones it was traced with.

Tiny random-init configs are written to a temp dir so the tests run offline and
fast on CPU.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sleap_nn.export.exporters import export_to_onnx
from sleap_nn.export.wrappers.base import BaseExportWrapper

from .conftest import requires_onnx, requires_onnxruntime

transformers = pytest.importorskip("transformers")

from sleap_nn.architectures.model import Model  # noqa: E402

pytestmark = [requires_onnx, requires_onnxruntime]


class _ConfmapsWrapper(BaseExportWrapper):
    """uint8 image -> raw confidence maps (no peak finding), for a dense check."""

    def forward(self, image):
        confmaps = self.model(self._normalize_uint8(image))
        return {"confmaps": self._extract_tensor(confmaps, ["confmap"])}


def _tiny_vit_dir(tmp_path, kind):
    """Save a tiny random-init ViT config so `PretrainedBackbone` loads offline."""
    if kind == "dinov2_with_registers":
        # Patch 14 (input resampled to a stride-16 grid) and a 16x16 positional
        # grid that the small export inputs *downsample* -> antialiased bicubic.
        config = transformers.Dinov2WithRegistersConfig(
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            intermediate_size=64,
            patch_size=14,
            image_size=224,
            num_register_tokens=4,
        )
    else:
        # Patch 16, plain (non-antialiased) bicubic positional interpolation.
        config = transformers.CLIPVisionConfig(
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            intermediate_size=64,
            patch_size=16,
            image_size=224,
        )
    path = tmp_path / kind
    config.save_pretrained(path)
    return path.as_posix()


def _caseC_model(model_dir):
    backbone_config = OmegaConf.create(
        dict(
            source="hf",
            model_name=model_dir,
            weights=False,
            mode="auto",
            in_channels=3,
            output_stride=4,
            max_stride=32,
            normalize=True,
            # Explicit stats: a bare config dir ships no image processor.
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )
    )
    heads = OmegaConf.create(
        {"confmaps": {"part_names": ["a", "b"], "sigma": 2.5, "output_stride": 4}}
    )
    torch.manual_seed(0)
    model = Model(
        backbone_type="pretrained",
        backbone_config=backbone_config,
        head_configs=heads,
        model_type="single_instance",
    ).eval()
    assert model.backbone.token_path and model.backbone.mode == "decoder"
    return model


@pytest.mark.parametrize("kind", ["dinov2_with_registers", "clip"])
def test_caseC_vit_onnx_export_parity(tmp_path, kind):
    """Exported Case C graph == PyTorch at a new batch size and input size."""
    import onnxruntime as ort

    model = _caseC_model(_tiny_vit_dir(tmp_path, kind))
    wrapper = _ConfmapsWrapper(model).eval()
    onnx_path = export_to_onnx(
        wrapper,
        tmp_path / f"{kind}.onnx",
        input_shape=(1, 3, 96, 96),
        output_names=["confmaps"],
        numerical_check=True,
        numerical_atol=1e-4,
        numerical_rtol=1e-4,
    )

    session = ort.InferenceSession(
        onnx_path.as_posix(), providers=["CPUExecutionProvider"]
    )
    for shape in [(1, 3, 96, 96), (3, 3, 64, 128), (2, 3, 160, 96)]:
        image = torch.randint(0, 256, shape, dtype=torch.uint8)
        with torch.no_grad():
            expected = wrapper(image)["confmaps"].numpy()
        (got,) = session.run(None, {"image": image.numpy()})
        assert (
            got.shape == expected.shape == (shape[0], 2, shape[2] // 4, shape[3] // 4)
        )
        np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


def test_rope_backbones_are_detected_for_static_export():
    """RoPE ViTs trace to a fixed grid, so the exporter must export them static."""
    timm = pytest.importorskip("timm")
    from sleap_nn.export.exporters.onnx_exporter import _has_rope

    assert _has_rope(timm.create_model("vit_small_patch16_dinov3", pretrained=False))
    assert _has_rope(timm.create_model("eva02_small_patch14_224", pretrained=False))
    assert not _has_rope(
        timm.create_model("vit_small_patch14_reg4_dinov2", pretrained=False)
    )


class RotaryEmbeddingStub(torch.nn.Module):
    """Stands in for a RoPE module: the exporter detects RoPE by class name."""


class _RopeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rope = RotaryEmbeddingStub()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def test_rope_backbone_exports_with_static_height_and_width(tmp_path):
    """A RoPE grid is baked into the trace, so only the batch stays dynamic."""
    import onnx

    onnx_path = export_to_onnx(
        _RopeNet(),
        tmp_path / "rope.onnx",
        input_shape=(1, 3, 64, 48),
        input_dtype=torch.float32,
        output_names=["out"],
    )
    dims = onnx.load(onnx_path.as_posix()).graph.input[0].type.tensor_type.shape.dim
    assert dims[0].dim_param  # batch stays dynamic
    assert [d.dim_value for d in dims[2:]] == [64, 48]
