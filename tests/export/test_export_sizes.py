"""Fixes from the review of #857 (export side): sizes the host sends, written to metadata."""

import json

import pytest
from click.testing import CliRunner


def _export(monkeypatch, ckpt, tmp_path, *extra, fill=None):
    """`sleap-nn export --format tensorrt` with the engine build stubbed out."""
    import sleap_nn.export.exporters as exporters

    seen = {}

    def fake_export_to_tensorrt(model, save_path, **kw):
        seen.update(kw)
        if fill and kw.get("build_info") is not None:
            kw["build_info"].update(fill)
        open(save_path, "wb").close()
        return save_path

    monkeypatch.setattr(exporters, "export_to_tensorrt", fake_export_to_tensorrt)
    from sleap_nn.export.cli import export

    out = tmp_path / "x"
    result = CliRunner().invoke(
        export, [str(ckpt), "-o", str(out), "--format", "tensorrt", *extra]
    )
    return result, seen, out


def test_network_only_traces_at_the_size_the_host_sends(
    monkeypatch, minimal_instance_centered_instance_ckpt, tmp_path
):
    """Crops are scaled and stride-padded on the host: graph and profile take that size."""
    onnx = pytest.importorskip("onnx")
    import math
    import shutil

    from omegaconf import OmegaConf

    from sleap_nn.export.cli import export

    ckpt = tmp_path / "ckpt"
    shutil.copytree(minimal_instance_centered_instance_ckpt, ckpt)
    cfg_path = ckpt / "training_config.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.data_config.preprocessing.scale = 0.5
    OmegaConf.save(cfg, cfg_path)
    crop = cfg.data_config.preprocessing.crop_size
    crop = int(crop[0]) if not isinstance(crop, int) else int(crop)
    bt = next(k for k, v in cfg.model_config.backbone_config.items() if v is not None)
    stride = int(cfg.model_config.backbone_config[bt]["max_stride"])
    want = math.ceil(crop * 0.5 / stride) * stride
    assert want < crop  # the case that used to fall below the engine's profile

    result, seen, out = _export(monkeypatch, ckpt, tmp_path, "--network-only")
    assert result.exit_code == 0, result.output
    for k in ("min_shape", "opt_shape", "max_shape"):
        assert tuple(seen[k][2:]) == (want, want), (k, seen[k])
    dims = onnx.load(str(out / "model.onnx")).graph.input[0].type.tensor_type.shape.dim
    assert all(d.dim_value in (0, want) for d in dims[2:])  # dynamic, or the host size
    meta = json.loads((out / "export_metadata.json").read_text())
    assert meta["input_scale"] == 0.5

    for flag in (["--crop-size", "64"], ["--input-scale", "1.0"]):
        result = CliRunner().invoke(
            export, [str(ckpt), "-o", str(tmp_path / "r"), "--network-only", *flag]
        )
        assert result.exit_code != 0 and "do not apply" in result.output


def test_checkpoint_wins_over_an_export_in_the_same_directory(tmp_path):
    """An export written next to best.ckpt does not replace the checkpoint (#857)."""
    from sleap_nn.inference.exported_network import is_network_export

    d = tmp_path / "m"
    d.mkdir()
    (d / "export_metadata.json").write_text(json.dumps({"network_only": True}))
    (d / "model.onnx").write_bytes(b"")
    assert is_network_export(d)
    (d / "best.ckpt").write_bytes(b"")
    assert not is_network_export(d)


def test_fp16_warning_for_vit_pose_models(capsys):
    """fp16 TensorRT exports of ViT-backbone pose models warn; UNets and fp32 do not."""
    from omegaconf import OmegaConf

    from sleap_nn.export.cli import _warn_fp16_vit_pose

    vit = OmegaConf.create(
        {
            "model_config": {
                "backbone_config": {
                    "pretrained": {
                        "model_name": "timm/vit_base_patch16_dinov3.lvd1689m"
                    }
                }
            }
        }
    )
    unet = OmegaConf.create(
        {"model_config": {"backbone_config": {"unet": {"filters": 16}}}}
    )
    _warn_fp16_vit_pose(vit, "centered_instance", "pretrained", "tensorrt", "fp16")
    assert "ViT backbone" in capsys.readouterr().out
    for args in [
        (vit, "centered_instance", "pretrained", "tensorrt", "bf16"),
        (vit, "centered_instance", "pretrained", "onnx", "fp16"),
        (vit, "bbox_detr", "pretrained", "tensorrt", "fp16"),
        (unet, "centered_instance", "unet", "tensorrt", "fp16"),
    ]:
        _warn_fp16_vit_pose(*args)
        assert capsys.readouterr().out == ""
