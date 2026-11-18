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


@pytest.mark.parametrize(
    "crop, scale, stride",
    [(258, 0.25, 32), (144, 0.45, 32), (160, 0.5, 16), (192, 1.0, 32), (100, 0.75, 8)],
)
def test_network_only_size_is_the_host_size(crop, scale, stride):
    """The traced size equals what the host's resize + stride padding produce."""
    import torch

    from sleap_nn.data.resizing import apply_pad_to_stride, resize_image
    from sleap_nn.export.cli import _network_only_hw

    host = apply_pad_to_stride(
        resize_image(torch.zeros(1, 1, crop, crop), scale), stride
    )
    assert _network_only_hw((crop, crop), scale, stride) == tuple(host.shape[-2:])


def test_vit_warning_covers_every_model_of_a_bundle(
    monkeypatch,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    tmp_path,
):
    """Two-model (centroid + instance) exports check each model for the ViT warning."""
    import sleap_nn.export.cli as export_cli

    calls = []
    monkeypatch.setattr(
        export_cli, "_warn_fp16_vit_pose", lambda c, mt, bt, f, p: calls.append(mt)
    )
    CliRunner().invoke(
        export_cli.export,
        [
            str(minimal_instance_centroid_ckpt),
            str(minimal_instance_centered_instance_ckpt),
            "-o",
            str(tmp_path / "x"),
            "--format",
            "onnx",
        ],
    )
    assert sorted(calls) == ["centered_instance", "centroid"]


def test_vit_warning_names_what_was_measured(capsys):
    """The text claims the DINOv3 measurement only and names no pose-only model kind."""
    from omegaconf import OmegaConf

    from sleap_nn.export.cli import _warn_fp16_vit_pose

    vit = OmegaConf.create(
        {
            "model_config": {
                "backbone_config": {
                    "pretrained": {"model_name": "timm/vit_base_patch16_dinov3"}
                }
            }
        }
    )
    _warn_fp16_vit_pose(vit, "embedding", "pretrained", "tensorrt", "fp16")
    out = capsys.readouterr().out
    assert "DINOv3" in out and "other ViTs not tested" in out
    assert "embedding model" in out and "pose model" not in out


def _fake_export_dir(path, meta, ckpt=False):
    path.mkdir(parents=True)
    (path / "export_metadata.json").write_text(json.dumps(meta))
    (path / "model.onnx").write_bytes(b"")
    if ckpt:
        (path / "best.ckpt").write_bytes(b"")
    return str(path)


def test_part_model_check_counts_checkpoint_directories(tmp_path):
    """--part_model refuses only what it refused before; best.ckpt directories load it."""
    import click

    from sleap_nn.cli import _check_part_stage_flags

    net = {"network_only": True, "model_type": "centered_instance"}
    det = {"model_type": "bbox_detr"}
    fused = {"model_type": "centered_instance"}
    pose_ck = _fake_export_dir(tmp_path / "pose_ck", net, ckpt=True)
    det_ck = _fake_export_dir(tmp_path / "det_ck", det, ckpt=True)
    tail_ck = _fake_export_dir(tmp_path / "tail_ck", net, ckpt=True)
    fused_ck = _fake_export_dir(tmp_path / "fused_ck", fused, ckpt=True)
    fused_only = _fake_export_dir(tmp_path / "fused_only", fused)

    def check(mps, part):
        kw = dict(model_paths=mps, part_model=part, part_crop_rule="rule.json")
        _check_part_stage_flags(kw, None)

    # bedae2f7 accepted these; the checkpoint-wins rule must not refuse them now.
    check([det_ck, pose_ck], tail_ck)
    check([pose_ck], tail_ck)
    # Several -m load best.ckpt: a fused export next to it is not what runs.
    check([det_ck, fused_ck], tail_ck)
    # Still refused as before: a fused export alone, and a single -m fused directory
    # (that route takes the export whatever else the directory holds).
    for mps in ([det_ck, fused_only], [fused_ck]):
        with pytest.raises(click.UsageError, match="not fused exported"):
            check(mps, tail_ck)


def test_provenance_lists_part_model_exports(monkeypatch):
    """A part model's exported graphs are listed at the top level, tagged stage: part."""
    from types import SimpleNamespace

    import sleap_nn.inference.provenance as prov
    from sleap_nn.inference.predictor import Predictor

    body, part = object(), object()
    graphs = {
        id(body): [{"file": "det/model.trt", "runtime": "tensorrt"}],
        id(part): [{"file": "tail/model.trt", "runtime": "tensorrt"}],
    }
    monkeypatch.setattr(prov, "_describe_exported_runtimes", lambda r: graphs[id(r)])
    me = SimpleNamespace(
        tracker_config=None,
        model_paths=["det", "pose"],
        layer=body,
        device="cpu",
        part_stage=SimpleNamespace(part_predictor=SimpleNamespace(layer=part)),
    )
    out = Predictor._build_inference_provenance(
        me, source=None, start_time=None, end_time=None, n_frames=0, inference_params={}
    )
    assert out["exported_models"] == [
        {"file": "det/model.trt", "runtime": "tensorrt"},
        {"file": "tail/model.trt", "runtime": "tensorrt", "stage": "part"},
    ]
    two_pass = {"exported_models": [{"file": "det/model.trt", "runtime": "tensorrt"}]}
    prov._add_part_exports(two_pass, [{"file": "tail/model.onnx", "runtime": "onnx"}])
    assert two_pass["exported_models"][-1] == {
        "file": "tail/model.onnx",
        "runtime": "onnx",
        "stage": "part",
    }
