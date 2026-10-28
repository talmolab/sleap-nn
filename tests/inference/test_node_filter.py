"""Inference-time node filter (`node_filter` / `--nodes` / `--node_prefix`)."""

import shutil

import numpy as np
import pytest
import sleap_io as sio
import torch
from click.testing import CliRunner
from omegaconf import OmegaConf
from torch import nn

from sleap_nn.architectures.heads import CenteredInstanceConfmapsHead
from sleap_nn.inference.loaders import load_model_assets
from sleap_nn.inference.node_filter import (
    check_flip_partners,
    filter_training_config,
    output_names,
    resolve_node_filter,
    restrict_skeleton,
    slice_head_conv,
)
from sleap_nn.inference.predictor import Predictor

NODES = ["ds1/nose", "ds1/l_ear", "ds1/r_ear", "ds2/nose", "ds2/tail"]


def test_resolve_node_filter():
    assert resolve_node_filter(NODES, "ds1/") == [0, 1, 2]
    # A list keeps model (channel) order.
    assert resolve_node_filter(NODES, ["ds2/tail", "ds1/nose"]) == [0, 4]
    with pytest.raises(ValueError, match="prefix"):
        resolve_node_filter(NODES, "ds3/")
    with pytest.raises(ValueError, match="not model nodes"):
        resolve_node_filter(NODES, ["ds1/nose", "nope"])
    with pytest.raises(ValueError, match="selects no node"):
        resolve_node_filter(NODES, [])


def test_output_names():
    keep = NODES[:3]
    assert output_names(keep, "ds1/", None) == keep
    assert output_names(keep, "ds1/", True) == ["nose", "l_ear", "r_ear"]
    assert output_names(keep, ["ds1/nose"], "ds1/") == ["nose", "l_ear", "r_ear"]
    with pytest.raises(ValueError, match="list of nodes"):
        output_names(keep, ["ds1/nose"], True)
    with pytest.raises(ValueError, match="repeated"):
        output_names(["a/x", "x"], ["a/x", "x"], "a/")


def _skeleton():
    return sio.Skeleton(
        nodes=list(NODES),
        edges=[
            ("ds1/nose", "ds1/l_ear"),
            ("ds1/nose", "ds2/nose"),
            ("ds2/nose", "ds2/tail"),
        ],
        symmetries=[("ds1/l_ear", "ds1/r_ear")],
        name="merged",
    )


def test_restrict_skeleton_keeps_inner_edges_and_symmetries():
    sk = restrict_skeleton(_skeleton(), NODES[:3], {n: n[4:] for n in NODES[:3]})
    assert sk.node_names == ["nose", "l_ear", "r_ear"]
    assert [(e.source.name, e.destination.name) for e in sk.edges] == [
        ("nose", "l_ear")
    ]
    assert [sorted(n.name for n in s.nodes) for s in sk.symmetries] == [
        ["l_ear", "r_ear"]
    ]
    assert sk.name == "merged"


def test_check_flip_partners():
    check_flip_partners(_skeleton(), NODES[:3])
    check_flip_partners(_skeleton(), NODES[3:])
    with pytest.raises(ValueError, match="symmetric partner"):
        check_flip_partners(_skeleton(), ["ds1/nose", "ds1/l_ear"])


@pytest.mark.parametrize("kernel_size", [1, 3])
def test_slice_head_conv_keeps_exact_channels(kernel_size):
    """A sliced head outputs exactly the selected channels of the full head."""
    torch.manual_seed(0)
    head = CenteredInstanceConfmapsHead(part_names=[f"n{i}" for i in range(300)])
    layer = head.make_head(x_in=32, kernel_size=kernel_size)
    model = nn.Module()
    model.heads = [head]
    model.head_layers = nn.ModuleList([layer])
    x = torch.randn(2, 32, 24, 20)
    with torch.no_grad():
        full = layer(x)
    idx = list(range(40, 55))
    new = slice_head_conv(model, head.name, idx)
    assert new.out_channels == 15 and new.kernel_size == (kernel_size,) * 2
    with torch.no_grad():
        sliced = model.head_layers[0](x)
    torch.testing.assert_close(sliced, full[:, idx], rtol=0, atol=0)


def _renamed_model_dir(src, dst, rename):
    """A copy of a model dir with its nodes renamed in the training config."""
    shutil.copytree(src, dst)
    cfg = OmegaConf.load(dst / "training_config.yaml")
    sk = cfg.data_config.skeletons[0]
    for node in sk.nodes:
        node.name = rename[node.name]
    for edge in sk.edges:
        edge.source.name = rename[edge.source.name]
        edge.destination.name = rename[edge.destination.name]
    for head in cfg.model_config.head_configs.values():
        if head is not None and head.get("confmaps") is not None:
            if head.confmaps.get("part_names"):
                head.confmaps.part_names = [rename[n] for n in head.confmaps.part_names]
    OmegaConf.save(cfg, dst / "training_config.yaml")
    return dst


def _module(assets, model_type):
    if model_type == "single_instance":
        return assets.inference_model.torch_model
    return assets.inference_model.instance_peaks.torch_model


@pytest.mark.parametrize(
    "fixture,model_type",
    [
        ("minimal_instance_centered_instance_ckpt", "centered_instance"),
        ("minimal_instance_single_instance_ckpt", "single_instance"),
    ],
)
def test_filtered_confmaps_equal_unfiltered_channels(fixture, model_type, request):
    model_dir = request.getfixturevalue(fixture)
    full, _ = load_model_assets([str(model_dir)], device="cpu")
    filt, _ = load_model_assets([str(model_dir)], device="cpu", node_filter=["B"])
    assert full.skeletons[0].node_names == ["A", "B"]
    assert filt.skeletons[0].node_names == ["B"]
    cfg = filt.confmap_config.model_config.head_configs[model_type].confmaps
    assert list(cfg.part_names) == ["B"]
    torch.manual_seed(0)
    x = torch.rand(2, 1, 1, 160, 160)
    with torch.no_grad():
        cms_full = _module(full, model_type)(x)
        cms_filt = _module(filt, model_type)(x)
    assert cms_full.shape[1] == 2 and cms_filt.shape[1] == 1
    torch.testing.assert_close(cms_filt, cms_full[:, [1]], rtol=0, atol=0)


def _points(labels):
    return np.stack(
        [inst.numpy() for lf in labels.labeled_frames for inst in lf.instances]
    )


@pytest.mark.parametrize("flip_test", [False, True])
def test_topdown_predictions_match_unfiltered_subset(
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    minimal_instance,
    flip_test,
):
    paths = [
        str(minimal_instance_centroid_ckpt),
        str(minimal_instance_centered_instance_ckpt),
    ]
    kw = dict(device="cpu", peak_threshold=0.0, flip_test=flip_test)
    full = Predictor.from_model_paths(paths, **kw).predict(str(minimal_instance))
    filt = Predictor.from_model_paths(paths, node_filter=["B"], **kw).predict(
        str(minimal_instance)
    )
    assert filt.skeletons[0].node_names == ["B"]
    p_full, p_filt = _points(full), _points(filt)
    assert p_filt.shape == (len(p_full), 1, 2)
    np.testing.assert_allclose(p_filt, p_full[:, [1]], atol=1e-4)


def test_single_instance_predictions_match_unfiltered_subset(
    minimal_instance_single_instance_ckpt, minimal_instance
):
    path = [str(minimal_instance_single_instance_ckpt)]
    full = Predictor.from_model_paths(path, device="cpu", peak_threshold=0.0)
    filt = Predictor.from_model_paths(
        path, device="cpu", peak_threshold=0.0, node_filter="A"
    )
    p_full = _points(full.predict(str(minimal_instance)))
    p_filt = _points(filt.predict(str(minimal_instance)))
    np.testing.assert_allclose(p_filt, p_full[:, [0]], atol=1e-4)


def test_prefix_filter_strips_names(
    minimal_instance_centered_instance_ckpt, minimal_instance, tmp_path
):
    model_dir = _renamed_model_dir(
        minimal_instance_centered_instance_ckpt,
        tmp_path / "ci",
        {"A": "ds1/A", "B": "ds2/B"},
    )
    pred = Predictor.from_model_paths(
        [str(model_dir)],
        device="cpu",
        peak_threshold=0.0,
        node_filter="ds2/",
        strip_prefix=True,
    )
    out = pred.predict(str(minimal_instance))
    assert out.skeletons[0].node_names == ["B"]
    assert all(len(i.numpy()) == 1 for lf in out for i in lf.instances)


def test_node_filter_rejects_other_model_types(minimal_instance_centroid_ckpt):
    with pytest.raises(ValueError, match="single_instance and centered_instance"):
        load_model_assets(
            [str(minimal_instance_centroid_ckpt)], device="cpu", node_filter=["A"]
        )
    with pytest.raises(ValueError, match="needs a node_filter"):
        load_model_assets(
            [str(minimal_instance_centroid_ckpt)], device="cpu", strip_prefix=True
        )


def test_filter_training_config(minimal_instance_centered_instance_ckpt, tmp_path):
    model_dir = _renamed_model_dir(
        minimal_instance_centered_instance_ckpt,
        tmp_path / "ci",
        {"A": "ds1/A", "B": "ds2/B"},
    )
    cfg = OmegaConf.load(model_dir / "training_config.yaml")
    out, idx = filter_training_config(cfg, "centered_instance", "ds2/", True)
    assert idx == [1]
    head = out.model_config.head_configs.centered_instance.confmaps
    assert list(head.part_names) == ["B"]
    sk = out.data_config.skeletons[0]
    assert [n["name"] for n in sk["nodes"]] == ["B"]
    assert sk["name"] == cfg.data_config.skeletons[0].name
    # The input config is untouched.
    assert list(
        cfg.model_config.head_configs.centered_instance.confmaps.part_names
    ) == [
        "ds1/A",
        "ds2/B",
    ]


def test_predict_cli_nodes(minimal_instance_centered_instance_ckpt, tmp_path):
    from sleap_nn.cli import cli

    data = "tests/assets/datasets/minimal_instance.pkg.slp"
    out = tmp_path / "out.slp"
    result = CliRunner().invoke(
        cli,
        [
            "predict",
            "--data_path",
            data,
            "--model_paths",
            str(minimal_instance_centered_instance_ckpt),
            "--nodes",
            "B",
            "--device",
            "cpu",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    labels = sio.load_slp(str(out))
    assert labels.skeletons[0].node_names == ["B"]

    result = CliRunner().invoke(
        cli,
        [
            "predict",
            "--data_path",
            data,
            "--model_paths",
            str(minimal_instance_centered_instance_ckpt),
            "--nodes",
            "B",
            "--node_prefix",
            "x/",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "not both" in result.output


def test_export_with_node_prefix(minimal_instance_centered_instance_ckpt, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    from sleap_nn.export.cli import export

    model_dir = _renamed_model_dir(
        minimal_instance_centered_instance_ckpt,
        tmp_path / "ci",
        {"A": "ds1/A", "B": "ds2/B"},
    )
    outs = {}
    for name, extra in (
        ("full", []),
        ("filt", ["--node-prefix", "ds2/", "--strip-prefix"]),
    ):
        result = CliRunner().invoke(
            export,
            [str(model_dir), "-o", str(tmp_path / name), "--format", "onnx"] + extra,
        )
        assert result.exit_code == 0, result.output
        outs[name] = tmp_path / name

    meta = OmegaConf.load(outs["filt"] / "training_config.yaml")
    assert list(
        meta.model_config.head_configs.centered_instance.confmaps.part_names
    ) == ["B"]
    import json

    md = json.loads((outs["filt"] / "export_metadata.json").read_text())
    assert md["node_names"] == ["B"] and md["n_nodes"] == 1

    rng = np.random.default_rng(0)
    sess = {k: ort.InferenceSession(str(v / "model.onnx")) for k, v in outs.items()}
    inp = sess["full"].get_inputs()[0]
    crop = int(meta.data_config.preprocessing.crop_size)
    shape = [1, 1, crop, crop]
    x = rng.integers(0, 255, shape, dtype=np.uint8)
    res = {k: s.run(None, {inp.name: x}) for k, s in sess.items()}
    peaks_full, vals_full = res["full"][:2]
    peaks_filt, vals_filt = res["filt"][:2]
    np.testing.assert_allclose(peaks_filt, peaks_full[:, [1]], atol=1e-4)
    np.testing.assert_allclose(vals_filt, vals_full[:, [1]], atol=1e-5)


def test_export_topdown_bundle_with_nodes(
    minimal_instance_centroid_ckpt, minimal_instance_centered_instance_ckpt, tmp_path
):
    """The centered-instance stage of a top-down bundle is exported filtered."""
    pytest.importorskip("onnx")
    import json

    from sleap_nn.export.cli import export

    out = tmp_path / "td"
    result = CliRunner().invoke(
        export,
        [
            str(minimal_instance_centroid_ckpt),
            str(minimal_instance_centered_instance_ckpt),
            "-o",
            str(out),
            "--format",
            "onnx",
            "--nodes",
            "B",
            "--no-verify",
        ],
    )
    assert result.exit_code == 0, result.output
    md = json.loads((out / "export_metadata.json").read_text())
    assert md["node_names"] == ["B"] and md["n_nodes"] == 1
    cfg = OmegaConf.load(out / "training_config_centered_instance.yaml")
    assert list(
        cfg.model_config.head_configs.centered_instance.confmaps.part_names
    ) == ["B"]

    # Filters target the pose model; a lone centroid model is refused.
    result = CliRunner().invoke(
        export,
        [
            str(minimal_instance_centroid_ckpt),
            "-o",
            str(tmp_path / "c"),
            "--nodes",
            "A",
        ],
    )
    assert (
        result.exit_code != 0
        and "single_instance or centered_instance" in result.output
    )
