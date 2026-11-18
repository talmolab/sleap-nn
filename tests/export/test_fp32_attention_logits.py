"""`--fp32-attention-logits`: fp16 engines with each attention logits product in fp32.

The graph-level parts run on synthetic ONNX graphs; the build path runs against the
stub ``tensorrt`` modules of ``test_onnx_precision.py`` (no TensorRT or GPU needed).
"""

import sys

import numpy as np
import pytest
import torch
from click.testing import CliRunner

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from sleap_nn.export.exporters.onnx_precision import (  # noqa: E402
    cast_onnx_precision,
    find_attention_logits,
)
from sleap_nn.export.exporters.tensorrt_exporter import (  # noqa: E402
    BACKBONE_SCOPE,
    _check_fp32_islands,
    _export_tensorrt_onnx,
    _write_typed_graph,
)
from tests.export.test_onnx_precision import _fake_trt, _graph  # noqa: E402

F = TensorProto.FLOAT
T, D = 4, 8  # tokens, width


def _attention_graph(kind="basic", blocks=1, prefix="/model/backbone/blocks"):
    """x -> [attention block] x blocks -> head MatMul (outside the backbone)."""
    rng = np.random.default_rng(0)
    inits, nodes = [], []
    inputs = [helper.make_tensor_value_info("x", F, [1, T, D])]

    def init(name, a):
        inits.append(numpy_helper.from_array(np.asarray(a, np.float32), name))
        return name

    def node(op, ins, name, **attrs):
        out = name + "_out"
        nodes.append(helper.make_node(op, ins, [out], name=name, **attrs))
        return out

    if kind == "mask":
        inputs.append(helper.make_tensor_value_info("mask", F, [1, T, T]))
    if kind == "where":
        inputs.append(
            helper.make_tensor_value_info("keep", TensorProto.BOOL, [1, T, T])
        )
    h = "x"
    for b in range(blocks):
        p = f"{prefix}.{b}/attn"
        q = node(
            "MatMul", [h, init(f"wq{b}", rng.normal(size=(D, D)))], f"{p}/q/MatMul"
        )
        k = node(
            "MatMul", [h, init(f"wk{b}", rng.normal(size=(D, D)))], f"{p}/k/MatMul"
        )
        v = node(
            "MatMul", [h, init(f"wv{b}", rng.normal(size=(D, D)))], f"{p}/v/MatMul"
        )
        scale = init(f"scale{b}", 1.0 / np.sqrt(D))
        if kind == "einsum":
            logits = node("Einsum", [q, k], f"{p}/Einsum", equation="btd,bsd->bts")
        else:
            if kind == "scale_before":
                q = node("Mul", [q, scale], f"{p}/Mul_q")
            kt = node("Transpose", [k], f"{p}/Transpose", perm=[0, 2, 1])
            logits = node("MatMul", [q, kt], f"{p}/MatMul")
        s = logits
        if kind == "div_after":
            s = node("Div", [s, init(f"sqrt{b}", np.sqrt(D))], f"{p}/Div")
        elif kind != "scale_before":
            s = node("Mul", [s, scale], f"{p}/Mul")
        if kind == "mask":
            s = node("Add", [s, "mask"], f"{p}/Add_mask")
        if kind == "where":
            s = node("Where", ["keep", s, init(f"neg{b}", -1e4)], f"{p}/Where")
        a = node("Softmax", [s], f"{p}/Softmax", axis=-1)
        o = node("MatMul", [a, v], f"{p}/MatMul_1")
        h = node("Add", [o, h], f"{prefix}.{b}/Add")
    y = node(
        "MatMul", [h, init("w_head", rng.normal(size=(D, 2)))], "/model/head/MatMul"
    )
    nodes.append(helper.make_node("Identity", [y], ["y"], name="/model/head/out"))
    graph = helper.make_graph(
        nodes, "attn", inputs, [helper.make_tensor_value_info("y", F, [1, T, 2])], inits
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _logits_names(m):
    return [
        n.output[0]
        for n in m.graph.node
        if n.name.endswith("/attn/MatMul") or n.name.endswith("/attn/Einsum")
    ]


@pytest.mark.parametrize(
    "kind", ["basic", "scale_before", "div_after", "mask", "where", "einsum"]
)
def test_finds_the_logits_product_by_structure(kind):
    m = _attention_graph(kind)
    products, unmatched = find_attention_logits(m)
    assert products == _logits_names(m) and len(products) == 1
    assert unmatched == []


def test_finds_one_product_per_block_and_reports_unmatched_softmax():
    m = _attention_graph(blocks=2)
    products, unmatched = find_attention_logits(m)
    assert products == _logits_names(m) and len(products) == 2
    # A softmax over a linear layer (constant weight) has no logits product.
    g = m.graph
    g.node.append(helper.make_node("Softmax", ["y"], ["probs"], name="/cls/Softmax"))
    g.output.append(helper.make_tensor_value_info("probs", F, [1, T, 2]))
    products, unmatched = find_attention_logits(m)
    assert len(products) == 2 and unmatched == ["/cls/Softmax"]


def _out_type(m, name):
    types = {
        v.name: v.type.tensor_type.elem_type
        for v in onnx.shape_inference.infer_shapes(m).graph.value_info
    }
    types.update({o.name: o.type.tensor_type.elem_type for o in m.graph.output})
    return types.get(name)


def test_recipe_keeps_only_the_logits_product_fp32():
    m = _attention_graph(blocks=2)
    products, _ = find_attention_logits(m)
    typed, info = cast_onnx_precision(m, "fp16", keep_fp32_outputs=products)
    onnx.checker.check_model(typed, full_check=True)
    by_out = {n.output[0]: n for n in typed.graph.node}
    for t in products:  # kept: same output name, fp32, inputs cast up from fp16
        assert t in by_out and _out_type(typed, t) == F
        producers = {o: n for n in typed.graph.node for o in n.output}
        assert all(producers[i].op_type == "Cast" for i in by_out[t].input)
    assert info["fp32_nodes_in_scope"]["fp32 island"] == 2
    # The value products and everything else stay fp16 (renamed __fp16 outputs).
    av = [n for n in typed.graph.node if n.name.endswith("/attn/MatMul_1")]
    assert av and all(n.output[0].endswith("__fp16") for n in av)
    # A do-nothing recipe reduces the logits products too: this test would fail.
    plain, _ = cast_onnx_precision(m, "fp16")
    assert not any(t in {n.output[0] for n in plain.graph.node} for t in products)


def test_default_rewrite_is_unchanged(tmp_path):
    m = _attention_graph()
    a, _ = cast_onnx_precision(m, "fp16")
    b, _ = cast_onnx_precision(m, "fp16", keep_fp32_outputs=None)
    c, _ = cast_onnx_precision(m, "fp16", keep_fp32_outputs=set())
    assert a.SerializeToString() == b.SerializeToString() == c.SerializeToString()
    onnx.save(m, str(tmp_path / "model.onnx"))
    out, info = _write_typed_graph(tmp_path / "model.onnx", "fp16", None)
    first = out.read_bytes()
    out, info2 = _write_typed_graph(tmp_path / "model.onnx", "fp16", None, None)
    assert out.read_bytes() == first and "fp32_islands" not in info2


def test_refusals():
    trt11, _ = _fake_trt("11.5.0.63")
    trt10, _ = _fake_trt("10.16.1.11")
    _check_fp32_islands("attention_logits", "fp16", "onnx", False, trt11)
    _check_fp32_islands("attention_logits", "fp16", "onnx", True, trt10)
    with pytest.raises(ValueError, match="strongly typed"):
        _check_fp32_islands("attention_logits", "fp16", "onnx", False, trt10)
    with pytest.raises(ValueError, match="strongly typed"):
        _check_fp32_islands("attention_logits", "fp16", "jit", True, trt11)
    for p in ("bf16", "fp32", "tf32"):
        with pytest.raises(ValueError, match="fp16 engines only"):
            _check_fp32_islands("attention_logits", p, "onnx", True, trt11)
    with pytest.raises(ValueError, match="Unknown fp32_islands"):
        _check_fp32_islands("everything", "fp16", "onnx", True, trt11)


def _build(monkeypatch, tmp_path, model, version="11.5.0.63", **kw):
    trt, calls = _fake_trt(version)
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tmp_path.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(tmp_path / "model.onnx"))
    info = {}
    shape = (1, 3, 8, 8)
    _export_tensorrt_onnx(
        None,
        tmp_path / "model.trt",
        shape,
        torch.float32,
        shape,
        shape,
        shape,
        "fp16",
        1 << 20,
        False,
        None,
        _build_info=info,
        **kw,
    )
    return onnx.load_from_string(calls["parsed"]), info


@pytest.mark.parametrize("version", ["11.5.0.63", "10.16.1.11"])
def test_build_records_the_products(monkeypatch, tmp_path, version):
    m = _attention_graph(blocks=2)
    kw = {} if version.startswith("11") else {"strongly_typed": True}
    parsed, info = _build(
        monkeypatch, tmp_path, m, version, fp32_islands="attention_logits", **kw
    )
    rec = info["fp32_islands"]
    assert (rec["recipe"], rec["products"], rec["softmax_unmatched"]) == (
        "attention_logits",
        2,
        0,
    )
    assert rec["first"] == [
        n
        for n in (
            "/model/backbone/blocks.0/attn/MatMul",
            "/model/backbone/blocks.1/attn/MatMul",
        )
    ]
    kept = {n.output[0] for n in parsed.graph.node}
    assert all(t in kept for t in _logits_names(m))


def test_no_attention_warns_and_builds_plain_fp16(monkeypatch, tmp_path, capsys):
    import sleap_nn.export.exporters.tensorrt_exporter as te

    monkeypatch.setattr(te, "_WARNED", set())
    plain, _ = _build(monkeypatch, tmp_path / "a", _graph())
    with_recipe, info = _build(
        monkeypatch, tmp_path / "b", _graph(), fp32_islands="attention_logits"
    )
    assert "found no attention logits" in capsys.readouterr().out
    assert info["fp32_islands"]["products"] == 0
    assert with_recipe.SerializeToString() == plain.SerializeToString()


def test_coordinate_graphs_keep_fp32_outside_the_backbone(monkeypatch, tmp_path):
    m = _attention_graph()
    parsed, info = _build(
        monkeypatch,
        tmp_path / "r",
        m,
        fp32_islands="attention_logits",
        _bf16_fp32_outside=BACKBONE_SCOPE,
    )
    reduced = [
        n.name
        for n in parsed.graph.node
        if n.op_type != "Cast" and n.output[0].endswith("__fp16")
    ]
    assert reduced and all(BACKBONE_SCOPE in n for n in reduced)
    # Without the recipe, fp16 stays whole-graph (unchanged).
    parsed, _ = _build(
        monkeypatch, tmp_path / "p", m, _bf16_fp32_outside=BACKBONE_SCOPE
    )
    reduced = [
        n.name
        for n in parsed.graph.node
        if n.op_type != "Cast" and n.output[0].endswith("__fp16")
    ]
    assert "/model/head/MatMul" in reduced


def test_vit_warning_names_the_remedy_and_is_silent_with_it(capsys):
    from omegaconf import OmegaConf

    from sleap_nn.export.cli import _warn_fp16_vit_pose

    vit = OmegaConf.create(
        {
            "model_config": {
                "backbone_config": {"pretrained": {"model_name": "vit_base_dinov3"}}
            }
        }
    )
    args = (vit, "centered_instance", "pretrained", "tensorrt", "fp16")
    _warn_fp16_vit_pose(*args)
    assert "--fp32-attention-logits" in capsys.readouterr().out
    _warn_fp16_vit_pose(*args, True)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--format", "onnx"], "shapes TensorRT engines"),
        (
            ["--format", "tensorrt", "--precision", "bf16"],
            "applies to --precision fp16",
        ),
        (["--format", "tensorrt"], "--strongly-typed"),
    ],
)
def test_cli_refusals(
    monkeypatch, minimal_instance_centered_instance_ckpt, tmp_path, extra, message
):
    from sleap_nn.export.cli import export

    import importlib.machinery

    trt10 = _fake_trt("10.16.1.11")[0]
    trt10.__spec__ = importlib.machinery.ModuleSpec("tensorrt", None)
    monkeypatch.setitem(sys.modules, "tensorrt", trt10)
    result = CliRunner().invoke(
        export,
        [
            str(minimal_instance_centered_instance_ckpt),
            "-o",
            str(tmp_path / "x"),
            "--fp32-attention-logits",
            *extra,
        ],
    )
    assert result.exit_code != 0 and message in result.output, result.output


def test_family_fallback_wins_over_the_recipe(monkeypatch):
    """A family built fp32 by #867's fallback records the recipe as not applied."""
    import sleap_nn.export.exporters.tensorrt_exporter as te

    class Fails(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

    seen = {}

    def fake_onnx_build(*a, fp32_islands=None, **kw):
        seen.update(precision=a[7], fp32_islands=fp32_islands)
        return a[1]

    monkeypatch.setitem(sys.modules, "tensorrt", _fake_trt("11.5.0.63")[0])
    monkeypatch.setattr(te, "_reduced_precision_fails", lambda: (Fails,))
    monkeypatch.setattr(te, "_export_tensorrt_onnx", fake_onnx_build)
    monkeypatch.setattr(te, "_WARNED", set())
    info = {}
    te.export_to_tensorrt(
        Fails(),
        "model.trt",
        input_shape=(1, 3, 8, 8),
        precision="fp16",
        verbose=False,
        _build_info=info,
        fp32_islands="attention_logits",
    )
    assert seen == {"precision": "fp32", "fp32_islands": None}
    assert info["fp32_islands"] == {"recipe": "attention_logits", "applied": False}
    assert info["precision_requested"] == "fp16"
