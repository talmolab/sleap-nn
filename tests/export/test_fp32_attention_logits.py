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
    _find_attention_logits,
    cast_onnx_precision,
)
from sleap_nn.export.exporters.tensorrt_exporter import (  # noqa: E402
    BACKBONE_SCOPE,
    _check_fp32_islands,
    _islands_record,
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


def find_attention_logits(m):
    f = _find_attention_logits(m)
    return f["products"], f["unmatched"]


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
    # TensorRT 10 without --strongly-typed: accepted (the build turns it on).
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
    assert info["fp32_islands"]["products"] == 1
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


def test_vit_warning_names_the_remedy_and_waits_for_the_build(capsys):
    from omegaconf import OmegaConf

    from sleap_nn.export.cli import _emit_deferred_warnings, _warn_fp16_vit_pose

    vit = OmegaConf.create(
        {
            "model_config": {
                "backbone_config": {"pretrained": {"model_name": "vit_base_dinov3"}}
            }
        }
    )
    _warn_fp16_vit_pose(vit, "centered_instance", "pretrained", "tensorrt", "fp16")
    out = capsys.readouterr().out
    assert "TensorRT >= 10.13" in out and "other ViTs not tested" in out
    assert "--no-fp32-attention-logits" not in out  # a flag the user may not pass
    shown = []
    held = [lambda: shown.append(1)]
    applied = _islands_record(
        "attention_logits", True, products=12, softmax_unmatched=0
    )
    _emit_deferred_warnings(held, {"fp32_islands": applied})
    assert shown == []  # products kept, none unmatched: dropped
    for rec in (
        _islands_record("attention_logits", False, products=0, softmax_unmatched=0),
        _islands_record("attention_logits", True, products=11, softmax_unmatched=1),
        _islands_record("attention_logits", True, products=12, softmax_in_subgraphs=1),
        _islands_record("attention_logits", False, reason="family built fp32"),
    ):
        _emit_deferred_warnings(held, {"fp32_islands": rec})
    _emit_deferred_warnings(held, {})  # also what a failed build passes
    assert shown == [1, 1, 1, 1, 1]


def test_a_failed_build_prints_the_held_back_warnings(
    monkeypatch, minimal_instance_centered_instance_ckpt, tmp_path
):
    """The ViT / bbox_detr warnings held back by the switch are not lost on failure."""
    import importlib.machinery

    import sleap_nn.export.cli as export_cli
    import sleap_nn.export.exporters as exporters

    trt11 = _fake_trt("11.5.0.63")[0]
    trt11.__spec__ = importlib.machinery.ModuleSpec("tensorrt", None)
    monkeypatch.setitem(sys.modules, "tensorrt", trt11)
    held = []
    monkeypatch.setattr(
        export_cli,
        "_warn_fp16_vit_pose",
        lambda *a: held.append("warned"),
    )

    def broken(*a, **kw):
        raise RuntimeError("engine build failed")

    monkeypatch.setattr(exporters, "export_to_tensorrt", broken)
    result = CliRunner().invoke(
        export_cli.export,
        [
            str(minimal_instance_centered_instance_ckpt),
            "-o",
            str(tmp_path / "x"),
            "--format",
            "tensorrt",
            "--fp32-attention-logits",
        ],
    )
    assert result.exit_code != 0
    assert held == ["warned"]


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--format", "onnx"], "shapes TensorRT engines"),
        (
            ["--format", "tensorrt", "--precision", "bf16"],
            "applies to --precision fp16",
        ),
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
    assert info["fp32_islands"] == _islands_record(
        "attention_logits", False, reason="family built fp32"
    )
    assert info["precision_requested"] == "fp16"


def test_unmatched_softmax_outside_the_scope_is_not_reported(
    monkeypatch, tmp_path, capsys
):
    """Outside the reduced scope everything is fp32: no warning for its softmaxes."""
    import sleap_nn.export.exporters.tensorrt_exporter as te

    m = _attention_graph()
    g = m.graph  # a softmax over a linear layer, outside the backbone (a decoder's)
    g.node.append(
        helper.make_node("Softmax", ["y"], ["probs"], name="/model/head/Softmax")
    )
    g.output.append(helper.make_tensor_value_info("probs", F, [1, T, 2]))
    monkeypatch.setattr(te, "_WARNED", set())
    _, info = _build(
        monkeypatch,
        tmp_path / "s",
        m,
        fp32_islands="attention_logits",
        _bf16_fp32_outside=BACKBONE_SCOPE,
    )
    assert info["fp32_islands"]["softmax_unmatched"] == 0
    assert "found no logits product" not in capsys.readouterr().out
    _, info = _build(monkeypatch, tmp_path / "w", m, fp32_islands="attention_logits")
    assert info["fp32_islands"]["softmax_unmatched"] == 1  # whole graph: reported


def _graph_of(nodes, inputs, outputs, inits=()):
    g = helper.make_graph(nodes, "g", inputs, outputs, list(inits))
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])


def _vi(name, shape, t=F):
    return helper.make_tensor_value_info(name, t, shape)


def test_relative_position_terms_do_not_hide_the_logits_product():
    """VitDet-style scores: Q.K^T + a nearer relative-position einsum (computed)."""
    nodes = [
        helper.make_node(
            "Transpose", ["k"], ["kt"], name="/a/Transpose", perm=[0, 2, 1]
        ),
        helper.make_node("MatMul", ["q", "kt"], ["qk"], name="/a/MatMul"),
        helper.make_node("Mul", ["qk", "s"], ["qks"], name="/a/Mul"),
        helper.make_node(
            "Einsum", ["q", "rel"], ["rb"], name="/a/Einsum_1", equation="btd,bsd->bts"
        ),
        helper.make_node("Add", ["rb", "qks"], ["z"], name="/a/Add"),
        helper.make_node("Softmax", ["z"], ["p"], name="/a/Softmax", axis=-1),
    ]
    m = _graph_of(
        nodes,
        [_vi("q", [1, T, D]), _vi("k", [1, T, D]), _vi("rel", [1, T, D])],
        [_vi("p", [1, T, T])],
        [numpy_helper.from_array(np.float32(0.3), "s")],
    )
    f = _find_attention_logits(m)
    # The nearest product is the relative-position einsum; Q.K^T must be kept too.
    assert set(f["products"]) == {"qk", "rb"}
    assert set(f["ambiguous"]["/a/Softmax"]) == {"qk", "rb"}
    assert f["unmatched"] == []


def test_scores_carried_from_the_previous_block():
    """softmax2(prev scores + this block's scores): both products kept, one shared."""
    nodes = [
        helper.make_node("MatMul", ["q1", "k1"], ["l1"], name="/b1/MatMul"),
        helper.make_node("Mul", ["l1", "s"], ["z1"], name="/b1/Mul"),
        helper.make_node("Softmax", ["z1"], ["p1"], name="/b1/Softmax", axis=-1),
        helper.make_node("MatMul", ["q2", "k2"], ["l2"], name="/b2/MatMul"),
        helper.make_node("Mul", ["l2", "s"], ["z2"], name="/b2/Mul"),
        helper.make_node("Add", ["z1", "z2"], ["zz"], name="/b2/Add"),
        helper.make_node("Softmax", ["zz"], ["p2"], name="/b2/Softmax", axis=-1),
    ]
    ins = [_vi(n, [1, T, T]) for n in ("q1", "k1", "q2", "k2")]
    m = _graph_of(
        nodes,
        ins,
        [_vi("p1", [1, T, T]), _vi("p2", [1, T, T])],
        [numpy_helper.from_array(np.float32(0.3), "s")],
    )
    f = _find_attention_logits(m)
    assert f["products"] == ["l1", "l2"]
    assert set(f["by_softmax"]["/b2/Softmax"]) == {"l1", "l2"}
    assert f["shared"] == {"l1": ["/b1/Softmax", "/b2/Softmax"]}


def test_a_transposed_weight_is_a_linear_layer_not_a_product():
    nodes = [
        helper.make_node("Transpose", ["w"], ["wt"], name="/h/Transpose", perm=[1, 0]),
        helper.make_node("MatMul", ["x", "wt"], ["y"], name="/h/MatMul"),
        helper.make_node("Softmax", ["y"], ["p"], name="/h/Softmax", axis=-1),
    ]
    m = _graph_of(
        nodes,
        [_vi("x", [1, T, D])],
        [_vi("p", [1, T, D])],
        [numpy_helper.from_array(np.ones((D, D), np.float32), "w")],
    )
    f = _find_attention_logits(m)
    assert f["products"] == [] and f["unmatched"] == ["/h/Softmax"]


def test_softmax_in_a_control_flow_body_is_counted(monkeypatch, tmp_path, capsys):
    import sleap_nn.export.exporters.tensorrt_exporter as te

    body = helper.make_graph(
        [helper.make_node("Softmax", ["y"], ["pb"], name="/if/Softmax", axis=-1)],
        "then",
        [],
        [_vi("pb", [1, T, 2])],
    )
    m = _attention_graph()
    g = m.graph
    g.node.append(
        helper.make_node(
            "Constant",
            [],
            ["c"],
            name="/c",
            value=numpy_helper.from_array(np.array(True)),
        )
    )
    g.node.append(
        helper.make_node(
            "If", ["c"], ["out_if"], name="/if", then_branch=body, else_branch=body
        )
    )
    g.output.append(_vi("out_if", [1, T, 2]))
    assert _find_attention_logits(m)["in_subgraphs"] == 2
    monkeypatch.setattr(te, "_WARNED", set())
    _, info = _build(monkeypatch, tmp_path / "cf", m, fp32_islands="attention_logits")
    assert info["fp32_islands"]["softmax_in_subgraphs"] == 2
    assert "inside control-flow bodies" in capsys.readouterr().out


def test_no_products_on_a_coordinate_graph_says_backbone_fp16(
    monkeypatch, tmp_path, capsys
):
    import sleap_nn.export.exporters.tensorrt_exporter as te

    monkeypatch.setattr(te, "_WARNED", set())
    # what the scoped rewrite gives with no islands (the recipe adds nothing here)
    from sleap_nn.export.exporters.onnx_precision import cast_onnx_precision as cast

    want, _ = cast(_graph(), "fp16", BACKBONE_SCOPE)
    got, info = _build(
        monkeypatch,
        tmp_path / "b",
        _graph(),
        fp32_islands="attention_logits",
        _bf16_fp32_outside=BACKBONE_SCOPE,
    )
    out = capsys.readouterr().out
    assert "found no attention logits" in out and "fp16 only inside" in out
    assert "plain fp16" not in out
    assert info["fp32_islands"]["products"] == 0
    assert got.SerializeToString() == want.SerializeToString()


def test_one_record_shape_applied_or_not():
    a = _islands_record("attention_logits", True, products=3, softmaxes_in_scope=3)
    b = _islands_record("attention_logits", False, reason="family built fp32")
    assert list(a) == list(b)


def test_cli_without_tensorrt_says_so(
    monkeypatch, minimal_instance_centered_instance_ckpt, tmp_path
):
    from sleap_nn.export.cli import export

    monkeypatch.setitem(sys.modules, "tensorrt", None)  # import raises ImportError
    result = CliRunner().invoke(
        export,
        [
            str(minimal_instance_centered_instance_ckpt),
            "-o",
            str(tmp_path / "x"),
            "--format",
            "tensorrt",
            "--fp32-attention-logits",
        ],
    )
    assert result.exit_code != 0 and "TensorRT is not installed" in result.output


def _plain_vit():
    """A module tree holding a plain-ViT backbone (built without weights)."""
    from sleap_nn.architectures.pretrained import PretrainedBackbone

    class SelfAttention(torch.nn.Module):
        pass

    vit = PretrainedBackbone.__new__(PretrainedBackbone)
    torch.nn.Module.__init__(vit)
    vit.isotropic = True
    vit.block = SelfAttention()
    model = torch.nn.Module()
    model.backbone = vit
    model.w = torch.nn.Parameter(torch.zeros(1))
    return model


def test_has_plain_vit():
    from sleap_nn.architectures.detr.dinov2_windowed import WindowedDinov2Backbone
    from sleap_nn.export.exporters.tensorrt_exporter import has_plain_vit

    assert has_plain_vit(_plain_vit())
    hierarchical = _plain_vit()
    hierarchical.backbone.isotropic = False  # a ConvNeXt / Swin encoder
    assert not has_plain_vit(hierarchical)
    no_attention = _plain_vit()
    del no_attention.backbone.block  # single-stride but no attention (RegNet ...)
    assert not has_plain_vit(no_attention)
    assert not has_plain_vit(torch.nn.Conv2d(1, 1, 1))
    windowed = WindowedDinov2Backbone.__new__(WindowedDinov2Backbone)
    torch.nn.Module.__init__(windowed)
    holder = torch.nn.Module()
    holder.enc = windowed
    assert has_plain_vit(holder)


@pytest.mark.parametrize(
    "version, vit, precision, method, islands, want, want_strong, fast_fp16",
    [
        # The default ("auto"): fp16 of a plain ViT keeps the logits fp32, strongly
        # typed on TensorRT 10 (its builder flags cannot), as built on 11.
        ("10.16.1.11", True, "fp16", "onnx", "auto", "attention_logits", True, True),
        ("11.5.0.63", True, "fp16", "onnx", "auto", "attention_logits", False, True),
        # Unchanged: other models, other precisions, the JIT method, the opt-out.
        ("10.16.1.11", False, "fp16", "onnx", "auto", None, False, True),
        ("10.16.1.11", True, "bf16", "onnx", "auto", None, False, True),
        ("10.16.1.11", True, "fp32", "onnx", "auto", None, False, True),
        ("10.16.1.11", True, "fp16", "jit", "auto", None, False, True),
        ("10.16.1.11", True, "fp16", "onnx", None, None, False, True),
        ("11.5.0.63", True, "fp16", "onnx", None, None, False, True),
        # The explicit recipe builds strongly typed on TensorRT 10.
        (
            "10.16.1.11",
            True,
            "fp16",
            "onnx",
            "attention_logits",
            "attention_logits",
            True,
            True,
        ),
        (
            "10.16.1.11",
            False,
            "fp16",
            "onnx",
            "attention_logits",
            "attention_logits",
            True,
            True,
        ),
        # Kept as before: TensorRT below 10.13 (warned), no fast fp16 (an fp32 build).
        ("10.8.0.43", True, "fp16", "onnx", "auto", None, False, True),
        ("10.16.1.11", True, "fp16", "onnx", "auto", None, False, False),
        # TensorRT 11 has no builder-flag path to fall back to.
        ("11.5.0.63", True, "fp16", "onnx", "auto", "attention_logits", False, False),
    ],
)
def test_vit_fp16_default_keeps_the_attention_logits_fp32(
    monkeypatch,
    capsys,
    version,
    vit,
    precision,
    method,
    islands,
    want,
    want_strong,
    fast_fp16,
):
    import sleap_nn.export.exporters.tensorrt_exporter as te

    seen = {}

    def fake_onnx_build(*a, fp32_islands=None, strongly_typed=False, **kw):
        seen.update(fp32_islands=fp32_islands, strongly_typed=strongly_typed)
        return a[1]

    def fake_jit_build(*a, **kw):
        seen.update(fp32_islands=None, strongly_typed=False)
        return a[1]

    monkeypatch.setitem(
        sys.modules, "tensorrt", _fake_trt(version, fast_fp16=fast_fp16)[0]
    )
    monkeypatch.setattr(te, "_export_tensorrt_onnx", fake_onnx_build)
    monkeypatch.setattr(te, "_export_tensorrt_jit", fake_jit_build)
    monkeypatch.setattr(te, "_WARNED", set())
    model = _plain_vit() if vit else torch.nn.Conv2d(3, 1, 1)
    info = {}
    te.export_to_tensorrt(
        model,
        "model.trt",
        input_shape=(1, 3, 8, 8),
        precision=precision,
        method=method,
        verbose=False,
        _build_info=info,
        fp32_islands=islands,
    )
    assert seen == {"fp32_islands": want, "strongly_typed": want_strong}
    out = capsys.readouterr().out
    mode = info.get("fp32_islands_mode")
    if islands == "attention_logits":
        assert mode == "requested"
        assert ("building strongly typed" in out) == version.startswith("10")
    elif want:
        assert mode.startswith("auto") and "--no-fp32-attention-logits" in out
    elif version.startswith("10.8") and vit and precision == "fp16":
        assert mode.startswith("not applied: TensorRT 10.8")
        assert out.count("may be inaccurate") == 1 and "--precision bf16" in out
    elif not fast_fp16 and vit and precision == "fp16" and method == "onnx" and islands:
        assert (
            mode == "not applied: no fast fp16 (built fp32)" and "plain-ViT" not in out
        )
    else:
        assert mode is None and "plain-ViT" not in out


@pytest.mark.parametrize(
    "extra, want",
    [
        ([], "auto"),
        (["--fp32-attention-logits", "--strongly-typed"], "attention_logits"),
        (["--no-fp32-attention-logits"], None),
    ],
)
def test_cli_passes_the_attention_logits_choice(
    monkeypatch, minimal_instance_centered_instance_ckpt, tmp_path, extra, want
):
    import importlib.machinery

    import sleap_nn.export.exporters as exporters
    from sleap_nn.export.cli import export

    trt10 = _fake_trt("10.16.1.11")[0]
    trt10.__spec__ = importlib.machinery.ModuleSpec("tensorrt", None)
    monkeypatch.setitem(sys.modules, "tensorrt", trt10)
    seen = {}

    def fake_export_to_tensorrt(model, save_path, **kw):
        seen.update(kw)
        open(save_path, "wb").close()
        return save_path

    monkeypatch.setattr(exporters, "export_to_tensorrt", fake_export_to_tensorrt)
    result = CliRunner().invoke(
        export,
        [str(minimal_instance_centered_instance_ckpt), "-o", str(tmp_path / "x")]
        + ["--format", "tensorrt", *extra],
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    assert seen["fp32_islands"] == want
