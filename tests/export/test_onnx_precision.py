"""Reduced precision carried by the ONNX graph (TensorRT 11 / strongly typed builds).

The TensorRT version branches run against stub ``tensorrt`` modules shaped like
TensorRT 10.x (builder flags, weak typing) and 11.x (no FP16 / BF16 / precision-
constraint flags, no EXPLICIT_BATCH, every network strongly typed), so they run
without TensorRT or a GPU.
"""

import json
import sys
import types

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from sleap_nn.export.exporters.onnx_precision import (  # noqa: E402
    _bf16_bits,
    cast_onnx_precision,
)
from sleap_nn.export.exporters.tensorrt_exporter import (  # noqa: E402
    BACKBONE_SCOPE,
    _export_tensorrt_onnx,
    has_weak_typing,
)


def _graph():
    """images -> backbone (Conv, LayerNorm, Relu, Resize, ConvTranspose) -> head Add."""
    rng = np.random.default_rng(0)
    inits = [
        numpy_helper.from_array(rng.normal(size=(4, 3, 3, 3)).astype(np.float32), "W"),
        numpy_helper.from_array(np.ones(8, np.float32), "ln_s"),
        numpy_helper.from_array(np.zeros(8, np.float32), "ln_b"),
        numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), "scales"),
        numpy_helper.from_array(rng.normal(size=(4, 2, 2, 2)).astype(np.float32), "Wt"),
        numpy_helper.from_array(np.full((1, 2, 1, 1), 0.5, np.float32), "bias"),
    ]
    nodes = [
        helper.make_node(
            "Conv",
            ["images", "W"],
            ["c"],
            name="/model/backbone/conv/Conv",
            pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            "LayerNormalization",
            ["c", "ln_s", "ln_b"],
            ["n"],
            name="/model/backbone/norm/LayerNormalization",
            axis=-1,
        ),
        helper.make_node("Relu", ["n"], ["r"], name="/model/backbone/act/Relu"),
        helper.make_node(
            "Resize",
            ["r", "", "scales"],
            ["u"],
            name="/model/backbone/up/Resize",
            mode="nearest",
        ),
        helper.make_node(
            "ConvTranspose",
            ["u", "Wt"],
            ["t"],
            name="/model/backbone/deconv/ConvTranspose",
            strides=[2, 2],
        ),
        helper.make_node("Add", ["t", "bias"], ["y"], name="/model/head/Add"),
    ]
    g = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 32, 32])],
        initializer=inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(m)
    return m


def _op_types(m):
    """Node name -> the element type its first output carries after the rewrite."""
    produced = {}
    for n in m.graph.node:
        if n.op_type == "Cast":
            continue
        produced[n.name] = n.output[0]
    return produced


def test_bf16_bits_round_like_torch():
    a = np.array(
        [0.0, 1.0, -2.5, 1 / 3, 65504.0, 1e-30, 3.14159265, -7.77e10, np.inf],
        np.float32,
    )
    want = (
        torch.from_numpy(a).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    )
    assert np.array_equal(_bf16_bits(a), want)
    assert _bf16_bits(np.array([np.nan], np.float32))[0] == 0x7FC0


def test_bf16_backbone_scope():
    m, info = cast_onnx_precision(_graph(), "bf16", scope=BACKBONE_SCOPE)
    onnx.checker.check_model(m)
    by_name = {n.name: n for n in m.graph.node}
    # Conv and Relu run in bf16; LayerNorm, the (bf16) ConvTranspose and the Resize
    # feeding it stay fp32; the head (outside the scope) stays fp32.
    assert by_name["/model/backbone/conv/Conv"].output[0].endswith("__bf16")
    assert by_name["/model/backbone/act/Relu"].output[0].endswith("__bf16")
    for name in (
        "/model/backbone/norm/LayerNormalization",
        "/model/backbone/up/Resize",
        "/model/backbone/deconv/ConvTranspose",
        "/model/head/Add",
    ):
        assert not by_name[name].output[0].endswith("__bf16"), name
    # Resize's scales input is float32 by definition: never cast.
    assert by_name["/model/backbone/up/Resize"].input[2] == "scales"
    inits = {i.name: i.data_type for i in m.graph.initializer}
    assert inits["W"] == TensorProto.BFLOAT16  # used only by a bf16 node
    assert inits["Wt"] == inits["bias"] == inits["scales"] == TensorProto.FLOAT
    assert [o.type.tensor_type.elem_type for o in m.graph.output] == [TensorProto.FLOAT]
    assert m.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert info["fp32_nodes_in_scope"] == {
        "ConvTranspose": 1,
        "LayerNormalization": 1,
        "input of ConvTranspose": 1,
    }
    assert info["reduced_nodes"] == 2 and info["initializers_converted"] == 1
    # images -> bf16 for the Conv; Conv out -> fp32 for LayerNorm; LayerNorm out ->
    # bf16 for Relu; Relu out -> fp32 for the Resize feeding the ConvTranspose.
    assert info["casts_inserted"] == 4
    # No cast sits directly on the transposed convolution's input (TensorRT 11 fuses
    # such a pair into a bf16 deconvolution it cannot build).
    producer = {o: n for n in m.graph.node for o in n.output}
    deconv = by_name["/model/backbone/deconv/ConvTranspose"]
    assert producer[deconv.input[0]].op_type != "Cast"
    for n in m.graph.node:
        if n.op_type == "Cast":
            to = helper.get_attribute_value(n.attribute[0])
            assert to == (
                TensorProto.FLOAT
                if not n.output[0].endswith("__bf16")
                else TensorProto.BFLOAT16
            )


def test_fp16_whole_graph():
    m, info = cast_onnx_precision(_graph(), "fp16")
    onnx.checker.check_model(m)
    by_name = {n.name: n for n in m.graph.node}
    for name in (
        "/model/backbone/conv/Conv",
        "/model/backbone/deconv/ConvTranspose",
        "/model/head/Add",
    ):
        assert (
            by_name[name].output[0].endswith("__fp16")
        ), name  # fp16 has transposed convs
    assert (
        not by_name["/model/backbone/norm/LayerNormalization"]
        .output[0]
        .endswith("__fp16")
    )
    out = m.graph.output[0].name
    cast_out = [n for n in m.graph.node if n.op_type == "Cast" and n.output[0] == out]
    assert len(cast_out) == 1  # the graph output is cast back to fp32
    inits = {i.name: i.data_type for i in m.graph.initializer}
    assert inits["W"] == inits["Wt"] == inits["bias"] == TensorProto.FLOAT16
    assert inits["scales"] == TensorProto.FLOAT
    assert info["fp32_nodes_in_scope"] == {"LayerNormalization": 1}


def test_unknown_scope_and_precision():
    with pytest.raises(ValueError, match="No node to convert"):
        cast_onnx_precision(_graph(), "bf16", scope="encoder/")
    with pytest.raises(ValueError, match="fp16 or bf16"):
        cast_onnx_precision(_graph(), "fp32")


def test_scope_from_module_path_metadata():
    """Dynamo-exported nodes (named node_*) match the scope by their module path."""
    g = _graph()
    for n in g.graph.node:
        path = "model.head" if "/head/" in n.name else "model.backbone.block"
        n.name = "node_" + n.op_type.lower()
        n.metadata_props.add(
            key="pkg.torch.onnx.name_scopes", value=str(["", "model", path])
        )
    m, info = cast_onnx_precision(g, "bf16", scope=BACKBONE_SCOPE)
    reduced = {
        n.name
        for n in m.graph.node
        if n.op_type != "Cast" and n.output[0].endswith("__bf16")
    }
    assert reduced == {"node_conv", "node_relu"}
    assert "node_add" not in reduced  # the head is outside the scope


def test_external_data_graph(tmp_path):
    """A graph whose weights live in a separate file (exports above 2 GB) is rewritten."""
    from sleap_nn.export.exporters.tensorrt_exporter import _write_typed_graph

    d = tmp_path / "exp"
    d.mkdir()
    onnx.save_model(
        _graph(),
        str(d / "model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=0,
    )
    out, info = _write_typed_graph(d / "model.onnx", "fp16", None)
    assert out == d / "tensorrt_graph" / "model.fp16.onnx"
    assert info["opset"] == 17 and info["reduced_nodes"] > 0
    m = onnx.load(str(out))
    onnx.checker.check_model(m)
    inits = {i.name: i.data_type for i in m.graph.initializer}
    assert inits["W"] == TensorProto.FLOAT16
    # The ONNX backend's *.onnx lookup of the export directory never sees it.
    assert sorted(p.name for p in d.glob("*.onnx")) == ["model.onnx"]


def test_fp16_rewrite_numbers():
    """The rewritten graph computes what the fp32 graph does, within fp16 rounding."""
    ort = pytest.importorskip("onnxruntime")
    from onnx import helper as h

    rng = np.random.default_rng(1)
    w = numpy_helper.from_array(rng.normal(size=(4, 3, 3, 3)).astype(np.float32), "W")
    b = numpy_helper.from_array(rng.normal(size=(1, 4, 1, 1)).astype(np.float32), "b")
    g = h.make_graph(
        [
            h.make_node(
                "Conv", ["x", "W"], ["c"], name="/model/backbone/Conv", pads=[1] * 4
            ),
            h.make_node("Relu", ["c"], ["r"], name="/model/backbone/Relu"),
            h.make_node("Add", ["r", "b"], ["y"], name="/model/head/Add"),
        ],
        "g",
        [h.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 16, 16])],
        [h.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4, 16, 16])],
        initializer=[w, b],
    )
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    typed, _ = cast_onnx_precision(m, "fp16")
    # The graph itself is fp16 (a do-nothing rewrite fails here: ONNX Runtime on the
    # CPU may compute fp16 nodes in fp32, so numbers alone cannot show it).
    onnx.checker.check_model(typed, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(typed, strict_mode=True)
    vtypes = {v.name: v.type.tensor_type.elem_type for v in inferred.graph.value_info}
    conv = next(n for n in typed.graph.node if n.op_type == "Conv")
    assert vtypes[conv.output[0]] == TensorProto.FLOAT16
    inits = {i.name: i.data_type for i in typed.graph.initializer}
    assert inits["W"] == inits["b"] == TensorProto.FLOAT16
    x = rng.normal(size=(2, 3, 16, 16)).astype(np.float32)
    try:
        want = ort.InferenceSession(m.SerializeToString()).run(None, {"x": x})[0]
        got = ort.InferenceSession(typed.SerializeToString()).run(None, {"x": x})[0]
    except Exception as e:  # noqa: BLE001 -- a runtime without fp16 CPU kernels
        pytest.skip(f"ONNX Runtime cannot run the fp16 graph here: {e}")
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-2)


def test_original_model_untouched():
    g = _graph()
    before = g.SerializeToString()
    cast_onnx_precision(g, "bf16", scope=BACKBONE_SCOPE)
    assert g.SerializeToString() == before


# --------------------------------------------------------------------------------------
# TensorRT version branches on stubs
# --------------------------------------------------------------------------------------


def _fake_trt(version, layer_names=(), fast_fp16=True):
    """A stub `tensorrt` module: 10.x (weak typing) or 11.x (strong typing only)."""
    v11 = version.startswith("11")
    calls = {"flags": [], "cleared": [], "network_flags": None, "parsed": None}
    trt = types.ModuleType("tensorrt")
    trt.__version__ = version
    flags = dict(TF32="TF32")
    if not v11:
        flags.update(FP16="FP16", BF16="BF16", PREFER_PRECISION_CONSTRAINTS="PREFER")
    trt.BuilderFlag = types.SimpleNamespace(**flags)
    creation = (
        dict(STRONGLY_TYPED=1) if v11 else dict(EXPLICIT_BATCH=0, STRONGLY_TYPED=1)
    )
    trt.NetworkDefinitionCreationFlag = types.SimpleNamespace(**creation)
    trt.MemoryPoolType = types.SimpleNamespace(WORKSPACE="WORKSPACE")
    trt.LayerType = types.SimpleNamespace(SHAPE="SHAPE", DECONVOLUTION="DECONVOLUTION")
    trt.float32 = "float32"

    class Logger:  # like trt.Logger: no .warning (sleap-nn logs through loguru)
        WARNING, INFO = "W", "I"

        def __init__(self, *a):
            pass

    class Inp:
        name, shape = "images", (-1, 3, 8, 8)

    class Out:
        dtype = "float32"

    class Layer:
        type, num_outputs = "CONV", 1

        def __init__(self, name):
            self.name = name

        def get_output(self, j):
            return Out()

        def set_output_type(self, j, t):
            calls.setdefault("pinned", []).append(self.name)

    class Network:
        num_layers = len(layer_names)

        def get_input(self, i):
            return Inp()

        def get_layer(self, i):
            return Layer(layer_names[i])

    class Config:
        def set_memory_pool_limit(self, *a):
            pass

        def set_flag(self, f):
            calls["flags"].append(f)

        def clear_flag(self, f):
            calls["cleared"].append(f)

        def add_optimization_profile(self, p):
            pass

    class Profile:
        def set_shape(self, *a):
            pass

    class Builder:
        def __init__(self, logger):
            pass

        def create_network(self, flags):
            calls["network_flags"] = flags
            return Network()

        def create_builder_config(self):
            return Config()

        def create_optimization_profile(self):
            return Profile()

        def build_serialized_network(self, network, config):
            return b"engine"

    if not v11:
        Builder.platform_has_fast_fp16 = fast_fp16

    class OnnxParser:
        num_errors = 0

        def __init__(self, network, logger):
            pass

        def parse(self, data, path=None):
            calls["parsed"] = data
            return True

        def parse_from_file(self, path):
            calls["parsed_file"] = path
            with open(path, "rb") as f:
                calls["parsed"] = f.read()
            return True

    trt.Logger, trt.Builder, trt.OnnxParser = Logger, Builder, OnnxParser
    return trt, calls


def _build(monkeypatch, tmp_path, version, precision, stub_kw=None, **kw):
    trt, calls = _fake_trt(version, **(stub_kw or {}))
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    # The real GPU's compute capability must not decide a stub test.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    onnx.save(_graph(), str(tmp_path / "model.onnx"))
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
        precision,
        1 << 20,
        False,
        None,
        _build_info=info,
        **kw,
    )
    parsed = onnx.load_from_string(calls["parsed"])
    return calls, info, parsed


def test_has_weak_typing():
    assert has_weak_typing(_fake_trt("10.16.1.11")[0])
    assert not has_weak_typing(_fake_trt("11.5.0.63")[0])


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_tensorrt10_default_uses_builder_flags(monkeypatch, tmp_path, precision):
    calls, info, parsed = _build(monkeypatch, tmp_path, "10.16.1.11", precision)
    assert calls["flags"] == ["FP16" if precision == "fp16" else "BF16"]
    assert calls["network_flags"] == 1  # EXPLICIT_BATCH only
    assert (
        parsed.SerializeToString() == _graph().SerializeToString()
    )  # graph as exported
    assert (info["typing"], info["tensorrt_version"], info["precision_built"]) == (
        "weak",
        "10.16.1.11",
        precision,
    )


@pytest.mark.parametrize("version", ["11.5.0.63", "10.16.1.11"])
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_strong_typing_carries_precision_in_the_graph(
    monkeypatch, tmp_path, version, precision
):
    kw = {} if version.startswith("11") else {"strongly_typed": True}
    calls, info, parsed = _build(
        monkeypatch,
        tmp_path,
        version,
        precision,
        _bf16_fp32_outside=BACKBONE_SCOPE,
        **kw,
    )
    assert calls["flags"] == []  # no precision flag (none exists on 11)
    want_flags = (1 << 1) | (0 if version.startswith("11") else 1 << 0)
    assert calls["network_flags"] == want_flags
    assert any(n.op_type == "Cast" for n in parsed.graph.node)
    assert info["typing"] == "strong" and info["precision_built"] == precision
    low = "__" + precision
    reduced = {
        n.name
        for n in parsed.graph.node
        if n.op_type != "Cast" and n.output[0].endswith(low)
    }
    if precision == "bf16":  # only inside the backbone scope
        assert reduced and all(BACKBONE_SCOPE in n for n in reduced)
        assert info["bf16_scope"].startswith("bf16 only in nodes named")
    else:  # fp16: the whole graph, as the TensorRT 10 flag
        assert "/model/head/Add" in reduced
    assert info["graph_casts"]["reduced_nodes"] == len(reduced)
    # The graph TensorRT parsed is saved in the export folder and recorded.
    assert calls["parsed_file"] == str(
        tmp_path / "tensorrt_graph" / f"model.{precision}.onnx"
    )
    assert info["tensorrt_graph"] == f"tensorrt_graph/model.{precision}.onnx"
    assert info["graph_casts"]["opset"] == 17
    assert info["precision_by_scope"]


def test_no_fast_fp16_builds_and_records_fp32(monkeypatch, tmp_path):
    calls, info, _ = _build(
        monkeypatch, tmp_path, "10.16.1.11", "fp16", stub_kw={"fast_fp16": False}
    )
    assert calls["flags"] == []
    assert info["precision_built"] == "fp32"


def test_weak_bf16_scope_matching_no_layer_records_fp32(monkeypatch, tmp_path):
    """Layers named node_* (dynamo export) carry no scope: the build is fp32, said so."""
    calls, info, _ = _build(
        monkeypatch,
        tmp_path,
        "10.16.1.11",
        "bf16",
        stub_kw={"layer_names": ["node_conv", "node_relu"]},
        _bf16_fp32_outside=BACKBONE_SCOPE,
    )
    assert calls["pinned"] == ["node_conv", "node_relu"]
    assert (
        info["precision_built"] == "fp32" and "no layer matched" in info["bf16_scope"]
    )
    calls, info, _ = _build(
        monkeypatch,
        tmp_path,
        "10.16.1.11",
        "bf16",
        stub_kw={"layer_names": ["/model/backbone/conv", "/model/head/conv"]},
        _bf16_fp32_outside=BACKBONE_SCOPE,
    )
    assert calls["pinned"] == ["/model/head/conv"]
    assert info["precision_built"] == "bf16" and info["bf16_fp32_layers"] == 1


@pytest.mark.parametrize("precision", ["fp32", "tf32"])
def test_tensorrt11_fp32_and_tf32(monkeypatch, tmp_path, precision):
    calls, info, parsed = _build(monkeypatch, tmp_path, "11.5.0.63", precision)
    assert calls["cleared"] == (["TF32"] if precision == "fp32" else [])
    assert calls["flags"] == []
    assert not any(n.op_type == "Cast" for n in parsed.graph.node)
    assert info["typing"] == "strong" and "graph_casts" not in info


def test_backend_names_the_building_version(tmp_path):
    from sleap_nn.inference.layers.backends.tensorrt_backend import (
        _load_failure_message,
    )

    eng = tmp_path / "model.trt"
    msg = _load_failure_message(eng, "11.5.0.63")
    assert "does not name the building version" in msg and "--runtime onnx" in msg
    (tmp_path / "model.trt.metadata.json").write_text(
        json.dumps({"tensorrt_version": "10.16.1.11"})
    )
    msg = _load_failure_message(eng, "11.5.0.63")
    assert "built with TensorRT 10.16.1.11, and this is TensorRT 11.5.0.63" in msg
    msg = _load_failure_message(eng, "10.16.1.11")
    assert "built with the same version, 10.16.1.11" in msg
    assert "not enough free GPU memory" in msg and "GPU model" in msg


def test_fp16_keeps_nodes_fed_by_constants_beyond_its_range():
    """A 1e9 sentinel (masked peaks) would become inf in fp16: its consumer stays fp32."""
    from onnx import helper as h

    nodes = [
        h.make_node("Relu", ["x"], ["r"], name="/model/backbone/Relu"),
        h.make_node(
            "Constant",
            [],
            ["big"],
            value=numpy_helper.from_array(np.array(1e9, np.float32), "big"),
        ),
        h.make_node(
            "ConstantOfShape",
            ["shape"],
            ["neg"],
            value=numpy_helper.from_array(np.array([-1e9], np.float32), "v"),
        ),
        h.make_node("Add", ["r", "big"], ["a"], name="/peaks/Add"),
        h.make_node("Mul", ["a", "neg"], ["y"], name="/peaks/Mul"),
    ]
    g = h.make_graph(
        nodes,
        "g",
        [h.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
        [h.make_tensor_value_info("y", TensorProto.FLOAT, [4])],
        initializer=[numpy_helper.from_array(np.array([4], np.int64), "shape")],
    )
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    typed, info = cast_onnx_precision(m, "fp16")
    by_name = {n.name: n for n in typed.graph.node if n.name}
    assert by_name["/model/backbone/Relu"].output[0].endswith("__fp16")
    assert not by_name["/peaks/Add"].output[0].endswith("__fp16")
    assert not by_name["/peaks/Mul"].output[0].endswith("__fp16")
    assert info["fp32_nodes_in_scope"] == {
        "constant beyond fp16 range": 2
    }  # Add, and Mul downstream
    # bf16 has fp32's range: the same nodes are converted.
    _, info = cast_onnx_precision(m, "bf16")
    assert "constant beyond fp16 range" not in info["fp32_nodes_in_scope"]


def _tiny(nodes, inits, out_shape=(4,), outputs=("y",), inp=("x", (4,))):
    from onnx import helper as h

    g = h.make_graph(
        nodes,
        "g",
        [h.make_tensor_value_info(inp[0], TensorProto.FLOAT, list(inp[1]))],
        [
            h.make_tensor_value_info(o, TensorProto.FLOAT, list(out_shape))
            for o in outputs
        ],
        initializer=inits,
    )
    return h.make_model(g, opset_imports=[h.make_opsetid("", 17)])


def test_control_flow_bodies_read_fp32():
    """Tensors an If body reads from outside keep fp32; such initializers stay fp32."""
    ort = pytest.importorskip("onnxruntime")
    from onnx import helper as h

    w = numpy_helper.from_array(np.full(4, 2.0, np.float32), "w")
    cond = numpy_helper.from_array(np.array(True), "cond")
    then = h.make_graph(
        [h.make_node("Mul", ["r", "w"], ["t_out"], name="then/Mul")],
        "then",
        [],
        [h.make_tensor_value_info("t_out", TensorProto.FLOAT, [4])],
    )
    other = h.make_graph(
        [h.make_node("Identity", ["r"], ["e_out"], name="else/Id")],
        "else",
        [],
        [h.make_tensor_value_info("e_out", TensorProto.FLOAT, [4])],
    )
    m = _tiny(
        [
            h.make_node("Mul", ["x", "w"], ["m"], name="/backbone/Mul"),
            h.make_node("Relu", ["m"], ["r"], name="/backbone/Relu"),
            h.make_node(
                "If", ["cond"], ["y"], name="/if", then_branch=then, else_branch=other
            ),
        ],
        [w, cond],
    )
    typed, info = cast_onnx_precision(m, "fp16")
    onnx.checker.check_model(typed, full_check=True)
    inits = {i.name: i.data_type for i in typed.graph.initializer}
    assert inits["w"] == TensorProto.FLOAT  # also read inside the If body
    produced = {o for n in typed.graph.node for o in n.output}
    assert "r" in produced  # the fp32 copy the body reads
    x = np.arange(4, dtype=np.float32)
    got = ort.InferenceSession(typed.SerializeToString()).run(None, {"x": x})[0]
    np.testing.assert_allclose(got, np.maximum(x * 2, 0) * 2, rtol=1e-3)


def test_initializer_with_reduced_and_fp32_readers_and_output_read_inside():
    ort = pytest.importorskip("onnxruntime")
    from onnx import helper as h

    w = numpy_helper.from_array(np.full(4, 2.0, np.float32), "w")
    m = _tiny(
        [
            h.make_node("Mul", ["x", "w"], ["y"], name="/backbone/Mul"),  # graph output
            h.make_node("Relu", ["y"], ["r"], name="/backbone/Relu"),  # reads it too
            h.make_node("ReduceSum", ["r"], ["s"], name="/backbone/Sum", keepdims=1),
            h.make_node("Pow", ["s", "w"], ["z"], name="/head/Pow"),  # w, fp32 reader
        ],
        [w],
        outputs=("y", "z"),
    )
    typed, info = cast_onnx_precision(m, "fp16")
    onnx.checker.check_model(typed, full_check=True)
    inits = {i.name: i.data_type for i in typed.graph.initializer}
    assert inits["w"] == TensorProto.FLOAT  # read by a reduced and an fp32 node
    assert [o.type.tensor_type.elem_type for o in typed.graph.output] == [
        TensorProto.FLOAT
    ] * 2
    x = np.arange(4, dtype=np.float32)
    y, z = ort.InferenceSession(typed.SerializeToString()).run(None, {"x": x})
    np.testing.assert_allclose(y, x * 2, rtol=1e-3)
    np.testing.assert_allclose(z, np.full(4, (x * 2).sum() ** 2), rtol=1e-3)


def test_untyped_nodes_are_counted():
    from onnx import helper as h

    m = _tiny(
        [
            h.make_node("Relu", ["x"], ["a"], name="/backbone/Relu"),
            h.make_node(
                "Plugin", ["a"], ["b"], name="/backbone/Plugin", domain="custom"
            ),
            h.make_node("Relu", ["b"], ["y"], name="/backbone/Relu2"),
        ],
        [],
    )
    m.opset_import.append(h.make_opsetid("custom", 1))
    _, info = cast_onnx_precision(m, "fp16")
    assert info["unknown_type_nodes"] >= 1
    assert info["fp32_nodes_in_scope"]["type unknown"] == info["unknown_type_nodes"]


@pytest.mark.parametrize("case", ["value_float", "int_cast", "product"])
def test_fp16_range_rule_cases(case):
    """Out-of-range values written or computed in other ways stay fp32 too."""
    from onnx import helper as h

    if case == "value_float":
        nodes = [h.make_node("Constant", [], ["k"], value_float=1e9)]
    elif case == "int_cast":
        nodes = [
            h.make_node("Constant", [], ["i"], value_int=10**9),
            h.make_node("Cast", ["i"], ["k"], to=TensorProto.FLOAT),
        ]
    else:
        nodes = [
            h.make_node("Constant", [], ["a"], value_float=1e3),  # in range;
            h.make_node("Constant", [], ["b"], value_float=1e3),  # product 1e6 is not
            h.make_node("Mul", ["a", "b"], ["k"], name="/c/Mul"),
        ]
    nodes += [
        h.make_node("Relu", ["x"], ["r"], name="/backbone/Relu"),
        h.make_node("Add", ["r", "k"], ["y"], name="/peaks/Add"),
    ]
    typed, info = cast_onnx_precision(_tiny(nodes, []), "fp16")
    by_name = {n.name: n for n in typed.graph.node if n.name}
    assert by_name["/backbone/Relu"].output[0].endswith("__fp16")
    assert not by_name["/peaks/Add"].output[0].endswith("__fp16")
    if case == "product":
        assert not by_name["/c/Mul"].output[0].endswith("__fp16")


def test_large_graph_save_branch(tmp_path, monkeypatch):
    """Above the size limit the rewritten graph keeps its weights in a .data file."""
    import sleap_nn.export.exporters.tensorrt_exporter as te

    monkeypatch.setattr(te, "_EXTERNAL_DATA_BYTES", 0)
    from onnx import helper as h

    d = tmp_path / "exp"
    d.mkdir()
    w = numpy_helper.from_array(np.ones((64, 3, 3, 3), np.float32), "W")  # > 1 kB
    g = h.make_graph(
        [
            h.make_node("Conv", ["x", "W"], ["c"], name="/backbone/Conv", pads=[1] * 4),
            h.make_node("Relu", ["c"], ["y"], name="/backbone/Relu"),
        ],
        "g",
        [h.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [h.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64, 8, 8])],
        initializer=[w],
    )
    onnx.save(
        h.make_model(g, opset_imports=[h.make_opsetid("", 17)]), str(d / "model.onnx")
    )
    out, info = te._write_typed_graph(d / "model.onnx", "fp16", None)
    assert (out.parent / (out.name + ".data")).exists()
    assert info["graph_validity"] == "valid ONNX (full check)"
    onnx.checker.check_model(onnx.load(str(out)), full_check=True)
    out, info = te._write_typed_graph(d / "model.onnx", "bf16", None)
    assert info["graph_validity"].startswith("for TensorRT only")


def test_tensorrt_version_decides_the_path():
    from sleap_nn.export.exporters.tensorrt_exporter import _uses_strong_typing

    assert _uses_strong_typing(_fake_trt("11.5.0.63")[0]) is True
    assert _uses_strong_typing(_fake_trt("10.16.1.11")[0]) is False
    # Below the supported minimum: a warning, and the builder flags as before.
    assert _uses_strong_typing(_fake_trt("10.12.0")[0]) is False
    odd = _fake_trt("11.0.0")[0]
    odd.BuilderFlag.FP16, odd.BuilderFlag.BF16 = "FP16", "BF16"
    with pytest.raises(RuntimeError, match="do not match its version"):
        _uses_strong_typing(odd)


def test_fp16_range_rule_scans_large_initializers():
    """An initializer above the folding cap holding one value beyond fp16 stays fp32."""
    from onnx import helper as h

    big = np.zeros(1_000_001, np.float32)
    big[7] = -1e9
    m = _tiny(
        [
            h.make_node("Relu", ["x"], ["r"], name="/backbone/Relu"),
            h.make_node("Add", ["r", "big"], ["y"], name="/peaks/Add"),
        ],
        [numpy_helper.from_array(big, "big")],
        out_shape=(1_000_001,),
        inp=("x", (1_000_001,)),
    )
    typed, info = cast_onnx_precision(m, "fp16")
    by_name = {n.name: n for n in typed.graph.node if n.name}
    assert not by_name["/peaks/Add"].output[0].endswith("__fp16")
    inits = {i.name: i.data_type for i in typed.graph.initializer}
    assert inits["big"] == TensorProto.FLOAT


def test_full_check_blames_only_the_rewrite(tmp_path, monkeypatch, capsys):
    """An fp32 graph that itself fails ONNX's full check is warned about, not refused."""
    import sleap_nn.export.exporters.tensorrt_exporter as te

    d = tmp_path / "exp"
    d.mkdir()
    m = _graph()
    m.graph.output[0].type.tensor_type.shape.dim[3].dim_value = 31  # inference says 32
    onnx.save(m, str(d / "model.onnx"))
    monkeypatch.setattr(te, "_WARNED", set())
    out, info = te._write_typed_graph(d / "model.onnx", "fp16", None)
    assert info["graph_validity"].startswith("full check not run")
    assert "not caused by the fp16 rewrite" in capsys.readouterr().out
