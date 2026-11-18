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
    # Conv, Relu and Resize run in bf16; LayerNorm and (bf16) ConvTranspose stay fp32;
    # the head (outside the scope) stays fp32.
    assert by_name["/model/backbone/conv/Conv"].output[0].endswith("__bf16")
    assert by_name["/model/backbone/act/Relu"].output[0].endswith("__bf16")
    assert by_name["/model/backbone/up/Resize"].output[0].endswith("__bf16")
    for name in (
        "/model/backbone/norm/LayerNormalization",
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
    assert info["fp32_nodes_in_scope"] == {"ConvTranspose": 1, "LayerNormalization": 1}
    assert info["reduced_nodes"] == 3 and info["initializers_converted"] == 1
    # images -> bf16 for the Conv; Conv out -> fp32 for LayerNorm; LayerNorm out ->
    # bf16 for Relu; Resize out -> fp32 for ConvTranspose.
    assert info["casts_inserted"] == 4
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


def test_original_model_untouched():
    g = _graph()
    before = g.SerializeToString()
    cast_onnx_precision(g, "bf16", scope=BACKBONE_SCOPE)
    assert g.SerializeToString() == before


# --------------------------------------------------------------------------------------
# TensorRT version branches on stubs
# --------------------------------------------------------------------------------------


def _fake_trt(version):
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

    class Logger:
        WARNING, INFO = "W", "I"

        def __init__(self, *a):
            pass

        def warning(self, *a):
            pass

    class Inp:
        name, shape = "images", (-1, 3, 8, 8)

    class Network:
        num_layers = 0

        def get_input(self, i):
            return Inp()

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
        Builder.platform_has_fast_fp16 = True

    class OnnxParser:
        num_errors = 0

        def __init__(self, network, logger):
            pass

        def parse(self, data, path=None):
            calls["parsed"] = data
            return True

    trt.Logger, trt.Builder, trt.OnnxParser = Logger, Builder, OnnxParser
    return trt, calls


def _build(monkeypatch, tmp_path, version, precision, **kw):
    trt, calls = _fake_trt(version)
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
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
        build_info=info,
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
        bf16_fp32_outside=BACKBONE_SCOPE,
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
    assert "(built with TensorRT 10.16.1.11)" in msg
