"""fp16 on TensorRT 10's builder flags: layers holding values beyond fp16's range stay fp32.

The exported peak finders mask absent peaks with a -1e9 sentinel. In an fp16 engine
built with builder flags (weak typing) TensorRT could run those layers in fp16, and
the value outputs (centroid / peak values) became -65504 or inf where the graph holds
-1e9. Runs on a stub network: no TensorRT or GPU needed.
"""

import types

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from sleap_nn.export.exporters.onnx_precision import (  # noqa: E402
    beyond_fp16_range_tensors,
)
from sleap_nn.export.exporters.tensorrt_exporter import (  # noqa: E402
    _hold_fp32_beyond_fp16_range,
)


def _sentinel_graph():
    """vals = TopK(Where(x > 0.2, x, ConstantOfShape(-1e9))), plus a Relu branch."""
    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="/Shape"),
        helper.make_node(
            "ConstantOfShape",
            ["shape"],
            ["fill"],
            name="/ConstantOfShape",
            value=numpy_helper.from_array(np.array([-1e9], np.float32)),
        ),
        helper.make_node("Greater", ["x", "thr"], ["is_peak"], name="/Greater"),
        helper.make_node("Where", ["is_peak", "x", "fill"], ["masked"], name="/Where"),
        helper.make_node(
            "TopK", ["masked", "k"], ["vals", "idx"], name="/TopK", axis=-1
        ),
        helper.make_node("Relu", ["x"], ["relu"], name="/Relu"),
    ]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
        [
            helper.make_tensor_value_info("vals", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("idx", TensorProto.INT64, [1, 2]),
            helper.make_tensor_value_info("relu", TensorProto.FLOAT, [1, 8]),
        ],
        [
            numpy_helper.from_array(np.array(0.2, np.float32), "thr"),
            numpy_helper.from_array(np.array([2], np.int64), "k"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _stub_network(trt):
    """The layers TensorRT's parser makes of that graph (constant + broadcast fill)."""

    class Tensor:
        def __init__(self, name, dtype):
            self.name, self.dtype = name, dtype

    class Layer:
        def __init__(self, name, type, ins, outs, weights=None):
            self.name, self.type, self._weights = name, type, weights
            self._ins, self._outs = ins, outs
            self.num_inputs, self.num_outputs = len(ins), len(outs)
            self.precision, self.output_types = None, {}

        @property
        def weights(self):
            return self._weights

        def get_input(self, j):
            return self._ins[j]

        def get_output(self, j):
            return self._outs[j]

        def set_output_type(self, j, t):
            self.output_types[j] = t

    trt.IConstantLayer = Layer  # the API's downcast is a no-op on the stub
    F, I, B = trt.float32, "int64", "bool"
    t = {
        n: Tensor(n, d)
        for n, d in [
            ("x", F),
            ("shape", I),
            ("w_out", F),
            ("bcast", F),
            ("fill", F),
            ("is_peak", B),
            ("masked", F),
            ("vals", F),
            ("idx", "int32"),
            ("relu", F),
            ("clip_max", F),
            ("thr", F),
        ]
    }
    L = trt.LayerType
    layers = [
        Layer("/Shape", L.SHAPE, [t["x"]], [t["shape"]]),
        Layer("tmp_weight", L.CONSTANT, [], [t["w_out"]], np.array([-1e9], np.float32)),
        Layer("bcast_shuffle", L.SHUFFLE, [t["w_out"]], [t["bcast"]]),
        Layer("bcast_slice", L.SLICE, [t["bcast"], t["shape"]], [t["fill"]]),
        Layer("thr_const", L.CONSTANT, [], [t["thr"]], np.array([0.2], np.float32)),
        Layer("/Greater", L.ELEMENTWISE, [t["x"], t["thr"]], [t["is_peak"]]),
        Layer("/Where", L.SELECT, [t["is_peak"], t["x"], t["fill"]], [t["masked"]]),
        Layer("/TopK", L.TOPK, [t["masked"]], [t["vals"], t["idx"]]),
        # A Clip without max: the parser's float32-max bound is not a value.
        Layer(
            "clip_const",
            L.CONSTANT,
            [],
            [t["clip_max"]],
            np.array([np.finfo(np.float32).max], np.float32),
        ),
        Layer("/Relu", L.ACTIVATION, [t["x"], t["clip_max"]], [t["relu"]]),
    ]
    return (
        types.SimpleNamespace(num_layers=len(layers), get_layer=lambda i: layers[i]),
        layers,
    )


def _stub_trt():
    trt = types.ModuleType("tensorrt")
    trt.float32, trt.float16 = "float32", "float16"
    trt.LayerType = types.SimpleNamespace(
        **{
            k: k
            for k in (
                "SHAPE",
                "CONSTANT",
                "SHUFFLE",
                "SLICE",
                "ELEMENTWISE",
                "SELECT",
                "TOPK",
                "ACTIVATION",
            )
        }
    )
    return trt


def test_onnx_rule_finds_the_sentinel_path():
    held = beyond_fp16_range_tensors(_sentinel_graph())
    assert held == {"fill", "masked", "vals"}


def test_weak_typing_keeps_the_sentinel_layers_in_fp32(tmp_path):
    path = tmp_path / "model.onnx"
    onnx.save(_sentinel_graph(), str(path))
    trt = _stub_trt()
    network, layers = _stub_network(trt)
    n = _hold_fp32_beyond_fp16_range(network, path, trt)
    pinned = [layer.name for layer in layers if layer.precision == trt.float32]
    assert pinned == ["tmp_weight", "bcast_shuffle", "bcast_slice", "/Where", "/TopK"]
    assert n == len(pinned)
    topk = layers[7]
    assert topk.output_types == {0: trt.float32}  # the values, not the indices


def test_weak_typing_without_large_values_pins_nothing(tmp_path):
    graph = _sentinel_graph()
    for node in graph.graph.node:
        if node.op_type == "ConstantOfShape":
            node.attribute[0].t.CopyFrom(
                numpy_helper.from_array(np.array([-1.0], np.float32))
            )
    path = tmp_path / "model.onnx"
    onnx.save(graph, str(path))
    trt = _stub_trt()
    network, layers = _stub_network(trt)
    layers[1]._weights = np.array([-1.0], np.float32)
    assert _hold_fp32_beyond_fp16_range(network, path, trt) == 0
    assert all(layer.precision is None for layer in layers)
