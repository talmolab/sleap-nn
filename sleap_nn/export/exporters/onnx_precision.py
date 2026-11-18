"""Reduced precision carried by an ONNX graph, for strongly typed TensorRT builds.

TensorRT 11 builds every network strongly typed: each tensor runs in the type the
graph gives it, and the builder flags that let TensorRT 10 pick fp16 / bf16 kernels
by itself (weak typing) are gone. :func:`cast_onnx_precision` rewrites an fp32 graph
so that the chosen nodes compute in fp16 or bf16: their float initializers are
converted, ``Cast`` nodes are inserted where a tensor crosses between fp32 and the
reduced type, and the graph's inputs and outputs keep their fp32 types.

Which nodes run in the reduced type follows the rules sleap-nn's TensorRT 10 builds
use, so the same ``--precision`` means the same thing on both versions:

- ``scope``: only nodes inside it are converted: the node's name (TorchScript export,
  ``/model/backbone/...``) or its module path (the dynamo exporter's
  ``pkg.torch.onnx.name_scopes``) contains it, e.g. ``"backbone/"``; ``None``
  converts every eligible node.
- Under fp16, nodes fed by a constant beyond fp16's range (|x| > 65504, e.g. the 1e9
  sentinels that mask absent peaks), and the float values computed from it downstream,
  stay fp32 (ModelOpt AutoCast keeps these from initializer and measured magnitudes).
- The normalization ops, reductions, ``Pow``, ``Sqrt``, ``Reciprocal``, ``Exp`` and
  ``Log`` stay in fp32 (:data:`KEEP_FP32_OPS`; fp16 overflows there first). A norm
  decomposed into elementwise ops keeps only those ops in fp32: its ``Sub`` / ``Add``
  / ``Div`` / ``Mul`` follow the scope (current exports use ``LayerNormalization``).
- Transposed convolutions stay in fp32 under bf16 (no bf16 kernel), and so does the
  node feeding one: TensorRT 11 fuses a bf16 -> fp32 -> bf16 cast pair around an
  fp32 deconvolution into a bf16 deconvolution, which it cannot build.
- Nodes whose output type is fixed by an attribute or that carry subgraphs
  (``Cast``, ``Constant``, ``ConstantOfShape``, ``If``, ``Loop`` ...) are never
  converted; neither are integer / boolean tensors or the float inputs that the
  ONNX operator definition fixes to float32 (``Resize`` roi / scales).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np

#: Ops computed in fp32 even inside the reduced-precision scope (overflow / accuracy).
KEEP_FP32_OPS = frozenset(
    {
        "LayerNormalization",
        "GroupNormalization",
        "InstanceNormalization",
        "BatchNormalization",
        "LpNormalization",
        "RMSNormalization",
        "ReduceMean",
        "ReduceSum",
        "ReduceSumSquare",
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceProd",
        "Pow",
        "Sqrt",
        "Reciprocal",
        "Exp",
        "Log",
        "CumSum",
    }
)

#: Ops kept in fp32 under bf16 only (no bf16 kernel in TensorRT 10).
KEEP_FP32_OPS_BF16 = frozenset({"ConvTranspose"})

#: Ops never converted: output type fixed by an attribute, subgraphs, quantization,
#: random / index generators.
NEVER_CONVERT = frozenset(
    {
        "Cast",
        "CastLike",
        "Constant",
        "ConstantOfShape",
        "EyeLike",
        "Range",
        "If",
        "Loop",
        "Scan",
        "QuantizeLinear",
        "DequantizeLinear",
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
        "Multinomial",
        "Shape",
        "Size",
        "NonZero",
    }
)

#: Largest finite fp16 magnitude: a node fed by a constant beyond it stays fp32 under fp16
#: (e.g. the 1e9 sentinels that mask absent peaks); ModelOpt AutoCast's ``init_max``.
FP16_MAX = 65504.0

#: Float inputs whose type the operator definition fixes (never cast).
FIXED_FLOAT_INPUTS = {"Resize": (1, 2), "Upsample": (1,)}


def _bf16_bits(a: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit patterns (round to nearest even), as uint16."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    nan = np.isnan(a)
    out = rounded.astype(np.uint16)
    if nan.any():
        out[nan] = 0x7FC0
    return out


def _convert_initializer(init, low: int) -> None:
    """Convert an fp32 TensorProto in place to fp16 / bf16."""
    from onnx import TensorProto, numpy_helper

    a = numpy_helper.to_array(init).astype(np.float32)
    if low == TensorProto.FLOAT16:
        raw = a.astype(np.float16).tobytes()
    else:
        raw = _bf16_bits(a).tobytes()
    dims = list(init.dims)
    name = init.name
    init.Clear()
    init.name = name
    init.dims.extend(dims)
    init.data_type = low
    init.raw_data = raw


def _tensor_types(model) -> Dict[str, int]:
    """Element type of every tensor the graph names (after shape inference)."""
    import onnx

    from onnx import shape_inference

    if model.ByteSize() > 1_500_000_000:
        # Types need no weights: infer on a copy whose initializers carry no data
        # (shape inference serializes the model, which protobuf caps at 2 GB).
        light = onnx.ModelProto()
        light.CopyFrom(model)
        for init in light.graph.initializer:
            init.ClearField("raw_data")
            for f in ("float_data", "int32_data", "int64_data", "double_data"):
                init.ClearField(f)
            del init.external_data[:]
            init.data_location = 0
        model_for_types = light
    else:
        model_for_types = model
    try:
        inferred = shape_inference.infer_shapes(model_for_types, strict_mode=False)
    except Exception:  # pragma: no cover - inference is best effort
        inferred = model
    g = inferred.graph
    types: Dict[str, int] = {}
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type:
            types[vi.name] = vi.type.tensor_type.elem_type
    for init in g.initializer:
        types[init.name] = init.data_type
    for node in g.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    types[node.output[0]] = attr.t.data_type
    return types


def _scope_path(node) -> str:
    """Where a node sits: its name, plus its module path when the exporter recorded one.

    TorchScript exports name nodes by module path (``/model/backbone/.../Conv``); the
    dynamo exporter names them ``node_*`` and records the path in the node's
    ``pkg.torch.onnx.name_scopes`` metadata (dotted module names), rendered here as
    ``/model/backbone/...`` so the same scope string matches both.
    """
    import ast

    path = node.name or ""
    for prop in node.metadata_props:
        if prop.key == "pkg.torch.onnx.name_scopes":
            try:
                scopes = ast.literal_eval(prop.value)
            except (ValueError, SyntaxError):
                continue
            if scopes:
                path += " /" + str(scopes[-1]).replace(".", "/") + "/"
    return path


def cast_onnx_precision(
    model,
    precision: str,
    scope: Optional[str] = None,
    keep_fp32_ops: Optional[Iterable[str]] = None,
) -> Tuple[object, dict]:
    """Rewrite an fp32 ONNX graph so chosen nodes compute in fp16 or bf16.

    Args:
        model: An ``onnx.ModelProto`` with an fp32 graph (not modified).
        precision: ``"fp16"`` or ``"bf16"``.
        scope: Convert only nodes whose name contains this string; ``None`` for all.
        keep_fp32_ops: Op types kept in fp32 (default :data:`KEEP_FP32_OPS`, plus
            :data:`KEEP_FP32_OPS_BF16` for bf16).

    Returns:
        ``(new_model, info)``: the rewritten ``ModelProto`` (graph inputs and outputs
        keep their types) and a summary dict: ``reduced_nodes``,
        ``fp32_nodes_in_scope`` (by op type), ``casts_inserted``,
        ``initializers_converted``.

    Raises:
        ValueError: For a precision other than fp16 / bf16, or a graph with nothing
            to convert in ``scope``.
    """
    import onnx
    from onnx import TensorProto, helper

    if precision not in ("fp16", "bf16"):
        raise ValueError(
            f"cast_onnx_precision: precision must be fp16 or bf16, got {precision!r}"
        )
    low = TensorProto.FLOAT16 if precision == "fp16" else TensorProto.BFLOAT16
    FLOAT = TensorProto.FLOAT
    keep = set(KEEP_FP32_OPS if keep_fp32_ops is None else keep_fp32_ops)
    if precision == "bf16" and keep_fp32_ops is None:
        keep |= KEEP_FP32_OPS_BF16

    copy = onnx.ModelProto()
    copy.CopyFrom(model)  # no serialization: works above protobuf's 2 GB limit
    model = copy
    g = model.graph
    types = _tensor_types(model)
    graph_outputs = {o.name for o in g.output}
    inits = {i.name: i for i in g.initializer}

    def float_in_slots(node) -> Set[int]:
        fixed = FIXED_FLOAT_INPUTS.get(node.op_type, ())
        return {
            i
            for i, t in enumerate(node.input)
            if t and i not in fixed and types.get(t) == FLOAT
        }

    # Constants beyond fp16's range (fp16 only): their consumers stay fp32.
    big: Set[str] = set()
    if precision == "fp16":
        from onnx import numpy_helper

        def too_big(t) -> bool:
            if t.data_type != FLOAT:
                return False
            a = numpy_helper.to_array(t)
            a = a[np.isfinite(a)]  # inf / NaN exist in fp16 (NaN fills absent peaks)
            return bool(a.size) and bool(np.abs(a).max() > FP16_MAX)

        big = {name for name, init in inits.items() if too_big(init)}
        for node in g.node:
            if node.op_type in ("Constant", "ConstantOfShape"):
                for attr in node.attribute:
                    if attr.name == "value" and attr.HasField("t") and too_big(attr.t):
                        big.add(node.output[0])
        # Values built from such a constant can stay that large downstream (e.g. a
        # mask times 1e9, then a max over it): follow them through the float ops.
        if big:
            for node in g.node:
                if any(t in big for t in node.input):
                    big.update(t for t in node.output if t and types.get(t) == FLOAT)

    # 1. Which nodes run in the reduced type.
    reduced: Set[int] = set()
    kept_in_scope: Dict[str, int] = defaultdict(int)
    for k, node in enumerate(g.node):
        if scope is not None and scope not in _scope_path(node):
            continue
        if node.op_type in NEVER_CONVERT or any(
            a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for a in node.attribute
        ):
            continue
        out_types = [types.get(t) for t in node.output if t]
        if not out_types or any(t is None for t in out_types):
            continue  # unknown output type: leave it alone
        if FLOAT not in out_types:
            continue
        if any(t and t not in types for t in node.input):
            continue  # an input of unknown type could not be cast consistently
        if node.op_type in keep:
            kept_in_scope[node.op_type] += 1
            continue
        if big and any(t in big for t in node.input):
            kept_in_scope["constant beyond fp16 range"] += 1
            continue
        reduced.add(k)
    if precision == "bf16" and "ConvTranspose" in keep:
        # The node feeding a kept (fp32) transposed convolution stays fp32 as well, so
        # no bf16 -> fp32 cast sits directly on the deconvolution's input.
        producer = {t: k for k, node in enumerate(g.node) for t in node.output if t}
        for node in g.node:
            if node.op_type != "ConvTranspose" or not node.input:
                continue
            k = producer.get(node.input[0])
            if k is not None and k in reduced:
                reduced.discard(k)
                kept_in_scope["input of ConvTranspose"] += 1
    if not reduced:
        raise ValueError(
            f"No node to convert to {precision}"
            + (f" in scope {scope!r}" if scope is not None else "")
            + ": check the scope against the graph's node names."
        )

    # Consumers of each tensor: (node index, input slot).
    consumers: Dict[str, list] = defaultdict(list)
    for k, node in enumerate(g.node):
        for i, t in enumerate(node.input):
            if t:
                consumers[t].append((k, i))

    def reads_low(k: int, i: int) -> bool:
        return k in reduced and i in float_in_slots(g.node[k])

    # 2. Initializers whose every use is a reduced-type slot are converted in place.
    converted_inits = 0
    low_name: Dict[str, str] = {}  # fp32 tensor -> its reduced-type twin
    for name, init in inits.items():
        if init.data_type != FLOAT or not consumers.get(name) or name in graph_outputs:
            continue
        if all(reads_low(k, i) for k, i in consumers[name]):
            _convert_initializer(init, low)
            types[name] = low
            low_name[name] = name
            converted_inits += 1

    # 3. Rebuild the node list in order, inserting casts next to their use.
    casts = 0
    suffix = "__" + precision
    new_nodes = []
    for k, node in enumerate(g.node):
        if k in reduced:
            slots = float_in_slots(node)
            for i in sorted(slots):
                t = node.input[i]
                if t not in low_name:
                    low_name[t] = t + suffix
                    new_nodes.append(
                        helper.make_node(
                            "Cast", [t], [t + suffix], to=low, name=f"{t}{suffix}/Cast"
                        )
                    )
                    casts += 1
                node.input[i] = low_name[t]
            for j, t in enumerate(node.output):
                if not t or types.get(t) != FLOAT:
                    continue
                lo = t + suffix
                node.output[j] = lo
                low_name[t] = lo
                types[lo] = low
            new_nodes.append(node)
            for lo in node.output:
                if not lo.endswith(suffix) or types.get(lo) != low:
                    continue
                t = lo[: -len(suffix)]
                needs_fp32 = t in graph_outputs or any(
                    not reads_low(kk, ii) for kk, ii in consumers.get(t, [])
                )
                if needs_fp32:
                    new_nodes.append(
                        helper.make_node(
                            "Cast", [lo], [t], to=FLOAT, name=f"{t}/Cast_fp32"
                        )
                    )
                    casts += 1
        else:
            new_nodes.append(node)
    del g.node[:]
    g.node.extend(new_nodes)
    del g.value_info[:]  # stale types; TensorRT re-derives them from the graph
    if model.ByteSize() < 2_000_000_000:
        onnx.checker.check_model(model)  # (larger graphs: checked from their file)
    info = dict(
        precision=precision,
        scope=scope,
        reduced_nodes=len(reduced),
        fp32_nodes_in_scope=dict(sorted(kept_in_scope.items())),
        casts_inserted=casts,
        initializers_converted=converted_inits,
    )
    return model, info
