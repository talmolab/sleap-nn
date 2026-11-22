"""TensorRT export utilities."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
from loguru import logger
from torch import nn

from sleap_nn.export.exporters.onnx_exporter import export_to_onnx, has_rope

#: Layers whose name contains this stay free to run in bf16 when a graph that computes
#: coordinates is built with ``precision="bf16"``; every other float layer (decoders,
#: heads, proposal / box / peak arithmetic) is kept in fp32 (bf16 keeps 8 significant
#: bits: a coordinate near 1,000 px rounds by up to ~4 px).
BACKBONE_SCOPE = "backbone/"


def check_precision_support(
    precision: str,
    trt_version: str,
    compute_capability: Optional[Tuple[int, int]],
) -> None:
    """Refuse a precision this GPU cannot build.

    Args:
        precision: ``"fp32"``, ``"tf32"``, ``"fp16"`` or ``"bf16"``.
        trt_version: ``tensorrt.__version__`` (named in the message).
        compute_capability: The building GPU's ``(major, minor)``, or ``None`` if
            unknown.

    Raises:
        ValueError: An unknown precision.
        RuntimeError: ``bf16`` on a GPU below compute capability 8.0.
    """
    if precision not in ("fp32", "tf32", "fp16", "bf16"):
        raise ValueError(
            f"Unknown TensorRT precision {precision!r}: use fp32, tf32, fp16 or bf16."
        )
    if (
        precision == "bf16"
        and compute_capability is not None
        and tuple(compute_capability) < (8, 0)
    ):
        raise RuntimeError(
            "bf16 engines need a GPU with bf16 support (compute capability 8.0, "
            f"Ampere, or newer); this GPU is {compute_capability[0]}."
            f"{compute_capability[1]} (TensorRT {trt_version}). Export with "
            "--precision fp32, tf32 or fp16."
        )


#: Oldest TensorRT this exporter supports (the `tensorrt` extra's pin).
_MIN_TENSORRT = (10, 13)


def _tensorrt_version(version: str) -> Tuple[int, int]:
    """``"11.5.0.63"`` -> ``(11, 5)``; unparsable parts count as 0."""
    parts = (str(version).split(".") + ["0", "0"])[:2]
    out = []
    for x in parts:
        digits = "".join(ch for ch in x if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return out[0], out[1]


def _uses_strong_typing(trt_module) -> bool:
    """Whether this TensorRT builds strongly typed networks only (11 and newer).

    Decided from the version number; the builder flags are a cross-check (TensorRT 11
    removed FP16 / BF16, 10.x has them).

    Raises:
        RuntimeError: When the flags contradict the version (10.13 and newer). Below
            the supported minimum a warning is printed and the flags decide, as
            before.
    """
    version = str(getattr(trt_module, "__version__", "0"))
    major_minor = _tensorrt_version(version)
    if major_minor < _MIN_TENSORRT:
        # Below the pin: warn, and build on the builder-flag path as before.
        _warn_once(
            f"Warning: TensorRT {version} is older than the supported minimum "
            f"{_MIN_TENSORRT[0]}.{_MIN_TENSORRT[1]} (pip install 'sleap-nn[tensorrt]'); "
            "building with its builder flags as before."
        )
        return not has_weak_typing(trt_module)
    strong = major_minor[0] >= 11
    if strong == has_weak_typing(trt_module):
        raise RuntimeError(
            f"TensorRT {version}: its builder flags "
            f"({'with' if has_weak_typing(trt_module) else 'without'} FP16 / BF16) do "
            "not match its version; reinstall TensorRT."
        )
    return strong


def has_weak_typing(trt_module) -> bool:
    """Whether this TensorRT picks reduced-precision kernels from builder flags.

    TensorRT 10 has the FP16 / BF16 builder flags and per-layer precision (weak
    typing); TensorRT 11 removed them and builds every network strongly typed, with
    each tensor in the type the graph gives it.
    """
    return hasattr(trt_module.BuilderFlag, "FP16") and hasattr(
        trt_module.BuilderFlag, "BF16"
    )


def _pin_static_dims(
    network_dims: Tuple[int, ...],
    min_shape: Tuple[int, ...],
    opt_shape: Tuple[int, ...],
    max_shape: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Clamp a TensorRT profile to a network input's static dimensions.

    Args:
        network_dims: The parsed network input's dims; ``-1`` marks a dynamic one.
        min_shape: Requested minimum input shape.
        opt_shape: Requested optimal input shape.
        max_shape: Requested maximum input shape.

    Returns:
        ``(min_shape, opt_shape, max_shape)`` with every static dimension set to
        its fixed value. A fixed height/width (a RoPE backbone) is logged.

    Raises:
        ValueError: If the batch is static and the requested batch range differs
            from it: the engine would silently accept only that batch size.
    """
    lo, opt, hi = (
        tuple(int(d) if d >= 0 else s for d, s in zip(network_dims, shape))
        for shape in (min_shape, opt_shape, max_shape)
    )
    if (lo[0], hi[0]) != (min_shape[0], max_shape[0]):
        raise ValueError(
            f"The ONNX graph has a static batch size of {lo[0]}, so a TensorRT "
            f"engine for batches {min_shape[0]}..{max_shape[0]} cannot be built. "
            f"Export with a max batch size of {lo[0]}, or from a graph with a "
            "dynamic batch."
        )
    if (lo, opt, hi) != (tuple(min_shape), tuple(opt_shape), tuple(max_shape)):
        logger.warning(
            f"The ONNX input has static dims {tuple(network_dims)}: TensorRT profile "
            f"narrowed from {tuple(min_shape)}/{tuple(opt_shape)}/{tuple(max_shape)} "
            f"to {lo}/{opt}/{hi}; the engine accepts only those sizes."
        )
    return lo, opt, hi


#: Recipes for ``export_to_tensorrt(fp32_islands=...)`` / ``--fp32-attention-logits``.
_FP32_ISLAND_RECIPES = ("attention_logits",)


def _islands_record(recipe: str, applied: bool, reason=None, **counts) -> dict:
    """The ``fp32_islands`` metadata record: one shape whether applied or not."""
    keys = (
        "softmaxes_in_scope",
        "products",
        "softmax_unmatched",
        "softmax_ambiguous",
        "products_shared",
        "softmax_in_subgraphs",
    )
    record = {"recipe": recipe, "applied": applied, "reason": reason}
    record.update({k: counts.get(k) for k in keys})
    for k in ("first", "unmatched_first", "ambiguous_first"):
        record[k] = list(counts.get(k) or [])
    return record


def _check_fp32_islands(recipe, precision, method, strongly_typed, trt_module) -> None:
    """Refuse an ``fp32_islands`` recipe where it cannot apply."""
    if recipe not in _FP32_ISLAND_RECIPES:
        raise ValueError(
            f"Unknown fp32_islands recipe {recipe!r}: use one of {_FP32_ISLAND_RECIPES}."
        )
    if precision != "fp16":
        raise ValueError(
            f"fp32 attention logits apply to fp16 engines only, not {precision}: bf16 "
            "already accumulates in fp32, and fp32 / tf32 keep everything in fp32."
        )
    if method != "onnx" or not (
        bool(strongly_typed) or _uses_strong_typing(trt_module)
    ):
        raise ValueError(
            "fp32 attention logits need the strongly typed build, where the graph "
            "carries each node's precision: TensorRT 11, or --strongly-typed "
            "(strongly_typed=True) on TensorRT 10, with the ONNX method. TensorRT 10's "
            "builder flags cannot keep single products in fp32."
        )


def export_to_tensorrt(
    model: nn.Module,
    save_path: str | Path,
    input_shape: Tuple[int, int, int, int] = (1, 1, 512, 512),
    input_dtype: torch.dtype = torch.uint8,
    precision: str = "fp16",
    min_shape: Optional[Tuple[int, int, int, int]] = None,
    opt_shape: Optional[Tuple[int, int, int, int]] = None,
    max_shape: Optional[Tuple[int, int, int, int]] = None,
    workspace_size: int = 2 << 30,  # 2GB default
    method: str = "onnx",
    verbose: bool = True,
    static_hw: Optional[bool] = None,
    _bf16_fp32_outside: Optional[str] = None,
    _build_info: Optional[dict] = None,
    strongly_typed: bool = False,
    force_precision: bool = False,
    fp32_islands: Optional[str] = None,
) -> Path:
    """Export a PyTorch model to TensorRT format.

    This function supports multiple compilation methods:
    - "onnx": Exports to ONNX first, then compiles with TensorRT (most reliable)
    - "jit": Uses torch.jit.trace + torch_tensorrt.compile (alternative)

    Args:
        model: The PyTorch model to export (typically an ONNX wrapper).
        save_path: Path to save the TensorRT engine (.trt file).
        input_shape: (B, C, H, W) optimal input tensor shape.
        input_dtype: Input tensor dtype (torch.uint8 or torch.float32).
        precision: Model precision - "fp32", "tf32", "fp16" or "bf16" (ONNX
            method; the JIT method takes fp32 / tf32 / fp16).
        min_shape: Minimum input shape for dynamic shapes (default: batch=1, H/W halved).
        opt_shape: Optimal input shape (default: same as input_shape).
        max_shape: Maximum input shape (default: batch=16, H/W doubled).
        workspace_size: TensorRT workspace size in bytes (default 2GB).
        method: Compilation method - "onnx" or "jit".
        verbose: Print export info.
        static_hw: Whether the engine takes only the traced height/width (a RoPE
            backbone's grid is fixed at trace time). ``None`` detects RoPE in
            ``model`` (`has_rope`); pass ``False`` for a wrapper whose RoPE model
            only sees fixed-size crops (a top-down instance model). For the
            ``"onnx"`` method an existing ONNX file's own static dims win.
        _bf16_fp32_outside: With ``precision="bf16"``: keep every float layer whose
            name does not contain this string (e.g. :data:`BACKBONE_SCOPE`) in fp32,
            so bf16 runs only inside that scope. ``None`` keeps only the transposed
            convolutions in fp32 (graphs whose outputs are feature maps, e.g.
            ``--network-only`` confidence maps).
        _build_info: Optional dict filled with what was actually built:
            ``precision_built`` (``"fp32"`` when fp16 fell back on a GPU without
            fast fp16), ``tensorrt_version``, ``typing`` (``"weak"``: builder
            flags; ``"strong"``: the precision carried by Cast nodes in the graph)
            and, for bf16, ``bf16_scope`` and ``bf16_fp32_layers``.
        force_precision: Build fp16 / bf16 even for a model family whose
            reduced-precision engines do not reproduce its fp32 graph (multi-class
            bottom-up). Without it such a family is built in fp32 with a warning, on
            the strongly typed path (TensorRT 11, ``strongly_typed``) and on TensorRT
            10's builder flags alike.
        strongly_typed: TensorRT 10: build a strongly typed engine whose fp16 /
            bf16 is carried by Cast nodes in the graph
            (:func:`~sleap_nn.export.exporters.onnx_precision.cast_onnx_precision`),
            the way TensorRT 11 builds always. Default off: TensorRT 10 builds
            use the builder flags as before. Ignored by the ``"jit"`` method.
        fp32_islands: Opt-in recipe for ``precision="fp16"`` on the strongly typed
            path (TensorRT 11, or ``strongly_typed`` on 10): ``"attention_logits"``
            keeps every attention block's logits product (Q.K^T, found by structure:
            :func:`~sleap_nn.export.exporters.onnx_precision._find_attention_logits`)
            in fp32, so TensorRT accumulates it in fp32; everything else is the
            usual fp16 rewrite. With ``_bf16_fp32_outside`` (graphs with coordinates)
            the fp16 runs only inside that scope, as bf16 does. Recorded as
            ``fp32_islands`` in the build info. ``None`` (default): plain fp16.

    Returns:
        Path to the exported TensorRT engine.

    Note:
        TensorRT models are NOT cross-platform. The exported model will only
        work on the same GPU architecture and TensorRT version used for export.
    """
    import tensorrt as trt

    model.eval()
    device = next(model.parameters()).device
    if str(precision).lower() not in ("fp32", "tf32", "fp16", "bf16"):
        raise ValueError(
            f"Unknown TensorRT precision {precision!r}: use fp32, tf32, fp16 or bf16."
        )
    precision = str(precision).lower()
    info = _build_info if _build_info is not None else {}
    if fp32_islands is not None:
        _check_fp32_islands(fp32_islands, precision, method, strongly_typed, trt)
    if precision in ("fp16", "bf16") and isinstance(model, _reduced_precision_fails()):
        # Built fp32 on every path (strongly typed or TensorRT 10's builder flags):
        # the reduced-precision engine of such a family is wrong, not just less exact.
        what = (
            f"a {type(model).__name__} engine does not reproduce its fp32 graph in "
            f"{precision} (multi-class bottom-up: 40-77% of peaks within 0.5 px)"
        )
        if not force_precision:
            msg = (
                f"Warning: {what}; building fp32 instead (force_precision to override)."
            )
            _warn_once(msg)
            info["precision_requested"] = precision
            info["precision_fallback"] = what + "; built fp32"
            precision = "fp32"
            if fp32_islands is not None:  # nothing left to keep in fp32
                info["fp32_islands"] = _islands_record(
                    fp32_islands, applied=False, reason="family built fp32"
                )
                fp32_islands = None
        else:
            msg = f"Warning: {what}; check the engine against fp32 on your data."
            _warn_once(msg)

    save_path = Path(save_path)
    if not save_path.suffix:
        save_path = save_path.with_suffix(".trt")

    B, C, H, W = input_shape

    if min_shape is None:
        min_shape = (1, C, H // 2, W // 2)
    if opt_shape is None:
        opt_shape = input_shape
    if max_shape is None:
        max_shape = (min(16, B * 4), C, H * 2, W * 2)
    if static_hw is None:
        static_hw = has_rope(model)
    if method == "jit" and static_hw:
        # The traced RoPE grid is only correct at the traced size (the ONNX route
        # gets the same restriction from the static ONNX input instead).
        pinned = tuple((s[0], s[1], H, W) for s in (min_shape, opt_shape, max_shape))
        if pinned != (tuple(min_shape), tuple(opt_shape), tuple(max_shape)):
            logger.warning(
                "The backbone uses rotary position embeddings, which trace to a "
                f"fixed token grid: TensorRT profile height/width pinned to {H}x{W}."
            )
        min_shape, opt_shape, max_shape = pinned

    if verbose:
        print(f"Exporting model to TensorRT...")
        print(f"  Input shape: {input_shape}")
        print(f"  Min/Opt/Max: {min_shape} / {opt_shape} / {max_shape}")
        print(f"  Precision: {precision}")
        print(f"  Workspace: {workspace_size / 1e9:.1f} GB")
        print(f"  Method: {method}")

    if method == "onnx":
        # Build on the model's GPU (--device cuda:N), and judge its precision support
        # there, not on whichever device is current.
        cuda_ctx = (
            torch.cuda.device(device)
            if device.type == "cuda" and torch.cuda.is_available()
            else nullcontext()
        )
        with cuda_ctx:
            return _export_tensorrt_onnx(
                model,
                save_path,
                input_shape,
                input_dtype,
                min_shape,
                opt_shape,
                max_shape,
                precision,
                workspace_size,
                verbose,
                static_hw,
                _bf16_fp32_outside=_bf16_fp32_outside,
                _build_info=info,
                strongly_typed=strongly_typed,
                fp32_islands=fp32_islands,
            )
    elif method == "jit":
        return _export_tensorrt_jit(
            model,
            save_path,
            input_shape,
            input_dtype,
            min_shape,
            opt_shape,
            max_shape,
            precision,
            workspace_size,
            verbose,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'onnx' or 'jit'.")


#: Rewritten graphs above this size keep their weights in a separate .data file.
_EXTERNAL_DATA_BYTES = 1_800_000_000


def _write_typed_graph(
    onnx_path: Path,
    precision: str,
    scope: Optional[str],
    fp32_islands: Optional[str] = None,
) -> Tuple[Path, dict]:
    """Write the fp16 / bf16 version of an exported graph for a strongly typed build.

    Args:
        onnx_path: The export's fp32 ``model.onnx`` (external weights next to it
            are read too).
        precision: ``"fp16"`` or ``"bf16"``.
        scope: Reduce only nodes inside this scope (:data:`BACKBONE_SCOPE`); ``None``
            for the whole graph.
        fp32_islands: ``"attention_logits"``: also keep the attention logits
            products inside the reduced scope in fp32 (``info["fp32_islands"]``, see
            :func:`_islands_record`). ``None``: none.

    Returns:
        ``(path, info)``: ``tensorrt_graph/<stem>.<precision>.onnx`` next to
        ``onnx_path`` (large weights in ``<name>.data`` beside it above 1.8 GB). An
        fp16 graph passes ONNX's full check; a bf16 graph is for TensorRT only (ONNX
        allows bfloat16 in Conv, pooling ... only from opset 22): structure checked,
        recorded in ``info["graph_validity"]``. ``info`` from
        :func:`~sleap_nn.export.exporters.onnx_precision.cast_onnx_precision` plus
        the graph's ``opset`` (that of ``model.onnx``: the rewrite adds no new op
        set).
    """
    import onnx

    from sleap_nn.export.exporters.onnx_precision import (
        _find_attention_logits,
        _scope_path,
        cast_onnx_precision,
    )

    if fp32_islands not in (None, "attention_logits"):
        raise ValueError(f"Unknown fp32_islands recipe {fp32_islands!r}.")
    original = onnx.load(str(onnx_path))
    islands = None
    if fp32_islands == "attention_logits":
        found = _find_attention_logits(original)
        nodes = {n.name or n.output[0]: n for n in original.graph.node if n.output}
        nodes.update({n.output[0]: n for n in original.graph.node if n.output})

        def inside(name):  # outside the scope everything is fp32 anyway
            return scope is None or scope in _scope_path(nodes[name])

        softmaxes = [x for x in found["softmaxes"] if inside(x)]
        products = [t for t in found["products"] if inside(t)]
        unmatched = [x for x in found["unmatched"] if inside(x)]
        ambiguous = {x: c for x, c in found["ambiguous"].items() if inside(x)}
        shared = {t: r for t, r in found["shared"].items() if inside(t)}
        islands = set(products)
        name = lambda t: nodes[t].name or t  # noqa: E731
        record = _islands_record(
            fp32_islands,
            applied=bool(products),
            softmaxes_in_scope=len(softmaxes),
            products=len(products),
            softmax_unmatched=len(unmatched),
            softmax_ambiguous=len(ambiguous),
            products_shared=len(shared),
            softmax_in_subgraphs=found["in_subgraphs"],
            first=[name(t) for t in products[:3]],
            unmatched_first=unmatched[:3],
            ambiguous_first=list(ambiguous)[:3],
        )
        where = (
            f"fp16 only inside '*{scope}*' and fp32 elsewhere"
            if scope is not None
            else f"plain {precision}"
        )
        if not products:
            _warn_once(
                f"Warning: --fp32-attention-logits found no attention logits in "
                f"{onnx_path.name}; the engine is {where}."
            )
        if unmatched:
            _warn_once(
                f"Warning: --fp32-attention-logits found no logits product for "
                f"{len(unmatched)} softmax node(s) ({', '.join(unmatched[:3])}"
                f"{' ...' if len(unmatched) > 3 else ''}); they run in {precision}."
            )
        if ambiguous:
            _warn_once(
                f"Warning: --fp32-attention-logits found more than one candidate "
                f"logits product for {len(ambiguous)} softmax node(s) "
                f"({', '.join(list(ambiguous)[:3])}{' ...' if len(ambiguous) > 3 else ''}"
                f"): all {sum(len(c) for c in ambiguous.values())} kept in fp32."
            )
        if shared:
            _warn_once(
                f"Note: --fp32-attention-logits: {len(shared)} product(s) feed more "
                f"than one softmax ({', '.join(name(t) for t in list(shared)[:3])}); "
                "kept in fp32."
            )
        if found["in_subgraphs"]:
            _warn_once(
                f"Warning: --fp32-attention-logits does not handle the "
                f"{found['in_subgraphs']} softmax node(s) inside control-flow bodies "
                f"(If / Loop); they run in {precision}."
            )
    typed, info = cast_onnx_precision(
        original, precision, scope, keep_fp32_outputs=islands
    )
    if fp32_islands is not None:
        info["fp32_islands"] = record
    out_dir = onnx_path.parent / "tensorrt_graph"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{onnx_path.stem}.{precision}.onnx"
    data = out_dir / (out.name + ".data")
    if data.exists():
        data.unlink()
    big = typed.ByteSize() > _EXTERNAL_DATA_BYTES
    info["opset"] = max(
        (o.version for o in typed.opset_import if o.domain in ("", "ai.onnx")),
        default=None,
    )
    if precision == "fp16" and typed.ByteSize() < 2_000_000_000:
        # The full check (types included), run before saving (saving with external
        # data rewrites the in-memory tensors as references). It aborts the export only
        # when the rewrite is at fault: the fp32 original passes and the rewrite fails.
        try:
            onnx.checker.check_model(original, full_check=True)
            original_ok = True
        except Exception as e:  # noqa: BLE001
            original_ok = False
            _warn_once(
                f"Warning: the exported graph {onnx_path} fails ONNX's full check "
                f"({type(e).__name__}); not caused by the fp16 rewrite, continuing."
            )
        if original_ok:
            onnx.checker.check_model(typed, full_check=True)
            info["graph_validity"] = "valid ONNX (full check)"
        else:
            info["graph_validity"] = "full check not run: the fp32 graph fails it"
    elif precision == "fp16":
        info["graph_validity"] = "structure checked (above 2 GB: types not checked)"
    else:
        # ONNX allows bfloat16 in Conv, pooling, TopK, GridSample ... only from opset
        # 22: at the export's opset the bf16 graph is for TensorRT's parser only.
        info["graph_validity"] = (
            f"for TensorRT only: bfloat16 ops are not valid ONNX at opset "
            f"{info['opset']} (structure checked, types not)"
        )
    onnx.save_model(
        typed,
        str(out),
        save_as_external_data=big,
        all_tensors_to_one_file=True,
        location=data.name,  # small tensors (shapes, scales) stay inline
    )
    onnx.checker.check_model(str(out))  # the file as written (structure)
    return out, info


def _export_tensorrt_onnx(
    model: nn.Module,
    save_path: Path,
    input_shape: Tuple[int, int, int, int],
    input_dtype: torch.dtype,
    min_shape: Tuple[int, int, int, int],
    opt_shape: Tuple[int, int, int, int],
    max_shape: Tuple[int, int, int, int],
    precision: str,
    workspace_size: int,
    verbose: bool,
    static_hw: Optional[bool] = None,
    _bf16_fp32_outside: Optional[str] = None,
    _build_info: Optional[dict] = None,
    strongly_typed: bool = False,
    fp32_islands: Optional[str] = None,
) -> Path:
    """Export via ONNX, then compile to TensorRT engine."""
    import tensorrt as trt

    capability = None
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()  # the current (= model's) GPU
    check_precision_support(precision, trt.__version__, capability)
    # TensorRT 11 has no weak typing: reduced precision must be in the graph.
    strong = bool(strongly_typed) or _uses_strong_typing(trt)
    info = _build_info if _build_info is not None else {}
    info["precision_built"] = precision
    info["tensorrt_version"] = str(trt.__version__)
    info["typing"] = "strong" if strong else "weak"

    # Check if ONNX file already exists (from prior export step)
    onnx_path = save_path.with_suffix(".onnx")

    if onnx_path.exists():
        if verbose:
            print(f"  Using existing ONNX file: {onnx_path}")
    else:
        # Need to export to ONNX first
        if verbose:
            print("  Exporting to ONNX first...")

        # Same exporter as `sleap-nn export --format onnx`: its fallbacks for ops
        # the legacy exporter cannot lower (antialiased resizes) and its static
        # height/width for RoPE backbones apply here too.
        export_to_onnx(
            model,
            onnx_path,
            input_shape=input_shape,
            input_dtype=input_dtype,
            input_names=["images"],
            dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"}},
            verify=False,
            static_hw=static_hw,
        )

        if verbose:
            print(f"  ONNX export complete: {onnx_path}")

    if verbose:
        print(f"  Building TensorRT engine (this may take a while)...")

    # Create TensorRT builder
    trt_logger = trt.Logger(trt.Logger.WARNING if not verbose else trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    # EXPLICIT_BATCH is the only mode (and a no-op flag) since TensorRT 10; 11 dropped it.
    creation = trt.NetworkDefinitionCreationFlag
    flags = 0
    if hasattr(creation, "EXPLICIT_BATCH"):
        flags |= 1 << int(creation.EXPLICIT_BATCH)
    if strong and hasattr(creation, "STRONGLY_TYPED"):
        # Opt-in on TensorRT 10; TensorRT 11 networks are strongly typed anyway.
        flags |= 1 << int(creation.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, trt_logger)

    # Parse ONNX model
    if strong and precision in ("fp16", "bf16"):
        # Strongly typed: the graph carries the precision. Same rules as the
        # weak-typing builds below: bf16 only inside _bf16_fp32_outside when given,
        # transposed convolutions (bf16) and normalizations in fp32. The rewritten
        # graph is saved next to the engine (tensorrt_graph/, never picked up as
        # the export's ONNX model) and checked before TensorRT parses it.
        # The fp16 attention-logits recipe takes the bf16 scope too: on graphs with
        # coordinates, everything outside the backbone stays fp32.
        typed_path, cast_info = _write_typed_graph(
            onnx_path,
            precision,
            (
                _bf16_fp32_outside
                if precision == "bf16" or (precision == "fp16" and fp32_islands)
                else None
            ),
            fp32_islands=fp32_islands if precision == "fp16" else None,
        )
        info["graph_casts"] = cast_info
        if "fp32_islands" in cast_info:
            info["fp32_islands"] = cast_info["fp32_islands"]
        info["tensorrt_graph"] = str(typed_path.relative_to(onnx_path.parent))
        if cast_info.get("unknown_type_nodes"):
            msg = (
                f"Warning: {cast_info['unknown_type_nodes']} graph nodes could not be "
                f"typed by ONNX shape inference; they (and nodes fed only through them) "
                f"stay fp32 in this {precision} engine (recorded in the metadata)."
            )
            _warn_once(msg)
        scope = cast_info["scope"]
        info["precision_by_scope"] = {
            (f"nodes in '*{scope}*'" if scope else "whole graph"): (
                f"{precision}, except {cast_info['fp32_nodes_in_scope'] or 'none'} "
                "in fp32"
            ),
            **({"every other node": "fp32"} if scope else {}),
        }
        if precision == "bf16":
            info["bf16_scope"] = (
                f"bf16 only in nodes named '*{scope}*'; every other float node fp32"
                if scope is not None
                else "bf16; transposed convolutions and normalizations fp32"
            )
        if verbose:
            print(
                f"  Strongly typed {precision}: {cast_info['reduced_nodes']} nodes in "
                f"{precision}, {cast_info['casts_inserted']} casts inserted "
                f"({typed_path})"
            )
        parsed = parser.parse_from_file(str(typed_path))
    else:
        with open(onnx_path, "rb") as f:
            parsed = parser.parse(f.read(), str(onnx_path))
    if not parsed:
        errors = []
        for i in range(parser.num_errors):
            errors.append(str(parser.get_error(i)))
        raise RuntimeError(f"ONNX parsing failed: {errors}")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    if strong and precision in ("fp16", "bf16"):
        pass  # the graph's types decide; no precision flag exists (or may be set)
    elif precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            # fp16 cannot hold a value beyond 65504: the layers that read or compute
            # one (the 1e9 sentinels for absent peaks) stay fp32, as on the strongly
            # typed path, so the outputs keep the value instead of an inf.
            held = _hold_fp32_beyond_fp16_range(network, onnx_path, trt)
            if held:
                config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            info["fp16_fp32_layers"] = held
            if verbose:
                print(
                    f"  Enabled FP16 mode ({held} layers holding values beyond fp16's "
                    "range kept in fp32)"
                )
        else:
            info["precision_built"] = "fp32"
            logger.warning(
                "This GPU has no fast fp16: the engine is built in fp32 (recorded as "
                "fp32 in the export metadata)."
            )
        info["precision_by_scope"] = {
            "whole graph": (
                "fp16 where TensorRT picks it"
                if info["precision_built"] == "fp16"
                else "fp32 (no fast fp16)"
            )
        }
        if info.get("fp16_fp32_layers"):
            info["precision_by_scope"][
                "layers reading values beyond fp16's range"
            ] = "fp32"
    elif precision == "bf16":
        # bf16 kernels where faster, with fp32's exponent range (fp16 can overflow
        # in large-activation transformers). Weak typing: TensorRT 10.x.
        config.set_flag(trt.BuilderFlag.BF16)
        # TensorRT 10 has no bf16 transposed convolution (the confmap decoders'
        # upsampling): those layers stay in fp32 so such models still build. With
        # _bf16_fp32_outside, every float layer outside that scope stays in fp32 too:
        # bf16 keeps 8 significant bits, so coordinate arithmetic in the graph
        # (box decoding, proposals, peak refinement) would round by pixels.
        pinned = free = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            is_deconv = layer.type == trt.LayerType.DECONVOLUTION
            outside = _bf16_fp32_outside is not None and _bf16_fp32_outside not in (
                layer.name or ""
            )
            if layer.type == trt.LayerType.SHAPE or not all(
                layer.get_output(j).dtype == trt.float32
                for j in range(layer.num_outputs)
            ):
                continue  # shape / integer / boolean layers keep their own types
            if not (is_deconv or outside):
                free += 1
                continue
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            pinned += 1
        if pinned:
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        info["bf16_scope"] = (
            f"bf16 only in layers named '*{_bf16_fp32_outside}*'; every other float "
            "layer fp32"
            if _bf16_fp32_outside is not None
            else "bf16; transposed convolutions fp32"
        )
        info["bf16_fp32_layers"] = pinned
        if _bf16_fp32_outside is not None and pinned and not free:
            # No layer name carries the scope (e.g. a graph from the dynamo exporter,
            # whose nodes are named node_*): every layer was held in fp32.
            info["precision_built"] = "fp32"
            info["bf16_scope"] += " (no layer matched the scope: built fp32)"
            logger.warning(
                f"No TensorRT layer name contains {_bf16_fp32_outside!r}: every float "
                "layer was held in fp32, so this bf16 engine computes in fp32 "
                "(recorded as fp32)."
            )
        info["precision_by_scope"] = (
            {
                f"layers in '*{_bf16_fp32_outside}*'": "bf16 where TensorRT picks it",
                "every other layer": "fp32",
            }
            if _bf16_fp32_outside is not None
            else {
                "whole graph": "bf16 where TensorRT picks it",
                "transposed convolutions": "fp32",
            }
        )
        if verbose:
            print(f"  Enabled BF16 mode ({pinned} float layers kept in fp32)")
    elif precision == "tf32":
        # TF32 mode: leave TF32 enabled (TRT default on Ampere+).
        # Uses 10-bit mantissa vs 23-bit for IEEE FP32.
        if verbose:
            print("  Using TF32 mode (TensorRT default)")
    elif precision == "fp32":
        # Explicitly disable TF32 to ensure strict IEEE FP32 arithmetic.
        # TensorRT enables TF32 by default which uses 10-bit mantissa vs
        # 23-bit for IEEE FP32, introducing small numerical differences.
        if hasattr(trt.BuilderFlag, "TF32"):
            config.clear_flag(trt.BuilderFlag.TF32)
        if verbose:
            print("  Using strict FP32 mode (TF32 disabled)")

    if precision in ("fp32", "tf32"):
        info["precision_by_scope"] = {
            "whole graph": "fp32, TF32 off" if precision == "fp32" else "tf32"
        }

    # Set up optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name

    # A static ONNX dimension (a RoPE backbone's height/width, or a batch the
    # torch.export fallback pinned) admits only its own value; a profile range
    # across it fails the engine build.
    min_shape, opt_shape, max_shape = _pin_static_dims(
        tuple(network.get_input(0).shape), min_shape, opt_shape, max_shape
    )
    if verbose:
        print(f"  Profile Min/Opt/Max: {min_shape} / {opt_shape} / {max_shape}")
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    engine_path = save_path.with_suffix(".trt")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    if verbose:
        import os

        print(f"  Exported TensorRT engine to: {engine_path}")
        print(f"  Engine size: {os.path.getsize(engine_path) / 1e6:.2f} MB")

    return engine_path


def _hold_fp32_beyond_fp16_range(network, onnx_path: Path, trt) -> int:
    """Keep the layers that read or compute values beyond fp16's range in fp32.

    TensorRT 10's builder flags (weak typing) may run any layer in fp16, where a
    constant beyond 65504 (e.g. the -1e9 sentinel that masks absent peaks) becomes
    inf, and so do the outputs computed from it. The float tensors holding such
    values are found on the ONNX graph
    (:func:`~sleap_nn.export.exporters.onnx_precision.beyond_fp16_range_tensors`, the
    rule of the strongly typed rewrite), matched to the parsed network by tensor
    name, plus the network's own constant layers beyond that range (the parser turns
    a fill value into a constant and a broadcast), and followed downstream through the
    network; every layer reading or writing one is pinned to fp32 (precision and float
    output types). Shape layers are left alone.

    Args:
        network: The parsed ``tensorrt.INetworkDefinition``.
        onnx_path: The ONNX file it was parsed from.
        trt: The ``tensorrt`` module.

    Returns:
        The number of layers pinned.
    """
    import onnx

    from sleap_nn.export.exporters.onnx_precision import beyond_fp16_range_tensors

    import numpy as np

    model = onnx.load(str(onnx_path))
    held = beyond_fp16_range_tensors(model)
    if not held:
        return 0
    # The parser makes constants of its own (an ONNX fill value becomes a constant
    # layer plus a broadcast): seed those holding such values too.
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type != trt.LayerType.CONSTANT:
            continue
        try:
            layer.__class__ = trt.IConstantLayer  # the Python API's downcast
            w = layer.weights
            a = np.asarray(w.numpy() if hasattr(w, "numpy") else w)
        except Exception:  # noqa: BLE001 -- weights not readable: judged by name
            continue
        if a.dtype.kind == "f" and a.size:
            # Finite values beyond fp16's range; float32's max is the parser's stand-in
            # for "no bound" (a Clip without max), not a value.
            a = np.abs(a[np.isfinite(a)].astype(np.float32))
            a = a[a < np.finfo(np.float32).max]
            if a.size and float(a.max()) > 65504.0:
                held.add(layer.get_output(0).name)
    float_types = {trt.float32, getattr(trt, "float16", None)} - {None}
    pinned = set()
    changed = True
    while changed:  # layers are in parse order; repeat in case one is not
        changed = False
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if i in pinned or layer.type == trt.LayerType.SHAPE:
                continue
            ins = [layer.get_input(j) for j in range(layer.num_inputs)]
            outs = [layer.get_output(j) for j in range(layer.num_outputs)]
            names = {t.name for t in ins + outs if t is not None}
            if not names & held:
                continue
            float_outs = [j for j, t in enumerate(outs) if t.dtype in float_types]
            layer.precision = trt.float32
            for j in float_outs:
                layer.set_output_type(j, trt.float32)
                held.add(outs[j].name)
            pinned.add(i)
            changed = True
    return len(pinned)


def _export_tensorrt_jit(
    model: nn.Module,
    save_path: Path,
    input_shape: Tuple[int, int, int, int],
    input_dtype: torch.dtype,
    min_shape: Tuple[int, int, int, int],
    opt_shape: Tuple[int, int, int, int],
    max_shape: Tuple[int, int, int, int],
    precision: str,
    workspace_size: int,
    verbose: bool,
) -> Path:
    """Export using torch.jit.trace + torch_tensorrt.compile."""
    import torch_tensorrt

    device = next(model.parameters()).device

    if verbose:
        print("  Tracing model with torch.jit...")

    # Create example input (float32 for tracing)
    if input_dtype == torch.uint8:
        example_input = torch.randint(
            0, 255, input_shape, dtype=torch.uint8, device=device
        )
    else:
        example_input = torch.randn(*input_shape, dtype=input_dtype, device=device)

    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    # Map precision to torch dtype
    precision_map = {
        "fp32": torch.float32,
        "tf32": torch.float32,
        "fp16": torch.float16,
    }
    if precision not in precision_map:
        raise ValueError(
            f"Unknown precision: {precision}. Use 'fp32', 'tf32', or 'fp16'"
        )

    enabled_precisions = {precision_map[precision]}
    if precision == "fp16":
        enabled_precisions.add(torch.float32)

    # Create input specs for TensorRT
    trt_inputs = [
        torch_tensorrt.Input(
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
            dtype=torch.float32,  # TRT internally uses float32 input spec
        )
    ]

    if verbose:
        print("  Compiling with TensorRT...")

    # Build compile kwargs
    compile_kwargs = dict(
        inputs=trt_inputs,
        enabled_precisions=enabled_precisions,
        workspace_size=workspace_size,
        truncate_long_and_double=True,
    )

    # Save TF32 flags so we can restore them after compilation
    prev_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32

    # Disable TF32 for strict FP32 mode in the JIT path
    if precision == "fp32":
        compile_kwargs["require_full_compilation"] = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if verbose:
            print("  Using strict FP32 mode (TF32 disabled)")

    # Compile with TensorRT
    try:
        trt_model = torch_tensorrt.compile(traced_model, **compile_kwargs)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_cuda_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    # Save as TorchScript
    ts_path = save_path.with_suffix(".ts")
    torch.jit.save(trt_model, ts_path)

    if verbose:
        import os

        print(f"  Exported TensorRT-accelerated model to: {ts_path}")
        print(f"  Model size: {os.path.getsize(ts_path) / 1e6:.2f} MB")

    return ts_path


def _reduced_precision_fails() -> tuple:
    """Wrapper classes whose fp16 / bf16 engines do not reproduce the fp32 graph."""
    from sleap_nn.export.wrappers.bottomup_multiclass import (
        BottomUpMultiClassONNXWrapper,
    )

    return (BottomUpMultiClassONNXWrapper,)


_WARNED: set = set()


def _warn_once(msg: str) -> None:
    """Print an export warning once per process (stdout, with the exporter's output)."""
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(msg)
