"""TensorRT export utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
from loguru import logger
from torch import nn

from sleap_nn.export.exporters.onnx_exporter import export_to_onnx, has_rope


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
        precision: Model precision - "fp32" or "fp16".
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

    Returns:
        Path to the exported TensorRT engine.

    Note:
        TensorRT models are NOT cross-platform. The exported model will only
        work on the same GPU architecture and TensorRT version used for export.
    """
    import tensorrt as trt

    model.eval()
    device = next(model.parameters()).device

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
) -> Path:
    """Export via ONNX, then compile to TensorRT engine."""
    import tensorrt as trt

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
    logger = trt.Logger(trt.Logger.WARNING if not verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = []
            for i in range(parser.num_errors):
                errors.append(str(parser.get_error(i)))
            raise RuntimeError(f"ONNX parsing failed: {errors}")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            if verbose:
                print("  Enabled FP16 mode")
        else:
            if verbose:
                print("  WARNING: Platform does not have fast FP16, using FP32")
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
