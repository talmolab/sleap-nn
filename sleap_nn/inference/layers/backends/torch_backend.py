"""``TorchBackend`` — wrap any ``nn.Module`` (or Lightning module) for inference.

Concrete ``ModelBackend`` for PyTorch. Adds opt-in optimizations
(``torch.compile``, FP16 half-precision casting, Conv+BN fusion, GPU
warmup) gated behind benchmark-informed defaults.

FP16 here uses ``torch.autocast`` (CUDA only): when ``use_fp16=True`` the
heavy conv / matmul ops run in half precision under an autocast context
while fp32 master weights and fp32-sensitive reductions are preserved;
outputs are cast back to float32 for downstream postprocessing. Autocast is
robust to model ``forward`` methods that change dtype internally (e.g.
``image / 255.0`` normalization) — a destructive ``model.half()`` would
instead raise an Input/weight dtype mismatch on those.

Defaults (from `12-design-review-and-revised-plan.md` §2 + CUDA validation):

- ``warmup_iterations = 1`` — default ON; 73× cold-start ratio on MPS
- ``fuse_layers      = False`` — opt-in; ≈0% on the test UNets
- ``use_compile      = False`` — opt-in
- ``use_fp16         = False`` — opt-in; tensor-core only, regresses at
  small batch on CUDA (0.65× FP32 at batch=1, 1.5× at batch=16)
- ``memory_format    = "contiguous"`` — opt-in ``"channels_last"``; 2.2× on
  a top-down UNet pair on an NVIDIA A100, no change on AMD ROCm

Warnings emitted on construction (verified by tests):

- ``UserWarning`` on CUDA with ``use_compile=True`` — numeric drift
- ``UserWarning`` on CUDA with ``use_fp16=True`` — drift + small-batch perf
- ``UserWarning`` on MPS with ``use_compile=True`` — disables and downgrades
- ``UserWarning`` on MPS with ``use_fp16=True`` — no tensor cores, no win
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import attrs
import torch
import torch.nn as nn

#: What a backend's ``autocast_dtype`` covers (``TorchBackend.autocast_scope``).
AUTOCAST_SCOPES = ("all", "encoder")


def supports_encoder_autocast(model: nn.Module) -> bool:
    """Whether ``model`` holds sleap-nn models whose encoders take encoder autocast.

    True when ``model`` holds at least one
    :class:`~sleap_nn.architectures.model.Model` and each is a ``bbox_detr``
    detector (its whole backbone is the encoder) or has a backbone that runs its
    encoder through :func:`~sleap_nn.architectures.common.run_encoder`
    (``supports_encoder_autocast``: UNet, including its stem; ConvNeXt; Swin;
    ``pretrained`` timm and HuggingFace backbones).
    """
    from sleap_nn.architectures.model import Model

    nets = [m for m in model.modules() if isinstance(m, Model)]
    return bool(nets) and all(
        net.model_type == "bbox_detr"
        or bool(getattr(net.backbone, "supports_encoder_autocast", False))
        for net in nets
    )


def autocast_forward(
    model: nn.Module,
    x: torch.Tensor,
    dtype: torch.dtype,
    scope: str = "all",
    device_type: str = "cuda",
) -> Any:
    """``model(x)`` under ``torch.autocast(device_type, dtype)``, whole or encoder only.

    Args:
        model: The module to run.
        x: Its input.
        dtype: The autocast dtype (``torch.bfloat16`` or ``torch.float16``).
        scope: ``"all"``: the whole forward runs under autocast. ``"encoder"``
            (experimental): only the backbones' encoders do
            (:func:`~sleap_nn.architectures.common.encoder_autocast`); their
            outputs are cast to fp32, and every layer after them (decoder, heads,
            a DETR detector's projector, transformer decoder and box head) runs in
            fp32. The request is process-wide while the call runs (see
            ``encoder_autocast``).
        device_type: Autocast device type (``TorchBackend`` passes ``"cuda"``;
            tests use ``"cpu"``, where bf16 autocast also works).

    Returns:
        The model's output, as autocast left it (``TorchBackend`` casts half
        floats back to fp32).

    Raises:
        ValueError: If ``scope`` is ``"encoder"`` and ``model`` does not support it.
        RuntimeError: If no encoder ran under the encoder scope.
    """
    from sleap_nn.architectures.common import encoder_autocast

    if scope == "all":
        with torch.autocast(device_type=device_type, dtype=dtype):
            return model(x)
    if scope != "encoder":
        raise ValueError(f"scope must be one of {AUTOCAST_SCOPES}; got {scope!r}.")
    if not supports_encoder_autocast(model):
        raise ValueError(
            "Encoder-only autocast needs sleap-nn models whose backbone supports it; "
            f"{type(model).__name__} does not."
        )
    with encoder_autocast(device_type, dtype) as state:
        out = model(x)
    if not state["calls"]:
        raise RuntimeError(
            "Encoder-only autocast was asked for but no encoder ran under it."
        )
    return out


# ──────────────────────────────────────────────────────────────────────────
# TorchBackend
# ──────────────────────────────────────────────────────────────────────────


def _check_scope_compile(backend: "TorchBackend", attribute: Any, value: str) -> None:
    """``autocast_scope="encoder"`` is not supported with ``use_compile``."""
    if value != "all" and backend.use_compile:
        raise ValueError(
            "autocast_scope='encoder' (precision '*-encoder') is not supported with "
            "use_compile=True."
        )


def _on_set_memory_format(backend: "TorchBackend", attribute: Any, value: str) -> str:
    """``on_setattr`` hook: convert the model when ``memory_format`` changes."""
    if value != backend.memory_format:
        backend._apply_memory_format(value)
    return value


@attrs.define(eq=False, slots=False)
class TorchBackend:
    """PyTorch ``nn.Module`` backend with opt-in compile / FP16 / fusion.

    Args:
        model: The forward-pass owner. Typically a Lightning module
            (``SingleInstanceLightningModule`` etc.) but any callable
            ``nn.Module`` works.
        device: ``"cpu"``, ``"cuda"``, ``"cuda:N"``, or ``"mps"``.
        use_compile: Wrap the model in ``torch.compile``. CUDA-only.
            Emits a numeric-drift warning.
        compile_mode: Forwarded to ``torch.compile`` when enabled.
        use_fp16: Run the heavy forward ops in float16 via ``torch.autocast``
            (CUDA only; fp32 master weights preserved, outputs cast back to
            fp32). Opt-in; benefits tensor-core hardware only.
            CUDA-only; counter-productive at batch < 4. Emits a drift warning.
        fuse_layers: Fold ``Conv2d → BatchNorm2d`` pairs in-place. Negligible
            speedup on the test UNets; opt-in.
        autocast_dtype: Run the forward under ``torch.autocast`` with this
            dtype (``torch.bfloat16`` or ``torch.float16``; CUDA only, outputs
            cast back to fp32). Takes precedence over ``use_fp16``. Set by
            ``Predictor.from_model_paths(precision=...)``, which warns when it
            has no effect (off CUDA, exported models); the backend itself
            emits no warning.
        autocast_scope: Experimental. What ``autocast_dtype`` covers:
            ``"all"`` (default, the whole forward) or ``"encoder"`` (only the
            backbones' encoders, see :func:`autocast_forward`; every layer after
            them runs in fp32 on fp32 inputs). Set by
            ``Predictor.from_model_paths(precision="bf16-encoder")``. Not with
            ``use_compile``.
        memory_format: ``"contiguous"`` (default, channels-first) or
            ``"channels_last"``: the memory layout of the model's weights.
            cuDNN's fast convolution kernels on NVIDIA GPUs want the
            channels last, so a channels-first model has its activations
            converted around every convolution. ``"channels_last"`` ran a
            top-down UNet pair 2.2× faster on an A100 (fp32) and made no
            difference on AMD (ROCm). Can also be set after construction;
            the model is converted right away.
        warmup_iterations: Number of dummy forwards to run inside
            :meth:`warmup`. ``1`` is enough on MPS / CUDA.

    Notes:
        ``slots=False`` is intentional — attrs-with-slots doesn't compose
        with Lightning's ``__getattr__`` (which forwards to ``nn.Module``).
        Using a regular class lets users mix the two without surprise.
    """

    model: nn.Module = attrs.field(repr=False)
    device: str = "cpu"
    use_compile: bool = False
    compile_mode: str = "reduce-overhead"
    use_fp16: bool = False
    fuse_layers: bool = False
    autocast_dtype: Optional[torch.dtype] = None
    autocast_scope: str = attrs.field(
        default="all",
        validator=[attrs.validators.in_(AUTOCAST_SCOPES), _check_scope_compile],
    )
    warmup_iterations: int = 1
    memory_format: str = attrs.field(
        default="contiguous",
        validator=attrs.validators.in_(("contiguous", "channels_last")),
        on_setattr=attrs.setters.pipe(attrs.setters.validate, _on_set_memory_format),
    )

    # Internal state, not part of the public surface.
    _compiled: Optional[nn.Module] = attrs.field(default=None, init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        """Validate device / feature combination, fuse, and (optionally) compile."""
        self.model = self.model.to(self.device).eval()
        self._validate_device_features()

        if self.fuse_layers:
            self._fuse_conv_bn()

        if self.memory_format != "contiguous":
            self._apply_memory_format(self.memory_format)

        # FP16 is applied at forward time via ``torch.autocast`` (CUDA only) —
        # see ``__call__``. We deliberately do NOT call ``model.half()``: a
        # destructive whole-model half cast raises an Input/weight dtype
        # mismatch on any ``forward`` that changes dtype internally (e.g.
        # ``image / 255.0`` normalization or an explicit ``.float()``) and gives
        # worse numerics than autocast's fp32 master weights. MPS/CPU keep fp32
        # (MPS has half kernels but no tensor cores — no win, warned above).

        if self.use_compile and self.device != "mps":
            self._compiled = torch.compile(
                self.model, mode=self.compile_mode, dynamic=False
            )

    # ──────────────────────────────────────────────────────────────────
    # Protocol surface
    # ──────────────────────────────────────────────────────────────────

    @property
    def does_baked_postproc(self) -> bool:
        """PyTorch returns raw confmaps; peak finding stays in Python."""
        return False

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass. Always returns a dict for protocol uniformity."""
        model = self._compiled if self._compiled is not None else self.model
        x = x.to(self.device, non_blocking=True)

        # FP16 (CUDA only) runs the heavy conv / matmul ops in half precision
        # via ``torch.autocast`` while keeping fp32 master weights and fp32
        # reductions. Autocast is robust to ``forward`` methods that change
        # dtype internally (``image / 255.0`` normalization, an explicit
        # ``.float()``, etc.) — a destructive ``model.half()`` would instead
        # raise an Input/weight dtype mismatch on those. Outputs are cast back
        # to fp32 below so downstream peak-finding sees fp32 confmaps.
        dtype = self._forward_autocast_dtype()
        use_autocast = dtype is not None and "cuda" in self.device
        with torch.inference_mode():
            if use_autocast:
                out = autocast_forward(
                    model, x, dtype, scope=self.autocast_scope, device_type="cuda"
                )
            else:
                out = model(x)

        half = (torch.float16, torch.bfloat16)
        if isinstance(out, torch.Tensor):
            if use_autocast and out.dtype in half:
                out = out.float()
            return {"output": out}

        if isinstance(out, dict):
            if use_autocast:
                out = {
                    k: (
                        v.float()
                        if isinstance(v, torch.Tensor) and v.dtype in half
                        else v
                    )
                    for k, v in out.items()
                }
            return out

        raise TypeError(
            f"TorchBackend got unexpected output type {type(out).__name__}; "
            "expected Tensor or Dict[str, Tensor]."
        )

    def warmup(self, input_shape: Tuple[int, ...]) -> None:
        """Prime the backend with ``warmup_iterations`` dummy forwards."""
        if self.device == "cpu":
            return
        # Autocast handles the fp16 path internally, so the warmup dummy stays
        # fp32 (matching the dtype of the real preprocessed input).
        dummy = torch.zeros(input_shape, device=self.device, dtype=torch.float32)
        for _ in range(self.warmup_iterations):
            self(dummy)
        if "cuda" in self.device:
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _forward_autocast_dtype(self) -> Optional[torch.dtype]:
        """Autocast dtype of the forward: ``autocast_dtype``, else fp16 if ``use_fp16``."""
        if self.autocast_dtype is not None:
            return self.autocast_dtype
        return torch.float16 if self.use_fp16 else None

    def _apply_memory_format(self, memory_format: str) -> None:
        """Convert the model's weights to ``memory_format`` (in place).

        Only the weights are converted. Inputs keep their layout: PyTorch
        runs a convolution channels-last when its weight is, and the
        activations stay channels-last from there. An existing compiled
        wrapper is rebuilt for the new layout.
        """
        fmt = (
            torch.channels_last
            if memory_format == "channels_last"
            else torch.contiguous_format
        )
        self.model.to(memory_format=fmt)
        if self._compiled is not None:
            self._compiled = torch.compile(
                self.model, mode=self.compile_mode, dynamic=False
            )

    def _validate_device_features(self) -> None:
        """Warn-and-disable feature combinations that don't work cleanly.

        - MPS + ``torch.compile``: unreliable; force ``use_compile=False``.
        - MPS + FP16: kernels exist but tensor cores don't — no speedup;
          we keep ``use_fp16`` enabled but warn.
        - CUDA + ``torch.compile``: works; warn about graph fusion changing
          numerics.
        - CUDA + FP16: warn about ~4e-3 drift and small-batch regression.
        """
        if self.device == "mps":
            if self.use_compile:
                warnings.warn(
                    "torch.compile is unreliable on MPS; disabling.",
                    stacklevel=3,
                )
                self.use_compile = False
            if self.use_fp16:
                warnings.warn(
                    "FP16 on MPS gives no throughput gain (no tensor cores).",
                    stacklevel=3,
                )

        if "cuda" in self.device:
            if self.use_compile:
                warnings.warn(
                    "torch.compile changes numerics: graph fusion can substitute "
                    "TF32/reduced-precision kernels for FP32, producing small "
                    "drift vs. eager. Disable for parity-critical comparisons; "
                    "validate downstream metrics before shipping.",
                    stacklevel=3,
                )
            if self.use_fp16:
                warnings.warn(
                    "FP16 trades precision for speed. This backend runs the heavy "
                    "ops in half precision via torch.autocast (fp32 master weights "
                    "preserved). Measured max-abs-diff vs FP32 on the test "
                    "single-instance UNet (A40, batch 1-16): ~4e-3. Note: FP16 is "
                    "*counterproductive at small batch* — at batch=1 it ran 0.65× "
                    "the FP32 speed because tensor cores aren't saturated and "
                    "kernel-launch overhead dominates. Validate parity tests AND "
                    "speed at your actual batch size before enabling.",
                    stacklevel=3,
                )

    def _fuse_conv_bn(self) -> None:
        """Fold ``Conv2d → BatchNorm2d`` pairs in-place inside Sequentials.

        Fusion is restricted to ``nn.Sequential`` blocks, where execution
        order is guaranteed to match registration order.

        Why only ``nn.Sequential``? ``named_children()`` yields submodules in
        *registration* order, which is **not** the same as *execution* order
        for a module with a custom ``forward`` that reorders or skips
        submodules. Fusing a (Conv2d, BatchNorm2d) pair that happens to be
        registered consecutively — but is not actually applied consecutively
        in ``forward`` — folds the BN into the wrong conv and replaces the BN
        with ``nn.Identity()``, silently changing the model's output (observed
        max-abs-diff up to ~1.7 on a model whose ``forward`` reorders).

        ``nn.Sequential`` is the one container whose ``forward`` is *defined*
        to run children in registration order, so adjacency there genuinely
        implies "executed consecutively". We therefore only fuse Conv→BN pairs
        that are immediate neighbours inside an ``nn.Sequential`` and skip
        every other module — preferring a missed (~0% win) fusion over a
        silent mis-fuse. The sleap-nn UNet backbones build their Conv→BN
        stacks as ``nn.Sequential`` blocks, so the useful fusions are still
        covered.

        ``fuse_conv_bn_eval`` requires eval mode (running stats frozen); the
        backend has already called ``model.eval()`` before this runs.
        """
        from torch.nn.utils.fusion import fuse_conv_bn_eval

        def _fuse_in(parent: nn.Module) -> None:
            # Recurse into every child first so nested Sequentials get fused.
            for child in parent.children():
                _fuse_in(child)

            # Only Sequential guarantees registration order == execution
            # order, so it is the only place adjacency is safe to fuse.
            if not isinstance(parent, nn.Sequential):
                return

            entries: Any = list(parent.named_children())
            for i in range(len(entries) - 1):
                name, child = entries[i]
                bn_name, bn = entries[i + 1]
                if isinstance(child, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                    fused = fuse_conv_bn_eval(child, bn)
                    setattr(parent, name, fused)
                    setattr(parent, bn_name, nn.Identity())

        _fuse_in(self.model)
