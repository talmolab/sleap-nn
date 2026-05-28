"""``TorchBackend`` — wrap any ``nn.Module`` (or Lightning module) for inference.

Concrete ``ModelBackend`` for PyTorch. Adds opt-in optimizations
(``torch.compile``, FP16 autocast, Conv+BN fusion, GPU warmup) gated
behind benchmark-informed defaults.

Defaults (from `12-design-review-and-revised-plan.md` §2 + CUDA validation):

- ``warmup_iterations = 1`` — default ON; 73× cold-start ratio on MPS
- ``fuse_layers      = False`` — opt-in; ≈0% on the test UNets
- ``use_compile      = False`` — opt-in; **also broken upstream** by #527
- ``use_fp16         = False`` — opt-in; tensor-core only, regresses at
  small batch on CUDA (0.65× FP32 at batch=1, 1.5× at batch=16)

Warnings emitted on construction (verified by tests):

- ``UserWarning`` on CUDA with ``use_compile=True`` — numeric drift
- ``UserWarning`` on CUDA with ``use_fp16=True`` — drift + small-batch perf
- ``UserWarning`` on MPS with ``use_compile=True`` — disables and downgrades
- ``UserWarning`` on MPS with ``use_fp16=True`` — no tensor cores, no win
- ``RuntimeError`` if ``use_compile=True`` while the swint ``torch.fx.wrap``
  registry is contaminated (#527 workaround). Raises with a clear message
  pointing at the upstream issue so users know it's not a config bug.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import attrs
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────
# #527 workaround — detect the contaminated fx.wrap registry
# ──────────────────────────────────────────────────────────────────────────


_FX_WRAPPED_NAMES = ("_patch_merging_pad", "_get_relative_position_bias")


def _swint_fx_wrap_blocks_compile() -> bool:
    """Detect the swint ``torch.fx.wrap`` registry contamination from #527.

    ``sleap_nn/architectures/swint.py`` calls ``torch.fx.wrap(...)`` at
    module-import time, which globally registers function names into
    ``torch.fx._symbolic_trace._wrapped_fns_to_patch``. The dynamo backend
    used by ``torch.compile`` chokes on these unknown names — even when the
    wrapped functions are never actually called by the model under compile.

    Returns ``True`` iff the registry is contaminated. Tested directly so we
    don't have to actually attempt a compile to detect breakage.
    """
    try:
        from torch.fx._symbolic_trace import _wrapped_fns_to_patch
    except ImportError:
        return False
    registered = {entry[1] for entry in _wrapped_fns_to_patch}
    return any(name in registered for name in _FX_WRAPPED_NAMES)


# ──────────────────────────────────────────────────────────────────────────
# TorchBackend
# ──────────────────────────────────────────────────────────────────────────


@attrs.define(eq=False, slots=False)
class TorchBackend:
    """PyTorch ``nn.Module`` backend with opt-in compile / FP16 / fusion.

    Args:
        model: The forward-pass owner. Typically a Lightning module
            (``SingleInstanceLightningModule`` etc.) but any callable
            ``nn.Module`` works.
        device: ``"cpu"``, ``"cuda"``, ``"cuda:N"``, or ``"mps"``.
        use_compile: Wrap the model in ``torch.compile``. CUDA-only.
            Emits a numeric-drift warning. Raises ``RuntimeError`` if the
            swint fx.wrap registry is contaminated (see #527).
        compile_mode: Forwarded to ``torch.compile`` when enabled.
        use_fp16: Run the forward pass in float16. CUDA-only;
            counter-productive at batch < 4. Emits a drift warning.
        fuse_layers: Fold ``Conv2d → BatchNorm2d`` pairs in-place. Negligible
            speedup on the test UNets; opt-in.
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
    warmup_iterations: int = 1

    # Internal state, not part of the public surface.
    _compiled: Optional[nn.Module] = attrs.field(default=None, init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        """Validate device / feature combination, fuse, and (optionally) compile."""
        self.model = self.model.to(self.device).eval()
        self._validate_device_features()

        if self.fuse_layers:
            self._fuse_conv_bn()

        if self.use_compile:
            if _swint_fx_wrap_blocks_compile():
                raise RuntimeError(
                    "torch.compile is currently blocked by the swint torch.fx.wrap "
                    "registry contamination (see issue #527). Importing "
                    "sleap_nn.architectures.swint registers '_patch_merging_pad' "
                    "into torch.fx._symbolic_trace._wrapped_fns_to_patch at "
                    "module-import time, which dynamo's compile pipeline cannot "
                    "handle even on non-SwinT checkpoints. Workaround: leave "
                    "use_compile=False until #527 lands. "
                    "Tracking: https://github.com/talmolab/sleap-nn/issues/527"
                )
            if self.device != "mps":
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

        if self.use_fp16 and x.is_floating_point() and "cuda" in self.device:
            x = x.half()

        with torch.inference_mode():
            out = model(x)

        # Convert back to fp32 if the upstream Lightning forward returned a
        # tensor — the wrapping layer expects fp32 confmaps regardless of
        # backend dtype.
        if isinstance(out, torch.Tensor):
            if self.use_fp16 and out.dtype == torch.float16:
                out = out.float()
            return {"output": out}

        if isinstance(out, dict):
            if self.use_fp16:
                out = {
                    k: (
                        v.float()
                        if isinstance(v, torch.Tensor) and v.dtype == torch.float16
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
        dtype = (
            torch.float16
            if (self.use_fp16 and "cuda" in self.device)
            else torch.float32
        )
        dummy = torch.zeros(input_shape, device=self.device, dtype=dtype)
        for _ in range(self.warmup_iterations):
            self(dummy)
        if "cuda" in self.device:
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _validate_device_features(self) -> None:
        """Warn-and-disable feature combinations that don't work cleanly.

        - MPS + ``torch.compile``: unreliable; force ``use_compile=False``.
        - MPS + FP16: kernels exist but tensor cores don't — no speedup;
          we keep ``use_fp16`` enabled but warn.
        - CUDA + ``torch.compile``: works (when #527 isn't blocking);
          warn about graph fusion changing numerics.
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
                    "FP16 trades precision for speed. Measured max-abs-diff vs "
                    "FP32 on the test single-instance UNet (A40, batch 1-16): "
                    "~4e-3 (autocast and model.half() are equivalent). Note: "
                    "FP16 is *counterproductive at small batch* — at batch=1 "
                    "it ran 0.65× the FP32 speed because tensor cores aren't "
                    "saturated and kernel-launch overhead dominates. Validate "
                    "parity tests AND speed at your actual batch size before "
                    "enabling.",
                    stacklevel=3,
                )

    def _fuse_conv_bn(self) -> None:
        """Fold sequential ``Conv2d → BatchNorm2d`` pairs in-place.

        Walks the module tree and replaces each (Conv2d, BatchNorm2d) pair
        with a single fused Conv2d using
        ``torch.nn.utils.fusion.fuse_conv_bn_eval``. Skips pairs where the
        BN tracks running stats and ``training`` mode is on (would be
        unsafe).
        """
        from torch.nn.utils.fusion import fuse_conv_bn_eval

        def _fuse_in(parent: nn.Module) -> None:
            children: Any = list(parent.named_children())
            for i, (name, child) in enumerate(children):
                _fuse_in(child)
                if (
                    i + 1 < len(children)
                    and isinstance(child, nn.Conv2d)
                    and isinstance(children[i + 1][1], nn.BatchNorm2d)
                ):
                    bn_name, bn = children[i + 1]
                    fused = fuse_conv_bn_eval(child, bn)
                    setattr(parent, name, fused)
                    setattr(parent, bn_name, nn.Identity())

        _fuse_in(self.model)
