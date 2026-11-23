"""Common utilities for architecture and model building."""

import contextlib
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

#: The active encoder-autocast request (see :func:`encoder_autocast`), or ``None``.
_ENCODER_AUTOCAST: Optional[Dict[str, Any]] = None
#: Guards changes to :data:`_ENCODER_AUTOCAST`.
_ENCODER_AUTOCAST_LOCK = threading.Lock()


@contextlib.contextmanager
def encoder_autocast(device_type: str, dtype: torch.dtype) -> Iterator[Dict[str, Any]]:
    """Ask the backbones run in this block to run their encoders under autocast.

    Experimental (``sleap-nn predict --precision bf16-encoder``). Backbones that
    support it (``supports_encoder_autocast = True``) run their encoder through
    :func:`run_encoder`, which enters ``torch.autocast(device_type, dtype)`` for
    the encoder only and casts its outputs back to fp32, so every layer after it
    runs in fp32. Outside this block :func:`run_encoder` is a plain call.

    The request belongs to the thread that made it: models run in other threads
    meanwhile run as without it, and only one thread can hold a request at a
    time (a second thread's ``encoder_autocast`` raises until the first exits).
    Requests nest within a thread.

    Yields:
        A dict whose ``"calls"`` counts the encoder calls that ran under it.

    Raises:
        RuntimeError: If another thread holds a request.
    """
    global _ENCODER_AUTOCAST
    owner = threading.get_ident()
    with _ENCODER_AUTOCAST_LOCK:
        previous = _ENCODER_AUTOCAST
        if previous is not None and previous["owner"] != owner:
            raise RuntimeError(
                "Encoder-only autocast is already active in another thread; run "
                "encoder-only reduced precision from one thread at a time."
            )
        state = {"device_type": device_type, "dtype": dtype, "calls": 0}
        state["owner"] = owner
        _ENCODER_AUTOCAST = state
    try:
        yield state
    finally:
        with _ENCODER_AUTOCAST_LOCK:
            _ENCODER_AUTOCAST = previous


def outputs_to_float32(out: Any) -> Any:
    """``out`` (a tensor or nested list / tuple / dict of them) with half floats as fp32."""
    if isinstance(out, torch.Tensor):
        return out.float() if out.dtype in (torch.float16, torch.bfloat16) else out
    if isinstance(out, dict):  # also a `transformers` ModelOutput: set in place
        for key in list(out.keys()):
            out[key] = outputs_to_float32(out[key])
        return out
    if isinstance(out, list):
        return [outputs_to_float32(v) for v in out]
    if isinstance(out, tuple):
        items = [outputs_to_float32(v) for v in out]
        return type(out)(*items) if hasattr(out, "_fields") else type(out)(items)
    return out


def encoder_autocast_requested() -> bool:
    """Whether an :func:`encoder_autocast` request is active (in any thread)."""
    return _ENCODER_AUTOCAST is not None


def run_encoder(fn: Callable, *args, **kwargs) -> Any:
    """``fn(*args, **kwargs)``: a backbone's encoder computation.

    A plain call, except inside :func:`encoder_autocast` in the thread that made
    the request, where it runs under that autocast and returns its outputs cast
    to fp32.
    """
    state = _ENCODER_AUTOCAST
    if state is None:
        return fn(*args, **kwargs)
    if state["owner"] != threading.get_ident():
        return fn(*args, **kwargs)  # another thread's request
    state["calls"] += 1  # only the owner thread counts
    with torch.autocast(device_type=state["device_type"], dtype=state["dtype"]):
        out = fn(*args, **kwargs)
    return outputs_to_float32(out)


class FreezableEncoderMixin:
    """Freeze a backbone's encoder and keep it in eval mode for the whole run.

    The encoder is the part of the backbone that pretrained weights load into: the
    HuggingFace model of a ``pretrained`` backbone, the ImageNet ``enc`` of a native
    ``convnext`` / ``swint`` backbone, the stem + encoder blocks of a ``unet``. Middle
    blocks and decoders are randomly initialized on those backbones and are NOT part
    of it. Subclasses whose encoder is not ``self.enc`` override ``encoder_modules``.

    Freezing means two things, and ``requires_grad_(False)`` alone does only the
    first: (1) no gradient reaches the encoder weights, and (2) the encoder runs in
    eval mode, so BatchNorm normalizes with its stored running statistics (and stops
    updating them) and dropout / stochastic depth are off. A "frozen" encoder that
    stays in train mode still changes: its BatchNorm statistics drift toward the new
    data and every training forward normalizes with batch statistics, so the
    features at inference differ from the ones the head was trained on.

    Place the mixin before ``nn.Module`` in the bases so its ``train`` wins.
    """

    #: Set by ``freeze_encoder``; read by ``train``.
    freeze: bool = False

    #: The backbone runs its encoder through :func:`run_encoder`, so
    #: :func:`encoder_autocast` (experimental) applies to it. Subclasses that do
    #: set it to ``True``.
    supports_encoder_autocast: bool = False

    def encoder_modules(self) -> List[nn.Module]:
        """Return the modules that make up the encoder (default: ``[self.enc]``)."""
        return [self.enc]

    def freeze_encoder(self) -> None:
        """Freeze the encoder: no gradient to its weights, eval mode from now on."""
        self.freeze = True
        # Remember what was trainable, so `unfreeze_encoder` does not revive
        # parameters that were excluded on purpose (e.g. ViT blocks after the last
        # tapped layer, which would break DDP).
        self._frozen_params = getattr(self, "_frozen_params", []) + [
            p
            for module in self.encoder_modules()
            for p in module.parameters()
            if p.requires_grad
        ]
        for module in self.encoder_modules():
            module.eval()
            module.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        """Undo ``freeze_encoder``: gradients flow again, encoder follows ``train()``.

        The caller is responsible for the optimizer: parameters frozen before the
        optimizer was built are not in it (see ``EncoderFreezeSchedule``).
        """
        self.freeze = False
        for p in getattr(self, "_frozen_params", []):
            p.requires_grad_(True)
        self._frozen_params = []
        for module in self.encoder_modules():
            module.train(self.training)

    def train(self, mode: bool = True):
        """Set train/eval mode, keeping a FROZEN encoder in eval.

        ``freeze_encoder`` puts the encoder in eval once, but Lightning calls
        ``model.train()`` at the start of every training epoch and that recurses
        into every submodule, so without this override the encoder goes back into
        train mode and its normalization layers keep updating. Weights stay frozen
        (``requires_grad_(False)``); the running statistics did not, which makes a
        "frozen" run irreproducible in a way that looks like seed noise (measured on
        ``microsoft/resnet-18``: ``running_mean`` moved 0.63 in a single forward).

        Only backbones with running statistics (BatchNorm: ResNet, BiT) or
        train-only stochasticity (the native ``swint``'s stochastic depth) are
        affected; LayerNorm-only encoders (ConvNeXt, ViT) compute the same thing in
        either mode.
        """
        super().train(mode)
        if getattr(self, "freeze", False):
            for module in self.encoder_modules():
                module.eval()
        return self


class MaxPool2dWithSamePadding(nn.MaxPool2d):
    """A MaxPool2d module with support for same padding.

    This class extends the torch.nn.MaxPool2d module and adds the ability
    to perform 'same' padding, similar to 'same' padding in convolutional
    layers. When 'same' padding is specified, the input tensor is padded
    with zeros to ensure that the output spatial dimensions match the input
    spatial dimensions as closely as possible.

    Args:
        nn.MaxPool2d arguments: Arguments that are passed to the parent
            torch.nn.MaxPool2d class.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass through the MaxPool2dWithSamePadding module.

    Note:
        The 'same' padding is applied only when self.padding is set to "same".

    Example:
        # Create an instance of MaxPool2dWithSamePadding
        maxpool_layer = MaxPool2dWithSamePadding(kernel_size=3, stride=2, padding="same")

        # Perform a forward pass on an input tensor
        input_tensor = torch.rand(1, 3, 32, 32)  # Example input tensor
        output = maxpool_layer(input_tensor)  # Apply the MaxPool2d operation with same padding.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the MaxPool2dWithSamePadding module."""
        super().__init__(*args, **kwargs)

    def _calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        """Calculate the required padding to achieve 'same' padding.

        Args:
            i (int): Input dimension (height or width).
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.

        Returns:
            int: The calculated padding value.

        Note:
            Integer arithmetic (``-(-i // s)`` is ``ceil(i / s)``), not a tensor
            + ``.item()``: the sizes are Python ints, and ``Tensor.item()`` makes
            ``torch.compile`` split the graph at every pooling layer (and fail
            with ``fullgraph=True`` on PyTorch 2.6). No float division either,
            so it stays exact and symbolic under ``torch.compile(dynamic=True)``.
        """
        return max((-(-i // s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MaxPool2dWithSamePadding module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the MaxPool2d operation.
        """
        padding = self.padding
        if padding == "same":
            ih, iw = x.size()[-2:]

            pad_h = self._calc_same_pad(
                i=ih,
                k=(
                    self.kernel_size
                    if type(self.kernel_size) is int
                    else self.kernel_size[0]
                ),
                s=self.stride if type(self.stride) is int else self.stride[0],
                d=self.dilation if type(self.dilation) is int else self.dilation[0],
            )
            pad_w = self._calc_same_pad(
                i=iw,
                k=(
                    self.kernel_size
                    if type(self.kernel_size) is int
                    else self.kernel_size[1]
                ),
                s=self.stride if type(self.stride) is int else self.stride[1],
                d=self.dilation if type(self.dilation) is int else self.dilation[1],
            )

            if pad_h > 0 or pad_w > 0:
                x = F.pad(
                    x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
                )
            # A local, not ``self.padding = 0``: mutating the module made "same"
            # padding apply on the first forward only.
            padding = 0

        return F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )
