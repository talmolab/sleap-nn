"""Common utilities for architecture and model building."""

from typing import List

import torch
from torch import nn
from torch.nn import functional as F


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

    def encoder_modules(self) -> List[nn.Module]:
        """Return the modules that make up the encoder (default: ``[self.enc]``)."""
        return [self.enc]

    def freeze_encoder(self) -> None:
        """Freeze the encoder: no gradient to its weights, eval mode from now on."""
        self.freeze = True
        for module in self.encoder_modules():
            module.eval()
            module.requires_grad_(False)

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
        """
        return int(
            max(
                (torch.ceil(torch.tensor(i / s)).item() - 1) * s + (k - 1) * d + 1 - i,
                0,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MaxPool2dWithSamePadding module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the MaxPool2d operation.
        """
        if self.padding == "same":
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
            self.padding = 0

        return F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )
