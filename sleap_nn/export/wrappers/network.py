"""Network-only export wrapper: image -> raw head outputs, nothing baked in."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn

from sleap_nn.export.wrappers.base import BaseExportWrapper


class NetworkONNXWrapper(BaseExportWrapper):
    """Wrap a model for a network-only export (``sleap-nn export --network-only``).

    The graph is exactly what the model's ``LightningModule.forward`` computes from
    an already-preprocessed input: ``[0, 255]`` image -> ``[0, 1]`` -> network ->
    the named head outputs, unchanged (raw confidence maps for a centered-instance
    model). No resize, peak finding or decoding is traced: an exported network runs
    inside the checkpoint inference pipeline in place of the checkpoint
    (:class:`~sleap_nn.inference.exported_network.ExportedNetwork`), whose host-side
    steps -- crops (fixed or box-scaled), peak finding and refinement, the part
    stage -- are the checkpoint path's own.

    Args:
        model: The underlying ``Model`` (backbone + heads).
        head_names: Head outputs to return, in order (the graph's output names).
    """

    def __init__(self, model: nn.Module, head_names: Sequence[str]):
        """Initialize."""
        super().__init__(model)
        self.head_names = list(head_names)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the network and return the raw head outputs."""
        out = self.model(self._normalize_uint8(image))
        return {name: out[name] for name in self.head_names}
