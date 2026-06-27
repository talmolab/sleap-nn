"""ONNX export wrapper for the ``embedding`` model type."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from sleap_nn.export.wrappers.base import BaseExportWrapper


class EmbeddingONNXWrapper(BaseExportWrapper):
    """Wrap an embedding model for ONNX export: crop -> appearance vector.

    The simplest wrapper (single input/output, no peak finding): WHOLE-crop per-crop
    standardize the input, run the encoder + head, optionally L2-normalize. Output:
    ``{"embedding": (B, D)}``.

    Parity: this exactly reproduces native inference for a ``burn_in=False`` embedder
    (whose ``_standardize`` also normalizes over the whole crop). A ``burn_in=True``
    embedder standardizes over the FOREGROUND (mask) only and fills the background, which
    this single-input graph cannot replicate — its exported embeddings therefore DIVERGE
    from native masked inference. The export CLI records ``burn_in``/``background_fill``
    in the metadata and warns on a ``burn_in=True`` export; use the native
    ``sleap-nn embed`` path for exact parity with such a model.
    """

    def __init__(self, model, normalize: bool = True, eps: float = 1e-5):
        """Initialize.

        Args:
            model: The underlying ``Model`` (encoder + EmbeddingHead).
            normalize: L2-normalize the output embedding.
            eps: Standardization epsilon.
        """
        super().__init__(model)
        self.normalize = normalize
        self.eps = eps

    def forward(self, image: torch.Tensor):
        """image: (B, C, H, W) [0, 255] -> {"embedding": (B, D)}.

        Replicates ``EmbeddingLightningModule._standardize``'s MASKLESS path exactly
        (i.e. the ``burn_in=False`` native path): a 1-channel ones "mask" makes the
        normalization count the spatial size (H*W), summing the numerator over all
        channels. For grayscale (C=1) this is the plain per-crop standardize; for RGB
        (C=3) it matches the PyTorch inference path (whose ``cnt`` is also H*W). Computing
        the count from a ones tensor (rather than a baked H*W constant) keeps the graph
        valid under dynamic spatial axes. NOTE: a ``burn_in=True`` model's masked
        (foreground-only) standardize is NOT reproduced here — see the class docstring.
        """
        x = image.float()
        ones = torch.ones_like(x[:, :1])
        cnt = ones.sum((1, 2, 3), keepdim=True).clamp(min=1)
        mu = (x * ones).sum((1, 2, 3), keepdim=True) / cnt
        var = ((x - mu) ** 2 * ones).sum((1, 2, 3), keepdim=True) / cnt
        x = (x - mu) / (var.sqrt() + self.eps)
        feat = self._extract_tensor(self.model(x), ["embedding", "vector"])
        if feat.dim() > 2:
            feat = feat.flatten(1)
        if self.normalize:
            feat = F.normalize(feat, p=2, dim=-1)
        return {"embedding": feat}
