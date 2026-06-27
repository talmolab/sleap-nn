"""Embedding (crop -> appearance vector, re-ID) inference layers.

Two pieces, mirroring the keypoint / segmentation top-down stacks:

* :class:`EmbeddingLayer` — the single-stage (precropped / mask-driven) embedder.
  Runs a trained ``embedding`` model on per-instance crops and returns one
  L2-normalized appearance vector per crop on ``Outputs.pred_embeddings``
  ``(B, I=1, D)``. It ALSO populates ``instance_scores`` / ``instance_valid``
  so the detection ENUMERATES correctly (SPEC §6 [AUDIT]: ``pred_embeddings``
  alone is insufficient — ``n_instances`` ignores it and ``to_instances``
  compacts all-NaN slots). The crop pipeline (mask burn-in + per-crop
  standardize) is IDENTICAL to training, via the LightningModule's
  ``_build_input`` — so the embeddings match the validation retrieval metrics.

* :class:`TopDownEmbeddingLayer` — subclasses
  :class:`~sleap_nn.inference.layers.topdown.TopDownLayer` to reuse its stage-1
  (centroid) + sizematch + crop machinery verbatim, overriding only stage 2 to
  run the embedder on each crop and pack ``Outputs.pred_embeddings`` ``(B, I, D)``
  (+ ``pred_centroids`` / ``instance_scores`` / ``instance_valid``). The
  GT-centroid fallback (``CentroidLayer(use_gt_centroids=True)``) covers the
  mask-only data the same way :class:`TopDownSegmentationLayer` does.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sleap_nn.data.instance_cropping import make_centered_bboxes
from sleap_nn.inference.layers.backends.base import ModelBackend
from sleap_nn.inference.layers.base import ImageInput, InferenceLayer
from sleap_nn.inference.layers.centroid import CentroidLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.layers.topdown import TopDownLayer
from sleap_nn.inference.ops.crops import crop_bboxes
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.preprocess_info import PreprocInfo


def _build_embedding_input(
    embedding_module,
    crops: torch.Tensor,
    masks: Optional[torch.Tensor],
    device: str,
) -> torch.Tensor:
    """Grayscale + mask burn-in + per-crop standardize, exactly like training.

    Reuses ``EmbeddingLightningModule._build_input`` so the inference crop
    pipeline is byte-for-byte the validation pipeline. When ``masks`` is
    ``None`` a whole-crop (all-ones) standardize is used (the no-mask fallback,
    matching ``EmbeddingLightningModule.forward``).

    Args:
        embedding_module: The trained ``EmbeddingLightningModule``.
        crops: ``(N, 1, H, W)`` grayscale crops (float).
        masks: Optional ``(N, 1, H, W)`` binary mask crops (float). ``None``
            falls back to whole-crop standardize.
        device: Device to run the build on.

    Returns:
        ``(N, 1, H, W)`` standardized, mask-burned-in input ready for the model.
    """
    gray = crops.to(device).to(torch.float32)
    if masks is not None:
        mask = masks.to(device).to(torch.float32)
    else:
        mask = torch.ones_like(gray[:, :1])
    return embedding_module._build_input(gray, mask)


class EmbeddingLayer(InferenceLayer):
    """Single-stage (precropped / mask-driven) appearance-embedding layer.

    Runs a trained ``embedding`` model on per-instance crops and returns one
    L2-normalized vector per crop on ``Outputs.pred_embeddings`` ``(B, I=1, D)``.
    The structural twin of
    :class:`~sleap_nn.inference.layers.centered_instance.CenteredInstanceLayer`,
    but returns appearance vectors rather than keypoints.

    Args:
        backend: Runtime backend wrapping ``embedding_module.model`` (its
            forward returns ``{"EmbeddingHead": (N, D)}``).
        embedding_module: The trained ``EmbeddingLightningModule`` — used for
            its ``_build_input`` (mask burn-in + per-crop standardize).
        embedding_dim: The output vector dimension ``D``.
        output_stride: Head map → crop-pixel stride (cosmetic for embeddings).
        max_stride: Backbone max stride.
        preprocess_config / postprocess_config: Standard knobs. The crops are
            already sized; only the model's own ``input_scale`` is applied.
    """

    _HEAD_OUTPUT_KEY: str = "EmbeddingHead"

    def __init__(
        self,
        backend: ModelBackend,
        embedding_module,
        embedding_dim: int,
        output_stride: int = 1,
        max_stride: int = 1,
        input_channels: int = 1,
        preprocess_config: Optional[PreprocessConfig] = None,
        postprocess_config: Optional[PostprocessConfig] = None,
    ) -> None:
        """Stash the embedder + configs."""
        super().__init__(
            backend=backend,
            preprocess_config=preprocess_config or PreprocessConfig(),
            postprocess_config=postprocess_config or PostprocessConfig(),
            output_stride=output_stride,
            max_stride=max_stride,
        )
        self.embedding_module = embedding_module
        self.embedding_dim = int(embedding_dim)
        # Data channels the model was trained on (1 = grayscale default, 3 = RGB). The
        # crop is coerced to this so inference matches the training input.
        self.input_channels = int(input_channels)
        # The composed TopDownLayer machinery inspects this on stage-2 layers.
        self.use_gt_peaks = False

    @property
    def warmup_input_shape(self):
        """Tiny single-channel warmup shape."""
        return (1, 1, 64, 64)

    def predict(
        self,
        crops: ImageInput,
        masks: Optional[ImageInput] = None,
    ) -> Outputs:
        """Embed a batch of crops → ``Outputs.pred_embeddings`` ``(N, 1, D)``.

        Args:
            crops: ``(N, C, H, W)`` (or any shape coercible by
                :meth:`InferenceLayer._to_4d_tensor`) grayscale crops.
            masks: Optional ``(N, 1, H, W)`` binary mask crops for mask burn-in.
                ``None`` uses whole-crop standardize.

        Returns:
            ``Outputs`` with ``pred_embeddings`` ``(N, 1, D)``, ``instance_scores``
            ``(N, 1)`` (all ones), and ``instance_valid`` ``(N, 1)`` (all True) so
            each crop enumerates as exactly one detection.
        """
        x = self._to_4d_tensor(crops)
        # Coerce to the training data channels (grayscale by default; RGB if the model
        # was trained with ensure_rgb).
        if self.input_channels == 1 and x.shape[1] != 1:
            from sleap_nn.data.normalization import convert_to_grayscale

            x = convert_to_grayscale(x.float())
        elif self.input_channels == 3 and x.shape[1] != 3:
            from sleap_nn.data.normalization import convert_to_rgb

            x = convert_to_rgb(x.float())
        m = None
        if masks is not None:
            m = self._to_4d_tensor(masks).float()
        inp = _build_embedding_input(
            self.embedding_module, x.float(), m, self.backend.device
        )
        raw = self.backend(inp)
        return self.postprocess(raw, info=None)

    def __call__(
        self,
        crops: ImageInput,
        masks: Optional[ImageInput] = None,
    ) -> Outputs:
        """Alias for :meth:`predict`."""
        return self.predict(crops, masks=masks)

    def postprocess(self, raw_out: dict, info: Optional[PreprocInfo]) -> Outputs:
        """Package the embedder output into ``Outputs.pred_embeddings``.

        Args:
            raw_out: Backend dict carrying the ``EmbeddingHead`` vectors ``(N, D)``.
            info: Unused (crops are already sized).

        Returns:
            ``Outputs`` with ``pred_embeddings`` ``(N, 1, D)`` + ``instance_scores``
            ``(N, 1)`` + ``instance_valid`` ``(N, 1)``.
        """
        emb = self._extract_confmaps(raw_out).detach()  # (N, D)
        n = emb.shape[0]
        device = emb.device
        return Outputs(
            pred_embeddings=emb.unsqueeze(1),  # (N, 1, D)
            instance_scores=torch.ones((n, 1), device=device),
            instance_valid=torch.ones((n, 1), dtype=torch.bool, device=device),
            preprocess_info=info,
        )


class TopDownEmbeddingLayer(TopDownLayer):
    """Composed centroid + per-crop-embedding two-stage layer.

    Subclasses :class:`TopDownLayer` to reuse stage 1 (centroid) + sizematch +
    crop extraction verbatim, overriding only :meth:`_run_stage_2` to emit one
    appearance vector per crop into ``Outputs.pred_embeddings`` instead of
    keypoints.

    Args:
        centroid_layer: Stage-1 :class:`CentroidLayer` (real model or
            ``use_gt_centroids=True`` for the GT-centroid fallback).
        centered_instance_layer: Stage-2 :class:`EmbeddingLayer`.
        crop_size: ``(crop_h, crop_w)`` of the per-instance crop.
        centroid_nms / centroid_nms_threshold: Optional centroid dedup
            (inherited).
    """

    def __init__(
        self,
        centroid_layer: CentroidLayer,
        centered_instance_layer: EmbeddingLayer,
        crop_size: Tuple[int, int],
        centroid_nms: bool = False,
        centroid_nms_threshold: float = 0.5,
    ) -> None:
        """Stash inner layers + crop size."""
        super().__init__(
            centroid_layer=centroid_layer,
            centered_instance_layer=centered_instance_layer,
            crop_size=crop_size,
            centroid_nms=centroid_nms,
            centroid_nms_threshold=centroid_nms_threshold,
            return_crops=False,
        )

    def _run_stage_2(
        self,
        image_4d: torch.Tensor,
        centroids: torch.Tensor,
        centroid_vals: torch.Tensor,
        valid_mask: torch.Tensor,
        eff_scale: Optional[torch.Tensor] = None,
    ) -> Outputs:
        """Crop around valid centroids, embed each crop, scatter into ``(B, I, D)``.

        ``image_4d`` / ``centroids`` are in **sized** space (after the centroid
        layer's sizematcher). Each valid crop is embedded; the resulting vector
        is scattered into ``pred_embeddings`` ``(B, max_inst, D)`` with all-NaN
        for empty slots. ``instance_valid`` marks the populated slots so the
        detection enumerates correctly.
        """
        B, max_inst, _ = centroids.shape
        crop_h, crop_w = self.crop_size
        D = self.centered_instance_layer.embedding_dim

        valid_idx = valid_mask.nonzero(as_tuple=False)  # (n_valid, 2) — (b, i)
        n_valid = valid_idx.shape[0]

        if eff_scale is None:
            eff_scale = torch.ones(B, dtype=torch.float32, device=centroids.device)
        else:
            eff_scale = eff_scale.to(centroids.device)
        centroids_in_image_space = centroids / eff_scale.view(-1, 1, 1)

        if n_valid == 0:
            return Outputs(
                pred_embeddings=torch.full(
                    (B, max_inst, D), float("nan"), device=centroids.device
                ),
                pred_centroids=centroids_in_image_space,
                pred_centroid_values=centroid_vals,
                instance_scores=centroid_vals,
                instance_valid=torch.zeros(
                    (B, max_inst), dtype=torch.bool, device=centroids.device
                ),
            )

        valid_centroids = centroids[valid_idx[:, 0], valid_idx[:, 1]]
        sample_inds = valid_idx[:, 0]  # (n_valid,)

        bboxes = make_centered_bboxes(valid_centroids, crop_h, crop_w)
        crops = crop_bboxes(image_4d, bboxes, sample_inds)

        stage2_out = self.centered_instance_layer.predict(crops)
        crop_emb = stage2_out.pred_embeddings.squeeze(1)  # (n_valid, D)
        device = crop_emb.device

        valid_idx = valid_idx.to(device)
        centroid_vals = centroid_vals.to(device)
        centroids_in_image_space = centroids_in_image_space.to(device)

        full_emb = torch.full((B, max_inst, D), float("nan"), device=device)
        full_emb[valid_idx[:, 0], valid_idx[:, 1]] = crop_emb
        full_valid = torch.zeros((B, max_inst), dtype=torch.bool, device=device)
        full_valid[valid_idx[:, 0], valid_idx[:, 1]] = True

        return Outputs(
            pred_embeddings=full_emb,
            pred_centroids=centroids_in_image_space,
            pred_centroid_values=centroid_vals,
            instance_scores=centroid_vals,
            instance_valid=full_valid,
        )
