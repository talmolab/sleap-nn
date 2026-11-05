"""``CentroidLayer`` — predicts instance centroids from a confmap model.

Single-stage layer used either standalone (centroid-only inference) or
composed with :class:`CenteredInstanceLayer` to form :class:`TopDownLayer`.

The ``use_gt_centroids=True`` flag skips the centroid model and reads
ground-truth centroids directly from a ``LabelsReader`` batch's
``"instances"`` field. Used for top-down inference when only the
centered_instance model is available.

The two GT fallback paths live on different layers:

* ``CentroidLayer.use_gt_centroids=True`` — GT *centroids* feed cropping
  for a real centered_instance model.
* ``CenteredInstanceLayer.use_gt_peaks=True`` — GT *keypoints* fill stage
  2 when only a centroid model is available.

Each is independently configurable on the layer that owns the role the
GT data plays.
"""

from __future__ import annotations

from typing import Optional

import attrs
import torch

from sleap_nn.data.instance_centroids import (
    degrade_anchor_if_unresolved,
    generate_centroids,
)
from sleap_nn.inference.layers.backends.base import ModelBackend
from sleap_nn.inference.layers.base import ImageInput, InferenceLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.ops.coord import (
    undo_eff_scale,
    undo_input_scale,
    undo_stride,
)
from sleap_nn.inference.ops.peaks import find_local_peaks
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.preprocess_info import PreprocInfo


class CentroidLayer(InferenceLayer):
    """Centroid prediction layer.

    Args:
        backend: Runtime backend for the centroid model. Required even when
            ``use_gt_centroids=True`` (the layer keeps the backend interface
            uniform; it just doesn't call it on the GT path).
        output_stride: Confmap → input-pixel stride from the head config.
        max_instances: Cap on returned centroids per frame. Below-cap results
            are NaN-padded; above-cap are truncated by ``topk`` confidence.
        max_stride: Maximum stride the model requires the input to be
            divisible by. Padding is applied bottom-right after the
            preprocess input-scale resize.
        centroid_method: How a GT centroid is derived from the instance's points
            when ``use_gt_centroids=True`` -- one of ``"center_of_mass"``,
            ``"bbox_center"``, ``"geometric_median"``, ``"anchor"``. ``None``
            (default) infers it from ``anchor_ind``, i.e. the historical behavior.
            Must match the training config; the loaders read it off the
            checkpoint's head config (#586).
        centroid_fallback: Reduce method used when the anchor node is not visible.
        anchor_ind: Skeleton-node index to use as the centroid anchor when
            ``use_gt_centroids=True``. ``None`` falls back to the NaN-ignoring
            mean of all visible nodes for each instance.
        use_gt_centroids: When ``True``, skip the model and read centroids
            from a batch's ``"instances"`` field (the LabelsReader path).
        preprocess_config / postprocess_config: Standard knobs.
    """

    _HEAD_OUTPUT_KEY: str = "CentroidConfmapsHead"

    def __init__(
        self,
        backend: ModelBackend,
        output_stride: int,
        max_instances: Optional[int] = None,
        max_stride: int = 1,
        anchor_ind: Optional[int] = None,
        centroid_method: Optional[str] = None,
        centroid_fallback: Optional[str] = None,
        use_gt_centroids: bool = False,
        preprocess_config: Optional[PreprocessConfig] = None,
        postprocess_config: Optional[PostprocessConfig] = None,
    ) -> None:
        """Compose the layer with default empty configs when omitted."""
        super().__init__(
            backend=backend,
            preprocess_config=preprocess_config or PreprocessConfig(),
            postprocess_config=postprocess_config
            or PostprocessConfig(
                max_instances=max_instances,
            ),
            output_stride=output_stride,
            max_stride=max_stride,
        )
        self.max_instances = max_instances
        self.anchor_ind = anchor_ind
        self.centroid_method, self.centroid_fallback = degrade_anchor_if_unresolved(
            centroid_method
            or ("anchor" if anchor_ind is not None else "center_of_mass"),
            centroid_fallback,
            anchor_ind,
        )
        self.use_gt_centroids = use_gt_centroids

    # ──────────────────────────────────────────────────────────────────
    # predict(): override to handle the use_gt_centroids branch
    # ──────────────────────────────────────────────────────────────────

    def predict(
        self,
        image: ImageInput,
        instances: Optional[torch.Tensor] = None,
    ) -> Outputs:
        """Run centroid prediction on ``image``.

        Args:
            image: ``np.ndarray`` or ``torch.Tensor`` in any of the shapes
                accepted by :meth:`InferenceLayer._to_4d_float_tensor`.
            instances: ``(B, max_instances, n_nodes, 2)`` GT instance
                keypoints. Required when ``use_gt_centroids=True``;
                ignored otherwise.

        Returns:
            ``Outputs`` populated with ``pred_centroids`` and
            ``pred_centroid_values`` (and optionally ``pred_confmaps`` if
            the postprocess config asks for it).
        """
        if self.use_gt_centroids:
            if instances is None:
                raise ValueError(
                    "use_gt_centroids=True requires `instances` to be passed "
                    "(typically from a LabelsReader's ground-truth field)."
                )
            return self._predict_from_gt(image, instances)
        return super().predict(image)

    def _predict_from_gt(self, image: ImageInput, instances: torch.Tensor) -> Outputs:
        """Compute centroids from GT instances, no model forward.

        Mirrors the legacy ``CentroidCrop(use_gt_centroids=True)`` branch:
        ``generate_centroids`` reduces ``(B, max_inst, n_nodes, 2)`` GT
        keypoints to ``(B, max_inst, 2)`` centroids. NaN-padded instance
        slots stay NaN; corresponding centroid_values are NaN-masked.
        Truncated/padded to ``self.max_instances`` if set.
        """
        x = self._to_4d_float_tensor(image)
        B = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]

        centroids = generate_centroids(
            instances,
            anchor_ind=self.anchor_ind,
            method=self.centroid_method,
            fallback=self.centroid_fallback,
        )
        # ``centroids`` shape: ``(B, max_inst, 2)`` (3D — same rank as
        # ``Outputs.pred_centroids``).
        device = centroids.device
        n_valid = centroids.shape[1]

        # Confidence = 1.0 where centroid is valid, NaN where padded.
        nan_mask = torch.isnan(centroids).any(dim=-1)  # (B, max_inst)
        centroid_vals = torch.where(
            nan_mask,
            torch.full((B, n_valid), float("nan"), device=device),
            torch.ones((B, n_valid), device=device),
        )

        # Honor max_instances cap; pad-with-NaN or truncate to it.
        # Prefer predict-time override (postprocess_config) over build-time value.
        max_inst = (
            getattr(self.postprocess_config, "max_instances", None)
            or self.max_instances
            or n_valid
        )
        if max_inst > n_valid:
            pad_n = max_inst - n_valid
            centroids = torch.cat(
                [
                    centroids,
                    torch.full((B, pad_n, 2), float("nan"), device=device),
                ],
                dim=1,
            )
            centroid_vals = torch.cat(
                [
                    centroid_vals,
                    torch.full((B, pad_n), float("nan"), device=device),
                ],
                dim=1,
            )
        elif max_inst < n_valid:
            centroids = centroids[:, :max_inst]
            centroid_vals = centroid_vals[:, :max_inst]

        info = PreprocInfo(
            original_size=(H, W),
            processed_size=(H, W),
            eff_scale=torch.ones(B, device=device),
            input_scale=1.0,
            output_stride=1,
        )
        return Outputs(
            pred_centroids=centroids,
            pred_centroid_values=centroid_vals,
            preprocess_info=info,
        )

    # ──────────────────────────────────────────────────────────────────
    # postprocess(): find_local_peaks + coord ladder + topk + NaN pad
    # ──────────────────────────────────────────────────────────────────

    def postprocess(self, raw_out: dict, info: PreprocInfo) -> Outputs:
        """Decode confmaps → centroids; coord-unscale; topk + NaN-pad.

        Mirrors the legacy ``CentroidCrop.forward()`` shape contract:
        returns ``(B, max_instances, 2)`` centroids and ``(B, max_instances)``
        values, NaN-padded where no detection.
        """
        # Always the torch decode path: this layer is only built with a
        # ``TorchBackend``; the exported path uses ``ExportedCentroidLayer``
        # (so no double coord ladder, #584).
        confmaps = self._extract_confmaps(raw_out)
        peaks, peak_vals, sample_inds, _channel_inds = find_local_peaks(
            confmaps.detach(),
            threshold=self.postprocess_config.peak_threshold,
            refinement=self.postprocess_config.effective_refinement,
            integral_patch_size=self.postprocess_config.integral_patch_size,
        )

        # Coord ladder: confmap → input pixels → original-image pixels.
        peaks = undo_stride(peaks, info.output_stride)
        peaks = undo_input_scale(peaks, info.input_scale)

        # Batch size from confmaps shape (always available on the torch path).
        B = int(confmaps.shape[0])

        max_instances = (
            getattr(self.postprocess_config, "max_instances", None)
            or self.max_instances
            or self._infer_max_instances(sample_inds)
        )
        if max_instances == 0:
            max_instances = 1  # always emit at least one slot for shape stability

        padded_peaks, padded_vals = pad_peaks_per_sample(
            peaks, peak_vals, sample_inds, B, max_instances
        )

        # Reverse the per-sample sizematcher last (matches legacy ordering).
        padded_peaks = undo_eff_scale(padded_peaks, info.eff_scale)

        outputs = Outputs(
            pred_centroids=padded_peaks,
            pred_centroid_values=padded_vals,
            preprocess_info=info,
        )
        if self.postprocess_config.return_confmaps and confmaps is not None:
            outputs = attrs.evolve(outputs, pred_confmaps=confmaps.detach())
        return outputs

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_max_instances(sample_inds: torch.Tensor) -> int:
        """Find the busiest sample's peak count."""
        if sample_inds.numel() == 0:
            return 0
        counts = torch.bincount(sample_inds.long())
        return int(counts.max().item())


def pad_peaks_per_sample(
    peaks: torch.Tensor,
    peak_vals: torch.Tensor,
    sample_inds: torch.Tensor,
    batch_size: int,
    max_instances: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group a batch's peaks by sample into NaN-padded ``(B, K)`` slots.

    A sample with at most ``max_instances`` (``K``) peaks keeps them in the
    order they come in; a sample with more keeps its ``K`` highest, highest
    first (what ``torch.topk`` returns). Peaks with exactly equal values
    keep their incoming order, where ``torch.topk`` leaves it unspecified.

    The whole batch is done in a few tensor ops on the peaks' device. A loop
    over samples (``peaks[sample_inds == b]``, then ``topk``) gives the same
    result but makes the CPU wait for the GPU several times per sample.

    Args:
        peaks: ``(n_peaks, 2)`` peak coordinates.
        peak_vals: ``(n_peaks,)`` peak values.
        sample_inds: ``(n_peaks,)`` sample index of each peak.
        batch_size: Number of samples ``B``.
        max_instances: Slots per sample ``K``.

    Returns:
        ``(B, K, 2)`` peaks and ``(B, K)`` values (float32), NaN where a
        sample has fewer than ``K`` peaks.
    """
    device = peaks.device
    B, K = batch_size, max_instances
    # One spare column: peaks past the K-th of their sample are written there
    # and dropped, so the number of kept peaks never has to be read back.
    padded_peaks = torch.full((B, K + 1, 2), float("nan"), device=device)
    padded_vals = torch.full((B, K + 1), float("nan"), device=device)
    n = sample_inds.shape[0]
    if n == 0:
        return padded_peaks[:, :K].contiguous(), padded_vals[:, :K].contiguous()

    sample_inds = sample_inds.long()
    counts = torch.bincount(sample_inds, minlength=B)
    starts = torch.cumsum(counts, 0) - counts
    arange = torch.arange(n, device=device)

    # Each peak's position among its sample's peaks, in incoming order.
    by_sample = torch.argsort(sample_inds, stable=True)
    pos = torch.empty_like(arange)
    pos[by_sample] = arange - starts[sample_inds[by_sample]]

    # Each peak's rank by value within its sample, highest first.
    by_val = torch.argsort(-peak_vals, stable=True)
    by_val = by_val[torch.argsort(sample_inds[by_val], stable=True)]
    rank = torch.empty_like(arange)
    rank[by_val] = arange - starts[sample_inds[by_val]]

    slot = torch.where(counts[sample_inds] > K, rank, pos)
    slot = slot.clamp(max=K)
    padded_peaks[sample_inds, slot] = peaks.to(padded_peaks.dtype)
    padded_vals[sample_inds, slot] = peak_vals.to(padded_vals.dtype)
    return padded_peaks[:, :K].contiguous(), padded_vals[:, :K].contiguous()
