"""``CentroidLayer`` — predicts instance centroids from a confmap model.

Single-stage layer used either standalone (centroid-only inference; PR 14
ships the saveable-output path #522) or composed with
:class:`CenteredInstanceLayer` to form :class:`TopDownLayer`.

The ``use_gt_centroids=True`` flag replaces the legacy
``CentroidCrop(use_gt_centroids=True)`` path: instead of running the
centroid model, the layer reads ground-truth centroids directly from a
``LabelsReader`` batch's ``"instances"`` field. Used for top-down
inference when only the centered_instance model is available — see issue
#508 docs and the user-facing comment in
``tests/utils/parity_goldens.py`` for context.

The two GT fallback paths are deliberately kept on different layers:

* ``CentroidLayer.use_gt_centroids=True`` — GT *centroids* feed cropping
  for a real centered_instance model.
* ``CenteredInstanceLayer.use_gt_peaks=True`` — GT *keypoints* fill stage
  2 when only a centroid model is available.

Each one is independently configurable on the layer that owns the role
the GT data plays.
"""

from __future__ import annotations

from typing import Optional, Tuple

import attrs
import torch

from sleap_nn.data.instance_centroids import generate_centroids
from sleap_nn.data.resizing import apply_pad_to_stride, resize_image
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

# Lightning's CentroidConfmapsHead returns a Tensor; TorchBackend wraps it
# under "output". ONNX/TRT wrappers (PR 7) emit baked peak fields.
_TORCH_OUTPUT_KEY = "output"
_HEAD_OUTPUT_KEY = "CentroidConfmapsHead"


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
        anchor_ind: Skeleton-node index to use as the centroid anchor when
            ``use_gt_centroids=True``. ``None`` falls back to the NaN-ignoring
            mean of all visible nodes for each instance.
        use_gt_centroids: When ``True``, skip the model and read centroids
            from a batch's ``"instances"`` field (the LabelsReader path).
        preprocess_config / postprocess_config: Standard knobs.
    """

    def __init__(
        self,
        backend: ModelBackend,
        output_stride: int,
        max_instances: Optional[int] = None,
        max_stride: int = 1,
        anchor_ind: Optional[int] = None,
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
        )
        self.max_instances = max_instances
        self.max_stride = max_stride
        self.anchor_ind = anchor_ind
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
        ``generate_centroids`` produces a ``(B, 1, max_inst, 2)`` tensor,
        which we reshape to the canonical ``Outputs`` shape and pad to
        ``max_instances`` with NaNs.
        """
        x = self._to_4d_float_tensor(image)
        B = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]

        centroids = generate_centroids(instances, anchor_ind=self.anchor_ind)
        # ``generate_centroids`` returns ``(B, 1, max_inst, 2)``; squeeze the
        # sample dim and pad each batch to the requested ``max_instances``.
        device = centroids.device
        centroid_vals = torch.ones(centroids.shape[:-1], device=device)
        peaks_per_b = [c[0] for c in centroids]  # list of (max_inst, 2)
        vals_per_b = [v[0] for v in centroid_vals]  # list of (max_inst,)
        max_instances = (
            self.max_instances
            if self.max_instances is not None
            else int(instances.shape[-3])
        )
        padded_peaks = torch.full((B, max_instances, 2), float("nan"), device=device)
        padded_vals = torch.full((B, max_instances), float("nan"), device=device)
        for b, (peaks_b, vals_b) in enumerate(zip(peaks_per_b, vals_per_b)):
            n = min(peaks_b.shape[0], max_instances)
            padded_peaks[b, :n] = peaks_b[:n]
            padded_vals[b, :n] = vals_b[:n]

        info = PreprocInfo(
            original_size=(H, W),
            processed_size=(H, W),
            eff_scale=torch.ones(B, device=device),
            input_scale=1.0,
            output_stride=1,
        )
        return Outputs(
            pred_centroids=padded_peaks,
            pred_centroid_values=padded_vals,
            preprocess_info=info,
        )

    # ──────────────────────────────────────────────────────────────────
    # preprocess(): scale + max-stride pad
    # ──────────────────────────────────────────────────────────────────

    def preprocess(self, image: ImageInput) -> Tuple[torch.Tensor, PreprocInfo]:
        """Run the full legacy-parity preprocessing chain on a raw frame.

        Delegates to :meth:`InferenceLayer._apply_full_preprocess`:
        ensure_rgb/grayscale → per-sample sizematcher (records eff_scale) →
        input_scale → pad_to_stride → ``n_samples`` wrap. The centroid
        Lightning forward unconditionally does ``torch.squeeze(img, dim=1)``,
        so the layer must hand the backend a 5D tensor.
        """
        x = self._to_4d_tensor(image)
        scaled_5d, eff_scale, orig_hw = self._apply_full_preprocess(
            x, max_stride=self.max_stride, unsqueeze_n_samples=True
        )

        info = PreprocInfo(
            original_size=orig_hw,
            processed_size=tuple(scaled_5d.shape[-2:]),
            eff_scale=eff_scale,
            input_scale=self.preprocess_config.scale,
            output_stride=self.output_stride,
        )
        return scaled_5d, info

    # ──────────────────────────────────────────────────────────────────
    # postprocess(): find_local_peaks + coord ladder + topk + NaN pad
    # ──────────────────────────────────────────────────────────────────

    def postprocess(self, raw_out: dict, info: PreprocInfo) -> Outputs:
        """Decode confmaps → centroids; coord-unscale; topk + NaN-pad.

        Mirrors the legacy ``CentroidCrop.forward()`` shape contract:
        returns ``(B, max_instances, 2)`` centroids and ``(B, max_instances)``
        values, NaN-padded where no detection.
        """
        if self.backend.does_baked_postproc:
            peaks = raw_out["peaks"]
            peak_vals = raw_out["peak_vals"]
            sample_inds = raw_out.get("peak_sample_inds")
            confmaps = raw_out.get("confmaps")
            if sample_inds is None:
                raise KeyError(
                    "baked-postproc backend must return peak_sample_inds for "
                    "the centroid layer (multi-peak per sample)."
                )
        else:
            confmaps = self._extract_confmaps(raw_out)
            refinement = (
                self.postprocess_config.refinement
                if self.postprocess_config.refinement != "none"
                else None
            )
            peaks, peak_vals, sample_inds, _channel_inds = find_local_peaks(
                confmaps.detach(),
                threshold=self.postprocess_config.peak_threshold,
                refinement=refinement,
                integral_patch_size=self.postprocess_config.integral_patch_size,
            )

        # Coord ladder: confmap → input pixels → original-image pixels.
        peaks = undo_stride(peaks, info.output_stride)
        peaks = undo_input_scale(peaks, info.input_scale)

        B = (
            info.processed_size and info.processed_size[0]
        )  # not used; B from sample_inds
        # Determine batch size from confmaps shape (always available on Torch).
        if confmaps is not None:
            B = int(confmaps.shape[0])
        else:
            B = int(sample_inds.max().item()) + 1 if sample_inds.numel() else 1

        max_instances = self.max_instances or self._infer_max_instances(sample_inds)
        if max_instances == 0:
            max_instances = 1  # always emit at least one slot for shape stability

        # Allocate the padded outputs on the same device as the peaks so the
        # scatter below doesn't trip the device check on cuda / mps. Falling
        # back to CPU produces correct results on CPU but silently routes
        # cuda / mps results through CPU (or errors on a downstream scatter).
        device = peaks.device
        padded_peaks = torch.full((B, max_instances, 2), float("nan"), device=device)
        padded_vals = torch.full((B, max_instances), float("nan"), device=device)

        for b in range(B):
            mask = sample_inds == b
            sample_peaks = peaks[mask]  # (n_b, 2)
            sample_vals = peak_vals[mask]  # (n_b,)
            if sample_peaks.numel() == 0:
                continue
            if sample_peaks.shape[0] > max_instances:
                sample_vals, idx = torch.topk(sample_vals, max_instances)
                sample_peaks = sample_peaks[idx]
            n = sample_peaks.shape[0]
            padded_peaks[b, :n] = sample_peaks
            padded_vals[b, :n] = sample_vals

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
    def _extract_confmaps(raw_out: dict) -> torch.Tensor:
        """Pull the confmap tensor out of the backend's dict."""
        if _TORCH_OUTPUT_KEY in raw_out:
            return raw_out[_TORCH_OUTPUT_KEY]
        if _HEAD_OUTPUT_KEY in raw_out:
            return raw_out[_HEAD_OUTPUT_KEY]
        tensors = [v for v in raw_out.values() if isinstance(v, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0]
        raise KeyError(
            f"CentroidLayer.postprocess could not find confmaps in raw_out "
            f"keys={list(raw_out.keys())}; expected {_TORCH_OUTPUT_KEY!r} or "
            f"{_HEAD_OUTPUT_KEY!r}."
        )

    @staticmethod
    def _infer_max_instances(sample_inds: torch.Tensor) -> int:
        """Find the busiest sample's peak count."""
        if sample_inds.numel() == 0:
            return 0
        counts = torch.bincount(sample_inds.long())
        return int(counts.max().item())
