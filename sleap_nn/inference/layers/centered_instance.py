"""``CenteredInstanceLayer`` — predicts keypoints from instance-centered crops.

Single-stage layer that runs a centered-instance model on
per-instance crops and decodes keypoints. Used either standalone
(testing / analysis) or composed with :class:`CentroidLayer` to form
:class:`TopDownLayer`.

The ``use_gt_peaks=True`` flag replaces the legacy
``FindInstancePeaksGroundTruth()`` path: instead of running the
centered-instance model, the layer matches each centroid to its
nearest ground-truth instance and returns the GT keypoints. Used for
top-down inference when only the centroid model is available.

The two GT fallback paths in the new design:

* :attr:`CentroidLayer.use_gt_centroids` — GT *centroids* feed cropping
  for a real centered_instance model.
* :attr:`CenteredInstanceLayer.use_gt_peaks` — GT *keypoints* fill stage
  2 when only a centroid model is available.

Each lives on the layer that owns the role the GT data plays.
"""

from __future__ import annotations

from typing import Optional, Tuple

import attrs
import torch

from sleap_nn.data.resizing import apply_pad_to_stride, resize_image
from sleap_nn.inference.layers.backends.base import ModelBackend
from sleap_nn.inference.layers.base import ImageInput, InferenceLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.ops.coord import undo_eff_scale, undo_input_scale, undo_stride
from sleap_nn.inference.ops.peaks import find_global_peaks
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.preprocess_info import PreprocInfo

# Lightning's CenteredInstanceConfmapsHead returns a Tensor; TorchBackend
# wraps under "output". ONNX/TRT (PR 7) emits baked peak fields.
_TORCH_OUTPUT_KEY = "output"
_HEAD_OUTPUT_KEY = "CenteredInstanceConfmapsHead"


class CenteredInstanceLayer(InferenceLayer):
    """Centered-instance keypoint prediction layer.

    Args:
        backend: Runtime backend wrapping the centered-instance model.
            Required even when ``use_gt_peaks=True`` (the layer keeps the
            backend interface uniform; it just doesn't call it on the GT
            path).
        output_stride: Confmap → input-pixel stride from the head config.
        max_stride: Maximum stride the model requires the input to be
            divisible by. Padding applied bottom-right after the
            preprocess input-scale resize.
        use_gt_peaks: When ``True``, skip the model and return the GT
            keypoints from the nearest matched instance.
        preprocess_config / postprocess_config: Standard knobs.
    """

    def __init__(
        self,
        backend: ModelBackend,
        output_stride: int,
        max_stride: int = 1,
        use_gt_peaks: bool = False,
        preprocess_config: Optional[PreprocessConfig] = None,
        postprocess_config: Optional[PostprocessConfig] = None,
    ) -> None:
        """Compose the layer with default empty configs when omitted."""
        super().__init__(
            backend=backend,
            preprocess_config=preprocess_config or PreprocessConfig(),
            postprocess_config=postprocess_config or PostprocessConfig(),
            output_stride=output_stride,
        )
        self.max_stride = max_stride
        self.use_gt_peaks = use_gt_peaks

    # ──────────────────────────────────────────────────────────────────
    # predict(): model path or GT path
    # ──────────────────────────────────────────────────────────────────

    def predict(
        self,
        crops: ImageInput,
        centroids: Optional[torch.Tensor] = None,
        instances: Optional[torch.Tensor] = None,
    ) -> Outputs:
        """Run keypoint prediction.

        Args:
            crops: Per-instance crops, ``(N, C, cH, cW)`` or any shape
                accepted by :meth:`InferenceLayer._to_4d_float_tensor`.
                Ignored on the GT path (``use_gt_peaks=True``).
            centroids: ``(B, max_inst, 2)`` predicted centroids. Required
                on the GT path so the layer can match each one to its
                nearest GT instance.
            instances: ``(B, max_inst, n_nodes, 2)`` GT keypoints from a
                LabelsReader. Required on the GT path.

        Returns:
            ``Outputs`` populated with ``pred_keypoints`` and
            ``pred_peak_values`` (and optionally ``pred_confmaps``).
        """
        if self.use_gt_peaks:
            if centroids is None or instances is None:
                raise ValueError(
                    "use_gt_peaks=True requires `centroids` and `instances` "
                    "to be passed (the layer matches each centroid to its "
                    "nearest GT instance)."
                )
            return self._predict_from_gt(centroids, instances)
        return super().predict(crops)

    # ──────────────────────────────────────────────────────────────────
    # GT path
    # ──────────────────────────────────────────────────────────────────

    def _predict_from_gt(
        self, centroids: torch.Tensor, instances: torch.Tensor
    ) -> Outputs:
        """Match each centroid to its nearest GT instance; return GT keypoints.

        Mirrors the legacy ``FindInstancePeaksGroundTruth.forward`` matching:
        for each centroid, find the GT instance whose nearest keypoint to
        the centroid is closest, then emit that instance's keypoints.
        """
        # ``centroids``: (B, max_inst, 2) — already in image-space.
        # ``instances``: (B, max_inst, n_nodes, 2) — GT keypoints.
        B, max_inst, _ = centroids.shape
        _B, _, n_nodes, _ = instances.shape

        # Distance from each centroid to each (instance, node) pair, then
        # min over nodes → distance from centroid to its nearest keypoint
        # in each candidate GT instance. Match each centroid to argmin.
        cents = centroids.unsqueeze(2).unsqueeze(3)  # (B, max_inst, 1, 1, 2)
        insts = instances.unsqueeze(1)  # (B, 1, max_inst, n_nodes, 2)
        sq = ((cents - insts) ** 2).sum(dim=-1)  # (B, max_inst, max_inst, n_nodes)
        # NaN keypoints (missing) → infinite distance so they don't win argmin.
        sq = torch.where(torch.isnan(sq), torch.full_like(sq, float("inf")), sq)
        nearest_node_dist = sq.min(dim=-1).values  # (B, max_inst_centroid, max_inst_gt)
        match_idx = nearest_node_dist.argmin(dim=-1)  # (B, max_inst_centroid)

        # Gather matched GT instance keypoints + assign full-confidence values.
        # Allocate b_idx + matched_vals on the centroids' device so the gather
        # + ``torch.where`` below don't trip the device check on cuda / mps.
        device = centroids.device
        b_idx = torch.arange(B, device=device).view(B, 1).expand(B, max_inst)
        matched_kpts = instances[b_idx, match_idx]  # (B, max_inst, n_nodes, 2)
        matched_vals = torch.ones(B, max_inst, n_nodes, device=device)

        # Centroids that were NaN-padded shouldn't pull a real GT instance —
        # mark their matched outputs back as NaN to preserve the "no peak"
        # sentinel through the final Outputs.
        nan_centroid = torch.isnan(centroids).any(dim=-1)  # (B, max_inst)
        matched_kpts = torch.where(
            nan_centroid.unsqueeze(-1).unsqueeze(-1),
            torch.full_like(matched_kpts, float("nan")),
            matched_kpts,
        )
        matched_vals = torch.where(
            nan_centroid,
            torch.full_like(matched_vals, float("nan")),
            matched_vals,
        )

        return Outputs(
            pred_keypoints=matched_kpts,
            pred_peak_values=matched_vals,
            pred_centroids=centroids,
            pred_centroid_values=torch.ones(B, max_inst, device=device),
        )

    # ──────────────────────────────────────────────────────────────────
    # Model path: preprocess + postprocess
    # ──────────────────────────────────────────────────────────────────

    def preprocess(self, image: ImageInput) -> Tuple[torch.Tensor, PreprocInfo]:
        """Resize, pad to stride, and wrap with an n_samples dim.

        Like :class:`CentroidLayer`, the centered-instance Lightning forward
        does ``torch.squeeze(img, dim=1)`` unconditionally; the layer hands
        the backend a 5D tensor that becomes 4D after the squeeze.
        """
        x = self._to_4d_float_tensor(image)
        B, _C, H, W = x.shape

        scaled = (
            resize_image(x, self.preprocess_config.scale)
            if self.preprocess_config.scale != 1.0
            else x
        )
        if self.max_stride != 1:
            scaled = apply_pad_to_stride(scaled, self.max_stride)

        scaled_5d = scaled.unsqueeze(1)

        info = PreprocInfo(
            original_size=(H, W),
            processed_size=tuple(scaled.shape[-2:]),
            eff_scale=torch.ones(B, device=scaled.device),
            input_scale=self.preprocess_config.scale,
            output_stride=self.output_stride,
        )
        return scaled_5d, info

    def postprocess(self, raw_out: dict, info: PreprocInfo) -> Outputs:
        """Decode confmaps → keypoints; un-scale; reshape to canonical shape.

        Centered-instance returns one keypoint set per crop. The Outputs
        canonical shape is ``(B, I=1, N, 2)`` where ``I=1`` because the
        crop is per-instance.
        """
        if self.backend.does_baked_postproc:
            peaks = raw_out["peaks"]
            vals = raw_out["peak_vals"]
            confmaps = raw_out.get("confmaps")
        else:
            confmaps = self._extract_confmaps(raw_out)
            refinement = (
                self.postprocess_config.refinement
                if self.postprocess_config.refinement != "none"
                else None
            )
            peaks, vals = find_global_peaks(
                confmaps.detach(),
                threshold=self.postprocess_config.peak_threshold,
                refinement=refinement,
                integral_patch_size=self.postprocess_config.integral_patch_size,
            )

        peaks = undo_stride(peaks, info.output_stride)
        peaks = undo_input_scale(peaks, info.input_scale)
        peaks = undo_eff_scale(peaks, info.eff_scale)

        peaks_BIN2 = peaks.unsqueeze(1)
        vals_BIN = vals.unsqueeze(1)

        outputs = Outputs(
            pred_keypoints=peaks_BIN2,
            pred_peak_values=vals_BIN,
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
            f"CenteredInstanceLayer.postprocess could not find confmaps in "
            f"raw_out keys={list(raw_out.keys())}; expected "
            f"{_TORCH_OUTPUT_KEY!r} or {_HEAD_OUTPUT_KEY!r}."
        )
