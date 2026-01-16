"""Inference modules for BottomUp models."""

import logging
from typing import Dict, Optional
import torch
import lightning as L
from sleap_nn.inference.peak_finding import find_local_peaks
from sleap_nn.inference.paf_grouping import PAFScorer
from sleap_nn.inference.identity import classify_peaks_from_maps

logger = logging.getLogger(__name__)


class BottomUpInferenceModel(L.LightningModule):
    """BottomUp Inference model.

    This model encapsulates the bottom-up approach. The images are passed to a peak detector
    to get the predicted instances and then fed into PAF to combine nodes belonging to
    the same instance.

    Attributes:
        torch_model: A `nn.Module` that accepts rank-5 images as input and predicts
            rank-4 confidence maps as output. This should be a model that is trained on
            MultiInstanceConfMaps.
        paf_scorer: A `sleap_nn.inference.paf_grouping.PAFScorer` instance configured to group
            instances based on peaks and PAFs produced by the model.
        cms_output_stride: Output stride of the model, denoting the scale of the output
            confidence maps relative to the images (after input scaling). This is used
            for adjusting the peak coordinates to the image grid.
        pafs_output_stride: Output stride of the model, denoting the scale of the output
            pafs relative to the images (after input scaling). This is used
            for adjusting the peak coordinates to the image grid.
        peak_threshold: Minimum confidence map value to consider a global peak as valid.
        refinement: If `None`, returns the grid-aligned peaks with no refinement. If
            `"integral"`, peaks will be refined with integral regression. If `"local"`,
            peaks will be refined with quarter pixel local gradient offset. This has no
            effect if the model has an offset regression head.
        integral_patch_size: Size of patches to crop around each rough peak for integral
            refinement as an integer scalar.
        return_confmaps: If `True`, the confidence maps will be returned together with
            the predicted peaks. This will result in slower inference times since
            the data must be copied off of the GPU, but is useful for visualizing the
            raw output of the model.
        return_pafs: If `True`, the part affinity fields will be returned together with
            the predicted instances. This will result in slower inference times since
            the data must be copied off of the GPU, but is useful for visualizing the
            raw output of the model.
        return_paf_graph: If `True`, the part affinity field graph will be returned
            together with the predicted instances. The graph is obtained by parsing the
            part affinity fields with the `paf_scorer` instance and is an intermediate
            representation used during instance grouping.
        input_scale: Float indicating if the images should be resized before being
            passed to the model.
    """

    def __init__(
        self,
        torch_model: L.LightningModule,
        paf_scorer: PAFScorer,
        cms_output_stride: Optional[int] = None,
        pafs_output_stride: Optional[int] = None,
        peak_threshold: float = 0.0,
        refinement: Optional[str] = "integral",
        integral_patch_size: int = 5,
        return_confmaps: Optional[bool] = False,
        return_pafs: Optional[bool] = False,
        return_paf_graph: Optional[bool] = False,
        input_scale: float = 1.0,
        max_peaks_per_node: Optional[int] = None,
    ):
        """Initialise the model attributes.

        Args:
            torch_model: A `nn.Module` that accepts images and predicts confidence maps.
            paf_scorer: A `PAFScorer` instance for grouping instances.
            cms_output_stride: Output stride of confidence maps relative to images.
            pafs_output_stride: Output stride of PAFs relative to images.
            peak_threshold: Minimum confidence map value for valid peaks.
            refinement: Peak refinement method: None, "integral", or "local".
            integral_patch_size: Size of patches for integral refinement.
            return_confmaps: If True, return confidence maps in output.
            return_pafs: If True, return PAFs in output.
            return_paf_graph: If True, return intermediate PAF graph in output.
            input_scale: Scale factor applied to input images.
            max_peaks_per_node: Maximum number of peaks allowed per node before
                skipping PAF scoring. If any node has more peaks than this limit,
                empty predictions are returned. This prevents combinatorial explosion
                during early training when confidence maps are noisy. Set to None to
                disable this check (default). Recommended value: 100.
        """
        super().__init__()
        self.torch_model = torch_model
        self.paf_scorer = paf_scorer
        self.peak_threshold = peak_threshold
        self.refinement = refinement
        self.integral_patch_size = integral_patch_size
        self.cms_output_stride = cms_output_stride
        self.pafs_output_stride = pafs_output_stride
        self.return_confmaps = return_confmaps
        self.return_pafs = return_pafs
        self.return_paf_graph = return_paf_graph
        self.input_scale = input_scale
        self.max_peaks_per_node = max_peaks_per_node

    def _generate_cms_peaks(self, cms):
        # TODO: append nans to batch them -> tensor (vectorize the initial paf grouping steps)
        peaks, peak_vals, sample_inds, peak_channel_inds = find_local_peaks(
            cms.detach(),
            threshold=self.peak_threshold,
            refinement=self.refinement,
            integral_patch_size=self.integral_patch_size,
        )
        # Adjust for stride and scale.
        peaks = peaks * self.cms_output_stride  # (n_centroids, 2)

        cms_peaks, cms_peak_vals, cms_peak_channel_inds = [], [], []

        for b in range(self.batch_size):
            cms_peaks.append(peaks[sample_inds == b])
            cms_peak_vals.append(peak_vals[sample_inds == b].to(torch.float32))
            cms_peak_channel_inds.append(peak_channel_inds[sample_inds == b])

        # cms_peaks: [(#nodes, 2), ...]
        return cms_peaks, cms_peak_vals, cms_peak_channel_inds

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict confidence maps and infer peak coordinates.

        Args:
            inputs: Dictionary with "image" as one of the keys.

        Returns:
            A dictionary of outputs with keys:

            `"pred_instance_peaks"`: The predicted peaks for each instance in the batch
                as a `torch.Tensor` of shape `(samples, nodes, 2)`.
            `"pred_peak_vals"`: The value of the confidence maps at the predicted
                peaks for each instance in the batch as a `torch.Tensor` of shape
                `(samples, nodes)`.

        """
        # Network forward pass.
        self.batch_size = inputs["image"].shape[0]
        output = self.torch_model(inputs["image"])
        cms = output["MultiInstanceConfmapsHead"]
        pafs = output["PartAffinityFieldsHead"].permute(
            0, 2, 3, 1
        )  # (batch, h, w, 2*edges)
        cms_peaks, cms_peak_vals, cms_peak_channel_inds = self._generate_cms_peaks(cms)

        # Check if too many peaks per node (prevents combinatorial explosion)
        skip_paf_scoring = False
        if self.max_peaks_per_node is not None:
            n_nodes = cms.shape[1]
            for b in range(self.batch_size):
                for node_idx in range(n_nodes):
                    n_peaks = int((cms_peak_channel_inds[b] == node_idx).sum().item())
                    if n_peaks > self.max_peaks_per_node:
                        logger.warning(
                            f"Skipping PAF scoring: node {node_idx} has {n_peaks} peaks "
                            f"(max_peaks_per_node={self.max_peaks_per_node}). "
                            f"Model may need more training."
                        )
                        skip_paf_scoring = True
                        break
                if skip_paf_scoring:
                    break

        if skip_paf_scoring:
            # Return empty predictions for each sample
            device = cms.device
            n_nodes = cms.shape[1]
            predicted_instances_adjusted = []
            predicted_peak_scores = []
            predicted_instance_scores = []
            for _ in range(self.batch_size):
                predicted_instances_adjusted.append(
                    torch.full((0, n_nodes, 2), float("nan"), device=device)
                )
                predicted_peak_scores.append(
                    torch.full((0, n_nodes), float("nan"), device=device)
                )
                predicted_instance_scores.append(torch.tensor([], device=device))
            edge_inds = [
                torch.tensor([], dtype=torch.int32, device=device)
            ] * self.batch_size
            edge_peak_inds = [
                torch.tensor([], dtype=torch.int32, device=device).reshape(0, 2)
            ] * self.batch_size
            line_scores = [torch.tensor([], device=device)] * self.batch_size
        else:
            (
                predicted_instances,
                predicted_peak_scores,
                predicted_instance_scores,
                edge_inds,
                edge_peak_inds,
                line_scores,
            ) = self.paf_scorer.predict(
                pafs=pafs,
                peaks=cms_peaks,
                peak_vals=cms_peak_vals,
                peak_channel_inds=cms_peak_channel_inds,
            )

            predicted_instances = [p / self.input_scale for p in predicted_instances]
            predicted_instances_adjusted = []
            for idx, p in enumerate(predicted_instances):
                predicted_instances_adjusted.append(
                    p / inputs["eff_scale"][idx].to(p.device)
                )

        out = {
            "pred_instance_peaks": predicted_instances_adjusted,
            "pred_peak_values": predicted_peak_scores,
            "instance_scores": predicted_instance_scores,
        }

        if self.return_confmaps:
            out["pred_confmaps"] = cms.detach()
        if self.return_pafs:
            out["pred_part_affinity_fields"] = pafs.detach()
        if self.return_paf_graph:
            out["peaks"] = cms_peaks
            out["peak_vals"] = cms_peak_vals
            out["peak_channel_inds"] = cms_peak_channel_inds
            out["edge_inds"] = edge_inds
            out["edge_peak_inds"] = edge_peak_inds
            out["line_scores"] = line_scores

        inputs.update(out)
        return [inputs]


class BottomUpMultiClassInferenceModel(L.LightningModule):
    """BottomUp Inference model for multi-class models.

    This model encapsulates the bottom-up approach. The images are passed to a local peak detector
    to get the predicted instances and then grouped into instances by their identity
    classifications.

    Attributes:
        torch_model: A `nn.Module` that accepts rank-5 images as input and predicts
            rank-4 confidence maps as output. This should be a model that is trained on
            MultiInstanceConfMaps.
        cms_output_stride: Output stride of the model, denoting the scale of the output
            confidence maps relative to the images (after input scaling). This is used
            for adjusting the peak coordinates to the image grid.
        class_maps_output_stride: Output stride of the model, denoting the scale of the output
            pafs relative to the images (after input scaling). This is used
            for adjusting the peak coordinates to the image grid.
        peak_threshold: Minimum confidence map value to consider a global peak as valid.
        refinement: If `None`, returns the grid-aligned peaks with no refinement. If
            `"integral"`, peaks will be refined with integral regression. If `"local"`,
            peaks will be refined with quarter pixel local gradient offset. This has no
            effect if the model has an offset regression head.
        integral_patch_size: Size of patches to crop around each rough peak for integral
            refinement as an integer scalar.
        return_confmaps: If `True`, the confidence maps will be returned together with
            the predicted peaks. This will result in slower inference times since
            the data must be copied off of the GPU, but is useful for visualizing the
            raw output of the model.
        return_class_maps: If `True`, the class maps will be returned together with
            the predicted instances. This will result in slower inference times since
            the data must be copied off of the GPU, but is useful for visualizing the
            raw output of the model.
        input_scale: Float indicating if the images should be resized before being
            passed to the model.
    """

    def __init__(
        self,
        torch_model: L.LightningModule,
        cms_output_stride: Optional[int] = None,
        class_maps_output_stride: Optional[int] = None,
        peak_threshold: float = 0.0,
        refinement: Optional[str] = "integral",
        integral_patch_size: int = 5,
        return_confmaps: Optional[bool] = False,
        return_class_maps: Optional[bool] = False,
        input_scale: float = 1.0,
    ):
        """Initialise the model attributes."""
        super().__init__()
        self.torch_model = torch_model
        self.peak_threshold = peak_threshold
        self.refinement = refinement
        self.integral_patch_size = integral_patch_size
        self.cms_output_stride = cms_output_stride
        self.class_maps_output_stride = class_maps_output_stride
        self.return_confmaps = return_confmaps
        self.return_class_maps = return_class_maps
        self.input_scale = input_scale

    def _generate_cms_peaks(self, cms):
        # TODO: append nans to batch them -> tensor (vectorize the initial paf grouping steps)
        peaks, peak_vals, sample_inds, peak_channel_inds = find_local_peaks(
            cms.detach(),
            threshold=self.peak_threshold,
            refinement=self.refinement,
            integral_patch_size=self.integral_patch_size,
        )
        # Adjust for stride and scale.
        peaks = peaks * self.cms_output_stride  # (n_centroids, 2)

        return peaks, peak_vals, sample_inds, peak_channel_inds

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict confidence maps and infer peak coordinates.

        Args:
            inputs: Dictionary with "image" as one of the keys.

        Returns:
            A dictionary of outputs with keys:

            `"pred_instance_peaks"`: The predicted peaks for each instance in the batch
                as a `torch.Tensor` of shape `(samples, nodes, 2)`.
            `"pred_peak_vals"`: The value of the confidence maps at the predicted
                peaks for each instance in the batch as a `torch.Tensor` of shape
                `(samples, nodes)`.

        """
        # Network forward pass.
        self.batch_size = inputs["image"].shape[0]
        output = self.torch_model(inputs["image"])
        cms = output["MultiInstanceConfmapsHead"]
        class_maps = output["ClassMapsHead"]  # (batch, n_classes, h, w)
        cms_peaks, cms_peak_vals, cms_peak_sample_inds, cms_peak_channel_inds = (
            self._generate_cms_peaks(cms.detach())
        )

        cms_peaks = cms_peaks / self.class_maps_output_stride
        (
            predicted_instances,
            predicted_peak_scores,
            predicted_instance_scores,
        ) = classify_peaks_from_maps(
            class_maps.detach(),
            cms_peaks,
            cms_peak_vals,
            cms_peak_sample_inds,
            cms_peak_channel_inds,
            n_channels=cms.shape[-3],
        )
        predicted_instances = [
            p * self.class_maps_output_stride for p in predicted_instances
        ]

        # Adjust for input scaling.
        if self.input_scale != 1.0:
            predicted_instances = [p / self.input_scale for p in predicted_instances]

        predicted_instances_adjusted = []
        for idx, p in enumerate(predicted_instances):
            predicted_instances_adjusted.append(
                p / inputs["eff_scale"][idx].to(p.device)
            )
        out = {
            "pred_instance_peaks": predicted_instances_adjusted,
            "pred_peak_values": predicted_peak_scores,
            "instance_scores": predicted_instance_scores,
        }

        if self.return_confmaps:
            out["pred_confmaps"] = cms.detach()
        if self.return_class_maps:
            out["pred_class_maps"] = class_maps.detach()

        inputs.update(out)
        return [inputs]
