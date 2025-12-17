"""Inference modules for TopDown centroid and centered-instance models."""

from typing import Dict, List, Optional, Union
import torch
import lightning as L
import numpy as np
from sleap_nn.data.resizing import (
    resize_image,
    apply_pad_to_stride,
)
from sleap_nn.inference.peak_finding import crop_bboxes
from sleap_nn.data.instance_centroids import generate_centroids
from sleap_nn.data.instance_cropping import make_centered_bboxes
from sleap_nn.inference.peak_finding import find_global_peaks, find_local_peaks
from sleap_nn.inference.identity import get_class_inds_from_vectors
from loguru import logger
from collections import defaultdict


class CentroidCrop(L.LightningModule):
    """Lightning Module for running inference for a centroid model.

    This layer encapsulates all of the inference operations requires for generating
    predictions from a centroid confidence map model. This includes model forward pass,
    generating crops for cenetered instance model, peak finding, coordinate adjustment
    and cropping.

    Attributes:
        torch_model: A `nn.Module` that accepts rank-5 images as input and predicts
            rank-4 confidence maps as output. This should be a model that is trained on
            centered instance confidence maps.
        max_instances: Max number of instances to consider during centroid predictions.
        output_stride: Output stride of the model, denoting the scale of the output
            confidence maps relative to the images (after input scaling). This is used
            for adjusting the peak coordinates to the image grid.
        peak_threshold: Minimum confidence map value to consider a global peak as valid.
        refinement: If `None`, returns the grid-aligned peaks with no refinement. If
            `"integral"`, peaks will be refined with integral regression. If `"local"`,
            peaks will be refined with quarter pixel local gradient offset. This has no
            effect if the model has an offset regression head.
        integral_patch_size: Size of patches to crop around each rough peak for integral
            refinement as an integer scalar.
        return_confmaps: If `True`, the confidence maps will be returned together with
            the predicted peaks.
        return_crops: If `True`, the output dictionary will also contain `instance_image`
            which is the cropped image of size (batch, 1, chn, crop_height, crop_width).
        crop_hw: Tuple (height, width) representing the crop size.
        input_scale: Float indicating if the images should be resized before being
            passed to the model.
        max_stride: Maximum stride in a model that the images must be divisible by.
            If > 1, this will pad the bottom and right of the images to ensure they meet
            this divisibility criteria. Padding is applied after the scaling specified
            in the `scale` attribute.
        use_gt_centroids: If `True`, then the crops are generated using ground-truth centroids.
            If `False`, then centroids are predicted using a trained centroid model.
        anchor_ind: The index of the node to use as the anchor for the centroid. If not
            provided or if not present in the instance, the midpoint of the bounding box
            is used instead.

    """

    def __init__(
        self,
        torch_model: Optional[L.LightningModule] = None,
        output_stride: int = 1,
        peak_threshold: float = 0.0,
        max_instances: Optional[int] = None,
        refinement: Optional[str] = None,
        integral_patch_size: int = 5,
        return_confmaps: bool = False,
        return_crops: bool = False,
        crop_hw: Optional[List[int]] = None,
        input_scale: float = 1.0,
        max_stride: int = 1,
        use_gt_centroids: bool = False,
        anchor_ind: Optional[int] = None,
        **kwargs,
    ):
        """Initialise the model attributes."""
        super().__init__(**kwargs)
        self.torch_model = torch_model
        self.peak_threshold = peak_threshold
        self.refinement = refinement
        self.integral_patch_size = integral_patch_size
        self.output_stride = output_stride
        self.return_confmaps = return_confmaps
        self.max_instances = max_instances
        self.return_crops = return_crops
        self.crop_hw = crop_hw
        self.input_scale = input_scale
        self.max_stride = max_stride
        self.use_gt_centroids = use_gt_centroids
        self.anchor_ind = anchor_ind

    def _generate_crops(self, inputs, cms: Optional[torch.Tensor] = None):
        """Generate Crops from the predicted centroids."""
        crops_dict = []
        if cms is not None:
            cms = cms.detach()
        for idx, (centroid, centroid_val, image, fidx, vidx, sz, eff_sc) in enumerate(
            zip(
                self.refined_peaks_batched,
                self.peak_vals_batched,
                inputs["image"],
                inputs["frame_idx"],
                inputs["video_idx"],
                inputs["orig_size"],
                inputs["eff_scale"],
            )
        ):
            if torch.any(torch.isnan(centroid)):
                if torch.all(torch.isnan(centroid)):
                    continue
                else:
                    non_nans = ~torch.any(centroid.isnan(), dim=-1)
                    centroid = centroid[non_nans]
                    if len(centroid.shape) == 1:
                        centroid = centroid.unsqueeze(dim=0)
                    centroid_val = centroid_val[non_nans]
            n = centroid.shape[0]
            box_size = (
                self.crop_hw[0],
                self.crop_hw[1],
            )
            instance_bbox = torch.unsqueeze(
                make_centered_bboxes(centroid, box_size[0], box_size[1]), 0
            )  # (1, n, 4, 2)

            # Generate cropped image of shape (n, C, crop_H, crop_W)
            instance_image = crop_bboxes(
                image,
                bboxes=instance_bbox.squeeze(dim=0),
                sample_inds=[0] * n,
            )

            # Access top left point (x,y) of bounding box and subtract this offset from
            # position of nodes.
            point = instance_bbox[0, :, 0]
            centered_centroid = centroid - point

            ex = {}
            ex["image"] = torch.cat([image] * n)
            ex["centroid"] = centered_centroid
            ex["centroid_val"] = centroid_val
            ex["frame_idx"] = torch.Tensor([fidx] * n)
            ex["video_idx"] = torch.Tensor([vidx] * n)
            ex["instance_bbox"] = instance_bbox.squeeze(dim=0).unsqueeze(dim=1)
            ex["instance_image"] = instance_image.unsqueeze(dim=1)
            ex["orig_size"] = torch.cat([torch.Tensor(sz)] * n)
            ex["eff_scale"] = torch.Tensor([eff_sc] * n)
            ex["pred_centroids"] = centroid
            if self.return_confmaps:
                ex["pred_centroid_confmaps"] = torch.cat(
                    [cms[idx].unsqueeze(dim=0)] * n
                )
            crops_dict.append(ex)

        return crops_dict

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict centroid confidence maps and crop around peaks.

        This layer can be chained with a `FindInstancePeaks` layer to create a top-down
        inference function from full images.

        Args:
            inputs: Dictionary with key `"image"`. Other keys will be passed down the pipeline.

        Returns:
            A list of dictionaries (size = batch size) where each dictionary has cropped
            images with key `instance_image` and `centroid_val` batched based on the
            number of centroids predicted for each image in the original batch if
            return_crops is True.
            If return_crops is not True, this module returns the dictionary with
            `centroids` and `centroid_val` keys with shapes (batch, 1, max_instances, 2)
            and (batch, max_instances) respectively which could then to passed to
            FindInstancePeaksGroundTruth class.
        """
        if self.use_gt_centroids:
            batch = inputs["video_idx"].shape[0]
            centroids = generate_centroids(
                inputs["instances"], anchor_ind=self.anchor_ind
            )
            centroid_vals = torch.ones(centroids.shape)[..., 0]
            self.refined_peaks_batched = [x[0] for x in centroids]
            self.peak_vals_batched = [x[0] for x in centroid_vals]

            max_instances = (
                self.max_instances
                if self.max_instances is not None
                else inputs["instances"].shape[-3]
            )

            refined_peaks_with_nans = torch.zeros((batch, max_instances, 2))
            peak_vals_with_nans = torch.zeros((batch, max_instances))
            for ind, (r, p) in enumerate(
                zip(self.refined_peaks_batched, self.peak_vals_batched)
            ):
                refined_peaks_with_nans[ind] = r
                peak_vals_with_nans[ind] = p

            inputs.update(
                {
                    "centroids": refined_peaks_with_nans.unsqueeze(dim=1),
                    "centroid_vals": peak_vals_with_nans,
                }
            )

            if self.return_crops:
                crops_dict = self._generate_crops(inputs)
                return crops_dict
            else:
                return inputs

        # Network forward pass.
        orig_image = inputs["image"]
        scaled_image = resize_image(orig_image, self.input_scale)
        if self.max_stride != 1:
            scaled_image = apply_pad_to_stride(scaled_image, self.max_stride)

        cms = self.torch_model(scaled_image)

        refined_peaks, peak_vals, peak_sample_inds, _ = find_local_peaks(
            cms.detach(),
            threshold=self.peak_threshold,
            refinement=self.refinement,
            integral_patch_size=self.integral_patch_size,
        )
        # Adjust for stride and scale.
        refined_peaks = refined_peaks * self.output_stride  # (n_centroids, 2)
        refined_peaks = refined_peaks / self.input_scale

        batch = cms.shape[0]

        # if max instances is not provided, find the max_instances for this batch
        num_instances = defaultdict(int)
        for p in peak_sample_inds:
            num_instances[int(p)] += 1

        if num_instances:
            max_instances = max(num_instances.values()) if num_instances else None
            if self.max_instances is not None:
                max_instances = self.max_instances

            self.refined_peaks_batched = []
            self.peak_vals_batched = []

            for b in range(batch):
                indices = (peak_sample_inds == b).nonzero()
                # list for predicted centroids and corresponding peak values for current batch.
                current_peaks = refined_peaks[indices].squeeze(dim=-2)
                current_peak_vals = peak_vals[indices].squeeze(dim=-1)
                # Choose top k centroids if max_instances is provided.
                if len(current_peaks) > max_instances:
                    current_peak_vals, indices = torch.topk(
                        current_peak_vals, max_instances
                    )
                    current_peaks = current_peaks[indices]
                    num_nans = 0
                else:
                    num_nans = max_instances - len(current_peaks)
                nans = torch.full((num_nans, 2), torch.nan)
                current_peaks = torch.cat(
                    [current_peaks, nans.to(current_peaks.device)], dim=0
                )
                nans = torch.full((num_nans,), torch.nan)
                current_peak_vals = torch.cat(
                    [current_peak_vals, nans.to(current_peak_vals.device)], dim=0
                )
                self.refined_peaks_batched.append(current_peaks)
                self.peak_vals_batched.append(current_peak_vals)

            # Generate crops if return_crops=True to pass the crops to CenteredInstance model.
            if self.return_crops:
                inputs.update(
                    {
                        "centroids": self.refined_peaks_batched,
                        "centroid_vals": self.peak_vals_batched,
                    }
                )
                crops_dict = self._generate_crops(inputs, cms)
                return crops_dict
            else:
                # batch the peaks to pass it to FindInstancePeaksGroundTruth class.
                refined_peaks_with_nans = torch.zeros((batch, max_instances, 2))
                peak_vals_with_nans = torch.zeros((batch, max_instances))
                for ind, (r, p) in enumerate(
                    zip(self.refined_peaks_batched, self.peak_vals_batched)
                ):
                    refined_peaks_with_nans[ind] = r
                    peak_vals_with_nans[ind] = p
                refined_peaks_with_nans = refined_peaks_with_nans / (
                    inputs["eff_scale"]
                    .unsqueeze(dim=1)
                    .unsqueeze(dim=2)
                    .to(refined_peaks_with_nans.device)
                )
                inputs.update(
                    {
                        "centroids": refined_peaks_with_nans.unsqueeze(dim=1),
                        "centroid_vals": peak_vals_with_nans,
                    }
                )
                if self.return_confmaps:
                    inputs.update(
                        {
                            "pred_centroid_confmaps": cms.detach(),
                        }
                    )

                return inputs

        else:
            # if there are no peak detections
            max_instances = 1 if self.max_instances is None else self.max_instances
            if self.return_crops:
                return None
            refined_peaks_with_nans = torch.zeros((batch, max_instances, 2))
            peak_vals_with_nans = torch.zeros((batch, max_instances))
            for b in range(batch):
                refined_peaks_with_nans[b] = torch.full((1, 2), torch.nan)
                peak_vals_with_nans[b] = torch.nan

            inputs.update(
                {
                    "centroids": refined_peaks_with_nans.unsqueeze(dim=1),
                    "centroid_vals": peak_vals_with_nans,
                }
            )
            if self.return_confmaps:
                inputs.update(
                    {
                        "pred_centroid_confmaps": cms.detach(),
                    }
                )
            return inputs


class FindInstancePeaksGroundTruth(L.LightningModule):
    """LightningModule that simulates a centered instance peaks model.

    This layer is useful for testing and evaluating centroid models.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """Initialise the model attributes."""
        super().__init__(**kwargs)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, np.array]:
        """Return the ground truth instance peaks given a set of crops."""
        b, _, _, nodes, _ = batch["instances"].shape
        # Use number of centroids as max_inst to ensure consistent output shape
        # This handles the case where max_instances limits centroids but instances
        # tensor has a different (global) max_instances from the labels file
        num_centroids = batch["centroids"].shape[2]
        inst = (
            batch["instances"].unsqueeze(dim=-4).float()
        )  # (batch, 1, 1, n_inst, nodes, 2)
        cent = (
            batch["centroids"].unsqueeze(dim=-2).unsqueeze(dim=-3).float()
        )  # (batch, 1, n_centroids, 1, 1, 2)
        dists = torch.sum(
            (inst - cent) ** 2, dim=-1
        )  # (batch, 1, n_centroids, n_inst, nodes)
        dists = torch.sqrt(dists)

        dists = torch.where(torch.isnan(dists), torch.inf, dists)
        dists = torch.min(dists, dim=-1).values  # (batch, 1, n_centroids, n_inst)

        # find nearest gt instance
        matches = torch.argmin(dists, dim=-1)  # (batch, 1, n_centroids)

        # filter matches without NaNs (nans have been converted to inf)
        subs = torch.argwhere(
            ~torch.all(dists == torch.inf, dim=-1)
        )  # each element represents an index with (batch, 1, n_centroids)
        valid_matches = matches[subs[:, 0], 0, subs[:, 2]]
        matched_batch_inds = subs[:, 0]

        counts = torch.bincount(matched_batch_inds.detach())
        peaks_list = batch["instances"][matched_batch_inds, 0, valid_matches, :, :]
        parsed = 0
        for i in range(b):
            if i not in matched_batch_inds:
                batch_peaks = torch.full((num_centroids, nodes, 2), torch.nan)
                vals = torch.full((num_centroids, nodes), torch.nan)
            else:
                c = counts[i]
                batch_peaks = peaks_list[parsed : parsed + c]
                num_inst = len(batch_peaks)
                vals = torch.ones((num_inst, nodes))
                if c < num_centroids:
                    batch_peaks = torch.cat(
                        [
                            batch_peaks,
                            torch.full((num_centroids - num_inst, nodes, 2), torch.nan),
                        ]
                    )
                    vals = torch.cat(
                        [vals, torch.full((num_centroids - num_inst, nodes), torch.nan)]
                    )
                else:
                    batch_peaks = batch_peaks[:num_centroids]
                    vals = vals[:num_centroids]
                parsed += c

            batch_peaks = batch_peaks.unsqueeze(dim=0)

            if i != 0:
                peaks = torch.cat([peaks, batch_peaks])
                peaks_vals = torch.cat([peaks_vals, vals])
            else:
                peaks = batch_peaks
                peaks_vals = vals

        peaks_output = batch
        if peaks.size(0) != 0:
            peaks = peaks / (
                batch["eff_scale"]
                .unsqueeze(dim=1)
                .unsqueeze(dim=2)
                .unsqueeze(dim=3)
                .to(peaks.device)
            )
        peaks_output["pred_instance_peaks"] = peaks
        peaks_output["pred_peak_values"] = peaks_vals

        batch_size = batch["centroids"].shape[0]
        output_dict = {}
        output_dict["centroid"] = batch["centroids"].squeeze(dim=1).reshape(-1, 1, 2)
        output_dict["centroid_val"] = batch["centroid_vals"].reshape(-1)
        output_dict["pred_instance_peaks"] = peaks_output[
            "pred_instance_peaks"
        ].reshape(-1, nodes, 2)
        output_dict["pred_peak_values"] = peaks_output["pred_peak_values"].reshape(
            -1, nodes
        )
        output_dict["instance_bbox"] = torch.zeros(
            (batch_size * num_centroids, 1, 4, 2)
        )
        frame_inds = []
        video_inds = []
        orig_szs = []
        images = []
        centroid_confmaps = []
        for b_idx in range(b):
            curr_batch_size = len(batch["centroids"][b_idx][0])
            frame_inds.extend([batch["frame_idx"][b_idx]] * curr_batch_size)
            video_inds.extend([batch["video_idx"][b_idx]] * curr_batch_size)
            orig_szs.append(torch.cat([batch["orig_size"][b_idx]] * curr_batch_size))
            images.append(
                batch["image"][b_idx].unsqueeze(0).repeat(curr_batch_size, 1, 1, 1, 1)
            )
            if "pred_centroid_confmaps" in batch:
                centroid_confmaps.append(
                    batch["pred_centroid_confmaps"][b_idx]
                    .unsqueeze(0)
                    .repeat(curr_batch_size, 1, 1, 1)
                )

        output_dict["frame_idx"] = torch.tensor(frame_inds)
        output_dict["video_idx"] = torch.tensor(video_inds)
        output_dict["orig_size"] = torch.concatenate(orig_szs, dim=0)
        output_dict["image"] = torch.cat(images, dim=0)
        if centroid_confmaps:
            output_dict["pred_centroid_confmaps"] = torch.cat(centroid_confmaps, dim=0)
        return output_dict


class FindInstancePeaks(L.LightningModule):
    """Lightning Module that predicts instance peaks from images using a trained model.

    This layer encapsulates all of the inference operations required for generating
    predictions from a centered instance confidence map model. This includes
    model forward pass, peak finding and coordinate adjustment.

    Attributes:
        torch_model: A `nn.Module` that accepts rank-5 images as input and predicts
            rank-4 confidence maps as output. This should be a model that is trained on
            centered instance confidence maps.
        output_stride: Output stride of the model, denoting the scale of the output
            confidence maps relative to the images (after input scaling). This is used
            for adjusting the peak coordinates to the image grid.
        peak_threshold: Minimum confidence map value to consider a global peak as valid.
        refinement: If `None`, returns the grid-aligned peaks with no refinement. If
            `"integral"`, peaks will be refined with integral regression. If `"local"`,
            peaks will be refined with quarter pixel local gradient offset. This has no
            effect if the model has an offset regression head.
        integral_patch_size: Size of patches to crop around each rough peak for integral
            refinement as an integer scalar.
        return_confmaps: If `True`, the confidence maps will be returned together with
            the predicted peaks.
        input_scale: Float indicating the scale with which the images were scaled before
            cropping.
        max_stride: Maximum stride in a model that the images must be divisible by.
            If > 1, this will pad the bottom and right of the images to ensure they meet
            this divisibility criteria. Padding is applied after the scaling specified
            in the `scale` attribute.

    """

    def __init__(
        self,
        torch_model: L.LightningModule,
        output_stride: Optional[int] = None,
        peak_threshold: float = 0.0,
        refinement: Optional[str] = None,
        integral_patch_size: int = 5,
        return_confmaps: Optional[bool] = False,
        input_scale: float = 1.0,
        max_stride: int = 1,
        **kwargs,
    ):
        """Initialise the model attributes."""
        super().__init__(**kwargs)
        self.torch_model = torch_model
        self.peak_threshold = peak_threshold
        self.refinement = refinement
        self.integral_patch_size = integral_patch_size
        self.output_stride = output_stride
        self.return_confmaps = return_confmaps
        self.input_scale = input_scale
        self.max_stride = max_stride

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Predict confidence maps and infer peak coordinates.

        This layer can be chained with a `CentroidCrop` layer to create a top-down
        inference function from full images.

        Args:
            inputs: Dictionary with keys:
                `"instance_image"`: Cropped images.
                Other keys will be passed down the pipeline.

        Returns:
            A dictionary of outputs with keys:

            `"pred_instance_peaks"`: The predicted peaks for each instance in the batch as a
                `torch.Tensor` of shape `(samples, nodes, 2)`.
            `"pred_peak_vals"`: The value of the confidence maps at the predicted
                peaks for each instance in the batch as a `torch.Tensor` of shape
                `(samples, nodes)`.

            If provided (e.g., from an input `CentroidCrop` layer), the centroids that
            generated the crops will also be included in the keys `"centroid"` and
            `"centroid_val"`.

        """
        # Network forward pass.
        # resize and pad the input image
        input_image = inputs["instance_image"]
        # resize the crop image
        input_image = resize_image(input_image, self.input_scale)
        if self.max_stride != 1:
            input_image = apply_pad_to_stride(input_image, self.max_stride)

        cms = self.torch_model(input_image)

        peak_points, peak_vals = find_global_peaks(
            cms.detach(),
            threshold=self.peak_threshold,
            refinement=self.refinement,
            integral_patch_size=self.integral_patch_size,
        )

        # Adjust for stride and scale.
        peak_points = peak_points * self.output_stride
        if self.input_scale != 1.0:
            peak_points = peak_points / self.input_scale

        peak_points = peak_points / (
            inputs["eff_scale"].unsqueeze(dim=1).unsqueeze(dim=2).to(peak_points.device)
        )

        inputs["instance_bbox"] = inputs["instance_bbox"] / (
            inputs["eff_scale"]
            .unsqueeze(dim=1)
            .unsqueeze(dim=2)
            .unsqueeze(dim=3)
            .to(inputs["instance_bbox"].device)
        )

        # Build outputs.
        outputs = {"pred_instance_peaks": peak_points, "pred_peak_values": peak_vals}
        if self.return_confmaps:
            outputs["pred_confmaps"] = cms.detach()
        inputs.update(outputs)
        return inputs


class TopDownMultiClassFindInstancePeaks(L.LightningModule):
    """Lightning Module that predicts instance peaks from images using a trained model.

    This layer encapsulates all of the inference operations required for generating
    predictions from a centered instance confidence map model. This includes
    model forward pass, peak finding and coordinate adjustment.

    Attributes:
        torch_model: A `nn.Module` that accepts rank-5 images as input and predicts
            rank-4 confidence maps as output. This should be a model that is trained on
            centered instance confidence maps.
        output_stride: Output stride of the model, denoting the scale of the output
            confidence maps relative to the images (after input scaling). This is used
            for adjusting the peak coordinates to the image grid.
        peak_threshold: Minimum confidence map value to consider a global peak as valid.
        refinement: If `None`, returns the grid-aligned peaks with no refinement. If
            `"integral"`, peaks will be refined with integral regression. If `"local"`,
            peaks will be refined with quarter pixel local gradient offset. This has no
            effect if the model has an offset regression head.
        integral_patch_size: Size of patches to crop around each rough peak for integral
            refinement as an integer scalar.
        return_confmaps: If `True`, the confidence maps will be returned together with
            the predicted peaks.
        return_class_vectors: If `True`, the classification probabilities will be
            returned together with the predicted peaks. This will not line up with the
            grouped instances, for which the associtated class probabilities will always
            be returned in `"instance_scores"`.
        input_scale: Float indicating the scale with which the images were scaled before
            cropping.
        max_stride: Maximum stride in a model that the images must be divisible by.
            If > 1, this will pad the bottom and right of the images to ensure they meet
            this divisibility criteria. Padding is applied after the scaling specified
            in the `scale` attribute.

    """

    def __init__(
        self,
        torch_model: L.LightningModule,
        output_stride: Optional[int] = None,
        peak_threshold: float = 0.0,
        refinement: Optional[str] = "integral",
        integral_patch_size: int = 5,
        return_confmaps: Optional[bool] = False,
        return_class_vectors: bool = False,
        input_scale: float = 1.0,
        max_stride: int = 1,
        **kwargs,
    ):
        """Initialise the model attributes."""
        super().__init__(**kwargs)
        self.torch_model = torch_model
        self.peak_threshold = peak_threshold
        self.refinement = refinement
        self.integral_patch_size = integral_patch_size
        self.output_stride = output_stride
        self.return_confmaps = return_confmaps
        self.return_class_vectors = return_class_vectors
        self.input_scale = input_scale
        self.max_stride = max_stride

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Predict confidence maps and infer peak coordinates.

        This layer can be chained with a `CentroidCrop` layer to create a top-down
        inference function from full images.

        Args:
            inputs: Dictionary with keys:
                `"instance_image"`: Cropped images.
                Other keys will be passed down the pipeline.

        Returns:
            A dictionary of outputs with keys:

            `"pred_instance_peaks"`: The predicted peaks for each instance in the batch as a
                `torch.Tensor` of shape `(samples, nodes, 2)`.
            `"pred_peak_vals"`: The value of the confidence maps at the predicted
                peaks for each instance in the batch as a `torch.Tensor` of shape
                `(samples, nodes)`.

            If provided (e.g., from an input `CentroidCrop` layer), the centroids that
            generated the crops will also be included in the keys `"centroid"` and
            `"centroid_val"`.

        """
        # Network forward pass.
        # resize and pad the input image
        input_image = inputs["instance_image"]
        # resize the crop image
        input_image = resize_image(input_image, self.input_scale)
        if self.max_stride != 1:
            input_image = apply_pad_to_stride(input_image, self.max_stride)

        out = self.torch_model(input_image)
        cms = out["CenteredInstanceConfmapsHead"].detach()
        peak_class_probs = out["ClassVectorsHead"].detach()

        peak_points, peak_vals = find_global_peaks(
            cms,
            threshold=self.peak_threshold,
            refinement=self.refinement,
            integral_patch_size=self.integral_patch_size,
        )

        # Adjust for stride and scale.
        peak_points = peak_points * self.output_stride
        if self.input_scale != 1.0:
            peak_points = peak_points / self.input_scale

        peak_points = peak_points / (
            inputs["eff_scale"].unsqueeze(dim=1).unsqueeze(dim=2).to(peak_points.device)
        )

        inputs["instance_bbox"] = inputs["instance_bbox"] / (
            inputs["eff_scale"]
            .unsqueeze(dim=1)
            .unsqueeze(dim=2)
            .unsqueeze(dim=3)
            .to(inputs["instance_bbox"].device)
        )

        (
            class_inds,
            class_probs,
        ) = get_class_inds_from_vectors(peak_class_probs)

        # Build outputs.
        outputs = {
            "pred_instance_peaks": peak_points,
            "pred_peak_values": peak_vals,
            "instance_scores": class_probs,
            "pred_class_inds": class_inds,
        }

        if self.return_confmaps:
            outputs["pred_confmaps"] = cms
        if self.return_class_vectors:
            outputs["pred_class_vectors"] = peak_class_probs
        inputs.update(outputs)
        return inputs


class TopDownInferenceModel(L.LightningModule):
    """Top-down instance prediction model.

    This model encapsulates the top-down approach where instances are first detected by
    local peak detection of an anchor point and then cropped. These instance-centered
    crops are then passed to an instance peak detector which is trained to detect all
    remaining body parts for the instance that is centered within the crop.

    Attributes:
        centroid_crop: A centroid cropping layer. This can be either `CentroidCrop` or
            `None`. This layer takes the full image as input and outputs a set of centroids
            and cropped boxes. If `None`, the centroids are calculated with the provided anchor index
            using InstanceCentroid module and the centroid vals are set as 1.
        instance_peaks: A instance peak detection layer. This can be either `FindInstancePeaks`
            or `FindInstancePeaksGroundTruth` or `TopDownMultiClassFindInstancePeaks`. This layer takes as input the output of the centroid cropper
            (if CentroidCrop not None else the image is cropped with the InstanceCropper module)
            and outputs the detected peaks for the instances within each crop.
    """

    def __init__(
        self,
        centroid_crop: Union[CentroidCrop, None],
        instance_peaks: Union[
            FindInstancePeaks,
            FindInstancePeaksGroundTruth,
            TopDownMultiClassFindInstancePeaks,
        ],
    ):
        """Initialize the class with Inference models."""
        super().__init__()
        self.centroid_crop = centroid_crop
        self.instance_peaks = instance_peaks

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict instances for one batch of images.

        Args:
            batch: This is a dictionary containing the image batch in the `image` key.
                   If centroid model is not provided, the dictionary should have
                   `instance_image` key.

        Returns:
            The predicted instances as a list of dictionaries of tensors with the
            entries in example along with the below keys:

            `"centroids": (batch_size, 1, 2)`: Instance centroids.
            `"centroid_val": (batch_size, 1)`: Instance centroid confidence
                values.
            `"pred_instance_peaks": (batch_size, n_nodes, 2)`: Instance skeleton
                points.
            `"pred_peak_vals": (batch_size, n_nodes)`: Confidence
                values for the instance skeleton points.
        """
        if isinstance(self.instance_peaks, FindInstancePeaksGroundTruth):
            if "instances" not in batch:
                message = (
                    "Ground truth data was not detected... "
                    "Please load both models when predicting on non-ground-truth data."
                )
                logger.error(message)
                raise ValueError(message)
        self.centroid_crop.eval()
        peaks_output = []
        batch = self.centroid_crop(batch)

        if batch is not None:
            if isinstance(self.instance_peaks, FindInstancePeaksGroundTruth):
                peaks_output.append(self.instance_peaks(batch))
            else:
                for i in batch:
                    self.instance_peaks.eval()
                    peaks_output.append(
                        self.instance_peaks(
                            i,
                        )
                    )
            return peaks_output
        return batch
