"""Inference modules for TopDown centroid and centered-instance models."""

from typing import Dict, Optional, Union
import torch
import lightning as L
import numpy as np
from sleap_nn.data.resizing import (
    resize_image,
    pad_to_stride,
)
from sleap_nn.inference.peak_finding import crop_bboxes
from sleap_nn.data.instance_cropping import make_centered_bboxes
from sleap_nn.inference.peak_finding import find_global_peaks, find_local_peaks


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

    """

    def __init__(
        self,
        torch_model: L.LightningModule,
        output_stride: int = 1,
        peak_threshold: float = 0.0,
        max_instances: Optional[int] = None,
        refinement: Optional[str] = None,
        integral_patch_size: int = 5,
        return_confmaps: bool = False,
        return_crops: bool = False,
        crop_hw: tuple = (160, 160),
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
        self.max_instances = max_instances
        self.return_crops = return_crops
        self.crop_hw = crop_hw
        self.input_scale = input_scale
        self.max_stride = max_stride

    def _generate_crops(self, inputs):
        """Generate Crops from the predicted centroids."""
        crops_dict = []
        for centroid, centroid_val, image, fidx, vidx, sz in zip(
            self.refined_peaks_batched,
            self.peak_vals_batched,
            inputs["image"],
            inputs["frame_idx"],
            inputs["video_idx"],
            inputs["orig_size"],
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
            and (batch, max_instances) repsectively which could then to passed to
            FindInstancePeaksGroundTruth class.
        """
        # Network forward pass.
        orig_image = inputs["image"]
        scaled_image = resize_image(orig_image, self.input_scale)
        if self.max_stride != 1:
            scaled_image = pad_to_stride(scaled_image, self.max_stride)

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

        self.refined_peaks_batched = []
        self.peak_vals_batched = []

        for b in range(batch):
            indices = (peak_sample_inds == b).nonzero()
            # list for predicted centroids and corresponding peak values for current batch.
            current_peaks = refined_peaks[indices].squeeze(dim=-2)
            current_peak_vals = peak_vals[indices].squeeze(dim=-1)
            # Choose top k centroids if max_instances is provided.
            if self.max_instances is not None:
                if len(current_peaks) > self.max_instances:
                    current_peak_vals, indices = torch.topk(
                        current_peak_vals, self.max_instances
                    )
                    current_peaks = current_peaks[indices]
                    num_nans = 0
                else:
                    num_nans = self.max_instances - len(current_peaks)
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
            crops_dict = self._generate_crops(inputs)
            return crops_dict
        else:
            # batch the peaks to pass it to FindInstancePeaksGroundTruth class.
            refined_peaks_with_nans = torch.zeros((batch, self.max_instances, 2))
            peak_vals_with_nans = torch.zeros((batch, self.max_instances))
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
        b, _, max_inst, nodes, _ = batch["instances"].shape
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
                batch_peaks = torch.full((max_inst, nodes, 2), torch.nan).unsqueeze(
                    dim=0
                )
                vals = torch.full((max_inst, nodes), torch.nan).unsqueeze(dim=0)
            else:
                c = counts[i]
                batch_peaks = peaks_list[parsed : parsed + c]
                num_inst = len(batch_peaks)
                vals = torch.ones((num_inst, nodes))
                if c != max_inst:
                    batch_peaks = torch.cat(
                        [
                            batch_peaks,
                            torch.full((abs(max_inst - num_inst), nodes, 2), torch.nan),
                        ]
                    )
                    vals = torch.cat(
                        [vals, torch.full((max_inst - num_inst, nodes), torch.nan)]
                    )
                parsed += c

            if i != 0:
                peaks = torch.cat([peaks, batch_peaks])
                peaks_vals = torch.cat([peaks_vals, vals])
            else:
                peaks = batch_peaks
                peaks_vals = vals

        peaks_output = batch
        peaks_output["pred_instance_peaks"] = peaks
        peaks_output["pred_peak_values"] = peaks_vals

        return peaks_output


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
        input_scale: Float indicating if the images should be resized before being
            passed to the model.
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

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        orig_image = inputs["instance_image"]
        scaled_image = resize_image(orig_image, self.input_scale)
        if self.max_stride != 1:
            scaled_image = pad_to_stride(scaled_image, self.max_stride)

        cms = self.torch_model(scaled_image)

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

        # Build outputs.
        outputs = {"pred_instance_peaks": peak_points, "pred_peak_values": peak_vals}
        if self.return_confmaps:
            outputs["pred_confmaps"] = cms.detach()
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
            or `None`. This layer takes as input the output of the centroid cropper
            (if CentroidCrop not None else the image is cropped with the InstanceCropper module)
            and outputs the detected peaks for the instances within each crop.
    """

    def __init__(
        self,
        centroid_crop: Union[CentroidCrop, None],
        instance_peaks: Union[FindInstancePeaks, FindInstancePeaksGroundTruth],
        **kwargs,
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
        batch_size = batch["video_idx"].shape[0]
        peaks_output = []
        if self.centroid_crop is None:
            batch["centroid_val"] = torch.ones(batch_size)
            if isinstance(self.instance_peaks, FindInstancePeaksGroundTruth):
                if "instances" in batch:
                    peaks_output.append(self.instance_peaks(batch))
                else:
                    raise ValueError(
                        "Ground truth data was not detected... "
                        "Please load both models when predicting on non-ground-truth data."
                    )
            else:
                self.instance_peaks.eval()
                peaks_output.append(self.instance_peaks(batch))

        else:
            self.centroid_crop.eval()
            if isinstance(self.instance_peaks, FindInstancePeaksGroundTruth):
                if "instances" in batch:
                    max_inst = batch["instances"].shape[-3]
                    self.centroid_crop.max_instances = max_inst
                else:
                    raise ValueError(
                        "Ground truth data was not detected... "
                        "Please load both models when predicting on non-ground-truth data."
                    )
            batch = self.centroid_crop(batch)
            if isinstance(self.instance_peaks, FindInstancePeaksGroundTruth):
                peaks_output.append(self.instance_peaks(batch))
            else:
                for i in batch:
                    self.instance_peaks.eval()
                    peaks_output.append(self.instance_peaks(i))
        return peaks_output