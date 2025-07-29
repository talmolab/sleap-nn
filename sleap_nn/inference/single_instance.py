"""Inference modules for SingleInstance models."""

from typing import Dict, List, Optional, Text
import torch
import lightning as L
from sleap_nn.inference.peak_finding import find_global_peaks


class SingleInstanceInferenceModel(L.LightningModule):
    """Single instance prediction model.

    This model encapsulates the basic single instance approach where it is assumed that
    there is only one instance in the frame. The images are passed to a peak detector
    which is trained to detect all body parts for the instance assuming a single peak
    per body part.

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
    ):
        """Initialise the model attributes."""
        super().__init__()
        self.torch_model = torch_model
        self.peak_threshold = peak_threshold
        self.refinement = refinement
        self.integral_patch_size = integral_patch_size
        self.output_stride = output_stride
        self.return_confmaps = return_confmaps
        self.input_scale = input_scale

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
        cms = self.torch_model(inputs["image"])

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
            inputs["eff_scale"].unsqueeze(dim=1).unsqueeze(dim=2)
        ).to(peak_points.device)

        # Build outputs.
        outputs = {"pred_instance_peaks": peak_points, "pred_peak_values": peak_vals}
        if self.return_confmaps:
            outputs["pred_confmaps"] = cms.detach()
        inputs.update(outputs)
        return [inputs]
