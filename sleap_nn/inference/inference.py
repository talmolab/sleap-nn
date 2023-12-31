"""Inference Pipelines."""

import numpy as np
import sleap_io as sio
from typing import Any, Dict, List, Optional, Union, Iterator, Text
import torch
import attr
import torch.nn as nn
from abc import ABC, abstractmethod
from collections import defaultdict
import lightning as L
from torch.utils.data.dataloader import DataLoader
from omegaconf.dictconfig import DictConfig
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.pipelines import (
    TopdownConfmapsPipeline,
    SingleInstanceConfmapsPipeline,
)
from sleap_nn.inference.peak_finding import find_global_peaks
from sleap_nn.model_trainer import TopDownCenteredInstanceModel

from time import time
from omegaconf import OmegaConf


class Predictor(ABC):
    """Base interface class for predictors."""

    @classmethod
    def from_model_paths(cls, ckpt_paths: Dict[Text, Text], model: Text) -> "Predictor":
        """Create the appropriate `Predictor` subclass from from the ckpt path.

        Args:
            ckpt_paths: Dict with keys as model names and values as paths to the checkpoint file having the trained model weights
            model: Model names. One of "single_instance", "topdown"

        Returns:
            A subclass of `Predictor`.

        See also: `SingleInstancePredictor`, `TopDownPredictor`, `BottomUpPredictor`,
            `MoveNetPredictor`, `TopDownMultiClassPredictor`,
            `BottomUpMultiClassPredictor`.
        """
        # Read configs and find model types.
        # configs={}
        # for p in ckpt_paths:
        #     ckpt = torch.load(p) # ???? should we load the checkpoint everytime
        #     configs[ckpt.config.model_name] = [ckpt.config, ckpt] # checkpoint or checkpoint path

        # ckpt_paths = {"centroid": ckpt_path, "centered": ckpt_path}
        model_names = ckpt_paths.keys()
        model = model.lower()

        if "single_instance" in model:
            pass

        elif "topdown" in model:
            centroid_ckpt_path = None
            confmap_ckpt_path = None
            if "centroid" in model_names:
                pass
            if "centered" in model_names:
                confmap_ckpt_path = ckpt_paths["centered"]

            # create an instance of the TopDown predictor class
            predictor = TopDownPredictor.from_trained_models(
                centroid_ckpt_path=centroid_ckpt_path,
                confmap_ckpt_path=confmap_ckpt_path,
            )

        else:
            raise ValueError(
                f"Could not create predictor from model name:\n{model}"
            )
        predictor.model_path = ckpt_paths
        return predictor

    @classmethod
    @abstractmethod
    def from_trained_models(cls, *args, **kwargs):
        """Function to initialize the Predictor class for certain type of model."""
        pass

    @property
    @abstractmethod
    def data_config(self) -> DictConfig:
        """Function to get the data parameters from the config."""
        pass

    @abstractmethod
    def make_pipeline(self):
        """Function to create the data pipeline."""
        pass

    def _initialize_inference_model(self):
        pass

    def _predict_generator(self) -> Iterator[Dict[str, np.ndarray]]:
        """Create a generator that yields batches of inference results.

        This method handles creating a pipeline object depending on the model type
        for loading the data, as well as looping over the batches and running inference.

        Returns:
            A generator yielding batches predicted results as dictionaries of numpy
            arrays.
        """
        # Initialize data pipeline and inference model if needed.
        self.make_pipeline()
        if self.inference_model is None:
            self._initialize_inference_model()

        def process_batch(ex):
            # Run inference on current batch.
            preds = self.inference_model(ex)

            # Add model outputs to the input data example.
            ex.update(preds)

            # Convert to numpy arrays if not already.
            for k, v in ex.items():
                if isinstance(v, torch.Tensor):
                    ex[k] = ex[k].cpu().numpy()

            return ex

        # Loop over data batches.
        for ex in self.data_pipeline:
            yield process_batch(ex)

    def predict(
        self,
        make_labels: bool = True,
        save_labels: bool = False,
        save_path: str = None,
    ) -> Union[List[Dict[str, np.ndarray]], sio.Labels]:
        """Run inference on a data source.

        Args:
            make_labels: If `True` (the default), returns a `sio.Labels` instance with
                `sio.PredictedInstance`s. If `False`, just return a list of
                dictionaries containing the raw arrays returned by the inference model.
            save_labels: If `True` , saves the labels object in a `.slp` file
            save_path: Path to save the labels file if `save_labels` is True.

        Returns:
            A `sio.Labels` with `sio.PredictedInstance`s if `make_labels` is `True`,
            otherwise a list of dictionaries containing batches of numpy arrays with the
            raw results.
        """
        # Initialize inference loop generator.
        generator = self._predict_generator()

        if make_labels:
            # Create SLEAP data structures from the predictions.
            pred_labels = self._make_labeled_frames_from_generator(generator)
            if save_labels:
                sio.io.slp.write_labels(save_path, pred_labels)
            return pred_labels

        else:
            # Just return the raw results.
            return list(generator)


class CentroidCrop(L.LightningModule):
    """Lightning Module to predict the centroids of instances using a trained centroid model."""

    pass


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
    """

    def __init__(
        self,
        torch_model: L.LightningModule,
        output_stride: Optional[int] = None,
        peak_threshold: float = 0.0,
        refinement: Optional[str] = "integral",
        integral_patch_size: int = 5,
        return_confmaps: Optional[bool] = False,
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

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict confidence maps and infer peak coordinates.

        This layer can be chained with a `CentroidCrop` layer to create a top-down
        inference function from full images.

        Args:
            inputs: Dictionary with
                keys:
                `"instance_image"`: Cropped images in either format above.
                `"centroid"`: (Optional) If provided, will be passed through to the
                    output.
                `"centroid_val"`: (Optional) If provided, will be passed through to the
                    output.

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
        cms = self.torch_model(inputs["instance_image"])

        peak_points, peak_vals = find_global_peaks(
            cms.detach(),
            threshold=self.peak_threshold,
            refinement=self.refinement,
            integral_patch_size=self.integral_patch_size,
        )

        # Adjust for stride and scale.
        peak_points = peak_points * self.output_stride

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
            `None`. This layer takes the full image as input and
            outputs a set of centroids and cropped boxes. If `None`, the centroids are calculated with the provided anchor index using InstanceCentroid module and the centroid vals are set as 1.
        instance_peaks: A instance peak detection layer. This can be either
            `FindInstancePeaks` or `None`. This layer takes as
            input the output of the centroid cropper (if CentroidCrop not None else the image is cropped with the InstanceCropper module) and outputs the detected peaks for
            the instances within each crop.
    """

    def __init__(
        self,
        centroid_crop: Union[CentroidCrop, None],
        instance_peaks: Union[FindInstancePeaks, None],
        **kwargs,
    ):
        """Initialize the class with Inference models."""
        super().__init__()
        self.centroid_crop = centroid_crop
        self.instance_peaks = instance_peaks

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict instances for one batch of images.

        Args:
            batch: This is a dictionary containing the image batch in the `"image"` key. If using a ground
                truth model for centroid cropping, the full
                example from a `TopDownConfMapsPipeline` is required for providing the metadata.

        Returns:
            The predicted instances as a dictionary of tensors with the entries in example along with the below keys:

            `"centroids": (batch_size, 1, 2)`: Instance centroids.
            `"centroid_val": (batch_size, 1)`: Instance centroid confidence
                values.
            `"pred_instance_peaks": (batch_size, n_nodes, 2)`: Instance skeleton
                points.
            `"pred_peak_vals": (batch_size, n_nodes)`: Confidence
                values for the instance skeleton points.
        """
        if self.centroid_crop is None:
            batch["centroid_val"] = torch.ones(batch["instance"].shape[0])

        else:
            pass

        if self.instance_peaks is None:
            if "instance" in batch:
                pass
            else:
                raise ValueError(
                    "Ground truth data was not detected... "
                    "Please load both models when predicting on non-ground-truth data."
                )
        else:
            self.instance_peaks.eval()
            peaks_output = self.instance_peaks(batch)
        return peaks_output


@attr.s(auto_attribs=True)
class TopDownPredictor(Predictor):
    """Top-down multi-instance predictor.

    This high-level class handles initialization, preprocessing and predicting using a trained top-down SLEAP-NN model.

    This should be initialized using the `from_trained_models()` constructor.

    Attributes:
        centroid_config: A Dictionary with the configs used for training the centroid model
        confmap_config: A Dictionary with the configs used for training the centered-instance model
        centroid_model: A LightningModule instance created from the trained weights for centroid model.
        confmap_model: A LightningModule instance created from the trained weights for centered-instance model.

    """

    centroid_config: Optional[DictConfig] = attr.ib(default=None)
    confmap_config: Optional[DictConfig] = attr.ib(default=None)
    centroid_model: Optional[L.LightningModule] = attr.ib(default=None)
    confmap_model: Optional[L.LightningModule] = attr.ib(default=None)

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained models and configuration."""
        self.model_config = OmegaConf.create()

        # Create an instance of CentroidLayer if centroid_config is not None
        if self.centroid_config is None:
            centroid_crop_layer = None
        else:
            self.model_config["centroid"] = self.centroid_config
            self.model_config["data"] = self.centroid_config.inference_config.data
            pass

        # Create an instance of FindInstancePeaks layer if confmap_config is not None
        if self.confmap_config is None:
            pass
        else:
            self.model_config["confmaps"] = self.confmap_config
            self.model_config["data"] = self.confmap_config.inference_config.data
            instance_peaks_layer = FindInstancePeaks(
                torch_model=self.confmap_model,
                peak_threshold=self.confmap_config.inference_config.peak_threshold,
                output_stride=self.confmap_config.inference_config.data.preprocessing.conf_map_gen.output_stride,
                refinement=self.confmap_config.inference_config.integral_refinement,
                integral_patch_size=self.confmap_config.inference_config.integral_patch_size,
                return_confmaps=self.confmap_config.inference_config.return_confmaps,
            )

        # Initialize the inference model with centroid and conf map layers
        self.inference_model = TopDownInferenceModel(
            centroid_crop=centroid_crop_layer, instance_peaks=instance_peaks_layer
        )

    @property
    def data_config(self) -> DictConfig:
        """Returns data config section from the overall config."""
        return self.model_config.data

    @classmethod
    def from_trained_models(
        cls,
        centroid_ckpt_path: Optional[Text] = None,
        confmap_ckpt_path: Optional[Text] = None,
    ) -> "TopDownPredictor":
        """Create predictor from saved models.

        Args:
            centroid_ckpt_path: Path to a centroid ckpt file.
            confmap_ckpt_path: Path to a centroid ckpt file.

        Returns:
            An instance of `TopDownPredictor` with the loaded models.

            One of the two models can be left as `None` to perform inference with ground
            truth data. This will only work with `LabelsReader` as the provider.
        """
        if centroid_ckpt_path is None and confmap_ckpt_path is None:
            raise ValueError(
                "Either the centroid or topdown confidence map model must be provided."
            )

        if centroid_ckpt_path is not None:
            # Load centroid model.
            pass

        else:
            centroid_config = None
            centroid_model = None

        if confmap_ckpt_path is not None:
            # Load confmap model.
            confmap_ckpt = torch.load(confmap_ckpt_path)
            skeleton = confmap_ckpt["skeleton"]
            confmap_config = confmap_ckpt["config"]
            confmap_model = TopDownCenteredInstanceModel.load_from_checkpoint(
                confmap_ckpt_path, config=confmap_config
            )
            confmap_model.to(confmap_config.inference_config.device)
            confmap_model.m_device = confmap_config.inference_config.device

        else:
            confmap_config = None
            confmap_model = None

        # create an instance of TopDownPredictor class
        obj = cls(
            centroid_config=centroid_config,
            centroid_model=centroid_model,
            confmap_config=confmap_config,
            confmap_model=confmap_model,
        )

        obj._initialize_inference_model()
        obj.skeleton = skeleton

        return obj

    def make_pipeline(self) -> TopdownConfmapsPipeline:
        """Make a data loading pipeline.

        Returns:
            DataLoader with the pipeline created with `sleap_nn.pipelines.TopdownConfmapsPipeline`

        Notes:
            This method also creates the class attribute `data_pipeline` and will be
            called automatically when predicting on data from a new source.
        """
        self.pipeline = TopdownConfmapsPipeline(data_config=self.data_config)

        provider = self.data_config.provider
        if provider == "LabelsReader":
            provider = LabelsReader

        labels = sio.load_slp(self.data_config.labels_path)
        self.videos = labels.videos
        provider_pipeline = provider(labels)
        self.pipeline = self.pipeline.make_training_pipeline(
            data_provider=provider_pipeline
        )

        # Remove duplicates.
        self.pipeline = self.pipeline.sharding_filter()

        self.data_pipeline = DataLoader(
            self.pipeline,
            **dict(self.data_config.data_loader),
        )

        return self.data_pipeline

    def _make_labeled_frames_from_generator(
        self,
        generator: Iterator[Dict[str, np.ndarray]],
    ) -> sio.Labels:
        """Create labeled frames from a generator that yields inference results.

        This method converts pure arrays into SLEAP-specific data structures.

        Args:
            generator: A generator that returns dictionaries with inference results.
                This should return dictionaries with keys `"instance_image"`, `"video_idx"`,
                `"frame_idx"`, `"pred_instance_peaks"`, `"pred_peak_values"`, and
                `"centroid_val"`. This can be created using the `_predict_generator()`
                method.

        Returns:
            A `sio.Labels` object with `sio.PredictedInstance`s created from
            arrays returned from the inference result generator.
        """
        preds = defaultdict(list)
        predicted_frames = []

        # Loop through each predicted instance.
        for ex in generator:
            # loop through each sample in a batch
            for (
                video_idx,
                frame_idx,
                bbox,
                pred_instances,
                pred_values,
                instance_score,
            ) in zip(
                ex["video_idx"],
                ex["frame_idx"],
                ex["instance_bbox"],
                ex["pred_instance_peaks"],
                ex["pred_peak_values"],
                ex["centroid_val"],
            ):
                pred_instances = pred_instances + bbox.squeeze(axis=0)[0, :]
                preds[(int(video_idx), int(frame_idx))].append(
                    sio.PredictedInstance.from_numpy(
                        points=pred_instances,
                        skeleton=self.skeleton[0],
                        point_scores=pred_values,
                        instance_score=instance_score,
                    )
                )

        for key, inst in preds.items():
            # Create list of LabeledFrames.
            video_idx, frame_idx = key
            predicted_frames.append(
                sio.LabeledFrame(
                    video=self.videos[video_idx],
                    frame_idx=frame_idx,
                    instances=inst,
                )
            )

        pred_labels = sio.Labels(
            videos=self.videos,
            skeletons=self.skeleton,
            labeled_frames=predicted_frames,
        )
        return pred_labels
