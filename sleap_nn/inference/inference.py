"""Inference Pipelines."""

import numpy as np
import sleap_io as sio
from typing import Any, Dict, List, Optional, Union, Iterator, Text
import torch
import attr
import torch.nn as nn
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
import lightning as L
from torch.utils.data.dataloader import DataLoader
from omegaconf.dictconfig import DictConfig
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.pipelines import (
    TopdownConfmapsPipeline,
    SingleInstanceConfmapsPipeline,
)
from sleap_nn.inference.peak_finding import find_global_peaks
from sleap_nn.model_trainer import TopDownCenteredInstanceModel, SingleInstanceModel
from omegaconf import OmegaConf


class Predictor(ABC):
    """Base interface class for predictors."""

    @classmethod
    def from_model_paths(cls, model_paths: List[Text]) -> "Predictor":
        """Create the appropriate `Predictor` subclass from from the ckpt path.

        Args:
            model_paths: List of paths to the directory where the best.ckpt and training_config.yaml are saved.

        Returns:
            A subclass of `Predictor`.

        See also: `SingleInstancePredictor`, `TopDownPredictor`, `BottomUpPredictor`,
            `MoveNetPredictor`, `TopDownMultiClassPredictor`,
            `BottomUpMultiClassPredictor`.
        """
        model_config_paths = [
            OmegaConf.load(f"{Path(c)}/training_config.yaml") for c in model_paths
        ]
        model_names = [
            (c.model_config.head_configs.head_type) for c in model_config_paths
        ]

        if "SingleInstanceConfmapsHead" in model_names:
            confmap_ckpt_path = model_paths[
                model_names.index("SingleInstanceConfmapsHead")
            ]
            predictor = SingleInstancePredictor.from_trained_models(confmap_ckpt_path)

        elif (
            "CentroidConfmapsHead" in model_names
            or "CenteredInstanceConfmapsHead" in model_names
        ):
            centroid_ckpt_path = None
            confmap_ckpt_path = None
            if "CentroidConfmapsHead" in model_names:
                pass
            if "CenteredInstanceConfmapsHead" in model_names:
                confmap_ckpt_path = model_paths[
                    model_names.index("CenteredInstanceConfmapsHead")
                ]

            # create an instance of the TopDown predictor class
            predictor = TopDownPredictor.from_trained_models(
                centroid_ckpt_path=centroid_ckpt_path,
                confmap_ckpt_path=confmap_ckpt_path,
            )
        else:
            raise ValueError(
                f"Could not create predictor from model paths:\n{model_paths}"
            )
        predictor.model_path = model_paths
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
        save_path: str = None,
    ) -> Union[List[Dict[str, np.ndarray]], sio.Labels]:
        """Run inference on a data source.

        Args:
            make_labels: If `True` (the default), returns a `sio.Labels` instance with
                `sio.PredictedInstance`s. If `False`, just return a list of
                dictionaries containing the raw arrays returned by the inference model.
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
            if save_path:
                sio.io.slp.write_labels(save_path, pred_labels)
            return pred_labels

        else:
            # Just return the raw results.
            return list(generator)


class CentroidCrop(L.LightningModule):
    """Lightning Module to predict the centroids of instances using a trained centroid model."""

    pass


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
        # num_inst = batch["num_instances"]
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

        counts = Counter(matched_batch_inds.detach().numpy())
        peaks_list = batch["instances"][matched_batch_inds, 0, valid_matches, :, :]
        parsed = 0
        for i in range(b):
            if i not in counts.keys():
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
                            torch.full((max_inst - num_inst, nodes, 2), torch.nan),
                        ]
                    ).unsqueeze(dim=0)
                    vals = torch.cat(
                        [vals, torch.full((max_inst - num_inst, nodes), torch.nan)]
                    ).unsqueeze(dim=0)
                parsed += c

            if i != 0:
                peaks = torch.cat([peaks, batch_peaks])
                peaks_vals = torch.cat([peaks_vals, vals])
            else:
                peaks = batch_peaks
                peaks_vals = vals

        peaks_output = batch.copy()
        peaks_output["pred_instance_peaks"] = peaks.unsqueeze(dim=1)
        peaks_output["pred_peak_values"] = peaks_vals.unsqueeze(dim=1)

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
            batch: This is a dictionary containing the image batch in the `"image"` key. If using a ground
                truth model for centroid cropping, the full
                example from a `TopDownConfMapsPipeline` is required for providing the metadata.

        Returns:
            The predicted instances as a dictionary of tensors with the entries in example
            along with the below keys:

            `"centroids": (batch_size, 1, 2)`: Instance centroids.
            `"centroid_val": (batch_size, 1)`: Instance centroid confidence
                values.
            `"pred_instance_peaks": (batch_size, n_nodes, 2)`: Instance skeleton
                points.
            `"pred_peak_vals": (batch_size, n_nodes)`: Confidence
                values for the instance skeleton points.
        """
        batch_size = batch["image"].shape[0]
        if self.centroid_crop is None:
            batch["centroid_val"] = torch.ones(batch_size)

        if isinstance(self.instance_peaks, FindInstancePeaksGroundTruth):
            if "instances" in batch:
                peaks_output = self.instance_peaks(batch)
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

    This high-level class handles initialization, preprocessing and predicting using a trained
    TopDown SLEAP-NN model.

    This should be initialized using the `from_trained_models()` constructor.

    Attributes:
        centroid_config: A Dictionary with the configs used for training the centroid model
        confmap_config: A Dictionary with the configs used for training the centered-instance model
        centroid_model: A LightningModule instance created from the trained weights for centroid model.
        confmap_model: A LightningModule instance created from the trained weights
                       for centered-instance model.

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
            self.model_config["skeletons"] = self.centroid_config.data_config.skeletons

        # Create an instance of FindInstancePeaks layer if confmap_config is not None
        if self.confmap_config is None:
            pass
        else:
            self.model_config["confmaps"] = self.confmap_config
            self.model_config["data"] = self.confmap_config.inference_config.data
            self.model_config["data"][
                "skeletons"
            ] = self.confmap_config.data_config.skeletons
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
            centroid_ckpt_path: Path to a centroid ckpt dir with model.ckpt and config.yaml.
            confmap_ckpt_path: Path to a centroid ckpt dir with model.ckpt and config.yaml.

        Returns:
            An instance of `TopDownPredictor` with the loaded models.

            One of the two models can be left as `None` to perform inference with ground
            truth data. This will only work with `LabelsReader` as the provider.
        """
        if centroid_ckpt_path is not None:
            # Load centroid model.
            pass

        else:
            centroid_config = None
            centroid_model = None

        if confmap_ckpt_path is not None:
            # Load confmap model.
            confmap_config = OmegaConf.load(f"{confmap_ckpt_path}/training_config.yaml")
            confmap_model = TopDownCenteredInstanceModel.load_from_checkpoint(
                f"{confmap_ckpt_path}/best.ckpt", config=confmap_config
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
        return obj

    def make_pipeline(self) -> TopdownConfmapsPipeline:
        """Make a data loading pipeline.

        Returns:
            DataLoader with the pipeline created with `sleap_nn.pipelines.TopdownConfmapsPipeline`

        Notes:
            This method also creates the class attribute `data_pipeline` and will be
            called automatically when predicting on data from a new source.
        """
        provider = self.data_config.provider
        if provider == "LabelsReader":
            provider = LabelsReader

        # load slp file
        labels = sio.load_slp(self.data_config.labels_path)
        self.videos = labels.videos

        # create pipeline
        self.pipeline = TopdownConfmapsPipeline(data_config=self.data_config)
        self.pipeline = self.pipeline.make_training_pipeline(
            data_provider=provider(
                labels,
                max_instances=self.data_config.max_instances,
                max_height=self.data_config.max_height,
                max_width=self.data_config.max_width,
                is_rgb=self.data_config.is_rgb,
            )
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

        skeletons = []
        for name in self.data_config.skeletons.keys():
            nodes = [
                sio.model.skeleton.Node(n["name"])
                for n in self.data_config.skeletons[name].nodes
            ]
            edges = [
                sio.model.skeleton.Edge(
                    sio.model.skeleton.Node(e["source"]["name"]),
                    sio.model.skeleton.Node(e["destination"]["name"]),
                )
                for e in self.confmap_config.data_config.skeletons[name].edges
            ]
            if self.data_config.skeletons[name].symmetries:
                list_args = [
                    set(
                        [
                            sio.model.skeleton.Node(s[0]["name"]),
                            sio.model.skeleton.Node(s[1]["name"]),
                        ]
                    )
                    for s in self.data_config.skeletons[name].symmetries
                ]
                symmetries = [sio.model.skeleton.Symmetry(x) for x in list_args]
            else:
                symmetries = []

            skeletons.append(
                sio.model.skeleton.Skeleton(nodes, edges, symmetries, name)
            )

        skeleton_idx = 0
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
                org_size,
            ) in zip(
                ex["video_idx"],
                ex["frame_idx"],
                ex["instance_bbox"],
                ex["pred_instance_peaks"],
                ex["pred_peak_values"],
                ex["centroid_val"],
                ex["orig_size"],
            ):
                pred_instances = (
                        pred_instances
                        + bbox.squeeze(axis=0)[0, :])
                if self.data_config.max_height is not None:
                    pad_height = (self.data_config.max_height - org_size[0]) // 2
                    pad_width = (self.data_config.max_width - org_size[1]) // 2
                    pred_instances = (
                        pred_instances
                        - [pad_height, pad_width]
                    )
                preds[(int(video_idx), int(frame_idx))].append(
                    sio.PredictedInstance.from_numpy(
                        points=pred_instances,
                        skeleton=skeletons[skeleton_idx],
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
            skeletons=skeletons,
            labeled_frames=predicted_frames,
        )
        return pred_labels


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

        Args:
            inputs: Dictionary with "image" as one of the keys.

        Returns:
            A dictionary of outputs with keys:

            `"pred_instance_peaks"`: The predicted peaks for each instance in the batch as a
                `torch.Tensor` of shape `(samples, nodes, 2)`.
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

        # Build outputs.
        outputs = {"pred_instance_peaks": peak_points, "pred_peak_values": peak_vals}
        if self.return_confmaps:
            outputs["pred_confmaps"] = cms.detach()
        inputs.update(outputs)
        return inputs


@attr.s(auto_attribs=True)
class SingleInstancePredictor(Predictor):
    """Single-Instance predictor.

    This high-level class handles initialization, preprocessing and predicting using a
    trained single instance SLEAP-NN model.

    This should be initialized using the `from_trained_models()` constructor.

    Attributes:
        confmap_config: A Dictionary with the configs used for training the single-instance model
        confmap_model: A LightningModule instance created from the trained weights for
                       single-instance model.

    """

    confmap_config: Optional[DictConfig] = attr.ib(default=None)
    confmap_model: Optional[L.LightningModule] = attr.ib(default=None)

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained models and configuration."""
        self.inference_model = SingleInstanceInferenceModel(
            torch_model=self.confmap_model,
            peak_threshold=self.confmap_config.inference_config.peak_threshold,
            output_stride=self.confmap_config.inference_config.data.preprocessing.conf_map_gen.output_stride,
            refinement=self.confmap_config.inference_config.integral_refinement,
            integral_patch_size=self.confmap_config.inference_config.integral_patch_size,
            return_confmaps=self.confmap_config.inference_config.return_confmaps,
        )

    @property
    def data_config(self) -> DictConfig:
        """Returns data config section from the overall config."""
        return self.confmap_config.inference_config.data

    @classmethod
    def from_trained_models(
        cls,
        confmap_ckpt_path: Optional[Text] = None,
    ) -> "TopDownPredictor":
        """Create predictor from saved models.

        Args:
            confmap_ckpt_path: Path to a centroid ckpt dir with model.ckpt and config.yaml.

        Returns:
            An instance of `SingleInstancePredictor` with the loaded models.

        """
        confmap_config = OmegaConf.load(f"{confmap_ckpt_path}/training_config.yaml")
        confmap_model = SingleInstanceModel.load_from_checkpoint(
            f"{confmap_ckpt_path}/best.ckpt", config=confmap_config
        )
        confmap_model.to(confmap_config.inference_config.device)
        confmap_model.m_device = confmap_config.inference_config.device

        # create an instance of SingleInstancePredictor class
        obj = cls(
            confmap_config=confmap_config,
            confmap_model=confmap_model,
        )

        obj._initialize_inference_model()
        return obj

    def make_pipeline(self) -> SingleInstanceConfmapsPipeline:
        """Make a data loading pipeline.

        Returns:
            DataLoader with the pipeline created with `sleap_nn.pipelines.SingleInstanceConfmapsPipeline`

        Notes:
            This method also creates the class attribute `data_pipeline` and will be
            called automatically when predicting on data from a new source.
        """
        self.pipeline = SingleInstanceConfmapsPipeline(data_config=self.data_config)

        provider = self.data_config.provider
        if provider == "LabelsReader":
            provider = LabelsReader

        labels = sio.load_slp(self.data_config.labels_path)
        self.videos = labels.videos
        provider_pipeline = provider(labels, max_instances=1)
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
                This should return dictionaries with keys `"image"`, `"video_idx"`,
                `"frame_idx"`, `"pred_instance_peaks"`, `"pred_peak_values"`.
                This can be created using the `_predict_generator()` method.

        Returns:
            A `sio.Labels` object with `sio.PredictedInstance`s created from
            arrays returned from the inference result generator.
        """
        predicted_frames = []
        skeletons = []
        for name in self.confmap_config.data_config.skeletons.keys():
            nodes = [
                sio.model.skeleton.Node(n["name"])
                for n in self.confmap_config.data_config.skeletons[name].nodes
            ]
            edges = [
                sio.model.skeleton.Edge(
                    sio.model.skeleton.Node(e["source"]["name"]),
                    sio.model.skeleton.Node(e["destination"]["name"]),
                )
                for e in self.confmap_config.data_config.skeletons[name].edges
            ]
            if self.confmap_config.data_config.skeletons[name].symmetries:
                list_args = [
                    set(
                        [
                            sio.model.skeleton.Node(s[0]["name"]),
                            sio.model.skeleton.Node(s[1]["name"]),
                        ]
                    )
                    for s in self.confmap_config.data_config.skeletons[name].symmetries
                ]
                symmetries = [sio.model.skeleton.Symmetry(x) for x in list_args]
            else:
                symmetries = []

            skeletons.append(
                sio.model.skeleton.Skeleton(nodes, edges, symmetries, name)
            )

        skeleton_idx = 0
        for ex in generator:
            # loop through each sample in a batch
            for (
                video_idx,
                frame_idx,
                pred_instances,
                pred_values,
                org_size,
            ) in zip(
                ex["video_idx"],
                ex["frame_idx"],
                ex["pred_instance_peaks"],
                ex["pred_peak_values"],
                ex["orig_size"],
            ):
                if self.data_config.max_height is not None:
                    pad_height = (self.data_config.max_height - org_size[0]) // 2
                    pad_width = (self.data_config.max_width - org_size[1]) // 2
                    pred_instances = (
                        pred_instances
                        - [pad_height, pad_width]
                    )
                inst = sio.PredictedInstance.from_numpy(
                    points=pred_instances,
                    skeleton=skeletons[skeleton_idx],
                    instance_score=np.nansum(pred_values),
                    point_scores=pred_values,
                )
                predicted_frames.append(
                    sio.LabeledFrame(
                        video=self.videos[video_idx],
                        frame_idx=frame_idx,
                        instances=[inst],
                    )
                )

        pred_labels = sio.Labels(
            videos=self.videos,
            skeletons=skeletons,
            labeled_frames=predicted_frames,
        )
        return pred_labels
