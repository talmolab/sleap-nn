"""Inference Pipelines."""

import numpy as np
import sleap_io as sio
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Text
import torch
import attr
import warnings
import torch.nn as nn
from abc import ABC, abstractmethod
import sleap_nn
import rich.progress
from collections import defaultdict
import json
from collections import deque
from rich.pretty import pprint
from torch.utils.data import DataLoader
from omegaconf.dictconfig import DictConfig
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.pipelines import TopdownConfmapsPipeline
from sleap_nn.architectures.model import Model, get_backbone, get_head
from time import time
from omegaconf import OmegaConf


def get_metadata(data_path: Text):
    labels = sio.load_slp(data_path)
    return labels.skeleton, labels.videos

class RateColumn(rich.progress.ProgressColumn):
    """Renders the progress rate."""

    def render(self, task: "Task") -> rich.progress.Text:
        """Show progress rate."""
        speed = task.speed
        if speed is None:
            return rich.progress.Text("?", style="progress.data.speed")
        return rich.progress.Text(f"{speed:.1f} FPS", style="progress.data.speed")

class Predictor(ABC):
    """Base interface class for predictors."""

    @classmethod
    def from_model_paths(
        cls,
        ckpt_paths: Dict,
        config_paths: Dict,
        ) -> "Predictor":
        """Create the appropriate `Predictor` subclass from from the ckpt path.

        Args:
            ckpt_paths: Dict with keys as model names and values as paths to the checkpoint file having the trained model weights
            config_paths: Dict with keys as model names and values as paths to the config.yaml file used for training

        Returns:
            A subclass of `Predictor`.

        See also: `SingleInstancePredictor`, `TopDownPredictor`, `BottomUpPredictor`,
            `MoveNetPredictor`, `TopDownMultiClassPredictor`,
            `BottomUpMultiClassPredictor`.
        """
        # Read configs and find model types.
        
        configs = dict({k:torch.load(v)["config"] for k,v in ckpt_paths})
        model_names = configs.keys()
        
        if "single_instance" in model_names:
            # predictor = SingleInstancePredictor.from_trained_models(
            #     model_path=model_paths["single_instance"],
            #     inference_config = inference_config
            # )
            pass

        elif "centroid" in model_names or "centered_instance" in model_names:
            centroid_model_path = None
            confmap_model_path = None
            if "centroid" in model_names:
                centroid_model_path = ckpt_paths["centroid"]
            if "centered_instance" in model_names:
                confmap_model_path = ckpt_paths["centered"]

            predictor = TopDownPredictor.from_trained_models(
                centroid_model_path=centroid_model_path,
                confmap_model_path=confmap_model_path,
            )

        else:
            raise ValueError(
                "Could not create predictor from model paths:" + "\n".join(ckpt_paths)
            )
        predictor.model_path = ckpt_paths
        return predictor

    @classmethod
    @abstractmethod
    def from_trained_models(cls, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def data_config(self) -> DictConfig:
        pass


    @abstractmethod
    def make_pipeline(self, data_path):
        pass

    
    def _initialize_inference_model(self):
        pass

    def _predict_generator(
        self, data_path: Text
    ) -> Iterator[Dict[str, np.ndarray]]:
        """Create a generator that yields batches of inference results.

        This method handles creating a pipeline object depending on the model type
        for loading the data, as well as looping over the batches and running inference.

        Returns:
            A generator yielding batches predicted results as dictionaries of numpy
            arrays.
        """
        # Initialize data pipeline and inference model if needed.
        self.make_pipeline(data_path)
        if self.inference_model is None:
            self._initialize_inference_model()

        def process_batch(ex):
            # Run inference on current batch.
            preds = self.inference_model(ex, numpy=True)

            # Add model outputs to the input data example.
            ex.update(preds)

            # Convert to numpy arrays if not already.
            if isinstance(ex["video_ind"], torch.Tensor):
                ex["video_ind"] = ex["video_ind"].numpy()
            if isinstance(ex["frame_ind"], torch.Tensor):
                ex["frame_ind"] = ex["frame_ind"].numpy()

            return ex

        # Loop over data batches with optional progress reporting.
        if self.verbosity == "rich":
            with rich.progress.Progress(
                "{task.description}",
                rich.progress.BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                "ETA:",
                rich.progress.TimeRemainingColumn(),
                RateColumn(),
                auto_refresh=False,
                refresh_per_second=self.report_rate,
                speed_estimate_period=5,
            ) as progress:
                task = progress.add_task("Predicting...", total=len(data_provider))
                last_report = time()
                for ex in self.pipeline.make_dataset():
                    ex = process_batch(ex)
                    progress.update(task, advance=len(ex["frame_ind"]))

                    # Handle refreshing manually to support notebooks.
                    elapsed_since_last_report = time() - last_report
                    if elapsed_since_last_report > self.report_period:
                        progress.refresh()

                    # Return results.
                    yield ex

        elif self.verbosity == "json":
            n_processed = 0
            n_total = len(data_provider)
            n_recent = deque(maxlen=30)
            elapsed_recent = deque(maxlen=30)
            last_report = time()
            t0_all = time()
            t0_batch = time()
            for ex in self.pipeline.make_dataset():
                # Process batch of examples.
                ex = process_batch(ex)

                # Track timing and progress.
                elapsed_batch = time() - t0_batch
                t0_batch = time()
                n_batch = len(ex["frame_ind"])
                n_processed += n_batch
                elapsed_all = time() - t0_all

                # Compute recent rate.
                n_recent.append(n_batch)
                elapsed_recent.append(elapsed_batch)
                rate = sum(n_recent) / sum(elapsed_recent)
                eta = (n_total - n_processed) / rate

                # Report.
                elapsed_since_last_report = time() - last_report
                if elapsed_since_last_report > self.report_period:
                    print(
                        json.dumps(
                            {
                                "n_processed": n_processed,
                                "n_total": n_total,
                                "elapsed": elapsed_all,
                                "rate": rate,
                                "eta": eta,
                            }
                        ),
                        flush=True,
                    )
                    last_report = time()

                # Return results.
                yield ex
        else:
            for ex in self.pipeline:
                yield process_batch(ex)

    def predict(
        self, data_path: Text, make_labels: bool = True, save_labels: bool = True, save_path: str=None,
    ) -> Union[List[Dict[str, np.ndarray]], sio.Labels]:
        """Run inference on a data source.

        Args:
            data: A `sleap.Labels` or `sleap.Video` to run inference over.
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
        generator = self._predict_generator(data_path)

        if make_labels:
            # Create SLEAP data structures while consuming results.
            pred_labels = self._make_labeled_frames_from_generator(generator)
            if save_labels:
                sio.io.slp.write_labels(save_path, pred_labels)
            return pred_labels
            
        else:
            # Just return the raw results.
            return list(generator)
    
class InferenceModule(nn.Module):
    """
    Inference model base class.

    This class wraps the `nn.Model` class to provide inference
    utilities such as handling different input data types, preprocessing and variable
    output shapes. This layer expects the same input as the model (rank-4 image).

    Attributes:
        model: A `torch.nn.Model` that will be called on the input to this layer.
        conf: OmegaConf file with inference related parameters.

    """

    def __init__(
        self,
        torch_model: Model,
        conf: OmegaConf,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.torch_model = torch_model
        self.config = conf
    
    def preprocess(self, imgs):
        return imgs

    def forward(
        self,
        data: Union[
            np.ndarray,
            torch.Tensor,
            Dict[str, torch.Tensor],
            sio.Video,
        ],
        **kwargs,
    ) -> Union[Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
        """Predict instances with the data.

        Args:
            data: Input data in any form. Possible types:
                - `np.ndarray`, `tf.Tensor`: Images of shape
                    `(samples, t, channels, height, width)`
                - `dict` with key `"image"` as a tensor
                - `torch.utils.data.DataLoader` that generates examples in one of the above formats.
0               - `sleap.Video` which will be converted into a pipeline that generates
                    batches of `batch_size` frames.

        Returns:
            The model outputs as a dictionary of tensors
        """

        ### TODO: check preprocess
        inputs = self.preprocess(data)
        outs = self.torch_model(inputs)
        return outs

class CentroidCropGroundTruth(nn.Module):
    """nn.Module layer that simulates a centroid cropping model using ground truth.

    This layer is useful for testing and evaluating centered instance models.

    Attributes:
        crop_size: The length of the square box to extract around each centroid.
    """

    def __init__(self, crop_size: int):
        super().__init__()
        self.crop_size = crop_size

    def forward(self, example_gt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return the ground truth instance crops.

        Args:
            example_gt: Dictionary generated from a labels pipeline with the keys:
                `"image": (batch_size, height, width, channels)`
                `"centroids": (batch_size, n_centroids, 2)`: The input centroids.
                These can be generated by the `InstanceCentroidFinder` module.

        Returns:
            Dictionary containing the output of the instance cropping layer with keys:
            `"crops": (batch_size, n_centroids, crop_size, crop_size, channels)`
            `"crop_bboxs": (batch_size, n_centroids, crop_size, crop_size, channels)`
                These contain the top-left coordinates of each crop in the full images.
            `"centroid": (batch_size, 1, 2)`
            `"centroid_vals": (batch_size, 1)`

            
            `"centroids"` are from the input example and `"centroid_vals"` will be
            filled with ones.
        """
        
        full_imgs = example_gt["image"]
        centroids = example_gt["centroids"] # (batch, 1, n_inst, 2)
        centroid_vals = torch.ones((centroids.shape[0], centroids.shape[-2])) # (n_inst, )
        crop_offsets = centroids - (self.crop_size/2)

        bboxes = sleap_nn.data.instance_cropping.make_centered_bboxes(centroids, self.crop_size, self.crop_size)
        sample_inds = range(0,bboxes.shape[-2])
        crops = sleap_nn.inference.peak_finding.crop_bboxes(full_imgs, bboxes, sample_inds)

        return dict(
            crops = crops,
            centroids = example_gt["centroids"],
            centroid_vals = centroid_vals,
            crop_offsets = crop_offsets
        )
        
class CentroidCrop(InferenceModule):
    pass

class FindInstancePeaksGroundTruth(nn.Module):
    pass

class FindInstancePeaks(InferenceModule):
    """nn.Module that predicts instance peaks from images using a trained model.

    This layer encapsulates all of the inference operations required for generating
    predictions from a centered instance confidence map model. This includes
    preprocessing, model forward pass, peak finding and coordinate adjustment.

    Attributes:
        torch_model: A `nn.Module` that accepts rank-4 images as input and predicts
            rank-4 confidence maps as output. This should be a model that is trained on
            centered instance confidence maps.
        output_stride: Output stride of the model, denoting the scale of the output
            confidence maps relative to the images (after input scaling). This is used
            for adjusting the peak coordinates to the image grid. This will be inferred
            from the model shapes if not provided.
        peak_threshold: Minimum confidence map value to consider a global peak as valid.
        refinement: If `None`, returns the grid-aligned peaks with no refinement. If
            `"integral"`, peaks will be refined with integral regression. If `"local"`,
            peaks will be refined with quarter pixel local gradient offset. This has no
            effect if the model has an offset regression head.
        integral_patch_size: Size of patches to crop around each rough peak for integral
            refinement as an integer scalar.
        return_confmaps: If `True`, the confidence maps will be returned together with
            the predicted peaks. This will result in slower inference times since the
            data must be copied off of the GPU, but is useful for visualizing the raw
            output of the model.
    """

    def __init__(
        self,
        torch_model: torch.nn.Module,
        output_stride: Optional[int] = None,
        peak_threshold: float = 0.2,
        refinement: Optional[str] = "local",
        integral_patch_size: int = 5,
        return_confmaps: bool = False,
        **kwargs,
    ):
        super().__init__(
            torch_model=torch_model, pad_to_stride=1, **kwargs
        )
        self.peak_threshold = peak_threshold
        self.refinement = refinement
        self.integral_patch_size = integral_patch_size
        self.return_confmaps = return_confmaps
        self.output_stride = output_stride

    def forward(
        self, inputs: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Predict confidence maps and infer peak coordinates.

        This layer can be chained with a `CentroidCrop` layer to create a top-down
        inference function from full images.

        Args:
            inputs: Instance-centered images as a `tf.Tensor` of shape
                `(samples, height, width, channels)` or `tf.RaggedTensor` of shape
                `(samples, ?, height, width, channels)` where images are grouped by
                sample and may contain a variable number of crops, or a dictionary with
                keys:
                `"crops"`: Cropped images in either format above.
                `"crop_offsets"`: (Optional) Coordinates of the top-left of the crops as
                    `(x, y)` offsets of shape `(samples, ?, 2)` for adjusting the
                    predicted peak coordinates. No adjustment is performed if not
                    provided.
                `"centroids"`: (Optional) If provided, will be passed through to the
                    output.
                `"centroid_vals"`: (Optional) If provided, will be passed through to the
                    output.

        Returns:
            A dictionary of outputs with keys:

            `"instance_peaks"`: The predicted peaks for each instance in the batch as a
                `tf.RaggedTensor` of shape `(samples, ?, nodes, 2)`.
            `"instance_peak_vals"`: The value of the confidence maps at the predicted
                peaks for each instance in the batch as a `tf.RaggedTensor` of shape
                `(samples, ?, nodes)`.

            If provided (e.g., from an input `CentroidCrop` layer), the centroids that
            generated the crops will also be included in the keys `"centroids"` and
            `"centroid_vals"`.

            If the `return_confmaps` attribute is set to `True`, the output will also
            contain a key named `"instance_confmaps"` containing a `tf.RaggedTensor` of
            shape `(samples, ?, output_height, output_width, nodes)` containing the
            confidence maps predicted by the model.
        """
        if isinstance(inputs, dict):
            crops = inputs["instance_image"]
        else:
            # Tensor input provided. We'll infer the extra fields in the expected input
            # dictionary.
            crops = inputs
            inputs = {}

        # Preprocess inputs (scaling, padding, colorspace, int to float).
        crops = self.preprocess(crops)

        # Network forward pass.
        cms = self.torch_model(crops)

        peak_points, peak_vals = sleap_nn.inference.peak_finding.find_global_peaks(
                cms,
                threshold=self.peak_threshold,
                refinement=self.refinement,
                integral_patch_size=self.integral_patch_size,
            )
        

        # Adjust for stride and scale.
        peak_points = peak_points * self.output_stride
        if self.input_scale != 1.0:
            # Note: We add 0.5 here to offset TensorFlow's weird image resizing. This
            # may not always(?) be the most correct approach.
            # See: https://github.com/tensorflow/tensorflow/issues/6720
            peak_points = (peak_points / self.input_scale) + 0.5

        # Build outputs.
        outputs = {"pred_instance_peaks": peak_points, "pred_peak_values": peak_vals}
        if "centroids" in inputs:
            outputs["centroids"] = inputs["centroids"]
        if "centroid_vals" in inputs:
            outputs["centroid_vals"] = inputs["centroid_vals"]
        if "centroid_confmaps" in inputs:
            outputs["centroid_confmaps"] = inputs["centroid_confmaps"]
        if self.return_confmaps:
            outputs["instance_confmaps"] = cms
        return outputs

class TopDownInferenceModel(nn.Module):
    """Top-down instance prediction model.

    This model encapsulates the top-down approach where instances are first detected by
    local peak detection of an anchor point and then cropped. These instance-centered
    crops are then passed to an instance peak detector which is trained to detect all
    remaining body parts for the instance that is centered within the crop.

    Attributes:
        centroid_crop: A centroid cropping layer. This can be either `CentroidCrop` or
            `CentroidCropGroundTruth`. This layer takes the full image as input and
            outputs a set of centroids and cropped boxes.
        instance_peaks: A instance peak detection layer. This can be either
            `FindInstancePeaks` or `FindInstancePeaksGroundTruth`. This layer takes as
            input the output of the centroid cropper and outputs the detected peaks for
            the instances within each crop.
    """

    def __init__(
        self,
        centroid_crop: Union[CentroidCrop, CentroidCropGroundTruth],
        instance_peaks: Union[FindInstancePeaks, FindInstancePeaksGroundTruth],
        **kwargs,
    ):
        super().__init__(
        )
        self.centroid_crop = centroid_crop
        self.instance_peaks = instance_peaks

    def forward(
        self, example: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Predict instances for one batch of images.

        Args:
            example: This may be either a single batch of images as a 4-D tensor of
                shape `(batch_size, height, width, channels)`, or a dictionary
                containing the image batch in the `"images"` key. If using a ground
                truth model for either centroid cropping or instance peaks, the full
                example from a `Pipeline` is required for providing the metadata.

        Returns:
            The predicted instances as a dictionary of tensors with keys:

            `"centroids": (batch_size, n_instances, 2)`: Instance centroids.
            `"centroid_vals": (batch_size, n_instances)`: Instance centroid confidence
                values.
            `"instance_peaks": (batch_size, n_instances, n_nodes, 2)`: Instance skeleton
                points.
            `"instance_peak_vals": (batch_size, n_instances, n_nodes)`: Confidence
                values for the instance skeleton points.
        """
        if isinstance(example, torch.Tensor):
            example = dict(image=example)

        crop_output = self.centroid_crop(example)

        if isinstance(self.instance_peaks, FindInstancePeaksGroundTruth):
            if "instances" in example:
                peaks_output = self.instance_peaks(example, crop_output)
            else:
                raise ValueError(
                    "Ground truth data was not detected... "
                    "Please load both models when predicting on non-ground-truth data."
                )
        else:
            peaks_output = self.instance_peaks(crop_output)
        return peaks_output

@attr.s(auto_attribs=True)
class TopDownPredictor(Predictor):
    """Top-down multi-instance predictor.

    This high-level class handles initialization, preprocessing and tracking using a
    trained top-down SLEAP model.

    This should be initialized using the `from_trained_models()` constructor.

    Attributes:
        centroid_ckpt_path: The `sleap.nn.config.TrainingJobConfig` containing the metadata
            for the trained centroid model. If `None`, ground truth centroids will be
        confmap_ckpt_path: The `sleap.nn.config.TrainingJobConfig` containing the metadata
            for the trained centered instance model. If `None`, ground truth instances
            will be used if available from the data source.

    """

    centroid_ckpt_path: Optional[DictConfig] = attr.ib(default=None)
    confmap_ckpt_path: Optional[DictConfig] = attr.ib(default=None)

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained models and configuration."""
        use_gt_centroid = self.centroid_config is None
        use_gt_confmap = self.confmap_config is None

        if use_gt_centroid:
            centroid_crop_layer = CentroidCropGroundTruth(
                crop_size=self.inference_config.data.instance_cropping.crop_size
            )
        else:
            if use_gt_confmap:
                crop_size = 1
            else:
                crop_size = self.inference_config.data.instance_cropping.crop_size
            # centroid_crop_layer = CentroidCrop(
            #     torch_model=self.centroid_model,
            #     crop_size=crop_size,
            #     input_scale=self.centroid_config.data.preprocessing.input_scaling,
            #     pad_to_stride=self.centroid_config.data.preprocessing.pad_to_stride,
            #     output_stride=self.centroid_config.model.heads.centroid.output_stride,
            #     peak_threshold=self.inference_config.peak_threshold,
            #     refinement="integral" if self.inference_config.integral_refinement else "local",
            #     integral_patch_size=self.inference_config.integral_patch_size,
            #     return_confmaps=False,
            # )

        if use_gt_confmap:
            pass
            # instance_peaks_layer = FindInstancePeaksGroundTruth()
        else:
            cfg = self.confmap_config
            instance_peaks_layer = FindInstancePeaks(
                torch_model=self.confmap_model,
                input_scale=self.inference_config.data.preprocessing.input_scaling,
                peak_threshold=self.inference_config.peak_threshold,
                output_stride=cfg.model.heads.head_config.output_stride,
                refinement=self.inference_config.integral_refinement,
                integral_patch_size=self.inference_config.integral_patch_size,
                return_confmaps=False,
            )

        self.inference_model = TopDownInferenceModel(
            centroid_crop=centroid_crop_layer, instance_peaks=instance_peaks_layer
        )

    @property
    def data_config(self) -> DictConfig:
        return (
            self.inference_config.data
        )

    @classmethod
    def from_trained_models(
        cls,
        centroid_model_path: Optional[Text] = None,
        confmap_model_path: Optional[Text] = None,
        inference_config: Optional[DictConfig] = None,
    ) -> "TopDownPredictor":
        """Create predictor from saved models.

        Args:
            centroid_model_path: Path to a centroid model folder or training job JSON
                file inside a model folder. This folder should contain
                `training_config.json` and `best_model.h5` files for a trained model.
            confmap_model_path: Path to a centered instance model folder or training job
                JSON file inside a model folder. This folder should contain
                `training_config.json` and `best_model.h5` files for a trained model.


        Returns:
            An instance of `TopDownPredictor` with the loaded models.

            One of the two models can be left as `None` to perform inference with ground
            truth data. This will only work with `LabelsReader` as the provider.
        """
        if centroid_model_path is None and confmap_model_path is None:
            raise ValueError(
                "Either the centroid or topdown confidence map model must be provided."
            )

        if centroid_model_path is not None:
            # Load centroid model.
            centroid_config = inference_config.centroid 
            centroid_model = Model(centroid_config.model.backbone, [centroid_config.model.heads])
            centroid_model.load_state_dict(torch.load(centroid_model_path), strict=False)

        else:
            centroid_config = None
            centroid_model = None

        if confmap_model_path is not None:
            # Load confmap model.
            confmap_config = inference_config.centered 
            confmap_model = Model(confmap_config.model.backbone, [confmap_config.model.heads])
            confmap_model.load_state_dict(torch.load(confmap_model_path), strict=False)

        else:
            confmap_config = None
            confmap_model = None

        obj = cls(
            centroid_config=centroid_config,
            centroid_model=centroid_model,
            confmap_config=confmap_config,
            confmap_model=confmap_model,
            inference_config=inference_config,
        )
        obj._initialize_inference_model()
        return obj

    @property
    def is_grayscale(self) -> bool:
        """Return whether the model expects grayscale inputs."""
        is_gray = False
        if self.centroid_model is not None:
            is_gray = self.centroid_model.input.shape[-1] == 1
        else:
            is_gray = self.confmap_model.input.shape[-1] == 1
        return is_gray

    def make_pipeline(self, data_path: Text, data_provider: Optional[LabelsReader] = None) -> TopdownConfmapsPipeline:
        """Make a data loading pipeline.

        Args:
            data_provider: If not `None`, the pipeline will be created with an instance
                of a `sleap.pipelines.Provider`.

        Returns:
            The created `sleap.pipelines.Pipeline` with batching and prefetching.

        Notes:
            This method also updates the class attribute for the pipeline and will be
            called automatically when predicting on data from a new source.
        """
        self.pipeline = TopdownConfmapsPipeline(self.data_config)
        self.skeleton, self.videos = get_metadata(data_path)
        self.pipeline = self.pipeline.make_training_pipeline(data_provider, data_path)
        self.pipeline = self.pipeline.sharding_filter()
        if self.data_config.num_workers is not None:
            num_workers = self.data_config.num_workers
        else:
            num_workers = None
        self.data_pipeline = DataLoader(self.pipeline, batch_size=self.data_config.batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

        return self.data_pipeline

    def _make_labeled_frames_from_generator(
        self, generator: Iterator[Dict[str, np.ndarray]], 
    ) -> List[sio.LabeledFrame]:
        """Create labeled frames from a generator that yields inference results.

        This method converts pure arrays into SLEAP-specific data structures and runs
        them through the tracker if it is specified.

        Args:
            generator: A generator that returns dictionaries with inference results.
                This should return dictionaries with keys `"image"`, `"video_ind"`,
                `"frame_ind"`, `"instance_peaks"`, `"instance_peak_vals"`, and
                `"centroid_vals"`. This can be created using the `_predict_generator()`
                method.
            data_provider: The `sleap.pipelines.Provider` that the predictions are being
                created from. This is used to retrieve the `sleap.Video` instance
                associated with each inference result.

        Returns:
            A list of `sleap.LabeledFrame`s with `sio.PredictedInstance`s created from
            arrays returned from the inference result generator.
        """

        predicted_frames = []
        preds = defaultdict(List)

        for ex in generator:
            # loop through each instance prediction
            for video_idx, frame_idx, bbox, pred_instances, pred_values, instance_score in zip(ex["video_idx"], ex["frame_idx"], ex["instance_bbox"], ex["pred_instance_peaks"], ex["pred_peak_values"], ex["centroid_vals"]):
                pred_instances = pred_instances + bbox
                preds[(int(video_idx), int(frame_idx))].append(sio.PredictedInstance.from_numpy(points=pred_instances, skeleton=self.skeleton, point_scores=pred_values, instance_score=instance_score))

            # create list of LabeledFrames
            for key, inst in preds.items():
                video_idx, frame_idx = key
                predicted_frames.append(sio.LabeledFrame(video=self.videos[video_idx], frame_idx=frame_idx, instances=inst))

        pred_labels = sio.Labels(videos=self.videos, skeletons=[self.skeleton], labeled_frames=predicted_frames)
        return pred_labels

       



    
