"""Predictors for running inference."""

from collections import defaultdict
from typing import Dict, List, Optional, Union, Iterator, Text
from queue import Queue
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import sleap_io as sio
import torch
import attrs
import lightning as L
from torch.utils.data.dataloader import DataLoader
from omegaconf import OmegaConf
from sleap_nn.data.providers import LabelsReader, VideoReader
from sleap_nn.data.resizing import (
    SizeMatcher,
    Resizer,
    PadToStride,
    resize_image,
    pad_to_stride,
)
from sleap_nn.data.normalization import Normalizer, convert_to_grayscale, convert_to_rgb
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.instance_cropping import InstanceCropper
from sleap_nn.data.general import KeyFilter
from sleap_nn.inference.paf_grouping import PAFScorer
from sleap_nn.training.model_trainer import (
    TopDownCenteredInstanceModel,
    SingleInstanceModel,
    CentroidModel,
    BottomUpModel,
)
from sleap_nn.inference.single_instance import SingleInstanceInferenceModel
from sleap_nn.inference.bottomup import BottomUpInferenceModel
from sleap_nn.inference.topdown import (
    CentroidCrop,
    FindInstancePeaks,
    FindInstancePeaksGroundTruth,
    TopDownInferenceModel,
)
from sleap_nn.inference.utils import get_skeleton_from_config


@attrs.define
class Predictor(ABC):
    """Base interface class for predictors.

    This is the base predictor class for different types of models.

    Attributes:
        preprocess: Only for VideoReader provider. True if preprocessing (reszizing and
            pad_to_stride) should be applied on the frames read in the video reader.
            Default: True.
        video_preprocess_config: Preprocessing config for VideoReader with keys: [`batch_size`,
            `scale`, `is_rgb`, `max_stride`]. Default: {"batch_size": 4, "scale": 1.0,
            "is_rgb": False, "max_stride": 1}
        provider: Provider for inference pipeline. One of ["LabelsReader", "VideoReader"].
            Default: LabelsReader.
        pipeline: If provider is LabelsReader, pipeline is a `DataLoader` object. If provider
            is VideoReader, pipeline is an instance of `sleap_nn.data.providers.VideoReader`
            class. Default: None.
        inference_model: Instance of one of the inference models ["TopDownInferenceModel",
            "SingleInstanceInferenceModel", "BottomUpInferenceModel"]. Default: None.
    """

    preprocess: bool = True
    video_preprocess_config: dict = {
        "batch_size": 4,
        "scale": 1.0,
        "is_rgb": False,
        "max_stride": 1,
    }
    provider: Union[LabelsReader, VideoReader] = LabelsReader
    pipeline: Optional[Union[DataLoader, VideoReader]] = None
    inference_model: Optional[
        Union[
            TopDownInferenceModel, SingleInstanceInferenceModel, BottomUpInferenceModel
        ]
    ] = None

    @classmethod
    def from_model_paths(cls, model_paths: List[Text]) -> "Predictor":
        """Create the appropriate `Predictor` subclass from from the ckpt path.

        Args:
            model_paths: List of paths to the directory where the best.ckpt and
                training_config.yaml are saved.

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
            (c.model_config.head_configs[0].head_type) for c in model_config_paths
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
                centroid_ckpt_path = model_paths[
                    model_names.index("CentroidConfmapsHead")
                ]
            if "CenteredInstanceConfmapsHead" in model_names:
                confmap_ckpt_path = model_paths[
                    model_names.index("CenteredInstanceConfmapsHead")
                ]

            # create an instance of the TopDown predictor class
            predictor = TopDownPredictor.from_trained_models(
                centroid_ckpt_path=centroid_ckpt_path,
                confmap_ckpt_path=confmap_ckpt_path,
            )

        elif (
            "MultiInstanceConfmapsHead" in model_names
            or "PartAffinityFieldsHead" in model_names
        ):
            bottomup_ckpt_path = model_paths[
                model_names.index("MultiInstanceConfmapsHead")
            ]
            predictor = BottomUpPredictor.from_trained_models(bottomup_ckpt_path)
        else:
            raise ValueError(
                f"Could not create predictor from model paths:\n{model_paths}"
            )
        return predictor

    @classmethod
    @abstractmethod
    def from_trained_models(cls, *args, **kwargs):
        """Initialize the Predictor class for certain type of model."""

    @property
    @abstractmethod
    def data_config(self) -> OmegaConf:
        """Get the data parameters from the config."""

    @abstractmethod
    def make_pipeline(self):
        """Create the data pipeline."""

    @abstractmethod
    def _initialize_inference_model(self):
        """Initialize the Inference model."""

    def _convert_tensors_to_numpy(self, output):
        """Convert tensors in output dictionary to numpy arrays."""
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                if v.is_nested:
                    output[k] = [i.cpu().numpy() for i in v]
                else:
                    output[k] = output[k].cpu().numpy()
        return output

    def _predict_generator(self) -> Iterator[Dict[str, np.ndarray]]:
        """Create a generator that yields batches of inference results.

        This method handles creating a pipeline object depending on the model type and
        provider for loading the data, as well as looping over the batches and
        running inference.

        Returns:
            A generator yielding batches predicted results as dictionaries of numpy
            arrays.
        """
        # Initialize data pipeline and inference model if needed.
        self.make_pipeline()
        if self.inference_model is None:
            self._initialize_inference_model()

        # Loop over data batches.
        if self.provider == "LabelsReader":
            for ex in self.pipeline:
                outputs_list = self.inference_model(ex)
                for output in outputs_list:
                    output = self._convert_tensors_to_numpy(output)
                    yield output

        elif self.provider == "VideoReader":
            try:
                self.pipeline.start()
                batch_size = self.video_preprocess_config["batch_size"]
                done = False
                while not done:
                    imgs = []
                    fidxs = []
                    org_szs = []
                    for _ in range(batch_size):
                        frame = self.pipeline.frame_buffer.get()
                        if frame[0] is None:
                            done = True
                            break
                        imgs.append(frame[0].unsqueeze(dim=0))
                        fidxs.append(frame[1])
                        org_szs.append(frame[2].unsqueeze(dim=0))
                    if imgs:
                        imgs = torch.concatenate(imgs, dim=0)
                        fidxs = torch.tensor(fidxs, dtype=torch.int32)
                        org_szs = torch.concatenate(org_szs, dim=0)
                        ex = {
                            "image": imgs,
                            "frame_idx": fidxs,
                            "video_idx": torch.tensor(
                                [0] * batch_size, dtype=torch.int32
                            ),
                            "orig_size": org_szs,
                        }
                        if not torch.is_floating_point(ex["image"]):  # normalization
                            ex["image"] = ex["image"].to(torch.float32) / 255.0
                        if self.video_preprocess_config["is_rgb"]:
                            ex["image"] = convert_to_rgb(ex["image"])
                        else:
                            ex["image"] = convert_to_grayscale(ex["image"])
                        if self.preprocess:
                            scale = self.video_preprocess_config["scale"]
                            if scale != 1.0:
                                ex["image"] = resize_image(ex["image"], scale)
                            ex["image"] = pad_to_stride(
                                ex["image"], self.video_preprocess_config["max_stride"]
                            )
                        outputs_list = self.inference_model(ex)
                        for output in outputs_list:
                            output = self._convert_tensors_to_numpy(output)
                            yield output

            except Exception as e:
                raise Exception(f"Error in VideoReader during data processing: {e}")

            finally:
                self.pipeline.join()

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
            save_path: Path to save the labels file if `make_labels` is True.

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

    @abstractmethod
    def _make_labeled_frames_from_generator(self, generator) -> sio.Labels:
        """Create `sio.Labels` object from the predictions."""


@attrs.define
class TopDownPredictor(Predictor):
    """Top-down multi-instance predictor.

    This high-level class handles initialization, preprocessing and predicting using a
    trained TopDown SLEAP-NN model. This should be initialized using the
    `from_trained_models()` constructor.

    Attributes:
        centroid_config: A Dictionary with the configs used for training the centroid model.
        confmap_config: A Dictionary with the configs used for training the
                        centered-instance model
        centroid_model: A LightningModule instance created from the trained weights
                        for centroid model.
        confmap_model: A LightningModule instance created from the trained weights
                       for centered-instance model.
        videos: List of `sio.Video` objects for creating the `sio.Labels` object from
                        the output predictions.
        skeletons: List of `sio.Skeleton` objects for creating `sio.Labels` object from
                        the output predictions.

    """

    centroid_config: Optional[OmegaConf] = attrs.field(default=None)
    confmap_config: Optional[OmegaConf] = attrs.field(default=None)
    centroid_model: Optional[L.LightningModule] = attrs.field(default=None)
    confmap_model: Optional[L.LightningModule] = attrs.field(default=None)
    videos: Optional[List[sio.Video]] = attrs.field(default=None)
    skeletons: Optional[List[sio.Skeleton]] = attrs.field(default=None)

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained models and configuration."""
        # Create an instance of CentroidLayer if centroid_config is not None
        return_crops = False
        if self.centroid_config is None:
            centroid_crop_layer = None
        else:
            max_stride = (
                self.centroid_config.model_config.backbone_config.backbone_config.max_stride
            )
            self.centroid_config.inference_config.data["skeletons"] = (
                self.centroid_config.data_config.skeletons
            )

            # if both centroid and centered-instance model are provided, set return crops to True
            if self.confmap_model:
                return_crops = True

            # initialize centroid crop layer
            centroid_crop_layer = CentroidCrop(
                torch_model=self.centroid_model,
                peak_threshold=self.centroid_config.inference_config.peak_threshold,
                output_stride=(
                    self.centroid_config.inference_config.data.preprocessing.output_stride
                ),
                refinement=self.centroid_config.inference_config.integral_refinement,
                integral_patch_size=self.centroid_config.inference_config.integral_patch_size,
                return_confmaps=self.centroid_config.inference_config.return_confmaps,
                return_crops=return_crops,
                max_instances=self.centroid_config.inference_config.data.max_instances,
                crop_hw=tuple(
                    self.centroid_config.inference_config.data.preprocessing.crop_hw
                ),
                input_scale=self.centroid_config.inference_config.data.scale,
                max_stride=max_stride,
            )

        # Create an instance of FindInstancePeaks layer if confmap_config is not None
        if self.confmap_config is None:
            instance_peaks_layer = FindInstancePeaksGroundTruth()
        else:
            self.confmap_config.inference_config.data["skeletons"] = (
                self.confmap_config.data_config.skeletons
            )
            max_stride = (
                self.confmap_config.model_config.backbone_config.backbone_config.max_stride
            )
            instance_peaks_layer = FindInstancePeaks(
                torch_model=self.confmap_model,
                peak_threshold=self.confmap_config.inference_config.peak_threshold,
                output_stride=self.confmap_config.inference_config.data.preprocessing.output_stride,
                refinement=self.confmap_config.inference_config.integral_refinement,
                integral_patch_size=self.confmap_config.inference_config.integral_patch_size,
                return_confmaps=self.confmap_config.inference_config.return_confmaps,
                input_scale=self.confmap_config.inference_config.data.scale,
                max_stride=max_stride,
            )

        # Initialize the inference model with centroid and instance peak layers
        self.inference_model = TopDownInferenceModel(
            centroid_crop=centroid_crop_layer, instance_peaks=instance_peaks_layer
        )

    @property
    def data_config(self) -> OmegaConf:
        """Returns data config section from the overall config."""
        if self.centroid_config:
            return self.centroid_config.inference_config.data
        return self.confmap_config.inference_config.data

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
            centroid_config = OmegaConf.load(
                f"{centroid_ckpt_path}/training_config.yaml"
            )
            skeletons = get_skeleton_from_config(centroid_config.data_config.skeletons)
            centroid_model = CentroidModel.load_from_checkpoint(
                f"{centroid_ckpt_path}/best.ckpt",
                config=centroid_config,
                skeletons=skeletons,
            )
            centroid_model.to(centroid_config.inference_config.device)
            centroid_model.m_device = centroid_config.inference_config.device

        else:
            centroid_config = None
            centroid_model = None

        if confmap_ckpt_path is not None:
            # Load confmap model.
            confmap_config = OmegaConf.load(f"{confmap_ckpt_path}/training_config.yaml")
            skeletons = get_skeleton_from_config(confmap_config.data_config.skeletons)
            confmap_model = TopDownCenteredInstanceModel.load_from_checkpoint(
                f"{confmap_ckpt_path}/best.ckpt",
                config=confmap_config,
                skeletons=skeletons,
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
            skeletons=skeletons,
        )

        obj._initialize_inference_model()
        return obj

    def make_pipeline(self):
        """Make a data loading pipeline.

        Returns:
            Torch DataLoader where each item is a dictionary with key `image` if provider
            is LabelsReader. If provider is VideoReader, this method initiates the reader
            class (doesn't return a pipeline) and the Thread is started in
            Predictor._predict_generator() method.

        Notes:
            This method creates the class attribute `pipeline` and will be
            called automatically when predicting on data from a new source only when the
            provider is LabelsReader.
        """
        self.provider = self.data_config.provider

        # LabelsReader provider
        if self.provider == "LabelsReader":
            provider = LabelsReader
            instances_key = True

            # no need of `instances` key for Centered-instance model
            if self.centroid_config and self.confmap_config:
                instances_key = False

            data_provider = provider.from_filename(
                self.data_config.path, instances_key=instances_key
            )

            self.videos = data_provider.labels.videos

            pipeline = Normalizer(data_provider, is_rgb=self.data_config.is_rgb)
            pipeline = SizeMatcher(
                pipeline,
                max_height=self.data_config.max_height,
                max_width=self.data_config.max_width,
                provider=data_provider,
            )

            if not self.centroid_model:
                pipeline = InstanceCentroidFinder(
                    pipeline,
                    anchor_ind=self.data_config.preprocessing.anchor_ind,
                )
                pipeline = InstanceCropper(
                    pipeline,
                    crop_hw=self.data_config.preprocessing.crop_hw,
                )

                pipeline = KeyFilter(
                    pipeline,
                    keep_keys=[
                        "image",
                        "video_idx",
                        "frame_idx",
                        "centroid",
                        "instance",
                        "instance_bbox",
                        "instance_image",
                        "confidence_maps",
                        "num_instances",
                        "orig_size",
                        "scale",
                    ],
                )

            # Remove duplicates.
            self.pipeline = pipeline.sharding_filter()

            self.pipeline = DataLoader(
                self.pipeline,
                **dict(self.data_config.data_loader),
            )

            return self.pipeline

        # VideoReader provider
        elif self.provider == "VideoReader":
            if self.centroid_config is None:
                raise ValueError(
                    "Ground truth data was not detected... "
                    "Please load both models when predicting on non-ground-truth data."
                )

            provider = VideoReader
            self.preprocess = False
            self.video_preprocess_config = {
                "batch_size": self.data_config.video_loader.batch_size,
                "scale": self.data_config.scale,
                "is_rgb": self.data_config.is_rgb,
                "max_stride": (
                    self.centroid_config.model_config.backbone_config.backbone_config.max_stride
                ),
            }

            frame_queue = Queue(
                maxsize=self.data_config.video_loader.queue_maxsize if not None else 16
            )
            self.pipeline = provider.from_filename(
                filename=self.data_config.path,
                frame_buffer=frame_queue,
                start_idx=self.data_config.video_loader.start_idx,
                end_idx=self.data_config.video_loader.end_idx,
            )
            self.videos = [self.pipeline.video]

        else:
            raise Exception(
                "Provider not recognised. Please use either `LabelsReader` or `VideoReader` as provider"
            )

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
                pred_instances = pred_instances + bbox.squeeze(axis=0)[0, :]
                preds[(int(video_idx), int(frame_idx))].append(
                    sio.PredictedInstance.from_numpy(
                        points=pred_instances,
                        skeleton=self.skeletons[skeleton_idx],
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
            skeletons=self.skeletons,
            labeled_frames=predicted_frames,
        )
        return pred_labels


@attrs.define
class SingleInstancePredictor(Predictor):
    """Single-Instance predictor.

    This high-level class handles initialization, preprocessing and predicting using a
    trained single instance SLEAP-NN model.

    This should be initialized using the `from_trained_models()` constructor.

    Attributes:
        confmap_config: A Dictionary with the configs used for training the
                        single-instance model.
        confmap_model: A LightningModule instance created from the trained weights for
                       single-instance model.
        videos: List of `sio.Video` objects for creating the `sio.Labels` object from
                        the output predictions.
        skeletons: List of `sio.Skeleton` objects for creating `sio.Labels` object from
                        the output predictions.

    """

    confmap_config: Optional[OmegaConf] = attrs.field(default=None)
    confmap_model: Optional[L.LightningModule] = attrs.field(default=None)
    videos: Optional[List[sio.Video]] = attrs.field(default=None)
    skeletons: Optional[List[sio.Skeleton]] = attrs.field(default=None)

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained models and configuration."""
        self.inference_model = SingleInstanceInferenceModel(
            torch_model=self.confmap_model,
            peak_threshold=self.confmap_config.inference_config.peak_threshold,
            output_stride=self.confmap_config.inference_config.data.preprocessing.output_stride,
            refinement=self.confmap_config.inference_config.integral_refinement,
            integral_patch_size=self.confmap_config.inference_config.integral_patch_size,
            return_confmaps=self.confmap_config.inference_config.return_confmaps,
            input_scale=self.confmap_config.inference_config.data.scale,
        )

    @property
    def data_config(self) -> OmegaConf:
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
        skeletons = get_skeleton_from_config(confmap_config.data_config.skeletons)
        confmap_model = SingleInstanceModel.load_from_checkpoint(
            f"{confmap_ckpt_path}/best.ckpt", config=confmap_config, skeletons=skeletons
        )
        confmap_model.to(confmap_config.inference_config.device)
        confmap_model.m_device = confmap_config.inference_config.device

        # create an instance of SingleInstancePredictor class
        obj = cls(
            confmap_config=confmap_config,
            confmap_model=confmap_model,
            skeletons=skeletons,
        )

        obj._initialize_inference_model()
        return obj

    def make_pipeline(self):
        """Make a data loading pipeline.

        Returns:
            Torch DataLoader where each item is a dictionary with key `image` if provider
            is LabelsReader. If provider is VideoReader, this method initiates the reader
            class (doesn't return a pipeline) and the Thread is started in
            Predictor._predict_generator() method.

        Notes:
            This method creates the class attribute `pipeline` and will be
            called automatically when predicting on data from a new source only when the
            provider is LabelsReader.
        """
        self.provider = self.data_config.provider
        if self.provider == "LabelsReader":
            provider = LabelsReader
            data_provider = provider.from_filename(self.data_config.path)
            self.videos = data_provider.labels.videos
            pipeline = Normalizer(data_provider, is_rgb=self.data_config.is_rgb)
            pipeline = SizeMatcher(
                pipeline,
                max_height=self.data_config.max_height,
                max_width=self.data_config.max_width,
                provider=data_provider,
            )
            pipeline = Resizer(pipeline, scale=self.data_config.scale)
            pipeline = PadToStride(
                pipeline,
                max_stride=self.confmap_config.model_config.backbone_config.backbone_config.max_stride,
            )

            # Remove duplicates.
            self.pipeline = pipeline.sharding_filter()

            self.pipeline = DataLoader(
                self.pipeline,
                **dict(self.data_config.data_loader),
            )

            return self.pipeline

        elif self.provider == "VideoReader":
            provider = VideoReader
            self.preprocess = True
            self.video_preprocess_config = {
                "batch_size": self.data_config.video_loader.batch_size,
                "scale": self.data_config.scale,
                "is_rgb": self.data_config.is_rgb,
                "max_stride": (
                    self.confmap_config.model_config.backbone_config.backbone_config.max_stride
                ),
            }
            frame_queue = Queue(
                maxsize=self.data_config.video_loader.queue_maxsize if not None else 16
            )
            self.pipeline = provider.from_filename(
                filename=self.data_config.path,
                frame_buffer=frame_queue,
                start_idx=self.data_config.video_loader.start_idx,
                end_idx=self.data_config.video_loader.end_idx,
            )
            self.videos = [self.pipeline.video]

        else:
            raise Exception(
                "Provider not recognised. Please use either `LabelsReader` or `VideoReader` as provider"
            )

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

                inst = sio.PredictedInstance.from_numpy(
                    points=pred_instances,
                    skeleton=self.skeletons[skeleton_idx],
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
            skeletons=self.skeletons,
            labeled_frames=predicted_frames,
        )
        return pred_labels


@attrs.define
class BottomUpPredictor(Predictor):
    """BottomUp model predictor.

    This high-level class handles initialization, preprocessing and predicting using a
    trained BottomUp SLEAP-NN model.

    This should be initialized using the `from_trained_models()` constructor.

    Attributes:
        bottomup_config: A OmegaConfig dictionary with the configs used for training the
                        bottom-up model.
        bottomup_model: A LightningModule instance created from the trained weights for
                       bottom-up model.
        max_edge_length_ratio: The maximum expected length of a connected pair of points
            as a fraction of the image size. Candidate connections longer than this
            length will be penalized during matching.
        dist_penalty_weight: A coefficient to scale weight of the distance penalty as
            a scalar float. Set to values greater than 1.0 to enforce the distance
            penalty more strictly.
        n_points: Number of points to sample along the line integral.
        min_instance_peaks: Minimum number of peaks the instance should have to be
                considered a real instance. Instances with fewer peaks than this will be
                discarded (useful for filtering spurious detections).
        min_line_scores: Minimum line score (between -1 and 1) required to form a match
            between candidate point pairs. Useful for rejecting spurious detections when
            there are no better ones.
        videos: List of `sio.Video` objects for creating the `sio.Labels` object from
                        the output predictions.
        skeletons: List of `sio.Skeleton` objects for creating `sio.Labels` object from
                        the output predictions.

    """

    bottomup_config: Optional[OmegaConf] = attrs.field(default=None)
    bottomup_model: Optional[L.LightningModule] = attrs.field(default=None)
    max_edge_length_ratio: float = 0.25
    dist_penalty_weight: float = 1.0
    n_points: int = 10
    min_instance_peaks: Union[int, float] = 0
    min_line_scores: float = 0.25
    videos: Optional[List[sio.Video]] = attrs.field(default=None)
    skeletons: Optional[List[sio.Skeleton]] = attrs.field(default=None)

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained models and configuration."""
        # get the index of pafs head configs
        paf_idx = [
            x.head_type == "PartAffinityFieldsHead"
            for x in self.bottomup_config.model_config.head_configs
        ].index(True)

        # get the index of confmap head configs
        confmaps_idx = [
            x.head_type == "MultiInstanceConfmapsHead"
            for x in self.bottomup_config.model_config.head_configs
        ].index(True)

        # initialize the paf scorer
        paf_scorer = PAFScorer.from_config(
            config=OmegaConf.create(
                {
                    "confmaps": self.bottomup_config.model_config.head_configs[
                        confmaps_idx
                    ].head_config,
                    "pafs": self.bottomup_config.model_config.head_configs[
                        paf_idx
                    ].head_config,
                }
            ),
            max_edge_length_ratio=(
                self.bottomup_config.inference_config.max_edge_length_ratio
                if "max_edge_length_ratio"
                in self.bottomup_config.inference_config.keys()
                else self.max_edge_length_ratio
            ),
            dist_penalty_weight=(
                self.bottomup_config.inference_config.dist_penalty_weight
                if "dist_penalty_weight" in self.bottomup_config.inference_config.keys()
                else self.dist_penalty_weight
            ),
            n_points=(
                self.bottomup_config.inference_config.n_points
                if "n_points" in self.bottomup_config.inference_config.keys()
                else self.n_points
            ),
            min_instance_peaks=(
                self.bottomup_config.inference_config.min_instance_peaks
                if "min_instance_peaks" in self.bottomup_config.inference_config.keys()
                else self.min_instance_peaks
            ),
            min_line_scores=(
                self.bottomup_config.inference_config.min_line_scores
                if "min_line_scores" in self.bottomup_config.inference_config.keys()
                else self.min_line_scores
            ),
        )

        # initialize the BottomUpInferenceModel
        self.inference_model = BottomUpInferenceModel(
            torch_model=self.bottomup_model,
            paf_scorer=paf_scorer,
            peak_threshold=self.bottomup_config.inference_config.peak_threshold,
            cms_output_stride=(
                self.bottomup_config.inference_config.data.preprocessing.output_stride
            ),
            pafs_output_stride=(
                self.bottomup_config.inference_config.data.preprocessing.pafs_output_stride
            ),
            refinement=self.bottomup_config.inference_config.integral_refinement,
            integral_patch_size=self.bottomup_config.inference_config.integral_patch_size,
            return_confmaps=self.bottomup_config.inference_config.return_confmaps,
            return_pafs=self.bottomup_config.inference_config.return_pafs,
            return_paf_graph=self.bottomup_config.inference_config.return_pafs,
            input_scale=self.bottomup_config.inference_config.data.scale,
        )

    @property
    def data_config(self) -> OmegaConf:
        """Returns data config section from the overall config."""
        return self.bottomup_config.inference_config.data

    @classmethod
    def from_trained_models(
        cls,
        bottomup_ckpt_path: Optional[Text] = None,
    ) -> "BottomUpPredictor":
        """Create predictor from saved models.

        Args:
            bottomup_ckpt_path: Path to a bottom-up ckpt dir with model.ckpt and config.yaml.

        Returns:
            An instance of `BottomUpPredictor` with the loaded models.

        """
        bottomup_config = OmegaConf.load(f"{bottomup_ckpt_path}/training_config.yaml")
        skeletons = get_skeleton_from_config(bottomup_config.data_config.skeletons)
        bottomup_model = BottomUpModel.load_from_checkpoint(
            f"{bottomup_ckpt_path}/best.ckpt",
            config=bottomup_config,
            skeletons=skeletons,
        )
        bottomup_model.to(bottomup_config.inference_config.device)
        bottomup_model.m_device = bottomup_config.inference_config.device

        # create an instance of SingleInstancePredictor class
        obj = cls(
            bottomup_config=bottomup_config,
            bottomup_model=bottomup_model,
            skeletons=skeletons,
        )
        bottomup_config.inference_config.data["skeletons"] = (
            bottomup_config.data_config.skeletons
        )

        obj._initialize_inference_model()
        return obj

    def make_pipeline(self):
        """Make a data loading pipeline.

        Returns:
            Torch DataLoader where each item is a dictionary with key `image` if provider
            is LabelsReader. If provider is VideoReader, this method initiates the reader
            class (doesn't return a pipeline) and the Thread is started in
            Predictor._predict_generator() method.

        Notes:
            This method creates the class attribute `pipeline` and will be
            called automatically when predicting on data from a new source only when the
            provider is LabelsReader.
        """
        self.provider = self.data_config.provider
        if self.provider == "LabelsReader":
            provider = LabelsReader
            data_provider = provider.from_filename(self.data_config.path)
            self.videos = data_provider.labels.videos
            pipeline = Normalizer(data_provider, is_rgb=self.data_config.is_rgb)
            pipeline = SizeMatcher(
                pipeline,
                max_height=self.data_config.max_height,
                max_width=self.data_config.max_width,
                provider=data_provider,
            )
            pipeline = Resizer(pipeline, scale=self.data_config.scale)
            max_stride = (
                self.bottomup_config.model_config.backbone_config.backbone_config.max_stride
            )
            pipeline = PadToStride(pipeline, max_stride=max_stride)

            # Remove duplicates.
            self.pipeline = pipeline.sharding_filter()

            self.pipeline = DataLoader(
                self.pipeline,
                **dict(self.data_config.data_loader),
            )

            return self.pipeline

        elif self.provider == "VideoReader":
            provider = VideoReader
            self.preprocess = True
            self.video_preprocess_config = {
                "batch_size": self.data_config.video_loader.batch_size,
                "scale": self.data_config.scale,
                "is_rgb": self.data_config.is_rgb,
                "max_stride": (
                    self.bottomup_config.model_config.backbone_config.backbone_config.max_stride
                ),
            }
            frame_queue = Queue(
                maxsize=self.data_config.video_loader.queue_maxsize if not None else 16
            )
            self.pipeline = provider.from_filename(
                filename=self.data_config.path,
                frame_buffer=frame_queue,
                start_idx=self.data_config.video_loader.start_idx,
                end_idx=self.data_config.video_loader.end_idx,
            )
            self.videos = [self.pipeline.video]

        else:
            raise Exception(
                "Provider not recognised. Please use either `LabelsReader` or `VideoReader` as provider"
            )

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
        predicted_frames = []

        skeleton_idx = 0
        for ex in generator:
            # loop through each sample in a batch
            for (
                video_idx,
                frame_idx,
                pred_instances,
                pred_values,
                instance_score,
            ) in zip(
                ex["video_idx"],
                ex["frame_idx"],
                ex["pred_instance_peaks"],
                ex["pred_peak_values"],
                ex["instance_scores"],
            ):

                # Loop over instances.
                predicted_instances = []
                for pts, confs, score in zip(
                    pred_instances, pred_values, instance_score
                ):
                    if np.isnan(pts).all():
                        continue

                    predicted_instances.append(
                        sio.PredictedInstance.from_numpy(
                            points=pts,
                            point_scores=confs,
                            instance_score=score,
                            skeleton=self.skeletons[skeleton_idx],
                        )
                    )

                max_instances = (
                    self.bottomup_config.inference_config.data.max_instances
                    if self.bottomup_config.inference_config.data.max_instances
                    is not None
                    else None
                )
                if max_instances is not None:
                    # Filter by score.
                    predicted_instances = sorted(
                        predicted_instances, key=lambda x: x.score, reverse=True
                    )
                    predicted_instances = predicted_instances[
                        : min(max_instances, len(predicted_instances))
                    ]

                predicted_frames.append(
                    sio.LabeledFrame(
                        video=self.videos[video_idx],
                        frame_idx=frame_idx,
                        instances=predicted_instances,
                    )
                )

        pred_labels = sio.Labels(
            videos=self.videos,
            skeletons=self.skeletons,
            labeled_frames=predicted_frames,
        )
        return pred_labels
