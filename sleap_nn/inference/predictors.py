"""Predictors for running inference."""

from collections import defaultdict
from typing import Dict, List, Optional, Union, Iterator, Text
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
import sleap_io as sio
import torchvision.transforms.v2.functional as F
import torch
import attrs
import lightning as L
from queue import Queue
from omegaconf import OmegaConf
from loguru import logger
from sleap_nn.data.providers import LabelsReader, VideoReader
from sleap_nn.data.resizing import (
    resize_image,
    apply_pad_to_stride,
    apply_sizematcher,
    apply_resizer,
)
from sleap_nn.config.utils import get_model_type_from_cfg
from sleap_nn.inference.paf_grouping import PAFScorer
from sleap_nn.training.lightning_modules import (
    TopDownCenteredInstanceLightningModule,
    SingleInstanceLightningModule,
    CentroidLightningModule,
    BottomUpLightningModule,
    BottomUpMultiClassLightningModule,
    TopDownCenteredInstanceMultiClassLightningModule,
)
from sleap_nn.inference.single_instance import SingleInstanceInferenceModel
from sleap_nn.inference.bottomup import (
    BottomUpInferenceModel,
    BottomUpMultiClassInferenceModel,
)
from sleap_nn.inference.topdown import (
    CentroidCrop,
    FindInstancePeaks,
    FindInstancePeaksGroundTruth,
    TopDownMultiClassFindInstancePeaks,
    TopDownInferenceModel,
)
from sleap_nn.inference.utils import get_skeleton_from_config
from sleap_nn.tracking.tracker import Tracker, run_tracker, connect_single_breaks
from sleap_nn.legacy_models import load_legacy_model
from sleap_nn.config.training_job_config import TrainingJobConfig
import rich
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from time import time
import json
import sys


def _filter_user_labeled_frames(
    labels: sio.Labels,
    video: sio.Video,
    frames: Optional[list],
    exclude_user_labeled: bool,
) -> Optional[list]:
    """Filter out user-labeled frames from a frame list.

    This function is used when running inference with VideoReader (video_index specified)
    to implement the exclude_user_labeled functionality.

    Args:
        labels: The Labels object containing labeled frames.
        video: The video to filter frames for.
        frames: List of frame indices to filter. If None, builds list of all frames.
        exclude_user_labeled: If True, filter out user-labeled frames.

    Returns:
        Filtered list of frame indices excluding user-labeled frames if
        exclude_user_labeled is True. Returns original frames if exclude_user_labeled
        is False or if there are no user-labeled frames.
    """
    if not exclude_user_labeled:
        return frames

    # Get user-labeled frame indices for this video
    user_frame_indices = {
        lf.frame_idx for lf in labels.find(video=video) if lf.has_user_instances
    }

    if not user_frame_indices:
        return frames

    # Build full frame list if frames is None
    if frames is None:
        frames = list(range(len(video)))

    # Filter out user-labeled frames
    return [f for f in frames if f not in user_frame_indices]


class RateColumn(rich.progress.ProgressColumn):
    """Renders the progress rate."""

    def render(self, task: "Task") -> rich.progress.Text:
        """Show progress rate."""
        speed = task.speed
        if speed is None:
            return rich.progress.Text("?", style="progress.data.speed")
        return rich.progress.Text(f"{speed:.1f} FPS", style="progress.data.speed")


@attrs.define
class Predictor(ABC):
    """Base interface class for predictors.

    This is the base predictor class for different types of models.

    Attributes:
        preprocess: True if preprocessing (resizing and
            apply_pad_to_stride) should be applied on the frames read in the video reader.
            Default: True.
        preprocess_config: Preprocessing config with keys: [`scale`,
            `ensure_rgb`, `ensure_grayscale`, `scale`, `max_height`, `max_width`, `crop_size`]. Default: {"scale": 1.0,
            "ensure_rgb": False, "ensure_grayscale": False, "max_height": None, "max_width": None, "crop_size": None}
        pipeline: If provider is LabelsReader, pipeline is a `DataLoader` object. If provider
            is VideoReader, pipeline is an instance of `sleap_nn.data.providers.VideoReader`
            class. Default: None.
        inference_model: Instance of one of the inference models ["TopDownInferenceModel",
            "SingleInstanceInferenceModel", "BottomUpInferenceModel"]. Default: None.
        instances_key: If `True`, then instances are appended to the data samples.
        max_stride: The maximum stride of the backbone network, as specified in the model's
            `backbone_config`. This determines the downsampling factor applied by the backbone,
            and is used to ensure that input images are padded or resized to be compatible
            with the model's architecture. Default: 16.
        gui: If True, outputs JSON progress lines for GUI integration instead of
            Rich progress bars. Default: False.
    """

    preprocess: bool = True
    preprocess_config: dict = {
        "scale": 1.0,
        "ensure_rgb": False,
        "ensure_grayscale": False,
        "crop_size": None,
        "max_height": None,
        "max_width": None,
    }
    pipeline: Optional[Union[LabelsReader, VideoReader]] = None
    inference_model: Optional[
        Union[
            TopDownInferenceModel, SingleInstanceInferenceModel, BottomUpInferenceModel
        ]
    ] = None
    instances_key: bool = False
    max_stride: int = 16
    gui: bool = False

    @classmethod
    def from_model_paths(
        cls,
        model_paths: List[Text],
        backbone_ckpt_path: Optional[str] = None,
        head_ckpt_path: Optional[str] = None,
        peak_threshold: Union[float, List[float]] = 0.2,
        integral_refinement: str = "integral",
        integral_patch_size: int = 5,
        batch_size: int = 4,
        max_instances: Optional[int] = None,
        return_confmaps: bool = False,
        device: str = "cpu",
        preprocess_config: Optional[OmegaConf] = None,
        anchor_part: Optional[str] = None,
    ) -> "Predictor":
        """Create the appropriate `Predictor` subclass from from the ckpt path.

        Args:
            model_paths: (List[str]) List of paths to the directory where the best.ckpt (or from SLEAP <=1.4 best_model - only UNet backbone is supported)
                and training_config.yaml (or from SLEAP <=1.4 training_config.json) are saved.
            backbone_ckpt_path: (str) To run inference on any `.ckpt` other than `best.ckpt`
                from the `model_paths` dir, the path to the `.ckpt` file should be passed here.
            head_ckpt_path: (str) Path to `.ckpt` file if a different set of head layer weights
                are to be used. If `None`, the `best.ckpt` from `model_paths` dir is used (or the ckpt
                from `backbone_ckpt_path` if provided.)
            peak_threshold: (float) Minimum confidence threshold. Peaks with values below
                this will be ignored. Default: 0.2. This can also be `List[float]` for topdown
                centroid and centered-instance model, where the first element corresponds
                to centroid model peak finding threshold and the second element is for
                centered-instance model peak finding.
            integral_refinement: If `None`, returns the grid-aligned peaks with no refinement.
                If `"integral"`, peaks will be refined with integral regression.
                Default: "integral".
            integral_patch_size: (int) Size of patches to crop around each rough peak as an
                integer scalar. Default: 5.
            batch_size: (int) Number of samples per batch. Default: 4.
            max_instances: (int) Max number of instances to consider from the predictions.
            return_confmaps: (bool) If `True`, predicted confidence maps will be returned
                along with the predicted peak values and points. Default: False.
            device: (str) Device on which torch.Tensor will be allocated. One of the
                ("cpu", "cuda", "mps").
                Default: "cpu"
            preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
                in the `data_config.preprocessing` section.
            anchor_part: (str) The name of the node to use as the anchor for the centroid. If not
                provided, the anchor part in the `training_config.yaml` is used instead. Default: None.

        Returns:
            A subclass of `Predictor`.

        See also: `SingleInstancePredictor`, `TopDownPredictor`, `BottomUpPredictor`,
            `MoveNetPredictor`, `TopDownMultiClassPredictor`,
            `BottomUpMultiClassPredictor`.
        """
        model_configs = []
        for model_path in model_paths:
            path = Path(model_path)
            if path / "training_config.yaml" in path.iterdir():
                model_configs.append(
                    OmegaConf.load((path / "training_config.yaml").as_posix())
                )
            elif path / "training_config.json" in path.iterdir():
                model_configs.append(
                    TrainingJobConfig.load_sleap_config(
                        (path / "training_config.json").as_posix()
                    )
                )
            else:
                raise ValueError(
                    f"Could not find training_config.yaml or training_config.json in {model_path}"
                )

        model_names = []
        for config in model_configs:
            model_names.append(get_model_type_from_cfg(config=config))

        if "single_instance" in model_names:
            confmap_ckpt_path = model_paths[model_names.index("single_instance")]
            predictor = SingleInstancePredictor.from_trained_models(
                confmap_ckpt_path,
                backbone_ckpt_path=backbone_ckpt_path,
                head_ckpt_path=head_ckpt_path,
                peak_threshold=peak_threshold,
                integral_refinement=integral_refinement,
                integral_patch_size=integral_patch_size,
                batch_size=batch_size,
                return_confmaps=return_confmaps,
                device=device,
                preprocess_config=preprocess_config,
            )

        elif (
            "centroid" in model_names
            or "centered_instance" in model_names
            or "multi_class_topdown" in model_names
        ):
            centroid_ckpt_path = None
            confmap_ckpt_path = None
            if "centroid" in model_names:
                centroid_ckpt_path = model_paths[model_names.index("centroid")]
                predictor = TopDownPredictor.from_trained_models(
                    centroid_ckpt_path=centroid_ckpt_path,
                    confmap_ckpt_path=confmap_ckpt_path,
                    backbone_ckpt_path=backbone_ckpt_path,
                    head_ckpt_path=head_ckpt_path,
                    peak_threshold=peak_threshold,
                    integral_refinement=integral_refinement,
                    integral_patch_size=integral_patch_size,
                    batch_size=batch_size,
                    max_instances=max_instances,
                    return_confmaps=return_confmaps,
                    device=device,
                    preprocess_config=preprocess_config,
                    anchor_part=anchor_part,
                )
            if "centered_instance" in model_names:
                confmap_ckpt_path = model_paths[model_names.index("centered_instance")]
                # create an instance of the TopDown predictor class
                predictor = TopDownPredictor.from_trained_models(
                    centroid_ckpt_path=centroid_ckpt_path,
                    confmap_ckpt_path=confmap_ckpt_path,
                    backbone_ckpt_path=backbone_ckpt_path,
                    head_ckpt_path=head_ckpt_path,
                    peak_threshold=peak_threshold,
                    integral_refinement=integral_refinement,
                    integral_patch_size=integral_patch_size,
                    batch_size=batch_size,
                    max_instances=max_instances,
                    return_confmaps=return_confmaps,
                    device=device,
                    preprocess_config=preprocess_config,
                    anchor_part=anchor_part,
                )
            elif "multi_class_topdown" in model_names:
                confmap_ckpt_path = model_paths[
                    model_names.index("multi_class_topdown")
                ]
                # create an instance of the TopDown predictor class
                predictor = TopDownMultiClassPredictor.from_trained_models(
                    centroid_ckpt_path=centroid_ckpt_path,
                    confmap_ckpt_path=confmap_ckpt_path,
                    backbone_ckpt_path=backbone_ckpt_path,
                    head_ckpt_path=head_ckpt_path,
                    peak_threshold=peak_threshold,
                    integral_refinement=integral_refinement,
                    integral_patch_size=integral_patch_size,
                    batch_size=batch_size,
                    max_instances=max_instances,
                    return_confmaps=return_confmaps,
                    device=device,
                    preprocess_config=preprocess_config,
                    anchor_part=anchor_part,
                )

        elif "bottomup" in model_names:
            bottomup_ckpt_path = model_paths[model_names.index("bottomup")]
            predictor = BottomUpPredictor.from_trained_models(
                bottomup_ckpt_path=bottomup_ckpt_path,
                backbone_ckpt_path=backbone_ckpt_path,
                head_ckpt_path=head_ckpt_path,
                peak_threshold=peak_threshold,
                integral_refinement=integral_refinement,
                integral_patch_size=integral_patch_size,
                batch_size=batch_size,
                max_instances=max_instances,
                return_confmaps=return_confmaps,
                device=device,
                preprocess_config=preprocess_config,
            )

        elif "multi_class_bottomup" in model_names:
            bottomup_ckpt_path = model_paths[model_names.index("multi_class_bottomup")]
            predictor = BottomUpMultiClassPredictor.from_trained_models(
                bottomup_ckpt_path=bottomup_ckpt_path,
                backbone_ckpt_path=backbone_ckpt_path,
                head_ckpt_path=head_ckpt_path,
                peak_threshold=peak_threshold,
                integral_refinement=integral_refinement,
                integral_patch_size=integral_patch_size,
                batch_size=batch_size,
                max_instances=max_instances,
                return_confmaps=return_confmaps,
                device=device,
                preprocess_config=preprocess_config,
            )

        else:
            message = f"Could not create predictor from model paths:\n{model_paths}"
            logger.error(message)
            raise ValueError(message)
        return predictor

    @classmethod
    @abstractmethod
    def from_trained_models(cls, *args, **kwargs):
        """Initialize the Predictor class for certain type of model."""

    @abstractmethod
    def make_pipeline(
        self,
        data_path: str,
        queue_maxsize: int = 8,
        frames: Optional[list] = None,
        only_labeled_frames: bool = False,
        only_suggested_frames: bool = False,
        exclude_user_labeled: bool = False,
        only_predicted_frames: bool = False,
        video_index: Optional[int] = None,
        video_dataset: Optional[str] = None,
        video_input_format: str = "channels_last",
    ):
        """Create the data pipeline."""

    @abstractmethod
    def _initialize_inference_model(self):
        """Initialize the Inference model."""

    def _convert_tensors_to_numpy(self, output):
        """Convert tensors in output dictionary to numpy arrays."""
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                output[k] = output[k].cpu().numpy()
            if isinstance(v, list) and isinstance(v[0], torch.Tensor):
                for n in range(len(v)):
                    v[n] = v[n].cpu().numpy()
        return output

    def _process_batch(self) -> tuple:
        """Process a single batch of frames from the pipeline.

        Returns:
            Tuple of (imgs, fidxs, vidxs, org_szs, instances, eff_scales, done)
            where done is True if the pipeline has finished.
        """
        imgs = []
        fidxs = []
        vidxs = []
        org_szs = []
        instances = []
        eff_scales = []
        done = False

        for _ in range(self.batch_size):
            frame = self.pipeline.frame_buffer.get()
            if frame["image"] is None:
                done = True
                break
            frame["image"], eff_scale = apply_sizematcher(
                frame["image"],
                self.preprocess_config["max_height"],
                self.preprocess_config["max_width"],
            )
            if self.instances_key:
                frame["instances"] = frame["instances"] * eff_scale
            if self.preprocess_config["ensure_rgb"] and frame["image"].shape[-3] != 3:
                frame["image"] = frame["image"].repeat(1, 3, 1, 1)
            elif (
                self.preprocess_config["ensure_grayscale"]
                and frame["image"].shape[-3] != 1
            ):
                frame["image"] = F.rgb_to_grayscale(
                    frame["image"], num_output_channels=1
                )

            eff_scales.append(torch.tensor(eff_scale))
            imgs.append(frame["image"].unsqueeze(dim=0))
            fidxs.append(frame["frame_idx"])
            vidxs.append(frame["video_idx"])
            org_szs.append(frame["orig_size"].unsqueeze(dim=0))
            if self.instances_key:
                instances.append(frame["instances"].unsqueeze(dim=0))

        return imgs, fidxs, vidxs, org_szs, instances, eff_scales, done

    def _run_inference_on_batch(
        self, imgs, fidxs, vidxs, org_szs, instances, eff_scales
    ) -> Iterator[Dict[str, np.ndarray]]:
        """Run inference on a prepared batch of frames.

        Args:
            imgs: List of image tensors.
            fidxs: List of frame indices.
            vidxs: List of video indices.
            org_szs: List of original sizes.
            instances: List of instance tensors.
            eff_scales: List of effective scales.

        Yields:
            Dictionaries containing inference results for each frame.
        """
        # TODO: all preprocessing should be moved into InferenceModels to be exportable.
        imgs = torch.concatenate(imgs, dim=0)
        fidxs = torch.tensor(fidxs, dtype=torch.int32)
        vidxs = torch.tensor(vidxs, dtype=torch.int32)
        org_szs = torch.concatenate(org_szs, dim=0)
        eff_scales = torch.tensor(eff_scales, dtype=torch.float32)
        if self.instances_key:
            instances = torch.concatenate(instances, dim=0)
        ex = {
            "image": imgs,
            "frame_idx": fidxs,
            "video_idx": vidxs,
            "orig_size": org_szs,
            "eff_scale": eff_scales,
        }
        if self.instances_key:
            ex["instances"] = instances
        if self.preprocess:
            scale = self.preprocess_config["scale"]
            if scale != 1.0:
                if self.instances_key:
                    ex["image"], ex["instances"] = apply_resizer(
                        ex["image"], ex["instances"]
                    )
                else:
                    ex["image"] = resize_image(ex["image"], scale)
            ex["image"] = apply_pad_to_stride(ex["image"], self.max_stride)
        outputs_list = self.inference_model(ex)
        if outputs_list is not None:
            for output in outputs_list:
                output = self._convert_tensors_to_numpy(output)
                yield output

    def _predict_generator(self) -> Iterator[Dict[str, np.ndarray]]:
        """Create a generator that yields batches of inference results.

        This method handles creating a pipeline object depending on the model type and
        provider for loading the data, as well as looping over the batches and
        running inference.

        Returns:
            A generator yielding batches predicted results as dictionaries of numpy
            arrays.
        """
        # Initialize inference model if needed.

        if self.inference_model is None:
            self._initialize_inference_model()

        # Loop over data batches.
        self.pipeline.start()
        total_frames = self.pipeline.total_len()

        try:
            if self.gui:
                # GUI mode: emit JSON progress lines
                yield from self._predict_generator_gui(total_frames)
            else:
                # Normal mode: use Rich progress bar
                yield from self._predict_generator_rich(total_frames)

        except KeyboardInterrupt:
            logger.info("Inference interrupted by user")
            raise KeyboardInterrupt

        except Exception as e:
            message = f"Error in _predict_generator: {e}"
            logger.error(message)
            raise Exception(message)

        self.pipeline.join()

    def _predict_generator_gui(
        self, total_frames: int
    ) -> Iterator[Dict[str, np.ndarray]]:
        """Generator for GUI mode with JSON progress output.

        Args:
            total_frames: Total number of frames to process.

        Yields:
            Dictionaries containing inference results for each frame.
        """
        start_time = time()
        frames_processed = 0
        last_report = time()
        done = False

        while not done:
            imgs, fidxs, vidxs, org_szs, instances, eff_scales, done = (
                self._process_batch()
            )

            if imgs:
                yield from self._run_inference_on_batch(
                    imgs, fidxs, vidxs, org_szs, instances, eff_scales
                )

                # Update progress
                num_frames = len(fidxs)
                frames_processed += num_frames

                # Emit JSON progress (throttled to ~4Hz)
                if time() - last_report > 0.25:
                    elapsed = time() - start_time
                    rate = frames_processed / elapsed if elapsed > 0 else 0
                    remaining = total_frames - frames_processed
                    eta = remaining / rate if rate > 0 else 0

                    progress_data = {
                        "n_processed": frames_processed,
                        "n_total": total_frames,
                        "rate": round(rate, 1),
                        "eta": round(eta, 1),
                    }
                    print(json.dumps(progress_data), flush=True)
                    last_report = time()

        # Final progress emit to ensure 100% is shown
        elapsed = time() - start_time
        progress_data = {
            "n_processed": total_frames,
            "n_total": total_frames,
            "rate": round(frames_processed / elapsed, 1) if elapsed > 0 else 0,
            "eta": 0,
        }
        print(json.dumps(progress_data), flush=True)

    def _predict_generator_rich(
        self, total_frames: int
    ) -> Iterator[Dict[str, np.ndarray]]:
        """Generator for normal mode with Rich progress bar.

        Args:
            total_frames: Total number of frames to process.

        Yields:
            Dictionaries containing inference results for each frame.
        """
        with Progress(
            "{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            MofNCompleteColumn(),
            "ETA:",
            TimeRemainingColumn(),
            "Elapsed:",
            TimeElapsedColumn(),
            RateColumn(),
            auto_refresh=False,
            refresh_per_second=4,
            speed_estimate_period=5,
        ) as progress:
            task = progress.add_task("Predicting...", total=total_frames)
            last_report = time()
            done = False

            while not done:
                imgs, fidxs, vidxs, org_szs, instances, eff_scales, done = (
                    self._process_batch()
                )

                if imgs:
                    yield from self._run_inference_on_batch(
                        imgs, fidxs, vidxs, org_szs, instances, eff_scales
                    )

                    # Advance progress
                    num_frames = len(fidxs)
                    progress.update(task, advance=num_frames)

                # Manually refresh progress bar
                if time() - last_report > 0.25:
                    progress.refresh()
                    last_report = time()

        self.pipeline.join()

    def predict(
        self,
        make_labels: bool = True,
    ) -> Union[List[Dict[str, np.ndarray]], sio.Labels]:
        """Run inference on a data source.

        Args:
            make_labels: If `True` (the default), returns a `sio.Labels` instance with
                `sio.PredictedInstance`s. If `False`, just return a list of
                dictionaries containing the raw arrays returned by the inference model.

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
        centroid_backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        centered_instance_backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        videos: List of `sio.Video` objects for creating the `sio.Labels` object from
                        the output predictions.
        skeletons: List of `sio.Skeleton` objects for creating `sio.Labels` object from
                        the output predictions.
        peak_threshold: (float) Minimum confidence threshold. Peaks with values below
                this will be ignored. Default: 0.2. This can also be `List[float]` for topdown
                centroid and centered-instance model, where the first element corresponds
                to centroid model peak finding threshold and the second element is for
                centered-instance model peak finding.
        integral_refinement: If `None`, returns the grid-aligned peaks with no refinement.
            If `"integral"`, peaks will be refined with integral regression.
            Default: "integral".
        integral_patch_size: (int) Size of patches to crop around each rough peak as an
            integer scalar. Default: 5.
        batch_size: (int) Number of samples per batch. Default: 4.
        max_instances: (int) Max number of instances to consider from the predictions.
        return_confmaps: (bool) If `True`, predicted confidence maps will be returned
            along with the predicted peak values and points. Default: False.
        device: (str) Device on which torch.Tensor will be allocated. One of the
            ("cpu", "cuda", "mps").
            Default: "cpu"
        preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
            in the `data_config.preprocessing` section.
        tracker: A `sleap_nn.tracking.Tracker` that will be called to associate
            detections over time. Predicted instances will not be assigned to tracks if
            if this is `None`.
        anchor_part: (str) The name of the node to use as the anchor for the centroid. If not
            provided, the anchor part in the `training_config.yaml` is used instead. Default: None.
        max_stride: The maximum stride of the backbone network, as specified in the model's
            `backbone_config`. This determines the downsampling factor applied by the backbone,
            and is used to ensure that input images are padded or resized to be compatible
            with the model's architecture. Default: 16.

    """

    centroid_config: Optional[OmegaConf] = None
    confmap_config: Optional[OmegaConf] = None
    centroid_model: Optional[L.LightningModule] = None
    confmap_model: Optional[L.LightningModule] = None
    centroid_backbone_type: Optional[str] = None
    centered_instance_backbone_type: Optional[str] = None
    videos: Optional[List[sio.Video]] = None
    skeletons: Optional[List[sio.Skeleton]] = None
    peak_threshold: Union[float, List[float]] = 0.2
    integral_refinement: str = "integral"
    integral_patch_size: int = 5
    batch_size: int = 4
    max_instances: Optional[int] = None
    return_confmaps: bool = False
    device: str = "cpu"
    preprocess_config: Optional[OmegaConf] = None
    tracker: Optional[Tracker] = None
    anchor_part: Optional[str] = None
    max_stride: int = 16

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained models and configuration."""
        # Create an instance of CentroidLayer if centroid_config is not None
        return_crops = False
        # if both centroid and centered-instance model are provided, set return crops to True
        if self.confmap_model:
            return_crops = True
        if isinstance(self.peak_threshold, list):
            centroid_peak_threshold = self.peak_threshold[0]
            centered_instance_peak_threshold = self.peak_threshold[1]
        else:
            centroid_peak_threshold = self.peak_threshold
            centered_instance_peak_threshold = self.peak_threshold

        if self.anchor_part is not None:
            anchor_ind = self.skeletons[0].node_names.index(self.anchor_part)
        else:
            anch_pt = None
            if self.centroid_config is not None:
                anch_pt = (
                    self.centroid_config.model_config.head_configs.centroid.confmaps.anchor_part
                )
            if self.confmap_config is not None:
                anch_pt = (
                    self.confmap_config.model_config.head_configs.centered_instance.confmaps.anchor_part
                )
            anchor_ind = (
                self.skeletons[0].node_names.index(anch_pt)
                if anch_pt is not None
                else None
            )

        if self.centroid_config is None:
            centroid_crop_layer = CentroidCrop(
                use_gt_centroids=True,
                crop_hw=(
                    self.preprocess_config.crop_size,
                    self.preprocess_config.crop_size,
                ),
                anchor_ind=anchor_ind,
                return_crops=return_crops,
            )

        else:
            max_stride = self.centroid_config.model_config.backbone_config[
                f"{self.centroid_backbone_type}"
            ]["max_stride"]
            # initialize centroid crop layer
            centroid_crop_layer = CentroidCrop(
                torch_model=self.centroid_model,
                peak_threshold=centroid_peak_threshold,
                output_stride=self.centroid_config.model_config.head_configs.centroid.confmaps.output_stride,
                refinement=self.integral_refinement,
                integral_patch_size=self.integral_patch_size,
                return_confmaps=self.return_confmaps,
                return_crops=return_crops,
                max_instances=self.max_instances,
                max_stride=max_stride,
                input_scale=self.centroid_config.data_config.preprocessing.scale,
                crop_hw=(
                    self.preprocess_config.crop_size,
                    self.preprocess_config.crop_size,
                ),
                use_gt_centroids=False,
            )

        # Create an instance of FindInstancePeaks layer if confmap_config is not None
        if self.confmap_config is None:
            instance_peaks_layer = FindInstancePeaksGroundTruth()
            self.instances_key = True
        else:
            max_stride = self.confmap_config.model_config.backbone_config[
                f"{self.centered_instance_backbone_type}"
            ]["max_stride"]
            instance_peaks_layer = FindInstancePeaks(
                torch_model=self.confmap_model,
                peak_threshold=centered_instance_peak_threshold,
                output_stride=self.confmap_config.model_config.head_configs.centered_instance.confmaps.output_stride,
                refinement=self.integral_refinement,
                integral_patch_size=self.integral_patch_size,
                return_confmaps=self.return_confmaps,
                max_stride=max_stride,
                input_scale=self.confmap_config.data_config.preprocessing.scale,
            )

        if self.centroid_config is None and self.confmap_config is not None:
            self.instances_key = (
                True  # we need `instances` to get ground-truth centroids
            )

        # Initialize the inference model with centroid and instance peak layers
        self.inference_model = TopDownInferenceModel(
            centroid_crop=centroid_crop_layer, instance_peaks=instance_peaks_layer
        )

    @classmethod
    def from_trained_models(
        cls,
        centroid_ckpt_path: Optional[Text] = None,
        confmap_ckpt_path: Optional[Text] = None,
        backbone_ckpt_path: Optional[str] = None,
        head_ckpt_path: Optional[str] = None,
        peak_threshold: float = 0.2,
        integral_refinement: str = "integral",
        integral_patch_size: int = 5,
        batch_size: int = 4,
        max_instances: Optional[int] = None,
        return_confmaps: bool = False,
        device: str = "cpu",
        preprocess_config: Optional[OmegaConf] = None,
        anchor_part: Optional[str] = None,
    ) -> "TopDownPredictor":
        """Create predictor from saved models.

        Args:
            centroid_ckpt_path: Path to a centroid ckpt dir with best.ckpt (or from SLEAP <=1.4 best_model.h5 - only UNet backbone is supported) and training_config.yaml (or from SLEAP <=1.4 training_config.json - only UNet backbone is supported).
            confmap_ckpt_path: Path to a centered-instance ckpt dir with best.ckpt (or from SLEAP <=1.4 best_model.h5 - only UNet backbone is supported) and training_config.yaml (or from SLEAP <=1.4 training_config.json - only UNet backbone is supported).
            backbone_ckpt_path: (str) To run inference on any `.ckpt` other than `best.ckpt`
                from the `model_paths` dir, the path to the `.ckpt` file should be passed here.
            head_ckpt_path: (str) Path to `.ckpt` file if a different set of head layer weights
                are to be used. If `None`, the `best.ckpt` from `model_paths` dir is used (or the ckpt
                from `backbone_ckpt_path` if provided.)
            peak_threshold: (float) Minimum confidence threshold. Peaks with values below
                this will be ignored. Default: 0.2
            integral_refinement: If `None`, returns the grid-aligned peaks with no refinement.
                If `"integral"`, peaks will be refined with integral regression.
                Default: "integral".
            integral_patch_size: (int) Size of patches to crop around each rough peak as an
                integer scalar. Default: 5.
            batch_size: (int) Number of samples per batch. Default: 4.
            max_instances: (int) Max number of instances to consider from the predictions.
            return_confmaps: (bool) If `True`, predicted confidence maps will be returned
                along with the predicted peak values and points. Default: False.
            device: (str) Device on which torch.Tensor will be allocated. One of the
                ("cpu", "cuda", "mps").
                Default: "cpu"
            preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
                in the `data_config.preprocessing` section.
            anchor_part: (str) The name of the node to use as the anchor for the centroid. If not
                provided, the anchor part in the `training_config.yaml` is used instead. Default: None.

        Returns:
            An instance of `TopDownPredictor` with the loaded models.

            One of the two models can be left as `None` to perform inference with ground
            truth data. This will only work with `LabelsReader` as the provider.

        """
        centered_instance_backbone_type = None
        centroid_backbone_type = None
        if centroid_ckpt_path is not None:
            is_sleap_ckpt = False
            # Load centroid model.
            if (
                Path(centroid_ckpt_path) / "training_config.yaml"
                in Path(centroid_ckpt_path).iterdir()
            ):
                centroid_config = OmegaConf.load(
                    (Path(centroid_ckpt_path) / "training_config.yaml").as_posix()
                )
            elif (
                Path(centroid_ckpt_path) / "training_config.json"
                in Path(centroid_ckpt_path).iterdir()
            ):
                is_sleap_ckpt = True
                centroid_config = TrainingJobConfig.load_sleap_config(
                    (Path(centroid_ckpt_path) / "training_config.json").as_posix()
                )

            skeletons = get_skeleton_from_config(centroid_config.data_config.skeletons)

            # check which backbone architecture
            for k, v in centroid_config.model_config.backbone_config.items():
                if v is not None:
                    centroid_backbone_type = k
                    break

            if not is_sleap_ckpt:
                ckpt_path = (Path(centroid_ckpt_path) / "best.ckpt").as_posix()
                centroid_model = CentroidLightningModule.load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    model_type="centroid",
                    backbone_type=centroid_backbone_type,
                    backbone_config=centroid_config.model_config.backbone_config,
                    head_configs=centroid_config.model_config.head_configs,
                    pretrained_backbone_weights=None,
                    pretrained_head_weights=None,
                    init_weights=centroid_config.model_config.init_weights,
                    lr_scheduler=centroid_config.trainer_config.lr_scheduler,
                    online_mining=centroid_config.trainer_config.online_hard_keypoint_mining.online_mining,
                    hard_to_easy_ratio=centroid_config.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
                    min_hard_keypoints=centroid_config.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
                    max_hard_keypoints=centroid_config.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
                    loss_scale=centroid_config.trainer_config.online_hard_keypoint_mining.loss_scale,
                    optimizer=centroid_config.trainer_config.optimizer_name,
                    learning_rate=centroid_config.trainer_config.optimizer.lr,
                    amsgrad=centroid_config.trainer_config.optimizer.amsgrad,
                    map_location=device,
                    weights_only=False,
                )
            else:
                # Load the converted model
                centroid_converted_model = load_legacy_model(
                    model_dir=f"{centroid_ckpt_path}"
                )
                centroid_model = CentroidLightningModule(
                    backbone_type=centroid_backbone_type,
                    model_type="centroid",
                    backbone_config=centroid_config.model_config.backbone_config,
                    head_configs=centroid_config.model_config.head_configs,
                    pretrained_backbone_weights=None,
                    pretrained_head_weights=None,
                    init_weights=centroid_config.model_config.init_weights,
                    lr_scheduler=centroid_config.trainer_config.lr_scheduler,
                    online_mining=centroid_config.trainer_config.online_hard_keypoint_mining.online_mining,
                    hard_to_easy_ratio=centroid_config.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
                    min_hard_keypoints=centroid_config.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
                    max_hard_keypoints=centroid_config.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
                    loss_scale=centroid_config.trainer_config.online_hard_keypoint_mining.loss_scale,
                    optimizer=centroid_config.trainer_config.optimizer_name,
                    learning_rate=centroid_config.trainer_config.optimizer.lr,
                    amsgrad=centroid_config.trainer_config.optimizer.amsgrad,
                )

                centroid_model.eval()
                centroid_model.model = centroid_converted_model
                centroid_model.to(device)

            centroid_model.eval()

            if backbone_ckpt_path is not None and head_ckpt_path is not None:
                logger.info(f"Loading backbone weights from `{backbone_ckpt_path}` ...")
                ckpt = torch.load(
                    backbone_ckpt_path, map_location=device, weights_only=False
                )
                ckpt["state_dict"] = {
                    k: ckpt["state_dict"][k]
                    for k in ckpt["state_dict"].keys()
                    if ".backbone" in k
                }
                centroid_model.load_state_dict(ckpt["state_dict"], strict=False)

            elif backbone_ckpt_path is not None:
                logger.info(f"Loading weights from `{backbone_ckpt_path}` ...")
                ckpt = torch.load(
                    backbone_ckpt_path, map_location=device, weights_only=False
                )
                centroid_model.load_state_dict(ckpt["state_dict"], strict=False)

            if head_ckpt_path is not None:
                logger.info(f"Loading head weights from `{head_ckpt_path}` ...")
                ckpt = torch.load(
                    head_ckpt_path, map_location=device, weights_only=False
                )
                ckpt["state_dict"] = {
                    k: ckpt["state_dict"][k]
                    for k in ckpt["state_dict"].keys()
                    if ".head_layers" in k
                }
                centroid_model.load_state_dict(ckpt["state_dict"], strict=False)

            centroid_model.to(device)

        else:
            centroid_config = None
            centroid_model = None

        if confmap_ckpt_path is not None:
            is_sleap_ckpt = False
            # Load confmap model.
            if (
                Path(confmap_ckpt_path) / "training_config.yaml"
                in Path(confmap_ckpt_path).iterdir()
            ):
                confmap_config = OmegaConf.load(
                    (Path(confmap_ckpt_path) / "training_config.yaml").as_posix()
                )
            elif (
                Path(confmap_ckpt_path) / "training_config.json"
                in Path(confmap_ckpt_path).iterdir()
            ):
                is_sleap_ckpt = True
                confmap_config = TrainingJobConfig.load_sleap_config(
                    (Path(confmap_ckpt_path) / "training_config.json").as_posix()
                )

            skeletons = get_skeleton_from_config(confmap_config.data_config.skeletons)

            # check which backbone architecture
            for k, v in confmap_config.model_config.backbone_config.items():
                if v is not None:
                    centered_instance_backbone_type = k
                    break

            if not is_sleap_ckpt:
                ckpt_path = (Path(confmap_ckpt_path) / "best.ckpt").as_posix()
                confmap_model = TopDownCenteredInstanceLightningModule.load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    model_type="centered_instance",
                    backbone_config=confmap_config.model_config.backbone_config,
                    head_configs=confmap_config.model_config.head_configs,
                    pretrained_backbone_weights=None,
                    pretrained_head_weights=None,
                    init_weights=confmap_config.model_config.init_weights,
                    lr_scheduler=confmap_config.trainer_config.lr_scheduler,
                    online_mining=confmap_config.trainer_config.online_hard_keypoint_mining.online_mining,
                    hard_to_easy_ratio=confmap_config.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
                    min_hard_keypoints=confmap_config.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
                    max_hard_keypoints=confmap_config.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
                    loss_scale=confmap_config.trainer_config.online_hard_keypoint_mining.loss_scale,
                    optimizer=confmap_config.trainer_config.optimizer_name,
                    learning_rate=confmap_config.trainer_config.optimizer.lr,
                    amsgrad=confmap_config.trainer_config.optimizer.amsgrad,
                    backbone_type=centered_instance_backbone_type,
                    map_location=device,
                    weights_only=False,
                )
            else:
                # Load the converted model
                confmap_converted_model = load_legacy_model(
                    model_dir=f"{confmap_ckpt_path}"
                )

                # Create a new LightningModule with the converted model
                confmap_model = TopDownCenteredInstanceLightningModule(
                    backbone_type=centered_instance_backbone_type,
                    model_type="centered_instance",
                    backbone_config=confmap_config.model_config.backbone_config,
                    head_configs=confmap_config.model_config.head_configs,
                    pretrained_backbone_weights=None,
                    pretrained_head_weights=None,
                    init_weights=confmap_config.model_config.init_weights,
                    lr_scheduler=confmap_config.trainer_config.lr_scheduler,
                    online_mining=confmap_config.trainer_config.online_hard_keypoint_mining.online_mining,
                    hard_to_easy_ratio=confmap_config.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
                    min_hard_keypoints=confmap_config.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
                    max_hard_keypoints=confmap_config.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
                    loss_scale=confmap_config.trainer_config.online_hard_keypoint_mining.loss_scale,
                    optimizer=confmap_config.trainer_config.optimizer_name,
                    learning_rate=confmap_config.trainer_config.optimizer.lr,
                    amsgrad=confmap_config.trainer_config.optimizer.amsgrad,
                )

                confmap_model.eval()
                confmap_model.model = confmap_converted_model
                confmap_model.to(device)

            confmap_model.eval()

            if backbone_ckpt_path is not None and head_ckpt_path is not None:
                logger.info(f"Loading backbone weights from `{backbone_ckpt_path}` ...")
                ckpt = torch.load(
                    backbone_ckpt_path, map_location=device, weights_only=False
                )
                ckpt["state_dict"] = {
                    k: ckpt["state_dict"][k]
                    for k in ckpt["state_dict"].keys()
                    if ".backbone" in k
                }
                confmap_model.load_state_dict(ckpt["state_dict"], strict=False)

            elif backbone_ckpt_path is not None:
                logger.info(f"Loading weights from `{backbone_ckpt_path}` ...")
                ckpt = torch.load(
                    backbone_ckpt_path, map_location=device, weights_only=False
                )
                confmap_model.load_state_dict(ckpt["state_dict"], strict=False)

            if head_ckpt_path is not None:
                logger.info(f"Loading head weights from `{head_ckpt_path}` ...")
                ckpt = torch.load(
                    head_ckpt_path, map_location=device, weights_only=False
                )
                ckpt["state_dict"] = {
                    k: ckpt["state_dict"][k]
                    for k in ckpt["state_dict"].keys()
                    if ".head_layers" in k
                }
                confmap_model.load_state_dict(ckpt["state_dict"], strict=False)

            confmap_model.to(device)

        else:
            confmap_config = None
            confmap_model = None

        if centroid_config is not None:
            preprocess_config["scale"] = (
                centroid_config.data_config.preprocessing.scale
                if preprocess_config["scale"] is None
                else preprocess_config["scale"]
            )
            preprocess_config["ensure_rgb"] = (
                centroid_config.data_config.preprocessing.ensure_rgb
                if preprocess_config["ensure_rgb"] is None
                else preprocess_config["ensure_rgb"]
            )
            preprocess_config["ensure_grayscale"] = (
                centroid_config.data_config.preprocessing.ensure_grayscale
                if preprocess_config["ensure_grayscale"] is None
                else preprocess_config["ensure_grayscale"]
            )
            preprocess_config["max_height"] = (
                centroid_config.data_config.preprocessing.max_height
                if preprocess_config["max_height"] is None
                else preprocess_config["max_height"]
            )
            preprocess_config["max_width"] = (
                centroid_config.data_config.preprocessing.max_width
                if preprocess_config["max_width"] is None
                else preprocess_config["max_width"]
            )

        else:
            preprocess_config["scale"] = (
                confmap_config.data_config.preprocessing.scale
                if preprocess_config["scale"] is None
                else preprocess_config["scale"]
            )
            preprocess_config["ensure_rgb"] = (
                confmap_config.data_config.preprocessing.ensure_rgb
                if preprocess_config["ensure_rgb"] is None
                else preprocess_config["ensure_rgb"]
            )
            preprocess_config["ensure_grayscale"] = (
                confmap_config.data_config.preprocessing.ensure_grayscale
                if preprocess_config["ensure_grayscale"] is None
                else preprocess_config["ensure_grayscale"]
            )
            preprocess_config["max_height"] = (
                confmap_config.data_config.preprocessing.max_height
                if preprocess_config["max_height"] is None
                else preprocess_config["max_height"]
            )
            preprocess_config["max_width"] = (
                confmap_config.data_config.preprocessing.max_width
                if preprocess_config["max_width"] is None
                else preprocess_config["max_width"]
            )

        preprocess_config["crop_size"] = (
            confmap_config.data_config.preprocessing.crop_size
            if preprocess_config["crop_size"] is None and confmap_config is not None
            else preprocess_config["crop_size"]
        )

        # create an instance of TopDownPredictor class
        obj = cls(
            centroid_config=centroid_config,
            centroid_model=centroid_model,
            confmap_config=confmap_config,
            confmap_model=confmap_model,
            centroid_backbone_type=centroid_backbone_type,
            centered_instance_backbone_type=centered_instance_backbone_type,
            skeletons=skeletons,
            peak_threshold=peak_threshold,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            batch_size=batch_size,
            max_instances=max_instances,
            return_confmaps=return_confmaps,
            device=device,
            preprocess_config=preprocess_config,
            anchor_part=anchor_part,
            max_stride=(
                centroid_config.model_config.backbone_config[
                    f"{centroid_backbone_type}"
                ]["max_stride"]
                if centroid_config is not None
                else confmap_config.model_config.backbone_config[
                    f"{centered_instance_backbone_type}"
                ]["max_stride"]
            ),
        )

        obj._initialize_inference_model()
        return obj

    def make_pipeline(
        self,
        inference_object: Union[str, Path, sio.Labels, sio.Video],
        queue_maxsize: int = 8,
        frames: Optional[list] = None,
        only_labeled_frames: bool = False,
        only_suggested_frames: bool = False,
        exclude_user_labeled: bool = False,
        only_predicted_frames: bool = False,
        video_index: Optional[int] = None,
        video_dataset: Optional[str] = None,
        video_input_format: str = "channels_last",
    ):
        """Make a data loading pipeline.

        Args:
            inference_object: (str) Path to `.slp` file or `.mp4` or sio.Labels or sio.Video to run inference on.
            queue_maxsize: (int) Maximum size of the frame buffer queue. Default: 8.
            frames: (list) List of frames indices. If `None`, all frames in the video are used. Default: None.
            only_labeled_frames: (bool) `True` if inference should be run only on user-labeled frames. Default: `False`.
            only_suggested_frames: (bool) `True` if inference should be run only on unlabeled suggested frames. Default: `False`.
            exclude_user_labeled: (bool) `True` to skip frames that have user-labeled instances. Default: `False`.
            only_predicted_frames: (bool) `True` to run inference only on frames that already have predictions. Default: `False`.
            video_index: (int) Integer index of video in .slp file to predict on. To be used
                with an .slp path as an alternative to specifying the video path.
            video_dataset: (str) The dataset for HDF5 videos.
            video_input_format: (str) The input_format for HDF5 videos.

        Returns:
            This method initiates the reader class (doesn't return a pipeline) and the
            Thread is started in Predictor._predict_generator() method.
        """
        if isinstance(inference_object, str) or isinstance(inference_object, Path):
            inference_object = (
                sio.load_slp(inference_object)
                if inference_object.endswith(".slp")
                else sio.load_video(
                    inference_object,
                    dataset=video_dataset,
                    input_format=video_input_format,
                )
            )

        # LabelsReader provider
        if isinstance(inference_object, sio.Labels) and video_index is None:
            provider = LabelsReader

            self.preprocess = False

            frame_buffer = Queue(maxsize=queue_maxsize)

            self.pipeline = provider(
                labels=inference_object,
                frame_buffer=frame_buffer,
                instances_key=self.instances_key,
                only_labeled_frames=only_labeled_frames,
                only_suggested_frames=only_suggested_frames,
                exclude_user_labeled=exclude_user_labeled,
                only_predicted_frames=only_predicted_frames,
            )
            self.videos = self.pipeline.labels.videos

        else:
            provider = VideoReader
            if self.centroid_config is None:
                message = (
                    "Ground truth data was not detected... "
                    "Please load both models when predicting on non-ground-truth data."
                )
                logger.error(message)
                raise ValueError(message)

            self.preprocess = False

            if isinstance(inference_object, sio.Labels) and video_index is not None:
                labels = inference_object
                video = labels.videos[video_index]
                # Filter out user-labeled frames if requested
                filtered_frames = _filter_user_labeled_frames(
                    labels, video, frames, exclude_user_labeled
                )
                self.pipeline = provider.from_video(
                    video=video,
                    queue_maxsize=queue_maxsize,
                    frames=filtered_frames,
                )

            else:  # for mp4 or hdf5 videos
                frame_buffer = Queue(maxsize=queue_maxsize)
                self.pipeline = provider(
                    video=inference_object,
                    frame_buffer=frame_buffer,
                    frames=frames,
                )

            self.videos = [self.pipeline.video]

    def _make_labeled_frames_from_generator(
        self,
        generator: Iterator[Dict[str, np.ndarray]],
    ) -> sio.Labels:
        """Create labeled frames from a generator that yields inference results.

        This method converts pure arrays into SLEAP-specific data structures and assigns
        tracks to the predicted instances if tracker is specified.

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
        # open video backend for tracking
        for video in self.videos:
            if not video.open_backend:
                video.open()

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
                if np.isnan(pred_instances).all():
                    continue
                pred_instances = pred_instances + bbox.squeeze(axis=0)[0, :]
                preds[(int(video_idx), int(frame_idx))].append(
                    sio.PredictedInstance.from_numpy(
                        points_data=pred_instances,
                        skeleton=self.skeletons[skeleton_idx],
                        point_scores=pred_values,
                        score=instance_score,
                    )
                )
        for key, inst in preds.items():
            # Create list of LabeledFrames.
            video_idx, frame_idx = key
            lf = sio.LabeledFrame(
                video=self.videos[video_idx],
                frame_idx=frame_idx,
                instances=inst,
            )

            if self.tracker:
                lf.instances = self.tracker.track(
                    untracked_instances=inst, frame_idx=frame_idx, image=lf.image
                )

            predicted_frames.append(lf)

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
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        videos: List of `sio.Video` objects for creating the `sio.Labels` object from
                        the output predictions.
        skeletons: List of `sio.Skeleton` objects for creating `sio.Labels` object from
                        the output predictions.
        peak_threshold: (float) Minimum confidence threshold. Peaks with values below
            this will be ignored. Default: 0.2
        integral_refinement: If `None`, returns the grid-aligned peaks with no refinement.
            If `"integral"`, peaks will be refined with integral regression.
            Default: "integral".
        integral_patch_size: (int) Size of patches to crop around each rough peak as an
            integer scalar. Default: 5.
        batch_size: (int) Number of samples per batch. Default: 4.
        return_confmaps: (bool) If `True`, predicted confidence maps will be returned
            along with the predicted peak values and points. Default: False.
        device: (str) Device on which torch.Tensor will be allocated. One of the
            ("cpu", "cuda", "mps").
            Default: "cpu"
        preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
                in the `data_config.preprocessing` section.
        max_stride: The maximum stride of the backbone network, as specified in the model's
            `backbone_config`. This determines the downsampling factor applied by the backbone,
            and is used to ensure that input images are padded or resized to be compatible
            with the model's architecture. Default: 16.

    """

    confmap_config: Optional[OmegaConf] = attrs.field(default=None)
    confmap_model: Optional[L.LightningModule] = attrs.field(default=None)
    backbone_type: str = "unet"
    videos: Optional[List[sio.Video]] = attrs.field(default=None)
    skeletons: Optional[List[sio.Skeleton]] = attrs.field(default=None)
    peak_threshold: float = 0.2
    integral_refinement: str = "integral"
    integral_patch_size: int = 5
    batch_size: int = 4
    return_confmaps: bool = False
    device: str = "cpu"
    preprocess_config: Optional[OmegaConf] = None
    max_stride: int = 16

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained models and configuration."""
        self.inference_model = SingleInstanceInferenceModel(
            torch_model=self.confmap_model,
            peak_threshold=self.peak_threshold,
            output_stride=self.confmap_config.model_config.head_configs.single_instance.confmaps.output_stride,
            refinement=self.integral_refinement,
            integral_patch_size=self.integral_patch_size,
            return_confmaps=self.return_confmaps,
            input_scale=self.confmap_config.data_config.preprocessing.scale,
        )

    @classmethod
    def from_trained_models(
        cls,
        confmap_ckpt_path: Optional[Text] = None,
        backbone_ckpt_path: Optional[str] = None,
        head_ckpt_path: Optional[str] = None,
        peak_threshold: float = 0.2,
        integral_refinement: str = "integral",
        integral_patch_size: int = 5,
        batch_size: int = 4,
        return_confmaps: bool = False,
        device: str = "cpu",
        preprocess_config: Optional[OmegaConf] = None,
        max_stride: int = 16,
    ) -> "SingleInstancePredictor":
        """Create predictor from saved models.

        Args:
            confmap_ckpt_path: Path to a single instance ckpt dir with best.ckpt (or from SLEAP <=1.4 best_model.h5 - only UNet backbone is supported) and training_config.yaml (or from SLEAP <=1.4 training_config.json - only UNet backbone is supported).
            backbone_ckpt_path: (str) To run inference on any `.ckpt` other than `best.ckpt`
                from the `model_paths` dir, the path to the `.ckpt` file should be passed here.
            head_ckpt_path: (str) Path to `.ckpt` file if a different set of head layer weights
                are to be used. If `None`, the `best.ckpt` from `model_paths` dir is used (or the ckpt
                from `backbone_ckpt_path` if provided.)
            peak_threshold: (float) Minimum confidence threshold. Peaks with values below
                this will be ignored. Default: 0.2
            integral_refinement: If `None`, returns the grid-aligned peaks with no refinement.
                If `"integral"`, peaks will be refined with integral regression.
                Default: "integral".
            integral_patch_size: (int) Size of patches to crop around each rough peak as an
                integer scalar. Default: 5.
            batch_size: (int) Number of samples per batch. Default: 4.
            return_confmaps: (bool) If `True`, predicted confidence maps will be returned
                along with the predicted peak values and points. Default: False.
            device: (str) Device on which torch.Tensor will be allocated. One of the
                ("cpu", "cuda", "mps").
                Default: "cpu"
            preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
                in the `data_config.preprocessing` section.
            max_stride: The maximum stride of the backbone network, as specified in the model's
                `backbone_config`. This determines the downsampling factor applied by the backbone,
                and is used to ensure that input images are padded or resized to be compatible
                with the model's architecture. Default: 16.

        Returns:
            An instance of `SingleInstancePredictor` with the loaded models.

        """
        is_sleap_ckpt = False
        if (
            Path(confmap_ckpt_path) / "training_config.yaml"
            in Path(confmap_ckpt_path).iterdir()
        ):
            confmap_config = OmegaConf.load(
                (Path(confmap_ckpt_path) / "training_config.yaml").as_posix()
            )
        elif (
            Path(confmap_ckpt_path) / "training_config.json"
            in Path(confmap_ckpt_path).iterdir()
        ):
            is_sleap_ckpt = True
            confmap_config = TrainingJobConfig.load_sleap_config(
                (Path(confmap_ckpt_path) / "training_config.json").as_posix()
            )

        # check which backbone architecture
        for k, v in confmap_config.model_config.backbone_config.items():
            if v is not None:
                backbone_type = k
                break

        if not is_sleap_ckpt:
            ckpt_path = (Path(confmap_ckpt_path) / "best.ckpt").as_posix()
            confmap_model = SingleInstanceLightningModule.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                model_type="single_instance",
                backbone_config=confmap_config.model_config.backbone_config,
                head_configs=confmap_config.model_config.head_configs,
                pretrained_backbone_weights=None,
                pretrained_head_weights=None,
                init_weights=confmap_config.model_config.init_weights,
                lr_scheduler=confmap_config.trainer_config.lr_scheduler,
                backbone_type=backbone_type,
                online_mining=confmap_config.trainer_config.online_hard_keypoint_mining.online_mining,
                hard_to_easy_ratio=confmap_config.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
                min_hard_keypoints=confmap_config.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
                max_hard_keypoints=confmap_config.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
                loss_scale=confmap_config.trainer_config.online_hard_keypoint_mining.loss_scale,
                optimizer=confmap_config.trainer_config.optimizer_name,
                learning_rate=confmap_config.trainer_config.optimizer.lr,
                amsgrad=confmap_config.trainer_config.optimizer.amsgrad,
                map_location=device,
                weights_only=False,
            )
        else:
            confmap_converted_model = load_legacy_model(
                model_dir=f"{confmap_ckpt_path}"
            )
            confmap_model = SingleInstanceLightningModule(
                backbone_type=backbone_type,
                backbone_config=confmap_config.model_config.backbone_config,
                head_configs=confmap_config.model_config.head_configs,
                pretrained_backbone_weights=None,
                pretrained_head_weights=None,
                init_weights=confmap_config.model_config.init_weights,
                lr_scheduler=confmap_config.trainer_config.lr_scheduler,
                online_mining=confmap_config.trainer_config.online_hard_keypoint_mining.online_mining,
                hard_to_easy_ratio=confmap_config.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
                min_hard_keypoints=confmap_config.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
                max_hard_keypoints=confmap_config.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
                loss_scale=confmap_config.trainer_config.online_hard_keypoint_mining.loss_scale,
                optimizer=confmap_config.trainer_config.optimizer_name,
                learning_rate=confmap_config.trainer_config.optimizer.lr,
                amsgrad=confmap_config.trainer_config.optimizer.amsgrad,
                model_type="single_instance",
            )
            confmap_model.eval()
            confmap_model.model = confmap_converted_model
            confmap_model.to(device)

        confmap_model.eval()

        skeletons = get_skeleton_from_config(confmap_config.data_config.skeletons)

        if backbone_ckpt_path is not None and head_ckpt_path is not None:
            logger.info(f"Loading backbone weights from `{backbone_ckpt_path}` ...")
            ckpt = torch.load(
                backbone_ckpt_path, map_location=device, weights_only=False
            )
            ckpt["state_dict"] = {
                k: ckpt["state_dict"][k]
                for k in ckpt["state_dict"].keys()
                if ".backbone" in k
            }
            confmap_model.load_state_dict(ckpt["state_dict"], strict=False)

        elif backbone_ckpt_path is not None:
            logger.info(f"Loading weights from `{backbone_ckpt_path}` ...")
            ckpt = torch.load(
                backbone_ckpt_path, map_location=device, weights_only=False
            )
            confmap_model.load_state_dict(ckpt["state_dict"], strict=False)

        if head_ckpt_path is not None:
            logger.info(f"Loading head weights from `{head_ckpt_path}` ...")
            ckpt = torch.load(head_ckpt_path, map_location=device, weights_only=False)
            ckpt["state_dict"] = {
                k: ckpt["state_dict"][k]
                for k in ckpt["state_dict"].keys()
                if ".head_layers" in k
            }
            confmap_model.load_state_dict(ckpt["state_dict"], strict=False)
        confmap_model.to(device)

        for k, v in preprocess_config.items():
            if v is None:
                preprocess_config[k] = (
                    confmap_config.data_config.preprocessing[k]
                    if k in confmap_config.data_config.preprocessing
                    else None
                )

        # create an instance of SingleInstancePredictor class
        obj = cls(
            confmap_config=confmap_config,
            confmap_model=confmap_model,
            backbone_type=backbone_type,
            skeletons=skeletons,
            peak_threshold=peak_threshold,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            batch_size=batch_size,
            return_confmaps=return_confmaps,
            device=device,
            preprocess_config=preprocess_config,
            max_stride=confmap_config.model_config.backbone_config[f"{backbone_type}"][
                "max_stride"
            ],
        )

        obj._initialize_inference_model()
        return obj

    def make_pipeline(
        self,
        inference_object: Union[str, Path, sio.Labels, sio.Video],
        queue_maxsize: int = 8,
        frames: Optional[list] = None,
        only_labeled_frames: bool = False,
        only_suggested_frames: bool = False,
        exclude_user_labeled: bool = False,
        only_predicted_frames: bool = False,
        video_index: Optional[int] = None,
        video_dataset: Optional[str] = None,
        video_input_format: str = "channels_last",
    ):
        """Make a data loading pipeline.

        Args:
            inference_object: (str) Path to `.slp` file or `.mp4` or sio.Labels or sio.Video to run inference on.
            queue_maxsize: (int) Maximum size of the frame buffer queue. Default: 8.
            frames: List of frames indices. If `None`, all frames in the video are used. Default: None.
            only_labeled_frames: (bool) `True` if inference should be run only on user-labeled frames. Default: `False`.
            only_suggested_frames: (bool) `True` if inference should be run only on unlabeled suggested frames. Default: `False`.
            exclude_user_labeled: (bool) `True` to skip frames that have user-labeled instances. Default: `False`.
            only_predicted_frames: (bool) `True` to run inference only on frames that already have predictions. Default: `False`.
            video_index: (int) Integer index of video in .slp file to predict on. To be used
                with an .slp path as an alternative to specifying the video path.
            video_dataset: (str) The dataset for HDF5 videos.
            video_input_format: (str) The input_format for HDF5 videos.

        Returns:
            This method initiates the reader class (doesn't return a pipeline) and the
            Thread is started in Predictor._predict_generator() method.

        """
        if isinstance(inference_object, str) or isinstance(inference_object, Path):
            inference_object = (
                sio.load_slp(inference_object)
                if inference_object.endswith(".slp")
                else sio.load_video(
                    inference_object,
                    dataset=video_dataset,
                    input_format=video_input_format,
                )
            )

        self.preprocess = True
        # LabelsReader provider
        if isinstance(inference_object, sio.Labels) and video_index is None:
            provider = LabelsReader

            frame_buffer = Queue(maxsize=queue_maxsize)

            self.pipeline = provider(
                labels=inference_object,
                frame_buffer=frame_buffer,
                only_labeled_frames=only_labeled_frames,
                only_suggested_frames=only_suggested_frames,
                exclude_user_labeled=exclude_user_labeled,
                only_predicted_frames=only_predicted_frames,
            )
            self.videos = self.pipeline.labels.videos

        else:
            provider = VideoReader

            if isinstance(inference_object, sio.Labels) and video_index is not None:
                labels = inference_object
                video = labels.videos[video_index]
                # Filter out user-labeled frames if requested
                filtered_frames = _filter_user_labeled_frames(
                    labels, video, frames, exclude_user_labeled
                )
                self.pipeline = provider.from_video(
                    video=video,
                    queue_maxsize=queue_maxsize,
                    frames=filtered_frames,
                )

            else:  # for mp4 or hdf5 videos
                frame_buffer = Queue(maxsize=queue_maxsize)
                self.pipeline = provider(
                    video=inference_object,
                    frame_buffer=frame_buffer,
                    frames=frames,
                )

            self.videos = [self.pipeline.video]

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
        # open video backend for tracking
        for video in self.videos:
            if not video.open_backend:
                video.open()

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
                if np.isnan(pred_instances).all():
                    continue
                inst = sio.PredictedInstance.from_numpy(
                    points_data=pred_instances,
                    skeleton=self.skeletons[skeleton_idx],
                    score=np.nansum(pred_values),
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
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
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
        peak_threshold: (float) Minimum confidence threshold. Peaks with values below
            this will be ignored. Default: 0.2
        integral_refinement: If `None`, returns the grid-aligned peaks with no refinement.
            If `"integral"`, peaks will be refined with integral regression.
            Default: "integral".
        integral_patch_size: (int) Size of patches to crop around each rough peak as an
            integer scalar. Default: 5.
        batch_size: (int) Number of samples per batch. Default: 4.
        max_instances: (int) Max number of instances to consider from the predictions.
        return_confmaps: (bool) If `True`, predicted confidence maps will be returned
            along with the predicted peak values and points. Default: False.
        device: (str) Device on which torch.Tensor will be allocated. One of the
            ("cpu", "cuda", "mps").
            Default: "cpu".
        preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
                in the `data_config.preprocessing` section.
        tracker: A `sleap.nn.tracking.Tracker` that will be called to associate
            detections over time. Predicted instances will not be assigned to tracks if
            if this is `None`.
        max_stride: The maximum stride of the backbone network, as specified in the model's
            `backbone_config`. This determines the downsampling factor applied by the backbone,
            and is used to ensure that input images are padded or resized to be compatible
            with the model's architecture. Default: 16.

    """

    bottomup_config: Optional[OmegaConf] = attrs.field(default=None)
    bottomup_model: Optional[L.LightningModule] = attrs.field(default=None)
    backbone_type: str = "unet"
    max_edge_length_ratio: float = 0.25
    dist_penalty_weight: float = 1.0
    n_points: int = 10
    min_instance_peaks: Union[int, float] = 0
    min_line_scores: float = 0.25
    videos: Optional[List[sio.Video]] = attrs.field(default=None)
    skeletons: Optional[List[sio.Skeleton]] = attrs.field(default=None)
    peak_threshold: float = 0.2
    integral_refinement: str = "integral"
    integral_patch_size: int = 5
    batch_size: int = 4
    max_instances: Optional[int] = None
    return_confmaps: bool = False
    device: str = "cpu"
    preprocess_config: Optional[OmegaConf] = None
    tracker: Optional[Tracker] = None
    max_stride: int = 16

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained models and configuration."""
        # initialize the paf scorer
        paf_scorer = PAFScorer.from_config(
            config=OmegaConf.create(
                {
                    "confmaps": self.bottomup_config.model_config.head_configs.bottomup[
                        "confmaps"
                    ],
                    "pafs": self.bottomup_config.model_config.head_configs.bottomup[
                        "pafs"
                    ],
                }
            ),
            max_edge_length_ratio=self.max_edge_length_ratio,
            dist_penalty_weight=self.dist_penalty_weight,
            n_points=self.n_points,
            min_instance_peaks=self.min_instance_peaks,
            min_line_scores=self.min_line_scores,
        )

        # initialize the BottomUpInferenceModel
        self.inference_model = BottomUpInferenceModel(
            torch_model=self.bottomup_model,
            paf_scorer=paf_scorer,
            peak_threshold=self.peak_threshold,
            cms_output_stride=self.bottomup_config.model_config.head_configs.bottomup.confmaps.output_stride,
            pafs_output_stride=self.bottomup_config.model_config.head_configs.bottomup.pafs.output_stride,
            refinement=self.integral_refinement,
            integral_patch_size=self.integral_patch_size,
            return_confmaps=self.return_confmaps,
            input_scale=self.bottomup_config.data_config.preprocessing.scale,
        )

    @classmethod
    def from_trained_models(
        cls,
        bottomup_ckpt_path: Optional[Text] = None,
        backbone_ckpt_path: Optional[str] = None,
        head_ckpt_path: Optional[str] = None,
        peak_threshold: float = 0.2,
        integral_refinement: str = "integral",
        integral_patch_size: int = 5,
        batch_size: int = 4,
        max_instances: Optional[int] = None,
        return_confmaps: bool = False,
        device: str = "cpu",
        preprocess_config: Optional[OmegaConf] = None,
        max_stride: int = 16,
    ) -> "BottomUpPredictor":
        """Create predictor from saved models.

        Args:
            bottomup_ckpt_path: Path to a bottom-up ckpt dir with best.ckpt (or from SLEAP <=1.4 best_model - only UNet backbone is supported) and training_config.yaml (or from SLEAP <=1.4 training_config.json - only UNet backbone is supported).
            backbone_ckpt_path: (str) To run inference on any `.ckpt` other than `best.ckpt`
                from the `model_paths` dir, the path to the `.ckpt` file should be passed here.
            head_ckpt_path: (str) Path to `.ckpt` file if a different set of head layer weights
                    are to be used. If `None`, the `best.ckpt` from `model_paths` dir is used (or the ckpt
                    from `backbone_ckpt_path` if provided.)
            peak_threshold: (float) Minimum confidence threshold. Peaks with values below
                this will be ignored. Default: 0.2
            integral_refinement: If `None`, returns the grid-aligned peaks with no refinement.
                If `"integral"`, peaks will be refined with integral regression.
                Default: "integral".
            integral_patch_size: (int) Size of patches to crop around each rough peak as an
                integer scalar. Default: 5.
            batch_size: (int) Number of samples per batch. Default: 4.
            max_instances: (int) Max number of instances to consider from the predictions.
            return_confmaps: (bool) If `True`, predicted confidence maps will be returned
                along with the predicted peak values and points. Default: False.
            device: (str) Device on which torch.Tensor will be allocated. One of the
                ("cpu", "cuda", "mps").
                Default: "cpu"
            preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
                in the `data_config.preprocessing` section.
            max_stride: The maximum stride of the backbone network, as specified in the model's
                `backbone_config`. This determines the downsampling factor applied by the backbone,
                and is used to ensure that input images are padded or resized to be compatible
                with the model's architecture. Default: 16.

        Returns:
            An instance of `BottomUpPredictor` with the loaded models.

        """
        is_sleap_ckpt = False
        if (
            Path(bottomup_ckpt_path) / "training_config.yaml"
            in Path(bottomup_ckpt_path).iterdir()
        ):
            bottomup_config = OmegaConf.load(
                (Path(bottomup_ckpt_path) / "training_config.yaml").as_posix()
            )
        elif (
            Path(bottomup_ckpt_path) / "training_config.json"
            in Path(bottomup_ckpt_path).iterdir()
        ):
            is_sleap_ckpt = True
            bottomup_config = TrainingJobConfig.load_sleap_config(
                (Path(bottomup_ckpt_path) / "training_config.json").as_posix()
            )

        # check which backbone architecture
        for k, v in bottomup_config.model_config.backbone_config.items():
            if v is not None:
                backbone_type = k
                break

        if not is_sleap_ckpt:
            ckpt_path = (Path(bottomup_ckpt_path) / "best.ckpt").as_posix()

            bottomup_model = BottomUpLightningModule.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                backbone_config=bottomup_config.model_config.backbone_config,
                head_configs=bottomup_config.model_config.head_configs,
                pretrained_backbone_weights=None,
                pretrained_head_weights=None,
                init_weights=bottomup_config.model_config.init_weights,
                lr_scheduler=bottomup_config.trainer_config.lr_scheduler,
                online_mining=bottomup_config.trainer_config.online_hard_keypoint_mining.online_mining,
                hard_to_easy_ratio=bottomup_config.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
                min_hard_keypoints=bottomup_config.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
                max_hard_keypoints=bottomup_config.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
                loss_scale=bottomup_config.trainer_config.online_hard_keypoint_mining.loss_scale,
                optimizer=bottomup_config.trainer_config.optimizer_name,
                learning_rate=bottomup_config.trainer_config.optimizer.lr,
                amsgrad=bottomup_config.trainer_config.optimizer.amsgrad,
                backbone_type=backbone_type,
                model_type="bottomup",
                map_location=device,
                weights_only=False,
            )
        else:
            bottomup_converted_model = load_legacy_model(
                model_dir=f"{bottomup_ckpt_path}"
            )
            bottomup_model = BottomUpLightningModule(
                backbone_config=bottomup_config.model_config.backbone_config,
                head_configs=bottomup_config.model_config.head_configs,
                pretrained_backbone_weights=None,
                pretrained_head_weights=None,
                init_weights=bottomup_config.model_config.init_weights,
                lr_scheduler=bottomup_config.trainer_config.lr_scheduler,
                backbone_type=backbone_type,
                model_type="bottomup",
                online_mining=bottomup_config.trainer_config.online_hard_keypoint_mining.online_mining,
                hard_to_easy_ratio=bottomup_config.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
                min_hard_keypoints=bottomup_config.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
                max_hard_keypoints=bottomup_config.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
                loss_scale=bottomup_config.trainer_config.online_hard_keypoint_mining.loss_scale,
                optimizer=bottomup_config.trainer_config.optimizer_name,
                learning_rate=bottomup_config.trainer_config.optimizer.lr,
                amsgrad=bottomup_config.trainer_config.optimizer.amsgrad,
            )
            bottomup_model.eval()
            bottomup_model.model = bottomup_converted_model
            bottomup_model.to(device)

        bottomup_model.eval()
        skeletons = get_skeleton_from_config(bottomup_config.data_config.skeletons)

        if backbone_ckpt_path is not None and head_ckpt_path is not None:
            logger.info(f"Loading backbone weights from `{backbone_ckpt_path}` ...")
            ckpt = torch.load(
                backbone_ckpt_path, map_location=device, weights_only=False
            )
            ckpt["state_dict"] = {
                k: ckpt["state_dict"][k]
                for k in ckpt["state_dict"].keys()
                if ".backbone" in k
            }
            bottomup_model.load_state_dict(ckpt["state_dict"], strict=False)

        elif backbone_ckpt_path is not None:
            logger.info(f"Loading weights from `{backbone_ckpt_path}` ...")
            ckpt = torch.load(
                backbone_ckpt_path, map_location=device, weights_only=False
            )
            bottomup_model.load_state_dict(ckpt["state_dict"], strict=False)

        if head_ckpt_path is not None:
            logger.info(f"Loading head weights from `{head_ckpt_path}` ...")
            ckpt = torch.load(head_ckpt_path, map_location=device, weights_only=False)
            ckpt["state_dict"] = {
                k: ckpt["state_dict"][k]
                for k in ckpt["state_dict"].keys()
                if ".head_layers" in k
            }
            bottomup_model.load_state_dict(ckpt["state_dict"], strict=False)
        bottomup_model.to(device)

        for k, v in preprocess_config.items():
            if v is None:
                preprocess_config[k] = (
                    bottomup_config.data_config.preprocessing[k]
                    if k in bottomup_config.data_config.preprocessing
                    else None
                )

        # create an instance of BottomUpPredictor class
        obj = cls(
            bottomup_config=bottomup_config,
            backbone_type=backbone_type,
            bottomup_model=bottomup_model,
            skeletons=skeletons,
            peak_threshold=peak_threshold,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            batch_size=batch_size,
            max_instances=max_instances,
            return_confmaps=return_confmaps,
            preprocess_config=preprocess_config,
            max_stride=bottomup_config.model_config.backbone_config[f"{backbone_type}"][
                "max_stride"
            ],
        )

        obj._initialize_inference_model()
        return obj

    def make_pipeline(
        self,
        inference_object: Union[str, Path, sio.Labels, sio.Video],
        queue_maxsize: int = 8,
        frames: Optional[list] = None,
        only_labeled_frames: bool = False,
        only_suggested_frames: bool = False,
        exclude_user_labeled: bool = False,
        only_predicted_frames: bool = False,
        video_index: Optional[int] = None,
        video_dataset: Optional[str] = None,
        video_input_format: str = "channels_last",
    ):
        """Make a data loading pipeline.

        Args:
            inference_object: (str) Path to `.slp` file or `.mp4` or sio.Labels or sio.Video to run inference on.
            queue_maxsize: (int) Maximum size of the frame buffer queue. Default: 8.
            frames: List of frames indices. If `None`, all frames in the video are used. Default: None.
            only_labeled_frames: (bool) `True` if inference should be run only on user-labeled frames. Default: `False`.
            only_suggested_frames: (bool) `True` if inference should be run only on unlabeled suggested frames. Default: `False`.
            exclude_user_labeled: (bool) `True` to skip frames that have user-labeled instances. Default: `False`.
            only_predicted_frames: (bool) `True` to run inference only on frames that already have predictions. Default: `False`.
            video_index: (int) Integer index of video in .slp file to predict on. To be used
                with an .slp path as an alternative to specifying the video path.
            video_dataset: (str) The dataset for HDF5 videos.
            video_input_format: (str) The input_format for HDF5 videos.

        Returns:
            This method initiates the reader class (doesn't return a pipeline) and the
            Thread is started in Predictor._predict_generator() method.
        """
        if isinstance(inference_object, str) or isinstance(inference_object, Path):
            inference_object = (
                sio.load_slp(inference_object)
                if inference_object.endswith(".slp")
                else sio.load_video(
                    inference_object,
                    dataset=video_dataset,
                    input_format=video_input_format,
                )
            )

        self.preprocess = True

        # LabelsReader provider
        if isinstance(inference_object, sio.Labels) and video_index is None:
            provider = LabelsReader

            frame_buffer = Queue(maxsize=queue_maxsize)

            self.pipeline = provider(
                labels=inference_object,
                frame_buffer=frame_buffer,
                only_labeled_frames=only_labeled_frames,
                only_suggested_frames=only_suggested_frames,
                exclude_user_labeled=exclude_user_labeled,
                only_predicted_frames=only_predicted_frames,
            )

            self.videos = self.pipeline.labels.videos

        else:
            provider = VideoReader

            if isinstance(inference_object, sio.Labels) and video_index is not None:
                labels = inference_object
                video = labels.videos[video_index]
                # Filter out user-labeled frames if requested
                filtered_frames = _filter_user_labeled_frames(
                    labels, video, frames, exclude_user_labeled
                )
                self.pipeline = provider.from_video(
                    video=video,
                    queue_maxsize=queue_maxsize,
                    frames=filtered_frames,
                )

            else:  # for mp4 or hdf5 videos
                frame_buffer = Queue(maxsize=queue_maxsize)
                self.pipeline = provider(
                    video=inference_object,
                    frame_buffer=frame_buffer,
                    frames=frames,
                )

            self.videos = [self.pipeline.video]

    def _make_labeled_frames_from_generator(
        self,
        generator: Iterator[Dict[str, np.ndarray]],
    ) -> sio.Labels:
        """Create labeled frames from a generator that yields inference results.

        This method converts pure arrays into SLEAP-specific data structures and assigns
        tracks to the predicted instances if tracker is specified.

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
        # open video backend for tracking
        for video in self.videos:
            if not video.open_backend:
                video.open()

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
                            points_data=pts,
                            point_scores=confs,
                            score=score,
                            skeleton=self.skeletons[skeleton_idx],
                        )
                    )

                max_instances = (
                    self.max_instances if self.max_instances is not None else None
                )
                if max_instances is not None:
                    # Filter by score.
                    predicted_instances = sorted(
                        predicted_instances, key=lambda x: x.score, reverse=True
                    )
                    predicted_instances = predicted_instances[
                        : min(max_instances, len(predicted_instances))
                    ]

                lf = sio.LabeledFrame(
                    video=self.videos[video_idx],
                    frame_idx=frame_idx,
                    instances=predicted_instances,
                )

                if self.tracker:
                    lf.instances = self.tracker.track(
                        untracked_instances=predicted_instances,
                        frame_idx=frame_idx,
                        image=lf.image,
                    )

                predicted_frames.append(lf)

        pred_labels = sio.Labels(
            videos=self.videos,
            skeletons=self.skeletons,
            labeled_frames=predicted_frames,
        )
        return pred_labels


@attrs.define
class BottomUpMultiClassPredictor(Predictor):
    """BottomUp ID model predictor.

    This high-level class handles initialization, preprocessing and predicting using a
    trained BottomUp SLEAP-NN model.

    This should be initialized using the `from_trained_models()` constructor.

    Attributes:
        bottomup_config: A OmegaConfig dictionary with the configs used for training the
                        multi_class_bottomup model.
        bottomup_model: A LightningModule instance created from the trained weights for
                       multi_class_bottomup model.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        videos: List of `sio.Video` objects for creating the `sio.Labels` object from
                        the output predictions.
        skeletons: List of `sio.Skeleton` objects for creating `sio.Labels` object from
                        the output predictions.
        peak_threshold: (float) Minimum confidence threshold. Peaks with values below
            this will be ignored. Default: 0.2
        integral_refinement: If `None`, returns the grid-aligned peaks with no refinement.
            If `"integral"`, peaks will be refined with integral regression.
            Default: "integral".
        integral_patch_size: (int) Size of patches to crop around each rough peak as an
            integer scalar. Default: 5.
        batch_size: (int) Number of samples per batch. Default: 4.
        max_instances: (int) Max number of instances to consider from the predictions.
        return_confmaps: (bool) If `True`, predicted confidence maps will be returned
            along with the predicted peak values and points. Default: False.
        device: (str) Device on which torch.Tensor will be allocated. One of the
            ("cpu", "cuda", "mps").
            Default: "cpu".
        preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
                in the `data_config.preprocessing` section.
        max_stride: The maximum stride of the backbone network, as specified in the model's
            `backbone_config`. This determines the downsampling factor applied by the backbone,
            and is used to ensure that input images are padded or resized to be compatible
            with the model's architecture. Default: 16.

    """

    bottomup_config: Optional[OmegaConf] = attrs.field(default=None)
    bottomup_model: Optional[L.LightningModule] = attrs.field(default=None)
    backbone_type: str = "unet"
    videos: Optional[List[sio.Video]] = attrs.field(default=None)
    skeletons: Optional[List[sio.Skeleton]] = attrs.field(default=None)
    peak_threshold: float = 0.2
    integral_refinement: str = "integral"
    integral_patch_size: int = 5
    batch_size: int = 4
    max_instances: Optional[int] = None
    return_confmaps: bool = False
    device: str = "cpu"
    preprocess_config: Optional[OmegaConf] = None
    max_stride: int = 16

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained models and configuration."""
        # initialize the BottomUpMultiClassInferenceModel
        self.inference_model = BottomUpMultiClassInferenceModel(
            torch_model=self.bottomup_model,
            peak_threshold=self.peak_threshold,
            cms_output_stride=self.bottomup_config.model_config.head_configs.multi_class_bottomup.confmaps.output_stride,
            class_maps_output_stride=self.bottomup_config.model_config.head_configs.multi_class_bottomup.class_maps.output_stride,
            refinement=self.integral_refinement,
            integral_patch_size=self.integral_patch_size,
            return_confmaps=self.return_confmaps,
            input_scale=self.bottomup_config.data_config.preprocessing.scale,
        )

    @classmethod
    def from_trained_models(
        cls,
        bottomup_ckpt_path: Optional[Text] = None,
        backbone_ckpt_path: Optional[str] = None,
        head_ckpt_path: Optional[str] = None,
        peak_threshold: float = 0.2,
        integral_refinement: str = "integral",
        integral_patch_size: int = 5,
        batch_size: int = 4,
        max_instances: Optional[int] = None,
        return_confmaps: bool = False,
        device: str = "cpu",
        preprocess_config: Optional[OmegaConf] = None,
        max_stride: int = 16,
    ) -> "BottomUpMultiClassPredictor":
        """Create predictor from saved models.

        Args:
            bottomup_ckpt_path: Path to a multi-class bottom-up ckpt dir with best.ckpt (or from SLEAP <=1.4 best_model.h5 - only UNet backbone is supported) and training_config.yaml (or from SLEAP <=1.4 training_config.json - only UNet backbone is supported).
            backbone_ckpt_path: (str) To run inference on any `.ckpt` other than `best.ckpt`
                from the `model_paths` dir, the path to the `.ckpt` file should be passed here.
            head_ckpt_path: (str) Path to `.ckpt` file if a different set of head layer weights
                    are to be used. If `None`, the `best.ckpt` from `model_paths` dir is used (or the ckpt
                    from `backbone_ckpt_path` if provided.)
            peak_threshold: (float) Minimum confidence threshold. Peaks with values below
                this will be ignored. Default: 0.2
            integral_refinement: If `None`, returns the grid-aligned peaks with no refinement.
                If `"integral"`, peaks will be refined with integral regression.
                Default: "integral".
            integral_patch_size: (int) Size of patches to crop around each rough peak as an
                integer scalar. Default: 5.
            batch_size: (int) Number of samples per batch. Default: 4.
            max_instances: (int) Max number of instances to consider from the predictions.
            return_confmaps: (bool) If `True`, predicted confidence maps will be returned
                along with the predicted peak values and points. Default: False.
            device: (str) Device on which torch.Tensor will be allocated. One of the
                ("cpu", "cuda", "mps").
                Default: "cpu"
            preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
                in the `data_config.preprocessing` section.
            max_stride: The maximum stride of the backbone network, as specified in the model's
                `backbone_config`. This determines the downsampling factor applied by the backbone,
                and is used to ensure that input images are padded or resized to be compatible
                with the model's architecture. Default: 16.

        Returns:
            An instance of `BottomUpPredictor` with the loaded models.

        """
        is_sleap_ckpt = False
        if (
            Path(bottomup_ckpt_path) / "training_config.yaml"
            in Path(bottomup_ckpt_path).iterdir()
        ):
            bottomup_config = OmegaConf.load(
                (Path(bottomup_ckpt_path) / "training_config.yaml").as_posix()
            )
        elif (
            Path(bottomup_ckpt_path) / "training_config.json"
            in Path(bottomup_ckpt_path).iterdir()
        ):
            is_sleap_ckpt = True
            bottomup_config = TrainingJobConfig.load_sleap_config(
                (Path(bottomup_ckpt_path) / "training_config.json").as_posix()
            )

        # check which backbone architecture
        for k, v in bottomup_config.model_config.backbone_config.items():
            if v is not None:
                backbone_type = k
                break

        if not is_sleap_ckpt:
            ckpt_path = (Path(bottomup_ckpt_path) / "best.ckpt").as_posix()

            bottomup_model = BottomUpMultiClassLightningModule.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                backbone_type=backbone_type,
                model_type="multi_class_bottomup",
                map_location=device,
                backbone_config=bottomup_config.model_config.backbone_config,
                head_configs=bottomup_config.model_config.head_configs,
                pretrained_backbone_weights=None,
                pretrained_head_weights=None,
                init_weights=bottomup_config.model_config.init_weights,
                lr_scheduler=bottomup_config.trainer_config.lr_scheduler,
                online_mining=bottomup_config.trainer_config.online_hard_keypoint_mining.online_mining,
                hard_to_easy_ratio=bottomup_config.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
                min_hard_keypoints=bottomup_config.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
                max_hard_keypoints=bottomup_config.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
                loss_scale=bottomup_config.trainer_config.online_hard_keypoint_mining.loss_scale,
                optimizer=bottomup_config.trainer_config.optimizer_name,
                learning_rate=bottomup_config.trainer_config.optimizer.lr,
                amsgrad=bottomup_config.trainer_config.optimizer.amsgrad,
                weights_only=False,
            )
        else:
            bottomup_converted_model = load_legacy_model(
                model_dir=f"{bottomup_ckpt_path}"
            )
            bottomup_model = BottomUpMultiClassLightningModule(
                backbone_type=backbone_type,
                model_type="multi_class_bottomup",
                backbone_config=bottomup_config.model_config.backbone_config,
                head_configs=bottomup_config.model_config.head_configs,
                pretrained_backbone_weights=None,
                pretrained_head_weights=None,
                init_weights=bottomup_config.model_config.init_weights,
                lr_scheduler=bottomup_config.trainer_config.lr_scheduler,
                online_mining=bottomup_config.trainer_config.online_hard_keypoint_mining.online_mining,
                hard_to_easy_ratio=bottomup_config.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
                min_hard_keypoints=bottomup_config.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
                max_hard_keypoints=bottomup_config.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
                loss_scale=bottomup_config.trainer_config.online_hard_keypoint_mining.loss_scale,
                optimizer=bottomup_config.trainer_config.optimizer_name,
                learning_rate=bottomup_config.trainer_config.optimizer.lr,
                amsgrad=bottomup_config.trainer_config.optimizer.amsgrad,
            )
            bottomup_model.eval()
            bottomup_model.model = bottomup_converted_model
            bottomup_model.to(device)

        bottomup_model.eval()
        skeletons = get_skeleton_from_config(bottomup_config.data_config.skeletons)

        if backbone_ckpt_path is not None and head_ckpt_path is not None:
            logger.info(f"Loading backbone weights from `{backbone_ckpt_path}` ...")
            ckpt = torch.load(
                backbone_ckpt_path,
                map_location=device,
                weights_only=False,
            )
            ckpt["state_dict"] = {
                k: ckpt["state_dict"][k]
                for k in ckpt["state_dict"].keys()
                if ".backbone" in k
            }
            bottomup_model.load_state_dict(ckpt["state_dict"], strict=False)

        elif backbone_ckpt_path is not None:
            logger.info(f"Loading weights from `{backbone_ckpt_path}` ...")
            ckpt = torch.load(
                backbone_ckpt_path,
                map_location=device,
                weights_only=False,
            )
            bottomup_model.load_state_dict(ckpt["state_dict"], strict=False)

        if head_ckpt_path is not None:
            logger.info(f"Loading head weights from `{head_ckpt_path}` ...")
            ckpt = torch.load(
                head_ckpt_path,
                map_location=device,
                weights_only=False,
            )
            ckpt["state_dict"] = {
                k: ckpt["state_dict"][k]
                for k in ckpt["state_dict"].keys()
                if ".head_layers" in k
            }
            bottomup_model.load_state_dict(ckpt["state_dict"], strict=False)
        bottomup_model.to(device)

        for k, v in preprocess_config.items():
            if v is None:
                preprocess_config[k] = (
                    bottomup_config.data_config.preprocessing[k]
                    if k in bottomup_config.data_config.preprocessing
                    else None
                )

        # create an instance of SingleInstancePredictor class
        obj = cls(
            bottomup_config=bottomup_config,
            backbone_type=backbone_type,
            bottomup_model=bottomup_model,
            skeletons=skeletons,
            peak_threshold=peak_threshold,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            batch_size=batch_size,
            max_instances=max_instances,
            return_confmaps=return_confmaps,
            preprocess_config=preprocess_config,
            max_stride=bottomup_config.model_config.backbone_config[f"{backbone_type}"][
                "max_stride"
            ],
        )

        obj._initialize_inference_model()
        return obj

    def make_pipeline(
        self,
        inference_object: Union[str, Path, sio.Labels, sio.Video],
        queue_maxsize: int = 8,
        frames: Optional[list] = None,
        only_labeled_frames: bool = False,
        only_suggested_frames: bool = False,
        exclude_user_labeled: bool = False,
        only_predicted_frames: bool = False,
        video_index: Optional[int] = None,
        video_dataset: Optional[str] = None,
        video_input_format: str = "channels_last",
    ):
        """Make a data loading pipeline.

        Args:
            inference_object: (str) Path to `.slp` file or `.mp4` or sio.Labels or sio.Video to run inference on.
            queue_maxsize: (int) Maximum size of the frame buffer queue. Default: 8.
            frames: List of frames indices. If `None`, all frames in the video are used. Default: None.
            only_labeled_frames: (bool) `True` if inference should be run only on user-labeled frames. Default: `False`.
            only_suggested_frames: (bool) `True` if inference should be run only on unlabeled suggested frames. Default: `False`.
            exclude_user_labeled: (bool) `True` to skip frames that have user-labeled instances. Default: `False`.
            only_predicted_frames: (bool) `True` to run inference only on frames that already have predictions. Default: `False`.
            video_index: (int) Integer index of video in .slp file to predict on. To be used
                with an .slp path as an alternative to specifying the video path.
            video_dataset: (str) The dataset for HDF5 videos.
            video_input_format: (str) The input_format for HDF5 videos.

        Returns:
            This method initiates the reader class (doesn't return a pipeline) and the
            Thread is started in Predictor._predict_generator() method.
        """
        if isinstance(inference_object, str) or isinstance(inference_object, Path):
            inference_object = (
                sio.load_slp(inference_object)
                if inference_object.endswith(".slp")
                else sio.load_video(
                    inference_object,
                    dataset=video_dataset,
                    input_format=video_input_format,
                )
            )

        self.preprocess = True
        # LabelsReader provider
        if isinstance(inference_object, sio.Labels) and video_index is None:
            provider = LabelsReader
            max_stride = self.bottomup_config.model_config.backbone_config[
                f"{self.backbone_type}"
            ]["max_stride"]

            frame_buffer = Queue(maxsize=queue_maxsize)

            self.pipeline = provider(
                labels=inference_object,
                frame_buffer=frame_buffer,
                only_labeled_frames=only_labeled_frames,
                only_suggested_frames=only_suggested_frames,
                exclude_user_labeled=exclude_user_labeled,
                only_predicted_frames=only_predicted_frames,
            )

            self.videos = self.pipeline.labels.videos

        else:
            provider = VideoReader

            if isinstance(inference_object, sio.Labels) and video_index is not None:
                labels = inference_object
                video = labels.videos[video_index]
                # Filter out user-labeled frames if requested
                filtered_frames = _filter_user_labeled_frames(
                    labels, video, frames, exclude_user_labeled
                )
                self.pipeline = provider.from_video(
                    video=video,
                    queue_maxsize=queue_maxsize,
                    frames=filtered_frames,
                )

            else:  # for mp4 or hdf5 videos
                frame_buffer = Queue(maxsize=queue_maxsize)
                self.pipeline = provider(
                    video=inference_object,
                    frame_buffer=frame_buffer,
                    frames=frames,
                )

            self.videos = [self.pipeline.video]

    def _make_labeled_frames_from_generator(
        self,
        generator: Iterator[Dict[str, np.ndarray]],
    ) -> sio.Labels:
        """Create labeled frames from a generator that yields inference results.

        This method converts pure arrays into SLEAP-specific data structures and assigns
        tracks to the predicted instances if tracker is specified.

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
        # open video backend for tracking
        for video in self.videos:
            if not video.open_backend:
                video.open()

        predicted_frames = []
        tracks = [
            sio.Track(name=x)
            for x in self.bottomup_config.model_config.head_configs.multi_class_bottomup.class_maps.classes
        ]

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
                for i, (pts, confs, score) in enumerate(
                    zip(pred_instances, pred_values, instance_score)
                ):
                    if np.isnan(pts).all():
                        continue

                    track = None
                    if tracks is not None and len(tracks) >= (i - 1):
                        track = tracks[i]

                    predicted_instances.append(
                        sio.PredictedInstance.from_numpy(
                            points_data=pts,
                            point_scores=confs,
                            score=np.nanmean(confs),
                            skeleton=self.skeletons[skeleton_idx],
                            track=track,
                            tracking_score=np.nanmean(score),
                        )
                    )

                max_instances = (
                    self.max_instances if self.max_instances is not None else None
                )
                if max_instances is not None:
                    # Filter by score.
                    predicted_instances = sorted(
                        predicted_instances, key=lambda x: x.score, reverse=True
                    )
                    predicted_instances = predicted_instances[
                        : min(max_instances, len(predicted_instances))
                    ]

                lf = sio.LabeledFrame(
                    video=self.videos[video_idx],
                    frame_idx=frame_idx,
                    instances=predicted_instances,
                )

                predicted_frames.append(lf)

        pred_labels = sio.Labels(
            videos=self.videos,
            skeletons=self.skeletons,
            labeled_frames=predicted_frames,
        )
        return pred_labels


@attrs.define
class TopDownMultiClassPredictor(Predictor):
    """Top-down multi-class predictor.

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
        centroid_backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        centered_instance_backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        videos: List of `sio.Video` objects for creating the `sio.Labels` object from
                        the output predictions.
        skeletons: List of `sio.Skeleton` objects for creating `sio.Labels` object from
                        the output predictions.
        peak_threshold: (float) Minimum confidence threshold. Peaks with values below
                this will be ignored. Default: 0.2. This can also be `List[float]` for topdown
                centroid and centered-instance model, where the first element corresponds
                to centroid model peak finding threshold and the second element is for
                centered-instance model peak finding.
        integral_refinement: If `None`, returns the grid-aligned peaks with no refinement.
            If `"integral"`, peaks will be refined with integral regression.
            Default: "integral".
        integral_patch_size: (int) Size of patches to crop around each rough peak as an
            integer scalar. Default: 5.
        batch_size: (int) Number of samples per batch. Default: 4.
        max_instances: (int) Max number of instances to consider from the predictions.
        return_confmaps: (bool) If `True`, predicted confidence maps will be returned
            along with the predicted peak values and points. Default: False.
        device: (str) Device on which torch.Tensor will be allocated. One of the
            ("cpu", "cuda", "mps").
            Default: "cpu"
        preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
            in the `data_config.preprocessing` section.
        anchor_part: (str) The name of the node to use as the anchor for the centroid. If not
            provided, the anchor part in the `training_config.yaml` is used instead. Default: None.
        max_stride: The maximum stride of the backbone network, as specified in the model's
            `backbone_config`. This determines the downsampling factor applied by the backbone,
            and is used to ensure that input images are padded or resized to be compatible
            with the model's architecture. Default: 16.

    """

    centroid_config: Optional[OmegaConf] = None
    confmap_config: Optional[OmegaConf] = None
    centroid_model: Optional[L.LightningModule] = None
    confmap_model: Optional[L.LightningModule] = None
    centroid_backbone_type: Optional[str] = None
    centered_instance_backbone_type: Optional[str] = None
    videos: Optional[List[sio.Video]] = None
    skeletons: Optional[List[sio.Skeleton]] = None
    peak_threshold: Union[float, List[float]] = 0.2
    integral_refinement: str = "integral"
    integral_patch_size: int = 5
    batch_size: int = 4
    max_instances: Optional[int] = None
    return_confmaps: bool = False
    device: str = "cpu"
    preprocess_config: Optional[OmegaConf] = None
    anchor_part: Optional[str] = None
    max_stride: int = 16

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained models and configuration."""
        # Create an instance of CentroidLayer if centroid_config is not None
        return_crops = False
        # if both centroid and centered-instance model are provided, set return crops to True
        if self.confmap_model:
            return_crops = True
        if isinstance(self.peak_threshold, list):
            centroid_peak_threshold = self.peak_threshold[0]
            centered_instance_peak_threshold = self.peak_threshold[1]
        else:
            centroid_peak_threshold = self.peak_threshold
            centered_instance_peak_threshold = self.peak_threshold

        if self.anchor_part is not None:
            anchor_ind = self.skeletons[0].node_names.index(self.anchor_part)
        else:
            anch_pt = None
            if self.centroid_config is not None:
                anch_pt = (
                    self.centroid_config.model_config.head_configs.centroid.confmaps.anchor_part
                )
            if self.confmap_config is not None:
                anch_pt = (
                    self.confmap_config.model_config.head_configs.multi_class_topdown.confmaps.anchor_part
                )
            anchor_ind = (
                self.skeletons[0].node_names.index(anch_pt)
                if anch_pt is not None
                else None
            )

        if self.centroid_config is None:
            centroid_crop_layer = CentroidCrop(
                use_gt_centroids=True,
                crop_hw=(
                    self.preprocess_config.crop_size,
                    self.preprocess_config.crop_size,
                ),
                anchor_ind=anchor_ind,
                return_crops=return_crops,
            )

        else:
            max_stride = self.centroid_config.model_config.backbone_config[
                f"{self.centroid_backbone_type}"
            ]["max_stride"]
            # initialize centroid crop layer
            centroid_crop_layer = CentroidCrop(
                torch_model=self.centroid_model,
                peak_threshold=centroid_peak_threshold,
                output_stride=self.centroid_config.model_config.head_configs.centroid.confmaps.output_stride,
                refinement=self.integral_refinement,
                integral_patch_size=self.integral_patch_size,
                return_confmaps=self.return_confmaps,
                return_crops=return_crops,
                max_instances=self.max_instances,
                max_stride=max_stride,
                input_scale=self.centroid_config.data_config.preprocessing.scale,
                crop_hw=(
                    self.preprocess_config.crop_size,
                    self.preprocess_config.crop_size,
                ),
                use_gt_centroids=False,
            )

        max_stride = self.confmap_config.model_config.backbone_config[
            f"{self.centered_instance_backbone_type}"
        ]["max_stride"]
        instance_peaks_layer = TopDownMultiClassFindInstancePeaks(
            torch_model=self.confmap_model,
            peak_threshold=centered_instance_peak_threshold,
            output_stride=self.confmap_config.model_config.head_configs.multi_class_topdown.confmaps.output_stride,
            refinement=self.integral_refinement,
            integral_patch_size=self.integral_patch_size,
            return_confmaps=self.return_confmaps,
            max_stride=max_stride,
            input_scale=self.confmap_config.data_config.preprocessing.scale,
        )

        if self.centroid_config is None:
            self.instances_key = (
                True  # we need `instances` to get ground-truth centroids
            )

        # Initialize the inference model with centroid and instance peak layers
        self.inference_model = TopDownInferenceModel(
            centroid_crop=centroid_crop_layer, instance_peaks=instance_peaks_layer
        )

    @classmethod
    def from_trained_models(
        cls,
        centroid_ckpt_path: Optional[Text] = None,
        confmap_ckpt_path: Optional[Text] = None,
        backbone_ckpt_path: Optional[str] = None,
        head_ckpt_path: Optional[str] = None,
        peak_threshold: float = 0.2,
        integral_refinement: str = "integral",
        integral_patch_size: int = 5,
        batch_size: int = 4,
        max_instances: Optional[int] = None,
        return_confmaps: bool = False,
        device: str = "cpu",
        preprocess_config: Optional[OmegaConf] = None,
        anchor_part: Optional[str] = None,
        max_stride: int = 16,
    ) -> "TopDownPredictor":
        """Create predictor from saved models.

        Args:
            centroid_ckpt_path: Path to a centroid ckpt dir with best.ckpt (or from SLEAP <=1.4 best_model.h5 - only UNet backbone is supported) and training_config.yaml (or from SLEAP <=1.4 training_config.json - only UNet backbone is supported).
            confmap_ckpt_path: Path to a centroid ckpt dir with best.ckpt (or from SLEAP <=1.4 best_model.h5 - only UNet backbone is supported) and training_config.yaml (or from SLEAP <=1.4 training_config.json - only UNet backbone is supported).
            backbone_ckpt_path: (str) To run inference on any `.ckpt` other than `best.ckpt`
                from the `model_paths` dir, the path to the `.ckpt` file should be passed here.
            head_ckpt_path: (str) Path to `.ckpt` file if a different set of head layer weights
                are to be used. If `None`, the `best.ckpt` from `model_paths` dir is used (or the ckpt
                from `backbone_ckpt_path` if provided.)
            peak_threshold: (float) Minimum confidence threshold. Peaks with values below
                this will be ignored. Default: 0.2
            integral_refinement: If `None`, returns the grid-aligned peaks with no refinement.
                If `"integral"`, peaks will be refined with integral regression.
                Default: "integral".
            integral_patch_size: (int) Size of patches to crop around each rough peak as an
                integer scalar. Default: 5.
            batch_size: (int) Number of samples per batch. Default: 4.
            max_instances: (int) Max number of instances to consider from the predictions.
            return_confmaps: (bool) If `True`, predicted confidence maps will be returned
                along with the predicted peak values and points. Default: False.
            device: (str) Device on which torch.Tensor will be allocated. One of the
                ("cpu", "cuda", "mps").
                Default: "cpu"
            preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
                in the `data_config.preprocessing` section.
            anchor_part: (str) The name of the node to use as the anchor for the centroid. If not
                provided, the anchor part in the `training_config.yaml` is used instead. Default: None.
            max_stride: The maximum stride of the backbone network, as specified in the model's
                `backbone_config`. This determines the downsampling factor applied by the backbone,
                and is used to ensure that input images are padded or resized to be compatible
                with the model's architecture. Default: 16.

        Returns:
            An instance of `TopDownPredictor` with the loaded models.

            One of the two models can be left as `None` to perform inference with ground
            truth data. This will only work with `LabelsReader` as the provider.

        """
        centered_instance_backbone_type = None
        centroid_backbone_type = None
        if centroid_ckpt_path is not None:
            is_sleap_ckpt = False
            if (
                Path(centroid_ckpt_path) / "training_config.yaml"
                in Path(centroid_ckpt_path).iterdir()
            ):
                centroid_config = OmegaConf.load(
                    (Path(centroid_ckpt_path) / "training_config.yaml").as_posix()
                )
            elif (
                Path(centroid_ckpt_path) / "training_config.json"
                in Path(centroid_ckpt_path).iterdir()
            ):
                is_sleap_ckpt = True
                centroid_config = TrainingJobConfig.load_sleap_config(
                    (Path(centroid_ckpt_path) / "training_config.json").as_posix()
                )

            # Load centroid model.
            skeletons = get_skeleton_from_config(centroid_config.data_config.skeletons)

            # check which backbone architecture
            for k, v in centroid_config.model_config.backbone_config.items():
                if v is not None:
                    centroid_backbone_type = k
                    break

            if not is_sleap_ckpt:
                ckpt_path = (Path(centroid_ckpt_path) / "best.ckpt").as_posix()

                centroid_model = CentroidLightningModule.load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    model_type="centroid",
                    backbone_type=centroid_backbone_type,
                    backbone_config=centroid_config.model_config.backbone_config,
                    head_configs=centroid_config.model_config.head_configs,
                    pretrained_backbone_weights=None,
                    pretrained_head_weights=None,
                    init_weights=centroid_config.model_config.init_weights,
                    lr_scheduler=centroid_config.trainer_config.lr_scheduler,
                    online_mining=centroid_config.trainer_config.online_hard_keypoint_mining.online_mining,
                    hard_to_easy_ratio=centroid_config.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
                    min_hard_keypoints=centroid_config.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
                    max_hard_keypoints=centroid_config.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
                    loss_scale=centroid_config.trainer_config.online_hard_keypoint_mining.loss_scale,
                    optimizer=centroid_config.trainer_config.optimizer_name,
                    learning_rate=centroid_config.trainer_config.optimizer.lr,
                    amsgrad=centroid_config.trainer_config.optimizer.amsgrad,
                    map_location=device,
                    weights_only=False,
                )

            else:
                centroid_converted_model = load_legacy_model(
                    model_dir=f"{centroid_ckpt_path}"
                )
                centroid_model = CentroidLightningModule(
                    model_type="centroid",
                    backbone_type=centroid_backbone_type,
                    backbone_config=centroid_config.model_config.backbone_config,
                    head_configs=centroid_config.model_config.head_configs,
                    pretrained_backbone_weights=None,
                    pretrained_head_weights=None,
                    init_weights=centroid_config.model_config.init_weights,
                    lr_scheduler=centroid_config.trainer_config.lr_scheduler,
                    online_mining=centroid_config.trainer_config.online_hard_keypoint_mining.online_mining,
                    hard_to_easy_ratio=centroid_config.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
                    min_hard_keypoints=centroid_config.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
                    max_hard_keypoints=centroid_config.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
                    loss_scale=centroid_config.trainer_config.online_hard_keypoint_mining.loss_scale,
                    optimizer=centroid_config.trainer_config.optimizer_name,
                    learning_rate=centroid_config.trainer_config.optimizer.lr,
                    amsgrad=centroid_config.trainer_config.optimizer.amsgrad,
                )
                centroid_model.eval()
                centroid_model.model = centroid_converted_model
                centroid_model.to(device)

            centroid_model.eval()

            if backbone_ckpt_path is not None and head_ckpt_path is not None:
                logger.info(f"Loading backbone weights from `{backbone_ckpt_path}` ...")
                ckpt = torch.load(
                    backbone_ckpt_path,
                    map_location=device,
                    weights_only=False,
                )
                ckpt["state_dict"] = {
                    k: ckpt["state_dict"][k]
                    for k in ckpt["state_dict"].keys()
                    if ".backbone" in k
                }
                centroid_model.load_state_dict(ckpt["state_dict"], strict=False)

            elif backbone_ckpt_path is not None:
                logger.info(f"Loading weights from `{backbone_ckpt_path}` ...")
                ckpt = torch.load(
                    backbone_ckpt_path,
                    map_location=device,
                    weights_only=False,
                )
                centroid_model.load_state_dict(
                    ckpt["state_dict"],
                    strict=False,
                    weights_only=False,
                )

            if head_ckpt_path is not None:
                logger.info(f"Loading head weights from `{head_ckpt_path}` ...")
                ckpt = torch.load(
                    head_ckpt_path,
                    map_location=device,
                    weights_only=False,
                )
                ckpt["state_dict"] = {
                    k: ckpt["state_dict"][k]
                    for k in ckpt["state_dict"].keys()
                    if ".head_layers" in k
                }
                centroid_model.load_state_dict(ckpt["state_dict"], strict=False)

            centroid_model.to(device)

        else:
            centroid_config = None
            centroid_model = None

        if confmap_ckpt_path is not None:
            # Load confmap model.
            is_sleap_ckpt = False
            if (
                Path(confmap_ckpt_path) / "training_config.yaml"
                in Path(confmap_ckpt_path).iterdir()
            ):
                confmap_config = OmegaConf.load(
                    (Path(confmap_ckpt_path) / "training_config.yaml").as_posix()
                )
            elif (
                Path(confmap_ckpt_path) / "training_config.json"
                in Path(confmap_ckpt_path).iterdir()
            ):
                is_sleap_ckpt = True
                confmap_config = TrainingJobConfig.load_sleap_config(
                    (Path(confmap_ckpt_path) / "training_config.json").as_posix()
                )

            # check which backbone architecture
            for k, v in confmap_config.model_config.backbone_config.items():
                if v is not None:
                    centered_instance_backbone_type = k
                    break

            if not is_sleap_ckpt:
                ckpt_path = (Path(confmap_ckpt_path) / "best.ckpt").as_posix()

                confmap_model = TopDownCenteredInstanceMultiClassLightningModule.load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    model_type="multi_class_topdown",
                    backbone_type=centered_instance_backbone_type,
                    backbone_config=confmap_config.model_config.backbone_config,
                    head_configs=confmap_config.model_config.head_configs,
                    pretrained_backbone_weights=None,
                    pretrained_head_weights=None,
                    init_weights=confmap_config.model_config.init_weights,
                    lr_scheduler=confmap_config.trainer_config.lr_scheduler,
                    online_mining=confmap_config.trainer_config.online_hard_keypoint_mining.online_mining,
                    hard_to_easy_ratio=confmap_config.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
                    min_hard_keypoints=confmap_config.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
                    max_hard_keypoints=confmap_config.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
                    loss_scale=confmap_config.trainer_config.online_hard_keypoint_mining.loss_scale,
                    optimizer=confmap_config.trainer_config.optimizer_name,
                    learning_rate=confmap_config.trainer_config.optimizer.lr,
                    amsgrad=confmap_config.trainer_config.optimizer.amsgrad,
                    map_location=device,
                    weights_only=False,
                )
            else:
                confmap_converted_model = load_legacy_model(
                    model_dir=f"{confmap_ckpt_path}"
                )
                confmap_model = TopDownCenteredInstanceMultiClassLightningModule(
                    model_type="multi_class_topdown",
                    backbone_type=centered_instance_backbone_type,
                    backbone_config=confmap_config.model_config.backbone_config,
                    head_configs=confmap_config.model_config.head_configs,
                    pretrained_backbone_weights=None,
                    pretrained_head_weights=None,
                    init_weights=confmap_config.model_config.init_weights,
                    lr_scheduler=confmap_config.trainer_config.lr_scheduler,
                    online_mining=confmap_config.trainer_config.online_hard_keypoint_mining.online_mining,
                    hard_to_easy_ratio=confmap_config.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
                    min_hard_keypoints=confmap_config.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
                    max_hard_keypoints=confmap_config.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
                    loss_scale=confmap_config.trainer_config.online_hard_keypoint_mining.loss_scale,
                    optimizer=confmap_config.trainer_config.optimizer_name,
                    learning_rate=confmap_config.trainer_config.optimizer.lr,
                    amsgrad=confmap_config.trainer_config.optimizer.amsgrad,
                )
                confmap_model.eval()
                confmap_model.model = confmap_converted_model
                confmap_model.to(device)

            confmap_model.eval()
            skeletons = get_skeleton_from_config(confmap_config.data_config.skeletons)

            if backbone_ckpt_path is not None and head_ckpt_path is not None:
                logger.info(f"Loading backbone weights from `{backbone_ckpt_path}` ...")
                ckpt = torch.load(
                    backbone_ckpt_path,
                    map_location=device,
                    weights_only=False,
                )
                ckpt["state_dict"] = {
                    k: ckpt["state_dict"][k]
                    for k in ckpt["state_dict"].keys()
                    if ".backbone" in k
                }
                confmap_model.load_state_dict(ckpt["state_dict"], strict=False)

            elif backbone_ckpt_path is not None:
                logger.info(f"Loading weights from `{backbone_ckpt_path}` ...")
                ckpt = torch.load(
                    backbone_ckpt_path,
                    map_location=device,
                    weights_only=False,
                )
                confmap_model.load_state_dict(ckpt["state_dict"], strict=False)

            if head_ckpt_path is not None:
                logger.info(f"Loading head weights from `{head_ckpt_path}` ...")
                ckpt = torch.load(
                    head_ckpt_path,
                    map_location=device,
                    weights_only=False,
                )
                ckpt["state_dict"] = {
                    k: ckpt["state_dict"][k]
                    for k in ckpt["state_dict"].keys()
                    if ".head_layers" in k
                }
                confmap_model.load_state_dict(ckpt["state_dict"], strict=False)

            confmap_model.to(device)

        else:
            confmap_config = None
            confmap_model = None

        if centroid_config is None and confmap_config is None:
            message = (
                "Both a centroid and a confidence map model must be provided to "
                "initialize a TopDownMultiClassPredictor."
            )
            logger.error(message)
            raise ValueError(message)

        if centroid_config is not None:
            preprocess_config["scale"] = (
                centroid_config.data_config.preprocessing.scale
                if preprocess_config["scale"] is None
                else preprocess_config["scale"]
            )
            preprocess_config["ensure_rgb"] = (
                centroid_config.data_config.preprocessing.ensure_rgb
                if preprocess_config["ensure_rgb"] is None
                else preprocess_config["ensure_rgb"]
            )
            preprocess_config["ensure_grayscale"] = (
                centroid_config.data_config.preprocessing.ensure_grayscale
                if preprocess_config["ensure_grayscale"] is None
                else preprocess_config["ensure_grayscale"]
            )
            preprocess_config["max_height"] = (
                centroid_config.data_config.preprocessing.max_height
                if preprocess_config["max_height"] is None
                else preprocess_config["max_height"]
            )
            preprocess_config["max_width"] = (
                centroid_config.data_config.preprocessing.max_width
                if preprocess_config["max_width"] is None
                else preprocess_config["max_width"]
            )

        else:
            preprocess_config["scale"] = (
                confmap_config.data_config.preprocessing.scale
                if preprocess_config["scale"] is None
                else preprocess_config["scale"]
            )
            preprocess_config["ensure_rgb"] = (
                confmap_config.data_config.preprocessing.ensure_rgb
                if preprocess_config["ensure_rgb"] is None
                else preprocess_config["ensure_rgb"]
            )
            preprocess_config["ensure_grayscale"] = (
                confmap_config.data_config.preprocessing.ensure_grayscale
                if preprocess_config["ensure_grayscale"] is None
                else preprocess_config["ensure_grayscale"]
            )
            preprocess_config["max_height"] = (
                confmap_config.data_config.preprocessing.max_height
                if preprocess_config["max_height"] is None
                else preprocess_config["max_height"]
            )
            preprocess_config["max_width"] = (
                confmap_config.data_config.preprocessing.max_width
                if preprocess_config["max_width"] is None
                else preprocess_config["max_width"]
            )

        preprocess_config["crop_size"] = (
            confmap_config.data_config.preprocessing.crop_size
            if preprocess_config["crop_size"] is None and confmap_config is not None
            else preprocess_config["crop_size"]
        )

        # create an instance of TopDownPredictor class
        obj = cls(
            centroid_config=centroid_config,
            centroid_model=centroid_model,
            confmap_config=confmap_config,
            confmap_model=confmap_model,
            centroid_backbone_type=centroid_backbone_type,
            centered_instance_backbone_type=centered_instance_backbone_type,
            skeletons=skeletons,
            peak_threshold=peak_threshold,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            batch_size=batch_size,
            max_instances=max_instances,
            return_confmaps=return_confmaps,
            device=device,
            preprocess_config=preprocess_config,
            anchor_part=anchor_part,
            max_stride=(
                centroid_config.model_config.backbone_config[
                    f"{centroid_backbone_type}"
                ]["max_stride"]
                if centroid_config is not None
                else confmap_config.model_config.backbone_config[
                    f"{centered_instance_backbone_type}"
                ]["max_stride"]
            ),
        )

        obj._initialize_inference_model()
        return obj

    def make_pipeline(
        self,
        inference_object: Union[str, Path, sio.Labels, sio.Video],
        queue_maxsize: int = 8,
        frames: Optional[list] = None,
        only_labeled_frames: bool = False,
        only_suggested_frames: bool = False,
        exclude_user_labeled: bool = False,
        only_predicted_frames: bool = False,
        video_index: Optional[int] = None,
        video_dataset: Optional[str] = None,
        video_input_format: str = "channels_last",
    ):
        """Make a data loading pipeline.

        Args:
            inference_object: (str) Path to `.slp` file or `.mp4` or sio.Labels or sio.Video to run inference on.
            queue_maxsize: (int) Maximum size of the frame buffer queue. Default: 8.
            frames: (list) List of frames indices. If `None`, all frames in the video are used. Default: None.
            only_labeled_frames: (bool) `True` if inference should be run only on user-labeled frames. Default: `False`.
            only_suggested_frames: (bool) `True` if inference should be run only on unlabeled suggested frames. Default: `False`.
            exclude_user_labeled: (bool) `True` to skip frames that have user-labeled instances. Default: `False`.
            only_predicted_frames: (bool) `True` to run inference only on frames that already have predictions. Default: `False`.
            video_index: (int) Integer index of video in .slp file to predict on. To be used
                with an .slp path as an alternative to specifying the video path.
            video_dataset: (str) The dataset for HDF5 videos.
            video_input_format: (str) The input_format for HDF5 videos.

        Returns:
            This method initiates the reader class (doesn't return a pipeline) and the
            Thread is started in Predictor._predict_generator() method.
        """
        if isinstance(inference_object, str) or isinstance(inference_object, Path):
            inference_object = (
                sio.load_slp(inference_object)
                if inference_object.endswith(".slp")
                else sio.load_video(
                    inference_object,
                    dataset=video_dataset,
                    input_format=video_input_format,
                )
            )

        # LabelsReader provider
        if isinstance(inference_object, sio.Labels) and video_index is None:
            provider = LabelsReader

            self.preprocess = False

            frame_buffer = Queue(maxsize=queue_maxsize)

            self.pipeline = provider(
                labels=inference_object,
                frame_buffer=frame_buffer,
                instances_key=self.instances_key,
                only_labeled_frames=only_labeled_frames,
                only_suggested_frames=only_suggested_frames,
                exclude_user_labeled=exclude_user_labeled,
                only_predicted_frames=only_predicted_frames,
            )
            self.videos = self.pipeline.labels.videos

        else:
            provider = VideoReader
            if self.centroid_config is None:
                message = (
                    "Ground truth data was not detected... "
                    "Please load both models when predicting on non-ground-truth data."
                )
                logger.error(message)
                raise ValueError(message)

            self.preprocess = False

            if isinstance(inference_object, sio.Labels) and video_index is not None:
                labels = inference_object
                video = labels.videos[video_index]
                # Filter out user-labeled frames if requested
                filtered_frames = _filter_user_labeled_frames(
                    labels, video, frames, exclude_user_labeled
                )
                self.pipeline = provider.from_video(
                    video=video,
                    queue_maxsize=queue_maxsize,
                    frames=filtered_frames,
                )

            else:  # for mp4 or hdf5 videos
                frame_buffer = Queue(maxsize=queue_maxsize)
                self.pipeline = provider(
                    video=inference_object,
                    frame_buffer=frame_buffer,
                    frames=frames,
                )

            self.videos = [self.pipeline.video]

    def _make_labeled_frames_from_generator(
        self,
        generator: Iterator[Dict[str, np.ndarray]],
    ) -> sio.Labels:
        """Create labeled frames from a generator that yields inference results.

        This method converts pure arrays into SLEAP-specific data structures and assigns
        tracks to the predicted instances if tracker is specified.

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
        # open video backend for tracking
        for video in self.videos:
            if not video.open_backend:
                video.open()

        preds = defaultdict(list)
        predicted_frames = []
        skeleton_idx = 0

        tracks = [
            sio.Track(name=x)
            for x in self.confmap_config.model_config.head_configs.multi_class_topdown.class_vectors.classes
        ]

        # Loop through each predicted instance.
        for ex in generator:
            # loop through each sample in a batch
            for (
                video_idx,
                frame_idx,
                bbox,
                pred_instances,
                pred_values,
                centroid_val,
                org_size,
                class_ind,
                instance_score,
            ) in zip(
                ex["video_idx"],
                ex["frame_idx"],
                ex["instance_bbox"],
                ex["pred_instance_peaks"],
                ex["pred_peak_values"],
                ex["centroid_val"],
                ex["orig_size"],
                ex["pred_class_inds"],
                ex["instance_scores"],
            ):
                if np.isnan(pred_instances).all():
                    continue
                pred_instances = pred_instances + bbox.squeeze(axis=0)[0, :]

                track = None
                if tracks is not None:
                    track = tracks[class_ind]

                preds[(int(video_idx), int(frame_idx))].append(
                    sio.PredictedInstance.from_numpy(
                        points_data=pred_instances,
                        skeleton=self.skeletons[skeleton_idx],
                        point_scores=pred_values,
                        score=centroid_val,
                        track=track,
                        tracking_score=instance_score,
                    )
                )
        for key, inst in preds.items():
            # Create list of LabeledFrames.
            video_idx, frame_idx = key
            lf = sio.LabeledFrame(
                video=self.videos[video_idx],
                frame_idx=frame_idx,
                instances=inst,
            )

            predicted_frames.append(lf)

        pred_labels = sio.Labels(
            videos=self.videos,
            skeletons=self.skeletons,
            labeled_frames=predicted_frames,
        )
        return pred_labels
