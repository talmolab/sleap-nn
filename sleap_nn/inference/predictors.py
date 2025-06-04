"""Predictors for running inference."""

from collections import defaultdict
from typing import Dict, List, Optional, Union, Iterator, Text
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import sleap_io as sio
import torchvision.transforms.v2.functional as F
import torch
import attrs
import lightning as L
from omegaconf import OmegaConf
from loguru import logger
from sleap_nn.data.providers import LabelsReader, VideoReader
from sleap_nn.data.resizing import (
    resize_image,
    apply_pad_to_stride,
    apply_sizematcher,
    apply_resizer,
)
from sleap_nn.data.normalization import (
    apply_normalization,
)
from sleap_nn.inference.paf_grouping import PAFScorer
from sleap_nn.training.lightning_modules import (
    TopDownCenteredInstanceLightningModule,
    SingleInstanceLightningModule,
    CentroidLightningModule,
    BottomUpLightningModule,
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
from sleap_nn.tracking.tracker import Tracker


@attrs.define
class Predictor(ABC):
    """Base interface class for predictors.

    This is the base predictor class for different types of models.

    Attributes:
        preprocess: Only for VideoReader provider. True if preprocessing (reszizing and
            apply_pad_to_stride) should be applied on the frames read in the video reader.
            Default: True.
        preprocess_config: Preprocessing config with keys: [`batch_size`,
            `scale`, `is_rgb`, `max_stride`]. Default: {"batch_size": 4, "scale": 1.0,
            "is_rgb": False, "max_stride": 1}
        provider: Provider for inference pipeline. One of ["LabelsReader", "VideoReader"].
            Default: LabelsReader.
        pipeline: If provider is LabelsReader, pipeline is a `DataLoader` object. If provider
            is VideoReader, pipeline is an instance of `sleap_nn.data.providers.VideoReader`
            class. Default: None.
        inference_model: Instance of one of the inference models ["TopDownInferenceModel",
            "SingleInstanceInferenceModel", "BottomUpInferenceModel"]. Default: None.
        instances_key: If `True`, then instances are appended to the data samples.
    """

    preprocess: bool = True
    preprocess_config: dict = {
        "batch_size": 4,
        "scale": 1.0,
        "is_rgb": False,
        "max_stride": 1,
        "max_height": None,
        "max_width": None,
    }
    provider: Union[LabelsReader, VideoReader] = LabelsReader
    pipeline: Optional[Union[LabelsReader, VideoReader]] = None
    inference_model: Optional[
        Union[
            TopDownInferenceModel, SingleInstanceInferenceModel, BottomUpInferenceModel
        ]
    ] = None
    instances_key: bool = False

    @classmethod
    def from_model_paths(
        cls,
        model_paths: List[Text],
        backbone_ckpt_path: Optional[str] = None,
        head_ckpt_path: Optional[str] = None,
        peak_threshold: Union[float, List[float]] = 0.2,
        integral_refinement: str = None,
        integral_patch_size: int = 5,
        batch_size: int = 4,
        max_instances: Optional[int] = None,
        return_confmaps: bool = False,
        device: str = "cpu",
        preprocess_config: Optional[OmegaConf] = None,
    ) -> "Predictor":
        """Create the appropriate `Predictor` subclass from from the ckpt path.

        Args:
            model_paths: (List[str]) List of paths to the directory where the best.ckpt
                and training_config.yaml are saved.
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
                Default: None.
            integral_patch_size: (int) Size of patches to crop around each rough peak as an
                integer scalar. Default: 5.
            batch_size: (int) Number of samples per batch. Default: 4.
            max_instances: (int) Max number of instances to consider from the predictions.
            return_confmaps: (bool) If `True`, predicted confidence maps will be returned
                along with the predicted peak values and points. Default: False.
            device: (str) Device on which torch.Tensor will be allocated. One of the
                ("cpu", "cuda", "mkldnn", "opengl", "opencl", "ideep", "hip", "msnpu").
                Default: "cpu"
            preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
                in the `data_config.preprocessing` section.

        Returns:
            A subclass of `Predictor`.

        See also: `SingleInstancePredictor`, `TopDownPredictor`, `BottomUpPredictor`,
            `MoveNetPredictor`, `TopDownMultiClassPredictor`,
            `BottomUpMultiClassPredictor`.
        """
        model_configs = [
            OmegaConf.load(f"{Path(c)}/training_config.yaml") for c in model_paths
        ]
        model_names = []
        for config in model_configs:
            for k, v in config.model_config.head_configs.items():
                if v is not None:
                    model_names.append(k)

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

        elif "centroid" in model_names or "centered_instance" in model_names:
            centroid_ckpt_path = None
            confmap_ckpt_path = None
            if "centroid" in model_names:
                centroid_ckpt_path = model_paths[model_names.index("centroid")]
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
        else:
            message = f"Could not create predictor from model paths:\n{model_paths}"
            logger.error(message)
            raise ValueError(message)
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
    def make_pipeline(
        self,
        provider: str,
        data_path: str,
        queue_maxsize: int = 8,
        frames: Optional[list] = None,
        only_labeled_frames: bool = False,
        only_suggested_frames: bool = False,
    ):
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
        # Initialize inference model if needed.

        if self.inference_model is None:
            self._initialize_inference_model()

        # Loop over data batches.
        self.pipeline.start()
        batch_size = self.preprocess_config["batch_size"]
        done = False
        while not done:
            imgs = []
            fidxs = []
            vidxs = []
            org_szs = []
            instances = []
            eff_scales = []
            for _ in range(batch_size):
                frame = self.pipeline.frame_buffer.get()
                if frame["image"] is None:
                    done = True
                    break
                frame["image"] = apply_normalization(frame["image"])
                frame["image"], eff_scale = apply_sizematcher(
                    frame["image"],
                    self.preprocess_config["max_height"],
                    self.preprocess_config["max_width"],
                )
                if self.instances_key:
                    frame["instances"] = frame["instances"] * eff_scale
                if self.preprocess_config["is_rgb"] and frame["image"].shape[-3] != 3:
                    frame["image"] = frame["image"].repeat(1, 3, 1, 1)
                elif not self.preprocess_config["is_rgb"]:
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
            if imgs:
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
                if self.preprocess_config["is_rgb"] and ex["image"].shape[-3] != 3:
                    ex["image"] = ex["image"].repeat(1, 1, 3, 1, 1)
                elif not self.preprocess_config["is_rgb"]:
                    ex["image"] = F.rgb_to_grayscale(ex["image"], num_output_channels=1)
                if self.preprocess:
                    scale = self.preprocess_config["scale"]
                    if scale != 1.0:
                        if self.instances_key:
                            ex["image"], ex["instances"] = apply_resizer(
                                ex["image"], ex["instances"]
                            )
                        else:
                            ex["image"] = resize_image(ex["image"], scale)
                    ex["image"] = apply_pad_to_stride(
                        ex["image"], self.preprocess_config["max_stride"]
                    )
                outputs_list = self.inference_model(ex)
                if outputs_list is not None:
                    for output in outputs_list:
                        output = self._convert_tensors_to_numpy(output)
                        yield output

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
                pred_labels.save(Path(save_path).as_posix())
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
            Default: None.
        integral_patch_size: (int) Size of patches to crop around each rough peak as an
            integer scalar. Default: 5.
        batch_size: (int) Number of samples per batch. Default: 4.
        max_instances: (int) Max number of instances to consider from the predictions.
        return_confmaps: (bool) If `True`, predicted confidence maps will be returned
            along with the predicted peak values and points. Default: False.
        device: (str) Device on which torch.Tensor will be allocated. One of the
            ("cpu", "cuda", "mkldnn", "opengl", "opencl", "ideep", "hip", "msnpu").
            Default: "cpu"
        preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
            in the `data_config.preprocessing` section.
        tracker: A `sleap_nn.tracking.Tracker` that will be called to associate
            detections over time. Predicted instances will not be assigned to tracks if
            if this is `None`.
        anchor_part: (str) The name of the node to use as the anchor for the centroid. If not
            provided, the anchor part in the `training_config.yaml` is used instead.

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
    integral_refinement: str = None
    integral_patch_size: int = 5
    batch_size: int = 4
    max_instances: Optional[int] = None
    return_confmaps: bool = False
    device: str = "cpu"
    preprocess_config: Optional[OmegaConf] = None
    tracker: Optional[Tracker] = None
    anchor_part: Optional[str] = None

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

        if self.data_config.crop_hw is None and self.confmap_config is not None:
            self.data_config.crop_hw = (
                self.confmap_config.data_config.preprocessing.crop_hw
            )

        if self.anchor_part is not None:
            anchor_ind = self.skeletons[0].node_names.index(self.anchor_part)
        else:
            anch_pt = None
            if self.centroid_config is not None:
                anch_pt = (
                    self.centroid_config.model_config.head_configs.centroid.confmaps.anchor_part
                )
            elif self.confmap_config is not None:
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
                crop_hw=self.data_config.crop_hw,
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
                crop_hw=self.data_config.crop_hw,
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
            centroid_crop_layer.precrop_resize = (
                self.confmap_config.data_config.preprocessing.scale
            )

        if self.centroid_config is None and self.confmap_config is not None:
            self.instances_key = (
                True  # we need `instances` to get ground-truth centroids
            )

        # Initialize the inference model with centroid and instance peak layers
        self.inference_model = TopDownInferenceModel(
            centroid_crop=centroid_crop_layer, instance_peaks=instance_peaks_layer
        )

    @property
    def data_config(self) -> OmegaConf:
        """Returns data config section from the overall config."""
        if self.centroid_config:
            data_config = self.centroid_config.data_config.preprocessing
        else:
            data_config = self.confmap_config.data_config.preprocessing
        if self.preprocess_config is None:
            return data_config
        return self.preprocess_config

    @classmethod
    def from_trained_models(
        cls,
        centroid_ckpt_path: Optional[Text] = None,
        confmap_ckpt_path: Optional[Text] = None,
        backbone_ckpt_path: Optional[str] = None,
        head_ckpt_path: Optional[str] = None,
        peak_threshold: float = 0.2,
        integral_refinement: str = None,
        integral_patch_size: int = 5,
        batch_size: int = 4,
        max_instances: Optional[int] = None,
        return_confmaps: bool = False,
        device: str = "cpu",
        preprocess_config: Optional[OmegaConf] = None,
    ) -> "TopDownPredictor":
        """Create predictor from saved models.

        Args:
            centroid_ckpt_path: Path to a centroid ckpt dir with model.ckpt and config.yaml.
            confmap_ckpt_path: Path to a centroid ckpt dir with model.ckpt and config.yaml.
            backbone_ckpt_path: (str) To run inference on any `.ckpt` other than `best.ckpt`
                from the `model_paths` dir, the path to the `.ckpt` file should be passed here.
            head_ckpt_path: (str) Path to `.ckpt` file if a different set of head layer weights
                are to be used. If `None`, the `best.ckpt` from `model_paths` dir is used (or the ckpt
                from `backbone_ckpt_path` if provided.)
            peak_threshold: (float) Minimum confidence threshold. Peaks with values below
                this will be ignored. Default: 0.2
            integral_refinement: If `None`, returns the grid-aligned peaks with no refinement.
                If `"integral"`, peaks will be refined with integral regression.
                Default: None.
            integral_patch_size: (int) Size of patches to crop around each rough peak as an
                integer scalar. Default: 5.
            batch_size: (int) Number of samples per batch. Default: 4.
            max_instances: (int) Max number of instances to consider from the predictions.
            return_confmaps: (bool) If `True`, predicted confidence maps will be returned
                along with the predicted peak values and points. Default: False.
            device: (str) Device on which torch.Tensor will be allocated. One of the
                ("cpu", "cuda", "mkldnn", "opengl", "opencl", "ideep", "hip", "msnpu").
                Default: "cpu"
            preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
                in the `data_config.preprocessing` section and the `anchor_part`.

        Returns:
            An instance of `TopDownPredictor` with the loaded models.

            One of the two models can be left as `None` to perform inference with ground
            truth data. This will only work with `LabelsReader` as the provider.

        """
        centered_instance_backbone_type = None
        centroid_backbone_type = None
        if centroid_ckpt_path is not None:
            # Load centroid model.
            centroid_config = OmegaConf.load(
                f"{centroid_ckpt_path}/training_config.yaml"
            )
            skeletons = get_skeleton_from_config(centroid_config.data_config.skeletons)
            ckpt_path = f"{centroid_ckpt_path}/best.ckpt"

            # check which backbone architecture
            for k, v in centroid_config.model_config.backbone_config.items():
                if v is not None:
                    centroid_backbone_type = k
                    break

            centroid_model = CentroidLightningModule.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                config=centroid_config,
                skeletons=skeletons,
                model_type="centroid",
                backbone_type=centroid_backbone_type,
            )

            if backbone_ckpt_path is not None and head_ckpt_path is not None:
                logger.info(f"Loading backbone weights from `{backbone_ckpt_path}` ...")
                ckpt = torch.load(backbone_ckpt_path)
                ckpt["state_dict"] = {
                    k: ckpt["state_dict"][k]
                    for k in ckpt["state_dict"].keys()
                    if ".backbone" in k
                }
                centroid_model.load_state_dict(ckpt["state_dict"], strict=False)

            elif backbone_ckpt_path is not None:
                logger.info(f"Loading weights from `{backbone_ckpt_path}` ...")
                ckpt = torch.load(backbone_ckpt_path)
                centroid_model.load_state_dict(ckpt["state_dict"], strict=False)

            if head_ckpt_path is not None:
                logger.info(f"Loading head weights from `{head_ckpt_path}` ...")
                ckpt = torch.load(head_ckpt_path)
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
            confmap_config = OmegaConf.load(f"{confmap_ckpt_path}/training_config.yaml")
            skeletons = get_skeleton_from_config(confmap_config.data_config.skeletons)
            ckpt_path = f"{confmap_ckpt_path}/best.ckpt"

            # check which backbone architecture
            for k, v in confmap_config.model_config.backbone_config.items():
                if v is not None:
                    centered_instance_backbone_type = k
                    break
            confmap_model = TopDownCenteredInstanceLightningModule.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                config=confmap_config,
                model_type="centered_instance",
                backbone_type=centered_instance_backbone_type,
            )
            if backbone_ckpt_path is not None and head_ckpt_path is not None:
                logger.info(f"Loading backbone weights from `{backbone_ckpt_path}` ...")
                ckpt = torch.load(backbone_ckpt_path)
                ckpt["state_dict"] = {
                    k: ckpt["state_dict"][k]
                    for k in ckpt["state_dict"].keys()
                    if ".backbone" in k
                }
                confmap_model.load_state_dict(ckpt["state_dict"], strict=False)

            elif backbone_ckpt_path is not None:
                logger.info(f"Loading weights from `{backbone_ckpt_path}` ...")
                ckpt = torch.load(backbone_ckpt_path)
                confmap_model.load_state_dict(ckpt["state_dict"], strict=False)

            if head_ckpt_path is not None:
                logger.info(f"Loading head weights from `{head_ckpt_path}` ...")
                ckpt = torch.load(head_ckpt_path)
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
            anchor_part=preprocess_config["anchor_part"],
        )

        obj._initialize_inference_model()
        return obj

    def make_pipeline(
        self,
        provider: str,
        data_path: str,
        queue_maxsize: int = 8,
        frames: Optional[list] = None,
        only_labeled_frames: bool = False,
        only_suggested_frames: bool = False,
    ):
        """Make a data loading pipeline.

        Args:
            provider: (str) Provider class to read the input sleap files.
                Either "LabelsReader" or "VideoReader".
            data_path: (str) Path to `.slp` file or `.mp4` to run inference on.
            queue_maxsize: (int) Maximum size of the frame buffer queue. Default: 8.
            frames: (list) List of frames indices. If `None`, all frames in the video are used. Default: None.
            only_labeled_frames: (bool) `True` if inference should be run only on user-labeled frames. Default: `False`.
            only_suggested_frames: (bool) `True` if inference should be run only on unlabeled suggested frames. Default: `False`.

        Returns:
            This method initiates the reader class (doesn't return a pipeline) and the
            Thread is started in Predictor._predict_generator() method.
        """
        self.provider = provider
        if self.centroid_config is not None:
            max_stride = self.centroid_config.model_config.backbone_config[
                f"{self.centroid_backbone_type}"
            ]["max_stride"]
            scale = self.centroid_config.data_config.preprocessing.scale
            max_height = self.centroid_config.data_config.preprocessing.max_height
            max_width = self.centroid_config.data_config.preprocessing.max_width
        else:
            max_stride = self.confmap_config.model_config.backbone_config[
                f"{self.centered_instance_backbone_type}"
            ]["max_stride"]
            scale = self.confmap_config.data_config.preprocessing.scale
            max_height = self.confmap_config.data_config.preprocessing.max_height
            max_width = self.confmap_config.data_config.preprocessing.max_width

        # LabelsReader provider
        if self.provider == "LabelsReader":
            provider = LabelsReader

            self.preprocess = False
            self.preprocess_config = {
                "batch_size": self.batch_size,
                "scale": scale,
                "is_rgb": self.data_config.is_rgb,
                "max_stride": max_stride,
                "max_height": (
                    self.data_config.max_height
                    if self.data_config.max_height is not None
                    else max_height
                ),
                "max_width": (
                    self.data_config.max_width
                    if self.data_config.max_width is not None
                    else max_width
                ),
            }

            self.pipeline = provider.from_filename(
                filename=data_path,
                queue_maxsize=queue_maxsize,
                instances_key=self.instances_key,
                only_labeled_frames=only_labeled_frames,
                only_suggested_frames=only_suggested_frames,
            )
            self.videos = self.pipeline.labels.videos

        # VideoReader provider
        elif self.provider == "VideoReader":
            if self.centroid_config is None:
                message = (
                    "Ground truth data was not detected... "
                    "Please load both models when predicting on non-ground-truth data."
                )
                logger.error(message)
                raise ValueError(message)

            provider = VideoReader
            self.preprocess = False
            self.preprocess_config = {
                "batch_size": self.batch_size,
                "scale": self.centroid_config.data_config.preprocessing.scale,
                "is_rgb": self.data_config.is_rgb,
                "max_stride": (
                    self.centroid_config.model_config.backbone_config[
                        f"{self.centroid_backbone_type}"
                    ]["max_stride"]
                ),
                "max_height": (
                    self.data_config.max_height
                    if self.data_config.max_height is not None
                    else max_height
                ),
                "max_width": (
                    self.data_config.max_width
                    if self.data_config.max_width is not None
                    else max_width
                ),
            }

            self.pipeline = provider.from_filename(
                filename=data_path, queue_maxsize=queue_maxsize, frames=frames
            )
            self.videos = [self.pipeline.video]

        else:
            message = "Provider not recognised. Please use either `LabelsReader` or `VideoReader` as provider"
            logger.error(message)
            raise Exception(message)

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
            Default: None.
        integral_patch_size: (int) Size of patches to crop around each rough peak as an
            integer scalar. Default: 5.
        batch_size: (int) Number of samples per batch. Default: 4.
        return_confmaps: (bool) If `True`, predicted confidence maps will be returned
            along with the predicted peak values and points. Default: False.
        device: (str) Device on which torch.Tensor will be allocated. One of the
            ("cpu", "cuda", "mkldnn", "opengl", "opencl", "ideep", "hip", "msnpu").
            Default: "cpu"
        preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
                in the `data_config.preprocessing` section.

    """

    confmap_config: Optional[OmegaConf] = attrs.field(default=None)
    confmap_model: Optional[L.LightningModule] = attrs.field(default=None)
    backbone_type: str = "unet"
    videos: Optional[List[sio.Video]] = attrs.field(default=None)
    skeletons: Optional[List[sio.Skeleton]] = attrs.field(default=None)
    peak_threshold: float = 0.2
    integral_refinement: str = None
    integral_patch_size: int = 5
    batch_size: int = 4
    return_confmaps: bool = False
    device: str = "cpu"
    preprocess_config: Optional[OmegaConf] = None

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

    @property
    def data_config(self) -> OmegaConf:
        """Returns data config section from the overall config."""
        data_config = self.confmap_config.data_config.preprocessing
        if self.preprocess_config is None:
            return data_config
        return self.preprocess_config

    @classmethod
    def from_trained_models(
        cls,
        confmap_ckpt_path: Optional[Text] = None,
        backbone_ckpt_path: Optional[str] = None,
        head_ckpt_path: Optional[str] = None,
        peak_threshold: float = 0.2,
        integral_refinement: str = None,
        integral_patch_size: int = 5,
        batch_size: int = 4,
        return_confmaps: bool = False,
        device: str = "cpu",
        preprocess_config: Optional[OmegaConf] = None,
    ) -> "SingleInstancePredictor":
        """Create predictor from saved models.

        Args:
            confmap_ckpt_path: Path to a centroid ckpt dir with model.ckpt and config.yaml.
            backbone_ckpt_path: (str) To run inference on any `.ckpt` other than `best.ckpt`
                from the `model_paths` dir, the path to the `.ckpt` file should be passed here.
            head_ckpt_path: (str) Path to `.ckpt` file if a different set of head layer weights
                are to be used. If `None`, the `best.ckpt` from `model_paths` dir is used (or the ckpt
                from `backbone_ckpt_path` if provided.)
            peak_threshold: (float) Minimum confidence threshold. Peaks with values below
                this will be ignored. Default: 0.2
            integral_refinement: If `None`, returns the grid-aligned peaks with no refinement.
                If `"integral"`, peaks will be refined with integral regression.
                Default: None.
            integral_patch_size: (int) Size of patches to crop around each rough peak as an
                integer scalar. Default: 5.
            batch_size: (int) Number of samples per batch. Default: 4.
            return_confmaps: (bool) If `True`, predicted confidence maps will be returned
                along with the predicted peak values and points. Default: False.
            device: (str) Device on which torch.Tensor will be allocated. One of the
                ("cpu", "cuda", "mkldnn", "opengl", "opencl", "ideep", "hip", "msnpu").
                Default: "cpu"
            preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
                in the `data_config.preprocessing` section.

        Returns:
            An instance of `SingleInstancePredictor` with the loaded models.

        """
        confmap_config = OmegaConf.load(f"{confmap_ckpt_path}/training_config.yaml")
        skeletons = get_skeleton_from_config(confmap_config.data_config.skeletons)
        ckpt_path = f"{confmap_ckpt_path}/best.ckpt"

        # check which backbone architecture
        for k, v in confmap_config.model_config.backbone_config.items():
            if v is not None:
                backbone_type = k
                break

        confmap_model = SingleInstanceLightningModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            config=confmap_config,
            model_type="single_instance",
            backbone_type=backbone_type,
        )
        if backbone_ckpt_path is not None and head_ckpt_path is not None:
            logger.info(f"Loading backbone weights from `{backbone_ckpt_path}` ...")
            ckpt = torch.load(backbone_ckpt_path)
            ckpt["state_dict"] = {
                k: ckpt["state_dict"][k]
                for k in ckpt["state_dict"].keys()
                if ".backbone" in k
            }
            confmap_model.load_state_dict(ckpt["state_dict"], strict=False)

        elif backbone_ckpt_path is not None:
            logger.info(f"Loading weights from `{backbone_ckpt_path}` ...")
            ckpt = torch.load(backbone_ckpt_path)
            confmap_model.load_state_dict(ckpt["state_dict"], strict=False)

        if head_ckpt_path is not None:
            logger.info(f"Loading head weights from `{head_ckpt_path}` ...")
            ckpt = torch.load(head_ckpt_path)
            ckpt["state_dict"] = {
                k: ckpt["state_dict"][k]
                for k in ckpt["state_dict"].keys()
                if ".head_layers" in k
            }
            confmap_model.load_state_dict(ckpt["state_dict"], strict=False)
        confmap_model.to(device)

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
        )

        obj._initialize_inference_model()
        return obj

    def make_pipeline(
        self,
        provider: str,
        data_path: str,
        queue_maxsize: int = 8,
        frames: Optional[list] = None,
        only_labeled_frames: bool = False,
        only_suggested_frames: bool = False,
    ):
        """Make a data loading pipeline.

        Args:
            provider: (str) Provider class to read the input sleap files.
                Either "LabelsReader" or "VideoReader".
            data_path: (str) Path to `.slp` file or `.mp4` to run inference on.
            queue_maxsize: (int) Maximum size of the frame buffer queue. Default: 8.
            frames: List of frames indices. If `None`, all frames in the video are used. Default: None.
            only_labeled_frames: (bool) `True` if inference should be run only on user-labeled frames. Default: `False`.
            only_suggested_frames: (bool) `True` if inference should be run only on unlabeled suggested frames. Default: `False`.

        Returns:
            This method initiates the reader class (doesn't return a pipeline) and the
            Thread is started in Predictor._predict_generator() method.

        """
        self.provider = provider

        # LabelsReader provider
        if self.provider == "LabelsReader":
            provider = LabelsReader

            max_stride = self.confmap_config.model_config.backbone_config[
                f"{self.backbone_type}"
            ]["max_stride"]

            self.preprocess = False
            self.preprocess_config = {
                "batch_size": self.batch_size,
                "scale": self.confmap_config.data_config.preprocessing.scale,
                "is_rgb": self.data_config.is_rgb,
                "max_stride": max_stride,
                "max_height": (
                    self.data_config.max_height
                    if self.data_config.max_height is not None
                    else self.confmap_config.data_config.preprocessing.max_height
                ),
                "max_width": (
                    self.data_config.max_width
                    if self.data_config.max_width is not None
                    else self.confmap_config.data_config.preprocessing.max_width
                ),
            }

            self.pipeline = provider.from_filename(
                filename=data_path,
                queue_maxsize=queue_maxsize,
                only_labeled_frames=only_labeled_frames,
                only_suggested_frames=only_suggested_frames,
            )
            self.videos = self.pipeline.labels.videos

        elif self.provider == "VideoReader":
            provider = VideoReader
            self.preprocess = True
            self.preprocess_config = {
                "batch_size": self.batch_size,
                "scale": self.confmap_config.data_config.preprocessing.scale,
                "is_rgb": self.data_config.is_rgb,
                "max_stride": (
                    self.confmap_config.model_config.backbone_config[
                        f"{self.backbone_type}"
                    ]["max_stride"]
                ),
                "max_height": (
                    self.data_config.max_height
                    if self.data_config.max_height is not None
                    else self.confmap_config.data_config.preprocessing.max_height
                ),
                "max_width": (
                    self.data_config.max_width
                    if self.data_config.max_width is not None
                    else self.confmap_config.data_config.preprocessing.max_width
                ),
            }

            self.pipeline = provider.from_filename(
                filename=data_path, queue_maxsize=queue_maxsize, frames=frames
            )

            self.videos = [self.pipeline.video]

        else:
            message = "Provider not recognised. Please use either `LabelsReader` or `VideoReader` as provider"
            logger.error(message)
            raise Exception(message)

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
            Default: None.
        integral_patch_size: (int) Size of patches to crop around each rough peak as an
            integer scalar. Default: 5.
        batch_size: (int) Number of samples per batch. Default: 4.
        max_instances: (int) Max number of instances to consider from the predictions.
        return_confmaps: (bool) If `True`, predicted confidence maps will be returned
            along with the predicted peak values and points. Default: False.
        device: (str) Device on which torch.Tensor will be allocated. One of the
            ("cpu", "cuda", "mkldnn", "opengl", "opencl", "ideep", "hip", "msnpu").
            Default: "cpu".
        preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
                in the `data_config.preprocessing` section.
        tracker: A `sleap.nn.tracking.Tracker` that will be called to associate
            detections over time. Predicted instances will not be assigned to tracks if
            if this is `None`.

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
    integral_refinement: str = None
    integral_patch_size: int = 5
    batch_size: int = 4
    max_instances: Optional[int] = None
    return_confmaps: bool = False
    device: str = "cpu"
    preprocess_config: Optional[OmegaConf] = None
    tracker: Optional[Tracker] = None

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

    @property
    def data_config(self) -> OmegaConf:
        """Returns data config section from the overall config."""
        data_config = self.bottomup_config.data_config.preprocessing
        if self.preprocess_config is None:
            return data_config
        return self.preprocess_config

    @classmethod
    def from_trained_models(
        cls,
        bottomup_ckpt_path: Optional[Text] = None,
        backbone_ckpt_path: Optional[str] = None,
        head_ckpt_path: Optional[str] = None,
        peak_threshold: float = 0.2,
        integral_refinement: str = None,
        integral_patch_size: int = 5,
        batch_size: int = 4,
        max_instances: Optional[int] = None,
        return_confmaps: bool = False,
        device: str = "cpu",
        preprocess_config: Optional[OmegaConf] = None,
    ) -> "BottomUpPredictor":
        """Create predictor from saved models.

        Args:
            bottomup_ckpt_path: Path to a bottom-up ckpt dir with model.ckpt and config.yaml.
            backbone_ckpt_path: (str) To run inference on any `.ckpt` other than `best.ckpt`
                from the `model_paths` dir, the path to the `.ckpt` file should be passed here.
            head_ckpt_path: (str) Path to `.ckpt` file if a different set of head layer weights
                    are to be used. If `None`, the `best.ckpt` from `model_paths` dir is used (or the ckpt
                    from `backbone_ckpt_path` if provided.)
            peak_threshold: (float) Minimum confidence threshold. Peaks with values below
                this will be ignored. Default: 0.2
            integral_refinement: If `None`, returns the grid-aligned peaks with no refinement.
                If `"integral"`, peaks will be refined with integral regression.
                Default: None.
            integral_patch_size: (int) Size of patches to crop around each rough peak as an
                integer scalar. Default: 5.
            batch_size: (int) Number of samples per batch. Default: 4.
            max_instances: (int) Max number of instances to consider from the predictions.
            return_confmaps: (bool) If `True`, predicted confidence maps will be returned
                along with the predicted peak values and points. Default: False.
            device: (str) Device on which torch.Tensor will be allocated. One of the
                ("cpu", "cuda", "mkldnn", "opengl", "opencl", "ideep", "hip", "msnpu").
                Default: "cpu"
            preprocess_config: (OmegaConf) OmegaConf object with keys as the parameters
                in the `data_config.preprocessing` section.

        Returns:
            An instance of `BottomUpPredictor` with the loaded models.

        """
        bottomup_config = OmegaConf.load(f"{bottomup_ckpt_path}/training_config.yaml")
        skeletons = get_skeleton_from_config(bottomup_config.data_config.skeletons)
        ckpt_path = f"{bottomup_ckpt_path}/best.ckpt"

        # check which backbone architecture
        for k, v in bottomup_config.model_config.backbone_config.items():
            if v is not None:
                backbone_type = k
                break

        bottomup_model = BottomUpLightningModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            config=bottomup_config,
            backbone_type=backbone_type,
            model_type="bottomup",
        )
        if backbone_ckpt_path is not None and head_ckpt_path is not None:
            logger.info(f"Loading backbone weights from `{backbone_ckpt_path}` ...")
            ckpt = torch.load(backbone_ckpt_path)
            ckpt["state_dict"] = {
                k: ckpt["state_dict"][k]
                for k in ckpt["state_dict"].keys()
                if ".backbone" in k
            }
            bottomup_model.load_state_dict(ckpt["state_dict"], strict=False)

        elif backbone_ckpt_path is not None:
            logger.info(f"Loading weights from `{backbone_ckpt_path}` ...")
            ckpt = torch.load(backbone_ckpt_path)
            bottomup_model.load_state_dict(ckpt["state_dict"], strict=False)

        if head_ckpt_path is not None:
            logger.info(f"Loading head weights from `{head_ckpt_path}` ...")
            ckpt = torch.load(head_ckpt_path)
            ckpt["state_dict"] = {
                k: ckpt["state_dict"][k]
                for k in ckpt["state_dict"].keys()
                if ".head_layers" in k
            }
            bottomup_model.load_state_dict(ckpt["state_dict"], strict=False)
        bottomup_model.to(device)

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
        )

        obj._initialize_inference_model()
        return obj

    def make_pipeline(
        self,
        provider: str,
        data_path: str,
        queue_maxsize: int = 8,
        frames: Optional[list] = None,
        only_labeled_frames: bool = False,
        only_suggested_frames: bool = False,
    ):
        """Make a data loading pipeline.

        Args:
            provider: (str) Provider class to read the input sleap files.
                Either "LabelsReader" or "VideoReader".
            data_path: (str) Path to `.slp` file or `.mp4` to run inference on.
            queue_maxsize: (int) Maximum size of the frame buffer queue. Default: 8.
            frames: List of frames indices. If `None`, all frames in the video are used. Default: None.
            only_labeled_frames: (bool) `True` if inference should be run only on user-labeled frames. Default: `False`.
            only_suggested_frames: (bool) `True` if inference should be run only on unlabeled suggested frames. Default: `False`.

        Returns:
            This method initiates the reader class (doesn't return a pipeline) and the
            Thread is started in Predictor._predict_generator() method.
        """
        self.provider = provider
        # LabelsReader provider
        if self.provider == "LabelsReader":
            provider = LabelsReader

            max_stride = self.bottomup_config.model_config.backbone_config[
                f"{self.backbone_type}"
            ]["max_stride"]

            self.preprocess = False
            self.preprocess_config = {
                "batch_size": self.batch_size,
                "scale": self.bottomup_config.data_config.preprocessing.scale,
                "is_rgb": self.data_config.is_rgb,
                "max_stride": max_stride,
                "max_height": (
                    self.data_config.max_height
                    if self.data_config.max_height is not None
                    else self.bottomup_config.data_config.preprocessing.max_height
                ),
                "max_width": (
                    self.data_config.max_width
                    if self.data_config.max_width is not None
                    else self.bottomup_config.data_config.preprocessing.max_width
                ),
            }

            self.pipeline = provider.from_filename(
                filename=data_path,
                queue_maxsize=queue_maxsize,
                only_labeled_frames=only_labeled_frames,
                only_suggested_frames=only_suggested_frames,
            )
            self.videos = self.pipeline.labels.videos

        elif self.provider == "VideoReader":
            provider = VideoReader
            self.preprocess = True
            self.preprocess_config = {
                "batch_size": self.batch_size,
                "scale": self.bottomup_config.data_config.preprocessing.scale,
                "is_rgb": self.data_config.is_rgb,
                "max_stride": (
                    self.bottomup_config.model_config.backbone_config[
                        f"{self.backbone_type}"
                    ]["max_stride"]
                ),
                "max_height": (
                    self.data_config.max_height
                    if self.data_config.max_height is not None
                    else self.bottomup_config.data_config.preprocessing.max_height
                ),
                "max_width": (
                    self.data_config.max_width
                    if self.data_config.max_width is not None
                    else self.bottomup_config.data_config.preprocessing.max_width
                ),
            }

            self.pipeline = provider.from_filename(
                filename=data_path, queue_maxsize=queue_maxsize, frames=frames
            )

            self.videos = [self.pipeline.video]

        else:
            message = "Provider not recognised. Please use either `LabelsReader` or `VideoReader` as provider"
            logger.error(message)
            raise Exception(message)

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


def run_inference(
    data_path: str,
    model_paths: List[str],
    backbone_ckpt_path: Optional[str] = None,
    head_ckpt_path: Optional[str] = None,
    max_instances: Optional[int] = None,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    is_rgb: bool = False,
    anchor_part: Optional[str] = None,
    provider: Optional[str] = None,
    only_labeled_frames: bool = False,
    only_suggested_frames: bool = False,
    batch_size: int = 4,
    queue_maxsize: int = 8,
    frames: Optional[list] = None,
    crop_size: Optional[int] = None,
    peak_threshold: Union[float, List[float]] = 0.2,
    ##
    integral_refinement: str = None,
    integral_patch_size: int = 5,
    return_confmaps: bool = False,
    return_pafs: bool = False,
    return_paf_graph: bool = False,
    max_edge_length_ratio: float = 0.25,
    dist_penalty_weight: float = 1.0,
    n_points: int = 10,
    min_instance_peaks: Union[int, float] = 0,
    min_line_scores: float = 0.25,
    make_labels: bool = True,
    ##
    output_path: str = "",
    device: str = "cpu",
    tracking: bool = False,
    tracking_window_size: int = 5,
    tracking_instance_score_threshold: float = 0.0,
    candidates_method: str = "fixed_window",
    features: str = "keypoints",
    scoring_method: str = "oks",
    scoring_reduction: str = "mean",
    track_matching_method: str = "hungarian",
    max_tracks: Optional[int] = None,
    use_flow: bool = False,
    of_img_scale: float = 1.0,
    of_window_size: int = 21,
    of_max_levels: int = 3,
):
    """Entry point to run inference on trained SLEAP-NN models.

    Args:
        data_path: (str) Path to `.slp` file or `.mp4` to run inference on.
        model_paths: (List[str]) List of paths to the directory where the best.ckpt
                and training_config.yaml are saved.
        backbone_ckpt_path: (str) To run inference on any `.ckpt` other than `best.ckpt`
                from the `model_paths` dir, the path to the `.ckpt` file should be passed here.
        head_ckpt_path: (str) Path to `.ckpt` file if a different set of head layer weights
                are to be used. If `None`, the `best.ckpt` from `model_paths` dir is used (or the ckpt
                from `backbone_ckpt_path` if provided.)
        max_instances: (int) Max number of instances to consider from the predictions.
        max_width: (int) Maximum width the image should be padded to. If not provided, the
                values from the training config are used. Default: None.
        max_height: (int) Maximum height the image should be padded to. If not provided, the
                values from the training config are used. Default: None.
        is_rgb: (bool) True if the image has 3 channels (RGB image). If input has only one
                channel when this is set to `True`, then the images from single-channel
                is replicated along the channel axis. If input has three channels and this
                is set to False, then we convert the image to grayscale (single-channel)
                image. Default: False.
        anchor_part: (str) The node name to use as the anchor for the centroid. If not
                provided, the anchor part in the `training_config.yaml` is used.
        provider: (str) Provider class to read the input sleap files.
                Either "LabelsReader" or "VideoReader". Default: None.
        only_labeled_frames: (bool) `True` if inference should be run only on user-labeled frames. Default: `False`.
        only_suggested_frames: (bool) `True` if inference should be run only on unlabeled suggested frames. Default: `False`.
        batch_size: (int) Number of samples per batch. Default: 4.
        queue_maxsize: (int) Maximum size of the frame buffer queue. Default: 8.
        frames: (list) List of frames indices. If `None`, all frames in the video are used. Default: None.
        crop_size: (int) Crop size. If not provided, the crop size from training_config.yaml is used.
                Default: None.
        peak_threshold: (float) Minimum confidence threshold. Peaks with values below
                this will be ignored. Default: 0.2. This can also be `List[float]` for topdown
                centroid and centered-instance model, where the first element corresponds
                to centroid model peak finding threshold and the second element is for
                centered-instance model peak finding.
        integral_refinement: (str) If `None`, returns the grid-aligned peaks with no refinement.
                If `"integral"`, peaks will be refined with integral regression.
                Default: None.
        integral_patch_size: (int) Size of patches to crop around each rough peak as an
                integer scalar. Default: 5.
        return_confmaps: (bool) If `True`, predicted confidence maps will be returned
                along with the predicted peak values and points. Default: False.
        return_pafs: (bool) If `True`, the part affinity fields will be returned together with
                the predicted instances. This will result in slower inference times since
                the data must be copied off of the GPU, but is useful for visualizing the
                raw output of the model. Default: False.
        return_paf_graph: (bool) If `True`, the part affinity field graph will be returned
                together with the predicted instances. The graph is obtained by parsing the
                part affinity fields with the `paf_scorer` instance and is an intermediate
                representation used during instance grouping. Default: False.
        max_edge_length_ratio: (float) The maximum expected length of a connected pair of points
                as a fraction of the image size. Candidate connections longer than this
                length will be penalized during matching. Default: 0.25.
        dist_penalty_weight: (float) A coefficient to scale weight of the distance penalty as
                a scalar float. Set to values greater than 1.0 to enforce the distance
                penalty more strictly.Default: 1.0.
        n_points: (int) Number of points to sample along the line integral. Default: 10.
        min_instance_peaks: Union[int, float] Minimum number of peaks the instance should
                have to be considered a real instance. Instances with fewer peaks than
                this will be discarded (useful for filtering spurious detections).
                Default: 0.
        min_line_scores: (float) Minimum line score (between -1 and 1) required to form a match
                between candidate point pairs. Useful for rejecting spurious detections when
                there are no better ones. Default: 0.25.
        make_labels: (bool) If `True` (the default), returns a `sio.Labels` instance with
                `sio.PredictedInstance`s. If `False`, just return a list of
                dictionaries containing the raw arrays returned by the inference model.
                Default: True.
        output_path: (str) Path to save the labels file if `make_labels` is True.
                Default is current working directory.
        device: (str) Device on which torch.Tensor will be allocated. One of the
                ("cpu", "cuda", "mkldnn", "opengl", "opencl", "ideep", "hip", "msnpu").
                Default: "cpu".
        tracking: (bool) If True, runs tracking on the predicted instances.
        tracking_window_size: Number of frames to look for in the candidate instances to match
                with the current detections. Default: 5.
        tracking_instance_score_threshold: Instance score threshold for creating new tracks.
            Default: 0.0.
        candidates_method: Either of `fixed_window` or `local_queues`. In fixed window
            method, candidates from the last `window_size` frames. In local queues,
            last `window_size` instances for each track ID is considered for matching
            against the current detection. Default: `fixed_window`.
        features: Feature representation for the candidates to update current detections.
            One of [`keypoints`, `centroids`, `bboxes`, `image`]. Default: `keypoints`.
        scoring_method: Method to compute association score between features from the
            current frame and the previous tracks. One of [`oks`, `cosine_sim`, `iou`,
            `euclidean_dist`]. Default: `oks`.
        scoring_reduction: Method to aggregate and reduce multiple scores if there are
            several detections associated with the same track. One of [`mean`, `max`,
            `weighted`]. Default: `mean`.
        track_matching_method: Track matching algorithm. One of `hungarian`, `greedy.
            Default: `hungarian`.
        max_tracks: Meaximum number of new tracks to be created to avoid redundant tracks.
            (only for local queues candidate) Default: None.
        use_flow: If True, `FlowShiftTracker` is used, where the poses are matched using
        optical flow shifts. Default: `False`.
        of_img_scale: Factor to scale the images by when computing optical flow. Decrease
            this to increase performance at the cost of finer accuracy. Sometimes
            decreasing the image scale can improve performance with fast movements.
            Default: 1.0. (only if `use_flow` is True)
        of_window_size: Optical flow window size to consider at each pyramid scale
            level. Default: 21. (only if `use_flow` is True)
        of_max_levels: Number of pyramid scale levels to consider. This is different
            from the scale parameter, which determines the initial image scaling.
            Default: 3. (only if `use_flow` is True)

    Returns:
        Returns `sio.Labels` object if `make_labels` is True. Else this function returns
            a list of Dictionaries with the predictions.

    """
    preprocess_config = {  # if not given, then use from training config
        "is_rgb": is_rgb,
        "crop_hw": (crop_size, crop_size) if crop_size is not None else None,
        "max_width": max_width,
        "max_height": max_height,
        "anchor_part": anchor_part,
    }

    if provider is None:
        if data_path.endswith(".slp"):
            provider = "LabelsReader"
        else:
            provider = "VideoReader"

    if provider == "VideoReader":
        preprocess_config["video_queue_maxsize"] = queue_maxsize
        preprocess_config["frames"] = frames

    # initializes the inference model
    predictor = Predictor.from_model_paths(
        model_paths,
        backbone_ckpt_path=backbone_ckpt_path,
        head_ckpt_path=head_ckpt_path,
        peak_threshold=peak_threshold,
        integral_refinement=integral_refinement,
        integral_patch_size=integral_patch_size,
        batch_size=batch_size,
        max_instances=max_instances,
        return_confmaps=return_confmaps,
        device=device,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    if tracking:
        predictor.tracker = Tracker.from_config(
            candidates_method=candidates_method,
            window_size=tracking_window_size,
            instance_score_threshold=tracking_instance_score_threshold,
            features=features,
            scoring_method=scoring_method,
            scoring_reduction=scoring_reduction,
            track_matching_method=track_matching_method,
            max_tracks=max_tracks,
            use_flow=use_flow,
            of_img_scale=of_img_scale,
            of_window_size=of_window_size,
            of_max_levels=of_max_levels,
        )

    if isinstance(predictor, BottomUpPredictor):
        predictor.inference_model.paf_scorer.max_edge_length_ratio = (
            max_edge_length_ratio
        )
        predictor.inference_model.paf_scorer.dist_penalty_weight = dist_penalty_weight
        predictor.inference_model.return_pafs = return_pafs
        predictor.inference_model.return_paf_graph = return_paf_graph
        predictor.inference_model.paf_scorer.max_edge_length_ratio = (
            max_edge_length_ratio
        )
        predictor.inference_model.paf_scorer.min_line_scores = min_line_scores
        predictor.inference_model.paf_scorer.min_instance_peaks = min_instance_peaks
        predictor.inference_model.paf_scorer.n_points = n_points

    # initialize make_pipeline function

    predictor.make_pipeline(
        provider,
        data_path,
        queue_maxsize,
        frames,
        only_labeled_frames,
        only_suggested_frames,
    )

    # run predict
    output = predictor.predict(
        make_labels=make_labels,
        save_path=output_path,
    )

    return output
