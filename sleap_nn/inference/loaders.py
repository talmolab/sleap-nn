"""Standalone checkpoint loading for the inference pipeline.

Nothing user-facing here -- the public API remains
:class:`sleap_nn.inference.Predictor` via
:meth:`Predictor.from_model_paths`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import attrs
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    import sleap_io as sio

from sleap_nn.config.training_job_config import TrainingJobConfig
from sleap_nn.config.utils import get_model_type_from_cfg, resolve_model_dir
from sleap_nn.inference.bottomup import (
    BottomUpInferenceModel,
    BottomUpMultiClassInferenceModel,
)
from sleap_nn.inference.paf_grouping import PAFScorer
from sleap_nn.inference.single_instance import SingleInstanceInferenceModel
from sleap_nn.inference.topdown import (
    CentroidCrop,
    FindInstancePeaks,
    FindInstancePeaksGroundTruth,
    TopDownInferenceModel,
    TopDownMultiClassFindInstancePeaks,
)
from sleap_nn.inference.utils import get_skeleton_from_config
from sleap_nn.legacy_models import load_legacy_model
from sleap_nn.inference.segmentation import BottomUpSegmentationInferenceModel
from sleap_nn.training.lightning_modules import (
    BottomUpLightningModule,
    BottomUpMultiClassLightningModule,
    BottomUpSegmentationLightningModule,
    CentroidLightningModule,
    SingleInstanceLightningModule,
    TopDownCenteredInstanceLightningModule,
    TopDownCenteredInstanceMultiClassLightningModule,
)

# ─────────────────────────────────────────────────────────────────────────
# LoadedAssets — the bag of attributes the factory's _build_*_layer helpers
# consume.
# ─────────────────────────────────────────────────────────────────────────


@attrs.define(eq=False, repr=False)
class LoadedAssets:
    """Everything the factory's ``_build_*_layer`` helpers need."""

    inference_model: Any  # Union of all *InferenceModel types
    preprocess_config: "DictConfig"
    skeletons: list["sio.Skeleton"]

    bottomup_config: Optional["DictConfig"] = None
    backbone_type: Optional[str] = None
    max_stride: Optional[int] = None

    centroid_config: Optional["DictConfig"] = None
    confmap_config: Optional["DictConfig"] = None

    # Cap on instances per frame. Threaded through to the bottom-up grouping
    # stage (the legacy ``BottomUpInferenceModel`` has no such field, so the
    # value is carried on the assets and read by ``_build_bottomup_layer``).
    max_instances: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────
# Per-checkpoint helpers (shared across all model types)
# ─────────────────────────────────────────────────────────────────────────


def _load_training_config(ckpt_dir: str) -> tuple["DictConfig", bool]:
    """Load ``training_config.{yaml,json}``.

    Returns:
        ``(config, is_legacy)`` where *is_legacy* is ``True`` for
        SLEAP <= 1.4 JSON configs.
    """
    p = Path(ckpt_dir)
    if (p / "training_config.yaml").exists():
        return OmegaConf.load((p / "training_config.yaml").as_posix()), False
    if (p / "training_config.json").exists():
        return (
            TrainingJobConfig.load_sleap_config(
                (p / "training_config.json").as_posix()
            ),
            True,
        )
    raise FileNotFoundError(
        f"No training_config.yaml or training_config.json in {ckpt_dir}"
    )


def _detect_backbone_type(config: Any) -> str:
    """Return the first non-None backbone key in ``model_config.backbone_config``."""
    for k, v in config.model_config.backbone_config.items():
        if v is not None:
            return k
    raise ValueError("No backbone found in model_config.backbone_config")


def _common_lightning_kwargs(config: Any, backbone_type: str, model_type: str) -> dict:
    """The kwargs every ``LightningModule.load_from_checkpoint`` call needs.

    Identical across all six predictor types in the legacy code.
    """
    tc = config.trainer_config
    hkm = tc.online_hard_keypoint_mining
    return dict(
        model_type=model_type,
        backbone_type=backbone_type,
        backbone_config=config.model_config.backbone_config,
        head_configs=config.model_config.head_configs,
        pretrained_backbone_weights=None,
        pretrained_head_weights=None,
        init_weights=config.model_config.init_weights,
        lr_scheduler=tc.lr_scheduler,
        online_mining=hkm.online_mining,
        hard_to_easy_ratio=hkm.hard_to_easy_ratio,
        min_hard_keypoints=hkm.min_hard_keypoints,
        max_hard_keypoints=hkm.max_hard_keypoints,
        loss_scale=hkm.loss_scale,
        optimizer=tc.optimizer_name,
        learning_rate=tc.optimizer.lr,
        amsgrad=tc.optimizer.amsgrad,
    )


def _apply_ckpt_overrides(
    module: torch.nn.Module,
    *,
    backbone_ckpt_path: Optional[str],
    head_ckpt_path: Optional[str],
    device: str,
) -> None:
    """Apply optional backbone / head checkpoint overrides.

    Three branches:
    (a) both overrides → only ``.backbone`` keys from backbone_ckpt_path
    (b) backbone only  → all keys from backbone_ckpt_path
    (c) head only      → only ``.head_layers`` keys from head_ckpt_path
    """
    if backbone_ckpt_path is not None and head_ckpt_path is not None:
        logger.info(f"Loading backbone weights from `{backbone_ckpt_path}` ...")
        ckpt = torch.load(backbone_ckpt_path, map_location=device, weights_only=False)
        ckpt["state_dict"] = {
            k: v for k, v in ckpt["state_dict"].items() if ".backbone" in k
        }
        module.load_state_dict(ckpt["state_dict"], strict=False)
    elif backbone_ckpt_path is not None:
        logger.info(f"Loading weights from `{backbone_ckpt_path}` ...")
        ckpt = torch.load(backbone_ckpt_path, map_location=device, weights_only=False)
        module.load_state_dict(ckpt["state_dict"], strict=False)

    if head_ckpt_path is not None:
        logger.info(f"Loading head weights from `{head_ckpt_path}` ...")
        ckpt = torch.load(head_ckpt_path, map_location=device, weights_only=False)
        ckpt["state_dict"] = {
            k: v for k, v in ckpt["state_dict"].items() if ".head_layers" in k
        }
        module.load_state_dict(ckpt["state_dict"], strict=False)


def _load_lightning_module(
    cls: type,
    ckpt_dir: str,
    *,
    model_type: str,
    device: str,
    backbone_ckpt_path: Optional[str] = None,
    head_ckpt_path: Optional[str] = None,
) -> tuple[torch.nn.Module, "DictConfig", str]:
    """Generic per-checkpoint loader.

    Returns:
        ``(module, config, backbone_type)``
    """
    config, is_legacy = _load_training_config(ckpt_dir)
    backbone_type = _detect_backbone_type(config)
    kwargs = _common_lightning_kwargs(config, backbone_type, model_type)

    if not is_legacy:
        ckpt_path = (Path(ckpt_dir) / "best.ckpt").as_posix()
        module = cls.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            map_location=device,
            weights_only=False,
            **kwargs,
        )
    else:
        converted = load_legacy_model(model_dir=ckpt_dir)
        module = cls(**kwargs)
        module.eval()
        module.model = converted
        module.to(device)

    module.eval()
    _apply_ckpt_overrides(
        module,
        backbone_ckpt_path=backbone_ckpt_path,
        head_ckpt_path=head_ckpt_path,
        device=device,
    )
    module.to(device)
    return module, config, backbone_type


def _resolve_preprocess_config(preprocess_config: Any, training_config: Any) -> Any:
    """Fill ``None`` fields in *preprocess_config* from the training config.

    Mirrors the resolution loop in every legacy ``from_trained_models``.
    """
    for k, v in preprocess_config.items():
        if v is None:
            preprocess_config[k] = (
                training_config.data_config.preprocessing[k]
                if k in training_config.data_config.preprocessing
                else None
            )
    return preprocess_config


# ─────────────────────────────────────────────────────────────────────────
# Per-model-type builders
# ─────────────────────────────────────────────────────────────────────────


def _build_single_instance(
    ckpt_path: str,
    *,
    device: str,
    backbone_ckpt_path: Optional[str],
    head_ckpt_path: Optional[str],
    peak_threshold: float,
    integral_refinement: str,
    integral_patch_size: int,
    return_confmaps: bool,
    preprocess_config: Any,
) -> LoadedAssets:
    module, config, backbone_type = _load_lightning_module(
        SingleInstanceLightningModule,
        ckpt_path,
        model_type="single_instance",
        device=device,
        backbone_ckpt_path=backbone_ckpt_path,
        head_ckpt_path=head_ckpt_path,
    )
    skeletons = get_skeleton_from_config(config.data_config.skeletons)
    max_stride = config.model_config.backbone_config[backbone_type]["max_stride"]

    preprocess_config = _resolve_preprocess_config(preprocess_config, config)

    inference_model = SingleInstanceInferenceModel(
        torch_model=module,
        peak_threshold=peak_threshold,
        output_stride=config.model_config.head_configs.single_instance.confmaps.output_stride,
        refinement=integral_refinement,
        integral_patch_size=integral_patch_size,
        return_confmaps=return_confmaps,
        input_scale=config.data_config.preprocessing.scale,
    )
    return LoadedAssets(
        inference_model=inference_model,
        preprocess_config=preprocess_config,
        skeletons=skeletons,
        backbone_type=backbone_type,
        max_stride=max_stride,
        confmap_config=config,
    )


def _build_bottomup(
    ckpt_path: str,
    *,
    device: str,
    backbone_ckpt_path: Optional[str],
    head_ckpt_path: Optional[str],
    peak_threshold: float,
    integral_refinement: str,
    integral_patch_size: int,
    max_instances: Optional[int],
    return_confmaps: bool,
    preprocess_config: Any,
    max_edge_length_ratio: float = 0.25,
    dist_penalty_weight: float = 1.0,
    n_points: int = 10,
    min_instance_peaks: Union[int, float] = 0,
    min_line_scores: float = 0.25,
) -> LoadedAssets:
    module, config, backbone_type = _load_lightning_module(
        BottomUpLightningModule,
        ckpt_path,
        model_type="bottomup",
        device=device,
        backbone_ckpt_path=backbone_ckpt_path,
        head_ckpt_path=head_ckpt_path,
    )
    skeletons = get_skeleton_from_config(config.data_config.skeletons)

    preprocess_config = _resolve_preprocess_config(preprocess_config, config)

    # Thread the bottom-up PAF grouping knobs from the CLI/API into the scorer
    # (legacy ran_inference set these on the live scorer; the new infer flow
    # silently dropped them). #583.
    paf_scorer = PAFScorer.from_config(
        config=OmegaConf.create(
            {
                "confmaps": config.model_config.head_configs.bottomup["confmaps"],
                "pafs": config.model_config.head_configs.bottomup["pafs"],
            }
        ),
        max_edge_length_ratio=max_edge_length_ratio,
        dist_penalty_weight=dist_penalty_weight,
        n_points=n_points,
        min_instance_peaks=min_instance_peaks,
        min_line_scores=min_line_scores,
    )

    inference_model = BottomUpInferenceModel(
        torch_model=module,
        paf_scorer=paf_scorer,
        peak_threshold=peak_threshold,
        cms_output_stride=config.model_config.head_configs.bottomup.confmaps.output_stride,
        pafs_output_stride=config.model_config.head_configs.bottomup.pafs.output_stride,
        refinement=integral_refinement,
        integral_patch_size=integral_patch_size,
        return_confmaps=return_confmaps,
        input_scale=config.data_config.preprocessing.scale,
    )
    return LoadedAssets(
        inference_model=inference_model,
        preprocess_config=preprocess_config,
        skeletons=skeletons,
        bottomup_config=config,
        backbone_type=backbone_type,
        max_instances=max_instances,
    )


def _build_bottomup_multiclass(
    ckpt_path: str,
    *,
    device: str,
    backbone_ckpt_path: Optional[str],
    head_ckpt_path: Optional[str],
    peak_threshold: float,
    integral_refinement: str,
    integral_patch_size: int,
    max_instances: Optional[int],
    return_confmaps: bool,
    preprocess_config: Any,
) -> LoadedAssets:
    module, config, backbone_type = _load_lightning_module(
        BottomUpMultiClassLightningModule,
        ckpt_path,
        model_type="multi_class_bottomup",
        device=device,
        backbone_ckpt_path=backbone_ckpt_path,
        head_ckpt_path=head_ckpt_path,
    )
    skeletons = get_skeleton_from_config(config.data_config.skeletons)

    preprocess_config = _resolve_preprocess_config(preprocess_config, config)

    inference_model = BottomUpMultiClassInferenceModel(
        torch_model=module,
        peak_threshold=peak_threshold,
        cms_output_stride=config.model_config.head_configs.multi_class_bottomup.confmaps.output_stride,
        class_maps_output_stride=config.model_config.head_configs.multi_class_bottomup.class_maps.output_stride,
        refinement=integral_refinement,
        integral_patch_size=integral_patch_size,
        return_confmaps=return_confmaps,
        input_scale=config.data_config.preprocessing.scale,
    )
    return LoadedAssets(
        inference_model=inference_model,
        preprocess_config=preprocess_config,
        skeletons=skeletons,
        bottomup_config=config,
        backbone_type=backbone_type,
        max_instances=max_instances,
    )


def _build_bottomup_segmentation(
    ckpt_path: str,
    *,
    device: str,
    backbone_ckpt_path: Optional[str],
    head_ckpt_path: Optional[str],
    peak_threshold: float,
    integral_refinement: str,
    integral_patch_size: int,
    return_confmaps: bool,
    preprocess_config: Any,
    fg_threshold: float = 0.5,
    min_mask_area: int = 0,
    max_instances: Optional[int] = None,
    center_nms_kernel: int = 3,
    mask_cleanup: bool = False,
) -> LoadedAssets:
    """Load a ``BottomUpSegmentationLightningModule`` and wrap it for inference.

    ``integral_refinement`` / ``integral_patch_size`` / ``return_confmaps`` are
    accepted for a uniform ``common_kwargs`` call signature but are unused
    (segmentation has no keypoint peak refinement / confmaps).

    ``min_mask_area`` (original-image pixels) drops tiny spurious predicted
    masks (over-segmentation); ``0`` disables it. It is carried on the
    inference model and applied in ``SegmentationLayer.postprocess``.
    """
    module, config, backbone_type = _load_lightning_module(
        BottomUpSegmentationLightningModule,
        ckpt_path,
        model_type="bottomup_segmentation",
        device=device,
        backbone_ckpt_path=backbone_ckpt_path,
        head_ckpt_path=head_ckpt_path,
    )
    # Mask-only labels may carry no skeleton; tolerate an empty/missing one.
    try:
        skeletons = get_skeleton_from_config(config.data_config.skeletons)
    except Exception:  # noqa: BLE001 — skeleton is optional for segmentation
        skeletons = []

    max_stride = config.model_config.backbone_config[backbone_type]["max_stride"]
    preprocess_config = _resolve_preprocess_config(preprocess_config, config)

    seg_cfg = config.model_config.head_configs.bottomup_segmentation
    output_stride = seg_cfg.segmentation.output_stride

    inference_model = BottomUpSegmentationInferenceModel(
        torch_model=module,
        fg_threshold=fg_threshold,
        peak_threshold=peak_threshold,
        output_stride=output_stride,
        input_scale=config.data_config.preprocessing.scale,
        min_mask_area=min_mask_area,
        max_instances=max_instances,
        center_nms_kernel=center_nms_kernel,
        mask_cleanup=mask_cleanup,
    )
    return LoadedAssets(
        inference_model=inference_model,
        preprocess_config=preprocess_config,
        skeletons=skeletons,
        bottomup_config=config,
        backbone_type=backbone_type,
        max_stride=max_stride,
    )


def _build_topdown(
    centroid_ckpt_path: Optional[str],
    confmap_ckpt_path: Optional[str],
    *,
    device: str,
    backbone_ckpt_path: Optional[str],
    head_ckpt_path: Optional[str],
    peak_threshold: Union[float, List[float]],
    integral_refinement: str,
    integral_patch_size: int,
    max_instances: Optional[int],
    return_confmaps: bool,
    preprocess_config: Any,
    anchor_part: Optional[str],
) -> LoadedAssets:
    if isinstance(peak_threshold, list):
        centroid_peak_threshold = peak_threshold[0]
        centered_instance_peak_threshold = peak_threshold[1]
    else:
        centroid_peak_threshold = peak_threshold
        centered_instance_peak_threshold = peak_threshold

    centroid_config = None
    centroid_model = None
    centroid_backbone_type = None
    confmap_config = None
    confmap_model = None
    centered_instance_backbone_type = None
    skeletons = None

    if centroid_ckpt_path is not None:
        centroid_model, centroid_config, centroid_backbone_type = (
            _load_lightning_module(
                CentroidLightningModule,
                centroid_ckpt_path,
                model_type="centroid",
                device=device,
                backbone_ckpt_path=backbone_ckpt_path,
                head_ckpt_path=head_ckpt_path,
            )
        )
        skeletons = get_skeleton_from_config(centroid_config.data_config.skeletons)

    if confmap_ckpt_path is not None:
        confmap_model, confmap_config, centered_instance_backbone_type = (
            _load_lightning_module(
                TopDownCenteredInstanceLightningModule,
                confmap_ckpt_path,
                model_type="centered_instance",
                device=device,
                backbone_ckpt_path=backbone_ckpt_path,
                head_ckpt_path=head_ckpt_path,
            )
        )
        skeletons = get_skeleton_from_config(confmap_config.data_config.skeletons)

    # Resolve preprocess_config from both training configs. The confmap
    # config supplies crop_size (absent from centroid training configs),
    # so resolve centroid first, then confmap to fill remaining Nones.
    # Capture whether the caller explicitly supplied a crop_size so the confmap
    # default below does not override an intentional user value.
    user_crop_size = preprocess_config.crop_size
    if centroid_config is not None:
        preprocess_config = _resolve_preprocess_config(
            preprocess_config, centroid_config
        )
    if confmap_config is not None:
        preprocess_config = _resolve_preprocess_config(
            preprocess_config, confmap_config
        )
        # crop_size is a centered-instance property: force it from the confmap
        # config so a stray non-null centroid crop_size can't win — but only
        # when the caller didn't supply an explicit crop_size (legacy parity;
        # #584).
        confmap_crop = confmap_config.data_config.preprocessing.crop_size
        if user_crop_size is None and confmap_crop is not None:
            preprocess_config.crop_size = confmap_crop

    # Resolve anchor_ind
    if anchor_part is not None:
        anchor_ind = skeletons[0].node_names.index(anchor_part)
    else:
        anch_pt = None
        if centroid_config is not None:
            anch_pt = (
                centroid_config.model_config.head_configs.centroid.confmaps.anchor_part
            )
        if confmap_config is not None:
            anch_pt = (
                confmap_config.model_config.head_configs.centered_instance.confmaps.anchor_part
            )
        anchor_ind = (
            skeletons[0].node_names.index(anch_pt) if anch_pt is not None else None
        )

    # Build CentroidCrop
    return_crops = confmap_model is not None
    if centroid_config is None:
        centroid_crop = CentroidCrop(
            use_gt_centroids=True,
            crop_hw=(preprocess_config.crop_size, preprocess_config.crop_size),
            anchor_ind=anchor_ind,
            return_crops=return_crops,
        )
    else:
        max_stride_centroid = centroid_config.model_config.backbone_config[
            centroid_backbone_type
        ]["max_stride"]
        centroid_crop = CentroidCrop(
            torch_model=centroid_model,
            peak_threshold=centroid_peak_threshold,
            output_stride=centroid_config.model_config.head_configs.centroid.confmaps.output_stride,
            refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            return_confmaps=return_confmaps,
            return_crops=return_crops,
            max_instances=max_instances,
            max_stride=max_stride_centroid,
            input_scale=centroid_config.data_config.preprocessing.scale,
            crop_hw=(preprocess_config.crop_size, preprocess_config.crop_size),
            use_gt_centroids=False,
            anchor_ind=anchor_ind,
        )

    # Build FindInstancePeaks
    if confmap_config is None:
        instance_peaks = FindInstancePeaksGroundTruth()
    else:
        max_stride_inst = confmap_config.model_config.backbone_config[
            centered_instance_backbone_type
        ]["max_stride"]
        instance_peaks = FindInstancePeaks(
            torch_model=confmap_model,
            peak_threshold=centered_instance_peak_threshold,
            output_stride=confmap_config.model_config.head_configs.centered_instance.confmaps.output_stride,
            refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            return_confmaps=return_confmaps,
            max_stride=max_stride_inst,
            input_scale=confmap_config.data_config.preprocessing.scale,
        )

    inference_model = TopDownInferenceModel(
        centroid_crop=centroid_crop, instance_peaks=instance_peaks
    )
    return LoadedAssets(
        inference_model=inference_model,
        preprocess_config=preprocess_config,
        skeletons=skeletons,
        centroid_config=centroid_config,
        confmap_config=confmap_config,
        backbone_type=centered_instance_backbone_type or centroid_backbone_type,
    )


def _build_topdown_multiclass(
    centroid_ckpt_path: Optional[str],
    confmap_ckpt_path: Optional[str],
    *,
    device: str,
    backbone_ckpt_path: Optional[str],
    head_ckpt_path: Optional[str],
    peak_threshold: Union[float, List[float]],
    integral_refinement: str,
    integral_patch_size: int,
    max_instances: Optional[int],
    return_confmaps: bool,
    preprocess_config: Any,
    anchor_part: Optional[str],
) -> LoadedAssets:
    if isinstance(peak_threshold, list):
        centroid_peak_threshold = peak_threshold[0]
        centered_instance_peak_threshold = peak_threshold[1]
    else:
        centroid_peak_threshold = peak_threshold
        centered_instance_peak_threshold = peak_threshold

    centroid_config = None
    centroid_model = None
    centroid_backbone_type = None
    confmap_config = None
    confmap_model = None
    centered_instance_backbone_type = None
    skeletons = None

    if centroid_ckpt_path is not None:
        centroid_model, centroid_config, centroid_backbone_type = (
            _load_lightning_module(
                CentroidLightningModule,
                centroid_ckpt_path,
                model_type="centroid",
                device=device,
                backbone_ckpt_path=backbone_ckpt_path,
                head_ckpt_path=head_ckpt_path,
            )
        )
        skeletons = get_skeleton_from_config(centroid_config.data_config.skeletons)

    if confmap_ckpt_path is not None:
        confmap_model, confmap_config, centered_instance_backbone_type = (
            _load_lightning_module(
                TopDownCenteredInstanceMultiClassLightningModule,
                confmap_ckpt_path,
                model_type="multi_class_topdown",
                device=device,
                backbone_ckpt_path=backbone_ckpt_path,
                head_ckpt_path=head_ckpt_path,
            )
        )
        skeletons = get_skeleton_from_config(confmap_config.data_config.skeletons)

    # Capture whether the caller explicitly supplied a crop_size so the confmap
    # default below does not override an intentional user value.
    user_crop_size = preprocess_config.crop_size
    if centroid_config is not None:
        preprocess_config = _resolve_preprocess_config(
            preprocess_config, centroid_config
        )
    if confmap_config is not None:
        preprocess_config = _resolve_preprocess_config(
            preprocess_config, confmap_config
        )
        # crop_size is a centered-instance property: force it from the confmap
        # config so a stray non-null centroid crop_size can't win — but only
        # when the caller didn't supply an explicit crop_size (legacy parity;
        # #584).
        confmap_crop = confmap_config.data_config.preprocessing.crop_size
        if user_crop_size is None and confmap_crop is not None:
            preprocess_config.crop_size = confmap_crop

    # Resolve anchor_ind
    if anchor_part is not None:
        anchor_ind = skeletons[0].node_names.index(anchor_part)
    else:
        anch_pt = None
        if centroid_config is not None:
            anch_pt = (
                centroid_config.model_config.head_configs.centroid.confmaps.anchor_part
            )
        if confmap_config is not None:
            anch_pt = (
                confmap_config.model_config.head_configs.multi_class_topdown.confmaps.anchor_part
            )
        anchor_ind = (
            skeletons[0].node_names.index(anch_pt) if anch_pt is not None else None
        )

    return_crops = confmap_model is not None
    if centroid_config is None:
        centroid_crop = CentroidCrop(
            use_gt_centroids=True,
            crop_hw=(preprocess_config.crop_size, preprocess_config.crop_size),
            anchor_ind=anchor_ind,
            return_crops=return_crops,
        )
    else:
        max_stride_centroid = centroid_config.model_config.backbone_config[
            centroid_backbone_type
        ]["max_stride"]
        centroid_crop = CentroidCrop(
            torch_model=centroid_model,
            peak_threshold=centroid_peak_threshold,
            output_stride=centroid_config.model_config.head_configs.centroid.confmaps.output_stride,
            refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            return_confmaps=return_confmaps,
            return_crops=return_crops,
            max_instances=max_instances,
            max_stride=max_stride_centroid,
            input_scale=centroid_config.data_config.preprocessing.scale,
            crop_hw=(preprocess_config.crop_size, preprocess_config.crop_size),
            use_gt_centroids=False,
            anchor_ind=anchor_ind,
        )

    max_stride_inst = confmap_config.model_config.backbone_config[
        centered_instance_backbone_type
    ]["max_stride"]
    instance_peaks = TopDownMultiClassFindInstancePeaks(
        torch_model=confmap_model,
        peak_threshold=centered_instance_peak_threshold,
        output_stride=confmap_config.model_config.head_configs.multi_class_topdown.confmaps.output_stride,
        refinement=integral_refinement,
        integral_patch_size=integral_patch_size,
        return_confmaps=return_confmaps,
        max_stride=max_stride_inst,
        input_scale=confmap_config.data_config.preprocessing.scale,
    )

    inference_model = TopDownInferenceModel(
        centroid_crop=centroid_crop, instance_peaks=instance_peaks
    )
    return LoadedAssets(
        inference_model=inference_model,
        preprocess_config=preprocess_config,
        skeletons=skeletons,
        centroid_config=centroid_config,
        confmap_config=confmap_config,
        backbone_type=centered_instance_backbone_type or centroid_backbone_type,
    )


# ─────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────


def load_model_assets(
    model_paths: List[str],
    *,
    device: str = "cpu",
    backbone_ckpt_path: Optional[str] = None,
    head_ckpt_path: Optional[str] = None,
    peak_threshold: Union[float, List[float]] = 0.2,
    integral_refinement: str = "integral",
    integral_patch_size: int = 5,
    max_instances: Optional[int] = None,
    return_confmaps: bool = False,
    preprocess_config: Optional["DictConfig"] = None,
    anchor_part: Optional[str] = None,
    max_edge_length_ratio: float = 0.25,
    dist_penalty_weight: float = 1.0,
    n_points: int = 10,
    min_instance_peaks: Union[int, float] = 0,
    min_line_scores: float = 0.25,
    fg_threshold: float = 0.5,
    min_mask_area: int = 0,
    center_nms_kernel: int = 3,
    mask_cleanup: bool = False,
) -> tuple[LoadedAssets, List[str]]:
    """Load checkpoints and build inference models.

    Each entry in ``model_paths`` may be a model directory, or a path to its
    ``best.ckpt`` checkpoint or ``training_config.{yaml,json}`` config file; every
    form is resolved to the model directory (#575).

    The five bottom-up PAF grouping knobs (``max_edge_length_ratio``,
    ``dist_penalty_weight``, ``n_points``, ``min_instance_peaks``,
    ``min_line_scores``) are forwarded ONLY to the plain bottom-up builder
    (legacy applied them only to ``BottomUpPredictor``); they are inert for
    other model types. Likewise ``fg_threshold`` / ``min_mask_area`` are
    forwarded ONLY to the bottom-up segmentation builder.

    Returns:
        ``(loaded_assets, model_types)`` — *model_types* is the list of
        detected model types (one per path in *model_paths*).
    """
    if preprocess_config is None:
        preprocess_config = OmegaConf.create(
            {
                "ensure_rgb": None,
                "ensure_grayscale": None,
                "crop_size": None,
                "max_width": None,
                "max_height": None,
                "scale": None,
            }
        )

    # Accept a model directory, a best.ckpt path, or a training_config.{yaml,json}
    # path for each entry; resolve every form to its model directory before
    # detection/dispatch (the builders below index back into model_paths and join
    # `best.ckpt` onto it). #575.
    model_paths = [resolve_model_dir(mp) for mp in model_paths]

    model_types: List[str] = []
    configs: List[Any] = []
    for mp in model_paths:
        cfg, _ = _load_training_config(mp)
        configs.append(cfg)
        model_types.append(get_model_type_from_cfg(config=cfg))

    common_kwargs = dict(
        device=device,
        backbone_ckpt_path=backbone_ckpt_path,
        head_ckpt_path=head_ckpt_path,
        peak_threshold=peak_threshold,
        integral_refinement=integral_refinement,
        integral_patch_size=integral_patch_size,
        return_confmaps=return_confmaps,
        preprocess_config=preprocess_config,
    )

    # Dispatch on detected model type. The order (bottomup checked before the
    # topdown family) differs from legacy but is inert for parity: each model
    # path is exactly one type and bottom-up is a single-stage model never
    # combined with a centroid/centered-instance pair, so the branches are
    # mutually exclusive (a mixed bottomup+topdown path list is not a supported
    # workflow and falls through to the final ValueError). #584.
    if "single_instance" in model_types:
        path = model_paths[model_types.index("single_instance")]
        assets = _build_single_instance(path, **common_kwargs)

    elif "bottomup" in model_types:
        path = model_paths[model_types.index("bottomup")]
        assets = _build_bottomup(
            path,
            max_instances=max_instances,
            max_edge_length_ratio=max_edge_length_ratio,
            dist_penalty_weight=dist_penalty_weight,
            n_points=n_points,
            min_instance_peaks=min_instance_peaks,
            min_line_scores=min_line_scores,
            **common_kwargs,
        )

    elif "multi_class_bottomup" in model_types:
        path = model_paths[model_types.index("multi_class_bottomup")]
        assets = _build_bottomup_multiclass(
            path, max_instances=max_instances, **common_kwargs
        )

    elif "bottomup_segmentation" in model_types:
        path = model_paths[model_types.index("bottomup_segmentation")]
        assets = _build_bottomup_segmentation(
            path,
            fg_threshold=fg_threshold,
            min_mask_area=min_mask_area,
            max_instances=max_instances,
            center_nms_kernel=center_nms_kernel,
            mask_cleanup=mask_cleanup,
            **common_kwargs,
        )

    elif (
        "centroid" in model_types
        or "centered_instance" in model_types
        or "multi_class_topdown" in model_types
    ):
        centroid_path = None
        confmap_path = None
        if "centroid" in model_types:
            centroid_path = model_paths[model_types.index("centroid")]
        if "centered_instance" in model_types:
            confmap_path = model_paths[model_types.index("centered_instance")]
            assets = _build_topdown(
                centroid_path,
                confmap_path,
                max_instances=max_instances,
                anchor_part=anchor_part,
                **common_kwargs,
            )
        elif "multi_class_topdown" in model_types:
            confmap_path = model_paths[model_types.index("multi_class_topdown")]
            assets = _build_topdown_multiclass(
                centroid_path,
                confmap_path,
                max_instances=max_instances,
                anchor_part=anchor_part,
                **common_kwargs,
            )
        else:
            # centroid-only: still goes through _build_topdown with confmap=None
            assets = _build_topdown(
                centroid_path,
                None,
                max_instances=max_instances,
                anchor_part=anchor_part,
                **common_kwargs,
            )
    else:
        raise ValueError(
            f"Could not create inference assets from model paths:\n{model_paths}\n"
            f"Detected types: {model_types}"
        )

    return assets, model_types
