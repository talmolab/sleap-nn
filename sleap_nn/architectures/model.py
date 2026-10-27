"""This module defines the main SLEAP model class for defining a trainable model.

This is a higher level wrapper around `nn.Module` that holds all the configuration
parameters required to construct the actual model. This allows for easy querying of the
model configuration without actually instantiating the model itself.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from torch import nn
import math
from loguru import logger
from sleap_nn.architectures.heads import (
    Head,
    CentroidConfmapsHead,
    SingleInstanceConfmapsHead,
    CenteredInstanceConfmapsHead,
    MultiInstanceConfmapsHead,
    PartAffinityFieldsHead,
    ClassMapsHead,
    ClassVectorsHead,
    OffsetRefinementHead,
    SegmentationHead,
    InstanceCenterHead,
    CenterOffsetHead,
    EmbeddingHead,
    BBoxCenterHead,
    BBoxSizeHead,
    BBoxOffsetHead,
    MultiScaleHead,
    SSDHead,
    default_embedding_pool,
    embedding_encoder_is_frozen,
)
from sleap_nn.architectures.unet import UNet
from sleap_nn.architectures.convnext import ConvNextWrapper
from sleap_nn.architectures.swint import SwinTWrapper
from sleap_nn.architectures.pretrained import PretrainedBackbone
from sleap_nn.architectures.detr import (
    DETR_BACKBONES,
    build_bbox_detr_head,
    build_detr_backbone,
)
import torchvision.transforms.v2.functional as F
from sleap_nn.config.model_config import DEFAULT_BBOX_CATEGORY, ssd_level_strides


def get_backbone(
    backbone: str, backbone_config: DictConfig, spatial_heads: bool = False
) -> nn.Module:
    """Get a backbone model `nn.Module` based on the provided name.

    This function returns an instance of a PyTorch `nn.Module`
    corresponding to the given backbone name.

    Args:
        backbone (str): Name of the backbone. Supported values are 'unet',
            'convnext', 'swint', and 'pretrained' (external HuggingFace backbone).
        backbone_config (DictConfig): A config for the backbone.
        spatial_heads (bool): Whether the model has a spatial head (confidence
            maps, PAFs, segmentation, ...). Only the `pretrained` backbone reads
            it: under `mode: auto` a plain ViT then gets its feature-pyramid neck
            and decoder instead of staying encoder-only. Training setup records
            the resolved mode in the config (`check_output_strides`); this covers
            a `Model` built from a config that did not go through it.

    Returns:
        nn.Module: An instance of the requested backbone model.

    Raises:
        KeyError: If the provided backbone name is not one of the supported values.
    """
    backbones = {
        "unet": UNet,
        "convnext": ConvNextWrapper,
        "swint": SwinTWrapper,
        "pretrained": PretrainedBackbone,
    }

    if backbone in DETR_BACKBONES:
        message = f"The {backbone} backbone is only supported for bbox_detr models."
        logger.error(message)
        raise KeyError(message)
    if backbone not in backbones:
        message = f"Unsupported backbone: {backbone}. Supported backbones are: {', '.join(backbones.keys())}"
        logger.error(message)
        raise KeyError(message)

    if backbone == "pretrained":
        return PretrainedBackbone.from_config(
            backbone_config, spatial_heads=spatial_heads
        )
    return backbones[backbone].from_config(backbone_config)


def get_head(model_type: str, head_config: DictConfig) -> Head:
    """Get a head `nn.Module` based on the provided name.

    This function returns an instance of a PyTorch `nn.Module`
    corresponding to the given head name.

    Args:
        model_type (str): Name of the head. Supported values are
            - 'single_instance'
            - 'centroid'
            - 'centered_instance'
            - 'bottomup'
            - 'multi_class_bottomup'
            - 'multi_class_topdown'
            - 'bottomup_segmentation'
            - 'centered_instance_segmentation'
            - 'embedding'
            - 'bbox_centernet'
            - 'bbox_ssd'
            ('bbox_detr' has no `Head`; `Model` builds its detector directly.)
        head_config (DictConfig): A config for the head.

    Returns:
        nn.Module: An instance of the requested head.
    """
    heads = []
    if model_type == "single_instance":
        heads.append(SingleInstanceConfmapsHead(**head_config.confmaps))

    elif model_type == "centered_instance":
        heads.append(CenteredInstanceConfmapsHead(**head_config.confmaps))

    elif model_type == "centroid":
        heads.append(CentroidConfmapsHead(**head_config.confmaps))

    elif model_type == "bottomup":
        heads.append(MultiInstanceConfmapsHead(**head_config.confmaps))
        heads.append(PartAffinityFieldsHead(**head_config.pafs))

    elif model_type == "multi_class_bottomup":
        heads.append(MultiInstanceConfmapsHead(**head_config.confmaps))
        heads.append(ClassMapsHead(**head_config.class_maps))

    elif model_type == "multi_class_topdown":
        heads.append(CenteredInstanceConfmapsHead(**head_config.confmaps))
        heads.append(ClassVectorsHead(**head_config.class_vectors))

    elif model_type == "bottomup_segmentation":
        # SegmentationHead takes only output_stride/loss_weight; the leaf's other
        # fields (bce/dice/pos_weight loss knobs, target_maxpool) are training/data
        # concerns consumed by the LightningModule + dataset, not the head arch.
        seg = head_config.segmentation
        heads.append(
            SegmentationHead(
                output_stride=seg.output_stride, loss_weight=seg.loss_weight
            )
        )
        heads.append(InstanceCenterHead(**head_config.center))
        heads.append(CenterOffsetHead(**head_config.offsets))

    elif model_type == "centered_instance_segmentation":
        # Top-down crop-centered segmentation: a lone foreground-mask head on a
        # centroid crop (the centered instance). `anchor_part` lives in the head
        # leaf but is a data-pipeline concern, not a SegmentationHead argument.
        seg = head_config.segmentation
        heads.append(
            SegmentationHead(
                output_stride=seg.output_stride, loss_weight=seg.loss_weight
            )
        )

    elif model_type == "semantic_segmentation":
        # Whole-frame semantic segmentation: a lone foreground-mask head on the
        # full frame (no crop, no center/offset, no grouping). Only output_stride /
        # loss_weight are head args; the leaf's loss/target knobs are consumed
        # downstream (LightningModule loss + dataset target), not by the head.
        seg = head_config.segmentation
        heads.append(
            SegmentationHead(
                output_stride=seg.output_stride, loss_weight=seg.loss_weight
            )
        )

    elif model_type == "embedding":
        # Lone pooled head (crop -> vector). The `objective` block lives in the head
        # leaf but is consumed by the LightningModule, not the head module.
        emb = head_config.embedding
        heads.append(
            EmbeddingHead(
                embedding_dim=emb.embedding_dim,
                num_fc_layers=emb.num_fc_layers,
                num_fc_units=emb.num_fc_units,
                pool=emb.pool,
                normalize=emb.normalize,
                output_stride=emb.output_stride,
            )
        )

    elif model_type == "bbox_centernet":
        # CenterNet-style detector: center heatmap + size + offset at one stride.
        # The leaves' target knobs (sigma, min_overlap, focal exponents) are read
        # by the dataset/LightningModule, not the heads.
        missing = [k for k in ("center", "size", "offset") if head_config[k] is None]
        if missing:
            message = (
                f"model_config.head_configs.bbox_centernet is missing {missing}; "
                "the center, size and offset heads are all required."
            )
            logger.error(message)
            raise ValueError(message)
        center = head_config.center
        heads.append(
            BBoxCenterHead(
                categories=list(center.categories or [DEFAULT_BBOX_CATEGORY]),
                output_stride=center.output_stride,
                loss_weight=center.loss_weight,
            )
        )
        heads.append(BBoxSizeHead(**head_config.size))
        heads.append(BBoxOffsetHead(**head_config.offset))

    elif model_type == "bbox_ssd":
        # SSD-style detector: one multi-scale anchor head over the pyramid levels.
        ssd = head_config.ssd
        heads.append(
            SSDHead(
                categories=list(ssd.categories or [DEFAULT_BBOX_CATEGORY]),
                anchors_per_location=len(ssd.anchor_scales) * len(ssd.aspect_ratios),
                output_strides=ssd_level_strides(ssd),
                head_channels=ssd.head_channels,
                head_convs=ssd.head_convs,
                loss_weight=ssd.loss_weight,
            )
        )

    else:
        message = f"{model_type} is not a defined model type. Please choose one of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`, `multi_class_topdown`, `bottomup_segmentation`, `centered_instance_segmentation`, `semantic_segmentation`, `embedding`, `bbox_centernet`, `bbox_ssd`."
        logger.error(message)
        raise Exception(message)

    return heads


class Model(nn.Module):
    """Model creates a model consisting of a backbone and head.

    Attributes:
        backbone_type: Backbone type. One of `unet`, `convnext`, `swint`, and
            `pretrained` (external HuggingFace backbone).
        backbone_config: An `DictConfig` configuration dictionary for the model backbone.
        head_configs: An `DictConfig` configuration dictionary for the model heads.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`, `multi_class_topdown`.

    A `bbox_detr` model has no `Head`s: its backbone is a DETR backbone
    (`sleap_nn.architectures.detr.build_detr_backbone`, returning tapped token maps)
    and everything after it is `self.detr` (a `BBoxDetrHead`), whose output dict
    (`pred_logits`, `pred_boxes`, `aux_outputs`, `enc_outputs`) `forward` returns.
    """

    def __init__(
        self,
        backbone_type: str,
        backbone_config: DictConfig,
        head_configs: DictConfig,
        model_type: str,
    ) -> None:
        """Initialize the backbone and head based on the backbone_config."""
        super().__init__()
        self.backbone_type = backbone_type
        self.backbone_config = backbone_config
        self.head_configs = head_configs
        self.model_type = model_type

        if model_type == "bbox_detr":
            self.heads = []
            self.head_layers = nn.ModuleList([])
            self.backbone = build_detr_backbone(backbone_type, backbone_config)
            self.detr = build_bbox_detr_head(
                head_configs.detr,
                in_channels=self.backbone.out_channels * self.backbone.num_taps,
            )
            return

        self.heads = get_head(model_type, self.head_configs)
        for head in self.heads:
            if isinstance(head, EmbeddingHead) and head.pool is None:
                # An unset pool means "the default for this backbone". Training
                # setup writes the resolved value into the saved config
                # (`check_output_strides`); this covers a model built from a config
                # that did not go through it.
                head.pool = default_embedding_pool(
                    embedding_encoder_is_frozen(
                        backbone_type, backbone_config, head_configs.embedding
                    )
                )

        output_strides = []
        for head_type in head_configs:
            head_config = head_configs[head_type]
            output_strides.append(head_config.output_stride)

        min_output_stride = min(output_strides)
        min_output_stride = min(min_output_stride, self.backbone_config.output_stride)

        self.backbone = get_backbone(
            self.backbone_type,
            backbone_config,
            spatial_heads=any(
                not isinstance(head, (ClassVectorsHead, EmbeddingHead))
                for head in self.heads
            ),
        )

        # Kernel size of the spatial heads' output conv: 1 unless the backbone's
        # decoder asks for more (a pretrained ViT's `vit_decoder: simple`).
        head_kernel_size = int(getattr(self.backbone, "head_kernel_size", 1))
        self.head_layers = nn.ModuleList([])
        for head in self.heads:
            if isinstance(head, MultiScaleHead):
                # One input per pyramid level. `decoder_stride_to_filters` covers
                # the decoder outputs and the bottleneck (the decoder's input).
                stride_to_filters = self.backbone.decoder_stride_to_filters
                absent = [s for s in head.output_strides if s not in stride_to_filters]
                if absent:
                    message = (
                        f"Head '{head.name}' reads strides {head.output_strides}, but "
                        f"backbone '{self.backbone_type}' produces strides "
                        f"{sorted(stride_to_filters)}; {absent} are missing. Lower "
                        f"the head's levels or raise the backbone's max_stride."
                    )
                    logger.error(message)
                    raise ValueError(message)
                self.head_layers.append(
                    head.make_head(
                        x_in=[int(stride_to_filters[s]) for s in head.output_strides]
                    )
                )
                continue
            if isinstance(head, (ClassVectorsHead, EmbeddingHead)):
                in_channels = int(self.backbone.middle_blocks[-1].filters)
                head_kwargs = {}
            else:
                stride_to_filters = self.backbone.decoder_stride_to_filters
                if head.output_stride not in stride_to_filters:
                    # An encoder-only backbone (e.g. an isotropic ViT like DINOv2
                    # under mode="auto") has no spatial decoder, so a spatial head
                    # has no feature map to bind to. Surface an actionable error
                    # instead of a bare KeyError.
                    produced = sorted(stride_to_filters) or "[] (encoder-only)"
                    message = (
                        f"Head '{head.name}' needs a spatial feature at "
                        f"output_stride {head.output_stride}, but backbone "
                        f"'{self.backbone_type}' produces strides {produced}. An "
                        f"encoder-only backbone supports only pooled heads "
                        f"(class-vectors); use a hierarchical backbone or "
                        f"mode='decoder' for spatial heads."
                    )
                    logger.error(message)
                    raise ValueError(message)
                in_channels = stride_to_filters[head.output_stride]
                head_kwargs = {"kernel_size": head_kernel_size}
            self.head_layers.append(head.make_head(x_in=in_channels, **head_kwargs))

    @classmethod
    def from_config(
        cls,
        backbone_type: str,
        backbone_config: DictConfig,
        head_configs: DictConfig,
        model_type: str,
    ) -> "Model":
        """Create the model from a config dictionary."""
        return cls(
            backbone_type=backbone_type,
            backbone_config=backbone_config,
            head_configs=head_configs,
            model_type=model_type,
        )

    def forward(
        self, x: torch.Tensor, return_backbone_outputs: bool = False
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], dict]]:
        """Forward pass through the model.

        Args:
            x: Input images ``(B, C, H, W)``.
            return_backbone_outputs: Also return the backbone's output dict (used by
                distillation to read intermediate features without hooks).

        Returns:
            ``{head name: output}``, or ``(that, backbone outputs)`` when
            ``return_backbone_outputs`` is set.
        """
        if x.shape[-3] != self.backbone_config.in_channels:
            if x.shape[-3] == 1:
                # convert grayscale to rgb
                x = x.repeat(1, 3, 1, 1)
            elif x.shape[-3] == 3:
                # convert rgb to grayscale
                x = F.rgb_to_grayscale(x, num_output_channels=1)

        backbone_outputs = self.backbone(x)
        if self.model_type == "bbox_detr":
            outputs = self.detr(backbone_outputs)
        else:
            outputs = self.forward_heads(backbone_outputs)
        if return_backbone_outputs:
            return outputs, backbone_outputs
        return outputs

    def forward_heads(
        self, backbone_outputs: dict, head_layers: Optional[nn.ModuleList] = None
    ) -> Dict[str, torch.Tensor]:
        """Run the heads on the backbone's output dict.

        Args:
            backbone_outputs: The backbone's output dict.
            head_layers: Head modules to run in place of ``self.head_layers`` (one
                per head, same layout), e.g. a frozen copy acting as a teacher.
        """
        if head_layers is None:
            head_layers = self.head_layers
        outputs = {}
        for head, head_layer in zip(self.heads, head_layers):
            if isinstance(head, MultiScaleHead):
                outputs[head.name] = head_layer(
                    [
                        _feature_at_stride(backbone_outputs, s)
                        for s in head.output_strides
                    ]
                )
            elif not len(backbone_outputs["outputs"]):
                outputs[head.name] = head_layer(backbone_outputs["middle_output"])
            else:
                if isinstance(head, (ClassVectorsHead, EmbeddingHead)):
                    backbone_out = backbone_outputs["intermediate_feat"]
                    outputs[head.name] = head_layer(backbone_out)
                else:
                    idx = backbone_outputs["strides"].index(head.output_stride)
                    outputs[head.name] = head_layer(backbone_outputs["outputs"][idx])

        return outputs


def _feature_at_stride(backbone_outputs: dict, stride: int) -> torch.Tensor:
    """The backbone feature map at ``stride``: a decoder output or the bottleneck.

    The decoder's outputs cover the strides below the bottleneck's; the bottleneck
    itself (``middle_output``, the decoder's input) is the deepest level.
    """
    strides = backbone_outputs["strides"]
    if stride in strides:
        return backbone_outputs["outputs"][strides.index(stride)]
    return backbone_outputs["middle_output"]
