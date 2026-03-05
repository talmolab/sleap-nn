"""This module defines the main SLEAP model class for defining a trainable model.

This is a higher level wrapper around `nn.Module` that holds all the configuration
parameters required to construct the actual model. This allows for easy querying of the
model configuration without actually instantiating the model itself.
"""

from typing import Dict, List, Optional
from collections import defaultdict
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
)
from sleap_nn.architectures.unet import UNet
from sleap_nn.architectures.convnext import ConvNextWrapper
from sleap_nn.architectures.swint import SwinTWrapper
import torchvision.transforms.v2.functional as F


def get_backbone(backbone: str, backbone_config: DictConfig) -> nn.Module:
    """Get a backbone model `nn.Module` based on the provided name.

    This function returns an instance of a PyTorch `nn.Module`
    corresponding to the given backbone name.

    Args:
        backbone (str): Name of the backbone. Supported values are 'unet'.
        backbone_config (DictConfig): A config for the backbone.

    Returns:
        nn.Module: An instance of the requested backbone model.

    Raises:
        KeyError: If the provided backbone name is not one of the supported values.
    """
    backbones = {"unet": UNet, "convnext": ConvNextWrapper, "swint": SwinTWrapper}

    if backbone not in backbones:
        message = f"Unsupported backbone: {backbone}. Supported backbones are: {', '.join(backbones.keys())}"
        logger.error(message)
        raise KeyError(message)

    backbone = backbones[backbone].from_config(backbone_config)

    return backbone


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

    else:
        message = f"{model_type} is not a defined model type. Please choose one of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`, `multi_class_topdown`."
        logger.error(message)
        raise Exception(message)

    return heads


class Model(nn.Module):
    """Model creates a model consisting of a backbone and head.

    Attributes:
        backbone_type: Backbone type. One of `unet`, `convnext` and `swint`.
        backbone_config: An `DictConfig` configuration dictionary for the model backbone.
        head_configs: An `DictConfig` configuration dictionary for the model heads.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`, `multi_class_topdown`.
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

        self.heads = get_head(model_type, self.head_configs)

        output_strides = []
        for head_type in head_configs:
            head_config = head_configs[head_type]
            output_strides.append(head_config.output_stride)

        min_output_stride = min(output_strides)
        min_output_stride = min(min_output_stride, self.backbone_config.output_stride)

        self.backbone = get_backbone(
            self.backbone_type,
            backbone_config,
        )

        self.head_layers = nn.ModuleList([])
        for head in self.heads:
            if isinstance(head, ClassVectorsHead):
                in_channels = int(self.backbone.middle_blocks[-1].filters)
            else:
                in_channels = self.backbone.decoder_stride_to_filters[
                    head.output_stride
                ]
            self.head_layers.append(head.make_head(x_in=in_channels))

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        if x.shape[-3] != self.backbone_config.in_channels:
            if x.shape[-3] == 1:
                # convert grayscale to rgb
                x = x.repeat(1, 3, 1, 1)
            elif x.shape[-3] == 3:
                # convert rgb to grayscale
                x = F.rgb_to_grayscale(x, num_output_channels=1)

        backbone_outputs = self.backbone(x)

        outputs = {}
        for head, head_layer in zip(self.heads, self.head_layers):
            if not len(backbone_outputs["outputs"]):
                outputs[head.name] = head_layer(backbone_outputs["middle_output"])
            else:
                if isinstance(head, ClassVectorsHead):
                    backbone_out = backbone_outputs["intermediate_feat"]
                    outputs[head.name] = head_layer(backbone_out)
                else:
                    idx = backbone_outputs["strides"].index(head.output_stride)
                    outputs[head.name] = head_layer(backbone_outputs["outputs"][idx])

        return outputs


class MultiHeadModel(nn.Module):
    """MultiHeadModel creates a model consisting of a shared backbone and multiple heads.

    This model architecture enables training with multiple datasets, where each dataset
    can have a different skeleton configuration. The backbone is shared across all
    datasets, while each dataset has its own dedicated head.

    Attributes:
        backbone_type: Backbone type. One of `unet`, `convnext` and `swint`.
        backbone_config: An `DictConfig` configuration dictionary for the model backbone.
        head_configs: An `DictConfig` configuration dictionary for the model heads.
            This should contain configurations for multiple heads indexed by dataset number.
        model_type: Type of the model. One of `single_instance`, `centered_instance`,
            `centroid`, `bottomup`, `multi_class_bottomup`, `multi_class_topdown`.
    """

    def __init__(
        self,
        backbone_type: str,
        backbone_config: DictConfig,
        head_configs: DictConfig,
        model_type: str,
    ) -> None:
        """Initialize the backbone and heads based on the configs.

        Args:
            backbone_type: Type of backbone network to use.
            backbone_config: Configuration for the backbone.
            head_configs: Configuration for the heads, with per-dataset configs.
            model_type: Type of model (e.g., 'centered_instance', 'centroid').
        """
        super().__init__()
        self.backbone_type = backbone_type
        self.backbone_config = backbone_config
        self.head_configs = head_configs
        self.model_type = model_type
        self.heads = []

        # Create heads based on model type
        if self.model_type == "centered_instance":
            for d_num in self.head_configs.confmaps:
                self.heads.append(
                    CenteredInstanceConfmapsHead(**self.head_configs.confmaps[d_num])
                )
        elif self.model_type == "centroid":
            for d_num in self.head_configs.confmaps:
                self.heads.append(
                    CentroidConfmapsHead(**self.head_configs.confmaps[d_num])
                )
        elif self.model_type == "single_instance":
            for d_num in self.head_configs.confmaps:
                self.heads.append(
                    SingleInstanceConfmapsHead(**self.head_configs.confmaps[d_num])
                )
        elif self.model_type == "bottomup":
            # For bottomup, we need confmaps and PAFs for each dataset
            for d_num in self.head_configs.confmaps:
                self.heads.append(
                    MultiInstanceConfmapsHead(**self.head_configs.confmaps[d_num])
                )
            for d_num in self.head_configs.pafs:
                self.heads.append(
                    PartAffinityFieldsHead(**self.head_configs.pafs[d_num])
                )
        else:
            message = f"{model_type} is not supported for multi-head models. Supported types: `single_instance`, `centered_instance`, `centroid`, `bottomup`."
            logger.error(message)
            raise ValueError(message)

        # Compute output strides from all heads
        output_strides = []
        for head_type in head_configs:
            head_config = head_configs[head_type]
            if isinstance(head_config, DictConfig):
                for cfg_key in head_config:
                    if hasattr(head_config[cfg_key], "output_stride"):
                        output_strides.append(head_config[cfg_key].output_stride)

        if output_strides:
            min_output_stride = min(output_strides)
            min_output_stride = min(min_output_stride, self.backbone_config.output_stride)

        # Initialize backbone
        self.backbone = get_backbone(
            self.backbone_type,
            backbone_config,
        )

        # Initialize head layers
        self.head_layers = nn.ModuleList([])
        for head in self.heads:
            if isinstance(head, ClassVectorsHead):
                in_channels = int(self.backbone.middle_blocks[-1].filters)
            else:
                in_channels = self.backbone.decoder_stride_to_filters[
                    head.output_stride
                ]
            self.head_layers.append(head.make_head(x_in=in_channels))

    @classmethod
    def from_config(
        cls,
        backbone_type: str,
        backbone_config: DictConfig,
        head_configs: DictConfig,
        model_type: str,
    ) -> "MultiHeadModel":
        """Create the model from a config dictionary.

        Args:
            backbone_type: Type of backbone network.
            backbone_config: Configuration for the backbone.
            head_configs: Configuration for the heads.
            model_type: Type of model.

        Returns:
            MultiHeadModel: An instance of the multi-head model.
        """
        return cls(
            backbone_type=backbone_type,
            backbone_config=backbone_config,
            head_configs=head_configs,
            model_type=model_type,
        )

    def forward(
        self,
        x: torch.Tensor,
        include_backbone_features: bool = False,
        backbone_outputs: Optional[str] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """Forward pass through the model.

        Args:
            x: Input image tensor of shape (batch, channels, height, width).
            include_backbone_features: If True, include backbone features in output.
            backbone_outputs: Which backbone outputs to return. One of "last" or "all".
                Only used if include_backbone_features is True.

        Returns:
            A dictionary with keys as head names and values as lists of output tensors,
            one per dataset/head. For example:
            {
                "CenteredInstanceConfmapsHead": [tensor_head_0, tensor_head_1, ...],
                "backbone_features": tensor (if include_backbone_features=True),
                "backbone_features_strides": stride (if include_backbone_features=True)
            }
        """
        # Handle channel conversion if needed
        if x.shape[-3] != self.backbone_config.in_channels:
            if x.shape[-3] == 1:
                # Convert grayscale to RGB
                x = x.repeat(1, 3, 1, 1)
            elif x.shape[-3] == 3:
                # Convert RGB to grayscale
                x = F.rgb_to_grayscale(x, num_output_channels=1)

        # Forward through backbone
        backbone_outs = self.backbone(x)

        # Initialize outputs dict with defaultdict for head outputs
        outputs = defaultdict(list)

        # Add backbone features if requested
        if include_backbone_features:
            if backbone_outputs is None:
                backbone_outputs = "last"
            if backbone_outputs == "last":
                outputs["backbone_features"] = backbone_outs["outputs"][-1]
                outputs["backbone_features_strides"] = backbone_outs["strides"][-1]
            elif backbone_outputs == "all":
                outputs["backbone_features"] = backbone_outs["outputs"]
                outputs["backbone_features_strides"] = backbone_outs["strides"]

        # Forward through each head
        for head, head_layer in zip(self.heads, self.head_layers):
            if not len(backbone_outs["outputs"]):
                outputs[head.name].append(head_layer(backbone_outs["middle_output"]))
            else:
                if isinstance(head, ClassVectorsHead):
                    backbone_out = backbone_outs["intermediate_feat"]
                    outputs[head.name].append(head_layer(backbone_out))
                else:
                    idx = backbone_outs["strides"].index(head.output_stride)
                    outputs[head.name].append(
                        head_layer(backbone_outs["outputs"][idx])
                    ) # eg: outputs = {"CenteredInstanceConfmapsHead" : [output_head_0, output_head_1, output_head_2, ...]}

        return dict(outputs)
