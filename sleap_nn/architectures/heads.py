"""Model head definitions for defining model output types."""

from typing import List, Optional, Sequence, Text, Tuple

import torch
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from torch import nn
from loguru import logger
from sleap_nn.architectures.utils import get_act_fn
from collections import OrderedDict


class Head:
    """Base class for model output heads.

    Attributes:
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

    def __init__(self, output_stride: int = 1, loss_weight: float = 1.0) -> None:
        """Initialize the object with the specified attributes."""
        self.output_stride = output_stride
        self.loss_weight = loss_weight

    @property
    def name(self) -> str:
        """Name of the head."""
        return type(self).__name__

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        message = "Subclasses must implement this method."
        logger.error(message)
        raise NotImplementedError(message)

    @property
    def activation(self) -> str:
        """Return the activation function of the head output layer."""
        return "identity"

    @property
    def loss_function(self) -> str:
        """Return the name of the loss function to use for this head."""
        return "mse"

    def make_head(self, x_in: int) -> nn.Sequential:
        """Make head output tensor from input feature tensor.

        Args:
            x_in: An int input for the input channels.

        Returns:
            A `nn.Sequential` with the correct shape for the head.
        """
        module_dict = OrderedDict()
        module_dict[self.name] = nn.Sequential(
            nn.Conv2d(
                in_channels=x_in,
                out_channels=self.channels,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            get_act_fn(self.activation),
        )

        return nn.Sequential(module_dict)


class SingleInstanceConfmapsHead(Head):
    """Head for specifying single instance confidence maps.

    Attributes:
        part_names: List of strings specifying the part names associated with channels.
        sigma: Spread of the confidence maps.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

    def __init__(
        self,
        part_names: List[Text],
        sigma: float = 5.0,
        output_stride: int = 1,
        loss_weight: float = 1.0,
    ) -> None:
        """Initialize the object with the specified attributes."""
        super().__init__(output_stride, loss_weight)
        self.part_names = part_names
        self.sigma = sigma

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return len(self.part_names)

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        part_names: Optional[List[Text]] = None,
    ) -> "SingleInstanceConfmapsHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `DictConfig` instance specifying the head
                parameters.
            part_names: Text name of the body parts (nodes) that the head will be
                configured to produce. The number of parts determines the number of
                channels in the output. This must be provided if the `part_names`
                attribute of the configuration is not set.

        Returns:
            The instantiated head with the specified configuration options.
        """
        if config.part_names is not None:
            part_names = config.part_names
        elif part_names is None:
            message = "Required attribute 'part_names' is missing in the configuration or in `from_config` input."
            logger.error(message)
            raise ValueError(message)
        return cls(
            part_names=part_names,
            sigma=config.sigma,
            output_stride=config.output_stride,
            loss_weight=config.loss_weight,
        )


class CentroidConfmapsHead(Head):
    """Head for specifying instance centroid confidence maps.

    Attributes:
        anchor_part: Name of the part to use as an anchor node. If not specified, the
            bounding box centroid will be used.
        sigma: Spread of the confidence maps.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

    def __init__(
        self,
        anchor_part: Optional[Text] = None,
        sigma: float = 5.0,
        output_stride: int = 1,
        loss_weight: float = 1.0,
    ) -> None:
        """Initialize the object with the specified attributes."""
        super().__init__(output_stride, loss_weight)
        self.anchor_part = anchor_part
        self.sigma = sigma

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return 1

    @classmethod
    def from_config(cls, config: DictConfig) -> "CentroidConfmapsHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `DictConfig` instance specifying the head parameters.

        Returns:
            The instantiated head with the specified configuration options.
        """
        return cls(
            anchor_part=config.anchor_part,
            sigma=config.sigma,
            output_stride=config.output_stride,
            loss_weight=config.loss_weight,
        )


class CenteredInstanceConfmapsHead(Head):
    """Head for specifying centered instance confidence maps.

    Attributes:
        part_names: List of strings specifying the part names associated with channels.
        anchor_part: Name of the part to use as an anchor node. If not specified, the
            bounding box centroid will be used.
        sigma: Spread of the confidence maps.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

    def __init__(
        self,
        part_names: List[Text],
        anchor_part: Optional[Text] = None,
        sigma: float = 5.0,
        output_stride: int = 1,
        loss_weight: float = 1.0,
    ) -> None:
        """Initialize the object with the specified attributes."""
        super().__init__(output_stride, loss_weight)
        self.part_names = part_names
        self.anchor_part = anchor_part
        self.sigma = sigma

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return len(self.part_names)

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        part_names: Optional[List[Text]] = None,
    ) -> "CenteredInstanceConfmapsHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `DictConfig` instance specifying the head
                parameters.
            part_names: Text name of the body parts (nodes) that the head will be
                configured to produce. The number of parts determines the number of
                channels in the output. This must be provided if the `part_names`
                attribute of the configuration is not set.

        Returns:
            The instantiated head with the specified configuration options.
        """
        if config.part_names is not None:
            part_names = config.part_names
        elif part_names is None:
            message = "Required attribute 'part_names' is missing in the configuration or in `from_config` input."
            logger.error(message)
            raise ValueError(message)
        return cls(
            part_names=part_names,
            anchor_part=config.anchor_part,
            sigma=config.sigma,
            output_stride=config.output_stride,
            loss_weight=config.loss_weight,
        )


class MultiInstanceConfmapsHead(Head):
    """Head for specifying multi-instance confidence maps.

    Attributes:
        part_names: List of strings specifying the part names associated with channels.
        sigma: Spread of the confidence maps.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

    def __init__(
        self,
        part_names: List[Text],
        sigma: float = 5.0,
        output_stride: int = 1,
        loss_weight: float = 1.0,
    ) -> None:
        """Initialize the object with the specified attributes."""
        super().__init__(output_stride, loss_weight)
        self.part_names = part_names
        self.sigma = sigma

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return len(self.part_names)

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        part_names: Optional[List[Text]] = None,
    ) -> "MultiInstanceConfmapsHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `DictConfig` instance specifying the head
                parameters.
            part_names: Text name of the body parts (nodes) that the head will be
                configured to produce. The number of parts determines the number of
                channels in the output. This must be provided if the `part_names`
                attribute of the configuration is not set.

        Returns:
            The instantiated head with the specified configuration options.
        """
        if config.part_names is not None:
            part_names = config.part_names
        elif part_names is None:
            message = "Required attribute 'part_names' is missing in the configuration or in `from_config` input."
            logger.error(message)
            raise ValueError(message)
        return cls(
            part_names=part_names,
            sigma=config.sigma,
            output_stride=config.output_stride,
            loss_weight=config.loss_weight,
        )


class PartAffinityFieldsHead(Head):
    """Head for specifying multi-instance part affinity fields.

    Attributes:
        edges: List of tuples of `(source, destination)` node names.
        sigma: Spread of the part affinity fields.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

    def __init__(
        self,
        edges: Sequence[Tuple[Text, Text]],
        sigma: float = 5.0,
        output_stride: int = 1,
        loss_weight: float = 1.0,
    ) -> None:
        """Initialize the object with the specified attributes."""
        super().__init__(output_stride, loss_weight)
        self.edges = edges
        self.sigma = sigma

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return int(len(self.edges) * 2)

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        edges: Optional[Sequence[Tuple[Text, Text]]] = None,
    ) -> "PartAffinityFieldsHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `DictConfig` instance specifying the head
                parameters.
            edges: List of 2-tuples of the form `(source_node, destination_node)` that
                define pairs of text names of the directed edges of the graph. This must
                be set if the `edges` attribute of the configuration is not set.

        Returns:
            The instantiated head with the specified configuration options.
        """
        if config.edges is not None:
            edges = config.edges
        return cls(
            edges=edges,
            sigma=config.sigma,
            output_stride=config.output_stride,
            loss_weight=config.loss_weight,
        )


class ClassMapsHead(Head):
    """Head for specifying class identity maps.

    Attributes:
        classes: List of string names of the classes.
        sigma: Spread of the class maps around each node.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

    def __init__(
        self,
        classes: List[Text],
        sigma: float = 5.0,
        output_stride: int = 1,
        loss_weight: float = 1.0,
    ) -> None:
        """Initialize the object with the specified attributes."""
        super().__init__(output_stride, loss_weight)
        self.classes = classes
        self.sigma = sigma

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return len(self.classes)

    @property
    def activation(self) -> str:
        """Return the activation function of the head output layer."""
        return "sigmoid"

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        classes: Optional[List[Text]] = None,
    ) -> "ClassMapsHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `DictConfig` instance specifying the head parameters.
            classes: List of string names of the classes that this head will predict.
                This must be set if the `classes` attribute of the configuration is not
                set.

        Returns:
            The instantiated head with the specified configuration options.
        """
        if config.classes is not None:
            classes = config.classes
        return cls(
            classes=classes,
            sigma=config.sigma,
            output_stride=config.output_stride,
            loss_weight=config.loss_weight,
        )


class ClassVectorsHead(Head):
    """Head for specifying classification heads.

    Attributes:
        classes: List of string names of the classes.
        num_fc_layers: Number of fully connected layers after flattening input features.
        num_fc_units: Number of units (dimensions) in fully connected layers prior to
            classification output.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

    def __init__(
        self,
        classes: List[Text],
        num_fc_layers: int = 1,
        num_fc_units: int = 64,
        global_pool: bool = True,
        output_stride: int = 1,
        loss_weight: float = 1.0,
    ) -> None:
        """Initialize the object with the specified attributes."""
        super().__init__(output_stride, loss_weight)
        self.classes = classes
        self.num_fc_layers = num_fc_layers
        self.num_fc_units = num_fc_units
        self.global_pool = global_pool

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return len(self.classes)

    @property
    def activation(self) -> str:
        """Return the activation function of the head output layer."""
        return "softmax"

    @property
    def loss_function(self) -> str:
        """Return the name of the loss function to use for this head."""
        return "categorical_crossentropy"

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        classes: Optional[List[Text]] = None,
    ) -> "ClassVectorsHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `DictConfig` instance specifying the head parameters.
            classes: List of string names of the classes that this head will predict.
                This must be set if the `classes` attribute of the configuration is not
                set.

        Returns:
            The instantiated head with the specified configuration options.
        """
        if config.classes is not None:
            classes = config.classes
        return cls(
            classes=classes,
            num_fc_layers=config.num_fc_layers,
            num_fc_units=config.num_fc_units,
            global_pool=config.global_pool,
            output_stride=config.output_stride,
            loss_weight=config.loss_weight,
        )

    def make_head(self, x_in: int) -> nn.Sequential:
        """Make head output tensor from input feature tensor.

        Args:
            x_in: An int for the input shape after applying AdaptiveMaxPool2d on dim=1, assuming inputs of shape (B, C, H, W).

        Returns:
            A `nn.Sequential` with the correct shape for the head.
        """
        from collections import OrderedDict

        module_dict = OrderedDict()

        if self.global_pool:
            module_dict[f"pre_classification_global_pool"] = nn.AdaptiveMaxPool2d(1)

        module_dict[f"pre_classification_flatten"] = nn.Flatten(start_dim=1)

        for i in range(self.num_fc_layers):
            if i == 0:
                module_dict[f"pre_classification{i}_fc"] = nn.Linear(
                    x_in, self.num_fc_units
                )
            else:
                module_dict[f"pre_classification{i}_fc"] = nn.Linear(
                    self.num_fc_units, self.num_fc_units
                )
            module_dict[f"pre_classification{i}_relu"] = get_act_fn("relu")

        module_dict[f"ClassVectorsHead"] = nn.Linear(self.num_fc_units, self.channels)
        module_dict[f"softmax"] = get_act_fn("softmax")

        return nn.Sequential(module_dict)


class GeM(nn.Module):
    """Generalized-mean pooling: ``(mean(x.clamp(min=eps)^p))^(1/p)`` over HxW.

    The exponent ``p`` is learnable (init 3.0). The ``clamp(min=eps)`` BEFORE the
    fractional power guards against NaNs (a fractional power of a negative/zero base).
    Returns a flattened ``[B, C]`` tensor.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        """Initialize the pooling layer.

        Args:
            p: Initial generalized-mean exponent (``p=1`` averages, larger ``p``
                approaches max-pooling).
            eps: Small floor applied to the activation before the power to avoid a
                fractional power of a non-positive value.
            learnable: If ``True``, ``p`` is a trainable parameter; otherwise it is a
                fixed buffer.
        """
        super().__init__()
        if learnable:
            self.p = nn.Parameter(torch.tensor(float(p)))
        else:
            self.register_buffer("p", torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool ``[B, C, H, W]`` to ``[B, C]`` by the generalized mean over ``HxW``."""
        # Clamp the (learnable) exponent to a sane floor: p -> 0 makes ``1/p`` explode
        # and a negative p inverts the mean — either can yield inf/NaN embeddings that
        # corrupt the whole batch's contrastive loss. The activation eps-clamp guards
        # the base; this guards the exponent.
        p = self.p.clamp(min=1.0)
        xp = x.clamp(min=self.eps).pow(p)
        return F.adaptive_avg_pool2d(xp, 1).pow(1.0 / p).flatten(1)


class L2Norm(nn.Module):
    """L2-normalize along ``dim`` (so embeddings live on the unit hypersphere)."""

    def __init__(self, dim: int = 1):
        """Initialize the layer.

        Args:
            dim: Dimension along which to L2-normalize.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalize ``x`` along ``dim``."""
        return F.normalize(x, dim=self.dim)


class EmbeddingHead(Head):
    """Head for crop -> embedding-vector (re-ID) models.

    Mirrors ``ClassVectorsHead`` (a pooled, non-spatial head): ``[pool] -> Flatten ->
    num_fc_layers x (Linear+ReLU) -> Linear(embedding_dim) -> [L2Norm]``. The pooled
    feature comes from the backbone's ``middle_output`` (lone head, empty decoder).

    Attributes:
        embedding_dim: Output embedding dimensionality.
        num_fc_layers: Number of FC layers before the embedding output.
        num_fc_units: Units in the pre-embedding FC layers.
        pool: Pooling over the encoder feature map: ``gem`` | ``max`` | ``avg``.
        normalize: L2-normalize the output embedding.
        output_stride: Should equal the backbone max_stride (so the decoder is empty
            and the head taps ``middle_output``).
        loss_weight: Weight of the loss term for this head.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_fc_layers: int = 1,
        num_fc_units: int = 256,
        pool: str = "gem",
        normalize: bool = True,
        output_stride: int = 1,
        loss_weight: float = 1.0,
    ) -> None:
        """Initialize the object with the specified attributes."""
        super().__init__(output_stride, loss_weight)
        self.embedding_dim = embedding_dim
        self.num_fc_layers = num_fc_layers
        self.num_fc_units = num_fc_units
        self.pool = pool
        self.normalize = normalize

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return self.embedding_dim

    @property
    def activation(self) -> str:
        """Return the activation function of the head output layer."""
        return "identity"

    @property
    def loss_function(self) -> str:
        """Return the loss-function name (informational).

        The contrastive loss is batch-level and is driven by the embedding
        LightningModule's ``objective``, not by this string.
        """
        return "supcon"

    @classmethod
    def from_config(cls, config: DictConfig) -> "EmbeddingHead":
        """Create this head from a head-leaf configuration."""
        return cls(
            embedding_dim=config.embedding_dim,
            num_fc_layers=config.num_fc_layers,
            num_fc_units=config.num_fc_units,
            pool=config.pool,
            normalize=config.normalize,
            output_stride=config.output_stride,
            loss_weight=config.loss_weight,
        )

    def make_head(self, x_in: int) -> nn.Sequential:
        """Make the head output module from the pooled-feature input channels.

        Args:
            x_in: Number of channels of the encoder feature map (its channel dim).

        Returns:
            An ``nn.Sequential`` mapping ``[B, x_in, H, W] -> [B, embedding_dim]``.
        """
        module_dict = OrderedDict()
        if self.pool == "gem":
            module_dict["pre_embedding_pool"] = GeM()
        elif self.pool == "max":
            module_dict["pre_embedding_pool"] = nn.AdaptiveMaxPool2d(1)
        elif self.pool == "avg":
            module_dict["pre_embedding_pool"] = nn.AdaptiveAvgPool2d(1)
        else:
            message = f"Unknown pool '{self.pool}'; choose one of gem|max|avg."
            logger.error(message)
            raise ValueError(message)

        module_dict["pre_embedding_flatten"] = nn.Flatten(start_dim=1)

        d_in = x_in
        for i in range(self.num_fc_layers):
            module_dict[f"pre_embedding{i}_fc"] = nn.Linear(d_in, self.num_fc_units)
            module_dict[f"pre_embedding{i}_relu"] = get_act_fn("relu")
            d_in = self.num_fc_units

        module_dict["EmbeddingHead"] = nn.Linear(d_in, self.embedding_dim)
        if self.normalize:
            module_dict["l2norm"] = L2Norm(dim=1)

        return nn.Sequential(module_dict)


class OffsetRefinementHead(Head):
    """Head for specifying offset refinement maps.

    Attributes:
        part_names: List of strings specifying the part names associated with channels.
        sigma_threshold: Threshold of confidence map values to use for defining the
            boundary of the offset maps.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

    def __init__(
        self,
        part_names: List[Text],
        sigma_threshold: float = 0.2,
        output_stride: int = 1,
        loss_weight: float = 1.0,
    ) -> None:
        """Initialize the object with the specified attributes."""
        super().__init__(output_stride, loss_weight)
        self.part_names = part_names
        self.sigma_threshold = sigma_threshold

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return int(len(self.part_names) * 2)

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        part_names: Optional[List[Text]] = None,
        sigma_threshold: float = 0.2,
        loss_weight: float = 1.0,
    ) -> "OffsetRefinementHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `DictConfig` instance specifying the head parameters.
            part_names: Text name of the body parts (nodes) that the head will be
                configured to produce. The number of parts determines the number of
                channels in the output. This must be provided if the `part_names`
                attribute of the configuration is not set.
            sigma_threshold: Minimum confidence map value below which offsets will be
                replaced with zeros.
            loss_weight: Weight of the loss associated with this head.

        Returns:
            The instantiated head with the specified configuration options.
        """
        if hasattr(config, "part_names"):
            if config.part_names is not None:
                part_names = config.part_names
        elif hasattr(config, "anchor_part"):
            part_names = [config.anchor_part]
        else:
            message = "Required attribute 'part_names' is missing in the configuration."
            logger.error(message)
            raise ValueError(message)
        return cls(
            part_names=part_names,
            output_stride=config.output_stride,
            sigma_threshold=sigma_threshold,
            loss_weight=loss_weight,
        )


class SegmentationHead(Head):
    """Head for predicting binary foreground segmentation masks.

    Outputs a single-channel map with sigmoid activation representing the
    probability that each pixel belongs to any instance (foreground).

    Attributes:
        output_stride: Stride of the output head tensor.
        loss_weight: Weight of the loss term for this head during optimization.
    """

    def __init__(
        self,
        output_stride: int = 2,
        loss_weight: float = 1.0,
    ) -> None:
        """Initialize the object with the specified attributes."""
        super().__init__(output_stride, loss_weight)

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return 1

    @property
    def activation(self) -> str:
        """Return the activation function of the head output layer."""
        return "identity"

    @property
    def loss_function(self) -> str:
        """Return the name of the loss function to use for this head."""
        return "bce_dice"


class InstanceCenterHead(Head):
    """Head for predicting instance center heatmaps.

    Outputs a single-channel Gaussian heatmap with peaks at each instance's
    mask centroid. Similar to CentroidConfmapsHead but for mask-derived centers.

    Attributes:
        sigma: Standard deviation of the Gaussian in pixels.
        output_stride: Stride of the output head tensor.
        loss_weight: Weight of the loss term for this head during optimization.
    """

    def __init__(
        self,
        sigma: float = 4.0,
        output_stride: int = 2,
        loss_weight: float = 1.0,
    ) -> None:
        """Initialize the object with the specified attributes."""
        super().__init__(output_stride, loss_weight)
        self.sigma = sigma

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return 1


class CenterOffsetHead(Head):
    """Head for predicting per-pixel offset vectors to instance centers.

    Outputs a 2-channel map where each pixel's value is (dx, dy) pointing
    from the pixel to its instance's center. Only meaningful on foreground pixels.

    Attributes:
        output_stride: Stride of the output head tensor.
        loss_weight: Weight of the loss term for this head during optimization.
    """

    def __init__(
        self,
        output_stride: int = 2,
        loss_weight: float = 0.1,
    ) -> None:
        """Initialize the object with the specified attributes."""
        super().__init__(output_stride, loss_weight)

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return 2

    @property
    def loss_function(self) -> str:
        """Return the name of the loss function to use for this head."""
        return "smooth_l1"
