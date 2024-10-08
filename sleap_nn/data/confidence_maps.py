"""Generate confidence maps."""

from typing import Dict, Iterator, Tuple

import torch
from torch.utils.data.datapipes.datapipe import IterDataPipe

from sleap_nn.data.utils import make_grid_vectors


def generate_confmaps(
    instance: torch.Tensor,
    img_hw: Tuple[int],
    sigma: float = 1.5,
    output_stride: int = 2,
) -> torch.Tensor:
    """Generate Confidence maps.

    Args:
        instance: Input keypoints. (n_samples, n_instances, n_nodes, 2) or
            (n_samples, n_nodes, 2).
        img_hw: Image size as tuple (height, width).
        sigma: The standard deviation of the Gaussian distribution that is used to
            generate confidence maps. Default: 1.5.
        output_stride: The relative stride to use when generating confidence maps.
            A larger stride will generate smaller confidence maps. Default: 2.

    Returns:
        Confidence maps for the input keypoints.
    """
    if instance.ndim != 3:
        instance = instance.view(instance.shape[0], -1, 2)
        # instances: (n_samples, n_nodes, 2)

    height, width = img_hw

    xv, yv = make_grid_vectors(height, width, output_stride)

    confidence_maps = make_confmaps(
        instance,
        xv,
        yv,
        sigma * output_stride,
    )  # (n_samples, n_nodes, height/ output_stride, width/ output_stride)

    return confidence_maps


def generate_multiconfmaps(
    instances: torch.Tensor,
    img_hw: Tuple[int],
    num_instances: int,
    sigma: float = 1.5,
    output_stride: int = 2,
    is_centroids: bool = False,
) -> torch.Tensor:
    """Generate multi-instance confidence maps.

    Args:
        instances: Input keypoints. (n_samples, n_instances, n_nodes, 2) or
            for centroids - (n_samples, n_instances, 2)
        img_hw: Image size as tuple (height, width).
        num_instances: Original number of instances in the frame.
        sigma: The standard deviation of the Gaussian distribution that is used to
            generate confidence maps. Default: 1.5.
        output_stride: The relative stride to use when generating confidence maps.
            A larger stride will generate smaller confidence maps. Default: 2.
        is_centroids: True if confidence maps should be generates for centroids else False.
            Default: False.

    Returns:
        Confidence maps for the input keypoints.
    """
    if is_centroids:
        points = instances[:, :num_instances, :].unsqueeze(dim=-2)
        # (n_samples, n_instances, 1, 2)
    else:
        points = instances[
            :, :num_instances, :, :
        ]  # (n_samples, n_instances, n_nodes, 2)

    height, width = img_hw

    xv, yv = make_grid_vectors(height, width, output_stride)

    confidence_maps = make_multi_confmaps(
        points,
        xv,
        yv,
        sigma * output_stride,
    )  # (n_samples, n_nodes, height/ output_stride, width/ output_stride).
    # If `is_centroids`, (n_samples, 1, height/ output_stride, width/ output_stride).

    return confidence_maps


def make_confmaps(
    points_batch: torch.Tensor, xv: torch.Tensor, yv: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Make confidence maps from a batch of points for multiple instances.

    Args:
        points_batch: A tensor of points of shape `(n_samples, n_nodes, 2)` and dtype `torch.float32` where
            the last axis corresponds to (x, y) pixel coordinates on the image for each instance.
            These can contain NaNs to indicate missing points.
        xv: Sampling grid vector for x-coordinates of shape `(grid_width,)` and dtype
            `torch.float32`. This can be generated by
            `sleap.nn.data.utils.make_grid_vectors`.
        yv: Sampling grid vector for y-coordinates of shape `(grid_height,)` and dtype
            `torch.float32`. This can be generated by
            `sleap.nn.data.utils.make_grid_vectors`.
        sigma: Standard deviation of the 2D Gaussian distribution sampled to generate
            confidence maps.

    Returns:
        Confidence maps as a tensor of shape `(n_samples, n_nodes, grid_height, grid_width)` of
        dtype `torch.float32`.
    """
    samples, n_nodes, _ = points_batch.shape

    x = torch.reshape(points_batch[:, :, 0], (samples, n_nodes, 1, 1))
    y = torch.reshape(points_batch[:, :, 1], (samples, n_nodes, 1, 1))

    xv_reshaped = torch.reshape(xv, (1, 1, 1, -1))
    yv_reshaped = torch.reshape(yv, (1, 1, -1, 1))

    cm = torch.exp(-((xv_reshaped - x) ** 2 + (yv_reshaped - y) ** 2) / (2 * sigma**2))

    # Replace NaNs with 0.
    cm = torch.nan_to_num(cm)

    return cm


def make_multi_confmaps(
    points_batch: torch.Tensor, xv: torch.Tensor, yv: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Make confidence maps for multiple instances through reduction.

    Args:
        points_batch: A tensor of shape `(n_samples, n_instances, n_nodes, 2)`
            and dtype `tf.float32` containing instance points where the last axis
            corresponds to (x, y) pixel coordinates on the image. This must be rank-3
            even if a single instance is present.
        xv: Sampling grid vector for x-coordinates of shape `(grid_width,)` and dtype
            `torch.float32`. This can be generated by
            `sleap.nn.data.utils.make_grid_vectors`.
        yv: Sampling grid vector for y-coordinates of shape `(grid_height,)` and dtype
            `torch.float32`. This can be generated by
            `sleap.nn.data.utils.make_grid_vectors`.
        sigma: Standard deviation of the 2D Gaussian distribution sampled to generate
            confidence maps.

    Returns:
        Confidence maps as a tensor of shape `(n_samples, n_nodes, grid_height, grid_width)` of
        dtype `torch.float32`.

        Each channel will contain the elementwise maximum of the confidence maps
        generated from all individual points for the associated node.

    """
    samples, n_inst, n_nodes, _ = points_batch.shape
    w, h = xv.shape[0], yv.shape[0]
    cms = torch.zeros((samples, n_nodes, h, w), dtype=torch.float32)
    points = points_batch.reshape(samples * n_inst, n_nodes, 2)
    for p in points:
        cm_instance = make_confmaps(p.unsqueeze(dim=0), xv, yv, sigma)
        cms = torch.maximum(cms, cm_instance)
    return cms


class MultiConfidenceMapGenerator(IterDataPipe):
    """IterDataPipe for generating multi-instance confidence maps.

    This IterDataPipe will generate confidence maps for examples from the input pipeline.
    Input examples must contain image of shape (n_samples, channels, crop_height, crop_width)
    and instance of shape (n_samples, n_instances, n_nodes, 2).

    Attributes:
        source_dp: The input `IterDataPipe` with examples that contain an instance and
            an image.
        sigma: The standard deviation of the Gaussian distribution that is used to
            generate confidence maps. This defines the spread in units of the input image's grid.
        output_stride: The relative stride to use when generating confidence maps.
            A larger stride will generate smaller confidence maps.
        centroids: If `True`, generate confidence maps for centroids rather than
            instance points.
        image_key: The name of the key where the image (n_samples, channels, crop_height, crop_width) is.
        instance_key: The name of the key where the instance points (n_samples, n_instances, n_nodes, 2) are.
    """

    def __init__(
        self,
        source_dp: IterDataPipe,
        sigma: int = 1.5,
        output_stride: int = 1,
        centroids: bool = True,
        image_key: str = "image",
        instance_key: str = "instances",
    ) -> None:
        """Initialize ConfidenceMapGenerator with input `IterDataPipe`, sigma, and output stride."""
        self.source_dp = source_dp
        self.sigma = sigma
        self.output_stride = output_stride
        self.centroids = centroids
        self.image_key = image_key
        self.instance_key = instance_key

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Generate confidence maps for each example."""
        for example in self.source_dp:
            if self.centroids:
                points = example["centroids"][
                    :, : example["num_instances"], :
                ].unsqueeze(
                    dim=-2
                )  # (n_samples, n_instances, 1, 2)
            else:
                points = example[
                    self.instance_key
                ]  # (n_samples, n_instances, n_nodes, 2)

            width = example[self.image_key].shape[-1]
            height = example[self.image_key].shape[-2]

            xv, yv = make_grid_vectors(height, width, self.output_stride)

            confidence_maps = make_multi_confmaps(
                points,
                xv,
                yv,
                self.sigma * self.output_stride,
            )

            if self.centroids:
                example["centroids_confidence_maps"] = (
                    confidence_maps  # (n_samples, 1, height, width)
                )
            else:
                example["confidence_maps"] = (
                    confidence_maps  # (n_samples, n_nodes, height, width)
                )
            yield example


class ConfidenceMapGenerator(IterDataPipe):
    """IterDataPipe for generating confidence maps.

    This IterDataPipe will generate confidence maps for examples from the input pipeline.
    Input examples must contain image of shape (n_samples, channels, crop_height, crop_width)
    and instance of shape (n_samples, n_instances, n_nodes, 2).

    Attributes:
        source_dp: The input `IterDataPipe` with examples that contain an instance and
            an image.
        sigma: The standard deviation of the Gaussian distribution that is used to
            generate confidence maps.
        output_stride: The relative stride to use when generating confidence maps.
            A larger stride will generate smaller confidence maps.
        image_key: The name of the key where the image (n_samples, channels, crop_height, crop_width) is.
        instance_key: The name of the key where the instance points (n_samples, n_instances, n_nodes, 2) are.
    """

    def __init__(
        self,
        source_dp: IterDataPipe,
        sigma: int = 1.5,
        output_stride: int = 1,
        image_key: str = "image",
        instance_key: str = "instances",
    ) -> None:
        """Initialize ConfidenceMapGenerator with input `IterDataPipe`, sigma, and output stride."""
        self.source_dp = source_dp
        self.sigma = sigma
        self.output_stride = output_stride
        self.image_key = image_key
        self.instance_key = instance_key

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Generate confidence maps for each example."""
        for example in self.source_dp:
            instance = example[self.instance_key]
            if self.instance_key == "instances":
                instance = instance.view(instance.shape[0], -1, 2)

            width = example[self.image_key].shape[-1]
            height = example[self.image_key].shape[-2]

            xv, yv = make_grid_vectors(height, width, self.output_stride)

            confidence_maps = make_confmaps(
                instance,
                xv,
                yv,
                self.sigma * self.output_stride,
            )  # (n_samples, n_nodes, height, width)

            example["confidence_maps"] = confidence_maps
            yield example
