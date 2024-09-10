"""Transformers for generating edge confidence maps and part affinity fields."""

from typing import Tuple, Optional, List, Text, Iterator, Dict
import torch
import attrs
from torch.utils.data.datapipes.datapipe import IterDataPipe

from sleap_nn.data.utils import (
    expand_to_rank,
    make_grid_vectors,
    ensure_list,
    gaussian_pdf,
)


def distance_to_edge(
    points: torch.Tensor, edge_source: torch.Tensor, edge_destination: torch.Tensor
) -> torch.Tensor:
    """Compute pairwise distance between points and undirected edges.

    Args:
        points: Tensor of dtype torch.float32 of shape (d_0, ..., d_n, 2) where the last
            axis corresponds to x- and y-coordinates. Distances will be broadcast across
            all point dimensions.
        edge_source: Tensor of dtype torch.float32 of shape (n_edges, 2) where the last
            axis corresponds to x- and y-coordinates of the source points of each edge.
        edge_destination: Tensor of dtype torch.float32 of shape (n_edges, 2) where the
            last axis corresponds to x- and y-coordinates of the source points of each
            edge.

    Returns:
        A tensor of dtype torch.float32 of shape (d_0, ..., d_n, n_edges) where the first
        axes correspond to the initial dimensions of `points`, and the last indicates
        the distance of each point to each edge.
    """
    # Ensure all points are at least rank 2.
    points = expand_to_rank(points, 2)
    edge_source = expand_to_rank(edge_source, 2)
    edge_destination = expand_to_rank(edge_destination, 2)

    # Compute number of point dimensions.
    n_pt_dims = points.dim() - 1

    # Direction vector.
    direction_vector = edge_destination - edge_source  # (n_edges, 2)

    # Edge length.
    edge_length = torch.maximum(
        direction_vector.square().sum(dim=1), torch.tensor(1.0)
    )  # (n_edges,)

    # Adjust query points relative to edge source point.
    source_relative_points = torch.unsqueeze(points, dim=-2) - expand_to_rank(
        edge_source, n_pt_dims + 2
    )  # (..., n_edges, 2)

    # Project points to edge line.
    line_projections = torch.sum(
        source_relative_points * expand_to_rank(direction_vector, n_pt_dims + 2), dim=3
    ) / expand_to_rank(
        edge_length, n_pt_dims + 1
    )  # (..., n_edges)

    # Crop to line segment.
    line_projections = torch.clamp(line_projections, min=0, max=1)

    # Compute distance from each point to the edge.
    distances = torch.sum(
        torch.square(
            (
                line_projections.unsqueeze(-1)
                * expand_to_rank(direction_vector, n_pt_dims + 2)
            )
            - source_relative_points
        ),
        dim=-1,
    )  # (..., n_edges)

    return distances


def make_edge_maps(
    xv: torch.Tensor,
    yv: torch.Tensor,
    edge_source: torch.Tensor,
    edge_destination: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Generate confidence maps for a set of undirected edges.

    Args:
        xv: Sampling grid vector for x-coordinates of shape (grid_width,) and dtype
            torch.float32. This can be generated by
            `sleap_nn.data.utils.make_grid_vectors`.
        yv: Sampling grid vector for y-coordinates of shape (grid_height,) and dtype
            torch.float32. This can be generated by
            `sleap_nn.data.utils.make_grid_vectors`.
        edge_source: Tensor of dtype torch.float32 of shape (n_edges, 2) where the last
            axis corresponds to x- and y-coordinates of the source points of each edge.
        edge_destination: Tensor of dtype torch.float32 of shape (n_edges, 2) where the
            last axis corresponds to x- and y-coordinates of the destination points of
            each edge.
        sigma: Standard deviation of the 2D Gaussian distribution sampled to generate
            confidence maps.

    Returns:
        A set of confidence maps corresponding to the probability of each point on a
        sampling grid being on each edge. These will be in a tensor of shape
        (grid_height, grid_width, n_edges) of dtype torch.float32.
    """
    yy, xx = torch.meshgrid(yv, xv, indexing="ij")
    sampling_grid = torch.stack((xx, yy), dim=-1)  # (height, width, 2)

    distances = distance_to_edge(
        sampling_grid, edge_source=edge_source, edge_destination=edge_destination
    )
    edge_maps = gaussian_pdf(distances, sigma=sigma)
    return edge_maps


def make_pafs(
    xv: torch.Tensor,
    yv: torch.Tensor,
    edge_source: torch.Tensor,
    edge_destination: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Generate part affinity fields for a set of directed edges.

    Args:
        xv: Sampling grid vector for x-coordinates of shape (grid_width,) and dtype
            torch.float32. This can be generated by
            `sleap_nn.data.utils.make_grid_vectors`.
        yv: Sampling grid vector for y-coordinates of shape (grid_height,) and dtype
            torch.float32. This can be generated by
            `sleap_nn.data.utils.make_grid_vectors`.
        edge_source: Tensor of dtype torch.float32 of shape (n_edges, 2) where the last
            axis corresponds to x- and y-coordinates of the source points of each edge.
        edge_destination: Tensor of dtype torch.float32 of shape (n_edges, 2) where the
            last axis corresponds to x- and y-coordinates of the destination points of
            each edge.
        sigma: Standard deviation of the 2D Gaussian distribution sampled to generate
            the edge maps for masking the PAFs.

    Returns:
        A set of part affinity fields corresponding to the unit vector pointing along
        the direction of each edge weighted by the probability of each point on a
        sampling grid being on each edge. These will be in a tensor of shape
        (grid_height, grid_width, n_edges, 2) of dtype torch.float32. The last axis
        corresponds to the x- and y-coordinates of the unit vectors.
    """
    unit_vectors = edge_destination - edge_source
    unit_vectors = unit_vectors / torch.norm(unit_vectors, dim=-1, keepdim=True)
    edge_confidence_map = make_edge_maps(
        xv=xv,
        yv=yv,
        edge_source=edge_source,
        edge_destination=edge_destination,
        sigma=sigma,
    )
    pafs = torch.unsqueeze(edge_confidence_map, dim=-1) * expand_to_rank(
        unit_vectors, 4
    )
    return pafs


def make_multi_pafs(
    xv: torch.Tensor,
    yv: torch.Tensor,
    edge_sources: torch.Tensor,
    edge_destinations: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Make multiple instance PAFs with addition reduction.

    Args:
        xv: Sampling grid vector for x-coordinates of shape (grid_width,) and dtype
            torch.float32. This can be generated by
            `sleap_nn.data.utils.make_grid_vectors`.
        yv: Sampling grid vector for y-coordinates of shape (grid_height,) and dtype
            torch.float32. This can be generated by
            `sleap_nn.data.utils.make_grid_vectors`.
        edge_sources: Tensor of dtype torch.float32 of shape (n_instances, n_edges, 2)
            where the last axis corresponds to x- and y-coordinates of the source points
            of each edge.
        edge_destinations: Tensor of dtype torch.float32 of shape (n_instances, n_edges, 2)
            where the last axis corresponds to x- and y-coordinates of the destination
            points of each edge.
        sigma: Standard deviation of the 2D Gaussian distribution sampled to generate
            the edge maps for masking the PAFs.

    Returns:
        A set of part affinity fields generated for each instance. These will be in a
        tensor of shape (grid_height, grid_width, n_edges, 2). If multiple instance
        PAFs are defined on the same pixel, they will be summed.
    """
    grid_height = yv.shape[0]
    grid_width = xv.shape[0]
    n_edges = edge_sources.shape[1]
    n_instances = edge_sources.shape[0]

    pafs = torch.zeros((grid_height, grid_width, n_edges, 2), dtype=torch.float32)

    for i in range(n_instances):
        edge_source = edge_sources[i, :]
        edge_destination = edge_destinations[i, :]

        paf = make_pafs(
            xv=xv,
            yv=yv,
            edge_source=edge_source,
            edge_destination=edge_destination,
            sigma=sigma,
        )

        paf[torch.isnan(paf)] = 0.0

        pafs += paf

    return pafs


def get_edge_points(
    instances: torch.Tensor, edge_inds: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the points in each instance that form a directed graph.

    Args:
        instances: A tensor of shape (n_instances, n_nodes, 2) and dtype torch.float32
            containing instance points where the last axis corresponds to (x, y) pixel
            coordinates on the image. This must be rank-3 even if a single instance is
            present.
        edge_inds: A tensor of shape (n_edges, 2) and dtype torch.int32 containing the node
            indices that define a directed graph, where the last axis corresponds to the
            source and destination node indices.

    Returns:
        Tuple of (edge_sources, edge_destinations) containing the edge and destination
        points respectively. Both will be tensors of shape (n_instances, n_edges, 2),
        where the last axis corresponds to (x, y) pixel coordinates on the image.
    """
    source_inds = edge_inds[:, 0].to(torch.int32)
    destination_inds = edge_inds[:, 1].to(torch.int32)

    edge_sources = instances[:, source_inds]
    edge_destinations = instances[:, destination_inds]
    return edge_sources, edge_destinations


def generate_pafs(
    instances: torch.Tensor,
    img_hw: Tuple[int],
    sigma: float = 1.5,
    output_stride=2,
    edge_inds: Optional[torch.Tensor] = attrs.field(
        default=None, converter=attrs.converters.optional(ensure_list)
    ),
    flatten_channels: bool = False,
) -> torch.Tensor:
    """Generate part-affinity fields.

    Args:
        instances: Input instances.
        img_hw: Image size as tuple (height, width).
        sigma: The standard deviation of the Gaussian distribution that is used to
            generate confidence maps. Default: 1.5.
        output_stride: The relative stride to use when generating confidence maps.
            A larger stride will generate smaller confidence maps. Default: 2.
        edge_inds: `torch.Tensor` to use for looking up the index of the
            edges.
        flatten_channels: If False, the generated tensors are of shape
            [height, width, n_edges, 2]. If True, generated tensors are of shape
            [height, width, n_edges * 2] by flattening the last 2 axes.
    """
    image_height, image_width = img_hw

    # Generate sampling grid vectors.
    xv, yv = make_grid_vectors(
        image_height=image_height,
        image_width=image_width,
        output_stride=output_stride,
    )
    grid_height = len(yv)
    grid_width = len(xv)
    n_edges = len(edge_inds)

    instances = instances[0]  # n_samples=1
    in_img = (instances > 0) & (instances < torch.stack([xv[-1], yv[-1]]).view(1, 1, 2))
    in_img = in_img.all(dim=-1).any(dim=1)
    assert len(in_img.shape) == 1
    instances = instances[in_img]

    edge_sources, edge_destinations = get_edge_points(instances, edge_inds)
    assert len(edge_sources.shape) == 3
    assert edge_sources.shape[1:] == (n_edges, 2)

    assert len(edge_destinations.shape) == 3
    assert edge_destinations.shape[1:] == (n_edges, 2)

    pafs = make_multi_pafs(
        xv=xv,
        yv=yv,
        edge_sources=edge_sources,
        edge_destinations=edge_destinations,
        sigma=sigma,
    )
    assert pafs.shape == (grid_height, grid_width, n_edges, 2)

    if flatten_channels:
        pafs = pafs.reshape(grid_height, grid_width, n_edges * 2)
        assert pafs.shape == (grid_height, grid_width, n_edges * 2)

    return pafs


class PartAffinityFieldsGenerator(IterDataPipe):
    """Transformer to generate part affinity fields.

    Attributes:
        sigma: Standard deviation of the 2D Gaussian distribution sampled to weight the
            part affinity fields by their distance to the edges. This defines the spread
            in units of the input image's grid, i.e., it does not take scaling in
            previous steps into account.
        output_stride: Relative stride of the generated confidence maps. This is
            effectively the reciprocal of the output scale, i.e., increase this to
            generate confidence maps that are smaller than the input images.
        edge_inds: `torch.Tensor` to use for looking up the index of the
            edges.
        flatten_channels: If False, the generated tensors are of shape
            [height, width, n_edges, 2]. If True, generated tensors are of shape
            [height, width, n_edges * 2] by flattening the last 2 axes.
    """

    def __init__(
        self,
        source_dp: IterDataPipe,
        sigma: float = attrs.field(default=1.0, converter=float),
        output_stride: int = attrs.field(default=1, converter=int),
        edge_inds: Optional[torch.Tensor] = attrs.field(
            default=None, converter=attrs.converters.optional(ensure_list)
        ),
        flatten_channels: bool = False,
    ) -> None:
        """Initialize PartAffinityFieldsGenerator with the source `DataPipe`."""
        self.source_dp = source_dp
        self.sigma = sigma
        self.output_stride = output_stride
        self.edge_inds = edge_inds
        self.flatten_channels = flatten_channels

    @property
    def input_keys(self) -> List[Text]:
        """Return the keys that incoming elements are expected to have."""
        return ["image", "instances", "skeleton_inds"]

    @property
    def output_keys(self) -> List[Text]:
        """Return the keys that outgoing elements will have."""
        return self.input_keys + ["part_affinity_fields"]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Create a dataset that contains the generated confidence maps.

        Args:
            input_ds: A dataset with elements that contain the keys "image",
                "instances" and "skeleton_inds".

        Returns:
            A `tf.data.Dataset` with the same keys as the input, as well as
            "part_affinity_fields".

            The "part_affinity_fields" key will be a tensor of shape
            (grid_height, grid_width, n_edges, 2) containing the combined part affinity
            fields of all instances in the frame.

            If the `flatten_channels` attribute is set to True, the last 2 axes of the
            "part_affinity_fields" are flattened to produce a tensor of shape
            (grid_height, grid_width, n_edges * 2). This is a convenient form when
            training models as a rank-4 (batched) tensor will generally be expected.

        Notes:
            The output stride is relative to the current scale of the image. To map
            points on the part affinity fields to the raw image, first multiply them by
            the output stride, and then scale the x- and y-coordinates by the "scale"
            key.

            Importantly, the sigma will be proportional to the current image grid, not
            the original grid prior to scaling operations.
        """
        for ex in self.source_dp:
            image_height = ex["image"].shape[2]
            image_width = ex["image"].shape[3]

            # Generate sampling grid vectors.
            xv, yv = make_grid_vectors(
                image_height=image_height,
                image_width=image_width,
                output_stride=self.output_stride,
            )
            grid_height = len(yv)
            grid_width = len(xv)
            n_edges = len(self.edge_inds)

            instances = ex["instances"][0]  # batch size=1
            in_img = (instances > 0) & (
                instances < torch.stack([xv[-1], yv[-1]]).view(1, 1, 2)
            )
            in_img = in_img.all(dim=-1).any(dim=1)
            assert len(in_img.shape) == 1
            instances = instances[in_img]

            edge_sources, edge_destinations = get_edge_points(instances, self.edge_inds)
            assert len(edge_sources.shape) == 3
            assert edge_sources.shape[1:] == (n_edges, 2)

            assert len(edge_destinations.shape) == 3
            assert edge_destinations.shape[1:] == (n_edges, 2)

            pafs = make_multi_pafs(
                xv=xv,
                yv=yv,
                edge_sources=edge_sources,
                edge_destinations=edge_destinations,
                sigma=self.sigma,
            )
            assert pafs.shape == (grid_height, grid_width, n_edges, 2)

            if self.flatten_channels:
                pafs = pafs.reshape(grid_height, grid_width, n_edges * 2)
                assert pafs.shape == (grid_height, grid_width, n_edges * 2)

            ex["part_affinity_fields"] = pafs

            yield ex
