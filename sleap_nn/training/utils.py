"""Miscellaneous utility functions for training."""

from dataclasses import dataclass, field
from io import BytesIO
import numpy as np
import matplotlib

matplotlib.use(
    "Agg"
)  # Use non-interactive backend to avoid tkinter issues on Windows CI
import matplotlib.pyplot as plt
from loguru import logger
from torch import nn
import torch.distributed as dist
import seaborn as sns
from typing import List, Optional
import shutil
import os
import subprocess


def is_distributed_initialized():
    """Check if distributed processes are initialized."""
    return dist.is_available() and dist.is_initialized()


def get_dist_rank():
    """Return the rank of the current process if torch.distributed is initialized."""
    return dist.get_rank() if is_distributed_initialized() else None


def get_gpu_memory() -> List[int]:
    """Get the available memory on each GPU.

    Returns:
        A list of the available memory on each GPU in MiB.
    """
    if shutil.which("nvidia-smi") is None:
        return []

    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv",
    ]

    try:
        memory_poll = subprocess.run(command, capture_output=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return []

    subprocess_result = memory_poll.stdout
    memory_string = subprocess_result.decode("ascii").split("\n")[1:-1]

    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    else:
        cuda_visible_devices = None

    memory_list = []
    for row in memory_string:
        gpu_index, available_memory = row.split(", ")
        available_memory = available_memory.split(" MiB")[0]

        if cuda_visible_devices is None or gpu_index in cuda_visible_devices:
            memory_list.append(int(available_memory))

    return memory_list


def xavier_init_weights(x):
    """Function to initilaise the model weights with Xavier initialization method."""
    if isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear):
        if x.weight is not None:
            nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


def imgfig(
    size: float | tuple = 6, dpi: int = 72, scale: float = 1.0
) -> matplotlib.figure.Figure:
    """Create a tight figure for image plotting.

    Args:
        size: Scalar or 2-tuple specifying the (width, height) of the figure in inches.
            If scalar, will assume equal width and height.
        dpi: Dots per inch, controlling the resolution of the image.
        scale: Factor to scale the size of the figure by. This is a convenience for
            increasing the size of the plot at the same DPI.

    Returns:
        A matplotlib.figure.Figure to use for plotting.
    """
    if not isinstance(size, (tuple, list)):
        size = (size, size)
    fig = plt.figure(figsize=(scale * size[0], scale * size[1]), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return fig


def plot_img(
    img: np.ndarray, dpi: int = 72, scale: float = 1.0
) -> matplotlib.figure.Figure:
    """Plot an image in a tight figure.

    Args:
        img: Image to plot of shape (height, width, channel).
        dpi: Dots per inch, controlling the resolution of the image.
        scale: Factor to scale the size of the figure by. This is a convenience for
            increasing the size of the plot at the same DPI.

    Returns:
        A matplotlib.figure.Figure to use for plotting.
    """
    if hasattr(img, "numpy"):
        img = img.numpy()

    if img.shape[0] == 1:
        # Squeeze out batch singleton dimension.
        img = img.squeeze(axis=0)

    # Check if image is grayscale (single channel).
    grayscale = img.shape[-1] == 1
    if grayscale:
        # Squeeze out singleton channel.
        img = img.squeeze(axis=-1)

    # Normalize the range of pixel values.
    img_min = img.min()
    img_max = img.max()
    if img_min < 0.0 or img_max > 1.0:
        img = (img - img_min) / (img_max - img_min)

    fig = imgfig(
        size=(float(img.shape[1]) / dpi, float(img.shape[0]) / dpi),
        dpi=dpi,
        scale=scale,
    )

    ax = fig.gca()
    ax.imshow(
        img,
        cmap="gray" if grayscale else None,
        origin="upper",
        extent=[-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5],
    )
    return fig


def plot_confmaps(confmaps: np.ndarray, output_scale: float = 1.0):
    """Plot confidence maps reduced over channels.

    Args:
        confmaps: Confidence maps to plot with shape (height, width, channel).
        output_scale: Factor to scale the size of the figure by.

    Returns:
        A matplotlib.figure.Figure to use for plotting.
    """
    ax = plt.gca()
    return ax.imshow(
        np.squeeze(confmaps.max(axis=-1)),
        alpha=0.5,
        origin="upper",
        vmin=0,
        vmax=1,
        extent=[
            -0.5,
            confmaps.shape[1] / output_scale - 0.5,
            confmaps.shape[0] / output_scale - 0.5,
            -0.5,
        ],
    )


def plot_peaks(
    pts_gt: np.ndarray, pts_pr: np.ndarray | None = None, paired: bool = False
):
    """Plot ground truth and detected peaks.

    Args:
        pts_gt: Ground-truth keypoints of shape (num_instances, nodes, 2). To plot centroids, shape: (1, num_instances, 2).
        pts_pr: Predicted keypoints of shape (num_instances, nodes, 2). To plot centroids, shape: (1, num_instances, 2)
        paired: True if error lines should be plotted else False.

    Returns:
        A matplotlib.figure.Figure to use for plotting.
    """
    handles = []
    ax = plt.gca()
    if paired and pts_pr is not None:
        for pt_gt, pt_pr in zip(pts_gt, pts_pr):
            for p_gt, p_pr in zip(pt_gt, pt_pr):
                handles.append(
                    ax.plot(
                        [p_gt[0], p_pr[0]], [p_gt[1], p_pr[1]], "r-", alpha=0.5, lw=2
                    )
                )
    if pts_pr is not None:
        handles.append(
            ax.plot(
                pts_gt[..., 0].ravel(),
                pts_gt[..., 1].ravel(),
                "g.",
                alpha=0.7,
                ms=10,
                mew=1,
                mec="w",
            )
        )
        handles.append(
            ax.plot(
                pts_pr[..., 0].ravel(),
                pts_pr[..., 1].ravel(),
                "r.",
                alpha=0.7,
                ms=10,
                mew=1,
                mec="w",
            )
        )
    else:
        cmap = sns.color_palette("tab20")
        for i, pts in enumerate(pts_gt):
            handles.append(
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    ".",
                    alpha=0.7,
                    ms=15,
                    mew=1,
                    mfc=cmap[i % len(cmap)],
                    mec="w",
                )
            )
    return handles


@dataclass
class VisualizationData:
    """Container for visualization data from a single sample.

    This dataclass decouples data extraction from rendering, allowing the same
    data to be rendered to different output targets (matplotlib, wandb, etc.).

    Attributes:
        image: Input image as (H, W, C) numpy array, normalized to [0, 1].
        pred_confmaps: Predicted confidence maps as (H, W, nodes) array, values in [0, 1].
        pred_peaks: Predicted keypoints as (instances, nodes, 2) or (nodes, 2) array.
        pred_peak_values: Confidence values as (instances, nodes) or (nodes,) array.
        gt_instances: Ground truth keypoints, same shape as pred_peaks.
        node_names: List of node/keypoint names, e.g., ["head", "thorax", ...].
        output_scale: Ratio of confmap size to image size (confmap_h / image_h).
        is_paired: Whether GT and predictions can be paired for error visualization.
        pred_pafs: Part affinity fields for bottom-up models, optional.
        pred_class_maps: Class maps for multi-class models, optional.
    """

    image: np.ndarray
    pred_confmaps: np.ndarray
    pred_peaks: np.ndarray
    pred_peak_values: np.ndarray
    gt_instances: np.ndarray
    node_names: List[str] = field(default_factory=list)
    output_scale: float = 1.0
    is_paired: bool = True
    pred_pafs: Optional[np.ndarray] = None
    pred_class_maps: Optional[np.ndarray] = None


class MatplotlibRenderer:
    """Renders VisualizationData to matplotlib figures."""

    def render(self, data: VisualizationData) -> matplotlib.figure.Figure:
        """Render visualization data to a matplotlib figure.

        Args:
            data: VisualizationData containing image, confmaps, peaks, etc.

        Returns:
            A matplotlib Figure object.
        """
        img = data.image
        scale = 1.0
        if img.shape[0] < 512:
            scale = 2.0
        if img.shape[0] < 256:
            scale = 4.0

        fig = plot_img(img, dpi=72 * scale, scale=scale)
        plot_confmaps(data.pred_confmaps, output_scale=data.output_scale)
        plot_peaks(data.gt_instances, data.pred_peaks, paired=data.is_paired)
        return fig

    def render_pafs(self, data: VisualizationData) -> matplotlib.figure.Figure:
        """Render PAF magnitude visualization.

        Args:
            data: VisualizationData with pred_pafs populated.

        Returns:
            A matplotlib Figure object showing PAF magnitudes.
        """
        if data.pred_pafs is None:
            raise ValueError("pred_pafs is None, cannot render PAFs")

        img = data.image
        scale = 1.0
        if img.shape[0] < 512:
            scale = 2.0
        if img.shape[0] < 256:
            scale = 4.0

        # Compute PAF magnitude
        pafs = data.pred_pafs  # (H, W, 2*edges) or (H, W, edges, 2)
        if pafs.ndim == 3:
            n_edges = pafs.shape[-1] // 2
            pafs = pafs.reshape(pafs.shape[0], pafs.shape[1], n_edges, 2)
        magnitude = np.sqrt(pafs[..., 0] ** 2 + pafs[..., 1] ** 2)
        magnitude = magnitude.max(axis=-1)  # Max over edges

        fig = plot_img(img, dpi=72 * scale, scale=scale)
        ax = plt.gca()

        # Calculate PAF output scale from actual PAF dimensions, not confmap output_scale
        # PAFs may have a different output_stride than confmaps
        paf_output_scale = magnitude.shape[0] / img.shape[0]

        ax.imshow(
            magnitude,
            alpha=0.5,
            origin="upper",
            cmap="viridis",
            extent=[
                -0.5,
                magnitude.shape[1] / paf_output_scale - 0.5,
                magnitude.shape[0] / paf_output_scale - 0.5,
                -0.5,
            ],
        )
        return fig


class WandBRenderer:
    """Renders VisualizationData to wandb.Image objects.

    Supports multiple rendering modes:
    - "direct": Pre-render with matplotlib, convert to wandb.Image
    - "boxes": Use wandb boxes for interactive keypoint visualization
    - "masks": Use wandb masks for confidence map overlay
    """

    def __init__(
        self,
        mode: str = "direct",
        box_size: float = 5.0,
        confmap_threshold: float = 0.1,
        min_size: int = 512,
    ):
        """Initialize the renderer.

        Args:
            mode: Rendering mode - "direct", "boxes", or "masks".
            box_size: Size of keypoint boxes in pixels (for "boxes" mode).
            confmap_threshold: Threshold for confmap mask (for "masks" mode).
            min_size: Minimum image dimension. Smaller images will be upscaled.
        """
        self.mode = mode
        self.box_size = box_size
        self.confmap_threshold = confmap_threshold
        self.min_size = min_size
        self._mpl_renderer = MatplotlibRenderer()

    def render(
        self, data: VisualizationData, caption: Optional[str] = None
    ) -> "wandb.Image":
        """Render visualization data to a wandb.Image.

        Args:
            data: VisualizationData containing image, confmaps, peaks, etc.
            caption: Optional caption for the image.

        Returns:
            A wandb.Image object.
        """
        import wandb

        if self.mode == "boxes":
            return self._render_with_boxes(data, caption)
        elif self.mode == "masks":
            return self._render_with_masks(data, caption)
        else:  # "direct"
            return self._render_direct(data, caption)

    def _get_scale_factor(self, img_h: int, img_w: int) -> int:
        """Calculate scale factor to ensure minimum image size."""
        min_dim = min(img_h, img_w)
        if min_dim >= self.min_size:
            return 1
        return int(np.ceil(self.min_size / min_dim))

    def _render_direct(
        self, data: VisualizationData, caption: Optional[str] = None
    ) -> "wandb.Image":
        """Pre-render with matplotlib, return as wandb.Image."""
        import wandb
        from PIL import Image

        fig = self._mpl_renderer.render(data)

        # Convert figure to PIL Image
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        plt.close(fig)

        pil_image = Image.open(buf)
        return wandb.Image(pil_image, caption=caption)

    def _render_with_boxes(
        self, data: VisualizationData, caption: Optional[str] = None
    ) -> "wandb.Image":
        """Use wandb boxes for interactive keypoint visualization."""
        import wandb
        from PIL import Image

        # Prepare class labels from node names
        class_labels = {i: name for i, name in enumerate(data.node_names)}
        if not class_labels:
            class_labels = {i: f"node_{i}" for i in range(data.pred_peaks.shape[-2])}

        # Convert image to uint8
        img_uint8 = (np.clip(data.image, 0, 1) * 255).astype(np.uint8)
        # Handle single-channel images: squeeze (H, W, 1) -> (H, W)
        if img_uint8.ndim == 3 and img_uint8.shape[2] == 1:
            img_uint8 = img_uint8.squeeze(axis=2)
        img_h, img_w = img_uint8.shape[:2]

        # Scale up small images for better visibility in wandb
        scale = self._get_scale_factor(img_h, img_w)
        if scale > 1:
            pil_img = Image.fromarray(img_uint8)
            pil_img = pil_img.resize(
                (img_w * scale, img_h * scale), resample=Image.BILINEAR
            )
            img_uint8 = np.array(pil_img)

        # Build ground truth boxes (use percent domain for proper thumbnail scaling)
        gt_box_data = self._peaks_to_boxes(
            data.gt_instances, data.node_names, img_w, img_h, is_gt=True
        )

        # Build prediction boxes
        pred_box_data = self._peaks_to_boxes(
            data.pred_peaks,
            data.node_names,
            img_w,
            img_h,
            peak_values=data.pred_peak_values,
            is_gt=False,
        )

        return wandb.Image(
            img_uint8,
            boxes={
                "ground_truth": {"box_data": gt_box_data, "class_labels": class_labels},
                "predictions": {
                    "box_data": pred_box_data,
                    "class_labels": class_labels,
                },
            },
            caption=caption,
        )

    def _peaks_to_boxes(
        self,
        peaks: np.ndarray,
        node_names: List[str],
        img_w: int,
        img_h: int,
        peak_values: Optional[np.ndarray] = None,
        is_gt: bool = False,
    ) -> List[dict]:
        """Convert peaks array to wandb box_data format.

        Args:
            peaks: Keypoints as (instances, nodes, 2) or (nodes, 2).
            node_names: List of node names.
            img_w: Image width in pixels.
            img_h: Image height in pixels.
            peak_values: Optional confidence values.
            is_gt: Whether these are ground truth points.

        Returns:
            List of box dictionaries for wandb.
        """
        box_data = []

        # Normalize shape to (instances, nodes, 2)
        if peaks.ndim == 2:
            peaks = peaks[np.newaxis, ...]
            if peak_values is not None and peak_values.ndim == 1:
                peak_values = peak_values[np.newaxis, ...]

        # Convert box_size from pixels to percent
        box_w_pct = self.box_size / img_w
        box_h_pct = self.box_size / img_h

        for inst_idx, instance in enumerate(peaks):
            for node_idx, (x, y) in enumerate(instance):
                if np.isnan(x) or np.isnan(y):
                    continue

                node_name = (
                    node_names[node_idx]
                    if node_idx < len(node_names)
                    else f"node_{node_idx}"
                )

                # Convert pixel coordinates to percent (0-1 range)
                x_pct = float(x) / img_w
                y_pct = float(y) / img_h

                box = {
                    "position": {
                        "middle": [x_pct, y_pct],
                        "width": box_w_pct,
                        "height": box_h_pct,
                    },
                    "domain": "percent",
                    "class_id": node_idx,
                }

                if is_gt:
                    box["box_caption"] = f"GT: {node_name}"
                else:
                    if peak_values is not None:
                        conf = float(peak_values[inst_idx, node_idx])
                        box["box_caption"] = f"{node_name} ({conf:.2f})"
                        box["scores"] = {"confidence": conf}
                    else:
                        box["box_caption"] = node_name

                box_data.append(box)

        return box_data

    def _render_with_masks(
        self, data: VisualizationData, caption: Optional[str] = None
    ) -> "wandb.Image":
        """Use wandb masks for confidence map overlay.

        Uses argmax approach: each pixel shows the dominant node.
        """
        import wandb

        # Prepare class labels (0 = background, 1+ = nodes)
        class_labels = {0: "background"}
        for i, name in enumerate(data.node_names):
            class_labels[i + 1] = name
        if not data.node_names:
            n_nodes = data.pred_confmaps.shape[-1]
            for i in range(n_nodes):
                class_labels[i + 1] = f"node_{i}"

        # Create argmax mask from confmaps
        confmaps = data.pred_confmaps  # (H/stride, W/stride, nodes)
        max_vals = confmaps.max(axis=-1)
        argmax_map = confmaps.argmax(axis=-1) + 1  # +1 for background offset
        argmax_map[max_vals < self.confmap_threshold] = 0  # Background

        # Convert image to uint8
        img_uint8 = (np.clip(data.image, 0, 1) * 255).astype(np.uint8)
        # Handle single-channel images: (H, W, 1) -> (H, W)
        if img_uint8.ndim == 3 and img_uint8.shape[2] == 1:
            img_uint8 = img_uint8.squeeze(axis=2)
        img_h, img_w = img_uint8.shape[:2]

        # Resize mask to match image dimensions (confmaps are H/stride, W/stride)
        from PIL import Image

        mask_pil = Image.fromarray(argmax_map.astype(np.uint8))
        mask_pil = mask_pil.resize((img_w, img_h), resample=Image.NEAREST)
        argmax_map = np.array(mask_pil)

        return wandb.Image(
            img_uint8,
            masks={
                "confidence_maps": {
                    "mask_data": argmax_map.astype(np.uint8),
                    "class_labels": class_labels,
                }
            },
            caption=caption,
        )
