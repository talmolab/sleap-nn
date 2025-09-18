"""Miscellaneous utility functions for training."""

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from torch import nn
import torch.distributed as dist
import matplotlib
import seaborn as sns
from typing import List
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
