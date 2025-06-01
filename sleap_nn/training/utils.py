"""Miscellaneous utility functions for training."""

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from torch import nn
import torch.distributed as dist
from typing import Optional, Tuple

import sleap_io as sio
from sleap_nn.data.providers import get_max_instances


def is_distributed_initialized():
    """Check if distributed processes are initialized."""
    return dist.is_available() and dist.is_initialized()


def get_dist_rank():
    """Return the rank of the current process if torch.distributed is initialized."""
    return dist.get_rank() if is_distributed_initialized() else None


def xavier_init_weights(x):
    """Function to initilaise the model weights with Xavier initialization method."""
    if isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear):
        nn.init.xavier_uniform_(x.weight)
        nn.init.constant_(x.bias, 0)


def plot_pafs(
    img: np.ndarray,
    pafs: np.ndarray,
    plot_title: Optional[str] = None,
):
    """Plot the predicted peaks on input image overlayed with confmaps.

    Args:
        img: Input image with shape (channel, height, width).
        pafs: Output pafs with shape (pafs_height, pafs_width, num_edges*2).
        plot_title: Title for the plot.
    """
    img_h, img_w = img.shape[-2:]
    img = img.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

    pafs = pafs.reshape((pafs.shape[0], pafs.shape[1], -1, 2))  # (h, w, edges, 2)
    pafs_mag = np.sqrt(pafs[..., 0] ** 2 + pafs[..., 1] ** 2)
    pafs_mag = np.squeeze(pafs_mag.max(axis=-1))

    fig, ax = plt.subplots()
    ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.imshow(img)

    ax.imshow(pafs_mag, alpha=0.5, extent=[0, img_w, img_h, 0])

    if plot_title is not None:
        ax.set_title(f"{plot_title}")

    ax.legend()

    return fig


def plot_pred_confmaps_peaks(
    img: np.ndarray,
    confmaps: np.ndarray,
    peaks: Optional[np.ndarray] = None,
    gt_instances: Optional[np.ndarray] = None,
    plot_title: Optional[str] = None,
):
    """Plot the predicted peaks on input image overlayed with confmaps.

    Args:
        img: Input image with shape (channel, height, width).
        confmaps: Output confmaps with shape (num_nodes, confmap_height, confmap_width).
        peaks: Predicted keypoints with shape (num_instances, num_nodes, 2).
        gt_instances: Ground-truth keypoints with shape (num_instances,  num_nodes, 2).
        plot_title: Title for the plot.
    """
    img_h, img_w = img.shape[-2:]
    img = img.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

    confmaps = confmaps.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    confmaps = np.max(np.abs(confmaps), axis=-1)

    fig, ax = plt.subplots()
    ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.imshow(img)

    ax.imshow(confmaps, alpha=0.5, extent=[0, img_w, img_h, 0])

    if gt_instances is not None:
        for instance in gt_instances:
            ax.plot(
                instance[:, 0],
                instance[:, 1],
                "go",
                markersize=8,
                markeredgewidth=2,
                label="GT keypoints",
            )

    if peaks is not None:
        for peak in peaks:
            ax.plot(
                peak[:, 0],
                peak[:, 1],
                "rx",
                markersize=8,
                markeredgewidth=2,
                label="Predicted peaks",
            )

    if plot_title is not None:
        ax.set_title(f"{plot_title}")

    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    return fig


def check_memory(
    labels: sio.Labels,
    max_hw: Tuple[int, int],
    model_type: str,
    input_scaling: float,
    crop_size: Optional[int],
):
    """Return memory required for caching the image samples."""
    if model_type == "centered_instance":
        num_samples = len(labels) * get_max_instances(labels)
        img = (labels[0].image / 255.0).astype(np.float32)
        img_mem = (crop_size**2) * img.shape[-1] * img.itemsize * num_samples

        return img_mem

    num_lfs = len(labels)
    img = (labels[0].image / 255.0).astype(np.float32)
    h, w = max_hw[0] * input_scaling, max_hw[1] * input_scaling
    img_mem = h * w * img.shape[-1] * img.itemsize * num_lfs

    return img_mem
