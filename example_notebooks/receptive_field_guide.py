# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "kornia==0.8.1",
#     "marimo",
#     "matplotlib==3.9.4",
#     "numpy",
#     "omegaconf==2.3.0",
#     "opencv-python==4.12.0.88",
#     "pillow==11.3.0",
#     "seaborn==0.13.2",
#     "sleap-io==0.4.1",
#     "torch==2.7.1",
#     "torchvision==0.22.1",
#     "zmq==0.0.0",
# ]
# ///

import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Receptive Field Guide""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    _**Note**_:
    This notebook executes automatically; there is no need to run individual cells, as all interactions are managed through the provided UI elements (sliders, buttons, etc.). Just upload a sample image and click `"Start exploring receptive fields!"` button!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This notebook shows how the [receptive field](https://distill.pub/2019/computing-receptive-fields/) changes when you adjust `max_stride` and `data_config.preprocessing.scale`, and how to pick good values. Receptive field is essentially the region of the input image that influence the value of a single pixel in the feature maps of a given layer—here, the last encoder layer.

    In sleap-nn, receptive field is governed primarily by `data_config.preprocessing.scale` and `max_stride`. For a UNet backbone, max_stride is configurable; for ConvNeXt and SwinT, the effective stride is fixed and adjustable only through the `stem_patch_stride` (for these, max_stride = stem_patch_stride × 16). Let's visualize how the receptive field varies under different settings!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Upload a sample image:""")
    return


@app.cell(hide_code=True)
def _(mo):
    image = mo.ui.file(kind="area", label="Upload a sample training image")
    image
    return (image,)


@app.cell(hide_code=True)
def _(mo):
    run_rf = mo.ui.run_button(label="Start exploring receptive fields!")
    run_rf
    return (run_rf,)


@app.cell(hide_code=True)
def _(Image, image, io, mo, run_rf):
    if not run_rf.value:
        mo.stop("Click `Start exploring receptive fields!` to start.")

    if image.value is not None:
        src_image = Image.open(io.BytesIO(image.value[0].contents))

    mo.vstack(
        [mo.image(src_image, caption="Source image", width=400, height=250)],
        align="center",
    )
    return (src_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Lowering the input scale (more downsampling) makes the receptive field larger relative to the original image but can erase fine details; increasing max_stride (leads to more down blocks in the backbone) also makes the receptive field larger but increases the number of model parameters, and thus, increases GPU memory and training time."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    scale = mo.ui.slider(
        0, 1.0, step=0.1, value=1.0, label="Choose an input scale value: "
    )
    max_stride = mo.ui.dropdown(
        options=[8, 16, 32, 64, 128], value=16, label="Choose a max_stride value: "
    )
    mo.vstack([scale, max_stride])
    return max_stride, scale


@app.cell(hide_code=True)
def _(np, plt):
    import matplotlib.patches as patches
    from typing import Optional, Tuple

    def compute_receptive_field(
        down_blocks: int, convs_per_block: int = 2, kernel_size: int = 3
    ) -> int:
        """
        Computes receptive field for specified model architecture.

        Based on SLEAP's implementation from:
        https://distill.pub/2019/computing-receptive-fields/ (Eq. 2)

        Args:
            down_blocks: Number of downsampling blocks
            convs_per_block: Number of convolutions per block (default: 2)
            kernel_size: Convolution kernel size (default: 3)

        Returns:
            Receptive field size in pixels
        """
        # Define the strides and kernel sizes for a single down block.
        # convs have stride 1, pooling has stride 2:
        block_strides = [1] * convs_per_block + [2]

        # convs have `kernel_size` x `kernel_size` kernels, pooling has 2 x 2 kernels:
        block_kernels = [kernel_size] * convs_per_block + [2]

        # Repeat block parameters by the total number of down blocks.
        strides = np.array(block_strides * down_blocks)
        kernels = np.array(block_kernels * down_blocks)

        # L = Total number of layers
        L = len(strides)

        # Compute the product term of the RF equation.
        rf = 1
        for l in range(L):
            rf += (kernels[l] - 1) * np.prod(strides[:l])

        return int(rf)

    def plot_receptive_field(
        image: np.ndarray,
        max_stride: int,
        scale: float = 1.0,
        convs_per_block: int = 2,
        kernel_size: int = 3,
        figsize: Tuple[int, int] = (10, 8),
        box_color: str = "red",
        box_linewidth: int = 3,
        show_info: bool = True,
    ) -> None:
        """
        Plot an image with a receptive field box.

        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            max_stride: Maximum stride value (determines number of down blocks)
            scale: Input scaling factor (how much the image will be scaled during training)
            convs_per_block: Number of convolutions per block (default: 2)
            kernel_size: Convolution kernel size (default: 3)
            figsize: Figure size as (width, height) tuple
            box_color: Color of the receptive field box
            box_linewidth: Line width of the box
            show_info: Whether to show receptive field information
        """
        # Compute receptive field size
        down_blocks = int(np.log2(max_stride))
        rf_size = compute_receptive_field(down_blocks, convs_per_block, kernel_size)

        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Display the image
        if len(image.shape) == 3:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap="gray")

        # Calculate box dimensions
        # The box size needs to be adjusted for the input scaling
        scaled_box_size = rf_size / scale

        # Calculate center position (center of the image)
        h, w = image.shape[:2]
        center_x = w / 2
        center_y = h / 2

        # Calculate box coordinates (top-left corner)
        box_x = center_x - scaled_box_size / 2
        box_y = center_y - scaled_box_size / 2

        # Create and add the receptive field box
        rect = patches.Rectangle(
            (box_x, box_y),
            scaled_box_size,
            scaled_box_size,
            linewidth=box_linewidth,
            edgecolor=box_color,
            facecolor="none",
        )
        ax.add_patch(rect)

        # Add title and labels
        ax.set_title(
            f"Receptive Field: {rf_size} pixels, Max stride: {max_stride}, Scale: {scale}",
            fontsize=14,
            fontweight="bold",
        )

        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

        # Add receptive field information if requested
        if show_info:
            info_text = f"""Receptive Field Info:
    • Size: {rf_size} pixels
    • Max Stride: {max_stride}
    • Down Blocks: {down_blocks}
    • Input Scale: {scale}"""

            # Position text in top-left corner
            ax.text(
                0.02,
                0.98,
                info_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.show()

        return rf_size

    def plot_receptive_field_comparison(
        image: np.ndarray,
        max_strides: list,
        scale: float = 1.0,
        convs_per_block: int = 2,
        kernel_size: int = 3,
        figsize: Tuple[int, int] = (15, 10),
        colors: list = None,
    ) -> None:
        """
        Plot an image with multiple receptive field boxes for different max_stride values.

        Args:
            image: Input image as numpy array
            max_strides: List of max_stride values to compare
            scale: Input scaling factor
            convs_per_block: Number of convolutions per block
            kernel_size: Convolution kernel size
            figsize: Figure size
            colors: List of colors for different boxes
        """
        if colors is None:
            colors = ["red", "blue", "green", "orange", "purple", "brown"]

        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Display the image
        if len(image.shape) == 3:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap="gray")

        # Calculate center position
        h, w = image.shape[:2]
        center_x = w / 2
        center_y = h / 2

        # Add boxes for each max_stride
        for i, max_stride in enumerate(max_strides):
            # Compute receptive field
            down_blocks = int(np.log2(max_stride))
            rf_size = compute_receptive_field(down_blocks, convs_per_block, kernel_size)

            # Calculate box dimensions
            scaled_box_size = rf_size / scale
            box_x = center_x - scaled_box_size / 2
            box_y = center_y - scaled_box_size / 2

            # Create box
            color = colors[i % len(colors)]
            rect = patches.Rectangle(
                (box_x, box_y),
                scaled_box_size,
                scaled_box_size,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
                label=f"Max Stride: {max_stride} (RF: {rf_size})",
            )
            ax.add_patch(rect)

        # Add title and legend
        ax.set_title(
            f"Receptive Field Comparison (scale: {scale})",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.show()

    return (plot_receptive_field,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The receptive field should be about the size of the animal—large enough to recognize what’s at its center using only the area inside the RF box. In top-down pipelines, use a larger RF for the centroid detector than for the instance/keypoint head: centroids benefit from broader context to identify an instance, while centered-instance model heads need finer detail to detect the keypoints."""
    )
    return


@app.cell(hide_code=True)
def _(max_stride, np, plot_receptive_field, scale, src_image):
    rf_size = plot_receptive_field(
        image=np.array(src_image),
        max_stride=max_stride.value,
        scale=scale.value,
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import cv2
    import torch
    import pprint
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torchvision import transforms

    from omegaconf import OmegaConf

    import random

    from PIL import Image
    import io

    return Image, io, mo, np, plt


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
