from typing import Tuple
import torch

def make_grid_vectors(
    image_height: int, image_width: int, output_stride: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make sampling grid vectors from image dimensions.

    This is a useful function for creating the x- and y-vectors that define a sampling
    grid over an image space. These vectors can be used to generate a full meshgrid or
    for equivalent broadcasting operations.

    Args:
        image_height: Height of the image grid that will be sampled, specified as a
            scalar integer.
        image_width: width of the image grid that will be sampled, specified as a
            scalar integer.
        output_stride: Sampling step size, specified as a scalar integer. This can be
            used to specify a sampling grid that has a smaller shape than the image
            grid but with values span the same range. This can be thought of as the
            reciprocal of the output scale, i.e., it will induce subsampling when set to
            values greater than 1.

    Returns:
        Tuple of grid vectors (xv, yv). These are tensors of dtype tf.float32 with
        shapes (grid_width,) and (grid_height,) respectively.

        The grid dimensions are calculated as:
            grid_width = image_width // output_stride
            grid_height = image_height // output_stride
    """
    xv = torch.arange(0, image_width, step=output_stride, dtype=torch.float32)
    yv = torch.arange(0, image_height, step=output_stride, dtype=torch.float32)
    return xv, yv