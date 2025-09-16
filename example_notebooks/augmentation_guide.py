# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "kornia==0.8.1",
#     "marimo",
#     "matplotlib==3.9.4",
#     "numpy",
#     "pillow==11.3.0",
#     "seaborn==0.13.2",
#     "torch==2.7.1",
#     "torchvision==0.22.1",
# ]
# ///

import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Visualizing Data Augmentations with SLEAP-NN""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    _**Note**_:
    This notebook executes automatically; there is no need to run individual cells, as all interactions are managed through the provided UI elements (sliders, buttons, etc.). Just upload a sample image and click `"Start exploring augmentations!"` button!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""This notebook provides a visual demo of the various data augmentations available through Kornia in `sleap-nn`. Users can interactively explore two broad classes of augmentations—intensity-based (e.g., brightness, contrast, noise) and geometric (e.g., rotation, scaling, translation)—by adjusting hyperparameters and observing their effects on sample images. This helps in understanding how different transformations impact the data and can guide effective augmentation strategies for training robust models."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Upload a sample image:""")
    return


@app.cell(hide_code=True)
def _(mo):
    file = mo.ui.file(kind="area")
    file
    return (file,)


@app.cell(hide_code=True)
def _(mo):
    run_aug = mo.ui.run_button(label="Start exploring augmentations!")
    run_aug
    return (run_aug,)


@app.cell(hide_code=True)
def _(Image, file, io, mo, run_aug):
    if not run_aug.value:
        mo.stop("Click `Start exploring augmentations!` to start.")

    if file.value is not None:
        src_image = Image.open(io.BytesIO(file.value[0].contents))

    mo.vstack(
        [mo.image(src_image, caption="Source image", width=400, height=250)],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Geometric Augmentations""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Rotation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Rotates the image.
    **_Note_**: `K.RandomAffine` takes in either an `int` or a `tuple` for rotation degree and it randomly samples an angle from (-degree, degree) if we provide an integer or from the given range. For demo purposes, we set min and max to same value
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    slider_rotation_angle = mo.ui.slider(
        0, 180, value=0, step=1.0, label="Rotation angle"
    )
    mo.vstack([slider_rotation_angle])
    return (slider_rotation_angle,)


@app.cell(hide_code=True)
def _(Image, K, file, io, mo, slider_rotation_angle, to_tensor, transforms):
    _aug = K.AugmentationSequential(
        K.RandomAffine(
            degrees=(slider_rotation_angle.value, slider_rotation_angle.value),
            p=1.0,
            keepdim=True,
            same_on_batch=True,
        ),
        data_keys=["input"],  # Just to define the future input here.
        same_on_batch=False,
    )

    if file.value is not None:
        _image = Image.open(io.BytesIO(file.value[0].contents))

        # Define the transformation to convert the image to a tensor
        _to_tensor = transforms.ToTensor()

        # Apply the transformation
        _tensor_image = to_tensor(_image)

        _aug_image = _aug(_tensor_image)

        _aug_image_pil = transforms.ToPILImage()(_aug_image)

    mo.hstack(
        [
            mo.image(_image, width=400, height=250),
            mo.image(_aug_image_pil, width=400, height=250),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Scale""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Scales the image without changing the output image size.
    **_Note_**: `K.RandomAffine` usually takes in a tuple: `(scale_min, scale_max)` and randomly samples a scaling fcator between the given range. For demo purposes, we set min and max to same value
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    slider_scale = mo.ui.slider(
        0, 10, value=1.0, step=0.01, label="Ratio to resize image to"
    )
    mo.vstack([slider_scale])
    return (slider_scale,)


@app.cell(hide_code=True)
def _(Image, K, file, io, mo, slider_scale, to_tensor, transforms):
    _aug = K.AugmentationSequential(
        K.RandomAffine(
            degrees=(0, 0),
            scale=(slider_scale.value, slider_scale.value),
            p=1.0,
            keepdim=True,
            same_on_batch=True,
        ),
        data_keys=["input"],  # Just to define the future input here.
        same_on_batch=False,
    )

    if file.value is not None:
        _image = Image.open(io.BytesIO(file.value[0].contents))

        # Define the transformation to convert the image to a tensor
        _to_tensor = transforms.ToTensor()

        # Apply the transformation
        _tensor_image = to_tensor(_image)

        _aug_image = _aug(_tensor_image)

        _aug_image_pil = transforms.ToPILImage()(_aug_image)

    mo.hstack(
        [
            mo.image(_image, width=400, height=250),
            mo.image(_aug_image_pil, width=400, height=250),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Translate""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Shifts the image horizontally and vertically by a fraction of its width and height.
    **_Note_**: shift is randomly computed within `-img_width * translate_width_ratio < dx < img_width * translate_width_ratio`
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    slider_translate_width = mo.ui.slider(
        0, 1, value=0.0, step=0.01, label="Translate width"
    )
    slider_translate_height = mo.ui.slider(
        0, 1, value=0.0, step=0.01, label="Translate height"
    )

    mo.vstack([slider_translate_width, slider_translate_height])
    return slider_translate_height, slider_translate_width


@app.cell(hide_code=True)
def _(
    Image,
    K,
    file,
    io,
    mo,
    slider_translate_height,
    slider_translate_width,
    to_tensor,
    transforms,
):
    _aug = K.AugmentationSequential(
        K.RandomAffine(
            degrees=(0, 0),
            translate=(slider_translate_width.value, slider_translate_height.value),
            p=1.0,
            keepdim=True,
            same_on_batch=True,
        ),
        data_keys=["input"],  # Just to define the future input here.
        same_on_batch=False,
    )

    if file.value is not None:
        _image = Image.open(io.BytesIO(file.value[0].contents))

        # Define the transformation to convert the image to a tensor
        _to_tensor = transforms.ToTensor()

        # Apply the transformation
        _tensor_image = to_tensor(_image)

        _aug_image = _aug(_tensor_image)

        _aug_image_pil = transforms.ToPILImage()(_aug_image)

    mo.hstack(
        [
            mo.image(_image, width=400, height=250),
            mo.image(_aug_image_pil, width=400, height=250),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Erase""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Erases a random rectangular region in the image by replacing its pixels with a constant value.

    **_Note_**: `K.RandomErasing` takes parameters like scale and ratio which takes in a tuple to specify the range of values, and kornia randomly samples from the range. For demo purposes, we use fix the max and min to be the same.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    slider_erase_scale = mo.ui.slider(
        0, 2, value=0.01, step=0.01, label="Proportion of area to be erased"
    )
    slider_erase_ratio = mo.ui.slider(
        0, 20, value=1.0, step=0.01, label="Aspect ratio of the area to be erased"
    )

    mo.vstack([slider_erase_scale, slider_erase_ratio])
    return slider_erase_ratio, slider_erase_scale


@app.cell(hide_code=True)
def _(
    Image,
    K,
    file,
    io,
    mo,
    slider_erase_ratio,
    slider_erase_scale,
    to_tensor,
    transforms,
):
    _aug = K.AugmentationSequential(
        K.RandomErasing(
            scale=(slider_erase_scale.value, slider_erase_scale.value),
            ratio=(slider_erase_ratio.value, slider_erase_ratio.value),
            p=1.0,
            keepdim=True,
            same_on_batch=True,
        ),
        data_keys=["input"],  # Just to define the future input here.
        same_on_batch=False,
    )

    if file.value is not None:
        _image = Image.open(io.BytesIO(file.value[0].contents))

        # Define the transformation to convert the image to a tensor
        _to_tensor = transforms.ToTensor()

        # Apply the transformation
        _tensor_image = to_tensor(_image)

        _aug_image = _aug(_tensor_image)

        _aug_image_pil = transforms.ToPILImage()(_aug_image)

    mo.hstack(
        [
            mo.image(_image, width=400, height=250),
            mo.image(_aug_image_pil, width=400, height=250),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Intensity Augmentations""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Uniform noise""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Adds pixel-wise uniform noise to the image.

    Note: `RandomUniformNoise` takes a noise parameter which takes in a tuple to specify the range of values, and kornia randomly samples from the range. For demo purposes, we use fix the max and min to be the same.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    slider_uniform_noise = mo.ui.slider(
        0, 1.0, value=0, step=0.01, label="Uniform noise"
    )
    mo.vstack([slider_uniform_noise])
    return (slider_uniform_noise,)


@app.cell(hide_code=True)
def _(torch):
    from typing import Any, Dict, Optional, Tuple
    from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
    from kornia.core import Tensor
    from kornia.augmentation.utils.param_validation import _range_bound

    class RandomUniformNoise(IntensityAugmentationBase2D):
        """Data transformer for applying random uniform noise to input images.

        This is a custom Kornia augmentation inheriting from `IntensityAugmentationBase2D`.
        Uniform noise within (min_val, max_val) is applied to the entire input image.

        Note: Inverse transform is not implemented and re-applying the same transformation
        in the example below does not work when included in an AugmentationSequential class.

        Args:
            noise: 2-tuple (min_val, max_val); 0.0 <= min_val <= max_val <= 1.0.
            p: probability for applying an augmentation. This param controls the augmentation probabilities
              element-wise for a batch.
            p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
              probabilities batch-wise.
            same_on_batch: apply the same transformation across the batch.
            keepdim: whether to keep the output shape the same as input `True` or broadcast it
              to the batch form `False`.

        Examples:
            >>> rng = torch.manual_seed(0)
            >>> img = torch.rand(1, 1, 2, 2)
            >>> RandomUniformNoise(min_val=0., max_val=0.1, p=1.)(img)
            tensor([[[[0.9607, 0.5865],
                      [0.2705, 0.5920]]]])

        To apply the exact augmentation again, you may take the advantage of the previous parameter state:
            >>> input = torch.rand(1, 3, 32, 32)
            >>> aug = RandomUniformNoise(min_val=0., max_val=0.1, p=1.)
            >>> (aug(input) == aug(input, params=aug._params)).all()
            tensor(True)

        Ref: `kornia.augmentation._2d.intensity.gaussian_noise
        <https://kornia.readthedocs.io/en/latest/_modules/kornia/augmentation/_2d/intensity/gaussian_noise.html#RandomGaussianNoise>`_.
        """

        def __init__(
            self,
            noise: Tuple[float, float],
            p: float = 0.5,
            p_batch: float = 1.0,
            clip_output: bool = True,
            same_on_batch: bool = False,
            keepdim: bool = False,
        ) -> None:
            """Initialize the class."""
            super().__init__(
                p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim
            )
            self.flags = {
                "uniform_noise": _range_bound(noise, "uniform_noise", bounds=(0.0, 1.0))
            }
            self.clip_output = clip_output

        def apply_transform(
            self,
            input: Tensor,
            params: Dict[str, Tensor],
            flags: Dict[str, Any],
            transform: Optional[Tensor] = None,
        ) -> Tensor:
            """Compute the uniform noise, add, and clamp output."""
            if "uniform_noise" in params:
                uniform_noise = params["uniform_noise"]
            else:
                uniform_noise = (
                    torch.FloatTensor(input.shape)
                    .uniform_(flags["uniform_noise"][0], flags["uniform_noise"][1])
                    .to(input.device)
                )
                self._params["uniform_noise"] = uniform_noise
            if self.clip_output:
                return torch.clamp(
                    input + uniform_noise, 0.0, 1.0
                )  # RandomGaussianNoise doesn't clamp.
            return input + uniform_noise

    return (RandomUniformNoise,)


@app.cell(hide_code=True)
def _(
    Image,
    K,
    RandomUniformNoise,
    file,
    io,
    mo,
    slider_uniform_noise,
    transforms,
):
    aug = K.AugmentationSequential(
        RandomUniformNoise(
            noise=(slider_uniform_noise.value, slider_uniform_noise.value),
            p=1.0,
            keepdim=True,
            same_on_batch=True,
        ),
        data_keys=["input"],  # Just to define the future input here.
        same_on_batch=False,
    )

    if file.value is not None:
        image_bytes = file.value[0].contents
        image = Image.open(io.BytesIO(image_bytes))

        # Define the transformation to convert the image to a tensor
        to_tensor = transforms.ToTensor()

        # Apply the transformation
        tensor_image = to_tensor(image)

        aug_image = aug(tensor_image)

        aug_image_pil = transforms.ToPILImage()(aug_image)

    mo.hstack(
        [
            mo.image(image, width=400, height=250),
            mo.image(aug_image_pil, width=400, height=250),
        ]
    )
    return (to_tensor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Gaussian noise""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Adds pixel-wise gaussian noise with a specified mean and std to the image."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    slider_gaussian_noise_mean = mo.ui.slider(
        0, 1.0, value=0, step=0.001, label="Gaussian noise mean"
    )
    slider_gaussian_noise_std = mo.ui.slider(
        0, 1.0, value=0, step=0.001, label="Gaussian noise standard deviation"
    )
    mo.vstack([slider_gaussian_noise_mean, slider_gaussian_noise_std])
    return slider_gaussian_noise_mean, slider_gaussian_noise_std


@app.cell(hide_code=True)
def _(
    Image,
    K,
    file,
    io,
    mo,
    slider_gaussian_noise_mean,
    slider_gaussian_noise_std,
    to_tensor,
    transforms,
):
    _aug = K.AugmentationSequential(
        K.RandomGaussianNoise(
            mean=slider_gaussian_noise_mean.value,
            std=slider_gaussian_noise_std.value,
            p=1.0,
            keepdim=True,
            same_on_batch=True,
        ),
        data_keys=["input"],  # Just to define the future input here.
        same_on_batch=False,
    )

    if file.value is not None:
        _image = Image.open(io.BytesIO(file.value[0].contents))

        # Define the transformation to convert the image to a tensor
        _to_tensor = transforms.ToTensor()

        # Apply the transformation
        _tensor_image = to_tensor(_image)

        _aug_image = _aug(_tensor_image)

        _aug_image_pil = transforms.ToPILImage()(_aug_image)

    mo.hstack(
        [
            mo.image(_image, width=400, height=250),
            mo.image(_aug_image_pil, width=400, height=250),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Contrast""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Modify image contrast.
    Note: `K.RandomContrast` takes in a tuple to specify the range of contrast to apply, and kornia randomly samples from the range. For demo purposes, we use fix the max and min to be the same.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    slider_contrast = mo.ui.slider(0, 10.0, value=0, step=0.01, label="Contrast")
    mo.vstack([slider_contrast])
    return (slider_contrast,)


@app.cell(hide_code=True)
def _(Image, K, file, io, mo, slider_contrast, to_tensor, transforms):
    _aug = K.AugmentationSequential(
        K.RandomContrast(
            contrast=(slider_contrast.value, slider_contrast.value),
            p=1.0,
            keepdim=True,
            same_on_batch=True,
        ),
        data_keys=["input"],  # Just to define the future input here.
        same_on_batch=False,
    )

    if file.value is not None:
        _image = Image.open(io.BytesIO(file.value[0].contents))

        # Define the transformation to convert the image to a tensor
        _to_tensor = transforms.ToTensor()

        # Apply the transformation
        _tensor_image = to_tensor(_image)

        _aug_image = _aug(_tensor_image)

        _aug_image_pil = transforms.ToPILImage()(_aug_image)

    mo.hstack(
        [
            mo.image(_image, width=400, height=250),
            mo.image(_aug_image_pil, width=400, height=250),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Brightness""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Modify image brightness.
    Note: `K.RandomBrightness` takes in a tuple to specify the range of brightness to apply, and kornia randomly samples from the range. For demo purposes, we use fix the max and min to be the same.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    slider_brightness = mo.ui.slider(0, 2.0, value=0, step=0.01, label="Brightness")
    mo.vstack([slider_brightness])
    return (slider_brightness,)


@app.cell(hide_code=True)
def _(Image, K, file, io, mo, slider_brightness, to_tensor, transforms):
    _aug = K.AugmentationSequential(
        K.RandomBrightness(
            brightness=(slider_brightness.value, slider_brightness.value),
            p=1.0,
            keepdim=True,
            same_on_batch=True,
        ),
        data_keys=["input"],  # Just to define the future input here.
        same_on_batch=False,
    )

    if file.value is not None:
        _image = Image.open(io.BytesIO(file.value[0].contents))

        # Define the transformation to convert the image to a tensor
        _to_tensor = transforms.ToTensor()

        # Apply the transformation
        _tensor_image = to_tensor(_image)

        _aug_image = _aug(_tensor_image)

        _aug_image_pil = transforms.ToPILImage()(_aug_image)

    mo.hstack(
        [
            mo.image(_image, width=400, height=250),
            mo.image(_aug_image_pil, width=400, height=250),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torchvision import transforms
    import kornia.augmentation as K
    from PIL import Image
    import io

    return Image, K, io, mo, torch, transforms


if __name__ == "__main__":
    app.run()
