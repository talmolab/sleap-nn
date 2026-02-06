# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib==3.9.4",
#     "numpy",
#     "pillow==11.3.0",
#     "seaborn==0.13.2",
#     "skia-python",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing Data Augmentations with SLEAP-NN
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _**Note**_:
    This notebook executes automatically; there is no need to run individual cells, as all interactions are managed through the provided UI elements (sliders, buttons, etc.). Just upload a sample image and click `"Start exploring augmentations!"` button!. If the run button in the bottom left corner is highlighted in yellow, click on the run button to start!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook provides a visual demo of the various data augmentations available in `sleap-nn`. Users can interactively explore two broad classes of augmentations—intensity-based (e.g., brightness, contrast, noise) and geometric (e.g., rotation, scaling, translation)—by adjusting hyperparameters and observing their effects on sample images. This helps in understanding how different transformations impact the data and can guide effective augmentation strategies for training robust models.

    SLEAP-NN uses **Skia** for augmentations, which operates on uint8 images and is ~4x faster than alternatives like Kornia.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Upload a sample image:
    """)
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
    mo.md(r"""
    ## Geometric Augmentations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Rotation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Rotates the image around its center using Skia's matrix transformation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    slider_rotation_angle = mo.ui.slider(
        -180, 180, value=0, step=1.0, label="Rotation angle (degrees)"
    )
    mo.vstack([slider_rotation_angle])
    return (slider_rotation_angle,)


@app.cell(hide_code=True)
def _(
    Image,
    file,
    io,
    mo,
    np,
    skia,
    slider_rotation_angle,
    transform_image_skia,
):
    if file.value is not None:
        _image = Image.open(io.BytesIO(file.value[0].contents))
        _img_np = np.array(_image)

        _h, _w = _img_np.shape[:2]
        _cx, _cy = _w / 2, _h / 2

        _matrix = skia.Matrix()
        _matrix.setRotate(slider_rotation_angle.value, _cx, _cy)

        _aug_np = transform_image_skia(_img_np, _matrix)
        _aug_image = Image.fromarray(_aug_np)

    mo.hstack(
        [
            mo.image(_image, width=400, height=250),
            mo.image(_aug_image, width=400, height=250),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scale
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Scales the image around its center without changing the output image size.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    slider_scale = mo.ui.slider(
        0.1, 3.0, value=1.0, step=0.01, label="Scale factor"
    )
    mo.vstack([slider_scale])
    return (slider_scale,)


@app.cell(hide_code=True)
def _(Image, file, io, mo, np, skia, slider_scale, transform_image_skia):
    if file.value is not None:
        _image = Image.open(io.BytesIO(file.value[0].contents))
        _img_np = np.array(_image)

        _h, _w = _img_np.shape[:2]
        _cx, _cy = _w / 2, _h / 2

        _matrix = skia.Matrix()
        _matrix.setScale(slider_scale.value, slider_scale.value, _cx, _cy)

        _aug_np = transform_image_skia(_img_np, _matrix)
        _aug_image = Image.fromarray(_aug_np)

    mo.hstack(
        [
            mo.image(_image, width=400, height=250),
            mo.image(_aug_image, width=400, height=250),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Translate
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Shifts the image horizontally and vertically by a fraction of its width and height.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    slider_translate_width = mo.ui.slider(
        -0.5, 0.5, value=0.0, step=0.01, label="Translate width (fraction)"
    )
    slider_translate_height = mo.ui.slider(
        -0.5, 0.5, value=0.0, step=0.01, label="Translate height (fraction)"
    )

    mo.vstack([slider_translate_width, slider_translate_height])
    return slider_translate_height, slider_translate_width


@app.cell(hide_code=True)
def _(
    Image,
    file,
    io,
    mo,
    np,
    skia,
    slider_translate_height,
    slider_translate_width,
    transform_image_skia,
):
    if file.value is not None:
        _image = Image.open(io.BytesIO(file.value[0].contents))
        _img_np = np.array(_image)

        _h, _w = _img_np.shape[:2]
        _tx = slider_translate_width.value * _w
        _ty = slider_translate_height.value * _h

        _matrix = skia.Matrix()
        _matrix.setTranslate(_tx, _ty)

        _aug_np = transform_image_skia(_img_np, _matrix)
        _aug_image = Image.fromarray(_aug_np)

    mo.hstack(
        [
            mo.image(_image, width=400, height=250),
            mo.image(_aug_image, width=400, height=250),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Erase
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Erases a random rectangular region in the image by replacing its pixels with a constant value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    slider_erase_scale = mo.ui.slider(
        0.01, 0.5, value=0.05, step=0.01, label="Proportion of area to erase"
    )
    slider_erase_ratio = mo.ui.slider(
        0.5, 2.0, value=1.0, step=0.1, label="Aspect ratio of erased area"
    )

    mo.vstack([slider_erase_scale, slider_erase_ratio])
    return slider_erase_ratio, slider_erase_scale


@app.cell(hide_code=True)
def _(
    Image,
    apply_random_erase,
    file,
    io,
    mo,
    np,
    slider_erase_ratio,
    slider_erase_scale,
):
    if file.value is not None:
        _image = Image.open(io.BytesIO(file.value[0].contents))
        _img_np = np.array(_image)

        _aug_np = apply_random_erase(
            _img_np,
            slider_erase_scale.value,
            slider_erase_scale.value,
            slider_erase_ratio.value,
            slider_erase_ratio.value,
        )
        _aug_image = Image.fromarray(_aug_np)

    mo.hstack(
        [
            mo.image(_image, width=400, height=250),
            mo.image(_aug_image, width=400, height=250),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Intensity Augmentations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Uniform Noise
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Adds pixel-wise uniform noise to the image. The noise is applied in uint8 space (0-255).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    slider_uniform_noise = mo.ui.slider(
        0, 100, value=0, step=1, label="Uniform noise (max intensity 0-255)"
    )
    mo.vstack([slider_uniform_noise])
    return (slider_uniform_noise,)


@app.cell(hide_code=True)
def _(Image, file, io, mo, np, slider_uniform_noise):
    if file.value is not None:
        _image = Image.open(io.BytesIO(file.value[0].contents))
        _img_np = np.array(_image)

        # Apply uniform noise in uint8 space
        _noise = np.random.randint(
            -slider_uniform_noise.value,
            slider_uniform_noise.value + 1,
            _img_np.shape,
            dtype=np.int16,
        )
        _aug_np = np.clip(_img_np.astype(np.int16) + _noise, 0, 255).astype(np.uint8)
        _aug_image = Image.fromarray(_aug_np)

    mo.hstack(
        [
            mo.image(_image, width=400, height=250),
            mo.image(_aug_image, width=400, height=250),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Gaussian Noise
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Adds pixel-wise Gaussian noise with a specified mean and standard deviation to the image.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    slider_gaussian_noise_mean = mo.ui.slider(
        0, 50, value=0, step=1, label="Gaussian noise mean"
    )
    slider_gaussian_noise_std = mo.ui.slider(
        0, 50, value=0, step=1, label="Gaussian noise std"
    )
    mo.vstack([slider_gaussian_noise_mean, slider_gaussian_noise_std])
    return slider_gaussian_noise_mean, slider_gaussian_noise_std


@app.cell(hide_code=True)
def _(
    Image,
    file,
    io,
    mo,
    np,
    slider_gaussian_noise_mean,
    slider_gaussian_noise_std,
):
    if file.value is not None:
        _image = Image.open(io.BytesIO(file.value[0].contents))
        _img_np = np.array(_image)

        # Apply Gaussian noise in uint8 space
        _noise = np.random.normal(
            slider_gaussian_noise_mean.value,
            slider_gaussian_noise_std.value,
            _img_np.shape,
        ).astype(np.int16)
        _aug_np = np.clip(_img_np.astype(np.int16) + _noise, 0, 255).astype(np.uint8)
        _aug_image = Image.fromarray(_aug_np)

    mo.hstack(
        [
            mo.image(_image, width=400, height=250),
            mo.image(_aug_image, width=400, height=250),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Contrast
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Adjusts image contrast using a lookup table (LUT). This operates entirely in uint8 space for efficiency.

    - Values < 1.0 reduce contrast
    - Values > 1.0 increase contrast
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    slider_contrast = mo.ui.slider(0.1, 3.0, value=1.0, step=0.01, label="Contrast factor")
    mo.vstack([slider_contrast])
    return (slider_contrast,)


@app.cell(hide_code=True)
def _(Image, file, io, mo, np, slider_contrast):
    if file.value is not None:
        _image = Image.open(io.BytesIO(file.value[0].contents))
        _img_np = np.array(_image)

        # Apply contrast using lookup table (pure uint8)
        _lut = np.arange(256, dtype=np.float32)
        _lut = np.clip((_lut - 127.5) * slider_contrast.value + 127.5, 0, 255).astype(np.uint8)
        _aug_np = _lut[_img_np]
        _aug_image = Image.fromarray(_aug_np)

    mo.hstack(
        [
            mo.image(_image, width=400, height=250),
            mo.image(_aug_image, width=400, height=250),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Brightness
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Adjusts image brightness using a lookup table (LUT). This operates entirely in uint8 space.

    - Values < 1.0 darken the image
    - Values > 1.0 brighten the image
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    slider_brightness = mo.ui.slider(0.1, 3.0, value=1.0, step=0.01, label="Brightness factor")
    mo.vstack([slider_brightness])
    return (slider_brightness,)


@app.cell(hide_code=True)
def _(Image, file, io, mo, np, slider_brightness):
    if file.value is not None:
        _image = Image.open(io.BytesIO(file.value[0].contents))
        _img_np = np.array(_image)

        # Apply brightness using lookup table (pure uint8)
        _lut = np.arange(256, dtype=np.float32)
        _lut = np.clip(_lut * slider_brightness.value, 0, 255).astype(np.uint8)
        _aug_np = _lut[_img_np]
        _aug_image = Image.fromarray(_aug_np)

    mo.hstack(
        [
            mo.image(_image, width=400, height=250),
            mo.image(_aug_image, width=400, height=250),
        ]
    )
    return


@app.cell(hide_code=True)
def _(np, skia):
    def transform_image_skia(image: np.ndarray, matrix: skia.Matrix) -> np.ndarray:
        """Transform image using Skia matrix (uint8 in, uint8 out).

        This is the same approach used in sleap-nn for fast augmentations.
        """
        h, w = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1

        # Skia needs RGBA format
        if channels == 1:
            image_rgba = np.stack(
                [image.squeeze()] * 3 + [np.full((h, w), 255, dtype=np.uint8)], axis=-1
            )
        elif channels == 3:
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            image_rgba = np.concatenate([image, alpha], axis=-1)
        elif channels == 4:
            image_rgba = image
        else:
            raise ValueError(f"Unsupported channels: {channels}")

        image_rgba = np.ascontiguousarray(image_rgba, dtype=np.uint8)
        skia_image = skia.Image.fromarray(
            image_rgba, colorType=skia.ColorType.kRGBA_8888_ColorType
        )

        surface = skia.Surface(w, h)
        canvas = surface.getCanvas()
        canvas.clear(skia.Color4f(0, 0, 0, 1))
        canvas.setMatrix(matrix)

        paint = skia.Paint()
        paint.setAntiAlias(True)
        sampling = skia.SamplingOptions(skia.FilterMode.kLinear)
        canvas.drawImage(skia_image, 0, 0, sampling, paint)

        result = surface.makeImageSnapshot().toarray()

        if channels == 1:
            return result[:, :, 0:1]
        elif channels == 3:
            return result[:, :, :3]
        return result
    return (transform_image_skia,)


@app.cell(hide_code=True)
def _(np):
    def apply_random_erase(
        image: np.ndarray,
        scale_min: float,
        scale_max: float,
        ratio_min: float,
        ratio_max: float,
    ) -> np.ndarray:
        """Apply random erasing (uint8).

        This matches the implementation in sleap-nn.
        """
        h, w = image.shape[:2]
        area = h * w

        erase_area = np.random.uniform(scale_min, scale_max) * area
        aspect_ratio = np.random.uniform(ratio_min, ratio_max)

        erase_h = int(np.sqrt(erase_area * aspect_ratio))
        erase_w = int(np.sqrt(erase_area / aspect_ratio))

        if erase_h >= h or erase_w >= w:
            return image

        y = np.random.randint(0, h - erase_h)
        x = np.random.randint(0, w - erase_w)

        result = image.copy()
        channels = image.shape[2] if image.ndim == 3 else 1
        fill = np.random.randint(0, 256, size=(channels,), dtype=np.uint8)
        result[y : y + erase_h, x : x + erase_w] = fill

        return result
    return (apply_random_erase,)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import skia
    from PIL import Image
    import io
    return Image, io, mo, np, skia


if __name__ == "__main__":
    app.run()
