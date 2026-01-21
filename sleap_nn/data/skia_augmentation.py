"""Skia-based augmentation functions that operate on uint8 tensors.

This module provides augmentation functions using skia-python that:
1. Match the exact API of sleap_nn.data.augmentation
2. Operate on uint8 tensors throughout (avoiding float32 conversions)
3. Provide ~1.5x faster augmentation compared to Kornia

Usage:
    from sleap_nn.data.skia_augmentation import (
        apply_intensity_augmentation_skia,
        apply_geometric_augmentation_skia,
    )

    # Apply augmentations (uint8 in, uint8 out)
    image, instances = apply_intensity_augmentation_skia(image, instances, **config)
    image, instances = apply_geometric_augmentation_skia(image, instances, **config)
"""

from typing import Optional, Tuple
import numpy as np
import torch
import skia


def apply_intensity_augmentation_skia(
    image: torch.Tensor,
    instances: torch.Tensor,
    uniform_noise_min: float = 0.0,
    uniform_noise_max: float = 0.04,
    uniform_noise_p: float = 0.0,
    gaussian_noise_mean: float = 0.02,
    gaussian_noise_std: float = 0.004,
    gaussian_noise_p: float = 0.0,
    contrast_min: float = 0.5,
    contrast_max: float = 2.0,
    contrast_p: float = 0.0,
    brightness_min: float = 1.0,
    brightness_max: float = 1.0,
    brightness_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply intensity augmentations on uint8 image tensor.

    Matches API of sleap_nn.data.augmentation.apply_intensity_augmentation.

    Args:
        image: Input tensor of shape (1, C, H, W) with dtype uint8 or float32.
        instances: Keypoints tensor (not modified, just passed through).
        uniform_noise_min: Minimum uniform noise (0-1 scale, maps to 0-255).
        uniform_noise_max: Maximum uniform noise (0-1 scale).
        uniform_noise_p: Probability of uniform noise.
        gaussian_noise_mean: Gaussian noise mean (0-1 scale).
        gaussian_noise_std: Gaussian noise std (0-1 scale).
        gaussian_noise_p: Probability of Gaussian noise.
        contrast_min: Minimum contrast factor.
        contrast_max: Maximum contrast factor.
        contrast_p: Probability of contrast adjustment.
        brightness_min: Minimum brightness factor.
        brightness_max: Maximum brightness factor.
        brightness_p: Probability of brightness adjustment.

    Returns:
        Tuple of (augmented_image, instances). Image dtype matches input.
    """
    # Convert to numpy for Skia processing
    is_float = image.dtype == torch.float32
    if is_float:
        img_np = (image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    else:
        img_np = image[0].permute(1, 2, 0).numpy()

    result = img_np.copy()

    # Apply uniform noise (in uint8 space)
    if uniform_noise_p > 0 and np.random.random() < uniform_noise_p:
        noise = np.random.randint(
            int(uniform_noise_min * 255),
            int(uniform_noise_max * 255) + 1,
            img_np.shape,
            dtype=np.int16,
        )
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Apply Gaussian noise (in uint8 space)
    if gaussian_noise_p > 0 and np.random.random() < gaussian_noise_p:
        noise = np.random.normal(
            gaussian_noise_mean * 255, gaussian_noise_std * 255, img_np.shape
        ).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Apply contrast using lookup table (pure uint8)
    if contrast_p > 0 and np.random.random() < contrast_p:
        factor = np.random.uniform(contrast_min, contrast_max)
        lut = np.arange(256, dtype=np.float32)
        lut = np.clip((lut - 127.5) * factor + 127.5, 0, 255).astype(np.uint8)
        result = lut[result]

    # Apply brightness using lookup table (pure uint8)
    if brightness_p > 0 and np.random.random() < brightness_p:
        factor = np.random.uniform(brightness_min, brightness_max)
        lut = np.arange(256, dtype=np.float32)
        lut = np.clip(lut * factor, 0, 255).astype(np.uint8)
        result = lut[result]

    # Convert back to tensor
    result_tensor = torch.from_numpy(result).permute(2, 0, 1).unsqueeze(0)
    if is_float:
        result_tensor = result_tensor.float() / 255.0

    return result_tensor, instances


def apply_geometric_augmentation_skia(
    image: torch.Tensor,
    instances: torch.Tensor,
    rotation_min: float = -15.0,
    rotation_max: float = 15.0,
    rotation_p: Optional[float] = None,
    scale_min: float = 0.9,
    scale_max: float = 1.1,
    scale_p: Optional[float] = None,
    translate_width: float = 0.02,
    translate_height: float = 0.02,
    translate_p: Optional[float] = None,
    affine_p: float = 0.0,
    erase_scale_min: float = 0.0001,
    erase_scale_max: float = 0.01,
    erase_ratio_min: float = 1.0,
    erase_ratio_max: float = 1.0,
    erase_p: float = 0.0,
    mixup_lambda_min: float = 0.01,
    mixup_lambda_max: float = 0.05,
    mixup_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply geometric augmentations using Skia.

    Matches API of sleap_nn.data.augmentation.apply_geometric_augmentation.

    Args:
        image: Input tensor of shape (1, C, H, W) with dtype uint8 or float32.
        instances: Keypoints tensor of shape (1, n_instances, n_nodes, 2) or (1, n_nodes, 2).
        rotation_min: Minimum rotation angle in degrees.
        rotation_max: Maximum rotation angle in degrees.
        rotation_p: Probability of rotation (independent). None = use affine_p.
        scale_min: Minimum scale factor.
        scale_max: Maximum scale factor.
        scale_p: Probability of scaling (independent). None = use affine_p.
        translate_width: Max horizontal translation as fraction of width.
        translate_height: Max vertical translation as fraction of height.
        translate_p: Probability of translation (independent). None = use affine_p.
        affine_p: Probability of bundled affine transform.
        erase_scale_min: Min proportion of image to erase.
        erase_scale_max: Max proportion of image to erase.
        erase_ratio_min: Min aspect ratio of erased area.
        erase_ratio_max: Max aspect ratio of erased area.
        erase_p: Probability of random erasing.
        mixup_lambda_min: Min mixup strength (not implemented).
        mixup_lambda_max: Max mixup strength (not implemented).
        mixup_p: Probability of mixup (not implemented).

    Returns:
        Tuple of (augmented_image, augmented_instances). Image dtype matches input.
    """
    # Convert to numpy for Skia processing
    is_float = image.dtype == torch.float32
    if is_float:
        img_np = (image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    else:
        img_np = image[0].permute(1, 2, 0).numpy().copy()

    h, w = img_np.shape[:2]
    cx, cy = w / 2, h / 2

    # Build transformation matrix
    matrix = skia.Matrix()
    has_transform = False

    use_independent = (
        rotation_p is not None or scale_p is not None or translate_p is not None
    )

    if use_independent:
        if (
            rotation_p is not None
            and rotation_p > 0
            and np.random.random() < rotation_p
        ):
            angle = np.random.uniform(rotation_min, rotation_max)
            rot_matrix = skia.Matrix()
            rot_matrix.setRotate(angle, cx, cy)
            matrix = matrix.preConcat(rot_matrix)
            has_transform = True

        if scale_p is not None and scale_p > 0 and np.random.random() < scale_p:
            scale = np.random.uniform(scale_min, scale_max)
            scale_matrix = skia.Matrix()
            scale_matrix.setScale(scale, scale, cx, cy)
            matrix = matrix.preConcat(scale_matrix)
            has_transform = True

        if (
            translate_p is not None
            and translate_p > 0
            and np.random.random() < translate_p
        ):
            tx = np.random.uniform(-translate_width, translate_width) * w
            ty = np.random.uniform(-translate_height, translate_height) * h
            trans_matrix = skia.Matrix()
            trans_matrix.setTranslate(tx, ty)
            matrix = matrix.preConcat(trans_matrix)
            has_transform = True

    elif affine_p > 0 and np.random.random() < affine_p:
        angle = np.random.uniform(rotation_min, rotation_max)
        scale = np.random.uniform(scale_min, scale_max)
        tx = np.random.uniform(-translate_width, translate_width) * w
        ty = np.random.uniform(-translate_height, translate_height) * h

        matrix.setRotate(angle, cx, cy)
        matrix.preScale(scale, scale, cx, cy)
        matrix.preTranslate(tx, ty)
        has_transform = True

    # Apply geometric transform
    if has_transform:
        img_np = _transform_image_skia(img_np, matrix)
        instances = _transform_keypoints_tensor(instances, matrix)

    # Apply random erasing
    if erase_p > 0 and np.random.random() < erase_p:
        img_np = _apply_random_erase(
            img_np, erase_scale_min, erase_scale_max, erase_ratio_min, erase_ratio_max
        )

    # Convert back to tensor
    result_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    if is_float:
        result_tensor = result_tensor.float() / 255.0

    return result_tensor, instances


def _transform_image_skia(image: np.ndarray, matrix: skia.Matrix) -> np.ndarray:
    """Transform image using Skia matrix (uint8 in, uint8 out)."""
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1

    # Skia needs RGBA
    if channels == 1:
        image_rgba = np.stack(
            [image.squeeze()] * 3 + [np.full((h, w), 255, dtype=np.uint8)], axis=-1
        )
    elif channels == 3:
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        image_rgba = np.concatenate([image, alpha], axis=-1)
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
    return result[:, :, :channels]


def _transform_keypoints_tensor(
    keypoints: torch.Tensor, matrix: skia.Matrix
) -> torch.Tensor:
    """Transform keypoints tensor using Skia matrix."""
    if keypoints.numel() == 0:
        return keypoints

    original_shape = keypoints.shape
    flat = keypoints.reshape(-1, 2).numpy()

    # Handle NaN values
    valid_mask = ~np.isnan(flat).any(axis=1)
    transformed = flat.copy()

    if valid_mask.any():
        valid_pts = flat[valid_mask]
        skia_pts = [skia.Point(float(p[0]), float(p[1])) for p in valid_pts]
        mapped = matrix.mapPoints(skia_pts)
        transformed[valid_mask] = np.array([[p.x(), p.y()] for p in mapped])

    return torch.from_numpy(transformed.reshape(original_shape).astype(np.float32))


def _apply_random_erase(
    image: np.ndarray,
    scale_min: float,
    scale_max: float,
    ratio_min: float,
    ratio_max: float,
) -> np.ndarray:
    """Apply random erasing (uint8)."""
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


def crop_and_resize_skia(
    image: torch.Tensor,
    boxes: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    """Crop and resize image regions using Skia.

    Replacement for kornia.geometry.transform.crop_and_resize.

    Args:
        image: Input tensor of shape (1, C, H, W).
        boxes: Bounding boxes tensor of shape (1, 4, 2) with corners:
            [top-left, top-right, bottom-right, bottom-left].
        size: Output size (height, width).

    Returns:
        Cropped and resized tensor of shape (1, C, out_h, out_w).
    """
    is_float = image.dtype == torch.float32
    if is_float:
        img_np = (image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    else:
        img_np = image[0].permute(1, 2, 0).numpy()

    h, w = img_np.shape[:2]
    out_h, out_w = size
    channels = img_np.shape[2] if img_np.ndim == 3 else 1

    # Get box coordinates (top-left and bottom-right)
    box = boxes[0].numpy()  # (4, 2)
    x1, y1 = box[0]  # top-left
    x2, y2 = box[2]  # bottom-right

    crop_w = x2 - x1
    crop_h = y2 - y1

    # Create transformation matrix
    matrix = skia.Matrix()
    scale_x = out_w / crop_w
    scale_y = out_h / crop_h
    matrix.setScale(scale_x, scale_y)
    matrix.preTranslate(-x1, -y1)

    # Skia needs RGBA
    if channels == 1:
        image_rgba = np.stack(
            [img_np.squeeze()] * 3 + [np.full((h, w), 255, dtype=np.uint8)], axis=-1
        )
    elif channels == 3:
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        image_rgba = np.concatenate([img_np, alpha], axis=-1)
    else:
        raise ValueError(f"Unsupported channels: {channels}")

    image_rgba = np.ascontiguousarray(image_rgba, dtype=np.uint8)
    skia_image = skia.Image.fromarray(
        image_rgba, colorType=skia.ColorType.kRGBA_8888_ColorType
    )

    surface = skia.Surface(out_w, out_h)
    canvas = surface.getCanvas()
    canvas.clear(skia.Color4f(0, 0, 0, 1))
    canvas.setMatrix(matrix)

    paint = skia.Paint()
    paint.setAntiAlias(True)
    sampling = skia.SamplingOptions(skia.FilterMode.kLinear)
    canvas.drawImage(skia_image, 0, 0, sampling, paint)

    result = surface.makeImageSnapshot().toarray()

    if channels == 1:
        result = result[:, :, 0:1]
    else:
        result = result[:, :, :channels]

    result_tensor = torch.from_numpy(result).permute(2, 0, 1).unsqueeze(0)
    if is_float:
        result_tensor = result_tensor.float() / 255.0

    return result_tensor
