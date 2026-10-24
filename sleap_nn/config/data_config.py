"""Serializable configuration classes for specifying all data configuration parameters.

These configuration classes are intended to specify all
the parameters required to initialize the data config.
"""

from attrs import define, field, validators
from omegaconf import MISSING
from typing import Optional, Tuple, Any, Dict, List, Union
from loguru import logger
import sleap_io as sio
import yaml
from sleap_io.io.skeleton import SkeletonDecoder, SkeletonYAMLEncoder


def validate_proportion(instance, attribute, value):
    """General Proportion Validation.

    Ensures all proportions are a 0<=float<=1.0
    """
    if not (0.0 <= value <= 1.0):
        message = f"{attribute.name} must be between 0.0 and 1.0, got {value}"
        logger.error(message)
        raise ValueError(message)


def validate_optional_positive_int(instance, attribute, value):
    """Allow None or a strictly-positive int.

    Used for `tile_size`, `tile_batch_size`, `samples_per_frame`, `steps_per_epoch`.
    """
    if value is None:
        return
    if not (isinstance(value, int) and value > 0):
        message = f"{attribute.name} must be a positive integer or None, got {value!r}"
        logger.error(message)
        raise ValueError(message)


def validate_optional_nonneg_int(instance, attribute, value):
    """Allow None or a non-negative int (e.g. `overlap`)."""
    if value is None:
        return
    if not (isinstance(value, int) and value >= 0):
        message = (
            f"{attribute.name} must be a non-negative integer or None, got {value!r}"
        )
        logger.error(message)
        raise ValueError(message)


def validate_fg_fraction(instance, attribute, value):
    """nnU-Net foreground oversample fraction: 0.0 <= value < 1.0 (never 1.0)."""
    if not (0.0 <= value < 1.0):
        message = f"{attribute.name} must be in [0.0, 1.0) (never 1.0), got {value}"
        logger.error(message)
        raise ValueError(message)


def validate_blend(instance, attribute, value):
    """Ensure the tiling blend window is a supported mode."""
    if value not in ("gaussian", "pyramid", "constant"):
        message = (
            f"blend must be one of ['gaussian', 'pyramid', 'constant'], got {value!r}"
        )
        logger.error(message)
        raise ValueError(message)


def validate_accumulator_device(instance, attribute, value):
    """Ensure the tiling accumulator device is a supported value."""
    if value not in ("auto", "cpu", "cuda"):
        message = (
            "accumulator_device must be one of ['auto', 'cpu', 'cuda'], "
            f"got {value!r}"
        )
        logger.error(message)
        raise ValueError(message)


def validate_sampling(instance, attribute, value):
    """Ensure the tiling sampling strategy is a supported value."""
    if value not in ("foreground", "grid"):
        message = f"sampling must be one of ['foreground', 'grid'], got {value!r}"
        logger.error(message)
        raise ValueError(message)


@define
class TilingConfig:
    """Configuration of tiled training/inference (Phase 0/A).

    Explicit opt-in only (`enabled=True`). Square tiles, constant-zero padding.
    Geometry (`tile_size`, `overlap`) is auto-sized from labels x backbone margin at
    train setup and written back into this config (persisted in training_config.yaml),
    then read + parity-checked at inference. Unsupported with pretrained backbones and
    with ClassVectorsHead / multi_class_topdown models (see `check_tiling`).

    Attributes:
        enabled: (bool) If `True`, engage tiled training/inference. Explicit opt-in; there is no engagement heuristic. *Default*: `False`.
        tile_size: (int) SQUARE tile side length in pixels. If `None`, auto-computed at train setup as a multiple of both the effective backbone max_stride and the head output_stride. *Default*: `None`.
        overlap: (int) Tile overlap in pixels. If `None`, auto-sized from object extent / confmap sigma plus a fixed backbone context margin; a conservative default is used (with a warning) when labels are sparse. An explicit value always overrides. *Default*: `None`.
        min_overlap_fraction: (float) Minimum overlap as a fraction of `tile_size`, enforced (overlap raised if needed) in `check_tiling`. Must be in [0.0, 1.0]. *Default*: `0.25`.
        blend: (str) Merge window for stitching tile predictions. One of ['gaussian', 'pyramid', 'constant']. *Default*: `"gaussian"`.
        sigma_scale: (float) Per-axis std of the Gaussian importance window as a fraction of the tile side (std = sigma_scale * tile). Must be in (0.0, 1.0]. *Default*: `0.125`.
        tile_batch_size: (int) Number of tiles forwarded per backend call at inference. Manual knob; a conservative default is used when `None` (no auto-tuner). *Default*: `None`.
        accumulator_device: (str) Device for the per-frame ACC/CNT merge buffers. One of ['auto', 'cpu', 'cuda']; 'auto' predicts placement and falls back to CPU on OOM. *Default*: `"auto"`.
        cpu_thresh: (float) Spill ACC/CNT to CPU when the buffers would exceed this fraction of free GPU memory. Must be in [0.0, 1.0]. *Default*: `0.40`.
        sampling: (str) Tile sampling strategy. 'foreground' (train, object-aware) or 'grid' (val/debug, full-coverage). *Default*: `"foreground"`.
        tile_fg_fraction: (float) Fraction of sampled train tiles forced to contain an object (nnU-Net oversampling). Must be in [0.0, 1.0) (never 1.0). *Default*: `0.5`.
        samples_per_frame: (int) Number of tiles emitted per decoded frame as a worker-aligned block. If `None`, a conservative default is used. *Default*: `None`.
        center_jitter: (float) Foreground-tile center jitter as a fraction of tile/2. Must be in [0.0, 1.0]. *Default*: `0.5`.
        min_visible_keypoints: (int) Keep an instance in a tile only if at least this many of its keypoints fall inside the tile. Must be >= 0. *Default*: `1`.
        steps_per_epoch: (int) Decouples the TRAIN epoch length from tile count. Validation is always full-coverage (not decoupled). If `None`, derived from the effective sample count. *Default*: `None`.
        full_frame_pass: (bool) Full-image mixing pass. DECLARED but INERT in Phase 0/A (wired in Phase C). *Default*: `False`.
    """

    enabled: bool = False
    tile_size: Optional[int] = field(
        default=None, validator=validate_optional_positive_int
    )
    overlap: Optional[int] = field(default=None, validator=validate_optional_nonneg_int)
    min_overlap_fraction: float = field(default=0.25, validator=validate_proportion)
    blend: str = field(default="gaussian", validator=validate_blend)
    sigma_scale: float = field(
        default=0.125, validator=[validators.gt(0), validators.le(1)]
    )
    tile_batch_size: Optional[int] = field(
        default=None, validator=validate_optional_positive_int
    )
    accumulator_device: str = field(
        default="auto", validator=validate_accumulator_device
    )
    cpu_thresh: float = field(default=0.40, validator=validate_proportion)
    sampling: str = field(default="foreground", validator=validate_sampling)
    tile_fg_fraction: float = field(default=0.5, validator=validate_fg_fraction)
    samples_per_frame: Optional[int] = field(
        default=None, validator=validate_optional_positive_int
    )
    center_jitter: float = field(default=0.5, validator=validate_proportion)
    min_visible_keypoints: int = field(default=1, validator=validators.ge(0))
    steps_per_epoch: Optional[int] = field(
        default=None, validator=validate_optional_positive_int
    )
    full_frame_pass: bool = False


@define
class PreprocessingConfig:
    """Configuration of Preprocessing.

    Attributes:
        ensure_rgb: (bool) True if the input image should have 3 channels (RGB image). If input has only one channel when this is set to `True`, then the images from single-channel is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. *Default*: `False`.
        ensure_grayscale: (bool) True if the input image should only have a single channel. If input has three channels (RGB) and this is set to True, then we convert the image to grayscale (single-channel) image. If the source image has only one channel and this is set to False, then we retain the single channel input. *Default*: `False`.
        max_height: (int) Maximum height the original image should be resized and padded to. If not provided, the original image size will be retained. *Default*: `None`.
        max_width: (int) Maximum width the original image should be resized and padded to. If not provided, the original image size will be retained. *Default*: `None`.
        scale: (float) Factor to resize the image dimensions by, specified as a float. *Default*: `1.0`.
        crop_size: (int or [height, width]) Crop size of each instance for centered-instance model. A `[height, width]` pair gives non-square crops (`crop_mode="bbox"` only; e.g. `[256, 192]` for people). If `None`, this would be automatically computed based on the largest instance in the `sio.Labels` file.
            If `scale` is provided, then the cropped image will be resized according to `scale`.*Default*: `None`.
        min_crop_size: (int) Minimum crop size to be used if `crop_size` is `None`. *Default*: `100`.
        crop_padding: (int) Padding in pixels to add around the instance bounding box when computing crop size.
            If `None`, padding is auto-computed based on augmentation settings (rotation/scale).
            Only used when `crop_size` is `None`. *Default*: `None`.
        crop_mode: (str) How a centered-instance crop is cut from the frame (the
            `centered_instance` and `multi_class_topdown` model types). `fixed` cuts a
            `crop_size` x `crop_size` window centered on the instance centroid, so the
            animal keeps its pixel size. `bbox` cuts the instance's bounding box (padded
            by `crop_bbox_padding` and shaped by `crop_bbox_aspect`) and resizes it to
            `crop_size` x `crop_size`, so every animal fills the crop the same way
            whatever its size. Inference then needs boxes: a `bbox_centernet` /
            `bbox_ssd` detector, or ground-truth boxes. The training box of an instance
            is derived like a detector's ground truth (`bbox_source`, `bbox_padding`),
            so pair the model with a detector trained with the same two values.
            *Default*: `"fixed"`.
        crop_bbox_aspect: (str) For `crop_mode="bbox"`: `square` expands the padded box
            to a square on its longest side (one scale for both axes; the frame around
            the box fills the margin), `letterbox` does the same but zeroes the pixels
            outside the padded box, and `stretch` resizes the padded box to the square
            crop on each axis separately (distorts the animal's aspect ratio).
            *Default*: `"square"`.
        crop_bbox_padding: (float) For `crop_mode="bbox"`: margin added on every side
            of the box, as a fraction of its longest side. *Default*: `0.1`.
        crop_bbox_min_size: (float) For `crop_mode="bbox"`: smallest crop window side,
            in size-matched frame pixels, which caps how much a tiny or partial box is
            magnified. If `None`, it is set at training time to a quarter of the median
            training window side and saved into the config. *Default*: `None`.
        crop_pipeline: (str) How `centered_instance` training crops are produced.
            `legacy` decodes the frame, size-matches it to
            `max_height`/`max_width`, cuts an oversized pre-crop, augments it and
            re-crops it. `warp` (requires `crop_mode="bbox"`) reads the frame and cuts
            the augmented box-scaled crop with ONE affine warp from the full frame
            (`sleap_nn.data.topdown_warp`): no size matcher, frames of any size, nodes
            matched by name across files (nodes a file lacks get zero loss), uint8
            samples (~4x the legacy loader throughput). *Default*: `"legacy"`.
        gpu_confmaps: (bool) With `crop_pipeline="warp"`, render the confidence maps
            on the GPU in the training step instead of in the dataloader workers.
            *Default*: `False`.
        crop_bbox_categories: (list of str) With `crop_pipeline="warp"`, only user
            boxes of these categories (e.g. `["person"]`) are crop boxes; other linked
            boxes (e.g. MPII head boxes) fall back to the keypoint box. *Default*:
            `None` (any category).
        tiling: Configuration of tiled training/inference. Inert unless `tiling.enabled` is `True`.
        burn_in: (bool) For the `embedding` model type: burn the instance mask into the
            crop (background pixels set to `background_fill`) so the embedder sees only
            the masked instance. *Default*: `False`.
        background_fill: (str) Fill value for masked-out background when `burn_in` is
            True (`embedding` model type). One of `black` (the original mask-multiply;
            foreground mean / 0 in standardized space), `grey` (mid-grey), `mean`
            (foreground mean — equivalent to `black` for standardized inputs), or
            `noise` (per-pixel noise). *Default*: `black`.
        crop_centering: (str) For the `embedding` model type, what the fixed-square crop
            is centered on. `auto` (mask mode -> mask center-of-mass; pose mode ->
            `head_configs.embedding.embedding.anchor_part` with a mean-of-visible-nodes
            fallback), `mask_com` (force the mask center-of-mass), or `bbox` (the mask
            bounding-box midpoint, robust to concave masks whose COM lands off the
            instance). Only `mask`-mode centering is affected; pose-mode centering is
            driven by `anchor_part`. *Default*: `auto`.
    """

    ensure_rgb: bool = False
    ensure_grayscale: bool = False
    max_height: Optional[int] = None
    max_width: Optional[int] = None
    scale: float = field(
        default=1.0, validator=lambda instance, attr, value: instance.validate_scale()
    )
    crop_size: Optional[Any] = field(
        default=None,
        validator=lambda instance, attr, value: instance.validate_crop_size(),
    )
    min_crop_size: Optional[int] = 100  # to help app work in case of error
    crop_padding: Optional[int] = None
    crop_mode: str = field(default="fixed", validator=validators.in_(("fixed", "bbox")))
    crop_bbox_aspect: str = field(
        default="square", validator=validators.in_(("square", "letterbox", "stretch"))
    )
    crop_bbox_padding: float = field(default=0.1, validator=validators.ge(0))
    crop_bbox_min_size: Optional[float] = field(
        default=None, validator=validators.optional(validators.gt(0))
    )
    crop_pipeline: str = field(
        default="legacy", validator=validators.in_(("legacy", "warp"))
    )
    gpu_confmaps: bool = False
    crop_bbox_categories: Optional[List[str]] = None
    tiling: TilingConfig = field(factory=TilingConfig)
    burn_in: bool = False
    background_fill: str = "black"
    crop_centering: str = "auto"

    def validate_crop_size(self):
        """Validate `crop_size`: an int, or `[height, width]` for warp crops."""
        value = self.crop_size
        if value is None or isinstance(value, (int, float)):
            return
        try:
            hw = [int(v) for v in value]
        except TypeError:
            raise ValueError(
                f"crop_size must be an int or [height, width], got {value!r}."
            )
        if len(hw) != 2 or min(hw) <= 0:
            raise ValueError(
                f"crop_size must be an int or [height, width], got {value!r}."
            )
        if self.crop_mode != "bbox" or self.crop_pipeline != "warp":
            raise ValueError(
                "A [height, width] crop_size needs crop_mode='bbox' and "
                "crop_pipeline='warp' (the other crop paths are square)."
            )

    def validate_scale(self):
        """Scale Validation.

        Ensures PreprocessingConfig's scale is a float>=0 or list of floats>=0
        """
        if isinstance(self.scale, float) and self.scale >= 0:
            return
        if isinstance(self.scale, list) and all(
            isinstance(x, float) and x >= 0 for x in self.scale
        ):
            return
        message = "PreprocessingConfig's scale must be a float or a list of floats."
        logger.error(message)
        raise ValueError(message)


@define
class IntensityConfig:
    """Configuration of Intensity (Optional).

    Attributes:
        uniform_noise_min: (float) Minimum value for uniform noise (0-1 scale, multiplied by 255 internally). *Default*: `0.0`.
        uniform_noise_max: (float) Maximum value for uniform noise (0-1 scale, multiplied by 255 internally). *Default*: `0.04`.
        uniform_noise_p: (float) Probability of applying random uniform noise. *Default*: `0.0`.
        gaussian_noise_mean: (float) The mean of the gaussian noise distribution (0-1 scale, multiplied by 255 internally). *Default*: `0.0`.
        gaussian_noise_std: (float) The standard deviation of the gaussian noise distribution (0-1 scale, multiplied by 255 internally). *Default*: `0.02`.
        gaussian_noise_p: (float) Probability of applying random gaussian noise. *Default*: `0.0`.
        contrast_min: (float) Minimum contrast factor to apply. *Default*: `0.9`.
        contrast_max: (float) Maximum contrast factor to apply. *Default*: `1.1`.
        contrast_p: (float) Probability of applying random contrast. *Default*: `0.0`.
        brightness_min: (float) Minimum brightness factor to apply. *Default*: `0.9`.
        brightness_max: (float) Maximum brightness factor to apply. *Default*: `1.1`.
        brightness_p: (float) Probability of applying random brightness. *Default*: `0.0`.
        hsv_p: (float) `crop_pipeline="warp"` only. Probability of a random HSV shift
            of the crop (mmdet `YOLOXHSVRandomAug`, which RTMPose applies always:
            `hsv_p: 1.0`). Each of hue / saturation / value is shifted with p=0.5 by
            an integer in `[-delta, delta]` (OpenCV 8-bit HSV: hue in [0, 180)).
            *Default*: `0.0`.
        hsv_hue_delta: (float) Max hue shift. *Default*: `5.0`.
        hsv_saturation_delta: (float) Max saturation shift. *Default*: `30.0`.
        hsv_value_delta: (float) Max value shift. *Default*: `30.0`.
        blur_p: (float) `crop_pipeline="warp"` only. Probability of a box blur with a
            random odd kernel in `[blur_kernel_min, blur_kernel_max]`
            (albumentations `Blur`). *Default*: `0.0`.
        blur_kernel_min: (int) Smallest box-blur kernel. *Default*: `3`.
        blur_kernel_max: (int) Largest box-blur kernel. *Default*: `7`.
        median_blur_p: (float) `crop_pipeline="warp"` only. Probability of a median
            blur (albumentations `MedianBlur`). *Default*: `0.0`.
        median_blur_kernel_min: (int) Smallest median kernel. *Default*: `3`.
        median_blur_kernel_max: (int) Largest median kernel. *Default*: `7`.
        gaussian_blur_p: (float) `crop_pipeline="warp"` only. Probability of a
            Gaussian blur with sigma in `[gaussian_blur_sigma_min,
            gaussian_blur_sigma_max]` crop pixels. *Default*: `0.0`.
        gaussian_blur_sigma_min: (float) Smallest blur sigma. *Default*: `0.1`.
        gaussian_blur_sigma_max: (float) Largest blur sigma. *Default*: `2.0`.
    """

    uniform_noise_min: float = field(default=0.0, validator=validators.ge(0))
    uniform_noise_max: float = field(default=0.04, validator=validators.le(1))
    uniform_noise_p: float = field(default=0.0, validator=validate_proportion)
    gaussian_noise_mean: float = 0.0
    gaussian_noise_std: float = 0.02
    gaussian_noise_p: float = field(default=0.0, validator=validate_proportion)
    contrast_min: float = field(default=0.9, validator=validators.ge(0))
    contrast_max: float = field(default=1.1, validator=validators.ge(0))
    contrast_p: float = field(default=0.0, validator=validate_proportion)
    brightness_min: float = field(default=0.9, validator=validators.ge(0))
    brightness_max: float = field(default=1.1, validator=validators.le(2))
    brightness_p: float = field(default=0.0, validator=validate_proportion)
    hsv_p: float = field(default=0.0, validator=validate_proportion)
    hsv_hue_delta: float = field(default=5.0, validator=validators.ge(0))
    hsv_saturation_delta: float = field(default=30.0, validator=validators.ge(0))
    hsv_value_delta: float = field(default=30.0, validator=validators.ge(0))
    blur_p: float = field(default=0.0, validator=validate_proportion)
    blur_kernel_min: int = field(default=3, validator=validators.ge(3))
    blur_kernel_max: int = field(default=7, validator=validators.ge(3))
    median_blur_p: float = field(default=0.0, validator=validate_proportion)
    median_blur_kernel_min: int = field(default=3, validator=validators.ge(3))
    median_blur_kernel_max: int = field(default=7, validator=validators.ge(3))
    gaussian_blur_p: float = field(default=0.0, validator=validate_proportion)
    gaussian_blur_sigma_min: float = field(default=0.1, validator=validators.gt(0))
    gaussian_blur_sigma_max: float = field(default=2.0, validator=validators.gt(0))


def _warn_mixup(instance, attribute, value):
    """Warn that `mixup_p` is not implemented (it used to be silently ignored)."""
    validate_proportion(instance, attribute, value)
    if value is not None and value > 0:
        logger.warning(
            f"geometric.mixup_p={value} is not implemented by any sleap-nn data "
            "pipeline and is ignored (crop_pipeline='warp' raises instead). Set "
            "mixup_p: 0 to silence this warning."
        )


@define
class GeometricConfig:
    """Configuration of Geometric (Optional).

    Attributes:
        rotation_min: (float) Minimum rotation angle in degrees. A random angle in (rotation_min, rotation_max) will be sampled and applied to both images and keypoints. Set to 0 to disable rotation augmentation. *Default*: `-15.0`.
        rotation_max: (float) Maximum rotation angle in degrees. A random angle in (rotation_min, rotation_max) will be sampled and applied to both images and keypoints. Set to 0 to disable rotation augmentation. *Default*: `15.0`.
        rotation_p: (float, optional) Probability of applying random rotation independently. If set, rotation is applied separately from scale/translate. If `None`, falls back to `affine_p` for bundled behavior. *Default*: `1.0`.
        scale_min: (float) Minimum scaling factor. If scale_min and scale_max are provided, the scale is randomly sampled from the range scale_min <= scale <= scale_max for isotropic scaling. *Default*: `0.9`.
        scale_max: (float) Maximum scaling factor. If scale_min and scale_max are provided, the scale is randomly sampled from the range scale_min <= scale <= scale_max for isotropic scaling. *Default*: `1.1`.
        scale_p: (float, optional) Probability of applying random scaling independently. If set, scaling is applied separately from rotation/translate. If `None`, falls back to `affine_p` for bundled behavior. *Default*: `1.0`.
        translate_width: (float) Maximum absolute fraction for horizontal translation. For example, if translate_width=a, then horizontal shift is randomly sampled in the range -img_width * a < dx < img_width * a. Will not translate by default. *Default*: `0.0`.
        translate_height: (float) Maximum absolute fraction for vertical translation. For example, if translate_height=a, then vertical shift is randomly sampled in the range -img_height * a < dy < img_height * a. Will not translate by default. *Default*: `0.0`.
        translate_p: (float, optional) Probability of applying random translation independently. If set, translation is applied separately from rotation/scale. If `None`, falls back to `affine_p` for bundled behavior. *Default*: `None`.
        affine_p: (float) Probability of applying random affine transformations (rotation, scale, translate bundled together). Used for backwards compatibility when individual `*_p` params are not set. *Default*: `0.0`.
        erase_scale_min: (float) Minimum value of range of proportion of erased area against input image. *Default*: `0.0001`.
        erase_scale_max: (float) Maximum value of range of proportion of erased area against input image. *Default*: `0.01`.
        erase_ratio_min: (float) Minimum value of range of aspect ratio of erased area. *Default*: `1.0`.
        erase_ratio_max: (float) Maximum value of range of aspect ratio of erased area. *Default*: `1.0`.
        erase_p: (float) Probability of applying random erase. *Default*: `1.0`.
        mixup_lambda_min: (float) Minimum mixup strength value. *Default*: `0.01`.
        mixup_lambda_max: (float) Maximum mixup strength value. *Default*: `0.05`.
        mixup_p: (float) NOT IMPLEMENTED: a value > 0 logs a warning (and raises with
            `crop_pipeline="warp"`). *Default*: `0.0`.
        affine_sampling: (str) `crop_pipeline="warp"` only. `"uniform"`: rotation,
            scale and translation are uniform in their ranges and `scale_*` scales the
            image content (> 1 zooms in), as in the stock pipeline. `"mmpose"`: mmpose
            `RandomBBoxTransform` semantics -- truncated-normal draws (+-1 sigma =
            the range ends) around the range middle, `scale_*` scales the crop *box*
            (content scale = 1 / box scale) and the shift moves the box before it is
            rotated, so `scale_min: 0.6, scale_max: 1.4, rotation_min: -80,
            rotation_max: 80` reproduce RTMPose's config. *Default*: `"uniform"`.
        half_body_p: (float) `crop_pipeline="warp"` only. Probability of mmpose's
            `RandomHalfBody`: crop to the upper or lower body only. The crop box is
            recomputed from the chosen half's visible nodes (centered on their mean,
            `half_body_padding` times their extent, fit to the crop aspect) before
            the single warp, so it costs nothing extra. *Default*: `0.0` (ViTPose /
            RTMPose: `0.3`).
        half_body_min_total_keypoints: (int) Only instances with at least this many
            visible (labeled, located) nodes are eligible. *Default*: `9`.
        half_body_min_upper_keypoints: (int) Visible upper-body nodes needed to
            choose the upper half. *Default*: `2`.
        half_body_min_lower_keypoints: (int) Visible lower-body nodes needed to
            choose the lower half. *Default*: `3`.
        half_body_padding: (float) Box size as a multiple of the half's keypoint
            extent. *Default*: `1.5`.
        half_body_upper_prob: (float) Probability of preferring the upper half when
            both qualify. *Default*: `0.7`.
        half_body_lower_nodes: (list of str) Lower-body node names. `None` takes
            the nodes whose name contains hip / knee / ankle / heel / toe / foot (COCO:
            hips, knees, ankles, like mmpose's `lower_body_ids`). *Default*: `None`.
        coarse_dropout_p: (float) `crop_pipeline="warp"` only. Probability of
            albumentations `CoarseDropout`: black holes in the crop. Keypoints inside
            holes keep their targets (as in mmpose). *Default*: `0.0` (RTMPose stage
            1: `1.0`, stage 2: `0.5`).
        coarse_dropout_min_holes: (int) Fewest holes. *Default*: `1`.
        coarse_dropout_max_holes: (int) Most holes. *Default*: `1`.
        coarse_dropout_min_size: (float) Smallest hole side as a fraction of the crop
            side (height and width drawn independently). *Default*: `0.2`.
        coarse_dropout_max_size: (float) Largest hole side fraction. *Default*: `0.4`.
        flip_p: (float) Probability of mirroring the image and keypoints left/right (`x' = (W-1) - x`). When applied, left/right symmetric body parts are swapped using the skeleton's symmetries so labels stay correct. *Correctness note*: if the skeleton has no symmetries, flipping is only valid when the animal is truly left/right symmetric in labeling; a warning is emitted otherwise. *Default*: `0.0` (disabled).
    """

    rotation_min: float = field(default=-15.0, validator=validators.ge(-180))
    rotation_max: float = field(default=15.0, validator=validators.le(180))
    rotation_p: Optional[float] = field(default=1.0)
    scale_min: float = field(default=0.9, validator=validators.ge(0))
    scale_max: float = field(default=1.1, validator=validators.ge(0))
    scale_p: Optional[float] = field(default=1.0)
    translate_width: float = 0.0
    translate_height: float = 0.0
    translate_p: Optional[float] = field(default=None)
    affine_p: float = field(default=0.0, validator=validate_proportion)
    erase_scale_min: float = 0.0001
    erase_scale_max: float = 0.01
    erase_ratio_min: float = 1.0
    erase_ratio_max: float = 1.0
    erase_p: float = field(default=0.0, validator=validate_proportion)
    mixup_lambda_min: float = field(default=0.01, validator=validators.ge(0))
    mixup_lambda_max: float = field(default=0.05, validator=validators.le(1))
    mixup_p: float = field(default=0.0, validator=_warn_mixup)
    flip_p: float = field(default=0.0, validator=validate_proportion)
    affine_sampling: str = field(
        default="uniform", validator=validators.in_(("uniform", "mmpose"))
    )
    half_body_p: float = field(default=0.0, validator=validate_proportion)
    half_body_min_total_keypoints: int = field(default=9, validator=validators.ge(0))
    half_body_min_upper_keypoints: int = field(default=2, validator=validators.ge(1))
    half_body_min_lower_keypoints: int = field(default=3, validator=validators.ge(1))
    half_body_padding: float = field(default=1.5, validator=validators.gt(0))
    half_body_upper_prob: float = field(default=0.7, validator=validate_proportion)
    half_body_lower_nodes: Optional[List[str]] = None
    coarse_dropout_p: float = field(default=0.0, validator=validate_proportion)
    coarse_dropout_min_holes: int = field(default=1, validator=validators.ge(0))
    coarse_dropout_max_holes: int = field(default=1, validator=validators.ge(0))
    coarse_dropout_min_size: float = field(default=0.2, validator=validate_proportion)
    coarse_dropout_max_size: float = field(default=0.4, validator=validate_proportion)


@define
class AugmentationStage2Config:
    """A second augmentation stage from a given epoch or step (warp pipeline only).

    The "stage-2 pipeline switch" of RTMPose (mmdet `PipelineSwitchHook`): the last
    epochs train with weaker augmentation. `geometric` / `intensity` are partial
    overrides merged over the stage-1 sections (e.g. `{rotation_min: -60,
    rotation_max: 60, coarse_dropout_p: 0.5}`). Applied by
    `AugmentationStageCallback`, which flips a shared-memory flag that persistent
    workers read per sample; batches already prefetched when the switch happens
    (at most `num_workers * prefetch_factor`) still use stage 1.

    Attributes:
        start_epoch: (int) First epoch (0-based) of stage 2. *Default*: `None`.
        start_step: (int) First global step of stage 2 (used when `start_epoch` is
            `None`). *Default*: `None`.
        geometric: (dict) Overrides of `augmentation_config.geometric`.
            *Default*: `None`.
        intensity: (dict) Overrides of `augmentation_config.intensity`.
            *Default*: `None`.
    """

    start_epoch: Optional[int] = field(
        default=None, validator=validators.optional(validators.ge(0))
    )
    start_step: Optional[int] = field(
        default=None, validator=validators.optional(validators.ge(0))
    )
    geometric: Optional[Dict[str, Any]] = None
    intensity: Optional[Dict[str, Any]] = None


@define
class AugmentationConfig:
    """Configuration of Augmentation.

    Attributes:
        intensity: Configuration options for intensity-based augmentations like brightness, contrast, etc. If None, no intensity augmentations will be applied.
        geometric: Configuration options for geometric augmentations like rotation, scaling, translation etc. If None, no geometric augmentations will be applied.
        stage2: (AugmentationStage2Config) Optional second augmentation stage
            (`crop_pipeline="warp"` only). *Default*: `None`.
    """

    intensity: Optional[IntensityConfig] = None
    geometric: Optional[GeometricConfig] = None
    stage2: Optional[AugmentationStage2Config] = None


def validate_test_file_path(instance, attribute, value):
    """Validate test_file_path to accept str or List[str].

    Args:
        instance: The instance being validated.
        attribute: The attribute being validated.
        value: The value to validate.

    Raises:
        ValueError: If value is not None, str, or list of strings.
    """
    if value is None:
        return
    if isinstance(value, str):
        return
    if isinstance(value, (list, tuple)) and all(isinstance(p, str) for p in value):
        return
    message = f"{attribute.name} must be a string or list of strings, got {type(value).__name__}"
    logger.error(message)
    raise ValueError(message)


@define
class IdentityConfig:
    """Declared identity-equality semantics for the `embedding` model type.

    Each positives/negatives source silently asserts "same/different animal". These
    fields DECLARE what the track labels mean so the objective can validate them
    (e.g. `positives.scope=global_id` needs a global identity per detection: a
    `sio.Identity`, or a track name under `track_names_are_global=True`).

    Attributes:
        tracks_are_proofread: (bool) Tracks are swap-free within a video. Gates
            `positives.scope=tracklet` (warn if False). *Default*: `False`.
        track_names_are_global: (bool) The same track name means the same animal
            across videos, so a track name can stand in for a `sio.Identity`. Under
            `positives.scope=global_id`, a detection without a `sio.Identity` is a
            sample only when this is True (training errors when the labels carry no
            `sio.Identity` and this is False). *Default*: `False`.
        detections_deduplicated: (bool) No duplicate/over-segmented detection per
            frame. Gates `same_frame` negatives (warn if False). *Default*: `True`.
    """

    tracks_are_proofread: bool = False
    track_names_are_global: bool = False
    detections_deduplicated: bool = True


@define
class SplitConfig:
    """Group-aware train/val split, decided before training.

    For the `embedding` model type the train/val partition *is* the generalization axis:
    the model must only ever see the training partition, with val/test held out by a group
    key so there is no leakage. When set on `DataConfig.split` (and no explicit
    `val_labels_path` is provided), the trainer partitions the training labels by
    `split_by` instead of the default frame-level random `validation_fraction` split.

    Attributes:
        split_by: (str) Group key for the partition. One of:
            `frame` (stratified-random over `LabeledFrame`s, identity-balanced; both train
            and val contain all identities), `video` (hold out whole videos by sio video
            index), or `identity` (hold out whole track names; disjoint identity sets).
            *Default*: `frame`.
        n_folds: (int) Number of CV folds; the val partition is `1 / n_folds` of the data.
            *Default*: `5`.
        fold: (int) Which fold (0-based) to hold out as validation. *Default*: `0`.
        seed: (int) Random seed for the (shuffled) splitter. *Default*: `0`.
    """

    split_by: str = "frame"
    n_folds: int = 5
    fold: int = 0
    seed: int = 0


@define
class InstanceFilterConfig:
    """Train-time instance filters (`crop_pipeline="warp"` only).

    Attributes:
        min_labeled_nodes: (int) Train only on instances with at least this many of
            `count_nodes` labeled. *Default*: `0`.
        count_nodes: (list of str) Nodes counted by `min_labeled_nodes` (e.g. the 12
            limb joints). `None` counts every model node. *Default*: `None`.
        min_bbox_size: (float) Train only on instances whose box has
            `sqrt(area) >= min_bbox_size` frame pixels. *Default*: `0.0`.
    """

    min_labeled_nodes: int = field(default=0, validator=validators.ge(0))
    count_nodes: Optional[List[str]] = None
    min_bbox_size: float = field(default=0.0, validator=validators.ge(0))


@define
class DataConfig:
    """Data configuration.

    Attributes:
        train_labels_path: (List[str]) List of paths to training data (`.slp` file(s)). *Default*: `None`.
        val_labels_path: (List[str]) List of paths to validation data (`.slp` file(s)). *Default*: `None`.
        validation_fraction: (float) Float between 0 and 1 specifying the fraction of the training set to sample for generating the validation set. The remaining labeled frames will be left in the training set. If the `validation_labels` are already specified, this has no effect. *Default*: `0.1`.
        use_same_data_for_val: (bool) If `True`, use the same data for both training and validation (train = val). Useful for intentional overfitting on small datasets. When enabled, `val_labels_path` and `validation_fraction` are ignored. *Default*: `False`.
        test_file_path: (str or List[str]) Path or list of paths to test dataset(s) (`.slp` file(s) or `.mp4` file(s)). *Note*: This is used only with CLI to get evaluation on test set after training is completed. *Default*: `None`.
        provider: (str) Provider class to read the input sleap files. Only "LabelsReader" is currently supported for the training pipeline. *Default*: `"LabelsReader"`.
        user_instances_only: (bool) `True` if only user labeled instances should be used for training. If `False`, both user labeled and predicted instances would be used. For the `embedding` model type, `True` leaves out only the predictions (`PredictedInstance`s and `PredictedSegmentationMask`s) that a user label on the same frame supersedes, by sleap-io's `LabeledFrame.unused_predictions` / `unused_predicted_masks` rule: a user `Instance` carrying the prediction's track (or linked to it by `from_predicted` on an untracked frame), or a user mask adopting the predicted one. Predictions nobody relabeled, e.g. predicted poses with proofread tracks, stay training samples; the post-training retrieval eval follows the same rule. *Default*: `True`.
        data_pipeline_fw: (str) Framework to create the data loaders. One of [`torch_dataset`, `torch_dataset_cache_img_memory`, `torch_dataset_cache_img_disk`]. *Default*: `"torch_dataset"`. (Note: When using `torch_dataset`, `num_workers` in `trainer_config` should be set to 0 as multiprocessing doesn't work with pickling video backends.)
        cache_img_path: (str) Path to save `.jpg` images created with `torch_dataset_cache_img_disk` data pipeline framework. If `None`, the path provided in `trainer_config.save_ckpt` is used. The `train_imgs` and `val_imgs` dirs are created inside this path. *Default*: `None`.
        use_existing_imgs: (bool) Use existing train and val images/ chunks in the `cache_img_path` for `torch_dataset_cache_img_disk` frameworks. If `True`, the `cache_img_path` should have `train_imgs` and `val_imgs` dirs. *Default*: `False`.
        delete_cache_imgs_after_training: (bool) If `False`, the images (torch_dataset_cache_img_disk) are retained after training. Else, the files are deleted. *Default*: `True`.
        parallel_caching: (bool) If `True`, use parallel processing to cache images (significantly faster for large datasets). *Default*: `True`.
        cache_workers: (int) Number of worker threads for parallel caching. If 0, uses min(4, cpu_count). *Default*: `0`.
        preprocessing: Configuration options related to data preprocessing.
        use_augmentations_train: (bool) True if the data augmentation should be applied to the training data, else False. *Default*: `True`.
        augmentation_config: Configurations related to augmentation. (only if `use_augmentations_train` is `True`)
        use_negative_frames: (bool) If ``True``, include all user-confirmed negative frames
            (``labels.negative_frames``) in the training set. These are frames the user explicitly
            marked as containing no instances. They produce all-zero confidence maps, teaching the model
            not to hallucinate detections on empty backgrounds. Use ``negative_loss_weight`` to control
            the relative importance of negative vs positive samples. *Default*: ``False``.
        negative_loss_weight: (float) Relative weight applied to the loss for negative samples. Must be > 0.
            Values < 1 down-weight negatives; values > 1 up-weight them. Only has effect when
            ``use_negative_frames`` is ``True``. *Default*: `1.0`.
        node_loss_mask: (bool) Per-node loss weighting for partially labeled
            skeletons (single-instance, centered-instance and bottom-up confmaps).
            Unlabeled points (sleap-io >= 0.10 ``labeled=False`` and
            ``visible=False``, e.g. COCO ``v=0`` or nodes another dataset of a merged
            skeleton annotates) get zero loss instead of being trained as absent, as
            do nodes listed as unannotated by the per-file stopgap
            ``Labels.provenance["labeled_nodes"]`` (node names or indices; all nodes
            when absent). Also the master switch for ``unlabeled_points``. Fully
            labeled data (every pre-SLP-3.0 file) with ``unlabeled_points: absent``
            trains exactly as with ``False``. *Default*: ``True``.
        unlabeled_points: (str) What a point with no location in the model input
            (NaN, e.g. COCO ``v=0``, or pushed out of the image/crop by augmentation)
            trains as. ``"absent"`` keeps weight 1 and trains an all-zero confmap
            (SLEAP's historical behavior); ``"ignore"`` gives it zero loss weight
            (as mmpose does). Negative frames always train as absent. Requires
            ``node_loss_mask``. *Default*: ``"absent"``.
        supervise_occluded: (bool) If ``True``, labeled-but-occluded points that carry
            estimated coordinates (sleap-io >= 0.10; COCO ``v=1``) are trained as
            Gaussians at those coordinates; if ``False`` they get zero loss. Crop sizes,
            crop boxes and centroids use them either way. *Default*: ``True``.
        skeletons: skeleton configuration for the `.slp` file. This will be pulled from the train dataset and saved to the `training_config.yaml`
        split: (Optional[SplitConfig]) Group-aware train/val split decided before training.
            When set and `val_labels_path` is not provided, the trainer
            partitions the training labels by the configured group key (`frame`/`video`/
            `identity`) instead of the default frame-level random `validation_fraction`
            split. *Default*: `None` (unchanged default behavior).
        centroids_from_masks: (str) Derive `UserCentroid` annotations from the labels'
            segmentation masks at load time, so a MASK-ONLY dataset (no pose
            annotations at all) can train a centroid model. The value names the
            derivation method — `center_of_mass` or `bbox_center`, the two
            `sio.SegmentationMask.to_centroid` offers — and `None` (default)
            disables it. (`geometric_median` is defined over a point set and does
            not apply to masks; it is available for pose-derived centroids via
            `centroid_method`.) Frames that
            already carry user centroids are left alone. Once derived, the ordinary
            `centroid_source="user"` path takes over unchanged; nothing downstream
            knows the centroids came from masks. *Default*: `None`.
        bbox_source: (str) Where the `bbox_*` model types read their ground-truth boxes
            from: `bboxes` (the frames' `UserBoundingBox` annotations,
            `LabeledFrame.bboxes`), `instances` (a tight box around each user pose
            instance's visible points, derived at load time with
            `sio.Instance.to_bbox`), `masks` (each user segmentation mask's extent,
            `sio.SegmentationMask.to_bbox`), or `auto` (default): per labels file,
            `bboxes` if it carries any user box, else `instances`, else `masks`.
            Derived boxes are added to the frames in memory only. Ignored by every
            other model type. *Default*: `"auto"`.
        bbox_padding: (float) Pixels added on every side of boxes derived from
            instances or masks (`bbox_source` resolving to `instances`/`masks`).
            Pose keypoints sit inside the animal's outline, so a tight keypoint box
            is slightly smaller than the animal; a few pixels of padding compensate.
            Never applied to existing `UserBoundingBox` annotations. *Default*: `0.0`.
        instance_filter: (InstanceFilterConfig) Train-time instance filters for
            `crop_pipeline="warp"` (validation keeps every instance). *Default*: `None`.
    """

    train_labels_path: Optional[List[str]] = None
    val_labels_path: Optional[List[str]] = None  # TODO : revisit MISSING!
    validation_fraction: float = 0.1
    use_same_data_for_val: bool = False
    test_file_path: Optional[Any] = field(
        default=None, validator=validate_test_file_path
    )
    provider: str = "LabelsReader"
    user_instances_only: bool = True
    data_pipeline_fw: str = "torch_dataset"
    cache_img_path: Optional[str] = None
    use_existing_imgs: bool = False
    delete_cache_imgs_after_training: bool = True
    parallel_caching: bool = True
    cache_workers: int = 0
    preprocessing: PreprocessingConfig = field(factory=PreprocessingConfig)
    use_augmentations_train: bool = True
    augmentation_config: Optional[AugmentationConfig] = field(
        factory=lambda: AugmentationConfig(geometric=GeometricConfig())
    )
    use_negative_frames: bool = False
    negative_loss_weight: float = field(default=1.0, validator=validators.gt(0))
    node_loss_mask: bool = True
    unlabeled_points: str = field(
        default="absent", validator=validators.in_(("absent", "ignore"))
    )
    supervise_occluded: bool = True
    skeletons: Optional[list] = None
    identity: Optional[IdentityConfig] = None
    split: Optional[SplitConfig] = None
    centroids_from_masks: Optional[str] = None
    bbox_source: str = field(
        default="auto",
        validator=validators.in_(("auto", "bboxes", "instances", "masks")),
    )
    bbox_padding: float = field(default=0.0, validator=validators.ge(0))
    instance_filter: Optional[InstanceFilterConfig] = None


def data_mapper(legacy_config: dict) -> DataConfig:
    """Maps the legacy data configuration to the new data configuration.

    Args:
        legacy_config: A dictionary containing the legacy data configuration.

    Returns:
        An instance of `DataConfig` with the mapped configuration.
    """
    legacy_config_data = legacy_config.get("data", {})
    legacy_config_optimization = legacy_config.get("optimization", {})
    train_labels_path = legacy_config_data.get("labels", {}).get(
        "training_labels", None
    )
    val_labels_path = legacy_config_data.get("labels", {}).get(
        "validation_labels", None
    )

    # get skeleton(s)
    json_skeletons = legacy_config_data.get("labels", {}).get("skeletons", None)
    skeletons_list = None
    if json_skeletons is not None:
        skeletons_list = []
        skeletons = SkeletonDecoder().decode(json_skeletons)
        skeletons = yaml.safe_load(SkeletonYAMLEncoder().encode(skeletons))
        for skl_name in skeletons.keys():
            skl = skeletons[skl_name]
            skl["name"] = skl_name
            skeletons_list.append(skl)

    data_cfg_args = {}
    preprocessing_args = {}
    intensity_args = {}
    geometric_args = {}

    if train_labels_path is not None:
        data_cfg_args["train_labels_path"] = [train_labels_path]
    if val_labels_path is not None:
        data_cfg_args["val_labels_path"] = [val_labels_path]
    if (
        legacy_config_data.get("labels", {}).get("validation_fraction", None)
        is not None
    ):
        data_cfg_args["validation_fraction"] = legacy_config_data["labels"][
            "validation_fraction"
        ]
    if legacy_config_data.get("labels", {}).get("test_labels", None) is not None:
        data_cfg_args["test_file_path"] = legacy_config_data["labels"]["test_labels"]

    # preprocessing
    if legacy_config_data.get("preprocessing", {}).get("ensure_rgb", None) is not None:
        preprocessing_args["ensure_rgb"] = legacy_config_data["preprocessing"][
            "ensure_rgb"
        ]
    if (
        legacy_config_data.get("preprocessing", {}).get("ensure_grayscale", None)
        is not None
    ):
        preprocessing_args["ensure_grayscale"] = legacy_config_data["preprocessing"][
            "ensure_grayscale"
        ]
    if (
        legacy_config_data.get("preprocessing", {}).get("target_height", None)
        is not None
    ):
        preprocessing_args["max_height"] = legacy_config_data["preprocessing"][
            "target_height"
        ]
    if (
        legacy_config_data.get("preprocessing", {}).get("target_width", None)
        is not None
    ):
        preprocessing_args["max_width"] = legacy_config_data["preprocessing"][
            "target_width"
        ]
    if (
        legacy_config_data.get("preprocessing", {}).get("input_scaling", None)
        is not None
    ):
        preprocessing_args["scale"] = legacy_config_data["preprocessing"][
            "input_scaling"
        ]
    if (
        legacy_config_data.get("instance_cropping", {}).get("crop_size", None)
        is not None
    ):
        size = legacy_config_data["instance_cropping"]["crop_size"]
        preprocessing_args["crop_size"] = size

    # augmentation
    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "uniform_noise_min_val", None
        )
        is not None
    ):
        intensity_args["uniform_noise_min"] = max(
            legacy_config_optimization["augmentation_config"]["uniform_noise_min_val"],
            0.0,
        )

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "uniform_noise_max_val", None
        )
        is not None
    ):
        intensity_args["uniform_noise_max"] = min(
            legacy_config_optimization["augmentation_config"]["uniform_noise_max_val"],
            1.0,
        )

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "uniform_noise", None
        )
        is not None
    ):
        intensity_args["uniform_noise_p"] = float(
            legacy_config_optimization["augmentation_config"]["uniform_noise"]
        )

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "gaussian_noise_mean", None
        )
        is not None
    ):
        intensity_args["gaussian_noise_mean"] = legacy_config_optimization[
            "augmentation_config"
        ]["gaussian_noise_mean"]

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "gaussian_noise_stddev", None
        )
        is not None
    ):
        intensity_args["gaussian_noise_std"] = legacy_config_optimization[
            "augmentation_config"
        ]["gaussian_noise_stddev"]

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "gaussian_noise", None
        )
        is not None
    ):
        intensity_args["gaussian_noise_p"] = float(
            legacy_config_optimization["augmentation_config"]["gaussian_noise"]
        )

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "contrast_min_gamma", None
        )
        is not None
    ):
        intensity_args["contrast_min"] = max(
            legacy_config_optimization["augmentation_config"]["contrast_min_gamma"],
            0.0,
        )

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "contrast_max_gamma", None
        )
        is not None
    ):
        intensity_args["contrast_max"] = max(
            legacy_config_optimization["augmentation_config"]["contrast_max_gamma"],
            0.0,
        )

    if (
        legacy_config_optimization.get("augmentation_config", {}).get("contrast", None)
        is not None
    ):
        intensity_args["contrast_p"] = float(
            legacy_config_optimization["augmentation_config"]["contrast"]
        )

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "brightness_min_val", None
        )
        is not None
    ):
        intensity_args["brightness_min"] = min(
            max(
                legacy_config_optimization["augmentation_config"]["brightness_min_val"],
                0.0,
            ),
            2.0,
        )

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "brightness_max_val", None
        )
        is not None
    ):
        intensity_args["brightness_max"] = min(
            legacy_config_optimization["augmentation_config"]["brightness_max_val"], 2.0
        )  # kornia brightness_max can only be 2.0

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "brightness", None
        )
        is not None
    ):
        intensity_args["brightness_p"] = float(
            legacy_config_optimization["augmentation_config"]["brightness"]
        )

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "rotation_min_angle", None
        )
        is not None
    ):
        geometric_args["rotation_min"] = legacy_config_optimization[
            "augmentation_config"
        ]["rotation_min_angle"]

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "rotation_max_angle", None
        )
        is not None
    ):
        geometric_args["rotation_max"] = legacy_config_optimization[
            "augmentation_config"
        ]["rotation_max_angle"]

    if (
        legacy_config_optimization.get("augmentation_config", {}).get("scale_min", None)
        is not None
    ):
        geometric_args["scale_min"] = legacy_config_optimization["augmentation_config"][
            "scale_min"
        ]

    if (
        legacy_config_optimization.get("augmentation_config", {}).get("scale_max", None)
        is not None
    ):
        geometric_args["scale_max"] = legacy_config_optimization["augmentation_config"][
            "scale_max"
        ]

    if legacy_config_optimization.get("augmentation_config", {}).get(
        "random_flip", False
    ):
        if legacy_config_optimization["augmentation_config"].get(
            "flip_horizontal", True
        ):
            geometric_args["flip_p"] = 0.5
        else:
            logger.warning(
                "Legacy config has vertical flip (flip_horizontal=False) enabled; "
                "vertical flip is not supported in sleap-nn and will be dropped."
            )

    # Legacy SLEAP applies `rotate`, `translate` and `scale` as independent
    # transforms, each with p=1.0 when its flag is on and not at all when off. Map
    # them onto the per-transform probabilities rather than the bundled `affine_p`:
    # `GeometricConfig` defaults `rotation_p`/`scale_p` to 1.0, and any non-None
    # per-transform probability makes the augmenter ignore `affine_p`, so leaving
    # them unset would apply rotation and scale regardless of the legacy flags.
    legacy_aug = legacy_config_optimization.get("augmentation_config", {}) or {}
    geometric_args["rotation_p"] = 1.0 if legacy_aug.get("rotate", False) else 0.0
    geometric_args["scale_p"] = 1.0 if legacy_aug.get("scale", False) else 0.0
    geometric_args["affine_p"] = 0.0

    if legacy_aug.get("translate", False):
        logger.warning(
            "Legacy config has translation augmentation enabled; legacy translation "
            "is specified in pixels while sleap-nn uses a fraction of the image size, "
            "so it cannot be converted and will be dropped. Set "
            "`data_config.augmentation_config.geometric.translate_width`/"
            "`translate_height`/`translate_p` to enable translation."
        )

    data_cfg_args["preprocessing"] = PreprocessingConfig(**preprocessing_args)
    data_cfg_args["augmentation_config"] = AugmentationConfig(
        intensity=IntensityConfig(**intensity_args),
        geometric=GeometricConfig(**geometric_args),
    )

    data_cfg_args["skeletons"] = (
        skeletons_list
        if skeletons_list is not None and len(skeletons_list) > 0
        else None
    )

    return DataConfig(**data_cfg_args)
