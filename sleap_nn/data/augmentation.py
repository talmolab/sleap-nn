import attr
from typing import Tuple, Dict, Any, Optional, Union
import torch
from torch.distributions import Uniform
import torchdata.datapipes as dp
import kornia as K
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor
from kornia.constants import Resample, SamplePadding
from kornia.geometry.transform import warp_affine
from kornia.augmentation.utils.param_validation import _range_bound


class RandomUniformNoise(IntensityAugmentationBase2D):
    """Data transformer for applying random uniform noise to input images.

    This is a custom Kornia augmentation inheriting from `IntensityAugmentationBase2D`.
    Uniform noise within (min_val, max_val) is applied to the entire input image.
    min_val and max_val must satisfy 0 <= min_val <= max_val <= 255.

    Note: min_val and max_val are int/float between 0 and 255, but Kornia expects
    images to be in float. Thus, min_val/max_val are normalized and the input image
    is expected to be a float tensor between 0 and 1.

    Note: Inverse transform is not implemented and re-applying the same transformation
    in the example below does not work when included in an AugmentationSequential class.

    Args:
        noise: 2-tuple (min_val, max_val).
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
        >>> RandomUniformNoise(min_val=0, max_val=10, p=1.)(img)
        tensor([[[[0.9607, 0.5865],
                  [0.2705, 0.5920]]]])

    To apply the exact augmentation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomUniformNoise(min_val=0, max_val=10, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    Ref: `kornia.augmentation._2d.intensity.gaussian_noise
    <https://kornia.readthedocs.io/en/latest/_modules/kornia/augmentation/_2d/intensity/gaussian_noise.html#RandomGaussianNoise>`_.
    """

    def __init__(
        self,
        noise: Tuple[int, int],
        p: float = 0.5,
        p_batch: float = 1.0,
        clip_output: bool = True,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim
        )
        self.flags = {"uniform_noise": _range_bound(noise, 'uniform_noise', bounds=(0.0, 255.0))/255.0}
        self.clip_output = clip_output

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
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


class RandomTranslatePx(GeometricAugmentationBase2D):
    """Data transformer for applying random pixel translations by pixel value to input images.

    This is a custom Kornia augmentation inheriting from `GeometricAugmentationBase2D`.
    Random pixel translations along the x and y axes are applied to the entire input image.
    The translations are uniformly sampled from the specified ranges (inclusive).

    Note: Inverse transform is not implemented and re-applying the same transformation
    in the example below does not work when included in an AugmentationSequential class.

    Args:
        translate_px: dictionary containing the ranges for random translations along x and y axes.
            It should have the format: {"x": (min_x, max_x), "y": (min_y, max_y)}.
        resample: the resampling algorithm to use during the transformation. Can be a string
            representing the resampling mode (e.g., "nearest", "bilinear", etc.), an integer
            representing the corresponding OpenCV resampling mode, or a Resample enum value.
        same_on_batch: if True, applies the same transformation across the entire batch.
        align_corners: if True, keeps the corners aligned during resampling.
        padding_mode: the padding mode to use during resampling. Can be a string representing the
            padding mode (e.g., "zeros", "border", etc.), an integer representing the corresponding
            PyTorch padding mode, or a SamplePadding enum value.
        p: probability for applying the augmentation. This parameter controls the augmentation
            probabilities element-wise for a batch.
        keepdim: whether to keep the output shape the same as the input (True) or broadcast it to
            the batch form (False).

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.rand(1, 1, 2, 2)
        >>> transformer = RandomTranslatePx(translate_px={"x": (-1, 1), "y": (-1, 1)}, p=1.)
        >>> transformer(img)
        tensor([[[[0.2515, 0.0253],
                  [0.0910, 0.0083]]]])

    To apply the exact augmentation again, you may take advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomTranslatePx(translate_px={"x": (-10, 10), "y": (-5, 5)}, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    Ref: `kornia.augmentation._2d.geometric.translate
    <https://kornia.readthedocs.io/en/latest/_modules/kornia/augmentation/_2d/geometric/translate.html#RandomTranslate>`_.
    """

    def __init__(
        self,
        translate_px: Dict[str, Tuple[int, int]],
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = False,
        padding_mode: Union[str, int, SamplePadding] = SamplePadding.ZEROS.name,
        p: float = 0.5,
        keepdim: bool = False,
    ):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.translate_px = translate_px
        self.flags = {
            "resample": Resample.get(resample),
            "padding_mode": SamplePadding.get(padding_mode),
            "align_corners": align_corners,
        }

    def compute_transformation(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        batch_size = input.shape[0]
        translations = torch.empty((batch_size, 2), dtype=torch.float32)
        translations[:, 0] = (
            torch.rand(batch_size)
            * (self.translate_px["x"][1] - self.translate_px["x"][0])
            + self.translate_px["x"][0]
        )
        translations[:, 1] = (
            torch.rand(batch_size)
            * (self.translate_px["y"][1] - self.translate_px["y"][0])
            + self.translate_px["y"][0]
        )

        trans_matrix = (
            torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        )
        trans_matrix[:, 0, 2] = translations[:, 0]
        trans_matrix[:, 1, 2] = translations[:, 1]

        return trans_matrix

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        if "translate_px" in params:
            tfm_matrix = params["translate_px"]
        else:
            tfm_matrix = transform[:, :2, :].to(input.device)
            self._params["translate_px"] = tfm_matrix
        return warp_affine(input, tfm_matrix, dsize=input.shape[-2:])

class RandomBrightnessAdd(IntensityAugmentationBase2D):
    """
    Randomly adds brightness of input images.

    This class applies random brightness augmentation to input images. The brightness adjustment is performed by adding
    a random value uniformly sampled from the specified range to the entire input image. The input images are expected
    to have values between 0 and 1, and the brightness range is specified as a tuple of integers between 0 and 255. The
    generated brightness values are divided by 255.0 to bring them into the appropriate range.

    Args:
        brightness: A tuple representing the range of brightness adjustment.
            The values are integers between 0 and 255. The brightness adjustment is performed by adding a random
            value uniformly sampled from this range to the input image.
        p: Probability for applying the augmentation to each element in the batch. Default: 0.5.
        p_batch: Probability for applying the augmentation to the entire batch. Default: 1.0.
        clip_output: If True, clip the output tensor to the [0, 1] range after the brightness
            augmentation. If False, the output tensor may have values outside the range. Default: True.
        same_on_batch: If True, apply the same transformation across the entire batch.
            Default: False.
        keepdim: If True, keep the output shape the same as input. If False, broadcast the output to
            the batch form. Default: False.

    Shape:
        - Input: (B, C, H, W) where B is the batch size, C is the number of channels, H is the height, and W is the width.
        - Output: (B, C, H, W) if keepdim is True, otherwise (C, H, W).

    Returns:
        Tensor: Randomly brightness-adjusted tensor with the same shape as the input.

    Example:
        >>> input = torch.rand(4, 3, 256, 256)  # Batch of 4 RGB images of size 256x256
        >>> brightness_augmenter = RandomBrightnessAdd(brightness=(30, 70), p=0.8)
        >>> output = brightness_augmenter(input)
    """

    def __init__(
        self,
        brightness: Tuple[int, int],
        p: float = 0.5,
        p_batch: float = 1.0,
        clip_output: bool = True,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ):
        super().__init__(
            p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim
        )
        self.flags = {
            "brightness": _range_bound(brightness, "brightness", bounds=(0.0, 255.0))
            / 255.0
        }
        self.clip_output = clip_output

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        # Generate random brightness values for each element in the batch.
        batch_size = input.size(0)
        if "brightness" in params:
            brightness = self._params["brightness"]
        else:
            brightness = (
                torch.FloatTensor(batch_size)
                .uniform_(flags["brightness"][0], flags["brightness"][1])
                .to(input.device)
                .to(input.dtype)
            )
            self._params["brightness"] = brightness
        # Add the brightness values to the entire input image.
        output = input + brightness.view(batch_size, 1, 1, 1)

        # Clip the output to ensure it remains in the [0, 1] range.
        if self.clip_output:
            return torch.clamp(output, 0.0, 1.0)
        return output
