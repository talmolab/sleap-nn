import attr
from typing import Text, Tuple, Dict, Any, Optional
import torch
import torchdata.datapipes as dp
import kornia as K
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor
from kornia.augmentation import random_generator as rg
from kornia.augmentation.utils import _range_bound
from kornia.enhance.adjust import adjust_contrast

class RandomUniformNoise(IntensityAugmentationBase2D):
    """Data transformer for applying random uniform noise to input images.

    This is a custom Kornia augmentation inheriting from `IntensityAugmentationBase2D`.
    Uniform noise within (min_val, max_val) is applied to the entire input image.
    min_val and max_val must satisfy 0 <= min_val <= max_val <= 255. 

    Note: min_val and max_val are int/float between 0 and 255, but Kornia expects
    images to be in float. Thus, min_val/max_val are normalized. The input image
    is expected to be a float tensor between 0 and 1.

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
            keepdim: bool = False
        ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.flags = {"min_val": noise[0], "max_val": noise[1]}
        self.clip_output = clip_output

    def generate_parameters(self, shape: Tuple[int, ...]) -> Dict[str, Tensor]:
        return {}
    
    def apply_transform(
            self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
        ) -> Tensor:
        if "uniform_noise" in params:
            uniform_noise = params["uniform_noise"]
        else:
            uniform_noise = torch.FloatTensor(input.shape).uniform_(flags["min_val"]/255.0, flags["max_val"]/255.0).to(input.device)
            self._params["uniform_noise"] = uniform_noise
        
        if self.clip_output:
            return torch.clamp(input + uniform_noise, 0.0, 1.0)  # RandomGaussianNoise doesn't clamp.
        return input + uniform_noise