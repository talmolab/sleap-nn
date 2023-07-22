"""This module implements data pipeline blocks for augmentations."""
from typing import Optional
from torch.utils.data.datapipes.datapipe import IterDataPipe
import kornia.augmentation as K


class KorniaAugmenter(IterDataPipe):
    """DataPipe for applying rotation and scaling augmentations using Kornia.

    This DataPipe will apply augmentations to images and instances in examples from the
    input pipeline.

    Attributes:
        source_dp: The input `IterDataPipe` with examples that contain `"instances"` and
            `"image"` keys.
        rotation: Angles in degrees as a scalar float of the amount of rotation. A
            random angle in `(-rotation, rotation)` will be sampled and applied to both
            images and keypoints. Set to 0 to disable rotation augmentation.
        scale: A scaling factor as a scalar float specifying the amount of scaling. A
            random factor between `(1 - scale, 1 + scale)` will be sampled and applied
            to both images and keypoints. If `None`, no scaling augmentation will be
            applied.
        probability: Probability of applying the transformations.

    Notes:
        This block expects the "image" and "instances" keys to be present in the input
        examples.

        The `"image"` key should contain a torch.Tensor of dtype torch.float32
        and of shape `(..., C, H, W)`, i.e., rank >= 3.

        The `"instances"` key should contain a torch.Tensor of dtype torch.float32 and
        of shape `(..., n_instances, n_nodes, 2)`, i.e., rank >= 3.

        The augmented versions will be returned with the same keys and shapes.
    """

    def __init__(
        self,
        source_dp: IterDataPipe,
        rotation: float = 15.0,
        scale: Optional[float] = 0.05,
        probability: float = 0.5,
    ):
        """Initialize the block and the augmentation pipeline."""
        self.source_dp = source_dp
        self.rotation = rotation
        self.scale = (1 - scale, 1 + scale)
        self.probability = probability
        self.augmenter = K.AugmentationSequential(
            K.RandomAffine(
                degrees=self.rotation,
                scale=self.scale,
                p=self.probability,
                keepdim=True,
                same_on_batch=True,
            ),
            data_keys=["input", "keypoints"],
            keepdim=True,
            same_on_batch=True,
        )

    def __iter__(self):
        """Return an example dictionary with the augmented image and instance."""
        for ex in self.source_dp:
            img = ex["image"]
            pts = ex["instances"]
            pts_shape = pts.shape
            pts = pts.reshape(-1, pts_shape[-2], pts_shape[-1])
            img, pts = self.augmenter(img, pts)
            pts = pts.reshape(pts_shape)
            ex["image"] = img
            ex["instances"] = pts
            yield ex
