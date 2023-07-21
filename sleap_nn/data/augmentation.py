from typing import List, Tuple, Union
import torchdata.datapipes.iter as dp
import torch
import kornia.augmentation as K


class KorniaAugmenter(dp.IterDataPipe):
    """DataPipe for applying Rotation and Scaling augmentations using Kornia

    This DataPipe will generate augmented samples containing the augmented frame and an sleap_io.Instance
    from sleap_io.Labels instance.

    Attributes:
        source_dp: DataPipe which is an instance of the LabelsReader class
        rotation: range of degrees to select from. If float, randomly selects a value from (-rotation, +rotation)
        probability: probability of applying transformation
        scale: scaling factor interval. Randomly selects a scale from the range

    """

    def __init__(
        self,
        source_dp: dp.IterDataPipe,
        rotation: Union[float, Tuple[float, float], List[float]] = 90,
        probability: float = 0.5,
        scale: Tuple[float, float] = (0.1, 0.3),
    ):
        """Initialize the class variables with the DataPipe and the augmenter with rotation and scaling"""

        self.source_dp = source_dp
        self.datapipe = self.source_dp.map(self.normalize)
        self.augmenter = K.AugmentationSequential(
            K.RandomRotation(degrees=rotation, p=probability, keepdim=True),
            K.RandomAffine(degrees=0, scale=scale, keepdim=True),
            data_keys=["input", "keypoints"],
            keepdim=True,
        )

    @classmethod
    def normalize(self, data):
        """Function to normalize the image

        This function will convert the image to type Double and normalizes it.

        Args:
            data: A dictionary sample (`image and key-points`) from the LabelsReader class

        Returns:
            A dictionary with the normalized image and instance

        """

        image = data["image"]
        instance = data["instance"]
        image = image.type(torch.DoubleTensor)
        image = image / 255
        return {"image": image, "instance": instance}

    def __iter__(self):
        """Returns a dictionary sample with the augmented image and the transformed instance"""

        for dict in self.datapipe:
            image = dict["image"]
            instance = dict["instance"]
            aug_image, aug_instance = self.augmenter(image, instance)
            yield {"image": aug_image, "instance": aug_instance}
