import torchvision.transforms as transforms
import torch
import random
from random import randrange


class Transformer:
    """
    Generate data sample transformation steps from torchvision, called in __getitem__()

    Attributes
    ----------
    config : dict
        config params for model training
    mode: str
        data augmentation stratrgy (default "default")
        "test": no augmentation
        "crop_only": random cropping applied
        "validation": predefined options for validation steps (random cropping applied)
        "default": predefined optimized combination of options

    Returns
    -------
    composed: torchvision.transforms.Compose(transforms)
        A combination of transformation steps with predefined order and probability,
        including random cropping, horizonal and vertical flipping, random perspective,
        affine transformation, rotation and erasing.

    """

    def __init__(self, config: dict, mode: str = "default"):
        self.cfg = config
        assert (
            self.cfg["training"]["crop_size"] >= 16
        ), "Crop size needs to be at least 16 pixels!"
        self.mode = mode

    def __call__(self):
        # data augumentation
        if self.mode == "test":
            return transforms.RandomCrop(
                size=self.cfg["training"]["original_size"]  # no transformation at all
            )
        elif self.mode == "crop_only":
            return transforms.RandomCrop(
                self.cfg["training"]["crop_size"]  # random cropping
            )
        elif self.mode == "validation":
            return transforms.Compose(
                [
                    transforms.RandomCrop(
                        size=self.cfg["training"]["crop_size"]
                    )  # crop at random location with size defined in config
                ]
            )
        elif self.mode == "default":
            return transforms.Compose(
                [
                    # transforms.RandomCrop(
                    #     size=self.cfg["training"]["crop_size"]
                    # ),  # crop at random location with size defined in config
                    transforms.RandomHorizontalFlip(p=0.5),  # 50% flip
                    transforms.RandomVerticalFlip(p=0.5),
                    WorldCoverRemoved(p=0.2),
                    # LanduseDropout(p=0.2),
                    transforms.RandomAffine(
                        180, shear=20
                    ),  # horizontal and vertical translations
                    transforms.RandomCrop(
                        size=self.cfg["training"]["crop_size"]
                    ),  # crop at random location with size defined in config
                    HorizontalMosaic(p=0.2),
                    VerticalMosaic(p=0.2),
                    transforms.RandomErasing(
                        p=0.2, scale=(0.01, 0.05)
                    ),  # remove random small fractions (default values)
                    transforms.RandomErasing(p=0.2, scale=(0.01, 0.05)),
                    transforms.RandomErasing(p=0.2, scale=(0.01, 0.05)),
                ]
            )
        else:
            raise ValueError("Invalid transformation option!")


class WorldCoverRemoved:
    # class LanduseDropout:
    """Convert image to remove land use.
    https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/transforms/transforms.py

    Args:
        p (float): probability that land use should be removed from image.

    Returns:
        Features with the land use class layer removed (replaced by zeros)

    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img: image features for model training

        Returns:
            img: image features with information for land use layer removed
        """
        if random.random() < self.p:
            img[
                -11:, :, :
            ] = 0  # last 11 channels are one hot encoding for land use class
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)


class HorizontalMosaic:
    """Convert image mosaic with horizontal cutline.

    Args:
        p (float): probability that mosaic transformation is applied to image.

    Returns:
        Features with horizontal parts swap with each other, applied to all bands.

    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img: image features for model training

        Returns:
            Features with horizontal parts swap with each other, applied to all bands.
        """
        if random.random() < self.p:
            size = img.size()[-1]
            random_cut = randrange(1, size)
            upper = img[:, :, 0:random_cut].detach().clone()
            lower = img[:, :, random_cut:].detach().clone()
            img[:, :, 0 : (size - random_cut)] = lower
            img[:, :, -random_cut:] = upper
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)


class VerticalMosaic:
    """Convert image mosaic with vertical cutline.

    Args:
        p (float): probability that mosaic transformation is applied to image.

    Returns:
        Features with vertical parts swap with each other, applied to all bands.

    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img: image features for model training

        Returns:
            Features with vertical parts swap with each other, applied to all bands.
        """
        if random.random() < self.p:
            size = img.size()[-2]
            random_cut = randrange(1, size)
            left = img[:, 0:random_cut, :].detach().clone()
            right = img[:, random_cut:, :].detach().clone()
            img[:, 0 : size - random_cut, :] = right
            img[:, -random_cut:, :] = left
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)
