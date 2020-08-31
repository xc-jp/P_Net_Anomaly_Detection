import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision


class Transform(object):

    def __init__(self, augment=False, resize_size=None, crop_size=None):
        self._crop_size = crop_size
        self._augment = augment
        if resize_size is None:
            self._resize = None
        else:
            self._resize = torchvision.transforms.Resize(resize_size)
        if not augment:
            if crop_size is not None:
                self._center_crop = torchvision.transforms.CenterCrop(crop_size)
            else:
                self._center_crop = None
        self._to_tensor = torchvision.transforms.ToTensor()
    
    def __call__(self, image1, image2):
        if self._resize is not None:
            image1 = self._resize(image1)
            image2 = self._resize(image2)
        if self._augment:
            x, y, h, w = torchvision.transforms.RandomCrop.get_params(image1, self._crop_size)
            image1 = torchvision.transforms.functional.crop(image1, x, y, h, w)
            image2 = torchvision.transforms.functional.crop(image2, x, y, h, w)
            if np.random.random() < 0.5:
                image1 = torchvision.transforms.functional.hflip(image1)
                image2 = torchvision.transforms.functional.hflip(image2)
        else:
            if not self._center_crop is None:
                image1 = self._center_crop(image1)
                image2 = self._center_crop(image2)
        image1 = self._to_tensor(image1)
        image2 = self._to_tensor(image2)
        return image1, image2


class ImageDataset(torch.utils.data.Dataset):
    """
    Images dataset.
    """

    def __init__(self, label_path, image_root, augment=False, resize_size=None, crop_size=None):
        """
        Creates dataset instance.

        Args:
            label_path (str): Label file path.
            image_root (str): Image root directory path.
            transform (Torchvision transformer): Transformer instance.
            augment (bool): If True this dataset will output augmented images.
            resize_size (tuple of ints): Image size to resize.
            crop_size (tuple of ints): Image size to crop.
        """

        self._image_root = image_root
        df = pd.read_csv(label_path, sep=',', header=None, dtype={0: str}, usecols=[0])
        self._image_paths = df.iloc[:, 0].values
        transforms = []
        if resize_size is not None:
            transforms.append(torchvision.transforms.Resize(resize_size))
        if crop_size is not None:
            if augment:
                transforms.append(torchvision.transforms.RandomCrop(crop_size))
            else:
                transforms.append(torchvision.transforms.CenterCrop(crop_size))
        if augment:
            transforms.append(torchvision.transforms.RandomHorizontalFlip())
        transforms.append(torchvision.transforms.ToTensor())
        self._transform = torchvision.transforms.Compose(transforms)

    def __len__(self):
        """
        Returns (int):
            Length of dataset.
        """
        return len(self._image_paths)

    def __getitem__(self, i):
        """
        Returns item.

        Args:
            i (int): Index of item to get.

        Returns (Tuple):
            Tuple of (image file path, PIL image)
            image file path and input image.
        """

        image_path = self._image_paths[i]
        image = Image.open(os.path.join(self._image_root, image_path))
        image = self._transform(image)
        return image, image_path


class SegmentationImageDataset(torch.utils.data.Dataset):
    """
    Images dataset.
    """

    def __init__(self, label_path, image_root, augment=False, resize_size=None, crop_size=None):
        """
        Creates dataset instance.

        Args:
            label_path (str): Label file path.
            image_root (str): Image root directory path.
            transform (Torchvision transformer): Transformer instance.
            augment (bool): If True this dataset will output augmented images.
            resize_size (tuple of ints): Image size to resize.
            crop_size (tuple of ints): Image size to crop.
        """

        self._image_root = image_root
        df = pd.read_csv(label_path, sep=',', header=None, dtype={0: str, 1: str}, usecols=[0, 1])
        self._image_paths = df.iloc[:, 0].values
        self._annotation_paths = df.iloc[:, 1].values
        self._transform = Transform(augment, resize_size, crop_size)

    def __len__(self):
        """
        Returns (int):
            Length of dataset.
        """
        return len(self._image_paths)

    def __getitem__(self, i):
        """
        Returns item.

        Args:
            i (int): Index of item to get.

        Returns (Tuple):
            Tuple of (image file path, PIL image, PIL image)
            image file path, input image and annotation image.
        """

        image_path = self._image_paths[i]
        image = Image.open(os.path.join(self._image_root, image_path))
        annotation_path = self._annotation_paths[i]
        if annotation_path == '':
            annotation = Image.new('L', image.size, 0)
        else:
            annotation = Image.open(os.path.join(self._image_root, annotation_path)).convert('L')
        image, annotation = self._transform(image, annotation)
        return image, annotation, image_path
