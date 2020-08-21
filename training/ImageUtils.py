import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from collections import defaultdict


class ImageUtils:
    """
    A collection of methods to maniulate images and their tensor
    representations.
    """

    @staticmethod
    def make_dataloader_from_folder(root, X_name, Y_name, image_size=256,
                                    minibatch_size=1):
        """
        Inputs:
        - [String] root: path containing subdirectories of images
        - [String] X_name: name of image folder for the X dataset (e.g., "horse")
        - [String] Y_name: name of image folder for the Y dataset (e.g., "zebra")
        - [int] image_size: dimension to scale images

        Returns DataLoaders for tensor images in folders X_name and Y_name.
        Note that samples from Y_dataloader are randomly drawn.
        """

        batches_by_class = ImageUtils.make_batch_from_folder(root, image_size)

        X_batch_list = batches_by_class[X_name]
        X_batch = torch.stack(X_batch_list)
        X_dataloader = DataLoader(X_batch)

        Y_batch_list = batches_by_class[Y_name]
        Y_batch = torch.stack(Y_batch_list)
        Y_dataloader = DataLoader(Y_batch, shuffle=True)

        return X_dataloader, Y_dataloader


    @staticmethod
    def make_batch_from_folder(root, image_size=256):
        """
        Inputs:
        - [String] root: path containing subdirectories of images
        - [int] image_size: dimension to scale images

        Returns a dictionary mapping subdirectory names to a list of PIL images.
        """

        # preprocess images to (image_size x image_size), then convert to tensors
        pipeline = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),  # makes images square
            T.ToTensor()
        ])

        image_folder = ImageFolder(root)
        class_names = image_folder.classes
        batches_by_class = defaultdict(list)

        for image_PIL, class_index in image_folder:
            class_name = class_names[class_index]
            batches_by_class[class_name].append(image_PIL)

        return batches_by_class


    # TODO: write method to convert tensors to images
    def tensor_to_image():
        pass
