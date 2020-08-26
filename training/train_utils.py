"""
A collection of helper methods for managing datasets.
"""

import os
from glob import glob
from collections import defaultdict
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import ImageFolder


def make_dataloader(batches_by_class, class_name, image_size=256,
                    minibatch_size=1, replacement=False):
    """
    Returns a Dataloader from the batch of tensor images stored in
    batches_by_class[class_name].

    Inputs:
    - [dict] batches_by_class: dictionary of class names to batch
    - [String] class_name: class name of desired batch
    - [int] image_size: dimension to scale images
    - [int] minibatch_size: number of samples to draw from X and Y
    - [bool] replacement: if True, create DataLoader with a RandomSampler that
      samples with replacement
    """

    # get batch for path/class_name
    batch = batches_by_class[class_name]

    # condense batches into (N+1)-dimensional tensors
    batch = torch.stack(batch)

    # convert to dataloader
    if replacement:
        sampler = RandomSampler(batch, replacement=True)
        dataloader = DataLoader(batch, batch_size=minibatch_size,
                                sampler=sampler, num_workers=10)
    else:
        dataloader = DataLoader(batch, shuffle=True, batch_size=minibatch_size,
                                num_workers=10)

    return dataloader


def get_batches_from_path(path, image_size=256):
    """
    Returns a dictionary mapping each subdirectory to a batch of its images.

    Inputs:
    - [String] path: path containing subdirectories of images
    - [int] image_size: dimension to scale images
    """

    # preprocess images to (image_size x image_size), then convert to tensors
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),  # makes images square
        T.ToTensor()
    ])

    image_folder = ImageFolder(path, transform=transform)
    class_names = image_folder.classes
    batches_by_class = defaultdict(list)

    for image_PIL, class_index in image_folder:
        class_name = class_names[class_index]
        batches_by_class[class_name].append(image_PIL)

    return batches_by_class


def purge_files(filelist):
    for file_ in glob(filelist):
        os.remove(file_)
