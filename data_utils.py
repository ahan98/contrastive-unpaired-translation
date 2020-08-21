import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from collections import defaultdict

def make_dataloaders(path, X_class_name, Y_class_name, X_train_split=0.8,
                     image_size=256, minibatch_size=1):
    """
    Returns DataLoaders for training and validation sets of X, and a single
    DataLoader for the entire batch of Y.

    Note that Y is not split into training and validation sets. During each
    training iteration, a random minibatch is drawn from the entire Y dataset.
    And during validation, we only care about how well the Generator works on
    inputs from X.

    Inputs:
    - [String] path: path containing subdirectories of images
    - [String] X_class_name: name of folder to use for X (e.g., "horses")
    - [String] Y_class_name: name of folder to use for Y (e.g., "zebras")
    - [float] X_train_split: the proportion of X used for training
    - [int] image_size: dimension to scale images
    - [int] minibatch_size: number of samples to draw from X and Y
    """

    # get batches for X and Y
    batches_by_class = DataUtils.make_batches_from_path(path, image_size)
    X_batch = batches_by_class[X_class_name]
    Y_batch = batches_by_class[Y_class_name]

    # get the batch item indices for train/val sets
    X_shuffled_idxs = torch.randperm(len(X_batch))
    partition = int(np.ceil(X_train_split * dataset_size))
    X_train_idxs = X_shuffled_idxs[:partition]
    X_val_idxs = X_shuffled_idxs[partition:]

    # create the train/val sets from their batch item indices
    X_train = X_batch[X_train_idxs]
    X_val = X_batch[X_val_idxs]

    # condense batches into (N+1)-dimensional tensors
    X_train, X_val = torch.stack(X_train), torch.stack(X_val)
    Y_batch = torch.stack(Y_batch)

    # initalize dataloaders
    X_train = DataLoader(X_train, shuffle=True, batch_size=minibatch_size)
    X_val = DataLoader(X_val, batch_size=minibatch_size)

    # define custom sampler for Y, since it is sampled with replacement
    sampler = RandomSampler(Y_batch, replacement=True)
    Y_batch = DataLoader(Y_batch, batch_size=minibatch_size, sampler=sampler)

    return X_train, X_val, Y_batch


def make_batches_from_path(path, image_size=256):
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

