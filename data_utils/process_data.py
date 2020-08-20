import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from collections import defaultdict
from .BatchDataset import BatchDataset


def folder_to_tensors(root, X_name, Y_name, image_size=256):
    """
    Inputs:
    - [String] root: path containing subdirectories of images
    - [String] X_name: name of image folder for the X dataset (e.g., "horse")
    - [String] Y_name: name of image folder for the Y dataset (e.g., "zebra")
    - [int] image_size: dimension to scale images

    Returns tuple of BatchDatasets (i.e., iterable list of image tensors) for
    root/X_name and root/Y_name.
    """

    batches_by_class = get_batch_tensors(root, image_size)
    X_tensors = batches_by_class[X_name]
    Y_tensors = batches_by_class[Y_name]
    X_dataset = BatchDataset(X_tensors)
    Y_dataset = BatchDataset(Y_tensors)
    return X_dataset, Y_dataset


def get_batch_tensors(root, image_size=256):
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


def shuffled_data_loader(dataset):
    """
    Inputs:
    [torch.utils.Dataset] dataset - our dataset

    Returns a shuffled DataLoader. This means the DataLoader will re-shuffle the
    dataset whenever all images in the set have been iterated through.
    """
    data_loader = DataLoader(dataset, shuffle=True)
    return data_loader
