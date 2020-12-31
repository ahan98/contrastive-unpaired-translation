from abc import ABC
from typing import Any, Tuple, Optional
import bbml
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import ImageFolder
from collections import defaultdict


class TrainingData(bbml.SafeTupleIterator, ABC):

    __x_iterator = None
    __y_iterator = None
    __next_tuple: Optional[Tuple[Any, Any]]

    def __init__(self, data_set_path: str, x_class_name: str, y_class_name: str, image_size: int = 256):
        super().__init__()

        # preprocess images to (image_size x image_size), then convert to tensors
        transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),  # makes images square
            T.ToTensor()
        ])

        image_folder = ImageFolder(data_set_path, transform=transform)
        class_names = image_folder.classes
        batches_by_class = defaultdict(list)

        # TODO: I kept the logic intact from the original version but we should rewrite this
        for image_PIL, class_index in image_folder:
            class_name = class_names[class_index]
            batches_by_class[class_name].append(image_PIL)

        self.__x_data_loader = self.__make_x_data_loader(batches_by_class, x_class_name)
        self.__y_data_loader = self.__make_y_data_loader(batches_by_class, y_class_name)

        self.reset()

    @staticmethod
    def __make_x_data_loader(batches_by_class: dict, class_name: str) -> DataLoader:
        batch = batches_by_class[class_name]
        batch = torch.stack(batch)
        return DataLoader(batch, shuffle=True, batch_size=1, num_workers=8)

    @staticmethod
    def __make_y_data_loader(batches_by_class: dict, class_name: str) -> DataLoader:
        batch = batches_by_class[class_name]
        batch = torch.stack(batch)
        sampler = RandomSampler(batch, replacement=True)
        return DataLoader(batch, batch_size=1, sampler=sampler, num_workers=8)

    ''' SafeTupleIterator '''

    def has_next(self) -> bool:
        return self.__next_tuple is not None

    def next(self) -> Optional[Tuple[Any, Any]]:
        next_tuple = self.__next_tuple
        self.__next_tuple = self.__load_next_tuple()
        return next_tuple

    def reset(self):
        self.__x_iterator = iter(self.__x_data_loader)
        self.__next_tuple = self.__load_next_tuple()

    def __load_next_tuple(self) -> Optional[Tuple[Any, Any]]:
        try:
            x_val = next(self.__x_iterator)
        except StopIteration:
            return None

        try:
            if self.__y_iterator is None:
                raise StopIteration

            y_val = next(self.__y_iterator)
        except StopIteration:
            self.__y_iterator = iter(self.__y_data_loader)
            y_val = next(self.__y_iterator)

        return x_val, y_val
