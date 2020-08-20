from torch.utils.data import IterableDataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from collections import defaultdict

class BatchDataset(IterableDataset):
    """ A simple class that wraps a batch of 4-D tensors into a Dataset. """

    def __init__(self, *tensors):
        super().__init__()
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)


def shuffle_data_loader(dataset):
    data_loader = DataLoader(dataset, shuffle=True)
    return data_loader


def preprocess_images(image_dataset, image_size=256):

    image_tensors = [pipeline(i) for i in image_dataset]
    return image_tensor_list


def get_batch_tensors(root, image_size=256):

    # preprocesses images to (image_size x image_size), then converts to tensors
    pipeline = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),  # makes sure images are square
        T.ToTensor()
    ])

    image_folder = ImageFolder(root)
    class_names = image_folder.classes
    batches_by_class = defaultdict(list)

    for image_PIL, class_index in image_folder:
        class_name = class_names[class_index]
        batches_by_class[class_name].append(image_PIL)

    return batches_by_class, class_names
