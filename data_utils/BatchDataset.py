from torch.utils.data import IterableDataset

class BatchDataset(IterableDataset):
    """
    A simple class that wraps a batch of 4-D tensors into an IterableDataset.
    """

    def __init__(self, *tensors):
        super().__init__()
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

