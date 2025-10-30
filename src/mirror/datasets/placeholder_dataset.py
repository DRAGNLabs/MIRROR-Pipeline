import torch
from mirror.datasets.mirror_dataset import MirrorDataset


class PlaceholderDataset(MirrorDataset):
    def __iter__(self):
        yield torch.tensor([1, 2, 3, 4]), torch.tensor([1, 1, 1, 1])
