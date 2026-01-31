from __future__ import annotations
from typing import Callable
from torch.utils.data import Dataset

from mirror.datasets.mirror_dataset import MirrorDataset

class PreprocessedDataset[RawT, ProcessedT](Dataset[ProcessedT]):
    def __init__(
            self, 
            raw_dataset: MirrorDataset[RawT], 
            preprocess: Callable[[RawT], ProcessedT]
    ):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> ProcessedT:
        return self.preprocess(self.raw_dataset[index])
        
