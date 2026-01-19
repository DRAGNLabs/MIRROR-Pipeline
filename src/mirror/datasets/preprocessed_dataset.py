from __future__ import annotations
from typing import Callable, Generic, TypeVar

import torch
from torch.utils.data import Dataset

from mirror.datasets.mirror_dataset import MirrorDataset

RawT = TypeVar("RawExampleT")
ProcessedT = TypeVar("ProcessedExampleT")

class PreprocessedDataset(Dataset[ProcessedT], Generic[RawT, ProcessedT]):
    def __init__(self, raw_dataset: MirrorDataset[RawT], preprocess: Callable[[RawT], ProcessedT]):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> ProcessedT:
        # TODO: use cached preprocessed data
        return self.preprocess(self.raw_dataset[index])
