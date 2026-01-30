from __future__ import annotations
from typing import Callable
import torch
from torch.utils.data import Dataset

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.types import AttentionMask, TokenTensor
from mirror.tokenizers.mirror_tokenizer import MirrorTokenizer
from mirror.util import mirror_data_path

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
        # TODO: use cached preprocessed data
        return self.preprocess(self.raw_dataset[index])
