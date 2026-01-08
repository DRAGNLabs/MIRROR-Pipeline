from typing import Tuple
import torch
from torch.utils.data import Dataset

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.types import AttentionMask, TokenTensor
from mirror.tokenizers.mirror_tokenizer import MirrorTokenizer
from mirror.util import device


class PreprocessedDataset(Dataset):
    def __init__(self, raw_dataset: MirrorDataset, tokenizer: MirrorTokenizer):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index) -> Tuple[TokenTensor, AttentionMask]:
        # TODO: use cached preprocessed data
        item = self.tokenizer.encode(self.raw_dataset[index])
        return item
