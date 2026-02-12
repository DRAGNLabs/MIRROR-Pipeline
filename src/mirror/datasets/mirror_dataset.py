from __future__ import annotations
from typing import Sized, Sequence
from torch.utils.data import Dataset
from abc import abstractmethod
import os

from mirror.tokenizers.mirror_tokenizer import MirrorTokenizer
from mirror.util import mirror_data_path

class MirrorDataset[RawT](Dataset[RawT], Sized):
    @property
    @abstractmethod
    def dataset_id(self) -> str:
        pass

    @property
    @abstractmethod
    def ds(self) -> str:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> RawT:
        pass

    def is_preprocessed(self, tokenization_id: str):
        dataset_id = f"{MirrorDataset.dataset_id}_TKID-{tokenization_id}".replace("/","-")
        preprocessed_dataset_path = mirror_data_path / f'tokenized_data/{dataset_id}'
        
        is_cached = os.path.exists(preprocessed_dataset_path)

        return is_cached