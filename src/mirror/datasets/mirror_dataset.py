from __future__ import annotations
from typing import Sized, Sequence
from mirror.types import TokenTensor
from torch.utils.data import Dataset
from abc import abstractmethod
from sys import stderr

class MirrorDataset[RawT](Dataset[RawT], Sized):
    
    @abstractmethod
    def __getitem__(self, index: int) -> RawT:
        pass
    
    def preprocess(self, preprocessor_function) -> Sequence[TokenTensor]:
        
        def mappable_preprocessor_function(row: dict):
            row['input_ids'] = preprocessor_function(row)
            return row

        self.ds = self.ds.map(mappable_preprocessor_function)
        print("Tokenization complete.", file=stderr)
        self.ds.set_format(type="torch", columns=["input_ids"])

        return self.ds["input_ids"]