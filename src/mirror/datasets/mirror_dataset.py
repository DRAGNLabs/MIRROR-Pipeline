from __future__ import annotations
from typing import Callable, Sequence, Sized
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from abc import abstractmethod
from sys import stderr

class MirrorDataset[RawT](Dataset[RawT], Sized):
    @property
    @abstractmethod
    def ds(self) -> HFDataset:
        pass

    @ds.setter
    @abstractmethod
    def ds(self, value: HFDataset) -> None:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> RawT:
        pass

    def preprocess[ProcessedT](self, preprocessor_function: Callable[[RawT], ProcessedT]) -> Sequence[ProcessedT]:

        def mappable_preprocessor_function(row: RawT) -> dict[str, ProcessedT]:
            return {"input_ids": preprocessor_function(row)}

        self.ds = self.ds.map(mappable_preprocessor_function)
        print("Preprocessing complete.", file=stderr)
        self.ds.set_format(type="torch", columns=["input_ids"])

        return self.ds["input_ids"]
    