from __future__ import annotations
from typing import Callable, Sequence, Sized
from torch.utils.data import Dataset
from abc import abstractmethod
from sys import stderr

class MirrorDataset[RawT](Dataset[RawT], Sized):

    @abstractmethod
    def __getitem__(self, index: int) -> RawT:
        pass

    def preprocess[ProcessedT](self, preprocessor_function: Callable[[RawT], ProcessedT], num_proc:int = 1) -> Sequence[ProcessedT]:

        def mappable_preprocessor_function(row: RawT) -> dict[str, ProcessedT]:
            return {"input_ids": preprocessor_function(row)}

        self.ds = self.ds.map(mappable_preprocessor_function, num_proc=num_proc)
        print("Preprocessing complete.", file=stderr)
        self.ds.set_format(type="torch", columns=["input_ids"])

        return self.ds["input_ids"]
    