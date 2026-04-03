from __future__ import annotations
from abc import abstractmethod
from sys import stderr
from typing import TYPE_CHECKING, Callable, Sequence, Sized
from torch.utils.data import Dataset
from mirror.util import _ds_cache_path_context

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset

class MirrorDataset[RawT](Dataset[RawT], Sized):
    @property
    @abstractmethod
    def ds(self) -> HFDataset:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> RawT:
        pass

    def preprocess[ProcessedT](self, preprocessor_function: Callable[[RawT], ProcessedT]) -> Sequence[ProcessedT]:

        def mappable_preprocessor_function(row: RawT) -> dict[str, ProcessedT]:
            return {"input_ids": preprocessor_function(row)}
        
        with _ds_cache_path_context():
            mapped = self.ds.map(mappable_preprocessor_function)

        print("Preprocessing complete.", file=stderr)
        mapped.set_format(type="torch", columns=["input_ids"])

        return mapped["input_ids"]
    