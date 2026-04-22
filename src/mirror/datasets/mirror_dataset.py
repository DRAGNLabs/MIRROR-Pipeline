from abc import abstractmethod
from sys import stderr
from typing import Callable, Sequence, Sized
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from mirror.util import _ds_cache_path_context

class MirrorDataset[RawT](Dataset[RawT], Sized):
    @property
    @abstractmethod
    def ds(self) -> HFDataset:
        pass

    @abstractmethod
    def to_row_type(self, ds_row: dict) -> RawT:
        pass

    def __getitem__(self, index: int) -> RawT:
        return self.to_row_type(self.ds[index])

    def preprocess[ProcessedT](self, preprocessor_function: Callable[[RawT], ProcessedT], num_nodes: int) -> Sequence[ProcessedT]:

        def mappable_preprocessor_function(row: dict) -> dict[str, ProcessedT]:
            return {"input_ids": preprocessor_function(self.to_row_type(row))}

        with _ds_cache_path_context():
            mapped = self.ds.map(mappable_preprocessor_function, num_proc=num_nodes)

        print("Preprocessing complete.", file=stderr)
        mapped.set_format(type="torch", columns=["input_ids"])

        return mapped["input_ids"]
