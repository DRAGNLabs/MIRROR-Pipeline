from abc import abstractmethod
from sys import stderr
from typing import Any, Callable, Sequence, Sized
from torch.utils.data import Dataset
from typed_datasets import TypedDataset
from mirror.util import _ds_cache_path_context


class MirrorDataset[RawT](Dataset[RawT], Sized):
    @property
    @abstractmethod
    def ds(self) -> TypedDataset[Any]:
        pass

    def to_row_type(self, ds_row: RawT) -> RawT:
        return ds_row

    def __getitem__(self, index: int) -> RawT:
        return self.to_row_type(self.ds[index])


def preprocess_dataset[RawT, ProcessedT](
    dataset: MirrorDataset[RawT],
    preprocessor_function: Callable[[RawT], ProcessedT],
) -> Sequence[ProcessedT]:
    def mappable_preprocessor_function(row: RawT) -> dict:
        return {"input_ids": preprocessor_function(dataset.to_row_type(row))}

    with _ds_cache_path_context():
        mapped = dataset.ds.map(mappable_preprocessor_function).unwrap()

    print("Preprocessing complete.", file=stderr)
    mapped.set_format(type="torch", columns=["input_ids"])

    return mapped["input_ids"]
