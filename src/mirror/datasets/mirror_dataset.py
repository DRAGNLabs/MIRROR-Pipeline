from abc import abstractmethod
from sys import stderr
from typing import Any, Callable, Mapping, Sequence, Sized, cast
from torch.utils.data import Dataset
from typed_datasets import TypedDataset
from mirror.util import _ds_cache_path_context


class MirrorDataset[RawT: Mapping[str, Any]](Dataset[RawT], Sized):
    @property
    @abstractmethod
    def ds(self) -> TypedDataset[RawT]:
        pass

    def to_row_type(self, ds_row: Mapping[str, Any]) -> RawT:
        return cast(RawT, ds_row)

    def __getitem__(self, index: int) -> RawT:
        return self.to_row_type(self.ds[index])


def preprocess_dataset[RawT: Mapping[str, Any], ProcessedT: Mapping[str, Any]](
    dataset: MirrorDataset[RawT],
    preprocessor_function: Callable[[RawT], ProcessedT],
) -> Sequence[ProcessedT]:
    raw = dataset.ds.unwrap()

    def mappable_preprocessor_function(row: dict) -> ProcessedT:
        return preprocessor_function(dataset.to_row_type(row))

    with _ds_cache_path_context():
        mapped = raw.map(mappable_preprocessor_function, remove_columns=raw.column_names)

    print("Preprocessing complete.", file=stderr)

    return cast(Sequence[ProcessedT], mapped)
