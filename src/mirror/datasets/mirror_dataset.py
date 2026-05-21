from abc import abstractmethod
from sys import stderr
from typing import Callable, Sequence, Sized, cast
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


def preprocess_dataset[RawT, ProcessedT](
    dataset: MirrorDataset[RawT],
    preprocessor_function: Callable[[RawT], ProcessedT],
) -> Sequence[ProcessedT]:
    def mappable_preprocessor_function(row: dict) -> dict:
        return cast(dict, preprocessor_function(dataset.to_row_type(row)))

    with _ds_cache_path_context():
        mapped = dataset.ds.map(
            mappable_preprocessor_function,
            remove_columns=dataset.ds.column_names,
        )

    print("Preprocessing complete.", file=stderr)
    mapped.set_format(type="torch", columns=["input_ids", "labels"])

    return cast(Sequence[ProcessedT], mapped)
