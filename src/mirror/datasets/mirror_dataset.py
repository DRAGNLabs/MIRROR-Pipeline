from abc import abstractmethod
from typing import Any, Mapping, Sized, cast
from torch.utils.data import Dataset
from typed_datasets import TypedDataset


class MirrorDataset[RawT: Mapping[str, Any]](Dataset[RawT], Sized):
    @property
    @abstractmethod
    def ds(self) -> TypedDataset[RawT]:
        pass

    def to_row_type(self, ds_row: Mapping[str, Any]) -> RawT:
        return cast(RawT, ds_row)

    def __getitem__(self, index: int) -> RawT:
        return self.to_row_type(self.ds[index])
