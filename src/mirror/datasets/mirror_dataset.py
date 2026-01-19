from __future__ import annotations
from typing import Generic, Sized, TypeVar
from torch.utils.data import Dataset
from abc import abstractmethod

RawExampleT = TypeVar("RawExampleT")

class MirrorDataset(Dataset[RawExampleT], Sized, Generic[RawExampleT]):
    @property
    @abstractmethod
    def dataset_id(self) -> str:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> RawExampleT:
        pass
