from __future__ import annotations
from typing import Generic, Sized, TypeVar
from torch.utils.data import Dataset
from abc import abstractmethod
from mirror.types import RawT

class MirrorDataset(Dataset[RawT], Sized, Generic[RawT]):
    @property
    @abstractmethod
    def dataset_id(self) -> str:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> RawT:
        pass
