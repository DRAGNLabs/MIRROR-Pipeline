from __future__ import annotations
from typing import Sized
from torch.utils.data import Dataset
from abc import abstractmethod

class MirrorDataset[RawT](Dataset[RawT], Sized):
    @property
    @abstractmethod
    def dataset_id(self) -> str:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> RawT:
        pass
