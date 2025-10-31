from typing import Sized
from torch import Tensor
from torch.utils.data import Dataset
from abc import abstractmethod
from jaxtyping import Int

TokenTensor = Int[Tensor, "T"]
AttentionMask = Int[Tensor, "T"]


class MirrorDataset(Dataset, Sized):
    @property
    @abstractmethod
    def dataset_id(self) -> str:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> str:
        pass
