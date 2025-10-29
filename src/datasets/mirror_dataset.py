from torch import Tensor
from torch.utils.data import IterableDataset
from abc import abstractmethod
from typing import Iterator, Tuple
from jaxtyping import Int

TokenTensor = Int[Tensor, "T"]
AttentionMask = Int[Tensor, "T"]


class MirrorDataset(IterableDataset):
    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[TokenTensor, AttentionMask]]:
        pass
