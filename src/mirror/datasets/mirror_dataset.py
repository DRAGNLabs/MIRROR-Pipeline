from torch.utils.data import IterableDataset
from abc import abstractmethod
from typing import Iterator, Tuple

from mirror.types import TokenTensor, AttentionMask


class MirrorDataset(IterableDataset):
    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[TokenTensor, AttentionMask]]:
        pass
