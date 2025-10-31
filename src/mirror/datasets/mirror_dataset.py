import itertools
import os
from pathlib import Path
import torch
from torch.utils.data import IterableDataset
from abc import ABC, abstractmethod
from typing import Collection, Iterator, Tuple

from mirror.types import TokenTensor, AttentionMask


class MirrorDataset(ABC, IterableDataset):
    def __init__(self, dir_path: Path, head: int | None, max_sequence_embeddings: int):
        super().__init__()
        assert os.path.isdir(dir_path)
        self.head = head
        self.max_sequence_embeddings = max_sequence_embeddings

    @abstractmethod
    def get_example_iter(self) -> Iterator[Collection[int]]:
        """
        Returns an iterator of collections of tokens. Each element yielded by the iterator
        is a single training example.
        """
        pass

    def __iter__(self) -> Iterator[Tuple[TokenTensor, AttentionMask]]:
        for (i, item) in enumerate(self.get_example_iter()):
            if self.head and i >= self.head:
                break

            item_sliced = itertools.islice(item, self.max_sequence_embeddings)
            item_tensor: TokenTensor = torch.tensor(item_sliced, device='auto')
            yield (
                item_tensor,
                torch.ones(item_tensor.shape[0], device='auto')
            )

