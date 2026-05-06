from __future__ import annotations
from bisect import bisect_right
from typing import Callable, Sequence
from torch.utils.data import Dataset

from mirror.datasets.mirror_dataset import MirrorDataset

class OnDemandPreprocessedDataset[RawT, ProcessedT](Dataset[ProcessedT]):
    def __init__(
            self,
            raw_dataset: MirrorDataset[RawT],
            preprocess: Callable[[RawT], Sequence[ProcessedT]]
    ):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.preprocess = preprocess
        # cumulative_lengths[i] = total processed items produced by raw_dataset[:i+1]
        cumulative = 0
        self._cumulative_lengths: list[int] = []
        for i in range(len(raw_dataset)):
            cumulative += len(preprocess(raw_dataset[i]))
            self._cumulative_lengths.append(cumulative)

    def __len__(self) -> int:
        return self._cumulative_lengths[-1] if self._cumulative_lengths else 0

    def __getitem__(self, index: int) -> ProcessedT:
        raw_index = bisect_right(self._cumulative_lengths, index)
        prev = self._cumulative_lengths[raw_index - 1] if raw_index > 0 else 0
        sub_index = index - prev
        return self.preprocess(self.raw_dataset[raw_index])[sub_index]
