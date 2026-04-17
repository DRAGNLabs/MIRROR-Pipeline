from __future__ import annotations

import math
from typing import Sequence, cast

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from mirror.datasets.mirror_dataset import MirrorDataset


class MixedDataset[RawT](MirrorDataset[RawT]):
    def __init__(
        self,
        weighted_datasets: Sequence[tuple[MirrorDataset[RawT], float]],
    ) -> None:
        super().__init__()

        total_weight = sum(w for _, w in weighted_datasets)
        normalized_weights = [w / total_weight for _, w in weighted_datasets]
        scale = max(len(ds) / w for (ds, _), w in zip(weighted_datasets, normalized_weights))

        selected: list[HFDataset] = []

        for (ds, _), w in zip(weighted_datasets, normalized_weights):
            target_count = math.ceil(scale * w)
            ds_len = len(ds)
            upsampled = [i % ds_len for i in range(target_count)]
            selected.append(ds.ds.select(upsampled))

        self._ds = concatenate_datasets(selected)

    @property
    def ds(self) -> HFDataset:
        return self._ds

    def __getitem__(self, index: int) -> RawT:
        return cast(RawT, self._ds[index])

    def __len__(self) -> int:
        return len(self.ds)
