from __future__ import annotations

import math
from typing import cast

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from mirror.datasets.mirror_dataset import MirrorDataset


class MixedDataset[RawT](MirrorDataset[RawT]):
    def __init__(
        self,
        datasets: list[MirrorDataset],
        weights: list[float] | None = None,
        start_fraction: float = 0.0,
        end_fraction: float = 1.0,
    ) -> None:
        super().__init__()

        if not 0.0 <= start_fraction <= end_fraction <= 1.0:
            raise ValueError(f"Invalid fractions: start={start_fraction}, end={end_fraction}")
        if weights is None:
            weights = [1.0] * len(datasets)
        if len(weights) != len(datasets):
            raise ValueError(f"weights ({len(weights)}) and datasets ({len(datasets)}) must align")

        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        scale = max(len(ds) / w for ds, w in zip(datasets, normalized_weights))

        selected: list[HFDataset] = []

        for ds, w in zip(datasets, normalized_weights):
            target_count = math.ceil(scale * w)
            start = int(start_fraction * target_count)
            end = int(end_fraction * target_count)
            ds_len = len(ds)
            upsampled = [i % ds_len for i in range(start, end)]
            selected.append(ds.ds.select(upsampled))

        self._ds = concatenate_datasets(selected)

    @property
    def ds(self) -> HFDataset:
        return self._ds

    def to_row_type(self, ds_row: dict) -> RawT:
        return cast(RawT, ds_row)

    def __getitem__(self, index: int) -> RawT:
        return cast(RawT, self._ds[index])

    def __len__(self) -> int:
        return len(self.ds)
