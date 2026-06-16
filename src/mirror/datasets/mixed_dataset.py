from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from typed_datasets import TypedDataset, concatenate

from mirror.datasets.mirror_dataset import MirrorDataset


class MixedDataset[RawT: Mapping[str, Any]](MirrorDataset[RawT]):
    def __init__(
        self,
        weighted_datasets: Sequence[tuple[MirrorDataset, float]],
        start_fraction: float = 0.0,
        end_fraction: float = 1.0,
    ) -> None:
        super().__init__()

        if not 0.0 <= start_fraction <= end_fraction <= 1.0:
            raise ValueError(f"Invalid fractions: start={start_fraction}, end={end_fraction}")

        total_weight = sum(w for _, w in weighted_datasets)
        normalized_weights = [w / total_weight for _, w in weighted_datasets]
        scale = max(len(ds) / w for (ds, _), w in zip(weighted_datasets, normalized_weights))

        selected: list[TypedDataset[RawT]] = []

        for (ds, _), w in zip(weighted_datasets, normalized_weights):
            target_count = math.ceil(scale * w)
            start = int(start_fraction * target_count)
            end = int(end_fraction * target_count)
            ds_len = len(ds)
            upsampled = [i % ds_len for i in range(start, end)]
            selected.append(ds.ds.select(upsampled))

        self._ds = concatenate(selected)

    @property
    def ds(self) -> TypedDataset[RawT]:
        return self._ds

    def __len__(self) -> int:
        return len(self.ds)
