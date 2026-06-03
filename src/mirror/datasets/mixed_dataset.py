from __future__ import annotations

import math
from typing import Any, Mapping

from typed_datasets import TypedDataset, concatenate

from mirror.datasets.mirror_dataset import MirrorDataset


class MixedDataset[RawT: Mapping[str, Any]](MirrorDataset[RawT]):
    # NOTE: parallel `datasets` + `weights` lists rather than the upstream
    # `weighted_datasets: Sequence[tuple[MirrorDataset, float]]` — jsonargparse can't
    # resolve a generic `MirrorDataset[RawT]` nested inside a tuple, so the tuple form
    # fails to parse from YAML. See the SSP mixed config.
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

        selected: list[TypedDataset[RawT]] = []

        for ds, w in zip(datasets, normalized_weights):
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
