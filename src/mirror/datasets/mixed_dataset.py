from __future__ import annotations

import math
from typing import Sequence

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from mirror.datasets.mirror_dataset import MirrorDataset


class MixedDataset[RawT](MirrorDataset[RawT]):
    """Combines multiple MirrorDataset sub-datasets with configurable sampling weights.

    Uses upsampling: every item from each sub-dataset appears at least once, with items
    repeated to achieve the specified sampling proportions.
    """

    def __init__(
        self,
        weighted_datasets: Sequence[tuple[MirrorDataset[RawT], float]],
    ) -> None:
        """Initialize a mixed dataset from weighted sub-datasets.

        Args:
            weighted_datasets: Sequence of (dataset, weight) tuples. Weights are
                normalized internally and may be any positive values.

        Raises:
            ValueError: If weighted_datasets is empty or weights are invalid.
        """
        super().__init__()

        if not weighted_datasets:
            raise ValueError("weighted_datasets must not be empty")

        total_weight = sum(w for _, w in weighted_datasets)
        if total_weight <= 0:
            raise ValueError("weights must sum to a positive value")

        norm_weights = [w / total_weight for _, w in weighted_datasets]

        # Compute scale: max dataset size relative to its normalized weight.
        # scale * norm_weight[i] >= len(dataset[i]) for all i, ensuring full coverage.
        scale = max(len(ds) / w for (ds, _), w in zip(weighted_datasets, norm_weights))

        selected: list[HFDataset] = []
        indices: list[tuple[int, int]] = []

        for ds_idx, ((ds, _), w) in enumerate(zip(weighted_datasets, norm_weights)):
            target_count = math.ceil(scale * w)
            ds_len = len(ds)
            upsampled = [i % ds_len for i in range(target_count)]

            selected.append(ds.ds.select(upsampled))

            for i in range(target_count):
                indices.append((ds_idx, i % ds_len))

        self._ds = concatenate_datasets(selected)
        self._datasets = [ds for ds, _ in weighted_datasets]
        self._indices = indices

    @property
    def ds(self) -> HFDataset:
        return self._ds

    def __getitem__(self, index: int) -> RawT:
        ds_idx, item_idx = self._indices[index]
        return self._datasets[ds_idx][item_idx]

    def __len__(self) -> int:
        return len(self._indices)
