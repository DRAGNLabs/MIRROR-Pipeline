from __future__ import annotations

import math
from typing import Sequence

from mirror.datasets.mirror_dataset import MirrorDataset


class MixedDataset[RawT](MirrorDataset[RawT]):
    """A dataset that combines multiple datasets of the same type into one.

    Sampling rates are specified per sub-dataset. Upsampling is used so that
    every item from every sub-dataset appears at least once. Some items may be
    repeated in order to satisfy the requested sampling proportions.

    Args:
        datasets: The sub-datasets to combine.
        sampling_rates: The desired proportion of samples to draw from each
            sub-dataset.  Values are relative (they will be normalized to sum
            to 1), so ``[1, 9]`` is equivalent to ``[0.1, 0.9]``.  If
            omitted, each sub-dataset is sampled equally.

    Example::

        mixed = MixedDataset(
            datasets=[dataset_a, dataset_b],
            sampling_rates=[0.1, 0.9],
        )
    """

    def __init__(
        self,
        datasets: Sequence[MirrorDataset[RawT]],
        sampling_rates: Sequence[float] | None = None,
    ) -> None:
        super().__init__()

        if len(datasets) == 0:
            raise ValueError("At least one sub-dataset must be provided.")

        if sampling_rates is None:
            sampling_rates = [1.0] * len(datasets)

        if len(sampling_rates) != len(datasets):
            raise ValueError(
                "The number of sampling rates must match the number of datasets."
            )

        if any(r <= 0 for r in sampling_rates):
            raise ValueError("All sampling rates must be positive.")

        self._datasets = list(datasets)

        # Normalise so that proportions sum to 1.
        total = sum(sampling_rates)
        self._rates = [r / total for r in sampling_rates]

        # Build an index list via upsampling.
        # Each entry is (dataset_index, item_index_within_dataset).
        self._index: list[tuple[int, int]] = self._build_index()

    # ------------------------------------------------------------------
    # MirrorDataset interface
    # ------------------------------------------------------------------

    @property
    def ds(self):
        raise NotImplementedError(
            "MixedDataset does not expose a single underlying HuggingFace dataset. "
            "Use the individual sub-datasets instead."
        )

    def __getitem__(self, index: int) -> RawT:
        ds_idx, item_idx = self._index[index]
        return self._datasets[ds_idx][item_idx]

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_index(self) -> list[tuple[int, int]]:
        """Build the upsampled index list.

        The strategy is:
        1. Determine the minimum number of *total* samples N such that, for
           every sub-dataset i, ``floor(N * rate_i) >= len(datasets[i])``.
           This guarantees that each item appears at least once.
        2. Enumerate ``floor(N * rate_i)`` samples for each sub-dataset by
           cycling through its items (i.e., wrapping around with modulo when
           we need more samples than the dataset contains).
        """
        sizes = [len(ds) for ds in self._datasets]
        rates = self._rates

        # Minimum total samples to guarantee full coverage of every dataset.
        # For dataset i we need: floor(N * rate_i) >= sizes[i]
        # => N >= sizes[i] / rate_i
        min_total = max(
            math.ceil(size / rate) for size, rate in zip(sizes, rates)
        )

        index: list[tuple[int, int]] = []
        for ds_idx, (size, rate) in enumerate(zip(sizes, rates)):
            n_samples = max(math.floor(min_total * rate), size)
            for k in range(n_samples):
                index.append((ds_idx, k % size))

        return index
