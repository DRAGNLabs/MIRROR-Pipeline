import os
from typing import Callable

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from mirror.util import assert_can_download, mirror_data_path

datasets_path = mirror_data_path / 'datasets'


def load_hf_from_cache_or_download(
        hf_dataset_path: str,
        hf_dataset_name: str | None = None,
        process: Callable[[Dataset | DatasetDict], Dataset] | None = None,
        use_cache: bool = True
) -> Dataset | DatasetDict:
    dataset_path = datasets_path / hf_dataset_path / hf_dataset_name \
        if hf_dataset_name else datasets_path / hf_dataset_path
    is_cached = use_cache and os.path.exists(dataset_path)

    if is_cached:
        ds = load_from_disk(dataset_path)
    else:
        assert_can_download(hf_dataset_path)
        ds = load_dataset(
            hf_dataset_path,
            hf_dataset_name,
            cache_dir=str(datasets_path / hf_dataset_path)
        )
        assert isinstance(ds, Dataset) or isinstance(ds, DatasetDict)
        if process:
            ds = process(ds)
        ds.save_to_disk(dataset_path)

    return ds
