import os
from typing import Callable

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from mirror.util import assert_can_download, mirror_data_path

datasets_path = mirror_data_path / 'datasets'


def load_hf_from_cache_or_download(
        hf_dataset_path: str,
        hf_dataset_name: str | None = None,
        process: Callable[[Dataset | DatasetDict], Dataset | DatasetDict] | None = None,
        use_cache: bool = True
) -> Dataset | DatasetDict:
    """
    The first time this is called with a particular path/name pair, it will download
    the dataset from huggingface and apply the process function to it. Thereafter,
    if it is called again with use_cache=True, it will used the cached data from the
    first run. Note that if you have changed your process method, you'll need to set
    use_cache=False to run the new processing.
    """
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
