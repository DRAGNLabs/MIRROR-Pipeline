import os
import shutil
from sys import stderr
from typing import Callable, Sequence
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from mirror.util import is_compute_node

from mirror.download_util import assert_can_download, mirror_data_path

datasets_path = mirror_data_path / 'datasets'


def load_hf_dataset(
        hf_dataset_path: str,
        hf_dataset_name: str | None = None,
        process: Callable[[Dataset | DatasetDict], Dataset | DatasetDict] | None = None,
        reset_cache: bool = False,
) -> Dataset | DatasetDict:
    """
    The first time this is called with a particular path/name pair, it will download
    the dataset from huggingface and apply the process function to it. Thereafter,
    if it is called again with reset_cache=False, it will used the cached data from the
    first run. Note that if you have changed your process method, you'll need to set
    reset_cache=True to run the new processing.

    Also note that the cached data will be used *any* time the path/name pair is passed.
    This means if your process function should not rely on information that might change
    independently of the path/name (such as split).
    """
    dataset_path = datasets_path / hf_dataset_path / hf_dataset_name \
        if hf_dataset_name else datasets_path / hf_dataset_path

    if not is_compute_node() and reset_cache:
        shutil.rmtree(dataset_path, ignore_errors=True)
    elif is_compute_node() and reset_cache:
        print("The attempted cache reset was aborted because it was initiated on a compute node.", file=stderr)

    is_cached = os.path.exists(dataset_path)

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
