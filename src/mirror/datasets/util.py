import os
import shutil
from typing import Callable
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from mirror.util import assert_can_download, mirror_data_path

datasets_path = mirror_data_path / 'datasets'


def load_hf_from_cache_or_download(
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

    if reset_cache:
        shutil.rmtree(dataset_path)

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


def tokenized_data_path(dataset_id: str) -> Path:
    return Path(f'/home/{os.environ['USER']}/nobackup/autodelete/mirror_data/tokenized_data/{dataset_id}_tokenized/')


def load_tokenized_from_cache(
        hf_dataset_path: str,
        hf_dataset_name: str | None = None,
        # process: Callable[[Dataset | DatasetDict], Dataset | DatasetDict] | None = None,
        tokenizer_function: Callable = None,
        reset_cache: bool = False,
) -> Dataset | DatasetDict:
    """
    The first time this is called with a particular dataset, it will tokenize
    the dataset. Thereafter, if it is called again with reset_cache=False, it will use 
    the cached tokenized data. Note that if you have changed your tokenizor method, 
    you'll need to set reset_cache=True to run the new processing.

    Also note that the cached data will be used *any* time the path/name pair is passed.
    This means if your process function should not rely on information that might change
    independently of the path/name (such as split).
    """
    dataset_path = datasets_path / hf_dataset_path / hf_dataset_name \
        if hf_dataset_name else datasets_path / hf_dataset_path

    if reset_cache:
        shutil.rmtree(dataset_path)

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
        if tokenizer_function:
            ds = ds.map(tokenizer_function)
        
        ds.save_to_disk(dataset_path)

    return ds
