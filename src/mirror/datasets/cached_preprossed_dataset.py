import os

from typing import Tuple
import torch
from torch.utils.data import Dataset, load_from_disk

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.types import AttentionMask, TokenTensor
from mirror.tokenizers.mirror_tokenizer import MirrorTokenizer
from mirror.util import device
from mirror.datasets import tokenized_data_path


class CachedPreprocessedDataset(Dataset):
    def __init__(self, raw_dataset: MirrorDataset, tokenizer: MirrorTokenizer):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.tokenizer_id = tokenizer.tokenization_id # A reference to the specific tokenizer that we are using hopefully
        self.dataset_id = f"{raw_dataset.dataset_id}_TKID-{self.tokenizer_id}" # Indicates that data is tokenized by tokenizer
        
        # if data is already tokenized and cached, use that. else tokenize and save.
        dataset_path = tokenized_data_path(self.dataset_id)
        
        if os.exists(dataset_path):
            self.data = load_from_disk(dataset_path) 
            # See mirror/datasets/utils.load_hf_from_cache_or_download
            #I may need to implement something more like that
        else:
            os.makedirs(dataset_path, exist_ok=True)
            raw_dataset.map(tokenizer.encode)
            raw_dataset.save_to_disk(dataset_path = self.dataset_path)
            self.data = raw_dataset

        # fabric.save_to_disk() or perhaps our dataset object has a save_to_disk method

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index) -> TokenTensor:
        # TODO: use cached preprocessed data
        item = self.tokenizer.encode(self.raw_dataset[index])
        return item