from __future__ import annotations
from typing import Sized, Sequence
from torch.utils.data import Dataset
from abc import abstractmethod

from mirror.datasets.util import load_tokenized_from_cache
from mirror.tokenizers.mirror_tokenizer import MirrorTokenizer
class MirrorDataset[RawT](Dataset[RawT], Sized):
    @property
    @abstractmethod
    def dataset_id(self) -> str:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> RawT:
        pass

    def preprocess(self, tokenizer: MirrorTokenizer, reset_cache: bool = False): # add types 
        dataset_id = f"{self.dataset_id}_TKID-{tokenizer.tokenization_id}".replace("/","-")

        self.ds, is_cached = load_tokenized_from_cache(
            dataset = self.ds,
            dataset_id = dataset_id,
            tokenizer_function = tokenizer.create_hf_map_function_(), #can do input_column
            reset_cache = reset_cache,
            preprocess = self.preprocess_,
        )

    def is_preprocessed(self, tokenizer: MirrorTokenizer):
        dataset_id = f"{self.dataset_id}_TKID-{tokenizer.tokenization_id}".replace("/","-")

        ds, is_cached = load_tokenized_from_cache(
            dataset = None,
            dataset_id = dataset_id,
            tokenizer_function = tokenizer.create_hf_map_function_(), #can do input_column
            preprocess = False,
        )

        if is_cached:
            ds.set_format(
                type='torch', 
                columns=['input_ids'], # 'attention_mask', 'labels',
            ) # THIS NEEDS TO FUNCTION DIFFERENTLY DEPENDING ON INPUT TYPE
            
            self.examples: Sequence[str] = ds[self.split]['input_ids']  # pyright: ignore
            if self.head:
                self.examples = self.examples[:self.head]
            return True
        else:
            self.examples: Sequence[str] = self.ds[self.split]['text']  # pyright: ignore
            if self.head:
                self.examples = self.examples[:self.head]
            return False
