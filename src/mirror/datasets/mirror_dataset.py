from __future__ import annotations
from typing import Sized, Sequence
from mirror.types import TokenTensor
from typing import List
from torch.utils.data import Dataset
from abc import abstractmethod
# import os

from mirror.tokenizers.mirror_tokenizer import MirrorTokenizer
# from mirror.util import mirror_data_path

class MirrorDataset[RawT](Dataset[RawT], Sized):
    @property
    @abstractmethod
    def dataset_id(self) -> str:
        pass

    # @property
    # @abstractmethod
    # def ds(self) -> str:
    #     pass

    @abstractmethod
    def __getitem__(self, index: int) -> RawT:
        pass

    # def is_preprocessed(self, tokenization_id: str):
    #     dataset_id = f"{MirrorDataset.dataset_id}_TKID-{tokenization_id}".replace("/","-")
    #     preprocessed_dataset_path = mirror_data_path / f'tokenized_data/{dataset_id}'
        
    #     is_cached = os.path.exists(preprocessed_dataset_path)

    #     return is_cached
    
    def preprocess(self, tokenizer_function) -> Sequence[TokenTensor]:
        import dill
        
        def mappable_tokenizer_function(row: dict):
            row['input_ids'] = tokenizer_function(row)
            return row

        def encode(text) -> List[int]:
            return [1, 2, 3, 4]
        
        def preprocess_example(example: dict) -> List[int]:
            return encode(example['text'])
        
        # try a function that accesses the model, tokenizer, and dataset in it
        # make a fake copy of the tokenizer and try that.
        try:
            print('try')
            dill.dumps(preprocess_example)
            print("preprocess example!")
            dill.dumps(tokenizer_function)
            print("Tokenizer function is picklable!")
            dill.dumps(mappable_tokenizer_function)
            print("Mappable function is picklable.")
        except Exception as e:
            print(f"Function is NOT picklable: {e}")

        
        self.ds = self.ds.map(mappable_tokenizer_function)
        self.ds.set_format(type="torch", columns=["input_ids"])


        return self.ds["input_ids"]