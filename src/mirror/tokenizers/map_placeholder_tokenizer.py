import torch

from mirror.tokenizers.mappable_tokenizer.py import MappableTokenizer
from mirror.util import device


class MapPlaceholderTokenizer(MappableTokenizer):
    @property
    def tokenization_id(self):
        return "placeholder"

    def encode(self, text) -> List[int]:
        return [1, 2, 3, 4]

    def decode(self, tokens):
        return "this is an example text"

    @property
    def pad_token_id(self):
        return -1