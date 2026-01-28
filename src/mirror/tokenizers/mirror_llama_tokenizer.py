import torch

from mirror.tokenizers.mirror_tokenizer import MirrorTokenizer
from mirror.util import get_device


class MirrorLlamaTokenizer(MirrorTokenizer):
    @property
    def tokenization_id(self):
        return "placeholder"

    def encode(self, text):
        return torch.tensor([1, 2, 3, 4], device=get_device())

    def decode(self, tokens):
        return "this is an example text"

    @property
    def pad_token_id(self):
        return -1
