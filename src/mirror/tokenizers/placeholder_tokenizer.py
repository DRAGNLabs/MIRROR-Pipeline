import torch

from mirror.tokenizers.mirror_tokenizer import MirrorTokenizer
from mirror.util import device


class PlaceholderTokenizer(MirrorTokenizer):
    @property
    def tokenization_id(self):
        return "placeholder"

    def encode(self, text):
        return torch.tensor([1, 2, 3, 4], device=device)

    def decode(self, tokens):
        return "this is an example text"
