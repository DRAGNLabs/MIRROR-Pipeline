import torch

from mirror.tokenizers.mirror_tokenizer import MirrorTokenizer
from mirror.util import device


class PlaceholderTokenizer(MirrorTokenizer):
    @property
    def tokenization_id(self):
        return "placeholder"

    def encode(self, text):
        L = int(torch.randint(low=2, high=8, size=(1,)).item())
        return torch.randint(low=1, high=5, size=(L,), device=device, dtype=torch.long)

        # return torch.tensor([1, 2, 3, 4], device=device)

    def decode(self, tokens):
        return "this is an example text"

    @property
    def pad_token_id(self):
        return -1