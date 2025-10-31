import torch

from tokenizers.mirror_tokenizer import MirrorTokenizer


class PlaceholderTokenizer(MirrorTokenizer):
    @property
    def tokenization_id(self):
        return "placeholder"

    def encode(self, text):
        return torch.tensor([1, 2, 3, 4], device='cuda')

    def decode(self, tokens):
        return "this is an example text"
