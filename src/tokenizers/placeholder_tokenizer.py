import torch

from tokenizers.mirror_tokenizer import MirrorTokenizer


class PlaceholderTokenizer(MirrorTokenizer):
    @property
    def tokenization_id(self):
        return "placeholder"

    def tokenize(self, text):
        return torch.tensor([1, 2, 3, 4], device='cuda')
