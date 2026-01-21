import torch
import torch.optim as optim
import torch.nn as nn

from mirror.models.mirror_model import MirrorModel
from mirror.tokenizers.placeholder_tokenizer import PlaceholderTokenizer
from mirror.util import get_device


class PlaceholderModel(MirrorModel):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor([0.0], device=get_device()))
        self._tokenizer = PlaceholderTokenizer()

    @property
    def tokenizer(self):
        return self._tokenizer

    def training_step(self, tokens, attention_mask):
        return self.parameter

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())
