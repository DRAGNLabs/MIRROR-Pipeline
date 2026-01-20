import torch
import torch.optim as optim
import torch.nn as nn

from mirror.models.mirror_model import MirrorModel
from mirror.tokenizers.placeholder_tokenizer import PlaceholderTokenizer
from mirror.util import device, pad_to_longest


class PlaceholderModel(MirrorModel):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor([0.0], device=device))
        self._tokenizer = PlaceholderTokenizer()

    @property
    def tokenizer(self):
        return self._tokenizer

    def preprocess_example(self, text: str):
        return self._tokenizer.encode(text)
    
    def training_step(self, batch):
        tokens, attention_mask = batch
        return self.parameter
    
    def collate(self, examples):
        return pad_to_longest(examples, pad_token=self.tokenizer.pad_token_id)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())