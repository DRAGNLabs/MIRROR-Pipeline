import torch
import torch.optim as optim
import torch.nn as nn

from mirror.models.mirror_model import MirrorModel
from mirror.tokenizers.placeholder_tokenizer import PlaceholderTokenizer
from mirror.util import device
from mirror.types import TrainStepOutput


class PlaceholderModel(MirrorModel[torch.Tensor]):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor([0.0], device=device))
        self._tokenizer = PlaceholderTokenizer()

    @property
    def tokenizer(self) -> PlaceholderTokenizer:
        return self._tokenizer

    def training_step(self, tokens, attention_mask) -> TrainStepOutput[torch.Tensor]:
        loss = self.parameter
        output = self.parameter
        return TrainStepOutput(loss=loss, output=output)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())