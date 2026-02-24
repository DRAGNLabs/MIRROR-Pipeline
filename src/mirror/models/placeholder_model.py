import torch
import torch.optim as optim
import torch.nn as nn
from typing import List

from mirror.models.mirror_model import MirrorModel
from mirror.preprocessors.placeholder_preprocessor import PlaceholderPreprocessor
from mirror.types import AttentionMaskBatch, Loss, TokenBatch, TokenTensor
from mirror.util import get_device

from mirror.row_types import TextRow

class PlaceholderModel(MirrorModel[TextRow, TokenTensor, tuple[TokenBatch, AttentionMaskBatch]]):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor([0.0], device=get_device()))
        self._preprocessor = PlaceholderPreprocessor()

    @property
    def preprocessor(self) -> PlaceholderPreprocessor:
        return self._preprocessor    

    def training_step(self, batch: tuple[TokenBatch, AttentionMaskBatch]) -> Loss:
        tokens, attention_mask = batch
        return self.parameter

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(self.parameters())