import torch
import torch.optim as optim
import torch.nn as nn

from mirror.models.mirror_model import MirrorModel
from mirror.preprocessors.placeholder_preprocessor import PlaceholderPreprocessor
from mirror.types import AttentionMaskBatch, Loss, TextRow, TokenBatch, TokenTensor, TrainStepOutput
from mirror.util import get_device

class PlaceholderModel(MirrorModel[TextRow, TokenTensor, tuple[TokenBatch, AttentionMaskBatch], None]):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor([0.0], device=get_device()))
        self._preprocessor = PlaceholderPreprocessor()

    @property
    def preprocessor(self) -> PlaceholderPreprocessor:
        return self._preprocessor    

    def training_step(self, batch: tuple[TokenBatch, AttentionMaskBatch]) -> TrainStepOutput[None]:
        tokens, attention_mask = batch
        return TrainStepOutput(loss=self.parameter, output=None)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(self.parameters())
