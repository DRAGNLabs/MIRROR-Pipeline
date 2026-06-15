import torch
import torch.optim as optim
import torch.nn as nn

from mirror.models.inference_model import InferenceModel
from mirror.models.trainable_model import TrainableModel
from mirror.formatters.placeholder_formatter import PlaceholderFormatter
from mirror.types import LabeledTokens, Loss, StandardBatch, TextRow
from mirror.util import get_device


class PlaceholderModel(
    TrainableModel[TextRow, LabeledTokens, StandardBatch],
    InferenceModel[TextRow, LabeledTokens, StandardBatch, None],
):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor([0.0], device=get_device()))
        self._formatter = PlaceholderFormatter()

    @property
    def formatter(self) -> PlaceholderFormatter:
        return self._formatter

    def forward(self, batch: StandardBatch) -> None:
        return None

    def training_step(self, batch: StandardBatch) -> Loss:
        return self.parameter

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(self.parameters())
