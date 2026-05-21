from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict
from torch.optim import Optimizer

if TYPE_CHECKING:
    from mirror.models.trainable_model import TrainableModel

class TextRow(TypedDict):
  text: str

class TextLabelRow(TextRow):
  label: str

class StateDict[RawT, ProcessedT, BatchT](TypedDict):
  model: TrainableModel[RawT, ProcessedT, BatchT]
  optimizer: Optimizer
  global_step: int | None
