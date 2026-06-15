from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, TypedDict
from torch.optim import Optimizer

if TYPE_CHECKING:
    from mirror.models.trainable_model import TrainableModel

class TextRow(TypedDict):
  text: str

class TextLabelRow(TextRow):
  label: str

class StateDict[RawT: Mapping[str, Any], FormattedT: Mapping[str, Any], BatchT](TypedDict):
  model: TrainableModel[RawT, FormattedT, BatchT]
  optimizer: Optimizer
  global_step: int | None
  optimization_step: int | None
