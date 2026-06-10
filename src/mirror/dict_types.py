from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, TypedDict
from torch.optim import Optimizer

if TYPE_CHECKING:
    from mirror.models.mirror_model import MirrorModel

class TextRow(TypedDict):
  text: str

class TextLabelRow(TextRow):
  label: str

class StateDict[RawT: Mapping[str, Any], ProcessedT: Mapping[str, Any], BatchT, ModelOutputT](TypedDict):
  model: MirrorModel[RawT, ProcessedT, BatchT, ModelOutputT]
  optimizer: Optimizer
  global_step: int | None
  optimization_step: int | None
