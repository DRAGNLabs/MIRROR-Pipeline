from typing import TypedDict
from mirror.models.mirror_model import MirrorModel
from torch.optim import Optimizer

class TextRow(TypedDict):
  text: str

class TextLabelRow(TextRow):
  label: str

class StateDict[RawT, ProcessedT, BatchT, ModelOutputT](TypedDict):
  model: MirrorModel[RawT, ProcessedT, BatchT, ModelOutputT]
  optimizer: Optimizer
  global_step: int | None
