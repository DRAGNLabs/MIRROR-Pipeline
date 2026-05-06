from typing import TypedDict
from mirror.models.mirror_model import MirrorModel
from torch.optim import Optimizer

class TextRow(TypedDict):
  text: str

class TextLabelRow(TextRow):
  label: str

class StateDict(TypedDict):
  model: MirrorModel
  optimizer: Optimizer
  global_step: int | None
