from typing import TypedDict
from torch.nn import Module
from torch.optim import Optimizer

class TextRow(TypedDict):
  text: str

class TextLabelRow(TextRow):
  label: str

class StateDict(TypedDict):
  model: Module
  optimizer: Optimizer
  global_step: int | None
