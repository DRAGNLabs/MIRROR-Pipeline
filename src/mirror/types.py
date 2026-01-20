from jaxtyping import Int, Float
from torch import Tensor
from dataclasses import dataclass
from typing import Generic, TypeVar

TokenTensor = Int[Tensor, "T"]
AttentionMask = Int[Tensor, "T"]

TokenBatch = Int[Tensor, "b t"]
AttentionMaskBatch = Int[Tensor, "b t"]
Loss = Float[Tensor, ""]


ProcessedT = TypeVar("ProcessedT")

@dataclass
class TrainStepOutput(Generic[ProcessedT]):
    loss: Tensor
    output: ProcessedT
