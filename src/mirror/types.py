from jaxtyping import Int, Float
from torch import Tensor
from dataclasses import dataclass

TokenTensor = Int[Tensor, "T"]
AttentionMask = Int[Tensor, "T"]

TokenBatch = Int[Tensor, "b t"]
AttentionMaskBatch = Int[Tensor, "b t"]
Loss = Float[Tensor, ""]

@dataclass
class TrainStepOutput[ModelOutputT]:
    loss: Tensor
    output: ModelOutputT
