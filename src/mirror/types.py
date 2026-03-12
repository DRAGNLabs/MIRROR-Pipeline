from jaxtyping import Int, Float
from torch import FloatTensor, LongTensor, Tensor
from dataclasses import dataclass

TokenTensor = list[int]
AttentionMask = Int[FloatTensor, "T"]

TokenBatch = Int[LongTensor, "b t"]
AttentionMaskBatch = Int[FloatTensor, "b t"]
Loss = Float[Tensor, ""]

@dataclass
class TrainStepOutput[ModelOutputT]:
    loss: Tensor
    output: ModelOutputT
