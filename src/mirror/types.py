from jaxtyping import Int, Float
from torch import Tensor

TokenTensor = Int[Tensor, "T"]
AttentionMask = Int[Tensor, "T"]

TokenBatch = Int[Tensor, "b t"]
AttentionMaskBatch = Int[Tensor, "b t"]
Loss = Float[Tensor, ""]
