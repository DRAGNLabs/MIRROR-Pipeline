import torch
import torch.nn as nn

from mirror.interventions.intervention import Intervention
from mirror.models.mirror_model import MirrorModel


def _zero_hook(
        _module: nn.Module,
        _input: tuple,
        output: torch.Tensor
) -> torch.Tensor:
    return torch.zeros_like(output)


class ZeroMLPIntervention(Intervention):
    def __init__(self, target: MirrorModel):
        super().__init__(target)
        mlps = self.target.mlp_modules()
        if not mlps:
            raise ValueError(
                f"{type(target).__name__} does not expose any MLP modules. "
                "Override mlp_modules() to support ZeroMLPIntervention."
            )
        for mlp in mlps:
            mlp.register_forward_hook(_zero_hook)

    def mlp_modules(self) -> list[nn.Module]:
        return []
