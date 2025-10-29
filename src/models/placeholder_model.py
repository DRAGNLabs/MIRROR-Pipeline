import torch
import torch.optim as optim
import torch.nn as nn
from models.mirror_model import MirrorModel


class PlaceholderModel(MirrorModel):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor(0.0))

    def training_step(self, tokens, attention_mask):
        return self.parameter

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())
