import torch
import torch.optim as optim
from models.mirror_model import MirrorModel


class PlaceholderModel(MirrorModel):
    def training_step(self, tokens, attention_mask):
        return torch.tensor(0.0)

    def configure_optimizers(self):
        return optim.AdamW([])
