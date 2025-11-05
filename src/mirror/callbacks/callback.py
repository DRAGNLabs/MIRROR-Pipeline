from lightning import Fabric
from torch.optim import Optimizer

from mirror.models.mirror_model import MirrorModel
from mirror.types import TokenBatch, AttentionMaskBatch, Loss


class Callback:
    """
    The names of the methods here are based on those of Lightning's 
    Callback class: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.Callback.html#lightning.pytorch.callbacks.Callback
    """

    def on_fit_start(self, fabric: Fabric, model: MirrorModel, optimizer: Optimizer, training_run_id: str):
        pass

    def on_fit_end(self, fabric: Fabric, model: MirrorModel, optimizer: Optimizer, training_run_id: str):
        pass

    def on_train_batch_end(self, fabric: Fabric, model: MirrorModel, loss: Loss, tokens: TokenBatch, attention_mask: AttentionMaskBatch, batch_idx: int):
        pass
