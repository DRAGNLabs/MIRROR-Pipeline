from lightning import Fabric

from models.mirror_model import MirrorModel
from mirror_types import TokenBatch, AttentionMaskBatch, Loss


class Callback:
    def on_train_batch_end(self, fabric: Fabric, model: MirrorModel, loss: Loss, tokens: TokenBatch, attention_mask: AttentionMaskBatch, batch_idx: int):
        pass
