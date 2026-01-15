from lightning import Fabric
from torch.optim import Optimizer

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models.mirror_model import MirrorModel
from mirror.types import TokenBatch, AttentionMaskBatch


class Callback:
    """
    The names of the methods here are based on those of Lightning's 
    Callback class: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.Callback.html#lightning.pytorch.callbacks.Callback
    """

    def __init__(self, is_singleton = False) -> None:
        """
        Args:
            is_singleton: whether or not there should only be one of this callback at a time
        """
        self.is_singleton = is_singleton

    def on_fit_start(
            self,
            fabric: Fabric,
            model: MirrorModel,
            optimizer: Optimizer,
            dataset: MirrorDataset,
            training_run_id: str,
            run_config_yaml: str,
            n_batches: int,
    ):
        pass

    def on_fit_end(self, fabric: Fabric, model: MirrorModel, optimizer: Optimizer, training_run_id: str):
        pass

    def on_train_batch_end(
            self,
            fabric: Fabric,
            model: MirrorModel,
            optimizer: Optimizer,
            loss: float,
            tokens: TokenBatch,
            attention_mask: AttentionMaskBatch,
            training_run_id: str,
            batch_idx: int,
    ):
        pass
