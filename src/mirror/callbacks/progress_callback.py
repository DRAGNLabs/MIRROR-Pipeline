from lightning import Fabric
import torch
from tqdm import tqdm
from torch.optim import Optimizer
from mirror.callbacks.callback import Callback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models.mirror_model import MirrorModel
from mirror.types import TokenBatch, AttentionMaskBatch

class ProgressCallback(Callback):
    def __init__(self, bar_refresh_interval = 5) -> None:
        super().__init__(is_singleton=True)
        self.progress_bar = None
        self.bar_refresh_interval = bar_refresh_interval


    def on_fit_start(
            self,
            fabric: Fabric,
            model: MirrorModel,
            optimizer: Optimizer,
            dataset: MirrorDataset,
            training_run_id: str,
            n_batches: int,
    ):
        if fabric.is_global_zero:
            self.progress_bar = tqdm(total=n_batches, desc="Training", mininterval=self.bar_refresh_interval)

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
        if fabric.is_global_zero and self.progress_bar is not None:
            self.progress_bar.set_postfix(Loss=f"{loss:.3f}")
            self.progress_bar.update(1)
