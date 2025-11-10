from lightning import Fabric
import torch
from torch.optim import Optimizer
from mirror.callbacks.callback import Callback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models.mirror_model import MirrorModel
from mirror.types import Loss, TokenBatch, AttentionMaskBatch

class CheckpointCallback(Callback):
    def __init__(self, every_n_train_steps: float | None = None) -> None:
        super().__init__(is_singleton=True)
        self.every_n_train_steps = every_n_train_steps

    def on_fit_start(
            self,
            fabric: Fabric,
            model: MirrorModel,
            optimizer: Optimizer,
            dataset: MirrorDataset,
            training_run_id: str,
    ):
        self._save_checkpoint(fabric, model, optimizer, CheckpointIdentifier(training_run_id, 'start'))

    def on_fit_end(self, fabric: Fabric, model: MirrorModel, optimizer: Optimizer, training_run_id: str):
        self._save_checkpoint(fabric, model, optimizer, CheckpointIdentifier(training_run_id, 'end'))

    def on_train_batch_end(
            self,
            fabric: Fabric,
            model: MirrorModel,
            optimizer: Optimizer,
            loss: Loss,
            tokens: TokenBatch,
            attention_mask: AttentionMaskBatch,
            training_run_id: str,
            batch_idx: int,
    ):
        if self.every_n_train_steps and batch_idx % self.every_n_train_steps == 0:
            self._save_checkpoint(
                fabric,
                model,
                optimizer,
                CheckpointIdentifier(training_run_id, str(batch_idx))
            )

    def _save_checkpoint(
            self,
            fabric: Fabric,
            model: MirrorModel,
            optimizer: Optimizer,
            checkpoint_identifier: CheckpointIdentifier,
    ):
        state = {
            'model': model,
            'optimizer': optimizer,
        }
        fabric.save(checkpoint_identifier.path, state)
