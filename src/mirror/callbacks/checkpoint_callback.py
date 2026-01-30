from lightning import Fabric
from torch.optim import Optimizer
from mirror.callbacks.callback import Callback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.models.mirror_model import MirrorModel

class CheckpointCallback[RawT, ProcessedT, BatchT, ModelOutputT](
       Callback[RawT, ProcessedT, BatchT, ModelOutputT]
):
    def __init__(self, every_n_train_steps: float | None = None) -> None:
        super().__init__(is_singleton=True)
        self.every_n_train_steps = every_n_train_steps

    def on_fit_start(
            self,
            *,
            fabric: Fabric,
            model: MirrorModel[RawT, ProcessedT, ModelOutputT],
            optimizer: Optimizer,
            training_run_id: str,
            **kwargs,
    ):
        self._save_checkpoint(fabric, model, optimizer, CheckpointIdentifier(training_run_id, 'start'))

    def on_fit_end(
            self, 
            *,
            fabric: Fabric, 
            model: MirrorModel[RawT, ProcessedT, ModelOutputT], 
            optimizer: Optimizer, 
            training_run_id: str
    ):
        self._save_checkpoint(fabric, model, optimizer, CheckpointIdentifier(training_run_id, 'end'))

    def on_train_batch_end(
            self,
            *,
            fabric: Fabric,
            model: MirrorModel[RawT, ProcessedT, ModelOutputT],
            optimizer: Optimizer,
            training_run_id: str,
            batch_idx: int,
            **kwargs,
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
            model: MirrorModel[RawT, ProcessedT, ModelOutputT],
            optimizer: Optimizer,
            checkpoint_identifier: CheckpointIdentifier,
    ):
        state = {
            'model': model,
            'optimizer': optimizer,
        }
        fabric.save(checkpoint_identifier.path, state)
