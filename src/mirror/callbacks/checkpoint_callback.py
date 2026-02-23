from lightning import Fabric
from torch.optim import Optimizer
from mirror.callbacks.callback import Callback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.models.mirror_model import MirrorModel

class CheckpointCallback[RawT, ProcessedT, BatchT, ModelOutputT](
       Callback[RawT, ProcessedT, BatchT, ModelOutputT]
):
    def __init__(self, every_n_train_steps: int | None = None) -> None:
        super().__init__(is_singleton=True)
        self.every_n_train_steps = every_n_train_steps

    def on_fit_start(
            self,
            *,
            fabric: Fabric,
            model: MirrorModel[RawT, ProcessedT, ModelOutputT],
            optimizer: Optimizer,
            training_run_id: str,
            epochs: int,
            n_batches: int,
            devices: int,
            **kwargs,
    ):
        self.n_print_digits = len(str(epochs*n_batches)) + 1
        self.n_batches = n_batches
        self._save_checkpoint(fabric, model, optimizer, CheckpointIdentifier(training_run_id, 'start'))
        self.devices = devices

    def on_fit_end(
            self, 
            *,
            fabric: Fabric, 
            model: MirrorModel[RawT, ProcessedT, ModelOutputT], 
            optimizer: Optimizer,
            training_run_id: str,
    ):
        self._save_checkpoint(fabric, model, optimizer, CheckpointIdentifier(training_run_id, 'end'))

    def on_train_batch_end(
            self,
            *,
            fabric: Fabric,
            model: MirrorModel[RawT, ProcessedT, ModelOutputT],
            optimizer: Optimizer,
            training_run_id: str,
            epoch: int,
            batch_idx: int,
            **kwargs,
    ):
        if self.every_n_train_steps and (batch_idx * self.devices + 1) % (self.every_n_train_steps) == 0:
            self._save_checkpoint(
                fabric,
                model,
                optimizer,
                CheckpointIdentifier(training_run_id, f"{epoch * self.n_batches + batch_idx:0{self.n_print_digits}d}")
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
