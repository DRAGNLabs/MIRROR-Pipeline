from lightning import Fabric
from torch.optim import Optimizer
from mirror.callbacks.callback import Callback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.models.mirror_model import MirrorModel

class CheckpointCallback[RawT, ProcessedT, BatchT, ModelOutputT](
       Callback[RawT, ProcessedT, BatchT, ModelOutputT]
):
    def __init__(self, every_n_training_steps: int | None = None) -> None:
        super().__init__(is_singleton=True)
        self.every_n_training_steps = every_n_training_steps

    def on_fit_start(
            self,
            *,
            fabric: Fabric,
            model: MirrorModel[RawT, ProcessedT, BatchT, ModelOutputT],
            optimizer: Optimizer,
            training_run_id: str,
            epochs: int,
            **kwargs,
    ):
        self.epochs = epochs,
        self._save_checkpoint(fabric, model, optimizer, CheckpointIdentifier(training_run_id, 'start'), 0)

    def on_fit_end(
            self,
            *,
            fabric: Fabric,
            model: MirrorModel[RawT, ProcessedT, BatchT, ModelOutputT],
            optimizer: Optimizer,
            training_run_id: str,
    ):
        if fabric.is_global_zero:
            self._save_checkpoint(fabric, model, optimizer, CheckpointIdentifier(training_run_id, 'end'), None)

    def on_train_batch_end(
            self,
            *,
            fabric: Fabric,
            model: MirrorModel[RawT, ProcessedT, BatchT, ModelOutputT],
            optimizer: Optimizer,
            training_run_id: str,
            epochs: int,
            n_batches: int,
            epoch_idx: int,
            batch_idx: int,
            **kwargs,
    ):
        n_print_digits = len(str(epochs*n_batches)) + 1
        self.n_batches = n_batches

        if fabric.is_global_zero and self.every_n_training_steps and (batch_idx + 1) % (self.every_n_training_steps) == 0:
            self._save_checkpoint(
                fabric,
                model,
                optimizer,
                CheckpointIdentifier(training_run_id, f"{epoch_idx * self.n_batches + batch_idx:0{n_print_digits}d}"),
                epoch_idx * self.n_batches + batch_idx,
            )

    def _save_checkpoint(
            self,
            fabric: Fabric,
            model: MirrorModel[RawT, ProcessedT, BatchT, ModelOutputT],
            optimizer: Optimizer,
            checkpoint_identifier: CheckpointIdentifier,
            global_step: int | None,
    ):
        state = {
            'model': model,
            'optimizer': optimizer,
            'global_step': global_step,
        }
        fabric.save(checkpoint_identifier.path, state)
