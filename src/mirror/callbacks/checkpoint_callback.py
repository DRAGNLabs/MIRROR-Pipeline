from typing import cast, Any
from lightning import Fabric
from torch.nn import Module
from torch.optim import Optimizer
from mirror.callbacks.callback import Callback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.models.mirror_model import MirrorModel
from mirror.dict_types import StateDict

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
            **kwargs,
    ):
        self._save_checkpoint(fabric, model, optimizer, CheckpointIdentifier(training_run_id, 'start'), 0)

    def on_fit_end(
            self,
            *,
            fabric: Fabric,
            model: MirrorModel[RawT, ProcessedT, BatchT, ModelOutputT],
            optimizer: Optimizer,
            training_run_id: str,
    ):
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
            global_step: int,
            **kwargs,
    ):
        n_print_digits = len(str(epochs*n_batches)) + 1

        if self.every_n_training_steps and (global_step + 1) % (self.every_n_training_steps) == 0:
            self._save_checkpoint(
                fabric,
                model,
                optimizer,
                CheckpointIdentifier(training_run_id, f"{global_step:0{n_print_digits}d}"),
                global_step,
            )

    def _save_checkpoint(
            self,
            fabric: Fabric,
            model: MirrorModel[RawT, ProcessedT, BatchT, ModelOutputT],
            optimizer: Optimizer,
            checkpoint_identifier: CheckpointIdentifier,
            global_step: int | None,
    ):
        state : StateDict = {
            'model': model,
            'optimizer': optimizer,
            'global_step': global_step,
        }
        fabric.save(checkpoint_identifier.path, cast(dict[str, Module | Optimizer | Any], state))
