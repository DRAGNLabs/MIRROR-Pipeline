from typing import cast, Any, Mapping
from lightning import Fabric
from torch.nn import Module
from torch.optim import Optimizer
from mirror.callbacks.callback import Callback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.models.trainable_model import TrainableModel
from mirror.dict_types import StateDict

class CheckpointCallback[RawT: Mapping[str, Any], FormattedT: Mapping[str, Any], BatchT](
       Callback[RawT, FormattedT, BatchT]
):
    def __init__(
            self,
            every_n_training_steps: int | None = None,
            every_n_epochs: int | None = None,
    ) -> None:
        super().__init__(is_singleton=True)
        self.every_n_training_steps = every_n_training_steps
        self.every_n_epochs = every_n_epochs

    def on_fit_start(
            self,
            *,
            fabric: Fabric,
            model: TrainableModel[RawT, FormattedT, BatchT],
            optimizer: Optimizer,
            training_run_id: str,
            **kwargs,
    ):
        self._save_checkpoint(
            fabric=fabric, 
            model=model,
            optimizer=optimizer, 
            checkpoint_identifier=CheckpointIdentifier(training_run_id, 'start'),
            global_step=0, 
            optimization_step=0,
        )

    def on_fit_end(
            self,
            *,
            fabric: Fabric,
            model: TrainableModel[RawT, FormattedT, BatchT],
            optimizer: Optimizer,
            training_run_id: str,
    ):
        self._save_checkpoint(
            fabric=fabric, 
            model=model,
            optimizer=optimizer, 
            checkpoint_identifier=CheckpointIdentifier(training_run_id, 'end'),
            global_step=None, 
            optimization_step=None,
        )

    def on_optimization_step(
            self,
            *,
            fabric: Fabric,
            model: TrainableModel[RawT, FormattedT, BatchT],
            optimizer: Optimizer,
            training_run_id: str,
            epochs: int,
            n_batches: int,
            global_step: int,
            optimization_step: int,
            **kwargs,
    ):
        n_print_digits = len(str(epochs*n_batches)) + 1

        if self.every_n_training_steps and optimization_step % self.every_n_training_steps == 0:
            self._save_checkpoint(
                fabric,
                model,
                optimizer,
                CheckpointIdentifier(training_run_id, f"{optimization_step:0{n_print_digits}d}"),
                global_step,
                optimization_step,
            )

    def on_train_batch_end(
            self,
            *,
            fabric: Fabric,
            model: TrainableModel[RawT, FormattedT, BatchT],
            optimizer: Optimizer,
            training_run_id: str,
            epochs: int,
            n_batches: int,
            batch_idx: int,
            global_step: int,
            optimization_step: int,
            **kwargs,
    ):
        if self.every_n_epochs is None or batch_idx != n_batches - 1:
            return
        completed_epoch = global_step // n_batches
        if completed_epoch % self.every_n_epochs != 0:
            return
        n_print_digits = len(str(epochs)) + 1
        self._save_checkpoint(
            fabric,
            model,
            optimizer,
            CheckpointIdentifier(training_run_id, f"epoch_{completed_epoch:0{n_print_digits}d}"),
            global_step,
            optimization_step,
        )

    def _save_checkpoint(
            self,
            fabric: Fabric,
            model: TrainableModel[RawT, FormattedT, BatchT],
            optimizer: Optimizer,
            checkpoint_identifier: CheckpointIdentifier,
            global_step: int | None,
            optimization_step: int | None,
    ):
        state : StateDict[RawT, FormattedT, BatchT] = {
            'model': model,
            'optimizer': optimizer,
            'global_step': global_step,
            'optimization_step': optimization_step,
        }
        fabric.save(checkpoint_identifier.path, cast(dict[str, Module | Optimizer | Any], state))
