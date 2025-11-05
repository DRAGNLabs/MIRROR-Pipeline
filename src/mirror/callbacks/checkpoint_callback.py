from lightning import Fabric
from torch.optim import Optimizer
from mirror.callbacks.callback import Callback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.models.mirror_model import MirrorModel

class CheckpointCallback(Callback):
    def on_fit_start(self, fabric: Fabric, model: MirrorModel, optimizer: Optimizer, training_run_id: str):
        self._save_checkpoint(fabric, model, optimizer, CheckpointIdentifier(training_run_id, 'start'))

    def on_fit_end(self, fabric: Fabric, model: MirrorModel, optimizer: Optimizer, training_run_id: str):
        self._save_checkpoint(fabric, model, optimizer, CheckpointIdentifier(training_run_id, 'end'))

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
