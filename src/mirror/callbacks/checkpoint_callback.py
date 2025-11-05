from lightning import Fabric
from torch.optim import Optimizer
from mirror.callbacks.callback import Callback
from mirror.models.mirror_model import MirrorModel
from mirror.util import mirror_data_path

class CheckpointCallback(Callback):
    def on_fit_start(self, fabric: Fabric, model: MirrorModel, optimizer: Optimizer, training_run_id: str):
        self._save_checkpoint(fabric, model, optimizer, training_run_id, 'start')

    def on_fit_end(self, fabric: Fabric, model: MirrorModel, optimizer: Optimizer, training_run_id: str):
        self._save_checkpoint(fabric, model, optimizer, training_run_id, 'end')

    def _save_checkpoint(
            self,
            fabric: Fabric,
            model: MirrorModel,
            optimizer: Optimizer,
            training_run_id: str,
            checkpoint_name: str,
    ):
        state = {
            'model': model,
            'optimizer': optimizer,
        }
        path = mirror_data_path / 'training_runs' / training_run_id / \
            'checkpoints' / f'{checkpoint_name}.ckpt'
        fabric.save(path, state)
