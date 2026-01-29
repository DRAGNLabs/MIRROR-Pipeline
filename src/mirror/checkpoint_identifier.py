from dataclasses import dataclass

from mirror.util import safe_training_run_path

@dataclass
class CheckpointIdentifier:
    """
    Checkpoints are found in your mirror_data folder under `training_runs/{training_run_id}/checkpoints`
    """
    training_run_id: str
    checkpoint_name: str

    @property
    def path(self):
        return safe_training_run_path(self.training_run_id) / \
            'checkpoints' / f'{self.checkpoint_name}.ckpt'

