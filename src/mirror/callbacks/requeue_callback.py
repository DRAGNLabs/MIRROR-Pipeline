import json
import os
import signal
from subprocess import call
import time
from types import FrameType
from typing import Dict, Literal

from lightning import Fabric
from torch.optim import Optimizer

from mirror.callbacks.callback import Callback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.fabric_util import rank_zero_log
from mirror.models.mirror_model import MirrorModel
from mirror.slurm_util import get_job_id
from mirror.util import is_power_of_ten, mirror_data_path

def requeue_handoff_path():
    slurm_job_id = get_job_id()
    return mirror_data_path / 'requeue_handoffs' / f'handoff-{slurm_job_id}.json'

RequeueHandoff = Dict[Literal['previous_training_run_id'], str]


class RequeueCallback[ProcessedT, ModelOutputT](Callback[ProcessedT, ModelOutputT]):
    def __init__(self, requeue_signal: int = signal.SIGHUP) -> None:
        super().__init__(is_singleton=True)
        self.requeue_signal = requeue_signal
        self.requeue_signal_recieved = False

        self.grace_period_seconds = 90
        self.last_train_batch_end_time = None
        self.num_iterations_too_long = 0 # the number of times a training step was longer than the grace period

    def on_fit_start(
            self,
            *,
            fabric: Fabric,
            model: MirrorModel[ProcessedT, ModelOutputT],
            optimizer: Optimizer,
            **kwargs,
    ):
        rank_zero_log(fabric, f'setting up requeue handler on signal {self.requeue_signal}')
        signal.signal(self.requeue_signal, self._make_requeue_handler(fabric))

        self._load_requeue_checkpoint_if_present(fabric, model, optimizer)

    def on_train_batch_end(
            self,
            *,
            fabric: Fabric,
            model: MirrorModel[ProcessedT, ModelOutputT],
            optimizer: Optimizer,
            training_run_id: str,
            **kwargs,
    ):
        self._warn_if_iteration_too_long(fabric)
        if self.requeue_signal_recieved:
            self._save_checkpoint(fabric, model, optimizer, training_run_id)
            if fabric.is_global_zero:
                self._create_requeue_handoff(training_run_id)
                self._requeue(fabric)
            exit()

    def _load_requeue_checkpoint_if_present(
            self, 
            fabric: Fabric, 
            model: MirrorModel[ProcessedT, ModelOutputT], 
            optimizer: Optimizer
    ):
        path = requeue_handoff_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            with open(path, 'r') as f:
                rank_zero_log(fabric, 'loading requeue checkpoint')
                handoff_json = f.read()
                handoff: RequeueHandoff = json.loads(handoff_json)

                checkpoint_id = self._requeue_checkpoint_id(handoff['previous_training_run_id'])
                fabric.load(checkpoint_id.path, {
                    'model': model,
                    'optimizer': optimizer,
                })
        except FileNotFoundError:
            pass

    def _requeue_checkpoint_id(
            self, 
            training_run_id: str
    ):
        return CheckpointIdentifier(training_run_id, checkpoint_name='requeue')


    def _save_checkpoint(
            self, 
            fabric: Fabric, 
            model: MirrorModel[ProcessedT, ModelOutputT], 
            optimizer: Optimizer, 
            training_run_id: str
    ):
        rank_zero_log(fabric, f'Saving requeue checkpoint for {training_run_id}')
        checkpoint_id = self._requeue_checkpoint_id(training_run_id)
        fabric.save(checkpoint_id.path, {
            'model': model,
            'optimizer': optimizer,
        })

    def _create_requeue_handoff(
            self,
            training_run_id: str
    ):
        path = requeue_handoff_path()
        handoff: RequeueHandoff = {
            'previous_training_run_id': training_run_id
        }
        handoff_json = json.dumps(handoff)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(handoff_json)


    def _requeue(self, fabric: Fabric):
        rank_zero_log(fabric, f'requeueing job')

        job_id = get_job_id()
        cmd = ['scontrol', 'requeue', job_id]
        try:
            result = call(cmd)
        except FileNotFoundError:
            # This can occur if a subprocess call to `scontrol` is run outside a shell context
            # Re-attempt call (now with shell context). If any error is raised, propagate to user.
            # When running a shell command, it should be passed as a single string.
            result = call(" ".join(cmd), shell=True)

        if result == 0:
            rank_zero_log(fabric, f'requeued job {job_id}')
        else:
            rank_zero_log(fabric, f'requeueing job {job_id} failed with error code {result}')

    def _warn_if_iteration_too_long(self, fabric: Fabric):
        if not fabric.is_global_zero:
            return

        current_time = time.time()
        if self.last_train_batch_end_time is not None:
            training_step_duration = current_time - self.last_train_batch_end_time
            if training_step_duration > self.grace_period_seconds:
                self.num_iterations_too_long += 1

            if is_power_of_ten(self.num_iterations_too_long):
                rank_zero_log(fabric, f'WARNING: the last training step took {training_step_duration} seconds, which is longer than the requeue grace period of {self.grace_period_seconds} seconds. This might cause your training run not to get properly checkpointed and requeued if it hits the walltime limit. {self.num_iterations_too_long} training step(s) have been longer than the grace period so far.')
        self.last_train_batch_end_time = current_time


    def _make_requeue_handler(self, fabric: Fabric):
        def requeue_hander(_signum: int, _stack: FrameType | None):
            rank_zero_log(fabric, f'requeue handler called with {_signum}')
            self.requeue_signal_recieved = True
        return requeue_hander
