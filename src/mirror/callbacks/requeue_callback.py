import signal
import time
from types import FrameType

from lightning import Fabric
from torch.optim import Optimizer

from mirror.callbacks.callback import Callback
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.fabric_util import rank_zero_log
from mirror.models.mirror_model import MirrorModel
from mirror.types import AttentionMaskBatch, Loss, TokenBatch
from mirror.util import is_power_of_ten


class RequeueCallback(Callback):
    def __init__(self, requeue_signal: int = signal.SIGHUP) -> None:
        super().__init__(is_singleton=True)
        self.requeue_signal = requeue_signal
        self.requeue_signal_recieved = False

        self.grace_period_seconds = 90
        self.last_train_batch_end_time = None
        self.num_iterations_too_long = 0 # the number of times a training step was longer than the grace period

    def on_fit_start(
            self,
            fabric: Fabric,
            model: MirrorModel,
            optimizer: Optimizer,
            dataset: MirrorDataset,
            training_run_id: str
    ):
        rank_zero_log(fabric, f'setting up requeue handler on signal {self.requeue_signal}')
        signal.signal(self.requeue_signal, self._make_requeue_handler(fabric))

    def on_train_batch_end(
            self,
            fabric: Fabric,
            model: MirrorModel,
            optimizer: Optimizer,
            loss: Loss,
            tokens: TokenBatch,
            attention_mask: AttentionMaskBatch,
            training_run_id: str,
            batch_idx: int
    ):
        self._warn_if_iteration_too_long(fabric)
        if self.requeue_signal_recieved:
            print('doing requeue work', flush=True)
            exit()

    def _warn_if_iteration_too_long(self, fabric: Fabric):
        if not fabric.is_global_zero:
            return

        current_time = time.time()
        if self.last_train_batch_end_time is not None:
            training_step_duration = current_time - self.last_train_batch_end_time
            if self.last_train_batch_end_time and training_step_duration > self.grace_period_seconds:
                self.num_iterations_too_long += 1

            if is_power_of_ten(self.num_iterations_too_long):
                rank_zero_log(fabric, f'WARNING: the last training step took {training_step_duration} seconds, which is longer than the requeue grace period of {self.grace_period_seconds} seconds. This might cause your training run not to get properly checkpointed and requeued if it hits the walltime limit. {self.num_iterations_too_long} training step(s) have been longer than the grace period so far.')
        self.last_train_batch_end_time = current_time


    def _make_requeue_handler(self, fabric: Fabric):
        def requeue_hander(_signum: int, _stack: FrameType | None):
            rank_zero_log(fabric, f'requeue handler called with {_signum}')
            self.requeue_signal_recieved = True
        return requeue_hander
