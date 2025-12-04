import signal
from types import FrameType

from lightning import Fabric
from torch.optim import Optimizer

from mirror.callbacks.callback import Callback
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models.mirror_model import MirrorModel
from mirror.types import AttentionMaskBatch, Loss, TokenBatch


class RequeueCallback(Callback):
    def __init__(self, requeue_signal: int = signal.SIGHUP) -> None:
        super().__init__(is_singleton=True)
        self.requeue_signal = requeue_signal
        self.requeue_signal_recieved = False

    def on_fit_start(
            self,
            fabric: Fabric,
            model: MirrorModel,
            optimizer: Optimizer,
            dataset: MirrorDataset,
            training_run_id: str
    ):
        print(f'setting up requeue handler on signal {self.requeue_signal}', flush=True)
        signal.signal(self.requeue_signal, self._requeue_handler)

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
        if self.requeue_signal_recieved:
            print('doing requeue work', flush=True)
            exit()

    def _requeue_handler(self, _signum: int, _stack: FrameType | None):
        print(f'requeue handler called with {_signum}', flush=True)
        self.requeue_signal_recieved = True
