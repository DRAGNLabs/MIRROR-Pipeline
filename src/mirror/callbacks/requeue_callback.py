import signal

from lightning import Fabric
from torch.optim import Optimizer

from mirror.callbacks.callback import Callback
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models.mirror_model import MirrorModel


class RequeueCallback(Callback):
    def __init__(self, requeue_signal: int = signal.SIGHUP) -> None:
        super().__init__(is_singleton=False)
        self.requeue_signal = requeue_signal

    def on_fit_start(
            self,
            fabric: Fabric,
            model: MirrorModel,
            optimizer: Optimizer,
            dataset: MirrorDataset,
            training_run_id: str
    ):
        print(self.requeue_signal)
