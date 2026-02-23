import torch
from lightning import Fabric
from tqdm import tqdm
from mirror.callbacks.callback import Callback

class ProgressCallback[RawT, ProcessedT, BatchT, ModelOutputT](
       Callback[RawT, ProcessedT, BatchT, ModelOutputT]
):
    def __init__(self, devices: int, bar_refresh_interval: int = 5) -> None:
        super().__init__(is_singleton=True)
        self.progress_bar = None
        self.bar_refresh_interval = bar_refresh_interval
        self.devices = devices


    def on_fit_start(
            self,
            *,
            n_batches: int,
            epochs: int,
            start_epoch: int,
            start_batch: int,
            **kwargs,
    ):
        if torch.distributed.get_rank() == 0:
            self.progress_bar = tqdm(total=(epochs * n_batches), initial=(start_epoch*n_batches + start_batch), desc="Training", mininterval=self.bar_refresh_interval)

    def on_train_batch_end(
            self,
            *,
            loss: float,
            **kwargs,
    ):
        if torch.distributed.get_rank() == 0 and self.progress_bar is not None:
            self.progress_bar.set_postfix(Loss=f"{loss:.3f}")
            self.progress_bar.update(self.devices)
