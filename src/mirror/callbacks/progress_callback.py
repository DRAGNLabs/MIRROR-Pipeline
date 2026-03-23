from lightning import Fabric
from tqdm import tqdm
from mirror.callbacks.callback import Callback

class ProgressCallback[RawT, ProcessedT, BatchT, ModelOutputT](
       Callback[RawT, ProcessedT, BatchT, ModelOutputT]
):
    def __init__(self, bar_refresh_interval = 5) -> None:
        super().__init__(is_singleton=True)
        self.progress_bar = None
        self.bar_refresh_interval = bar_refresh_interval


    def on_fit_start(
            self,
            *,
            fabric: Fabric,
            n_batches: int,
            epochs: int,
            **kwargs,
    ):
        if fabric.is_global_zero:
            self.progress_bar = tqdm(total=(epochs * n_batches), desc="Training", mininterval=self.bar_refresh_interval)

    def on_train_batch_end(
            self,
            *,
            fabric: Fabric,
            loss: float,
            **kwargs,
    ):
        if fabric.is_global_zero and self.progress_bar is not None:
            self.progress_bar.set_postfix(Loss=f"{loss:.3f}", refresh=False)
            self.progress_bar.update(1)

    def on_validation_epoch_end(
            self,
            *,
            fabric: Fabric,
            val_loss,
            epoch,
            **kwargs,
    ):
        if fabric.is_global_zero and self.progress_bar is not None:
            self.progress_bar.set_postfix(Val_Loss=f"{val_loss:.3f}", Epoch=epoch, refresh=False)

    def on_test_epoch_end(
            self,
            *,
            fabric: Fabric,
            test_loss,
            **kwargs,
    ):
        if fabric.is_global_zero and self.progress_bar is not None:
            self.progress_bar.set_postfix(Test_Loss=f"{test_loss:.3f}", refresh=False)

    def on_fit_end(
            self,
            **kwargs,
    ):
        if self.progress_bar is not None:
            self.progress_bar.refresh()
            self.progress_bar.disable = True
            self.progress_bar.close()
            self.progress_bar = None