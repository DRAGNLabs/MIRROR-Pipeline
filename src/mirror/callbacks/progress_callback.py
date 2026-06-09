from typing import Any, Mapping
from lightning import Fabric
from tqdm import tqdm
from mirror.callbacks.callback import Callback
from mirror.metrics.extra_metrics_getter import ExtraMetricsGetter
from mirror.models.mirror_model import MirrorModel

class ProgressCallback[RawT: Mapping[str, Any], ProcessedT: Mapping[str, Any], BatchT, ModelOutputT](
       Callback[RawT, ProcessedT, BatchT, ModelOutputT]
):
    def __init__(
            self,
            bar_refresh_interval: int = 5,
            extra_metrics_getter: ExtraMetricsGetter | None = None,
            extra_metrics_every_n_steps: int = 1,
    ) -> None:
        super().__init__(is_singleton=True)
        self.progress_bar = None
        self.bar_refresh_interval = bar_refresh_interval
        self.extra_metrics_getter = extra_metrics_getter
        self.extra_metrics_every_n_steps = extra_metrics_every_n_steps
        self.step = 0
        self.last_extra_metrics: dict = {}

    def on_fit_start(
            self,
            *,
            fabric: Fabric,
            n_batches: int,
            epochs: int,
            start_epoch: int,
            start_batch: int,
            **kwargs,
    ):
        if fabric.is_global_zero:
            self.progress_bar = tqdm(total=(epochs * n_batches), initial=(start_epoch*n_batches + start_batch), desc="Training", mininterval=self.bar_refresh_interval)

    def on_train_batch_end(
            self,
            *,
            fabric: Fabric,
            model: MirrorModel,
            loss: float,
            **kwargs,
    ):
        self.step += 1
        if self.extra_metrics_getter is not None and self.step % self.extra_metrics_every_n_steps == 0:
            self.last_extra_metrics = self.extra_metrics_getter.get_metrics(model, fabric)
        if fabric.is_global_zero and self.progress_bar is not None:
            self.progress_bar.set_postfix(Loss=f"{loss:.3f}", **self.last_extra_metrics, refresh=False)
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
