from typing import Literal

import wandb

from lightning import Fabric

from mirror.callbacks.callback import Callback
from mirror.config import RuntimeEnvironment, get_config
from mirror.util import mirror_data_path

from wandb.sdk.wandb_run import Run as WandbRun

WandbMode = Literal["online", "offline"]


class WandbCallback[RawT, ProcessedT, BatchT, ModelOutputT](
    Callback[RawT, ProcessedT, BatchT, ModelOutputT]
):
    def __init__(self) -> None:
        super().__init__(is_singleton=True)
        self.run: WandbRun | None = None
        self.step = 0

    def on_fit_start(
        self,
        *,
        fabric: Fabric,
        training_run_id: str,
        run_config_yaml: str,
        n_batches: int,
        epochs: int,
        **kwargs,
    ):
        if not fabric.is_global_zero:
            return

        wandb_dir = mirror_data_path / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)

        self.run = wandb.init(
            project="mirror",
            name=training_run_id,
            mode=self._mode(),
            dir=str(wandb_dir),
        )
        self.step = 0

        self.run.config.update({
            "training_run_id": training_run_id,
            "epochs": epochs,
            "n_batches": n_batches,
            "run_config_yaml": run_config_yaml,
        })
        
    def on_train_batch_end(
        self,
        *,
        loss: float,
        **kwargs,
    ):
        self.step += 1
        self.run.log({"train/loss": loss}, step=self.step)

    def on_fit_end(
        self,
        **kwargs,
    ):
        self.run.finish()
        self.run = None

    def _mode(self) -> WandbMode:
        if get_config()["environment"] == RuntimeEnvironment.SLURM_COMPUTE:
            return "offline"
        return "online"