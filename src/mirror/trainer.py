from lightning import Fabric
from torch.utils.data import DataLoader
from typing import List
import datetime
import os
import torch

from lightning.fabric.strategies.strategy import Strategy
from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.fabric.strategies.single_device import SingleDeviceStrategy

from mirror.callbacks.callback import Callback
from mirror.callbacks.checkpoint_callback import CheckpointCallback
from mirror.callbacks.progress_callback import ProgressCallback
from mirror.callbacks.requeue_callback import RequeueCallback
from mirror.callbacks.wandb_callback import WandbCallback
from mirror.callbacks.config_snapshot_callback import ConfigSnapshotCallback
from mirror.callbacks.print_step_callback import PrintStepCallback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.datasets.on_demand_preprocessed_dataset import OnDemandPreprocessedDataset
from mirror.models.mirror_model import MirrorModel
from mirror.config import RuntimeEnvironment, get_config


class Trainer[RawT, ProcessedT, BatchT, ModelOutputT]:
    def __init__(
            self,
            strategy: Strategy = FSDPStrategy(),
            devices: int = 1,
            num_nodes: int = 1,
            callbacks: List[Callback[RawT, ProcessedT, ModelOutputT]] = [],
    ) -> None:
        config = get_config()
        self.config = config
        if config['device'] == "cpu" and isinstance(strategy, FSDPStrategy):
            self.strategy = SingleDeviceStrategy(device="cpu")
        else:
            self.strategy = strategy
        self.devices = devices
        self.num_nodes = num_nodes
        default_callbacks: List[Callback[RawT, ProcessedT, ModelOutputT]] = [
            CheckpointCallback(),
            ConfigSnapshotCallback(),
            ProgressCallback(),
            WandbCallback()
        ]
        if os.getenv("MIRROR_PRINT_STEP_LOSS", "").lower() == "true":
            default_callbacks.append(PrintStepCallback())
        if config['environment'] == RuntimeEnvironment.SLURM_COMPUTE:
            default_callbacks.append(RequeueCallback())

        default_singleton_cbs, default_non_singleton_cbs = separate_singletons(default_callbacks)
        input_singleton_cbs, input_non_singleton_cbs = separate_singletons(callbacks)

        singleton_cbs = {cb.__class__:cb for cb in [*default_singleton_cbs, *input_singleton_cbs]}.values()

        callbacks = [*singleton_cbs, *default_non_singleton_cbs, *input_non_singleton_cbs]
        self.callbacks = callbacks
        self.fabric = self._make_fabric(self.strategy, config['device'])

    def launch(self):
        try:
            self.fabric.launch()
        except torch.AcceleratorError as exc:
            if self.config['device'] == 'cuda':
                print("WARNING: CUDA unavailable or busy, running on CPU instead.")
                self.config['device'] = 'cpu'
                strategy = self.strategy
                if isinstance(strategy, FSDPStrategy):
                    strategy = SingleDeviceStrategy(device="cpu")
                self.fabric = self._make_fabric(strategy, 'cpu')
                self.fabric.launch()
                return
            raise

    def fit(
            self,
            model: MirrorModel[RawT, ProcessedT, BatchT],
            dataset: MirrorDataset[RawT],
            checkpoint: CheckpointIdentifier | None = None,
            epochs: int = 1,
            batch_size: int = 1,
            do_preprocess: bool = False,
            run_config_yaml: str = "",
            val_dataset: MirrorDataset[RawT] | None = None,
            test_dataset: MirrorDataset[RawT] | None = None,
            val_check_interval: float = 1.0,
    ):
        training_run_id = datetime.datetime.now().isoformat()

        model, optimizer = self.fabric.setup(
            model,
            model.configure_optimizers(),
            move_to_device=self.config['device'] == 'cuda'
        )

        if checkpoint:
            # models and optimizers are treated specially: they are populated via their load_state_dict
            # methods internally to fabric.load. Anything else in the state dict is just set in place.
            state = {
                'model': model,
                'optimizer': optimizer,
            }
            self.fabric.load(checkpoint.path, state)

        dataloader = self._make_dataloader(dataset, model, batch_size, do_preprocess)

        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = self._make_dataloader(val_dataset, mode1l, batch_size, do_preprocess)

        test_dataloader = None
        if test_dataset is not None:
            test_dataloader = self._make_dataloader(test_dataset, model, batch_size, do_preprocess)

        self.fabric.call('on_fit_start', fabric=self.fabric, model=model, optimizer=optimizer, dataset=dataset,
            training_run_id=training_run_id, n_batches=len(dataloader), epochs=epochs, run_config_yaml=run_config_yaml)

        next_val_epoch = val_check_interval
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(dataloader):
                batch: BatchT = batch

                optimizer.zero_grad()
                loss = model.training_step(batch)
                loss_value = loss.item()
                self.fabric.backward(loss)
                optimizer.step()

                self.fabric.call(
                    'on_train_batch_end',
                    fabric=self.fabric,
                    model=model,
                    optimizer=optimizer,
                    loss=loss_value,
                    training_run_id=training_run_id,
                    batch_idx=batch_idx
                )

            if val_dataloader is not None and (epoch + 1) >= next_val_epoch:
                val_loss = self._eval_loop(model, val_dataloader)
                self.fabric.call(
                    'on_validation_epoch_end',
                    fabric=self.fabric,
                    model=model,
                    optimizer=optimizer,
                    val_loss=val_loss,
                    training_run_id=training_run_id,
                    epoch=epoch,
                )
                next_val_epoch = epoch + 1 + val_check_interval

        if test_dataloader is not None:
            test_loss = self._eval_loop(model, test_dataloader)
            self.fabric.call(
                'on_test_epoch_end',
                fabric=self.fabric,
                model=model,
                optimizer=optimizer,
                test_loss=test_loss,
                training_run_id=training_run_id,
            )

        self.fabric.call(
            'on_fit_end',
            fabric=self.fabric,
            model=model,
            optimizer=optimizer,
            training_run_id=training_run_id,
        )

    def _eval_loop(self, model, dataloader) -> float:
        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                loss = model.training_step(batch)
                total_loss += loss.item()
                n_batches += 1
        model.train()
        return total_loss / n_batches if n_batches > 0 else 0.0

    def _make_dataloader(self, dataset, model, batch_size, do_preprocess):
        if do_preprocess:
            preprocessed = dataset.preprocess(model.preprocessor.preprocess_example)
        else:
            preprocessed = OnDemandPreprocessedDataset(dataset, model.preprocessor.preprocess_example)
        dataloader = DataLoader(
            preprocessed, 
            batch_size=batch_size,
            collate_fn=model.preprocessor.collate,
            drop_last=False,
        )
        return self.fabric.setup_dataloaders(dataloader, move_to_device=self.config['device'] == 'cuda')

    def _make_fabric(self, strategy: Strategy, accelerator: str) -> Fabric:
        return Fabric(
            strategy=strategy,
            devices=self.devices,
            num_nodes=self.num_nodes,
            callbacks=self.callbacks,
            accelerator=accelerator,
        )

def separate_singletons[RawT, ProcessedT, ModelOutputT](
       callbacks: List[Callback[RawT, ProcessedT, ModelOutputT]]
) -> tuple[
   List[Callback[RawT, ProcessedT, ModelOutputT]],
   List[Callback[RawT, ProcessedT, ModelOutputT]]
]:
    singletons = [c for c in callbacks if c.is_singleton]
    non_singletons = [c for c in callbacks if not c.is_singleton]
    return singletons, non_singletons
