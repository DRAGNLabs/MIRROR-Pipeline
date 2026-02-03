from lightning import Fabric
from torch.utils.data import DataLoader
from typing import List
import datetime
import torch

from lightning.fabric.strategies.strategy import Strategy
from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.fabric.strategies.single_device import SingleDeviceStrategy

from mirror.callbacks.callback import Callback
from mirror.callbacks.checkpoint_callback import CheckpointCallback
from mirror.callbacks.progress_callback import ProgressCallback
from mirror.callbacks.requeue_callback import RequeueCallback
from mirror.callbacks.config_snapshot_callback import ConfigSnapshotCallback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.datasets.preprocessed_dataset import PreprocessedDataset
from mirror.models.mirror_model import MirrorModel
from mirror.config import RuntimeEnvironment, get_config


class Trainer[RawT, ProcessedT, BatchT, ModelOutputT]:
    def __init__(
            self,
            strategy: Strategy = FSDPStrategy(),
            devices: int = 1,
            num_nodes: int = 1,
            callbacks: List[Callback[RawT, ProcessedT, BatchT, ModelOutputT]] = [],
    ) -> None:
        config = get_config()
        self.config = config
        self.strategy = strategy
        self.devices = devices
        self.num_nodes = num_nodes
        default_callbacks: List[Callback[RawT, ProcessedT, BatchT, ModelOutputT]] = [
            CheckpointCallback(),
            RequeueCallback(),
            ConfigSnapshotCallback(),
            ProgressCallback(),
        ]
        if config['environment'] != RuntimeEnvironment.LOCAL:
            default_callbacks.append(RequeueCallback())

        default_singleton_cbs, default_non_singleton_cbs = separate_singletons(default_callbacks)
        input_singleton_cbs, input_non_singleton_cbs = separate_singletons(callbacks)

        singleton_cbs = {cb.__class__:cb for cb in [*default_singleton_cbs, *input_singleton_cbs]}.values()

        callbacks = [*singleton_cbs, *default_non_singleton_cbs, *input_non_singleton_cbs]
        self.callbacks = callbacks
        self.fabric = self._make_fabric(strategy, config['device'])

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
            run_config_yaml: str = ""
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

        if dataset.is_preprocessed(model.tokenizer):
            preprocessed_dataset = dataset
        else:
            preprocessed_dataset = PreprocessedDataset[RawT, ProcessedT](dataset, model.preprocess_example)
        
        dataloader = DataLoader(
            preprocessed_dataset, 
            batch_size=batch_size, 
            collate_fn=model.collate, 
            drop_last=False,
        )
        dataloader = self.fabric.setup_dataloaders(dataloader, move_to_device=self.config['device'] == 'cuda')

        self.fabric.call('on_fit_start', fabric=self.fabric, model=model, optimizer=optimizer, dataset=dataset,
            training_run_id=training_run_id, n_batches=len(dataloader), epochs=epochs, run_config_yaml=run_config_yaml)

        for _ in range(epochs):
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
                    batch=batch, 
                    training_run_id=training_run_id, 
                    batch_idx=batch_idx
                )

        self.fabric.call(
            'on_fit_end', 
            fabric=self.fabric,
            model=model, 
            optimizer=optimizer, 
            training_run_id=training_run_id
        )

    def _make_fabric(self, strategy: Strategy, accelerator: str) -> Fabric:
        return Fabric(
            strategy=strategy,
            devices=self.devices,
            num_nodes=self.num_nodes,
            callbacks=self.callbacks,
            accelerator=accelerator,
        )


def separate_singletons[RawT, ProcessedT, BatchT, ModelOutputT](
       callbacks: List[Callback[RawT, ProcessedT, BatchT, ModelOutputT]]
) -> tuple[
   List[Callback[RawT, ProcessedT, BatchT, ModelOutputT]],
   List[Callback[RawT, ProcessedT, BatchT, ModelOutputT]]
]:
    singletons = [c for c in callbacks if c.is_singleton]
    non_singletons = [c for c in callbacks if not c.is_singleton]
    return singletons, non_singletons
