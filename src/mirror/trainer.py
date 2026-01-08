from lightning import Fabric
from torch.utils.data import DataLoader
from typing import List
import datetime
import math

from lightning.fabric.strategies.strategy import Strategy
from lightning.fabric.strategies.fsdp import FSDPStrategy

from mirror.callbacks.callback import Callback
from mirror.callbacks.checkpoint_callback import CheckpointCallback
from mirror.callbacks.requeue_callback import RequeueCallback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.datasets.preprocessed_dataset import PreprocessedDataset
from mirror.models.mirror_model import MirrorModel
from mirror.util import is_login_node, pad_to_longest


class Trainer:
    def __init__(
            self,
            strategy: Strategy = FSDPStrategy(),
            devices: int = 1,
            num_nodes: int = 1,
            callbacks: List[Callback] = [], 
    ) -> None:
        default_callbacks: List[Callback] = [
            CheckpointCallback(),
            RequeueCallback(),
        ]

        default_singleton_cbs, default_non_singleton_cbs = separate_singletons(default_callbacks)
        input_singleton_cbs, input_non_singleton_cbs = separate_singletons(callbacks)

        singleton_cbs = {cb.__class__:cb for cb in [*default_singleton_cbs, *input_singleton_cbs]}.values()

        callbacks = [*singleton_cbs, *default_non_singleton_cbs, *input_non_singleton_cbs]
        self.fabric = Fabric(strategy=strategy, devices=devices, num_nodes=num_nodes, callbacks=callbacks)

    def launch(self):
        self.fabric.launch()

    def fit(self, model: MirrorModel, dataset: MirrorDataset, checkpoint: CheckpointIdentifier | None = None, batch_size=1):
        training_run_id = datetime.datetime.now().isoformat()

        model, optimizer = self.fabric.setup(
            model,
            model.configure_optimizers(),
            move_to_device=not is_login_node()
        )

        if checkpoint:
            # models and optimizers are treated specially: they are populated via their load_state_dict
            # methods internally to fabric.load. Anything else in the state dict is just set in place.
            state = {
                'model': model,
                'optimizer': optimizer,
            }
            self.fabric.load(checkpoint.path, state)

        def collate(batch):
            return pad_to_longest(batch, pad_token=model.tokenizer.pad_token_id)

        preprocessed_dataset = PreprocessedDataset(dataset, model.tokenizer)
        dataloader = DataLoader(preprocessed_dataset, batch_size=batch_size, collate_fn=collate, drop_last=True)
        dataloader = self.fabric.setup_dataloaders(dataloader, move_to_device=not is_login_node())

        self.fabric.call('on_fit_start', fabric=self.fabric, model=model, optimizer=optimizer, dataset=dataset,
            training_run_id=training_run_id)

        for batch_idx, (tokens, attention_mask) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model.training_step(tokens, attention_mask)
            self.fabric.backward(loss)
            optimizer.step()

            self.fabric.call('on_train_batch_end', fabric=self.fabric, model=model, optimizer=optimizer, loss=loss, 
                tokens=tokens, attention_mask=attention_mask, training_run_id=training_run_id, batch_idx=batch_idx)

        self.fabric.call('on_fit_end', fabric=self.fabric, model=model, optimizer=optimizer, 
            training_run_id=training_run_id)


def separate_singletons(callbacks: List[Callback]):
    singletons = [c for c in callbacks if c.is_singleton]
    non_singletons = [c for c in callbacks if not c.is_singleton]
    return singletons, non_singletons


