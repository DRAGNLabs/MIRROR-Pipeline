from lightning import Fabric
from torch.utils.data import DataLoader
from typing import List
import datetime

from mirror.callbacks.callback import Callback
from mirror.callbacks.checkpoint_callback import CheckpointCallback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.datasets.preprocessed_dataset import PreprocessedDataset
from mirror.models.mirror_model import MirrorModel


class Trainer:
    def __init__(self, callbacks: List[Callback] = []) -> None:
        default_callbacks = [
            CheckpointCallback()
        ]

        callbacks = [*default_callbacks, *callbacks]
        self.fabric = Fabric(callbacks=callbacks)

    def fit(self, model: MirrorModel, dataset: MirrorDataset, checkpoint: CheckpointIdentifier | None = None):
        training_run_id = datetime.datetime.now().isoformat()

        model, optimizer = self.fabric.setup(model, model.configure_optimizers())

        if checkpoint:
            state = self.fabric.load(checkpoint.path)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])

        preprocessed_dataset = PreprocessedDataset(dataset, model.tokenizer)
        dataloader = DataLoader(preprocessed_dataset)
        dataloader = self.fabric.setup_dataloaders(dataloader)

        self.fabric.call('on_fit_start', fabric=self.fabric, model=model, optimizer=optimizer, training_run_id=training_run_id)

        for batch_idx, (tokens, attention_mask) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model.training_step(tokens, attention_mask)
            self.fabric.backward(loss)
            optimizer.step()

            self.fabric.call('on_train_batch_end', fabric=self.fabric, model=model, loss=loss, tokens=tokens, attention_mask=attention_mask, batch_idx=batch_idx)

        self.fabric.call('on_fit_end', fabric=self.fabric, model=model, optimizer=optimizer, training_run_id=training_run_id)
