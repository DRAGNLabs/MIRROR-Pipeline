from lightning import Fabric
from torch.utils.data import DataLoader
from typing import List
import datetime

from mirror.callbacks.callback import Callback
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.datasets.preprocessed_dataset import PreprocessedDataset
from mirror.models.mirror_model import MirrorModel


class Trainer:
    def __init__(self, callbacks: List[Callback] = []) -> None:
        default_callbacks = []

        callbacks = [*default_callbacks, *callbacks]
        self.fabric = Fabric(callbacks=callbacks)

    def fit(self, model: MirrorModel, dataset: MirrorDataset):
        training_run_id = datetime.datetime.now().isoformat()

        model, optimizer = self.fabric.setup(model, model.configure_optimizers())

        preprocessed_dataset = PreprocessedDataset(dataset, model.tokenizer)
        dataloader = DataLoader(preprocessed_dataset)
        dataloader = self.fabric.setup_dataloaders(dataloader)

        for batch_idx, (tokens, attention_mask) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model.training_step(tokens, attention_mask)
            self.fabric.backward(loss)
            optimizer.step()

            self.fabric.call('on_train_batch_end', fabric=self.fabric, model=model, loss=loss, tokens=tokens, attention_mask=attention_mask, batch_idx=batch_idx)
