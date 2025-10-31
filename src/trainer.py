from lightning import Fabric
from torch.utils.data import DataLoader
import datetime

from datasets.mirror_dataset import MirrorDataset
from datasets.preprocessed_dataset import PreprocessedDataset
from models.mirror_model import MirrorModel


class Trainer:
    def __init__(self) -> None:
        self.fabric = Fabric()

    def fit(self, model: MirrorModel, dataset: MirrorDataset):
        training_run_id = datetime.datetime.now().isoformat()

        model, optimizer = self.fabric.setup(model, model.configure_optimizers())

        preprocessed_dataset = PreprocessedDataset(dataset, model.tokenizer)
        dataloader = DataLoader(preprocessed_dataset)
        dataloader = self.fabric.setup_dataloaders(dataloader)

        for tokens, attention_mask in dataloader:
            optimizer.zero_grad()
            loss = model.training_step(tokens, attention_mask)
            self.fabric.backward(loss)
            optimizer.step()
