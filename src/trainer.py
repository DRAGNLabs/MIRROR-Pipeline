from lightning import Fabric

from datasets.mirror_dataset import MirrorDataset
from models.mirror_model import MirrorModel


class Trainer:
    def __init__(self) -> None:
        self.fabric = Fabric()

    def fit(self, model: MirrorModel, dataset: MirrorDataset):
        pass
