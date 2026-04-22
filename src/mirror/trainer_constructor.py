from lightning.fabric.strategies.strategy import Strategy
from lightning.fabric.strategies.fsdp import FSDPStrategy
from mirror.callbacks.callback import Callback
from mirror.trainer import Trainer


class TrainerConstructor:
    def __init__(
            self,
            strategy: Strategy = FSDPStrategy(),
            devices: int = 1,
            num_nodes: int = 1,
            callbacks: list[Callback] = [],
    ) -> None:
        self.strategy = strategy
        self.devices = devices
        self.num_nodes = num_nodes
        self.callbacks = callbacks

    def construct_trainer[RawT, ProcessedT, BatchT, ModelOutputT](self) -> Trainer[RawT, ProcessedT, BatchT, ModelOutputT]:
        return Trainer(self.strategy, self.devices, self.num_nodes, self.callbacks)
