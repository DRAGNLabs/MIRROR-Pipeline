from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightning.fabric.strategies.strategy import Strategy
    from mirror.callbacks.callback import Callback
    from mirror.trainer import Trainer


class TrainerConstructor:
    def __init__(
            self,
            strategy: Strategy | None = None,
            devices: int = 1,
            num_nodes: int = 1,
            callbacks: list[Callback] = [],
    ) -> None:
        if strategy is None:
            from lightning.fabric.strategies.fsdp import FSDPStrategy
            self.strategy = FSDPStrategy()
        else:
            self.strategy = strategy
        self.devices = devices
        self.num_nodes = num_nodes
        self.callbacks = callbacks

    def construct_trainer(self) -> Trainer:
        from mirror.trainer import Trainer

        return Trainer(self.strategy, self.devices, self.num_nodes, self.callbacks)
