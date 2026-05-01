from lightning.fabric.strategies.strategy import Strategy
from lightning.fabric.strategies.fsdp import FSDPStrategy
from mirror.callbacks.callback import Callback
from mirror.metrics.extra_metrics_getter import ExtraMetricsGetter
from mirror.trainer import Trainer


class TrainerConstructor:
    def __init__(
            self,
            strategy: Strategy = FSDPStrategy(),
            devices: int = 1,
            num_nodes: int = 1,
            callbacks: list[Callback] = [],
            extra_metrics_getter: ExtraMetricsGetter | None = None,
    ) -> None:
        self.strategy = strategy
        self.devices = devices
        self.num_nodes = num_nodes
        self.callbacks = callbacks
        self.extra_metrics_getter = extra_metrics_getter

    def construct_trainer(self) -> Trainer:
        return Trainer(
            self.strategy,
            self.devices,
            self.num_nodes,
            self.callbacks,
            self.extra_metrics_getter,
        )
