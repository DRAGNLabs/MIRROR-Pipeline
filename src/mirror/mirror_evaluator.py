from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.fabric.strategies.strategy import Strategy

from mirror.metrics.mirror_metric import MirrorMetric
from mirror.models.mirror_model import MirrorModel
from mirror.slurm_util import SlurmConfig


class MirrorEvaluator[RawT, ProcessedT, BatchT, ModelOutputT]:
    def __init__(
            self,
            strategy: Strategy = FSDPStrategy(),
    ) -> None:
        self.strategy = strategy

    def evaluate(
            self,
            model: MirrorModel[RawT, ProcessedT, BatchT, ModelOutputT],
            metrics: dict[str, MirrorMetric[RawT, ProcessedT, BatchT, ModelOutputT]],
            checkpoint_path: str | None = None,
            slurm: SlurmConfig = SlurmConfig(),
    ) -> dict:
        from mirror.config import get_config
        from mirror.fabric_util import make_fabric

        config = get_config()
        fabric = make_fabric(
            self.strategy,
            config['device'],
            devices=slurm.ntasks_per_node or 1,
            num_nodes=slurm.nodes or 1,
        )
        fabric.launch()

        model = fabric.setup(model)

        if checkpoint_path:
            fabric.load(checkpoint_path, {'model': model})

        model.eval()

        results = {}
        for label, metric in metrics.items():
            result = metric.get_metrics(model, fabric)
            results[label] = result

        return results
