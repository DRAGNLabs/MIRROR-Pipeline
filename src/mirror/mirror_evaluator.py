from lightning.fabric.strategies.ddp import DDPStrategy
from lightning.fabric.strategies.strategy import Strategy

from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.metrics.mirror_metric import MirrorMetric
from mirror.models.mirror_model import MirrorModel
from mirror.slurm_util import SlurmConfig


class MirrorEvaluator:
    def __init__(
            self,
            metrics: dict[str, MirrorMetric],
            strategy: Strategy = DDPStrategy(),
    ) -> None:
        self.metrics = metrics
        self.strategy = strategy

    def evaluate(
            self, 
            model: MirrorModel, 
            checkpoint: CheckpointIdentifier | None = None, 
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

        if checkpoint:
            fabric.load(checkpoint.path, {'model': model})

        model.eval()

        results = {}
        for label, metric in self.metrics.items():
            result = metric.get_metrics(model, fabric)
            results[label] = result

        return results
