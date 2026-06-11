from lightning import Fabric

from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.metrics.mirror_metric import MirrorMetric
from mirror.optimization.optimization_strategy import OptimizationStrategy
from mirror.schedulers.configure_scheduler import ConfigureScheduler
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models.mirror_model import MirrorModel
from mirror.formatters.mirror_formatter import MirrorFormatter
from mirror.slurm_util import SlurmConfig
from mirror.trainer import Trainer


def fit(
        data: MirrorDataset,
        model: MirrorModel,
        trainer: Trainer,
        formatter: MirrorFormatter | None = None,
        checkpoint: CheckpointIdentifier | None = None,
        slurm: SlurmConfig = SlurmConfig(),
        epochs: int = 1,
        batch_size: int = 1,
        run_config_yaml: str = '',
        val_data: MirrorDataset | None = None,
        test_data: MirrorDataset | None = None,
        val_check_interval: int = 1,
        configure_scheduler: ConfigureScheduler | None = None,
        shuffle: bool = True,
        optimization_strategy: OptimizationStrategy | None = None,
):
    trainer.fit(
        model=model,
        dataset=data,
        formatter=formatter,
        checkpoint=checkpoint,
        epochs=epochs,
        batch_size=batch_size,
        run_config_yaml=run_config_yaml,
        val_dataset=val_data,
        test_dataset=test_data,
        val_check_interval=val_check_interval,
        configure_scheduler=configure_scheduler,
        shuffle=shuffle,
        optimization_strategy=optimization_strategy,
    )

def evaluation(
        model: MirrorModel,
        metrics: dict[str, MirrorMetric],
        fabric: Fabric,
        checkpoint_path: str | None = None,
        slurm: SlurmConfig = SlurmConfig(),
) -> None:
    model = fabric.setup(model)

    if checkpoint_path:
        fabric.load(checkpoint_path, {'model': model})

    model.eval()

    results = {}
    for label, metric in metrics.items():
        result = metric.get_metrics(model, fabric)
        results[label] = result

    for label, result in results.items():
        print(f"{label}: {result}")

def format(
        data: MirrorDataset,
        formatter: MirrorFormatter,
) -> None:
    formatter.format_data(data)
