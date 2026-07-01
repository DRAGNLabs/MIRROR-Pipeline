from lightning import Fabric

from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.metrics.mirror_metric import MirrorMetric
from mirror.optimization.optimization_strategy import OptimizationStrategy
from mirror.schedulers.configure_scheduler import ConfigureScheduler
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models.trainable_model import TrainableModel
from mirror.models.inference_model import InferenceModel
from mirror.formatters.infer_friendly_formatter import InferFriendlyFormatter
from mirror.formatters.mirror_formatter import MirrorFormatter
from mirror.slurm_util import SlurmConfig
from mirror.trainer import Trainer


def fit(
        data: MirrorDataset,
        model: TrainableModel,
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
        model: TrainableModel,
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

def infer(
        model: InferenceModel,  # type: ignore[type-arg]
        fabric: Fabric,
        max_new_tokens: int,
        text: str | None = None,
        interactive: bool = False,
        checkpoint_path: str | None = None,
        formatter: InferFriendlyFormatter | None = None,
        temperature: float = 1.0,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        slurm: SlurmConfig = SlurmConfig(),
) -> None:
    from mirror.predictor import Predictor
    predictor = Predictor()

    if interactive:
        predictor.predict_interactive(
            model=model,
            fabric=fabric,
            checkpoint_path=checkpoint_path,
            formatter=formatter,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        return

    if text is None:
        raise ValueError("`text` is required unless `--interactive` is set.")

    result = predictor.predict(
        model=model,
        fabric=fabric,
        checkpoint_path=checkpoint_path,
        formatter=formatter,
        text=text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    print(result)

def format(
        data: MirrorDataset,
        formatter: MirrorFormatter,
) -> None:
    formatter.format_data(data)
