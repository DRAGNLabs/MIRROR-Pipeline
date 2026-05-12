from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.metrics.mirror_metric import MirrorMetric
from mirror.mirror_evaluator import MirrorEvaluator
from mirror.schedulers.configure_scheduler import ConfigureScheduler
from mirror.datasets.mirror_dataset import MirrorDataset, preprocess_dataset
from mirror.models.mirror_model import MirrorModel
from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.slurm_util import SlurmConfig
from mirror.trainer import Trainer


def fit(
        data: MirrorDataset,
        model: MirrorModel,
        trainer: Trainer,
        preprocessor: MirrorPreprocessor | None = None,
        checkpoint: CheckpointIdentifier | None = None,
        slurm: SlurmConfig = SlurmConfig(),
        epochs: int = 1,
        batch_size: int = 1,
        do_preprocess: bool = False,
        run_config_yaml: str = '',
        val_data: MirrorDataset | None = None,
        test_data: MirrorDataset | None = None,
        val_check_interval: int = 1,
        configure_scheduler: ConfigureScheduler | None = None,
        shuffle: bool = True,
):
    trainer.fit(
        model=model,
        dataset=data,
        preprocessor=preprocessor,
        checkpoint=checkpoint,
        epochs=epochs,
        batch_size=batch_size,
        do_preprocess=do_preprocess,
        run_config_yaml=run_config_yaml,
        val_dataset=val_data,
        test_dataset=test_data,
        val_check_interval=val_check_interval,
        configure_scheduler=configure_scheduler,
        shuffle=shuffle,
    )

def evaluation(
        model: MirrorModel,
        evaluator: MirrorEvaluator,
        metrics: dict[str, MirrorMetric],
        checkpoint_path: str | None = None,
        slurm: SlurmConfig = SlurmConfig(),
) -> None:
    results = evaluator.evaluate(model, metrics, checkpoint_path, slurm)
    for label, result in results.items():
        print(f"{label}: {result}")

def preprocess(
        data: MirrorDataset,
        preprocessor: MirrorPreprocessor,
        slurm: SlurmConfig = SlurmConfig()
) -> None:
    preprocess_dataset(data, preprocessor.preprocess_example, slurm.nodes or 1)
