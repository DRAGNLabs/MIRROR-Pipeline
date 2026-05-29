from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.optimization.optimization_strategy import OptimizationStrategy
from mirror.schedulers.configure_scheduler import ConfigureScheduler
from mirror.datasets.mirror_dataset import MirrorDataset, preprocess_dataset
from mirror.models.inference_model import InferenceFriendlyModel
from mirror.models.mirror_model import MirrorModel
from mirror.preprocessors.mirror_preprocessor import InferenceFriendlyPreprocessor, MirrorPreprocessor
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
        optimization_strategy: OptimizationStrategy | None = None,
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
        optimization_strategy=optimization_strategy,
    )

def preprocess(
        data: MirrorDataset,
        preprocessor: MirrorPreprocessor,
) -> None:
    preprocess_dataset(data, preprocessor.preprocess_example)


def infer(
        model: InferenceFriendlyModel,
        text: str,
        num_tokens: int,
        checkpoint_path: str | None = None,
        preprocessor: InferenceFriendlyPreprocessor | None = None,
        temperature: float = 1.0,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        slurm: SlurmConfig = SlurmConfig(),
) -> None:
    from mirror.predictor import Predictor
    result = Predictor().predict(
        model=model,
        checkpoint_path=checkpoint_path,
        text=text,
        num_tokens=num_tokens,
        preprocessor=preprocessor,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    print(result)
