from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.schedulers.configure_scheduler import ConfigureScheduler
from mirror.datasets.mirror_dataset import MirrorDataset
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
):
    trainer.fit(
        model,
        data,
        preprocessor,
        checkpoint,
        epochs,
        batch_size,
        do_preprocess,
        run_config_yaml,
        val_data,
        test_data,
        val_check_interval,
        configure_scheduler,
    )

def preprocess(
        data: MirrorDataset,
        preprocessor: MirrorPreprocessor,
        slurm: SlurmConfig = SlurmConfig()
) -> None:
    data.preprocess(preprocessor.preprocess_example, slurm.nodes or 1)
