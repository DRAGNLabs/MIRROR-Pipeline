from dataclasses import asdict
from pathlib import Path
import shlex
import subprocess
import sys

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.schedulers.configure_scheduler import ConfigureScheduler
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models.mirror_model import MirrorModel
from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.slurm_util import SlurmConfig
from mirror.trainer import Trainer
from mirror.util import is_login_node


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
    if slurm.job_type == "compute" and is_login_node():
        job_id = _submit_slurm_job(
            python_args=sys.argv[1:],
            slurm=slurm,
            num_nodes=trainer.num_nodes,
            devices=trainer.devices,
        )
        print(f"Submitted batch job {job_id}")
        return

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

def preprocess(
        data: MirrorDataset,
        preprocessor: MirrorPreprocessor,
        slurm: SlurmConfig = SlurmConfig()
) -> None:
    if slurm.job_type == "compute" and is_login_node():
        job_id = _submit_slurm_job(
            python_args=sys.argv[1:],
            slurm=slurm,
            num_nodes=slurm.nodes or 1,
            devices=slurm.ntasks_per_node or 1,
        )
        print(f"Submitted batch job {job_id}")
        return
    
    data.preprocess(preprocessor.preprocess_example, slurm.nodes or 1)

def _submit_slurm_job(
        *,
        python_args: list[str],
        slurm: SlurmConfig,
        num_nodes: int,
        devices: int
) -> str:
    # Prevent recursion: job run should not submit again
    args = [a for a in python_args if not a.startswith("--slurm.submit")]
    args.append("--slurm.submit=false")

    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("slurm.jinja")

    slurm_ctx = asdict(slurm)

    if slurm_ctx["nodes"] is None:
        slurm_ctx["nodes"] = num_nodes

    if slurm_ctx["ntasks_per_node"] is None:
        slurm_ctx["ntasks_per_node"] = devices

    if slurm_ctx["gpus_per_node"] is None:
        slurm_ctx["gpus_per_node"] = devices

    context = {
        **slurm_ctx,
        "chdir": str(Path.cwd()),
        "activate_cmd": "mamba activate ./.env",
        "run_cmd": f"srun python {sys.argv[0]} {shlex.join(python_args)}",
    }

    script = template.render(**context)

    res = subprocess.run(["sbatch"], input=script, text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (exit {res.returncode}):\n{res.stderr}\n\nGenerated script:\n{script}"
        )
    return res.stdout.strip().split()[-1]
