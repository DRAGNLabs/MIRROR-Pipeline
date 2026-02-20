
from dataclasses import asdict
from pathlib import Path
import shlex
import subprocess
import sys

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models.mirror_model import MirrorModel
from mirror.slurm_util import SlurmConfig
from mirror.trainer import Trainer
from mirror.util import is_login_node


def fit(
    data: MirrorDataset,
    model: MirrorModel,
    trainer: Trainer,
    checkpoint: CheckpointIdentifier | None = None,
    slurm: SlurmConfig = SlurmConfig(),
    epochs: int = 1,
    batch_size: int = 1,
    run_config_yaml: str = ''
):
    if slurm.submit and is_login_node():
        job_id = _submit_slurm_job(
            python_args=sys.argv[1:],
            slurm=slurm,
            num_nodes=trainer.num_nodes,
            devices=trainer.devices,
        )
        print(f"Submitted batch job {job_id}")
        return

    trainer.fit(
        model,
        data,
        checkpoint,
        epochs,
        batch_size,
        run_config_yaml,
    )

def preprocess(dataset: MirrorDataset, model: MirrorModel) -> None:
    pass

def _submit_slurm_job(*, python_args: list[str], slurm: SlurmConfig, num_nodes: int, devices: int) -> str:
    # Prevent recursion: job run should not submit again
    args = [a for a in python_args if not a.startswith("--slurm.submit")]
    args.append("--slurm.submit=false")

    repo_root = Path(__file__).resolve().parents[1]
    templates_dir = repo_root / "mirror" / "templates"

    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
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
        "chdir": str(repo_root),
        "activate_cmd": "mamba activate ./.env",
        "run_cmd": f"srun python src/main.py {shlex.join(python_args)}",
    }

    script = template.render(**context)
    
    res = subprocess.run(["sbatch"], input=script, text=True, capture_output=True, check=True)
    return res.stdout.strip().split()[-1]

