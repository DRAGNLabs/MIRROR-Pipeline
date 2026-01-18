from jsonargparse import auto_parser
from typing import List, Literal
import warnings
import subprocess
import sys
import shlex
from pathlib import Path

from lightning.fabric.strategies.strategy import Strategy
from lightning.fabric.strategies.fsdp import FSDPStrategy

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from mirror.callbacks.callback import Callback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models.placeholder_model import PlaceholderModel
from mirror.trainer import Trainer
from mirror.util import is_login_node
from mirror.slurm_util import SlurmConfig

from dataclasses import asdict

# These are required so that their items can be found easily by jsonargparse without
# having to give the full classpath
import lightning.fabric.strategies
import mirror.datasets

Subcommand = Literal['fit'] | Literal['test']


def main(
    subcommand: Subcommand,
    data: MirrorDataset,
    strategy: Strategy = FSDPStrategy(),
    devices: int = 1,
    num_nodes: int = 1,
    callbacks: List[Callback] = [],
    checkpoint: CheckpointIdentifier | None = None,
    slurm: SlurmConfig = SlurmConfig(),
):
    # These warnings happen internal to Fabric, so there's not much we can do about them.
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*Please use DTensor instead and we are deprecating ShardedTensor.*')
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*`load_state_dict` is deprecated and will be removed in future versions\\. Please use `load` instead.*')
    warnings.filterwarnings('ignore', category=UserWarning, message='.*Please use the new API settings to control TF32 behavior.*')
    warnings.filterwarnings('ignore', category=UserWarning, message='.*`_get_pg_default_device` will be deprecated, it only stays for backward-compatibility reason.*')

    if slurm.submit and is_login_node():
        job_id = _submit_slurm_job(python_args=sys.argv[1:], slurm=slurm, num_nodes=num_nodes, devices=devices)
        print(f"Submitted batch job {job_id}")
        return
    
    match subcommand:
        case 'fit':
            fit(data, strategy, devices, num_nodes, callbacks, checkpoint)
        case _:
            print(f'unimplemented subcommand: {subcommand}')


def fit(
    dataset: MirrorDataset,
    strategy: Strategy,
    devices: int,
    num_nodes: int,
    callbacks: List[Callback],
    checkpoint: CheckpointIdentifier | None,
):
    trainer = Trainer(strategy, devices, num_nodes, callbacks)

    trainer.launch()

    with trainer.fabric.init_module():
        model = PlaceholderModel()

    trainer.fit(model, dataset, checkpoint)

def _submit_slurm_job(*, python_args: list[str], slurm: SlurmConfig, num_nodes: int, devices: int) -> str:
    # Prevent recursion: job run should not submit again
    args = [a for a in python_args if not a.startswith("--slurm.submit")]
    args.append("--slurm.submit=false")

    repo_root = Path(__file__).resolve().parents[1]
    templates_dir = repo_root / "src" / "mirror" / "templates"

    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("slurm.jinja")

    slurm_ctx = asdict(slurm)

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

if __name__ == '__main__':
    parser = auto_parser(main)
    cfg = parser.parse_args()
    if hasattr(cfg, 'config'):
        del cfg.config  # pyright: ignore

    init = parser.instantiate_classes(cfg)

    main(**init)
