from jsonargparse import auto_parser
from typing import List, Literal
import warnings
import subprocess
import sys
import shlex
from pathlib import Path

from lightning.fabric.strategies.strategy import Strategy
from lightning.fabric.strategies.fsdp import FSDPStrategy

from mirror.callbacks.callback import Callback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models.placeholder_model import PlaceholderModel
from mirror.trainer import Trainer
from mirror.util import is_login_node
from mirror.slurm_util import SlurmConfig

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

    match subcommand:
        case 'fit':
            fit(data, strategy, devices, num_nodes, callbacks, checkpoint, slurm)
        case _:
            print(f'unimplemented subcommand: {subcommand}')


def fit(
    dataset: MirrorDataset,
    strategy: Strategy,
    devices: int,
    num_nodes: int,
    callbacks: List[Callback],
    checkpoint: CheckpointIdentifier | None, 
    slurm: SlurmConfig,
):
    if slurm.submit and is_login_node():
        job_id = _submit_slurm_job(python_args=sys.argv[1:], slurm=slurm, num_nodes=num_nodes)
        print(f"Submitted batch job {job_id}")
        return
    
    trainer = Trainer(strategy, devices, num_nodes, callbacks)

    trainer.launch()

    with trainer.fabric.init_module():
        model = PlaceholderModel()

    trainer.fit(model, dataset, checkpoint)

def _submit_slurm_job(*, python_args: list[str], slurm: SlurmConfig, num_nodes: int) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    (repo_root / "slurm_logs").mkdir(exist_ok=True)

    sbatch_lines = [
        "#!/bin/bash --login",
        f"#SBATCH --time={slurm.time}",
        f"#SBATCH --ntasks-per-node={slurm.ntasks_per_node}",
        f"#SBATCH --nodes={num_nodes}",
        f"#SBATCH --gpus-per-node={slurm.gpus_per_node}",
        f"#SBATCH --mem-per-cpu={slurm.mem_per_cpu}",
        f"#SBATCH --output={slurm.output}",
        f"#SBATCH --open-mode={slurm.open_mode}",
        f"#SBATCH --signal={slurm.signal}",
    ]

    if slurm.requeue:
        sbatch_lines.append("SBATCH --requeue")

    sbatch_lines += [
        f"#SBATCH --chdir={repo_root}",
        "",
        "set -euo pipefail",
        "mamba activate ./.env",
        "",
        f"srun python src/main.py {shlex.join(python_args)}",
        "",
    ]

    script = "\n".join(sbatch_lines)

    res = subprocess.run(["sbatch"], input=script, text=True, capture_output=True, check=True)

    job_id = res.stdout.strip().split()[-1]
    return job_id
    
if __name__ == '__main__':
    parser = auto_parser(main)
    cfg = parser.parse_args()
    if hasattr(cfg, 'config'):
        del cfg.config  # pyright: ignore

    init = parser.instantiate_classes(cfg)

    main(**init)
