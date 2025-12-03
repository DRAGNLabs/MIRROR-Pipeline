from jsonargparse import auto_parser
from typing import List, Literal
import warnings

from lightning.fabric.strategies.strategy import Strategy
from lightning.fabric.strategies.fsdp import FSDPStrategy

from mirror.callbacks.callback import Callback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models.placeholder_model import PlaceholderModel
from mirror.trainer import Trainer

# These are required so that their items can be found easily by jsonargparse without
# having to give the full classpath
import lightning.fabric.strategies
import mirror.datasets

from jinja2 import Environment, FileSystemLoader
import subprocess

Subcommand = Literal['fit'] | Literal['test']

def main(
    subcommand: Subcommand,
    data: MirrorDataset,
    time: str = '1:00:00',
    ntasks_per_node: int = 1,
    nodes: int = 1,
    gpus_per_node: int = 1,
    mem_per_cpu: str = '128g',
    strategy: Strategy = FSDPStrategy(),
    devices: int = 1,
    num_nodes: int = 1,
    callbacks: List[Callback] = [],
    checkpoint: CheckpointIdentifier | None = None,
    
):
    try:
        result = subprocess.run(
            ["sbatch"],
            input=slurm_job.encode('utf-8'), # Encode the string to bytes
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error submitting SLURM job: {e}")
        print(f"Stderr: {e.stderr}")


if __name__ == '__main__':
    env = Environment(loader = FileSystemLoader('templates'))
    template = env.get_template('slurm.jinja')

    parser = auto_parser(main)
    cfg = parser.parse_args()
    if hasattr(cfg, 'config'):
        del cfg.config  # pyright: ignore

    init = parser.instantiate_classes(cfg)

    slurm_job = template.render(**init)
    main(slurm_job, **init)
