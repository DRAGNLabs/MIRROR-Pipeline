from jsonargparse import ArgumentParser, auto_parser
from typing import Literal
import warnings
import subprocess
import sys
import shlex
from pathlib import Path

from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.fabric.strategies.single_device import SingleDeviceStrategy
from lightning.fabric.utilities.warnings import PossibleUserWarning

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.config import get_config, init_config
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models import __init__
from mirror.models.mirror_model import MirrorModel
from mirror.models.model_util import instantiate_model
from mirror.trainer import Trainer
from mirror.util import is_login_node
from mirror.slurm_util import SlurmConfig

from dataclasses import asdict

# These are required so that their items can be found easily by jsonargparse without
# having to give the full classpath
import lightning.fabric.strategies
import mirror.callbacks
import mirror.datasets

Subcommand = Literal['fit'] | Literal['test']
run_config_yaml = ""


def main(subcommand: Subcommand):
    # These warnings happen internal to Fabric, so there's not much we can do about them.
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*Please use DTensor instead and we are deprecating ShardedTensor.*')
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*`load_state_dict` is deprecated and will be removed in future versions\\. Please use `load` instead.*')
    warnings.filterwarnings('ignore', category=UserWarning, message='.*Please use the new API settings to control TF32 behavior.*')
    warnings.filterwarnings('ignore', category=UserWarning, message='.*`_get_pg_default_device` will be deprecated, it only stays for backward-compatibility reason.*')
    # Local development warning
    warnings.filterwarnings('ignore', category=PossibleUserWarning, message='.*`srun` command is available on your system but is not used.*')

    match subcommand:
        case 'fit':
            parser = auto_parser(fit, as_positional=False)
            parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
            cfg = parser.parse_args(sys.argv[2:])

            global run_config_yaml
            run_config_yaml = f"subcommand: fit\n{parser.dump(cfg)}"

            if hasattr(cfg, 'config'):
                del cfg.config  # pyright: ignore

            init_config(cfg.device)
            init = parser.instantiate_classes(cfg)
            fit(
                data=init.data,
                model=init.model,
                trainer=init.trainer,
                checkpoint=init.checkpoint,
                slurm=init.slurm,
                epochs=init.epochs,
                batch_size=init.batch_size,
            )

        case _:
            print(f'unimplemented subcommand: {subcommand}')


def fit(
    data: MirrorDataset,
    model: MirrorModel,
    trainer: Trainer | None = None,
    checkpoint: CheckpointIdentifier | None = None,
    slurm: SlurmConfig = SlurmConfig(),
    epochs: int = 1,
    batch_size: int = 1,
):
    if trainer is None:
        if get_config()['device'] == 'cpu':
            trainer = Trainer(strategy=SingleDeviceStrategy(device="cpu"))
        else:
            trainer = Trainer()

    if slurm.submit and is_login_node():
        job_id = _submit_slurm_job(
            python_args=sys.argv[1:],
            slurm=slurm,
            num_nodes=trainer.num_nodes,
            devices=trainer.devices,
        )
        print(f"Submitted batch job {job_id}")
        return

    if get_config()['device'] == 'cpu' and isinstance(trainer.strategy, FSDPStrategy):
        trainer = Trainer(
            strategy=SingleDeviceStrategy(device="cpu"),
            devices=trainer.devices,
            num_nodes=trainer.num_nodes,
            callbacks=trainer.callbacks,
        )

    trainer.launch()

    model = instantiate_model(model, fabric=trainer.fabric)

    if is_login_node():
        print("Model downloaded/cached. Re-run on a compute node.")
        return
    
    trainer.fit(
        model,
        data,
        checkpoint,
        epochs,
        batch_size,
        run_config_yaml,
    )

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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("subcommand", type=Subcommand)
    cfg = parser.parse_args(sys.argv[1:2])
    main(cfg.subcommand)
