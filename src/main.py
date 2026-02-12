from jsonargparse import auto_parser
from typing import List, Literal
import warnings
import subprocess
import sys
import shlex
from pathlib import Path

from lightning.fabric.strategies.strategy import Strategy
from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.fabric.strategies.single_device import SingleDeviceStrategy
from lightning.fabric.utilities.warnings import PossibleUserWarning

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from mirror.callbacks.callback import Callback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.config import init_config
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.datasets.util import load_hf_tk_from_cache_or_map
from mirror.models.placeholder_model import PlaceholderModel
from mirror.trainer import Trainer
from mirror.util import is_login_node
from mirror.slurm_util import SlurmConfig

from dataclasses import asdict

# These are required so that their items can be found easily by jsonargparse without
# having to give the full classpath
import lightning.fabric.strategies
import mirror.callbacks
import mirror.datasets

Subcommand = Literal['fit'] | Literal['test'] | Literal['preprocess']

# This is only ever assigned by the parser dump 
# Could change to pass as parameter to main when parser is updated/changed
run_config_yaml = ""

def main(
    subcommand: Subcommand,
    data: MirrorDataset,
    strategy: Strategy = FSDPStrategy(),
    devices: int = 1,
    num_nodes: int = 1,
    callbacks: List[Callback] = [],
    checkpoint: CheckpointIdentifier | None = None,
    slurm: SlurmConfig = SlurmConfig(),
    epochs: int = 1,
    batch_size: int = 1,
    # reset_cache: bool = False,
    device: Literal['cpu', 'cuda'] | None = None,
):
    # These warnings happen internal to Fabric, so there's not much we can do about them.
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*Please use DTensor instead and we are deprecating ShardedTensor.*')
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*`load_state_dict` is deprecated and will be removed in future versions\\. Please use `load` instead.*')
    warnings.filterwarnings('ignore', category=UserWarning, message='.*Please use the new API settings to control TF32 behavior.*')
    warnings.filterwarnings('ignore', category=UserWarning, message='.*`_get_pg_default_device` will be deprecated, it only stays for backward-compatibility reason.*')
    # Local development warning
    warnings.filterwarnings('ignore', category=PossibleUserWarning, message='.*`srun` command is available on your system but is not used.*')

    config = init_config(device)
    if config['device'] == 'cpu' and isinstance(strategy, FSDPStrategy):
        strategy = SingleDeviceStrategy(device="cpu")

    if slurm.submit and is_login_node():
        job_id = _submit_slurm_job(python_args=sys.argv[1:], slurm=slurm, num_nodes=num_nodes, devices=devices)
        print(f"Submitted batch job {job_id}")
        return
    
    match subcommand:
        case 'preprocess':
            preprocess(data, 
            # reset_cache
            )
        case 'fit':
            fit(data, strategy, devices, num_nodes, callbacks, checkpoint, epochs, batch_size)
        case _:
            print(f'unimplemented subcommand: {subcommand}')

def preprocess(
    dataset: MirrorDataset,
    # reset_cache: bool,
    # model: MirrorModel,
):
    print("Beginning preprocess...") #####################################################################################
    model = PlaceholderModel()
    tk_id = model.tokenizer.tokenization_id
    model.get_column_names(dataset.data_column, dataset.label_column)
    print("Pre-if-statement... ") ########################################################################################
    
    ds = load_hf_tk_from_cache_or_map(dataset, tk_id, model.preprocess_row, dataset.reset_cache) 
        # TODO: fix preprocess example to expect ds row when we have a hf_ds

    print("Preprocessing complete.")

def fit(
    dataset: MirrorDataset,
    strategy: Strategy,
    devices: int,
    num_nodes: int,
    callbacks: List[Callback],
    checkpoint: CheckpointIdentifier | None,
    epochs: int,
    batch_size: int,
):
    trainer = Trainer(strategy, devices, num_nodes, callbacks)

    trainer.launch()

    with trainer.fabric.init_module():
        model = PlaceholderModel()

    trainer.fit(model, dataset, checkpoint, epochs, batch_size, run_config_yaml=run_config_yaml)

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
    print("Name does = __main__...") ################################################################################
    parser = auto_parser(main)
    cfg = parser.parse_args()
    run_config_yaml = parser.dump(cfg)
    if hasattr(cfg, 'config'):
        del cfg.config  # pyright: ignore

    init = parser.instantiate_classes(cfg)
    print("Calling main...") ########################################################################################
    main(**init)
