from jsonargparse import ActionConfigFile, ArgumentParser
from typing import Literal
from inspect import signature
import warnings
import sys

from lightning.fabric.utilities.warnings import PossibleUserWarning

from mirror.config import init_config
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models.mirror_model import MirrorModel
from mirror.models.model_util import instantiate_model
from mirror.subcommands import fit, preprocess
from mirror.trainer import Trainer
from mirror.util import is_login_node

# These are required so that their items can be found easily by jsonargparse without
# having to give the full classpath
import lightning.fabric.strategies
import mirror.callbacks
import mirror.datasets
import mirror.models

Subcommand = Literal['fit'] | Literal['test'] | Literal['preprocess']

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
            parser = ArgumentParser()
            parser.add_argument("--config", action=ActionConfigFile)
            parser.add_function_arguments(fit, as_positional=False, skip={"model", "trainer", "run_config_yaml"})
            parser.add_subclass_arguments(MirrorModel, "model", required=True, instantiate=False)
            parser.add_subclass_arguments(Trainer, "trainer", required=False, instantiate=True)
            parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
            cfg = parser.parse_args(sys.argv[2:])
            
            run_config_yaml = f"subcommand: fit\n{parser.dump(cfg)}"

            if hasattr(cfg, 'config'):
                del cfg.config  # pyright: ignore

            init_config(cfg.device)
            init_cfg = cfg.clone()
            init = parser.instantiate_classes(init_cfg)
            trainer = init.trainer or Trainer()
            model = init.model

            trainer.launch()
            if not (is_login_node() and init.slurm.job_type == "compute"):
                model = instantiate_model(model, fabric=trainer.fabric)

            if is_login_node() and init.slurm.job_type == "local-download":
                print("Model downloaded/cached. Re-run on a compute node.")
                return
                
            del init.model # pyright: ignore
            del init.device # pyright: ignore

            fit(**{**init, "model": model, "trainer": trainer, "run_config_yaml": run_config_yaml})

        case 'preprocess':
            parser = ArgumentParser()
            parser.add_argument("--config", action=ActionConfigFile)
            parser.add_subclass_arguments(MirrorDataset, "dataset", required=True, instantiate=True)
            parser.add_subclass_arguments(MirrorModel, "model", required=True, instantiate=False)
            cfg = parser.parse_args(sys.argv[2:])
            init = parser.instantiate_classes(cfg)

            dataset = init.dataset
            model = instantiate_model(init.model, fabric=None)

            preprocess(dataset, model)
            
        case _:
            print(f'unimplemented subcommand: {subcommand}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("subcommand", type=Subcommand)
    cfg = parser.parse_args(sys.argv[1:2])
    main(cfg.subcommand)
