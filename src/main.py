import sys
from typing import Literal

Subcommand = Literal['fit'] | Literal['test'] | Literal['format'] | Literal['eval'] | Literal['infer']


def main(subcommand: Subcommand):
    from mirror.slurm_launcher import submit_slurm_job
    from mirror.slurm_util import parse_slurm_config
    submit_slurm_job(parse_slurm_config(sys.argv[1:]), sys.argv[1:])

    _run(subcommand)


def _run(subcommand: Subcommand):
    import warnings

    from lightning.fabric.utilities.warnings import PossibleUserWarning

    from mirror.cli_parsers import build_parser
    from mirror.config import init_config
    from mirror.models.model_util import instantiate_model
    from mirror.subcommands import evaluation, fit, format, infer
    from mirror.trainer_constructor import TrainerConstructor
    from mirror.util import is_login_node, resolve_config_args

    # These warnings happen internal to Fabric, so there's not much we can do about them.
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*Please use DTensor instead and we are deprecating ShardedTensor.*')
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*`load_state_dict` is deprecated and will be removed in future versions\\. Please use `load` instead.*')
    warnings.filterwarnings('ignore', category=UserWarning, message='.*Please use the new API settings to control TF32 behavior.*')
    warnings.filterwarnings('ignore', category=UserWarning, message='.*`_get_pg_default_device` will be deprecated, it only stays for backward-compatibility reason.*')
    # Local development warning
    warnings.filterwarnings('ignore', category=PossibleUserWarning, message='.*`srun` command is available on your system but is not used.*')

    match subcommand:
        case 'fit':
            parser = build_parser('fit')
            cfg = parser.parse_args(resolve_config_args(sys.argv[2:]))

            run_config_yaml = f"subcommand: fit\n{parser.dump(cfg)}"

            if hasattr(cfg, 'config'):
                del cfg.config  # pyright: ignore

            init_config(cfg.device)
            init = parser.instantiate_classes(cfg)
            trainer = init.trainer or TrainerConstructor()
            model = init.model

            trainer = trainer.construct_trainer()
            trainer.launch()
            model = instantiate_model(model, fabric=trainer.fabric)

            if is_login_node() and init.slurm.job_type == "local-download":
                print("Model downloaded/cached. Re-run on a compute node.")
                return

            del init.model # pyright: ignore
            del init.device # pyright: ignore

            fit(**{**init, "model": model, "trainer": trainer, "run_config_yaml": run_config_yaml})

        case 'format':
            parser = build_parser('format')
            cfg = parser.parse_args(resolve_config_args(sys.argv[2:]))

            if hasattr(cfg, 'config'):
                del cfg.config  # pyright: ignore

            init = parser.instantiate_classes(cfg)
            format(**init)

        case 'eval':
            parser = build_parser('eval')
            cfg = parser.parse_args(resolve_config_args(sys.argv[2:]))

            if hasattr(cfg, 'config'):
                del cfg.config  # pyright: ignore

            init_config(cfg.device)
            init = parser.instantiate_classes(cfg)

            from mirror.config import get_config
            from mirror.fabric_util import make_fabric

            config = get_config()
            fabric = make_fabric(
                init.strategy,
                config['device'],
                devices=init.slurm.ntasks_per_node or 1,
                num_nodes=init.slurm.nodes or 1,
            )
            fabric.launch()

            model = instantiate_model(init.model, fabric=fabric)

            if is_login_node() and init.slurm.job_type == "local-download":
                print("Model downloaded/cached. Re-run on a compute node.")
                return

            del init.model  # pyright: ignore
            del init.device  # pyright: ignore
            del init.strategy  # pyright: ignore

            evaluation(**{**init, "model": model, "fabric": fabric})

        case 'infer':
            from mirror.config import get_config
            from mirror.fabric_util import make_fabric

            parser = build_parser('infer')
            cfg = parser.parse_args(resolve_config_args(sys.argv[2:]))

            if hasattr(cfg, 'config'):
                del cfg.config  # pyright: ignore

            init_config(cfg.device)
            init_cfg = cfg.clone()
            init = parser.instantiate_classes(init_cfg)

            config = get_config()
            fabric = make_fabric(
                init.strategy,
                config['device'],
                devices=init.slurm.ntasks_per_node or 1,
                num_nodes=init.slurm.nodes or 1,
            )
            fabric.launch()

            model = instantiate_model(init.model, fabric=fabric)

            del init.model  # pyright: ignore
            del init.device  # pyright: ignore
            del init.strategy  # pyright: ignore

            infer(**{**init, "model": model, "fabric": fabric})

        case _:
            print(f'unimplemented subcommand: {subcommand}')


if __name__ == '__main__':
    from jsonargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("subcommand", type=Subcommand)
    cfg = parser.parse_args(sys.argv[1:2])
    main(cfg.subcommand)
