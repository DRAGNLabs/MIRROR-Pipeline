from jsonargparse import ActionConfigFile, ArgumentParser
from lightning.fabric.strategies.strategy import Strategy

from mirror.subcommands import evaluation, fit, format, infer
from mirror.models.trainable_model import TrainableModel
from mirror.models.inference_model import InferenceModel
from mirror.trainer_constructor import TrainerConstructor

# These imports register the subclasses so jsonargparse can resolve them by
# short name (e.g. `class_path: WikitextDataset`) in config files.
import lightning.fabric.strategies  # noqa: F401
import mirror.callbacks  # noqa: F401
import mirror.datasets  # noqa: F401
import mirror.models  # noqa: F401
import mirror.optimization  # noqa: F401
import mirror.formatters  # noqa: F401
import mirror.schedulers  # noqa: F401
import mirror.interventions  # noqa: F401
import mirror.metrics  # noqa: F401


def build_parser(subcommand: str) -> ArgumentParser:
    """Build the argument parser for a subcommand.

    This is the single source of truth for each subcommand's arguments, shared
    by the runtime (main.py) and the config-schema generator (config_schema.py)
    so the two never drift apart.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    match subcommand:
        case 'fit':
            parser.add_function_arguments(fit, as_positional=False, skip={"model", "trainer", "run_config_yaml"})
            parser.add_subclass_arguments(TrainableModel, "model", required=True, instantiate=False)
            parser.add_subclass_arguments(TrainerConstructor, "trainer", required=False, instantiate=True)
            parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
        case 'format':
            parser.add_function_arguments(format, as_positional=False)
        case 'eval':
            parser.add_function_arguments(evaluation, as_positional=False, skip={"model", "fabric"})
            parser.add_subclass_arguments(TrainableModel, "model", required=True, instantiate=False)
            parser.add_argument("--strategy", type=Strategy)
            parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
        case 'infer':
            parser.add_function_arguments(infer, as_positional=False, skip={"model", "fabric"})
            parser.add_subclass_arguments(InferenceModel, "model", required=True, instantiate=False)
            parser.add_argument("--strategy", type=Strategy, default="fsdp")
            parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    return parser
