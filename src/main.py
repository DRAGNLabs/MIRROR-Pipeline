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
):
    # These warnings happen internal to Fabric, so there's not much we can do about them.
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*Please use DTensor instead and we are deprecating ShardedTensor.*')
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*`load_state_dict` is deprecated and will be removed in future versions\\. Please use `load` instead.*')
    warnings.filterwarnings('ignore', category=UserWarning, message='.*Please use the new API settings to control TF32 behavior.*')
    warnings.filterwarnings('ignore', category=UserWarning, message='.*`_get_pg_default_device` will be deprecated, it only stays for backward-compatibility reason.*')

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
    checkpoint: CheckpointIdentifier | None
):
    trainer = Trainer(strategy, devices, num_nodes, callbacks)

    trainer.launch()

    with trainer.fabric.init_module():
        model = PlaceholderModel()

    trainer.fit(model, dataset, checkpoint)


if __name__ == '__main__':
    parser = auto_parser(main)
    cfg = parser.parse_args()
    if hasattr(cfg, 'config'):
        del cfg.config  # pyright: ignore

    init = parser.instantiate_classes(cfg)

    main(**init)
