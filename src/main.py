from jsonargparse import auto_parser
from typing import List, Literal

from lightning.fabric.strategies.strategy import Strategy
from lightning.fabric.strategies.fsdp import FSDPStrategy

from mirror.callbacks.callback import Callback
from mirror.checkpoint_identifier import CheckpointIdentifier
import mirror.datasets
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models.placeholder_model import PlaceholderModel
from mirror.trainer import Trainer

Subcommand = Literal['fit'] | Literal['test']


def main(
    subcommand: Subcommand,
    data: MirrorDataset,
    strategy: Strategy = FSDPStrategy(),
    callbacks: List[Callback] = [],
    checkpoint: CheckpointIdentifier | None = None,
):
    match subcommand:
        case 'fit':
            fit(data, strategy, callbacks, checkpoint)
        case _:
            print(f'unimplemented subcommand: {subcommand}')


def fit(
    dataset: MirrorDataset,
    strategy: Strategy,
    callbacks: List[Callback],
    checkpoint: CheckpointIdentifier | None
):
    trainer = Trainer(strategy, callbacks)

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
