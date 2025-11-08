from jsonargparse import auto_parser
from typing import List, Literal

from mirror.callbacks.callback import Callback
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.datasets.placeholder_dataset import PlaceholderDataset
import mirror.datasets
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models.placeholder_model import PlaceholderModel
from mirror.trainer import Trainer

Subcommand = Literal['fit'] | Literal['test']


def main(
    subcommand: Subcommand,
    callbacks: List[Callback] = [],
    checkpoint: CheckpointIdentifier | None = None,
):
    match subcommand:
        case 'fit':
            fit(callbacks, checkpoint)
        case _:
            print(f'unimplemented subcommand: {subcommand}')


def fit(callbacks: List[Callback], checkpoint: CheckpointIdentifier | None):
    trainer = Trainer(callbacks)

    with trainer.fabric.init_module():
        model = PlaceholderModel()

    dataset = PlaceholderDataset()
    trainer.fit(model, dataset, checkpoint)


if __name__ == '__main__':
    parser = auto_parser(main)
    cfg = parser.parse_args()
    if hasattr(cfg, 'config'):
        del cfg.config  # pyright: ignore

    init = parser.instantiate_classes(cfg)

    main(**init)
