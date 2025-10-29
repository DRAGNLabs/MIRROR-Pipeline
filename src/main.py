from jsonargparse import auto_parser
from typing import Literal

from datasets.placeholder_dataset import PlaceholderDataset
from models.placeholder_model import PlaceholderModel
from trainer import Trainer

Subcommand = Literal['fit'] | Literal['test']


def main(subcommand: Subcommand):
    match subcommand:
        case 'fit':
            fit()
        case _:
            print(f'unimplemented subcommand: {subcommand}')


def fit():
    trainer = Trainer()

    with trainer.fabric.init_module():
        model = PlaceholderModel()

    dataset = PlaceholderDataset()
    trainer.fit(model, dataset)


if __name__ == '__main__':
    parser = auto_parser(main)
    cfg = parser.parse_args()
    if hasattr(cfg, 'config'):
        del cfg.config  # pyright: ignore

    main(**cfg)
