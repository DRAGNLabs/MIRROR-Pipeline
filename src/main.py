from jsonargparse import auto_parser
from typing import List, Literal

from mirror.callbacks.callback import Callback
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.models.placeholder_model import PlaceholderModel
from mirror.trainer import Trainer

Subcommand = Literal['fit'] | Literal['test']


def main(
        subcommand: Subcommand,
        data: MirrorDataset,
        callbacks: List[Callback] = []
):
    match subcommand:
        case 'fit':
            fit(data, callbacks)
        case _:
            print(f'unimplemented subcommand: {subcommand}')


def fit(data: MirrorDataset, callbacks: List[Callback]):
    trainer = Trainer(callbacks)

    with trainer.fabric.init_module():
        model = PlaceholderModel()

    trainer.fit(model, data)


if __name__ == '__main__':
    parser = auto_parser(main)
    cfg = parser.parse_args()
    if hasattr(cfg, 'config'):
        del cfg.config  # pyright: ignore

    init = parser.instantiate_classes(cfg)

    main(**init)
