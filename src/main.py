from jsonargparse import auto_parser
from typing import Literal

Subcommand = Literal['fit'] | Literal['test']


def main(subcommand: Subcommand):
    print(subcommand)


if __name__ == '__main__':
    parser = auto_parser(main)
    cfg = parser.parse_args()
    if hasattr(cfg, 'config'):
        del cfg.config  # pyright: ignore

    main(**cfg)
