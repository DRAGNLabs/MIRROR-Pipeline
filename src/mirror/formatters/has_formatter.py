from abc import ABC, abstractmethod
from typing import Any, Mapping

from mirror.formatters.mirror_formatter import MirrorFormatter


class HasFormatter[RawT: Mapping[str, Any], FormattedT: Mapping[str, Any], BatchT](ABC):
    @property
    @abstractmethod
    def formatter(self) -> MirrorFormatter[RawT, FormattedT, BatchT]:
        pass
