from abc import ABC, abstractmethod
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class ConfigureScheduler(ABC):
    @abstractmethod
    def __call__(self, optimizer: Optimizer, total_training_steps: int) -> LRScheduler:
        pass
