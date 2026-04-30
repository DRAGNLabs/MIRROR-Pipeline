from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from mirror.schedulers.configure_scheduler import ConfigureScheduler


class CosineAnnealingScheduler(ConfigureScheduler):
    """Decays the learning rate following a cosine curve over all training steps."""

    def __call__(self, optimizer: Optimizer, total_training_steps: int) -> LRScheduler:
        return CosineAnnealingLR(optimizer, T_max=total_training_steps)
