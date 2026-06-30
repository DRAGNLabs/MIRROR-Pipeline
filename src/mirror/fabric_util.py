import sys

from lightning import Fabric
from lightning.fabric.connector import _PRECISION_INPUT
from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.fabric.strategies.single_device import SingleDeviceStrategy
from lightning.fabric.strategies.strategy import Strategy


def cpu_safe_strategy(strategy: str | Strategy, device: str) -> str | Strategy:
    """FSDP is GPU-only, so fall back to single-device when running on CPU."""
    if device == "cpu" and (strategy == "fsdp" or isinstance(strategy, FSDPStrategy)):
        print("FSDP is GPU-only; falling back to the single-device CPU strategy.", file=sys.stderr)
        return SingleDeviceStrategy(device="cpu")
    return strategy


def make_fabric(
        strategy: Strategy,
        accelerator: str,
        devices: int = 1,
        num_nodes: int = 1,
        callbacks: list = [],
        precision: _PRECISION_INPUT | None = None,
) -> Fabric:
    return Fabric(
        strategy=strategy,
        devices=devices,
        num_nodes=num_nodes,
        callbacks=callbacks,
        accelerator=accelerator,
        precision=precision,
    )


def rank_zero_log(fabric: Fabric, *args):
    if fabric.is_global_zero:
        # stderr because by default python buffers stdout, which can be confusing for debugging.
        # a lot of libraries seem to use this strategy of printing to stderr instead
        print(*args, file=sys.stderr)
