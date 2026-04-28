import json
import signal
import time
from pathlib import Path
from types import FrameType
from typing import Literal, TypedDict, cast

import torch
from torch import Tensor
from lightning import Fabric

from mirror.callbacks.callback import Callback
from mirror.util import count_params, mirror_data_path


class BenchmarkEntryMetadata(TypedDict):
    num_nodes: int
    devices_per_node: int
    batch_size: int
    param_count: int


class _BenchmarkFailureEntry(BenchmarkEntryMetadata):
    error: str


class BenchmarkSuccessEntry(BenchmarkEntryMetadata):
    status: Literal["success"]
    duration_seconds: float
    is_estimated: bool


class BenchmarkOomEntry(_BenchmarkFailureEntry):
    status: Literal["oom"]


class BenchmarkErrorEntry(_BenchmarkFailureEntry):
    status: Literal["error"]


BenchmarkLogEntry = BenchmarkSuccessEntry | BenchmarkOomEntry | BenchmarkErrorEntry


def benchmark_run_key(num_nodes: int, devices_per_node: int, batch_size: int, param_count: int) -> str:
    return f"{num_nodes}n_{devices_per_node}d_bs{batch_size}_p{param_count}"


def benchmark_lock_path(lock_dir: Path, num_nodes: int, devices_per_node: int, batch_size: int, param_count: int) -> Path:
    return lock_dir / f"{benchmark_run_key(num_nodes, devices_per_node, batch_size, param_count)}.lock"


class TimerCallback[RawT, ProcessedT, BatchT, ModelOutputT](
    Callback[RawT, ProcessedT, BatchT, ModelOutputT]
):
    def __init__(
        self,
        log_file: Path = mirror_data_path / "benchmark_log.jsonl",
        lock_dir: Path = mirror_data_path / "timer_locks",
    ) -> None:
        super().__init__(is_singleton=False)
        self._log_file = log_file
        self._lock_dir = lock_dir
        self._run_metadata: BenchmarkEntryMetadata | None = None
        self._start_time: float | None = None
        self._steps_done: int = 0
        self._total_steps: int = 0

    def on_fit_start(
        self,
        *,
        fabric: Fabric,
        model,
        batch_size: int,
        num_nodes: int,
        n_batches: int,
        epochs: int,
        start_epoch: int,
        start_batch: int,
        **kwargs,
    ) -> None:
        # All ranks must participate in the reduce before branching on is_global_zero.
        reduced = cast(Tensor, fabric.all_reduce(torch.tensor(count_params(model)), reduce_op="sum"))
        param_count = int(reduced.item())

        if not fabric.is_global_zero:
            return

        devices_per_node = fabric.world_size // num_nodes

        self._run_metadata = BenchmarkEntryMetadata(
            num_nodes=num_nodes,
            devices_per_node=devices_per_node,
            batch_size=batch_size,
            param_count=param_count,
        )
        self._total_steps = (epochs - start_epoch) * n_batches - start_batch

        print(f"[TimerCallback] starting run: {dict(self._run_metadata)}")

        lock_path = benchmark_lock_path(
            self._lock_dir, num_nodes, devices_per_node, batch_size, param_count
        )
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path.touch()
        self._start_time = time.perf_counter()

        signal.signal(signal.SIGHUP, self._make_timeout_handler())

    def on_train_batch_end(self, *, fabric: Fabric, **kwargs) -> None:
        if not fabric.is_global_zero:
            return
        self._steps_done += 1

    def on_fit_end(self, *, fabric: Fabric, **kwargs) -> None:
        if not fabric.is_global_zero:
            return
        assert self._start_time is not None and self._run_metadata is not None
        duration = time.perf_counter() - self._start_time
        entry = BenchmarkSuccessEntry(
            **self._run_metadata,
            status="success",
            duration_seconds=duration,
            is_estimated=False,
        )
        self._write_log(entry)
        self._release_lock()

    def on_training_error(self, *, fabric: Fabric, error: Exception, **kwargs) -> None:
        if not fabric.is_global_zero:
            return
        if self._run_metadata is None:
            return
        failure: BenchmarkOomEntry | BenchmarkErrorEntry
        if isinstance(error, torch.cuda.OutOfMemoryError):
            failure = BenchmarkOomEntry(**self._run_metadata, status="oom", error=str(error))
        else:
            failure = BenchmarkErrorEntry(**self._run_metadata, status="error", error=str(error))
        self._write_log(failure)
        self._release_lock()

    def _make_timeout_handler(self):
        def handler(_signum: int, _frame: FrameType | None) -> None:
            if self._run_metadata is None or self._start_time is None or self._steps_done == 0:
                return
            elapsed = time.perf_counter() - self._start_time
            estimated = elapsed * (self._total_steps / self._steps_done)
            entry = BenchmarkSuccessEntry(
                **self._run_metadata,
                status="success",
                duration_seconds=estimated,
                is_estimated=True,
            )
            self._write_log(entry)
            self._release_lock()
        return handler

    def _write_log(self, entry: BenchmarkLogEntry) -> None:
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _release_lock(self) -> None:
        assert self._run_metadata is not None
        lock_path = benchmark_lock_path(
            self._lock_dir,
            self._run_metadata["num_nodes"],
            self._run_metadata["devices_per_node"],
            self._run_metadata["batch_size"],
            self._run_metadata["param_count"],
        )
        if lock_path.exists():
            lock_path.unlink()
