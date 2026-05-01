#!/usr/bin/env python
"""Visualize benchmark results from the JSONL log.

Produces:
  1. A table printed to stdout: rows = (device config, batch size),
     columns = model size, cells = duration or error status.
  2. A line chart saved to --output: training time vs param count,
     one line per device config, coloured by batch size (cividis log scale).
"""
import argparse
import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from mirror.callbacks.timer_callback import BenchmarkLogEntry, BenchmarkSuccessEntry
from mirror.util import mirror_data_path


def _load(log_file: Path) -> list[BenchmarkLogEntry]:
    if not log_file.exists():
        raise FileNotFoundError(log_file)
    entries: list[BenchmarkLogEntry] = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(cast(BenchmarkLogEntry, json.loads(line)))
    return entries


def _row_label(num_nodes: int, devices_per_node: int, batch_size: int) -> str:
    return f"{num_nodes}x{devices_per_node} GPUs  bs={batch_size}"


def _cell(entry: BenchmarkLogEntry) -> str:
    match entry["status"]:
        case "success":
            return f"{entry['duration_seconds']:.1f}s"
        case "oom":
            return "OOM"
        case _:
            return "ERR"


def print_table(entries: list[BenchmarkLogEntry]) -> None:
    param_counts = sorted({e["param_count"] for e in entries})
    col_labels = [f"{p:,}" for p in param_counts]

    lookup: dict[tuple[int, int, int, int], BenchmarkLogEntry] = {
        (e["num_nodes"], e["devices_per_node"], e["batch_size"], e["param_count"]): e
        for e in entries
    }

    device_configs = sorted(
        {(e["num_nodes"], e["devices_per_node"]) for e in entries}
    )
    batch_sizes = sorted({e["batch_size"] for e in entries})

    col_w = max(len(c) for c in col_labels) + 2
    row_w = max(len(_row_label(n, d, b)) for n, d in device_configs for b in batch_sizes) + 2

    header = f"{'':>{row_w}}" + "".join(f"{c:>{col_w}}" for c in col_labels)
    print(header)
    print("-" * len(header))

    for num_nodes, devices_per_node in device_configs:
        for batch_size in batch_sizes:
            row_label = _row_label(num_nodes, devices_per_node, batch_size)
            cells = []
            for param_count in param_counts:
                entry = lookup.get((num_nodes, devices_per_node, batch_size, param_count))
                cells.append(_cell(entry) if entry else "-")
            print(f"{row_label:>{row_w}}" + "".join(f"{c:>{col_w}}" for c in cells))


def plot_chart(entries: list[BenchmarkLogEntry], output: Path, log_y: bool) -> None:
    successes: list[BenchmarkSuccessEntry] = [e for e in entries if e["status"] == "success"]
    if not successes:
        print("No successful entries to plot.")
        return

    device_configs = sorted({(e["num_nodes"], e["devices_per_node"]) for e in successes})
    batch_sizes = sorted({e["batch_size"] for e in successes})

    norm = mcolors.LogNorm(vmin=min(batch_sizes), vmax=max(batch_sizes))
    cmap = plt.colormaps["cividis"]
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    device_marker = {cfg: markers[i % len(markers)] for i, cfg in enumerate(device_configs)}

    fig, ax = plt.subplots(figsize=(10, 6))

    for num_nodes, devices_per_node in device_configs:
        for batch_size in batch_sizes:
            subset = sorted(
                [e for e in successes
                 if e["num_nodes"] == num_nodes
                 and e["devices_per_node"] == devices_per_node
                 and e["batch_size"] == batch_size],
                key=lambda e: e["param_count"],
            )
            if not subset:
                continue
            xs = [e["param_count"] for e in subset]
            ys = [e["duration_seconds"] for e in subset]
            color = cmap(norm(batch_size))
            marker = device_marker[(num_nodes, devices_per_node)]
            label = f"{num_nodes}x{devices_per_node} GPUs  bs={batch_size}"
            ax.plot(xs, ys, marker=marker, color=color, label=label)

    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("Parameter count")
    ax.set_ylabel("Training time (s)")
    ax.set_title("Training time vs model size")
    ax.legend(loc="upper left", fontsize=7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label="Batch size")

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Chart saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=Path, default=mirror_data_path / "benchmark_log.jsonl")
    parser.add_argument("--output",   type=Path, default=mirror_data_path / "benchmark_chart.png")
    parser.add_argument("--log-y", action='store_true')
    args = parser.parse_args()

    entries = _load(args.log_file)
    print_table(entries)
    plot_chart(entries, args.output, args.log_y)


if __name__ == "__main__":
    main()
