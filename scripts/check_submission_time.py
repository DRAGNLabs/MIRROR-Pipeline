#!/usr/bin/env python3
"""Time how long it takes to reach SLURM submission, intercepting sbatch.

Protects against regressions like a stray top-level heavy import
re-inflating SLURM submission time (it was ~40s before #298).
"""
import argparse
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Timing:
    end: float | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(_REPO_ROOT / "submit_test_config.yaml"))
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="Maximum allowed seconds to reach sbatch")
    return parser.parse_args()


def _patch_login_node() -> None:
    socket.gethostname = lambda: "login01"


def _patch_sbatch(timing: Timing) -> None:
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] == "sbatch":
            timing.end = time.perf_counter()
            return subprocess.CompletedProcess(
                args=cmd, returncode=0,
                stdout="Submitted batch job 12345\n", stderr="",
            )
        return real_run(cmd, *args, **kwargs)

    subprocess.run = fake_run


def _invoke_main(config: str) -> None:
    sys.argv = [str(_REPO_ROOT / "src/main.py"), "fit", "--config", config]
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    import main as main_module
    main_module.main("fit")


def main() -> int:
    args = _parse_args()
    timing = Timing()
    _patch_login_node()
    _patch_sbatch(timing)
    start = time.perf_counter()
    try:
        _invoke_main(args.config)
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(f"main exited with code {exc.code}", file=sys.stderr)
            return int(exc.code) if isinstance(exc.code, int) else 1
    if timing.end is None:
        print("FAIL: did not reach simulated sbatch call", file=sys.stderr)
        return 1
    elapsed = timing.end - start
    if elapsed > args.threshold:
        print(f"FAIL: submission took {elapsed:.3f}s (threshold: {args.threshold:.3f}s)",
              file=sys.stderr)
        return 1
    print(f"PASS: submission took {elapsed:.3f}s (threshold: {args.threshold:.3f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
