#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a training config and assert that loss drops enough to indicate overfitting."
    )
    parser.add_argument("--config", default="overfit_test_config.yaml")
    parser.add_argument("--min-relative-drop", type=float, default=0.5)
    parser.add_argument("--min-absolute-drop", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cmd = [sys.executable, "src/main.py", "fit", "--config", args.config]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    combined_output = proc.stdout + proc.stderr
    print(combined_output, end="")

    if proc.returncode != 0:
        print(f"Training command failed with exit code {proc.returncode}.", file=sys.stderr)
        return proc.returncode

    loss_pattern = r"(?:loss:\s*|Loss=)([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    losses = [float(match) for match in re.findall(loss_pattern, combined_output)]
    if len(losses) < 2:
        print("Could not parse at least two loss values from training output.", file=sys.stderr)
        return 1

    initial_loss = losses[0]
    final_loss = losses[-1]
    absolute_drop = initial_loss - final_loss
    relative_drop = absolute_drop / initial_loss if initial_loss > 0 else 0.0

    print(
        f"Overfit check: initial_loss={initial_loss:.6f}, final_loss={final_loss:.6f}, "
        f"absolute_drop={absolute_drop:.6f}, relative_drop={relative_drop:.3f}"
    )

    if absolute_drop < args.min_absolute_drop or relative_drop < args.min_relative_drop:
        print(
            "Overfit check failed: loss did not drop enough. "
            f"Required absolute_drop >= {args.min_absolute_drop} and relative_drop >= {args.min_relative_drop}.",
            file=sys.stderr,
        )
        return 1

    print("Overfit check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
