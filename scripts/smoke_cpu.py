#!/usr/bin/env python3
"""
CPU-only smoke test runner for HRM.

Runs a short synthetic-data training loop on CPU with small defaults.
Accepts Hydra-style overrides as additional CLI args.

Examples:
  python scripts/smoke_cpu.py
  python scripts/smoke_cpu.py epochs=10 global_batch_size=32

Env knobs:
  OMP_NUM_THREADS (default 4)
  EPOCHS (default 20), BATCH_SIZE (default 64), EVAL_INTERVAL (default 20)
  ENABLE_WANDB (default false)
"""
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    pretrain = repo_root / "HRM" / "pretrain.py"

    omp = os.environ.get("OMP_NUM_THREADS", "4")
    epochs = os.environ.get("EPOCHS", "20")
    batch_size = os.environ.get("BATCH_SIZE", "64")
    eval_interval = os.environ.get("EVAL_INTERVAL", epochs)
    enable_wandb = os.environ.get("ENABLE_WANDB", "false")

    overrides = [
        "device=cpu",
        "data_path=/nonexistent",  # triggers synthetic data path
        f"epochs={epochs}",
        f"eval_interval={eval_interval}",
        f"global_batch_size={batch_size}",
        f"enable_wandb={enable_wandb}",
    ]

    # Pass through any additional Hydra overrides from CLI
    overrides.extend(sys.argv[1:])

    cmd = [sys.executable, str(pretrain)] + overrides
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = omp

    print("Running:", " ".join(cmd))
    try:
        return subprocess.call(cmd, cwd=str(repo_root), env=env)
    except FileNotFoundError as e:
        print("Python executable not found or script missing:", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
