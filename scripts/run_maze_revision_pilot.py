"""Train a fixed-budget HRM with feudal goals on Maze-Hard, preserving its checkpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from scripts.run_meta_agents_fixed_p import (
    build_overrides,
    git_provenance,
    resolved_config,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "HRM/data/maze-30x30-hard-1k"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maze_group(grid):
    square = grid.reshape(30, 30)
    variants = [np.rot90(x, k).tobytes() for x in (square, square.T) for k in range(4)]
    return hashlib.sha256(min(variants)).hexdigest()


def prepare(directory):
    src = SOURCE / "train"
    x = np.load(src / "all__inputs.npy")
    y = np.load(src / "all__labels.npy")
    groups = [maze_group(row) for row in x]
    # Sorting SHA-256 keys gives a deterministic split independent of task outcomes.
    dev = set(sorted(set(groups))[:64])
    manifest = {}
    for split, use_dev in (("train", False), ("test", True)):
        dest = directory / split
        dest.mkdir(parents=True)
        rows = [i for i, group in enumerate(groups) if (group in dev) == use_dev]
        manifest[split] = rows
        n = len(rows)
        np.save(dest / "all__inputs.npy", x[rows])
        np.save(dest / "all__labels.npy", y[rows])
        np.save(dest / "all__puzzle_identifiers.npy", np.zeros(n, dtype=np.int32))
        for key in ("group_indices", "puzzle_indices"):
            np.save(dest / f"all__{key}.npy", np.arange(n + 1, dtype=np.int32))
        metadata = json.loads((src / "dataset.json").read_text())
        metadata.update(total_groups=n, mean_puzzle_examples=1.0)
        (dest / "dataset.json").write_text(json.dumps(metadata))
    manifest["source_sha256"] = {
        p.name: digest(p) for p in src.glob("*") if p.is_file()
    }
    manifest["dev_copy_input_token_accuracy"] = float(
        (x[manifest["test"]] == y[manifest["test"]]).mean()
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 8:
        parser.error("--steps must be positive and divisible by eight")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = prepare(out / "data")
    config = argparse.Namespace(
        device="cpu",
        training_steps=args.steps,
        dataset=str(out / "data"),
        batch_size=4,
        epochs=1 + (args.steps * 4 // len(manifest["train"])),
        final_eval=True,
        experiment_name="maze_revision_pilot",
        fixed_refinement_steps=8,
        halt_max_steps=4,
        hidden_size=32,
        h_layers=1,
        l_layers=1,
        h_cycles=1,
        l_cycles=1,
        num_heads=2,
        expansion=2,
        feudal_loss_weight=0.05,
    )
    overrides = build_overrides(config, "4", 0, out / "summary.json")
    overrides.append(f"+checkpoint_path={out / 'checkpoints'}")
    cmd = [sys.executable, str(ROOT / "HRM/pretrain.py"), *overrides]
    manifest.update(
        command=cmd, git=git_provenance(), config=resolved_config(overrides)
    )
    record = out / "manifest.json"
    record.write_text(json.dumps(manifest, indent=2))
    start = time.monotonic()
    env = dict(
        os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2"
    )
    with (out / "training.log").open("w") as log:
        subprocess.run(
            cmd,
            env=env,
            cwd=ROOT / "HRM",
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    manifest["runtime_seconds"] = time.monotonic() - start
    manifest["checkpoint_sha256"] = digest(out / f"checkpoints/step_{args.steps}")
    record.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
