#!/usr/bin/env python3
"""Independent-process CPU probe for experiment-seed reproducibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
HRM_ROOT = ROOT / "HRM"
DATA_ROOT = HRM_ROOT / "data" / "conceptarc-mini"
if str(HRM_ROOT) not in sys.path:
    sys.path.insert(0, str(HRM_ROOT))

from pretrain import (  # noqa: E402
    PretrainConfig,
    create_dataloader,
    init_train_state,
    train_batch,
    validate_training_config,
)
from puzzle_dataset import _sample_batch  # noqa: E402
from utils.seeding import seed_everything  # noqa: E402


def resolved_config(seed: int) -> PretrainConfig:
    overrides = [
        "device=cpu",
        f"seed={seed}",
        "max_steps=8",
        "enable_wandb=false",
        f"data_path={DATA_ROOT}",
        "global_batch_size=4",
        "epochs=1",
        "eval_interval=null",
        "+final_eval=false",
        "lr_warmup_steps=0",
        "arch.fixed_refinement_steps=8",
        "arch.halt_max_steps=4",
        "arch.hidden_size=8",
        "arch.subgoal_head.hidden_size=8",
        "arch.subgoal_head.goal_dim=8",
        "arch.H_layers=1",
        "arch.L_layers=1",
        "arch.H_cycles=1",
        "arch.L_cycles=1",
        "arch.num_heads=2",
        "arch.expansion=2",
        "arch.puzzle_emb_ndim=8",
        "+arch.forward_dtype=float32",
        "arch.loss.feudal_loss_weight=0.05",
        "arch.subgoal_head.manager_period=3",
        "arch.subgoal_head.initial_goal_only=false",
        "arch.subgoal_head.directional_displacement=true",
    ]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(HRM_ROOT / "config"), version_base=None):
        hydra_config = compose(config_name="cfg_pretrain", overrides=overrides)
    resolved = OmegaConf.to_container(hydra_config, resolve=True)
    GlobalHydra.instance().clear()
    assert isinstance(resolved, dict)
    config = PretrainConfig(**resolved)
    config.project_name = "seed_reproducibility_probe"
    config.run_name = f"seed_{seed}"
    config.checkpoint_path = None
    return config


def sample_trace(seed: int, count: int) -> dict[str, list[int]]:
    puzzle_indices = np.load(DATA_ROOT / "train" / "all__puzzle_indices.npy")
    group_indices = np.load(DATA_ROOT / "train" / "all__group_indices.npy")
    puzzle_identifiers = np.load(DATA_ROOT / "train" / "all__puzzle_identifiers.npy")

    # PuzzleDataset increments _iters before constructing this exact generator.
    rng = np.random.Generator(np.random.Philox(seed=seed + 1))
    group_order = rng.permutation(group_indices.size - 1)
    start = 0
    examples: list[int] = []
    puzzle_ids: list[int] = []
    identifiers: list[int] = []
    while len(examples) < count:
        start, batch_examples, batch_puzzles = _sample_batch(
            rng,
            group_order=group_order,
            puzzle_indices=puzzle_indices,
            group_indices=group_indices,
            start_index=start,
            global_batch_size=4,
        )
        examples.extend(int(value) for value in batch_examples)
        puzzle_ids.extend(int(value) for value in batch_puzzles)
        identifiers.extend(int(value) for value in puzzle_identifiers[batch_puzzles])
    return {
        "example_indices": examples[:count],
        "puzzle_indices": puzzle_ids[:count],
        "puzzle_identifiers": identifiers[:count],
    }


def model_checksum(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_worker(seed: int, output: Path, trace_count: int, loss_steps: int) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    config = resolved_config(seed)
    effective_seed = seed_everything(config.seed, rank=0)
    train_loader, metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )
    config = validate_training_config(config, metadata, rank=0)
    state = init_train_state(config, metadata, world_size=1)
    checksum = model_checksum(state.model)

    losses: list[float] = []
    state.model.train()
    for _, batch, global_batch_size in train_loader:
        metrics = train_batch(
            config,
            state,
            batch,
            global_batch_size,
            rank=0,
            world_size=1,
        )
        assert metrics is not None
        losses.append(float(metrics["train/lm_loss"]))
        if len(losses) == loss_steps:
            break

    payload = {
        "seed": seed,
        "effective_seed": effective_seed,
        "sample_trace": sample_trace(seed, trace_count),
        "initial_model_sha256": checksum,
        "early_train_lm_losses": losses,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def invoke_worker(seed: int, output: Path, trace_count: int, loss_steps: int) -> None:
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--seed",
            str(seed),
            "--output",
            str(output),
            "--trace-count",
            str(trace_count),
            "--loss-steps",
            str(loss_steps),
        ],
        cwd=ROOT,
        check=True,
    )


def losses_match(left: list[float], right: list[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=0.0) for a, b in zip(left, right)
    )


def run_probe(
    seed: int, different_seed: int, trace_count: int, loss_steps: int
) -> None:
    with tempfile.TemporaryDirectory(prefix="hrm_seed_probe_") as temp_dir:
        temp = Path(temp_dir)
        paths = [temp / "same_a.json", temp / "same_b.json", temp / "different.json"]
        invoke_worker(seed, paths[0], trace_count, loss_steps)
        invoke_worker(seed, paths[1], trace_count, loss_steps)
        invoke_worker(different_seed, paths[2], trace_count, loss_steps)
        same_a, same_b, different = (
            json.loads(path.read_text(encoding="utf-8")) for path in paths
        )

    same_trace = same_a["sample_trace"] == same_b["sample_trace"]
    same_model = same_a["initial_model_sha256"] == same_b["initial_model_sha256"]
    same_losses = losses_match(
        same_a["early_train_lm_losses"], same_b["early_train_lm_losses"]
    )
    different_trace = same_a["sample_trace"] != different["sample_trace"]
    different_model = (
        same_a["initial_model_sha256"] != different["initial_model_sha256"]
    )

    assert same_trace, "same-seed sample traces differ"
    assert same_model, "same-seed initial model checksums differ"
    assert same_losses, "same-seed early losses differ"
    assert (
        different_trace or different_model
    ), "different seed changed neither samples nor model"

    report: dict[str, Any] = {
        "same_seed": {
            "seed": seed,
            "sample_trace_equal": same_trace,
            "model_checksum_equal": same_model,
            "early_losses_equal": same_losses,
            "run_a": same_a,
            "run_b": same_b,
        },
        "different_seed": {
            "seed": different_seed,
            "sample_trace_changed": different_trace,
            "model_checksum_changed": different_model,
            "run": different,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--different-seed", type=int, default=1)
    parser.add_argument("--trace-count", type=int, default=16)
    parser.add_argument("--loss-steps", type=int, default=4)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if args.output is None:
            parser.error("--worker requires --output")
        run_worker(args.seed, args.output, args.trace_count, args.loss_steps)
    else:
        run_probe(args.seed, args.different_seed, args.trace_count, args.loss_steps)


if __name__ == "__main__":
    main()
