#!/usr/bin/env python3
"""Reproducible fixed-P, fixed-refinement baseline runner.

Execution is opt-in. ``single`` and ``matrix`` print their commands unless
``--execute`` is supplied, preventing accidental launch of the full matrix.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import torch


ROOT = Path(__file__).resolve().parents[1]
HRM_ROOT = ROOT / "HRM"
OUTPUT_ROOT = ROOT / "experiments" / "meta_agents_fixed_p"
SCHEMA_VERSION = "meta_agents_fixed_p_corrected_directional_v1"


def parse_period(value: str) -> tuple[int, bool, str]:
    normalized = value.strip().lower()
    if normalized in {"inf", "infinity", "∞"}:
        return 1, True, "inf"
    period = int(normalized)
    if period < 1:
        raise argparse.ArgumentTypeError("manager period must be >= 1 or 'inf'")
    return period, False, str(period)


def schedule(manager_period: str, fixed_steps: int) -> dict[str, Any]:
    period, initial_only, label = parse_period(manager_period)
    emissions = [1]
    if not initial_only:
        emissions.extend(step for step in range(2, fixed_steps) if step % period == 0)
    durations = [
        (emissions[index + 1] if index + 1 < len(emissions) else fixed_steps) - emission
        for index, emission in enumerate(emissions)
    ]
    diagnostics = {
        "manager_period": label,
        "emission_positions": emissions,
        "commitment_durations_all_emissions": durations,
        "consumed_commitment_durations": [
            duration for duration in durations if duration
        ],
        "unconsumed_terminal_emissions": sum(duration == 0 for duration in durations),
        "interventions_per_episode": len(emissions),
    }
    if diagnostics["unconsumed_terminal_emissions"] != 0:
        raise AssertionError("fixed-compute schedules must not emit terminal goals")
    return diagnostics


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    status = run("status", "--short").splitlines()
    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(status),
        "dirty_paths": status,
    }


def build_overrides(
    args: argparse.Namespace, period_value: str, seed: int, summary: Path
):
    period, initial_only, _ = parse_period(period_value)
    return [
        f"device={args.device}",
        f"seed={seed}",
        f"max_steps={args.training_steps}",
        "enable_wandb=false",
        f"data_path={args.dataset}",
        f"global_batch_size={args.batch_size}",
        f"epochs={args.epochs}",
        "eval_interval=null",
        f"+final_eval={str(args.final_eval).lower()}",
        "lr_warmup_steps=0",
        f"run_summary_path={summary}",
        "+project_name=meta_agents_fixed_p",
        f"+run_name={args.experiment_name}_p{period_value}_seed{seed}",
        f"arch.fixed_refinement_steps={args.fixed_refinement_steps}",
        f"arch.halt_max_steps={args.halt_max_steps}",
        f"arch.hidden_size={args.hidden_size}",
        f"arch.subgoal_head.hidden_size={args.hidden_size}",
        f"arch.subgoal_head.goal_dim={args.hidden_size}",
        f"arch.H_layers={args.h_layers}",
        f"arch.L_layers={args.l_layers}",
        f"arch.H_cycles={args.h_cycles}",
        f"arch.L_cycles={args.l_cycles}",
        f"arch.num_heads={args.num_heads}",
        f"arch.expansion={args.expansion}",
        f"arch.puzzle_emb_ndim={args.hidden_size}",
        f"arch.loss.feudal_loss_weight={args.feudal_loss_weight}",
        f"arch.subgoal_head.manager_period={period}",
        f"arch.subgoal_head.initial_goal_only={str(initial_only).lower()}",
        "arch.subgoal_head.directional_displacement=true",
    ]


def resolved_config(overrides: list[str]) -> dict[str, Any]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(HRM_ROOT / "config"), version_base=None):
        config = compose(config_name="cfg_pretrain", overrides=overrides)
    result = OmegaConf.to_container(config, resolve=True)
    GlobalHydra.instance().clear()
    assert isinstance(result, dict)
    return result


def effective_architecture(config: dict[str, Any], batch_size: int) -> dict[str, Any]:
    arch = config["arch"]
    subgoal = arch["subgoal_head"]
    return {
        "H_layers": arch["H_layers"],
        "L_layers": arch["L_layers"],
        "H_cycles": arch["H_cycles"],
        "L_cycles": arch["L_cycles"],
        "hidden_size": arch["hidden_size"],
        "halt_max_steps": arch["halt_max_steps"],
        "fixed_refinement_steps": arch["fixed_refinement_steps"],
        "batch_size": batch_size,
        "goal_dim": subgoal["goal_dim"],
        "directional_displacement": subgoal["directional_displacement"],
        "V_L": {"present": True, "bias": False},
        "feudal_loss_weight": arch["loss"]["feudal_loss_weight"],
        "manager_period": (
            "inf" if subgoal["initial_goal_only"] else subgoal["manager_period"]
        ),
        "initial_goal_only": subgoal["initial_goal_only"],
    }


def run_one(args: argparse.Namespace, period_value: str, seed: int) -> dict[str, Any]:
    if args.training_steps % args.fixed_refinement_steps:
        raise ValueError(
            "--training-steps must be a multiple of --fixed-refinement-steps so "
            "the run ends on complete episode boundaries"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _, _, label = parse_period(period_value)
    run_id = f"{timestamp}_{args.experiment_name}_p{label}_seed{seed}"
    run_dir = OUTPUT_ROOT / "runs" / run_id
    runtime_summary = run_dir / "runtime_summary.json"
    overrides = build_overrides(args, period_value, seed, runtime_summary)
    config = resolved_config(overrides)
    command = [sys.executable, str(HRM_ROOT / "pretrain.py"), *overrides]

    base_result = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "experiment_name": args.experiment_name,
            "run_id": run_id,
            "timestamp_utc": timestamp,
            "seed": seed,
            "dataset": args.dataset,
            "git": git_provenance(),
        },
        "condition": {
            "manager_period": label,
            "fixed_refinement_steps": args.fixed_refinement_steps,
            "compute_matched": True,
        },
        "effective_architecture": effective_architecture(config, args.batch_size),
        "resolved_configuration": config,
        "manager_diagnostics": schedule(label, args.fixed_refinement_steps),
        "command": command,
    }

    if not args.execute:
        print("DRY RUN:", " ".join(command))
        return {**base_result, "status": "dry_run"}

    run_dir.mkdir(parents=True, exist_ok=False)
    process = subprocess.run(
        command, cwd=HRM_ROOT, capture_output=True, text=True, check=False
    )
    (run_dir / "stdout.log").write_text(process.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
    runtime = (
        json.loads(runtime_summary.read_text(encoding="utf-8"))
        if runtime_summary.exists()
        else None
    )
    result = {
        **base_result,
        "status": "completed" if process.returncode == 0 else "failed",
        "return_code": process.returncode,
        "metrics": runtime,
    }

    if process.returncode == 0:
        aggregates = runtime["aggregates"]
        observed = aggregates["mean_executed_refinement_passes_per_episode"]
        if observed is None or abs(observed - args.fixed_refinement_steps) > 1e-9:
            raise AssertionError(
                f"fixed-compute invariant failed: observed {observed}, "
                f"expected {args.fixed_refinement_steps}"
            )
        if aggregates.get("fixed_compute_violations", 0.0) != 0.0:
            raise AssertionError("runtime reported a fixed-compute violation")
        if aggregates.get("unconsumed_terminal_emissions", 0.0) != 0.0:
            raise AssertionError("runtime reported an unconsumed terminal emission")

    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{result['status']}: {result_path}")
    return result


def smoke(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(HRM_ROOT))
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1

    fixed_steps = args.fixed_refinement_steps
    for period_value in ("1", "3", "4", "6", "inf"):
        period, initial_only, label = parse_period(period_value)
        config = {
            "batch_size": 1,
            "seq_len": 4,
            "vocab_size": 16,
            "num_puzzle_identifiers": 2,
            "H_cycles": 1,
            "L_cycles": 1,
            "H_layers": 1,
            "L_layers": 1,
            "hidden_size": 8,
            "expansion": 2,
            "num_heads": 2,
            "pos_encodings": "rope",
            "halt_max_steps": 100,
            "fixed_refinement_steps": fixed_steps,
            "halt_exploration_prob": 1.0,
            "forward_dtype": "float32",
            "subgoal_head": {
                "hidden_size": 8,
                "goal_dim": 8,
                "manager_period": period,
                "initial_goal_only": initial_only,
                "directional_displacement": True,
            },
        }
        torch.manual_seed(11)
        model = HierarchicalReasoningModel_ACTV1(config).train()
        with torch.no_grad():
            model.inner.q_head.bias.copy_(torch.tensor([100.0, -100.0]))
        batch = {
            "inputs": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[2, 3, 4, 5]]),
            "puzzle_identifiers": torch.zeros(1, dtype=torch.long),
        }
        carry = model.initial_carry(batch)
        previous_active = None
        observed_emissions = []
        active_consumption_passes = []
        print(f"P={label}")
        with torch.no_grad():
            for pass_index in range(1, fixed_steps + 1):
                carry, outputs = model(carry, batch)
                active = outputs["subgoal_active"].bool().item()
                cosine = None
                if active and previous_active is not None:
                    cosine = torch.nn.functional.cosine_similarity(
                        outputs["subgoal_goal"], previous_active, dim=-1
                    ).item()
                if active:
                    previous_active = outputs["subgoal_goal"].clone()
                    active_consumption_passes.append(pass_index)
                emitted = outputs["subgoal_updated"].bool().item()
                if emitted:
                    observed_emissions.append(pass_index)
                print(
                    f"  pass={pass_index} emitted={emitted} active={active} "
                    f"cos_prev={cosine} halted={carry.halted.item()}"
                )
        expected = schedule(label, fixed_steps)
        assert observed_emissions == expected["emission_positions"]
        assert carry.steps.item() == fixed_steps and carry.halted.item()
        assert expected["unconsumed_terminal_emissions"] == 0
        print(
            f"  interventions={len(observed_emissions)} "
            f"active_passes={active_consumption_passes} "
            f"dwell={expected['consumed_commitment_durations']} "
            f"executed={carry.steps.item()}"
        )


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fixed-refinement-steps", type=int, required=True)
    parser.add_argument("--training-steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dataset", default="data/conceptarc-mini")
    parser.add_argument("--experiment-name", default="corrected_fixed_p")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--h-layers", type=int, default=1)
    parser.add_argument("--l-layers", type=int, default=1)
    parser.add_argument("--h-cycles", type=int, default=1)
    parser.add_argument("--l-cycles", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--halt-max-steps", type=int, default=4)
    parser.add_argument("--feudal-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--final-eval",
        action="store_true",
        help="Run held-out evaluation after the fixed training budget",
    )
    parser.add_argument("--execute", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    single = subparsers.add_parser("single")
    add_common(single)
    single.add_argument("--manager-period", required=True)
    single.add_argument("--seed", type=int, default=0)

    matrix = subparsers.add_parser("matrix")
    add_common(matrix)
    matrix.add_argument(
        "--manager-periods", nargs="+", default=["1", "3", "4", "6", "inf"]
    )
    matrix.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--fixed-refinement-steps", type=int, default=8)
    args = parser.parse_args()

    if args.mode == "smoke":
        smoke(args)
    elif args.mode == "single":
        run_one(args, args.manager_period, args.seed)
    else:
        results = [
            run_one(args, period, seed)
            for period in args.manager_periods
            for seed in args.seeds
        ]
        if not args.execute:
            print(
                f"Dry-run matrix contains {len(results)} runs; add --execute to launch."
            )


if __name__ == "__main__":
    main()
