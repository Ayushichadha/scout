#!/usr/bin/env python3
"""Run the first compute-matched 400-step adaptive replanning experiment.

This is deliberately eta=0 and preserves the audited 96-step mechanism
configuration.  Execution is opt-in; without ``--execute`` the script performs
the full configuration-diff preflight and prints the resolved command.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
HRM_ROOT = ROOT / "HRM"
OUTPUT_ROOT = ROOT / "experiments" / "meta_agents_adaptive_400step"
REFERENCE_CONFIG = ROOT / (
    "experiments/meta_agents_adaptive_sanity/"
    "20260816T144122Z_adaptive_eta0_seed0_96step_sanity/resolved_config.yaml"
)
SCHEMA_VERSION = "meta_agents_adaptive_400step_eta0_v1"
TRAINING_STEPS = 400


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def build_overrides(run_dir: Path, seed: int = 0) -> list[str]:
    return [
        "device=cpu",
        f"seed={seed}",
        f"max_steps={TRAINING_STEPS}",
        "enable_wandb=false",
        "data_path=data/conceptarc-mini",
        "global_batch_size=4",
        "epochs=1",
        "eval_interval=null",
        "+final_eval=true",
        "lr_warmup_steps=0",
        f"+checkpoint_path={run_dir / 'checkpoints'}",
        f"run_summary_path={run_dir / 'runtime_summary.json'}",
        "+project_name=meta_agents_adaptive_400step",
        f"+run_name=adaptive_eta0_seed{seed}_400step",
        "arch.fixed_refinement_steps=8",
        "arch.halt_max_steps=4",
        "arch.hidden_size=32",
        "arch.subgoal_head.hidden_size=32",
        "arch.subgoal_head.goal_dim=32",
        "arch.H_layers=1",
        "arch.L_layers=1",
        "arch.H_cycles=1",
        "arch.L_cycles=1",
        "arch.num_heads=2",
        "arch.expansion=2",
        "arch.puzzle_emb_ndim=32",
        "arch.loss.feudal_loss_weight=0.05",
        "arch.loss.intervention_weight=0.0",
        "arch.subgoal_head.replan_mode=adaptive",
        "arch.subgoal_head.directional_displacement=true",
        "arch.subgoal_head.detach_goals=true",
        "arch.subgoal_head.initial_goal_only=false",
        "arch.subgoal_head.trigger_stochastic_train=true",
        "arch.subgoal_head.trigger_threshold=0.5",
    ]


def resolve_config(overrides: list[str]) -> dict[str, Any]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(HRM_ROOT / "config"), version_base=None):
        config = compose(config_name="cfg_pretrain", overrides=overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    GlobalHydra.instance().clear()
    if not isinstance(resolved, dict):
        raise TypeError("resolved Hydra config is not a mapping")
    return resolved


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, path))
        return result
    return {prefix: value}


def preflight_against_96step(
    candidate: dict[str, Any], seed: int = 0
) -> dict[str, Any]:
    reference = yaml.safe_load(REFERENCE_CONFIG.read_text())
    reference_flat = flatten(reference)
    candidate_flat = flatten(candidate)
    keys = sorted(set(reference_flat) | set(candidate_flat))
    differences = {
        key: {
            "reference": reference_flat.get(key, "<missing>"),
            "candidate": candidate_flat.get(key, "<missing>"),
        }
        for key in keys
        if reference_flat.get(key, "<missing>") != candidate_flat.get(key, "<missing>")
    }
    allowed = {
        "max_steps",
        "checkpoint_path",
        "run_summary_path",
        "project_name",
        "run_name",
    }
    if seed != 0:
        allowed.add("seed")
    unexpected = sorted(set(differences) - allowed)
    if unexpected:
        raise RuntimeError(
            "400-step candidate differs from audited 96-step config in "
            f"unexpected fields: {unexpected}"
        )
    required = {
        "max_steps": TRAINING_STEPS,
        "device": "cpu",
        "seed": seed,
        "global_batch_size": 4,
        "arch.fixed_refinement_steps": 8,
        "arch.loss.intervention_weight": 0.0,
        "arch.subgoal_head.replan_mode": "adaptive",
        "arch.subgoal_head.trigger_threshold": 0.5,
        "arch.subgoal_head.trigger_stochastic_train": True,
    }
    for key, expected in required.items():
        actual = candidate_flat.get(key)
        if actual != expected:
            raise RuntimeError(f"preflight {key}={actual!r}, expected {expected!r}")
    return differences


def verify_checkpoint(checkpoint: Path) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"final checkpoint missing: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("checkpoint is not a non-empty state dict")
    nonfinite = []
    for name, tensor in state.items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            nonfinite.append(name)
    if nonfinite:
        raise FloatingPointError(f"non-finite checkpoint tensors: {nonfinite}")
    trigger_keys = sorted(key for key in state if "adaptive_trigger.linear" in key)
    if len(trigger_keys) != 2:
        raise RuntimeError(f"expected two adaptive-trigger tensors, got {trigger_keys}")
    return {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "tensor_count": len(state),
        "all_finite": True,
        "adaptive_trigger_keys": trigger_keys,
    }


def verify_runtime_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("training_steps") != TRAINING_STEPS:
        raise RuntimeError("runtime summary does not report 400 optimizer steps")
    aggregates = payload.get("aggregates", {})
    checks = {
        "mean_executed_refinement_passes_per_episode": 8.0,
        "fixed_compute_violations": 0.0,
        "unconsumed_terminal_emissions": 0.0,
    }
    for key, expected in checks.items():
        actual = aggregates.get(key)
        if actual is None or not math.isclose(float(actual), expected, abs_tol=1e-9):
            raise RuntimeError(f"runtime invariant {key}={actual}, expected {expected}")
    return payload


def run(
    run_dir: Path,
    resolved: dict[str, Any],
    overrides: list[str],
    *,
    seed: int = 0,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_path = run_dir / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    differences = preflight_against_96step(resolved, seed)
    command = [sys.executable, str(HRM_ROOT / "pretrain.py"), *overrides]
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "training-budget-matched adaptive replication",
        "seed": seed,
        "training_budget": TRAINING_STEPS,
        "intervention_weight_eta": 0.0,
        "threshold_role": "0.5 is not calibrated; matched-budget calibration follows training",
        "reference_96step_config": str(REFERENCE_CONFIG),
        "reference_96step_config_sha256": sha256_file(REFERENCE_CONFIG),
        "allowed_config_differences": differences,
        "resolved_config_sha256": sha256_file(resolved_path),
        "command": command,
        "git": git_provenance(),
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    log_path = run_dir / "training.log"
    print("launch:", " ".join(command), flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=HRM_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code != 0:
        failure = {
            "status": "failed",
            "return_code": return_code,
            "log": str(log_path),
        }
        (run_dir / "run_failure.json").write_text(json.dumps(failure, indent=2))
        raise RuntimeError(f"adaptive training exited with code {return_code}")

    runtime_path = run_dir / "runtime_summary.json"
    runtime = verify_runtime_summary(runtime_path)
    checkpoint = run_dir / "checkpoints" / f"step_{TRAINING_STEPS}"
    checkpoint_result = verify_checkpoint(checkpoint)
    hashes = {
        "checkpoint": checkpoint_result,
        "resolved_config_sha256": sha256_file(resolved_path),
        "runtime_summary_sha256": sha256_file(runtime_path),
        "source_files": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                HRM_ROOT / "pretrain.py",
                HRM_ROOT / "models/hrm/hrm_act_v1.py",
                HRM_ROOT / "models/losses.py",
                HRM_ROOT / "models/subgoal_head.py",
            )
        },
    }
    (run_dir / "hashes.json").write_text(json.dumps(hashes, indent=2))
    result = {
        "status": "completed",
        "checkpoint": checkpoint_result,
        "training_steps": runtime["training_steps"],
        "aggregates": runtime["aggregates"],
        "validation_metrics": runtime.get("validation_metrics"),
        "ready_for_matched_budget_calibration": True,
    }
    (run_dir / "run_result.json").write_text(json.dumps(result, indent=2))
    print(f"completed: {run_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir or OUTPUT_ROOT / (
        f"{timestamp}_adaptive_eta0_seed{args.seed}_400step"
    )
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    overrides = build_overrides(run_dir, args.seed)
    resolved = resolve_config(overrides)
    differences = preflight_against_96step(resolved, args.seed)
    if not args.execute:
        print("DRY RUN — no training")
        print("result_directory:", run_dir)
        print("allowed_config_differences:", json.dumps(differences, indent=2))
        print(
            "command:",
            " ".join([sys.executable, str(HRM_ROOT / "pretrain.py"), *overrides]),
        )
        return
    run(run_dir, resolved, overrides, seed=args.seed)


if __name__ == "__main__":
    main()
