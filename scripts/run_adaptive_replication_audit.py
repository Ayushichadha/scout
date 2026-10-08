#!/usr/bin/env python3
"""Audit one 400-step adaptive replication at matched K=2 on CPU.

This runner is intentionally adaptive-only. It freezes one trained checkpoint,
calibrates theta from intervention count on the canonical calibration split,
evaluates task metrics once on the canonical final split, measures clock
likeness, and evaluates the preregistered forced schedules [1,2], [1,4], and
[1,6] without retraining.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
HRM_ROOT = ROOT / "HRM"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HRM_ROOT))

from scripts.eval_checkpoints_local import (  # noqa: E402
    build_model,
    dataset_path_from_config,
    load_config,
    load_weights,
)
from scripts.run_adaptive_matched_budget import (  # noqa: E402
    BudgetSummary,
    calibrate_threshold,
    deterministic_split,
    evaluate_budget_only,
    evaluate_final,
    load_held_out_episodes,
    logical_state_sha256,
    sha256_file,
    split_provenance,
    summarize_schedules,
    write_json,
)
from scripts.run_adaptive_step400_audit import (  # noqa: E402
    FEATURE_NAMES,
    build_reference_audit,
    outward_search_bounds,
    trigger_parameters,
    write_csv,
)
from scripts.run_frozen_schedule_control import (  # noqa: E402
    evaluate_forced_schedule,
)


CANONICAL_SPLIT = ROOT / (
    "experiments/meta_agents_adaptive_matched_budget/"
    "20260820T110102Z_eta0_step400_k2/split_provenance.json"
)
SCHEMA_VERSION = "meta_agents_adaptive_replication_audit_v1"
FORCED_SECOND_PASSES = (2, 4, 6)


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def validate_config(config: dict[str, Any], seed: int) -> None:
    checks = {
        "seed": (config.get("seed"), seed),
        "max_steps": (config.get("max_steps"), 400),
        "device": (config.get("device"), "cpu"),
        "batch_size": (config.get("global_batch_size"), 4),
        "fixed_passes": (config["arch"].get("fixed_refinement_steps"), 8),
        "eta": (config["arch"]["loss"].get("intervention_weight"), 0.0),
        "mode": (config["arch"]["subgoal_head"].get("replan_mode"), "adaptive"),
    }
    failures = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in checks.items()
        if observed != expected
    }
    if failures:
        raise RuntimeError(f"replication config mismatch: {failures}")


def calibrated_row(
    final_budget: BudgetSummary,
    final_task: dict[str, float],
    schedule: dict[str, object],
) -> dict[str, object]:
    dominant = schedule["most_common_schedules"][0]
    return {
        "condition": "adaptive_calibrated",
        "actual_mean_k": final_budget.mean_total_interventions,
        **final_task,
        "dominant_schedule": json.dumps(dominant["positions"]),
        "dominant_schedule_fraction": dominant["fraction"],
        "schedule_entropy_bits": schedule["schedule_entropy_bits"],
    }


def run(*, seed: int, adaptive_run: Path, output_dir: Path) -> None:
    checkpoint = adaptive_run / "checkpoints/step_400"
    config_path = adaptive_run / "resolved_config.yaml"
    for path in (checkpoint, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw_config = yaml.safe_load(config_path.read_text())
    validate_config(raw_config, seed)
    config = load_config(config_path)
    checkpoint_sha = sha256_file(checkpoint)
    config_sha = sha256_file(config_path)

    metadata, episodes = load_held_out_episodes(dataset_path_from_config(config))
    split = deterministic_split(episodes, seed=20260819, calibration_fraction=0.20)
    split_payload = split_provenance(split)
    canonical = json.loads(CANONICAL_SPLIT.read_text())
    for key in ("calibration_ordered_digest", "final_ordered_digest"):
        if split_payload[key] != canonical[key]:
            raise RuntimeError(f"canonical split mismatch: {key}")

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "split_provenance.json", split_payload)
    model = build_model(config, metadata, batch_size=4, device="cpu")
    load_weights(model, checkpoint, "cpu")
    model.eval()
    state_before = logical_state_sha256(model)
    weights, bias = trigger_parameters(model)

    print(
        f"seed {seed}: reference audit theta=0.5 on {len(episodes)} episodes",
        flush=True,
    )
    reference = evaluate_budget_only(
        model, episodes, metadata, threshold=0.5, batch_size=4, device="cpu"
    )
    feature_stats, clock, decision_rows = build_reference_audit(
        reference, weights, bias
    )
    write_csv(
        output_dir / "decision_level_metrics.csv",
        decision_rows,
        ("episode_id", "eligible_pass", "beta", "hard_intervene", *FEATURE_NAMES),
    )
    write_json(output_dir / "feature_beta_statistics.json", feature_stats)
    write_json(output_dir / "clock_likeness.json", clock)

    print(f"seed {seed}: calibration reference theta=0.5", flush=True)
    calibration_reference = evaluate_budget_only(
        model,
        split.calibration,
        metadata,
        threshold=0.5,
        batch_size=4,
        device="cpu",
    )
    low, high = outward_search_bounds(calibration_reference)
    cache: dict[float, BudgetSummary] = {0.5: calibration_reference}

    def evaluator(threshold: float) -> BudgetSummary:
        if threshold not in cache:
            print(f"seed {seed}: calibration theta={threshold:.6f}", flush=True)
            before = logical_state_sha256(model)
            cache[threshold] = evaluate_budget_only(
                model,
                split.calibration,
                metadata,
                threshold=threshold,
                batch_size=4,
                device="cpu",
            )
            after = logical_state_sha256(model)
            if before != state_before or after != state_before:
                raise RuntimeError("model state changed during calibration")
            print(
                f"seed {seed}: mean_K={cache[threshold].mean_total_interventions:.6f}",
                flush=True,
            )
        return cache[threshold]

    selected_threshold, search_rows = calibrate_threshold(
        evaluator,
        coarse_low=low,
        coarse_high=high,
        steps=(1e-4, 1e-5, 1e-6),
    )
    selected = cache[selected_threshold]
    write_csv(
        output_dir / "calibration_threshold_search.csv",
        (asdict(row) for row in search_rows),
        tuple(asdict(search_rows[0])),
    )
    write_json(
        output_dir / "calibration_summary.json",
        {
            "selected_threshold": selected_threshold,
            "target_mean_total_interventions": 2.0,
            "selected_mean_total_interventions": selected.mean_total_interventions,
            "task_metrics_available_to_selector": False,
            "coarse_interval": [low, high],
            "refinement_steps": [1e-4, 1e-5, 1e-6],
            "schedule": summarize_schedules(selected),
        },
    )

    print(
        f"seed {seed}: final adaptive evaluation theta={selected_threshold:.6f}",
        flush=True,
    )
    final_budget, final_task = evaluate_final(
        model,
        split.final,
        metadata,
        frozen_threshold=selected_threshold,
        batch_size=4,
        device="cpu",
    )
    final_schedule = summarize_schedules(final_budget)
    write_json(output_dir / "adaptive_schedule_summary.json", final_schedule)
    rows = [calibrated_row(final_budget, final_task, final_schedule)]
    diagnostics: dict[str, object] = {}
    for second_pass in FORCED_SECOND_PASSES:
        print(f"seed {seed}: forced schedule [1,{second_pass}]", flush=True)
        row, detail = evaluate_forced_schedule(
            model,
            split.final,
            metadata,
            second_pass=second_pass,
            batch_size=4,
            device="cpu",
        )
        row["dominant_schedule_fraction"] = 1.0
        row["schedule_entropy_bits"] = 0.0
        rows.append(row)
        diagnostics[f"forced_[1,{second_pass}]"] = detail
    write_json(output_dir / "forced_schedule_diagnostics.json", diagnostics)
    fields = tuple(rows[0])
    write_csv(
        output_dir / "schedule_comparison.csv",
        ({field: row[field] for field in fields} for row in rows),
        fields,
    )

    state_after = logical_state_sha256(model)
    if state_after != state_before:
        raise RuntimeError("model state changed during replication audit")
    if sha256_file(checkpoint) != checkpoint_sha:
        raise RuntimeError("checkpoint file changed during replication audit")
    write_json(
        output_dir / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "adaptive_run": str(adaptive_run),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "resolved_config_sha256": config_sha,
            "loaded_state_before": state_before,
            "loaded_state_after": state_after,
            "selected_threshold": selected_threshold,
            "device": "cpu",
            "batch_size": 4,
            "training_steps": 400,
            "fixed_refinement_steps": 8,
            "forced_schedules": [[1, value] for value in FORCED_SECOND_PASSES],
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_status": git_value("status", "--short").splitlines(),
        },
    )

    dominant = final_schedule["most_common_schedules"][0]
    variance = feature_stats["variance_decomposition"]
    time_corr = clock["time_only_vs_full"]["spearman"]
    report = f"""# Adaptive replication audit — seed {seed}

- Checkpoint SHA-256: `{checkpoint_sha}`
- Calibration/final episodes: {len(split.calibration)} / {len(split.final)}
- Selected theta / calibration K: {selected_threshold:.6f} / {selected.mean_total_interventions:.6f}
- Final K: {final_budget.mean_total_interventions:.6f}
- Final token accuracy: {final_task['token_accuracy']:.6f}
- Final LM loss/pass: {final_task['lm_loss_per_executed_pass']:.6f}
- Dominant schedule: {dominant['positions']} ({dominant['fraction']:.2%})
- Schedule entropy: {final_schedule['schedule_entropy_bits']:.6f} bits
- Position-explained beta variance: {variance['percent_explained_by_position']:.3f}%
- Time-only/full-score Spearman: {time_corr:.4f}

Forced-schedule results are in `schedule_comparison.csv`. This report is a
precommitted seed replication; all outcomes must be retained regardless of
direction.
"""
    (output_dir / "ADAPTIVE_REPLICATION_AUDIT.md").write_text(report)
    print(f"seed {seed}: completed {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--adaptive-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    adaptive_run = args.adaptive_run.resolve()
    checkpoint = adaptive_run / "checkpoints/step_400"
    config_path = adaptive_run / "resolved_config.yaml"
    raw_config = yaml.safe_load(config_path.read_text())
    validate_config(raw_config, args.seed)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "seed": args.seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "config": str(config_path),
                    "config_sha256": sha256_file(config_path),
                },
                indent=2,
            )
        )
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or ROOT / (
        "experiments/meta_agents_adaptive_replications/"
        f"{timestamp}_seed{args.seed}_step400_k2"
    )
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    run(seed=args.seed, adaptive_run=adaptive_run, output_dir=output_dir)


if __name__ == "__main__":
    main()
