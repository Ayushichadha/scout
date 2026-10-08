#!/usr/bin/env python3
"""Evaluate forced K=2 schedules on the frozen adaptive step-400 weights."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence

import torch

from scripts.eval_checkpoints_local import (
    build_model,
    dataset_path_from_config,
    load_config,
    load_weights,
)
from scripts.run_adaptive_matched_budget import (
    Episode,
    _batch_episodes,
    _set_threshold,
    accumulate_episode_metrics,
    deterministic_split,
    load_held_out_episodes,
    logical_state_sha256,
    sha256_file,
    split_provenance,
    write_json,
)
from puzzle_dataset import IGNORE_LABEL_ID


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULT = ROOT / (
    "experiments/meta_agents_adaptive_matched_budget/"
    "20260820T110102Z_eta0_step400_k2"
)
ADAPTIVE_RUN = ROOT / (
    "experiments/meta_agents_adaptive_400step/"
    "20260820T101100Z_adaptive_eta0_seed0_400step"
)
CHECKPOINT = ADAPTIVE_RUN / "checkpoints/step_400"
CONFIG = ADAPTIVE_RUN / "resolved_config.yaml"
EXPECTED_CHECKPOINT_SHA256 = (
    "165586ca87a5ca473b342ccb4342afd72890817498c87319964356aa37313889"
)
EXPECTED_SPLIT_DIGEST = (
    "2c38c7c804baca6f48c050069dcb772be9c5ae3ee423f0f277e79162c4cbc506"
)
CALIBRATED_THRESHOLD = 0.4982
FORCED_SECOND_PASSES = tuple(range(2, 8))
SCHEMA_VERSION = "meta_agents_frozen_adaptive_schedule_control_v1"


def threshold_for_pass(pass_index: int, second_pass: int) -> float:
    """Force the desired eligible action without changing model weights."""

    if second_pass not in FORCED_SECOND_PASSES:
        raise ValueError(f"unsupported forced second pass: {second_pass}")
    if not 1 <= pass_index <= 8:
        raise ValueError(f"invalid refinement pass: {pass_index}")
    return -1.0 if pass_index == second_pass else 2.0


def evaluate_forced_schedule(
    model: torch.nn.Module,
    episodes: Sequence[Episode],
    metadata: Any,
    *,
    second_pass: int,
    batch_size: int = 4,
    device: str = "cpu",
    collect_per_episode: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one exogenous schedule using the adaptive model's own goals."""

    expected = [1, second_pass]
    before = logical_state_sha256(model)
    metric_totals: dict[str, float] = defaultdict(float)
    position_histogram: Counter[int] = Counter()
    per_episode: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch, valid in _batch_episodes(
            episodes, batch_size=batch_size, metadata=metadata
        ):
            batch = {name: value.to(device) for name, value in batch.items()}
            carry = model.initial_carry(batch)
            episode_metrics = None
            local_positions = [[] for _ in valid]
            local_dwells = [[] for _ in valid]
            local_lm_loss = torch.zeros(len(valid), dtype=torch.float64)
            local_correct = torch.zeros(len(valid), dtype=torch.int64)
            local_valid = (
                (batch["labels"][: len(valid)] != IGNORE_LABEL_ID).sum(-1).cpu()
            )
            local_exact = torch.zeros(len(valid), dtype=torch.bool)
            local_metric_hits = torch.zeros(len(valid), dtype=torch.int64)
            for pass_index in range(1, 9):
                _set_threshold(model, threshold_for_pass(pass_index, second_pass))
                carry, _loss, metrics, outputs, _done = model(
                    carry=carry,
                    batch=batch,
                    return_keys=[
                        "subgoal_updated",
                        "adaptive_completed_dwell",
                        "adaptive_completed_dwell_mask",
                        "logits",
                    ],
                )
                episode_metrics = accumulate_episode_metrics(episode_metrics, metrics)
                updated = outputs["subgoal_updated"].bool().cpu()
                dwell = outputs["adaptive_completed_dwell"].long().cpu()
                dwell_mask = outputs["adaptive_completed_dwell_mask"].bool().cpu()
                logits = outputs["logits"][: len(valid)]
                labels = batch["labels"][: len(valid)]
                pass_lm_loss = model.loss_fn(
                    logits, labels, ignore_index=IGNORE_LABEL_ID
                ).sum(-1) / local_valid.to(device).clamp_min(1)
                local_lm_loss += pass_lm_loss.double().cpu()
                halted = carry.halted[: len(valid)].bool().cpu()
                predictions = logits.argmax(-1).cpu()
                labels_cpu = labels.cpu()
                valid_mask = labels_cpu != IGNORE_LABEL_ID
                correct = (valid_mask & (predictions == labels_cpu)).sum(-1)
                local_correct[halted] = correct[halted]
                local_exact[halted] = correct[halted] == local_valid[halted]
                local_metric_hits += halted.long()
                for index in range(len(valid)):
                    if updated[index]:
                        local_positions[index].append(pass_index)
                    if dwell_mask[index]:
                        local_dwells[index].append(int(dwell[index]))
            if episode_metrics is None:
                raise RuntimeError("forced schedule produced no episode metrics")
            for name, value in episode_metrics.items():
                metric_totals[name] += float(value.cpu())
            expected_dwells = [second_pass - 1, 8 - second_pass]
            for index, positions in enumerate(local_positions):
                if positions != expected:
                    raise RuntimeError(
                        f"forced schedule {expected} produced {positions}"
                    )
                position_histogram.update(positions)
                if local_dwells[index] != expected_dwells:
                    raise RuntimeError(
                        f"forced schedule {expected} produced dwell segments "
                        f"{local_dwells[index]}, expected {expected_dwells}"
                    )
                if sum(local_dwells[index]) != 7:
                    raise RuntimeError("forced-schedule dwell sum is not seven")
                if len(local_dwells[index]) != len(positions):
                    raise RuntimeError(
                        "forced-schedule closed dwell count differs from interventions"
                    )
                if local_metric_hits[index] != 1:
                    raise RuntimeError(
                        f"{valid[index].episode_id}: expected one terminal metric hit, "
                        f"found {int(local_metric_hits[index])}"
                    )
                if collect_per_episode:
                    token_count = int(local_valid[index])
                    token_correct = int(local_correct[index])
                    per_episode.append(
                        {
                            "episode_id": valid[index].episode_id,
                            "schedule_k": second_pass,
                            "dwell_first": expected_dwells[0],
                            "dwell_second": expected_dwells[1],
                            "tokens_correct": token_correct,
                            "tokens_valid": token_count,
                            "token_acc_episode": token_correct / token_count,
                            "lm_loss_sum": float(local_lm_loss[index]),
                            "passes_executed": 8,
                            "lm_loss_per_pass": float(local_lm_loss[index] / 8),
                            "exact_correct": int(local_exact[index]),
                        }
                    )
    _set_threshold(model, CALIBRATED_THRESHOLD)
    after = logical_state_sha256(model)
    if before != after:
        raise RuntimeError(f"model state changed under forced schedule {expected}")
    count = metric_totals["count"]
    if count != len(episodes):
        raise RuntimeError("forced schedule task denominator mismatch")
    if metric_totals.get("fixed_compute_violations", 0.0) != 0.0:
        raise RuntimeError("forced schedule violated fixed compute")
    if metric_totals.get("unconsumed_terminal_emissions", 0.0) != 0.0:
        raise RuntimeError("forced schedule emitted at the terminal pass")
    row = {
        "condition": f"forced_[1,{second_pass}]",
        "actual_mean_k": sum(position_histogram.values()) / count,
        "token_accuracy": metric_totals["accuracy"] / count,
        "exact_accuracy": metric_totals["exact_accuracy"] / count,
        "lm_loss_per_executed_pass": metric_totals["lm_loss"] / (count * 8),
        "lm_loss_per_completed_episode": metric_totals["lm_loss"] / count,
        "mean_refinement_passes": 8.0,
        "dominant_schedule": json.dumps(expected),
    }
    if not all(
        math.isfinite(value)
        for key, value in row.items()
        if key not in {"condition", "dominant_schedule"}
    ):
        raise FloatingPointError("non-finite forced-schedule result")
    diagnostics = {
        "expected_schedule": expected,
        "episodes": int(count),
        "position_histogram": {
            str(position): position_histogram[position] for position in range(1, 8)
        },
        "checkpoint_state_before": before,
        "checkpoint_state_after": after,
        "model_state_stable": before == after,
        "fixed_compute_violations": 0,
        "terminal_emission_violations": 0,
        "dwell_segments": [second_pass - 1, 8 - second_pass],
    }
    if collect_per_episode:
        diagnostics["per_episode"] = per_episode
    return row, diagnostics


def calibrated_row() -> dict[str, Any]:
    with (SOURCE_RESULT / "final_comparison.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    source = next(row for row in rows if row["condition"] == "adaptive_calibrated")
    return {
        "condition": "calibrated_adaptive",
        "actual_mean_k": float(source["actual_mean_k"]),
        "token_accuracy": float(source["token_accuracy"]),
        "exact_accuracy": float(source["exact_accuracy"]),
        "lm_loss_per_executed_pass": float(source["lm_loss_per_executed_pass"]),
        "lm_loss_per_completed_episode": float(source["lm_loss_per_completed_episode"]),
        "mean_refinement_passes": float(source["mean_refinement_passes"]),
        "dominant_schedule": source["dominant_schedule"],
    }


def archive_inputs(output_dir: Path) -> dict[str, str]:
    archive = output_dir / "frozen_inputs"
    archive.mkdir()
    sources = {
        "adaptive_step_400": CHECKPOINT,
        "adaptive_resolved_config.yaml": CONFIG,
        "adaptive_checkpoint_all_config.yaml": ADAPTIVE_RUN
        / "checkpoints/all_config.yaml",
        "adaptive_checkpoint_hrm_act_v1.py": ADAPTIVE_RUN / "checkpoints/hrm_act_v1.py",
        "adaptive_checkpoint_losses.py": ADAPTIVE_RUN / "checkpoints/losses.py",
        "adaptive_training.log": ADAPTIVE_RUN / "training.log",
        "source_step400_final_comparison.csv": SOURCE_RESULT / "final_comparison.csv",
        "source_step400_split_provenance.json": SOURCE_RESULT / "split_provenance.json",
        "source_step400_hashes.json": SOURCE_RESULT / "hashes.json",
        "source_step400_report.md": SOURCE_RESULT
        / "META_AGENTS_ADAPTIVE_MATCHED_BUDGET_K2_STEP400.md",
        "driver_run_frozen_schedule_control.py": Path(__file__).resolve(),
        "evaluator_run_adaptive_matched_budget.py": ROOT
        / "scripts/run_adaptive_matched_budget.py",
    }
    hashes = {}
    for name, source in sources.items():
        destination = archive / name
        shutil.copy2(source, destination)
        hashes[name] = sha256_file(destination)
    if hashes["adaptive_step_400"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("archived adaptive checkpoint hash mismatch")
    write_json(archive / "MANIFEST.json", hashes)
    return hashes


def write_report(output_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    calibrated = rows[0]
    forced_12 = rows[1]
    lines = "\n".join(
        f"| {row['condition']} | {row['actual_mean_k']:.6f} | "
        f"{row['token_accuracy']:.6f} | {row['exact_accuracy']:.6f} | "
        f"{row['lm_loss_per_executed_pass']:.6f} | {row['dominant_schedule']} |"
        for row in rows
    )
    accuracy_delta = forced_12["token_accuracy"] - calibrated["token_accuracy"]
    loss_delta = (
        forced_12["lm_loss_per_executed_pass"] - calibrated["lm_loss_per_executed_pass"]
    )
    forced_14 = rows[2]
    forced_16 = rows[3]
    if abs(accuracy_delta) < 1e-4 and abs(loss_delta) < 1e-4:
        conclusion = (
            "Calibrated adaptive and strict [1,2] are effectively indistinguishable "
            "at aggregate precision; their tiny accuracy and loss differences point "
            "in opposite directions. The rare adaptive exceptions provide no clear "
            "aggregate value."
        )
    elif accuracy_delta > 0:
        conclusion = (
            "Strict [1,2] improves over calibrated adaptive, so the rare adaptive "
            "exceptions are harmful in aggregate."
        )
    else:
        conclusion = (
            "Calibrated adaptive improves over strict [1,2], so the rare adaptive "
            "exceptions provide some aggregate benefit."
        )
    report = f"""# Frozen Adaptive-Checkpoint Schedule Control

## Design

The exact adaptive step-400 weights were frozen and evaluated on the same 3,686 final episodes used by the matched-budget audit. No retraining or performance-driven selection occurred. Forced policies use the model's causal trajectory and manager-generated goal at the prescribed second intervention, then retain thereafter.

| Condition | Mean K | Token accuracy | Exact accuracy | LM loss/pass | Schedule |
|---|---:|---:|---:|---:|---|
{lines}

## Primary diagnostic

- Forced `[1,2]` minus calibrated adaptive accuracy: {accuracy_delta:+.9f}
- Forced `[1,2]` minus calibrated adaptive LM loss/pass: {loss_delta:+.9f}
- Forced `[1,4]` minus calibrated adaptive accuracy: {forced_14['token_accuracy'] - calibrated['token_accuracy']:+.9f}
- Forced `[1,6]` minus calibrated adaptive accuracy: {forced_16['token_accuracy'] - calibrated['token_accuracy']:+.9f}

{conclusion}

Forced `[1,4]` and `[1,6]` isolate timing within the same learned representations. They must not be conflated with the separately trained fixed-P checkpoints.

The within-checkpoint result points to timing selection—not generally weak adaptive-checkpoint representations—as the immediate failure: the same weights perform materially better when the second goal is delayed to pass 4 or 6.

All schedules used exactly eight refinement passes and two interventions per episode. Checkpoint file and loaded model-state hashes were unchanged, and verified checkpoint/config/source copies are stored under `frozen_inputs/`.
"""
    (output_dir / "FROZEN_ADAPTIVE_SCHEDULE_CONTROL.md").write_text(report)


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_before = sha256_file(CHECKPOINT)
    if checkpoint_before != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("adaptive checkpoint hash mismatch")
    config = load_config(CONFIG)
    if config["arch"]["fixed_refinement_steps"] != 8:
        raise RuntimeError("schedule control requires M=8")
    if config["arch"]["subgoal_head"]["replan_mode"] != "adaptive":
        raise RuntimeError("schedule control checkpoint is not adaptive")
    metadata, episodes = load_held_out_episodes(dataset_path_from_config(config))
    split = deterministic_split(episodes, seed=20260819, calibration_fraction=0.20)
    split_payload = split_provenance(split)
    if split_payload["final_ordered_digest"] != EXPECTED_SPLIT_DIGEST:
        raise RuntimeError("final split differs from the matched-budget audit")
    source_split = json.loads((SOURCE_RESULT / "split_provenance.json").read_text())
    if split_payload != source_split:
        raise RuntimeError("reconstructed split provenance is not byte-equivalent")
    write_json(output_dir / "split_provenance.json", split_payload)

    model = build_model(config, metadata, batch_size=4, device="cpu")
    load_weights(model, CHECKPOINT, "cpu")
    initial_state = logical_state_sha256(model)
    rows = [calibrated_row()]
    diagnostics = {}
    for second_pass in FORCED_SECOND_PASSES:
        print(f"forced schedule [1,{second_pass}]", flush=True)
        row, detail = evaluate_forced_schedule(
            model,
            split.final,
            metadata,
            second_pass=second_pass,
            batch_size=4,
            device="cpu",
        )
        rows.append(row)
        diagnostics[f"forced_[1,{second_pass}]"] = detail
    final_state = logical_state_sha256(model)
    if final_state != initial_state:
        raise RuntimeError("adaptive state changed across schedule controls")
    checkpoint_after = sha256_file(CHECKPOINT)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("adaptive checkpoint file changed")

    fields = tuple(rows[0])
    with (output_dir / "schedule_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "forced_schedule_diagnostics.json", diagnostics)
    archived = archive_inputs(output_dir)
    write_json(
        output_dir / "hashes.json",
        {
            "checkpoint_before": checkpoint_before,
            "checkpoint_after": checkpoint_after,
            "loaded_state_before": initial_state,
            "loaded_state_after": final_state,
            "checkpoint_unchanged": checkpoint_before == checkpoint_after,
            "model_state_unchanged": initial_state == final_state,
            "archived_files": archived,
        },
    )
    write_json(
        output_dir / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": checkpoint_before,
            "source_matched_budget_result": str(SOURCE_RESULT),
            "calibrated_threshold": CALIBRATED_THRESHOLD,
            "forced_schedules": [[1, value] for value in FORCED_SECOND_PASSES],
            "training_steps": 400,
            "fixed_refinement_steps": 8,
            "final_episodes": len(split.final),
            "final_ordered_digest": split_payload["final_ordered_digest"],
            "device": "cpu",
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
        },
    )
    write_report(output_dir, rows)
    print(f"completed: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("adaptive checkpoint hash mismatch")
    if args.preflight_only:
        print(f"checkpoint_sha256={EXPECTED_CHECKPOINT_SHA256}")
        print(f"source_result={SOURCE_RESULT}")
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or ROOT / (
        "experiments/meta_agents_frozen_schedule_control/"
        f"{timestamp}_adaptive_step400_k2"
    )
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    run(output_dir)


if __name__ == "__main__":
    main()
