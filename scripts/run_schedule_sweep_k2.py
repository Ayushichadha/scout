#!/usr/bin/env python3
"""Exact six-clock K=2 sweep on the frozen adaptive step-400 checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HRM_ROOT = ROOT / "HRM"
for import_root in (ROOT, HRM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.eval_checkpoints_local import (  # noqa: E402
    build_model,
    dataset_path_from_config,
    load_config,
    load_weights,
)
from scripts.run_adaptive_matched_budget import (  # noqa: E402
    deterministic_split,
    load_held_out_episodes,
    logical_state_sha256,
    sha256_file,
    split_provenance,
    write_json,
)
from scripts.run_frozen_schedule_control import (  # noqa: E402
    CHECKPOINT,
    CONFIG,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_SPLIT_DIGEST,
    SOURCE_RESULT,
    evaluate_forced_schedule,
)


OUTPUT_DIR = ROOT / "experiments/schedule_sweep_k2"
SCHEDULES = tuple(range(2, 8))
MATRIX_FIELDS = (
    "episode_id",
    "schedule_k",
    "dwell_first",
    "dwell_second",
    "tokens_correct",
    "tokens_valid",
    "token_acc_episode",
    "lm_loss_sum",
    "passes_executed",
    "lm_loss_per_pass",
    "exact_correct",
)
EVALUATION_SEED = 20260819
BOOTSTRAP_SEED = 20260822
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_REGRESSION = {
    2: (0.4670020091060369, 1.9578407899625816),
    4: (0.4875110190223278, 1.956718238878651),
    6: (0.48662219732864376, 1.9574397634551794),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def schedule_positions(k: int) -> tuple[int, ...]:
    if k not in SCHEDULES:
        raise ValueError("k must be in 2..7")
    return tuple(position for position in range(1, 9) if position in (1, k))


def dwell_segments(k: int) -> tuple[int, int]:
    if k not in SCHEDULES:
        raise ValueError("k must be in 2..7")
    return k - 1, 8 - k


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    correct = sum(int(row["tokens_correct"]) for row in rows)
    valid = sum(int(row["tokens_valid"]) for row in rows)
    if not rows or valid <= 0:
        raise ValueError("cannot aggregate empty or invalid episode rows")
    return {
        "micro": correct / valid,
        "macro": float(np.mean([float(row["token_acc_episode"]) for row in rows])),
        "exact": float(np.mean([int(row["exact_correct"]) for row in rows])),
        "lm_loss_per_pass": sum(float(row["lm_loss_sum"]) for row in rows)
        / (len(rows) * 8),
    }


def matrix_arrays(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    grouped = {k: [] for k in SCHEDULES}
    for row in rows:
        grouped[int(row["schedule_k"])].append(row)
    ids = [str(row["episode_id"]) for row in grouped[2]]
    for k in SCHEDULES:
        if [str(row["episode_id"]) for row in grouped[k]] != ids:
            raise RuntimeError(f"episode IDs/order differ for schedule [1,{k}]")
    correct = np.column_stack(
        [[int(row["tokens_correct"]) for row in grouped[k]] for k in SCHEDULES]
    )
    valid = np.column_stack(
        [[int(row["tokens_valid"]) for row in grouped[k]] for k in SCHEDULES]
    )
    if not np.all(valid == valid[:, [0]]):
        raise RuntimeError("per-episode valid-token denominators vary by schedule")
    return ids, correct, valid


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_analysis(
    correct: np.ndarray,
    valid: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[tuple[float, float], dict[tuple[int, int], tuple[float, float]]]:
    rng = np.random.Generator(np.random.Philox(seed))
    headroom = np.empty(replicates, dtype=np.float64)
    differences = {
        pair: np.empty(replicates, dtype=np.float64)
        for pair in combinations(SCHEDULES, 2)
    }
    oracle_column = np.argmax(correct / valid, axis=1)
    oracle_correct = correct[np.arange(len(correct)), oracle_column]
    oracle_valid = valid[np.arange(len(valid)), oracle_column]
    for start in range(0, replicates, 250):
        stop = min(start + 250, replicates)
        indices = rng.integers(0, len(correct), size=(stop - start, len(correct)))
        sampled_correct = correct[indices].sum(axis=1)
        sampled_valid = valid[indices].sum(axis=1)
        fixed = sampled_correct / sampled_valid
        oracle = oracle_correct[indices].sum(axis=1) / oracle_valid[indices].sum(axis=1)
        headroom[start:stop] = oracle - fixed.max(axis=1)
        for (left, right), values in differences.items():
            left_index = SCHEDULES.index(left)
            right_index = SCHEDULES.index(right)
            values[start:stop] = fixed[:, left_index] - fixed[:, right_index]
    return percentile_ci(headroom), {
        pair: percentile_ci(values) for pair, values in differences.items()
    }


def analyze(rows: Sequence[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    grouped = {
        k: [row for row in rows if int(row["schedule_k"]) == k] for k in SCHEDULES
    }
    aggregates = {k: aggregate_rows(grouped[k]) for k in SCHEDULES}
    ids, correct, valid = matrix_arrays(rows)
    micro = np.asarray([aggregates[k]["micro"] for k in SCHEDULES])
    best_index = int(np.argmax(micro))
    best_k = SCHEDULES[best_index]
    episode_accuracy = correct / valid
    oracle_column = np.argmax(
        episode_accuracy, axis=1
    )  # documented smallest-k tie break
    oracle_correct = correct[np.arange(len(ids)), oracle_column]
    oracle_valid = valid[np.arange(len(ids)), oracle_column]
    ceiling = float(oracle_correct.sum() / oracle_valid.sum())
    floor = float(micro.mean())
    best_fixed = float(micro[best_index])
    headroom = ceiling - best_fixed
    distribution = {
        k: int((oracle_column == index).sum()) for index, k in enumerate(SCHEDULES)
    }
    ties = int(
        np.sum(
            (episode_accuracy == episode_accuracy.max(axis=1, keepdims=True)).sum(
                axis=1
            )
            > 1
        )
    )
    headroom_ci, pair_cis = bootstrap_analysis(correct, valid)
    pair_values = {
        pair: aggregates[pair[0]]["micro"] - aggregates[pair[1]]["micro"]
        for pair in combinations(SCHEDULES, 2)
    }

    schedule_lines = "\n".join(
        f"| `[1,{k}]` | ({k - 1}, {8 - k}) | {aggregates[k]['micro']:.9f} | "
        f"{aggregates[k]['macro']:.9f} | {aggregates[k]['lm_loss_per_pass']:.9f} | "
        f"{aggregates[k]['exact']:.9f} |"
        for k in SCHEDULES
    )
    pair_lines = "\n".join(
        f"| `[1,{left}] - [1,{right}]` | {pair_values[(left, right)]:+.9f} | "
        f"[{pair_cis[(left, right)][0]:+.9f}, {pair_cis[(left, right)][1]:+.9f}] |"
        for left, right in combinations(SCHEDULES, 2)
    )
    matched = ((2, 7), (3, 6), (4, 5))
    matched_lines = []
    all_indistinguishable = True
    for left, right in matched:
        ci = pair_cis[(left, right)]
        distinguishable = ci[0] > 0 or ci[1] < 0
        all_indistinguishable &= not distinguishable
        verdict = "distinguishable" if distinguishable else "not distinguishable"
        matched_lines.append(
            f"| `[1,{left}] - [1,{right}]` | {pair_values[(left, right)]:+.9f} | "
            f"[{ci[0]:+.9f}, {ci[1]:+.9f}] | {verdict} |"
        )
    if all_indistinguishable:
        order_conclusion = (
            "All three reversed-order pairs are statistically indistinguishable; "
            "under these tests, schedule value is a function of the dwell multiset alone."
        )
    else:
        separated = [
            f"[1,{left}] vs [1,{right}]"
            for left, right in matched
            if pair_cis[(left, right)][0] > 0 or pair_cis[(left, right)][1] < 0
        ]
        order_conclusion = (
            "At least one reversed-order pair separates ("
            + ", ".join(separated)
            + "); position carries information beyond dwell."
        )
    dominant_k = max(distribution, key=distribution.get)
    dominant_fraction = distribution[dominant_k] / len(ids)
    thin_text = (
        f"Schedule `[1,{dominant_k}]` receives the deterministic argmax for "
        f"{dominant_fraction:.1%} of episodes. "
        + (
            "The reported allocation is concentrated and the measured headroom is thin. "
            "Because most episodes tie and ties go to the smallest `k`, the identity and "
            "share of the dominant schedule are tie-break-sensitive; the ceiling and "
            "headroom are not."
            if dominant_fraction > 0.5
            else "The oracle allocation is not dominated by one schedule."
        )
    )
    report = (
        f"""# Exact K=2 Forced-Schedule Ceiling Analysis

The headline aggregation throughout this report is **micro token accuracy** (total correct valid tokens divided by total valid tokens). Macro means of per-episode token accuracy are shown separately. Every schedule contains the same {len(ids):,} ordered episodes.

## Schedule aggregates

| Schedule | Dwells | Micro token accuracy | Macro token accuracy | LM loss/pass | Exact accuracy |
|---|---|---:|---:|---:|---:|
{schedule_lines}

## Fixed policy, floor, and held-out-label upper bound

- Best fixed clock: `[1,{best_k}]`, micro token accuracy `{best_fixed:.9f}`.
- Floor (analytical expectation under a uniform random schedule per episode): `{floor:.9f}`.
- **Oracle ceiling (held-out-label upper bound; not an achievable policy or performance result):** `{ceiling:.9f}`.
- Total K=2 state-conditioned timing headroom: `{headroom:.9f}` (95% episode-bootstrap CI `[{headroom_ci[0]:.9f}, {headroom_ci[1]:.9f}]`).

The bootstrap resamples episodes with replacement {BOOTSTRAP_REPLICATES:,} times using NumPy Philox seed `{BOOTSTRAP_SEED}`. In each headroom replicate, the best fixed schedule is recomputed; the oracle choice remains the per-episode held-out-label argmax. Percentile intervals are reported.

## Oracle argmax distribution

Ties are broken deterministically in favor of the smallest `k`; {ties:,} episodes have two or more tied maxima.

| Schedule | Episodes | Fraction |
|---|---:|---:|
"""
        + "\n".join(
            f"| `[1,{k}]` | {distribution[k]} | {distribution[k] / len(ids):.6f} |"
            for k in SCHEDULES
        )
        + f"""

{thin_text}

## All pairwise schedule differences

Differences are left minus right in headline micro token accuracy, with paired episode-bootstrap 95% CIs.

| Difference | Estimate | 95% CI |
|---|---:|---:|
{pair_lines}

## Paired reversed-order dwell test

| Matched pair difference | Estimate | 95% CI | Verdict |
|---|---:|---:|---|
{chr(10).join(matched_lines)}

{order_conclusion}

## Preregistered predictions

The three predictions equated the reversed-order pairs `[1,5]`/`[1,4]`, `[1,3]`/`[1,6]`, and `[1,7]`/`[1,2]`. A prediction is counted as holding when its paired CI includes zero. Therefore, {sum(pair_cis[p][0] <= 0 <= pair_cis[p][1] for p in matched)} of 3 predictions held by the preregistered paired-order criterion.
"""
    )
    summary = {
        "aggregates": {str(k): aggregates[k] for k in SCHEDULES},
        "best_fixed_k": best_k,
        "best_fixed_micro": best_fixed,
        "floor_micro": floor,
        "oracle_ceiling_micro": ceiling,
        "headroom_micro": headroom,
        "headroom_ci95": headroom_ci,
        "argmax_distribution": distribution,
        "argmax_ties": ties,
        "pair_differences": {f"{a}-{b}": pair_values[(a, b)] for a, b in pair_values},
        "pair_ci95": {f"{a}-{b}": pair_cis[(a, b)] for a, b in pair_cis},
        "all_matched_pairs_indistinguishable": all_indistinguishable,
    }
    return report, summary


def write_matrix(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_matrix(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    ).stdout


def artifact_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.relative_to(OUTPUT_DIR)): sha256_file(path) for path in paths}


def finalize_existing(output_dir: Path) -> None:
    """Validate preserved schedule outputs and finish downstream artifacts."""

    finalize_started = time.monotonic()
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA-256 mismatch during finalization")
    source_split = json.loads((SOURCE_RESULT / "split_provenance.json").read_text())
    saved_split = json.loads((output_dir / "split_provenance.json").read_text())
    if (
        saved_split != source_split
        or saved_split["final_ordered_digest"] != EXPECTED_SPLIT_DIGEST
    ):
        raise RuntimeError("preserved split differs from the Aug 20 control")
    expected_ids = [item["id"] for item in saved_split["final_episodes"]]
    all_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    state_hashes: set[str] = set()
    prior_mtime = (output_dir / "split_provenance.json").stat().st_mtime
    for k in SCHEDULES:
        rows_path = output_dir / f"schedule_1_{k}_rows.csv"
        diagnostics_path = output_dir / f"schedule_1_{k}_diagnostics.json"
        rows = read_matrix(rows_path)
        diagnostics_payload = json.loads(diagnostics_path.read_text())
        aggregate = diagnostics_payload["aggregate"]
        invariants = diagnostics_payload["invariants"]
        if len(rows) != len(expected_ids):
            raise RuntimeError(f"preserved [1,{k}] row count mismatch")
        if [row["episode_id"] for row in rows] != expected_ids:
            raise RuntimeError(f"preserved [1,{k}] episode order mismatch")
        for row in rows:
            if int(row["schedule_k"]) != k:
                raise RuntimeError(f"preserved [1,{k}] contains a wrong schedule label")
            if (int(row["dwell_first"]), int(row["dwell_second"])) != dwell_segments(k):
                raise RuntimeError(f"preserved [1,{k}] contains wrong dwell segments")
            if int(row["passes_executed"]) != 8:
                raise RuntimeError(f"preserved [1,{k}] did not execute eight passes")
            if int(row["tokens_valid"]) <= 0:
                raise RuntimeError(f"preserved [1,{k}] has no valid tokens")
            if (
                abs(
                    float(row["token_acc_episode"])
                    - int(row["tokens_correct"]) / int(row["tokens_valid"])
                )
                > 1e-12
            ):
                raise RuntimeError(
                    f"preserved [1,{k}] episode accuracy is inconsistent"
                )
            if (
                abs(float(row["lm_loss_per_pass"]) - float(row["lm_loss_sum"]) / 8)
                > 1e-12
            ):
                raise RuntimeError(f"preserved [1,{k}] per-pass loss is inconsistent")
        if invariants["expected_schedule"] != [1, k]:
            raise RuntimeError(f"preserved [1,{k}] intervention schedule mismatch")
        if invariants["dwell_segments"] != list(dwell_segments(k)):
            raise RuntimeError(f"preserved [1,{k}] diagnostic dwell mismatch")
        if not invariants["model_state_stable"]:
            raise RuntimeError(f"preserved [1,{k}] model-state invariant failed")
        if (
            invariants["checkpoint_state_before"]
            != invariants["checkpoint_state_after"]
        ):
            raise RuntimeError(f"preserved [1,{k}] before/after state hashes differ")
        state_hashes.add(invariants["checkpoint_state_before"])
        computed = aggregate_rows(rows)
        if abs(computed["macro"] - float(aggregate["token_accuracy"])) > 1e-6:
            raise RuntimeError(f"preserved [1,{k}] accuracy aggregate mismatch")
        if (
            abs(
                computed["lm_loss_per_pass"]
                - float(aggregate["lm_loss_per_executed_pass"])
            )
            > 1e-6
        ):
            raise RuntimeError(f"preserved [1,{k}] loss aggregate mismatch")
        if k in EXPECTED_REGRESSION:
            expected_acc, expected_loss = EXPECTED_REGRESSION[k]
            if abs(float(aggregate["token_accuracy"]) - expected_acc) >= 5e-6:
                raise RuntimeError(f"preserved [1,{k}] accuracy regression failed")
            if (
                abs(float(aggregate["lm_loss_per_executed_pass"]) - expected_loss)
                >= 5e-6
            ):
                raise RuntimeError(f"preserved [1,{k}] loss regression failed")
        completed_mtime = diagnostics_path.stat().st_mtime
        run_records.append(
            {
                "schedule": [1, k],
                "output_paths": [str(rows_path), str(diagnostics_path)],
                "wall_clock_seconds_inferred_from_consecutive_artifact_mtimes": max(
                    0.0, completed_mtime - prior_mtime
                ),
            }
        )
        prior_mtime = completed_mtime
        all_rows.extend(rows)
    if len(state_hashes) != 1:
        raise RuntimeError(
            "loaded model-state hash differs across preserved evaluations"
        )

    matrix_path = output_dir / "per_episode_matrix.csv"
    write_matrix(matrix_path, all_rows)
    report, summary = analyze(all_rows)
    analysis_path = output_dir / "ceiling_analysis.md"
    analysis_path.write_text(report)
    summary_path = output_dir / "analysis_summary.json"
    write_json(summary_path, summary)
    provenance_path = output_dir / "provenance.json"
    provenance = {
        "schema_version": "schedule_sweep_k2_v1",
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256_expected": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_sha256_at_finalization": sha256_file(CHECKPOINT),
        "loaded_model_state_sha256": next(iter(state_hashes)),
        "model_state_verified_after_every_evaluation": True,
        "split_digest": saved_split["final_ordered_digest"],
        "split_size": len(expected_ids),
        "git_head": git_output("rev-parse", "HEAD").strip(),
        "git_status_full": git_output("status"),
        "seeds": {"evaluation": EVALUATION_SEED, "bootstrap": BOOTSTRAP_SEED},
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "deterministic_algorithms_during_evaluation": True,
        "library_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "wall_clock": {
            "evaluation_elapsed_seconds_inferred_from_artifact_mtimes": max(
                0.0,
                (output_dir / "schedule_1_7_diagnostics.json").stat().st_mtime
                - (output_dir / "split_provenance.json").stat().st_mtime,
            ),
            "finalization_seconds": time.monotonic() - finalize_started,
            "note": "Evaluation frontend detached after schedule 7; elapsed evaluation time is reconstructed from preserved artifact mtimes and includes detached/paused wall time.",
        },
        "runs": run_records,
        "artifact_paths": {
            "matrix": str(matrix_path),
            "analysis": str(analysis_path),
            "summary": str(summary_path),
            "split_provenance": str(output_dir / "split_provenance.json"),
        },
    }
    write_json(provenance_path, provenance)
    artifacts = [
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "hashes.json"
    ]
    write_json(
        output_dir / "hashes.json",
        {
            "algorithm": "sha256",
            "self_excluded": "hashes.json cannot contain its own stable digest",
            "artifacts": artifact_hashes(sorted(artifacts)),
        },
    )
    print(f"finalized exact K=2 sweep: {output_dir}", flush=True)


def run(output_dir: Path) -> None:
    started_wall = time.monotonic()
    started_utc = datetime.now(timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=True)
    prereg = output_dir / "PREREGISTRATION.md"
    if (
        not prereg.exists()
        or not git_output("log", "-1", "--format=%H", "--", str(prereg)).strip()
    ):
        raise RuntimeError("committed preregistration is required before evaluation")
    checkpoint_before = sha256_file(CHECKPOINT)
    if checkpoint_before != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA-256 mismatch before loading")
    seed_everything(EVALUATION_SEED)
    config = load_config(CONFIG)
    if config["arch"]["fixed_refinement_steps"] != 8:
        raise RuntimeError("K=2 sweep requires exactly M=8")
    metadata, episodes = load_held_out_episodes(dataset_path_from_config(config))
    split = deterministic_split(
        episodes, seed=EVALUATION_SEED, calibration_fraction=0.20
    )
    split_payload = split_provenance(split)
    source_split = json.loads((SOURCE_RESULT / "split_provenance.json").read_text())
    if (
        split_payload["final_ordered_digest"] != EXPECTED_SPLIT_DIGEST
        or split_payload != source_split
    ):
        raise RuntimeError("ordered final split differs from the Aug 20 control")
    write_json(output_dir / "split_provenance.json", split_payload)

    model = build_model(config, metadata, batch_size=4, device="cpu")
    load_weights(model, CHECKPOINT, "cpu")
    loaded_state = logical_state_sha256(model)
    all_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    for k in SCHEDULES:
        run_started = time.monotonic()
        if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError(f"checkpoint file changed before schedule [1,{k}]")
        aggregate, diagnostics = evaluate_forced_schedule(
            model,
            split.final,
            metadata,
            second_pass=k,
            batch_size=4,
            device="cpu",
            collect_per_episode=True,
        )
        rows = diagnostics.pop("per_episode")
        rows_path = output_dir / f"schedule_1_{k}_rows.csv"
        write_matrix(rows_path, rows)
        if len(rows) != len(split.final):
            raise RuntimeError(f"schedule [1,{k}] emitted the wrong row count")
        expected_ids = [episode.episode_id for episode in split.final]
        if [row["episode_id"] for row in rows] != expected_ids:
            raise RuntimeError(f"schedule [1,{k}] changed episode order")
        if logical_state_sha256(model) != loaded_state:
            raise RuntimeError(f"model state changed after schedule [1,{k}]")
        if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError(f"checkpoint file changed after schedule [1,{k}]")
        computed = aggregate_rows(rows)
        if abs(computed["macro"] - float(aggregate["token_accuracy"])) > 1e-6:
            raise RuntimeError(f"per-episode accuracy mismatch for schedule [1,{k}]")
        if (
            abs(
                computed["lm_loss_per_pass"]
                - float(aggregate["lm_loss_per_executed_pass"])
            )
            > 1e-6
        ):
            raise RuntimeError(f"per-episode LM loss mismatch for schedule [1,{k}]")
        if k in EXPECTED_REGRESSION:
            expected_acc, expected_loss = EXPECTED_REGRESSION[k]
            if abs(float(aggregate["token_accuracy"]) - expected_acc) >= 5e-6:
                raise RuntimeError(f"[1,{k}] accuracy failed five-decimal regression")
            if (
                abs(float(aggregate["lm_loss_per_executed_pass"]) - expected_loss)
                >= 5e-6
            ):
                raise RuntimeError(f"[1,{k}] LM loss failed five-decimal regression")
        all_rows.extend(rows)
        diagnostics_path = output_dir / f"schedule_1_{k}_diagnostics.json"
        write_json(
            diagnostics_path, {"aggregate": aggregate, "invariants": diagnostics}
        )
        run_records.append(
            {
                "schedule": [1, k],
                "output_paths": [str(rows_path), str(diagnostics_path)],
                "wall_clock_seconds": time.monotonic() - run_started,
            }
        )
        print(
            f"completed [1,{k}]: accuracy={aggregate['token_accuracy']:.9f}", flush=True
        )

    matrix_path = output_dir / "per_episode_matrix.csv"
    write_matrix(matrix_path, all_rows)
    report, summary = analyze(all_rows)
    analysis_path = output_dir / "ceiling_analysis.md"
    analysis_path.write_text(report)
    summary_path = output_dir / "analysis_summary.json"
    write_json(summary_path, summary)
    finished_utc = datetime.now(timezone.utc)
    provenance_path = output_dir / "provenance.json"
    provenance = {
        "schema_version": "schedule_sweep_k2_v1",
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256_expected": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": sha256_file(CHECKPOINT),
        "loaded_model_state_sha256": loaded_state,
        "model_state_verified_after_every_evaluation": True,
        "split_digest": split_payload["final_ordered_digest"],
        "split_size": len(split.final),
        "git_head": git_output("rev-parse", "HEAD").strip(),
        "git_status_full": git_output("status"),
        "seeds": {"evaluation": EVALUATION_SEED, "bootstrap": BOOTSTRAP_SEED},
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "library_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "environment": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
        "started_at_utc": started_utc.isoformat(),
        "finished_at_utc": finished_utc.isoformat(),
        "wall_clock_seconds": time.monotonic() - started_wall,
        "runs": run_records,
        "artifact_paths": {
            "matrix": str(matrix_path),
            "analysis": str(analysis_path),
            "summary": str(summary_path),
            "split_provenance": str(output_dir / "split_provenance.json"),
        },
    }
    if provenance["checkpoint_sha256_after"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA-256 changed during sweep")
    write_json(provenance_path, provenance)
    artifacts = [
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "hashes.json"
    ]
    write_json(
        output_dir / "hashes.json",
        {
            "algorithm": "sha256",
            "self_excluded": "hashes.json cannot contain its own stable digest",
            "artifacts": artifact_hashes(sorted(artifacts)),
        },
    )
    print(f"completed exact K=2 sweep: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if args.finalize_existing:
        finalize_existing(args.output_dir.resolve())
    else:
        run(args.output_dir.resolve())


if __name__ == "__main__":
    main()
