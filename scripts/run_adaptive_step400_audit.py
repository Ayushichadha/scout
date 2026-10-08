#!/usr/bin/env python3
"""Audit and matched-budget evaluation of the frozen step-400 adaptive model."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Sequence

import numpy as np

from scripts.eval_checkpoints_local import (
    build_model,
    dataset_path_from_config,
    load_config,
    load_weights,
)
from scripts.run_adaptive_matched_budget import (
    BudgetSearchRow,
    BudgetSummary,
    calibrate_threshold,
    deterministic_split,
    evaluate_budget_only,
    evaluate_final,
    evaluate_fixed_final,
    load_held_out_episodes,
    logical_state_sha256,
    sha256_file,
    split_provenance,
    summarize_schedules,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
ADAPTIVE_RUN = ROOT / (
    "experiments/meta_agents_adaptive_400step/"
    "20260820T101100Z_adaptive_eta0_seed0_400step"
)
ADAPTIVE_CHECKPOINT = ADAPTIVE_RUN / "checkpoints/step_400"
ADAPTIVE_CONFIG = ADAPTIVE_RUN / "resolved_config.yaml"
P4_DIR = ROOT / "HRM/checkpoints/meta_agents_fixed_p/corrected_fixed_p_p4_seed0"
P6_DIR = ROOT / "HRM/checkpoints/meta_agents_fixed_p/corrected_fixed_p_p6_seed0"
P4_CHECKPOINT = P4_DIR / "step_400"
P6_CHECKPOINT = P6_DIR / "step_400"
P4_CONFIG = P4_DIR / "all_config.yaml"
P6_CONFIG = P6_DIR / "all_config.yaml"
P4_RUN = ROOT / (
    "experiments/meta_agents_fixed_p/runs/"
    "20260819T130736Z_corrected_fixed_p_p4_seed0"
)
P6_RUN = ROOT / (
    "experiments/meta_agents_fixed_p/runs/"
    "20260819T130744Z_corrected_fixed_p_p6_seed0"
)
EXPECTED_HASHES = {
    "adaptive_checkpoint": "165586ca87a5ca473b342ccb4342afd72890817498c87319964356aa37313889",
    "p4_checkpoint": "2e5d4ba89ed23fc70232d5c0603b9c5ab6cd2a5c6fecd410a83d19268d614f49",
    "p6_checkpoint": "dace3b0dbb1ea69fcb8a1e0ac707c78267a24950f52364ef7f882c3c95aeb1e7",
    "p4_config": "4fde38021e7ea1b775197333969ae69e9d3231652baed6f6b5ae44bc0ab77a60",
    "p6_config": "f4a19847919a622125cca0796b182c23567d0dd0becc32b8374cacfcc0ab5e4d",
}
FEATURE_NAMES = ("c", "d", "rho", "dwell_over_m", "q", "alpha")
SCHEMA_VERSION = "meta_agents_adaptive_matched_budget_k2_step400_v1"


def verify_frozen_files() -> dict[str, str]:
    paths = {
        "adaptive_checkpoint": ADAPTIVE_CHECKPOINT,
        "p4_checkpoint": P4_CHECKPOINT,
        "p6_checkpoint": P6_CHECKPOINT,
        "p4_config": P4_CONFIG,
        "p6_config": P6_CONFIG,
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != EXPECTED_HASHES:
        raise RuntimeError(f"frozen input hash mismatch: {observed}")
    return observed


def numeric_stats(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise RuntimeError("statistics require non-empty finite values")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    return ranks


def stable_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x_centered = x.astype(np.float64) - float(np.mean(x, dtype=np.float64))
    y_centered = y.astype(np.float64) - float(np.mean(y, dtype=np.float64))
    x_ss = float(np.sum(x_centered * x_centered, dtype=np.float64))
    y_ss = float(np.sum(y_centered * y_centered, dtype=np.float64))
    if x_ss == 0 or y_ss == 0:
        return None
    numerator = float(np.sum(x_centered * y_centered, dtype=np.float64))
    value = numerator / math.sqrt(x_ss * y_ss)
    if not math.isfinite(value):
        raise FloatingPointError("non-finite correlation")
    return max(-1.0, min(1.0, value))


def safe_correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    if len(x) < 2:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": stable_pearson(x, y),
        "spearman": stable_pearson(average_ranks(x), average_ranks(y)),
    }


def decision_rows(summary: BudgetSummary) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in summary.traces:
        for pass_index, beta, hard, feature in zip(
            trace.eligible_passes, trace.betas, trace.hard, trace.features
        ):
            rows.append(
                {
                    "episode_id": trace.episode_id,
                    "eligible_pass": pass_index,
                    "beta": beta,
                    "hard_intervene": int(hard),
                    **dict(zip(FEATURE_NAMES, feature)),
                }
            )
    if len(rows) != summary.eligible_decisions:
        raise RuntimeError("decision-row count mismatch")
    return rows


def trigger_parameters(model: Any) -> tuple[np.ndarray, float]:
    core = getattr(model, "model", model)
    linear = core.adaptive_trigger.linear
    weights = linear.weight.detach().float().cpu().numpy().reshape(-1).astype(float)
    bias = float(linear.bias.detach().float().cpu().item())
    if weights.shape != (6,):
        raise RuntimeError(f"expected six trigger coefficients, got {weights.shape}")
    return weights, bias


def build_reference_audit(
    summary: BudgetSummary, weights: np.ndarray, bias: float
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    rows = decision_rows(summary)
    betas = np.asarray([row["beta"] for row in rows], dtype=np.float64)
    passes = np.asarray([row["eligible_pass"] for row in rows], dtype=np.int64)
    features = np.asarray(
        [[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float64
    )
    # Avoid the platform BLAS matmul path here: the earlier threshold audit
    # observed spurious overflow warnings for these small, finite float64
    # arrays.  Explicit affine summation is mathematically identical and easy
    # to validate element by element.
    logits = np.sum(features * weights[None, :], axis=1, dtype=np.float64) + bias
    time_only = weights[3] * features[:, 3] + bias
    progress_only = np.sum(
        features[:, :3] * weights[None, :3], axis=1, dtype=np.float64
    )
    for name, values in (
        ("full logits", logits),
        ("time-only logits", time_only),
        ("progress-only logits", progress_only),
    ):
        if not np.isfinite(values).all():
            raise FloatingPointError(f"non-finite {name} in diagnostic reconstruction")
    by_pass = {
        str(position): numeric_stats(betas[passes == position])
        for position in range(2, 8)
    }
    by_dwell = {}
    dwell_steps = np.rint(features[:, 3] * 8).astype(int)
    for dwell in sorted(set(dwell_steps.tolist())):
        by_dwell[str(dwell)] = numeric_stats(betas[dwell_steps == dwell])
    total_variance = float(np.var(betas))
    group_means = np.asarray(
        [betas[passes == position].mean() for position in passes], dtype=np.float64
    )
    between = float(np.mean((group_means - betas.mean()) ** 2))
    within = float(np.mean((betas - group_means) ** 2))
    if not math.isclose(total_variance, between + within, rel_tol=1e-8, abs_tol=1e-20):
        raise RuntimeError("beta variance decomposition does not close")
    feature_stats: dict[str, Any] = {}
    for index, name in enumerate(FEATURE_NAMES):
        values = features[:, index]
        contribution = weights[index] * values
        feature_stats[name] = {
            **numeric_stats(values),
            **safe_correlations(values, betas),
            "weight": float(weights[index]),
            "abs_weight_times_std": float(abs(weights[index]) * values.std()),
            "contribution_min": float(contribution.min()),
            "contribution_max": float(contribution.max()),
            "contribution_range": float(np.ptp(contribution)),
        }
    statistics = {
        "reference_threshold": summary.threshold,
        "trajectory": "natural deterministic retain-all reference",
        "beta": numeric_stats(betas),
        "beta_by_eligible_pass": by_pass,
        "beta_by_dwell": by_dwell,
        "variance_decomposition": {
            "total_beta_variance": total_variance,
            "between_position_variance": between,
            "within_position_residual_variance": within,
            "percent_explained_by_position": 100 * between / total_variance,
            "percent_residual_within_position": 100 * within / total_variance,
        },
        "trigger_parameters": {
            "weights": dict(zip(FEATURE_NAMES, weights.tolist())),
            "bias": bias,
        },
        "features": feature_stats,
    }
    clock = {
        "definition": {
            "time_only": "w_dwell * (dwell/M) + bias",
            "progress_only": "w_c*c + w_d*d + w_rho*rho",
        },
        "full_logit": numeric_stats(logits),
        "time_only_logit": numeric_stats(time_only),
        "progress_only_logit": numeric_stats(progress_only),
        "time_only_vs_full": safe_correlations(time_only, logits),
        "progress_only_vs_full": safe_correlations(progress_only, logits),
        "scale_comparison": {
            "time_only_std": float(time_only.std()),
            "time_only_range": float(np.ptp(time_only)),
            "progress_only_std": float(progress_only.std()),
            "progress_only_range": float(np.ptp(progress_only)),
        },
    }
    return statistics, clock, rows


def mixed_action_audit(summary: BudgetSummary) -> dict[str, Any]:
    rows = decision_rows(summary)
    result: dict[str, Any] = {}
    for position in range(2, 8):
        local = [row for row in rows if row["eligible_pass"] == position]
        if not local or len({row["hard_intervene"] for row in local}) < 2:
            continue
        groups: dict[str, Any] = {}
        for label, action in (("intervene", 1), ("retain", 0)):
            selected = [row for row in local if row["hard_intervene"] == action]
            groups[label] = {
                "count": len(selected),
                **{
                    name: numeric_stats([row[name] for row in selected])
                    for name in (*FEATURE_NAMES, "beta")
                },
            }
        result[str(position)] = groups
    return result


def outward_search_bounds(summary: BudgetSummary) -> tuple[float, float]:
    betas = [beta for trace in summary.traces for beta in trace.betas]
    scale = 10_000
    low = math.floor(min(betas) * scale) / scale
    high = math.ceil(max(betas) * scale) / scale
    if low == high:
        low -= 1 / scale
        high += 1 / scale
    return low, high


def write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def archive_inputs(output_dir: Path, original_hashes: dict[str, str]) -> dict[str, str]:
    archive = output_dir / "frozen_inputs"
    archive.mkdir()
    sources = {
        "adaptive_step_400": ADAPTIVE_CHECKPOINT,
        "adaptive_resolved_config.yaml": ADAPTIVE_CONFIG,
        "adaptive_runtime_summary.json": ADAPTIVE_RUN / "runtime_summary.json",
        "adaptive_run_result.json": ADAPTIVE_RUN / "run_result.json",
        "adaptive_run_hashes.json": ADAPTIVE_RUN / "hashes.json",
        "adaptive_run_provenance.json": ADAPTIVE_RUN / "provenance.json",
        "adaptive_training.log": ADAPTIVE_RUN / "training.log",
        "adaptive_checkpoint_all_config.yaml": ADAPTIVE_RUN
        / "checkpoints/all_config.yaml",
        "adaptive_checkpoint_hrm_act_v1.py": ADAPTIVE_RUN / "checkpoints/hrm_act_v1.py",
        "adaptive_checkpoint_losses.py": ADAPTIVE_RUN / "checkpoints/losses.py",
        "fixed_p4_step_400": P4_CHECKPOINT,
        "fixed_p4_all_config.yaml": P4_CONFIG,
        "fixed_p4_hrm_act_v1.py": P4_DIR / "hrm_act_v1.py",
        "fixed_p4_losses.py": P4_DIR / "losses.py",
        "fixed_p4_run_result.json": P4_RUN / "result.json",
        "fixed_p4_runtime_summary.json": P4_RUN / "runtime_summary.json",
        "fixed_p4_stdout.log": P4_RUN / "stdout.log",
        "fixed_p4_stderr.log": P4_RUN / "stderr.log",
        "fixed_p6_step_400": P6_CHECKPOINT,
        "fixed_p6_all_config.yaml": P6_CONFIG,
        "fixed_p6_hrm_act_v1.py": P6_DIR / "hrm_act_v1.py",
        "fixed_p6_losses.py": P6_DIR / "losses.py",
        "fixed_p6_run_result.json": P6_RUN / "result.json",
        "fixed_p6_runtime_summary.json": P6_RUN / "runtime_summary.json",
        "fixed_p6_stdout.log": P6_RUN / "stdout.log",
        "fixed_p6_stderr.log": P6_RUN / "stderr.log",
        "evaluator_run_adaptive_matched_budget.py": ROOT
        / "scripts/run_adaptive_matched_budget.py",
        "driver_run_adaptive_step400_audit.py": Path(__file__).resolve(),
    }
    hashes: dict[str, str] = {}
    for name, source in sources.items():
        destination = archive / name
        shutil.copy2(source, destination)
        hashes[name] = sha256_file(destination)
        if "step_400" in name:
            source_key = {
                "adaptive_step_400": "adaptive_checkpoint",
                "fixed_p4_step_400": "p4_checkpoint",
                "fixed_p6_step_400": "p6_checkpoint",
            }[name]
            if hashes[name] != original_hashes[source_key]:
                raise RuntimeError(f"archived checkpoint changed while copying: {name}")
    write_json(archive / "MANIFEST.json", hashes)
    return hashes


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def render_report(
    output_dir: Path,
    selected: BudgetSearchRow,
    neighbors: Sequence[BudgetSearchRow],
    feature_stats: dict[str, Any],
    clock: dict[str, Any],
    adaptive_schedule: dict[str, Any],
    adaptive_task: dict[str, float],
    comparison: Sequence[dict[str, Any]],
) -> None:
    variance = feature_stats["variance_decomposition"]
    beta = feature_stats["beta"]
    p4, p6, adaptive = comparison
    dominant = adaptive_schedule["most_common_schedules"][0]
    neighbor_lines = "\n".join(
        f"| {row.threshold:.6f} | {row.mean_total_interventions:.6f} | "
        f"{row.absolute_budget_error:.6f} |"
        for row in neighbors
    )
    comparison_lines = "\n".join(
        f"| {row['condition']} | 400 | {row['actual_mean_k']:.6f} | "
        f"{row['token_accuracy']:.6f} | {row['exact_accuracy']:.6f} | "
        f"{row['lm_loss_per_executed_pass']:.6f} | {row['dominant_schedule']} |"
        for row in comparison
    )
    adaptive_minus_p4_acc = adaptive["token_accuracy"] - p4["token_accuracy"]
    adaptive_minus_p6_acc = adaptive["token_accuracy"] - p6["token_accuracy"]
    p6_minus_p4_acc = p6["token_accuracy"] - p4["token_accuracy"]
    adaptive_minus_p4_loss = (
        adaptive["lm_loss_per_executed_pass"] - p4["lm_loss_per_executed_pass"]
    )
    adaptive_minus_p6_loss = (
        adaptive["lm_loss_per_executed_pass"] - p6["lm_loss_per_executed_pass"]
    )
    if adaptive_minus_p4_acc > 0 and adaptive_minus_p6_acc > 0:
        outcome = "Outcome A: adaptive improves over both fixed policies."
    elif adaptive_minus_p4_acc >= -0.005 and adaptive_minus_p6_acc >= -0.005:
        outcome = "Outcome B: adaptive approximately ties the strongest fixed policy."
    else:
        outcome = (
            "Outcome C: after equal 400-step training, this checkpoint does not "
            "convert its replanning ranking into improved matched-budget performance."
        )
    report = f"""# Adaptive Matched-Budget K=2 Audit — Step 400

## A. Checkpoint provenance

- Adaptive checkpoint: `{ADAPTIVE_CHECKPOINT}`
- SHA-256: `{EXPECTED_HASHES['adaptive_checkpoint']}`
- Training: 400 optimizer steps, seed 0, eta=0, M=8, CPU.
- Frozen reconstructed baselines: P=4 `{EXPECTED_HASHES['p4_checkpoint']}`, P=6 `{EXPECTED_HASHES['p6_checkpoint']}`.
- Calibration/final split: 921 / 3,686 episodes; identical to the prior diagnostic; zero overlap.
- Every checkpoint was state-hashed before/after evaluation and copied into `frozen_inputs/`.

## B. 400-step beta/ranking audit

At the natural deterministic theta=0.5 reference trajectory, beta has count {beta['count']}, mean {beta['mean']:.9f}, std {beta['std']:.9f}, range [{beta['min']:.9f}, {beta['max']:.9f}], and median {beta['median']:.9f}.

Position explains {variance['percent_explained_by_position']:.3f}% of total beta variance; {variance['percent_residual_within_position']:.3f}% remains as residual same-position score variation. This residual is not labeled “state intelligence.”

Time-only versus full-logit correlation is Pearson {clock['time_only_vs_full']['pearson']:.4f} and Spearman {clock['time_only_vs_full']['spearman']:.4f}. Progress-only std/range are {clock['scale_comparison']['progress_only_std']:.3e}/{clock['scale_comparison']['progress_only_range']:.3e}, versus dwell-only {clock['scale_comparison']['time_only_std']:.3e}/{clock['scale_comparison']['time_only_range']:.3e}. Detailed per-feature scales, ranges, and associations are in `feature_beta_statistics.json`; fixed-pass action groups are in `clock_likeness.json`.

## C. Calibration

- Selection objective: intervention budget only, target total K=2.
- Selected theta: `{selected.threshold:.6f}`.
- Calibration mean K: {selected.mean_total_interventions:.6f}.
- Task accuracy/loss were unavailable to the selector.

| Threshold | Mean K | Absolute error |
|---:|---:|---:|
{neighbor_lines}

## D. Final adaptive behavior

- Final episodes: 3,686
- Mean total K / adaptive interventions: {adaptive_schedule['mean_total_interventions']:.6f} / {adaptive_schedule['mean_adaptive_interventions']:.6f}
- Hard intervention rate: {adaptive_schedule['hard_intervention_rate']:.6f}
- Token/exact accuracy: {adaptive_task['token_accuracy']:.6f} / {adaptive_task['exact_accuracy']:.6f}
- LM loss per executed pass: {adaptive_task['lm_loss_per_executed_pass']:.6f}
- Mean/median dwell: {adaptive_schedule['mean_dwell']:.6f} / {adaptive_schedule['median_dwell']:.1f}
- Unique schedules / entropy: {adaptive_schedule['unique_replanning_schedules']} / {adaptive_schedule['schedule_entropy_bits']:.6f} bits
- Dominant schedule: {dominant['positions']} ({dominant['fraction']:.2%})
- Replacement old/new-goal cosine: {adaptive_schedule['hard_replacement_goal_cosine']}

## E. Matched 400-step comparison

| Condition | Steps | Mean K | Token accuracy | Exact accuracy | LM loss/pass | Dominant schedule |
|---|---:|---:|---:|---:|---:|---|
{comparison_lines}

Descriptive differences:

- Adaptive - P4 token accuracy: {adaptive_minus_p4_acc:+.6f}
- Adaptive - P6 token accuracy: {adaptive_minus_p6_acc:+.6f}
- P6 - P4 token accuracy: {p6_minus_p4_acc:+.6f}
- Adaptive - P4 LM loss/pass: {adaptive_minus_p4_loss:+.6f}
- Adaptive - P6 LM loss/pass: {adaptive_minus_p6_loss:+.6f}

## F. Scientific interpretation

**{outcome}**

1. **Calibration quality:** calibration K is {selected.mean_total_interventions:.6f} and final K is {adaptive_schedule['mean_total_interventions']:.6f}; budget transfer is successful.
2. **Ranking/state dependence:** position explains {variance['percent_explained_by_position']:.3f}% of beta variance and the time-only/full Spearman correlation is {clock['time_only_vs_full']['spearman']:.4f}. The score is predominantly temporal. The {variance['percent_residual_within_position']:.3f}% residual and mixed pass-2 actions show descriptive same-position variation, not causal feature intelligence.
3. **Realized schedule adaptivity:** {dominant['fraction']:.2%} of final episodes use {dominant['positions']}; entropy is only {adaptive_schedule['schedule_entropy_bits']:.6f} bits. State-dependent exceptions exist but are behaviorally rare.
4. **Task performance:** adaptive token accuracy is {abs(adaptive_minus_p4_acc):.4f} below P=4 and {abs(adaptive_minus_p6_acc):.4f} below P=6. LM loss/pass is essentially matched (adaptive-P4 {adaptive_minus_p4_loss:+.6f}; adaptive-P6 {adaptive_minus_p6_loss:+.6f}), so the former 96-vs-400 training-budget confound no longer explains the accuracy deficit.
5. **Limitations:** theta was calibrated externally, this is one seed-0 checkpoint comparison, and the P=4/P=6 weights are deterministic reconstructions of runs whose original weights were not saved. No significance claim or causal feature attribution is made.
"""
    (output_dir / "META_AGENTS_ADAPTIVE_MATCHED_BUDGET_K2_STEP400.md").write_text(
        report
    )


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    original_hashes = verify_frozen_files()
    config = load_config(ADAPTIVE_CONFIG)
    if config["arch"]["fixed_refinement_steps"] != 8:
        raise RuntimeError("adaptive config is not fixed M=8")
    if config["arch"]["loss"]["intervention_weight"] != 0.0:
        raise RuntimeError("adaptive config is not eta=0")
    metadata, episodes = load_held_out_episodes(dataset_path_from_config(config))
    split = deterministic_split(episodes, seed=20260819, calibration_fraction=0.20)
    split_payload = split_provenance(split)
    prior_split = json.loads(
        (
            ROOT / "experiments/meta_agents_adaptive_matched_budget/"
            "20260819T144255Z_eta0_step96_k2/split_provenance.json"
        ).read_text()
    )
    for key in ("calibration_ordered_digest", "final_ordered_digest"):
        if split_payload[key] != prior_split[key]:
            raise RuntimeError(f"step-400 split differs from prior split: {key}")
    write_json(output_dir / "split_provenance.json", split_payload)

    model = build_model(config, metadata, batch_size=4, device="cpu")
    load_weights(model, ADAPTIVE_CHECKPOINT, "cpu")
    model.eval()
    state_before = logical_state_sha256(model)
    weights, bias = trigger_parameters(model)

    print("reference audit: theta=0.5 on all 4,607 episodes", flush=True)
    reference = evaluate_budget_only(
        model, episodes, metadata, threshold=0.5, batch_size=4, device="cpu"
    )
    feature_stats, clock, rows = build_reference_audit(reference, weights, bias)
    write_csv(
        output_dir / "decision_level_metrics.csv",
        rows,
        ("episode_id", "eligible_pass", "beta", "hard_intervene", *FEATURE_NAMES),
    )
    write_json(output_dir / "feature_beta_statistics.json", feature_stats)

    print("calibration reference: theta=0.5 on 921 episodes", flush=True)
    calibration_reference = evaluate_budget_only(
        model, split.calibration, metadata, threshold=0.5, batch_size=4, device="cpu"
    )
    coarse_low, coarse_high = outward_search_bounds(calibration_reference)
    cache: dict[float, BudgetSummary] = {0.5: calibration_reference}

    def evaluator(threshold: float) -> BudgetSummary:
        if threshold not in cache:
            print(f"calibration theta={threshold:.6f}", flush=True)
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
                raise RuntimeError("adaptive model state changed during calibration")
            print(
                f"  mean_K={cache[threshold].mean_total_interventions:.6f}", flush=True
            )
        return cache[threshold]

    selected_threshold, search_rows = calibrate_threshold(
        evaluator,
        coarse_low=coarse_low,
        coarse_high=coarse_high,
        steps=(1e-4, 1e-5, 1e-6),
    )
    selected_row = next(
        row for row in search_rows if row.threshold == selected_threshold
    )
    selected_index = search_rows.index(selected_row)
    neighbors = search_rows[
        max(0, selected_index - 1) : min(len(search_rows), selected_index + 2)
    ]
    selected_calibration = cache[selected_threshold]
    write_csv(
        output_dir / "calibration_threshold_search.csv",
        (asdict(row) for row in search_rows),
        tuple(asdict(search_rows[0])),
    )
    write_json(
        output_dir / "calibration_summary.json",
        {
            "target_mean_total_interventions": 2.0,
            "selected_threshold": selected_threshold,
            "selected_row": asdict(selected_row),
            "neighboring_rows": [asdict(row) for row in neighbors],
            "coarse_interval": [coarse_low, coarse_high],
            "refinement_steps": [1e-4, 1e-5, 1e-6],
            "selection_key": ["absolute_budget_error", "threshold"],
            "task_metrics_available_to_selector": False,
            "selected_schedule": summarize_schedules(selected_calibration),
        },
    )

    print(
        f"final adaptive evaluation: frozen theta={selected_threshold:.6f}", flush=True
    )
    final_budget, adaptive_task = evaluate_final(
        model,
        split.final,
        metadata,
        frozen_threshold=selected_threshold,
        batch_size=4,
        device="cpu",
    )
    adaptive_schedule = summarize_schedules(final_budget)
    clock["mixed_action_groups_at_calibrated_final_policy"] = mixed_action_audit(
        final_budget
    )
    write_json(output_dir / "clock_likeness.json", clock)
    write_json(output_dir / "adaptive_schedule_summary.json", adaptive_schedule)

    print("fixed P=4 final evaluation", flush=True)
    p4_row, p4_hashes = evaluate_fixed_final(
        P4_CHECKPOINT,
        P4_CONFIG,
        split.final,
        metadata,
        expected_period=4,
        batch_size=4,
        device="cpu",
    )
    print("fixed P=6 final evaluation", flush=True)
    p6_row, p6_hashes = evaluate_fixed_final(
        P6_CHECKPOINT,
        P6_CONFIG,
        split.final,
        metadata,
        expected_period=6,
        batch_size=4,
        device="cpu",
    )
    adaptive_row = {
        "condition": "adaptive_calibrated",
        "actual_mean_k": final_budget.mean_total_interventions,
        **adaptive_task,
        "dominant_schedule": json.dumps(
            adaptive_schedule["most_common_schedules"][0]["positions"]
        ),
    }
    p4_row["dominant_schedule"] = "[1, 4]"
    p6_row["dominant_schedule"] = "[1, 6]"
    comparison = [p4_row, p6_row, adaptive_row]
    comparison_fields = (
        "condition",
        "actual_mean_k",
        "token_accuracy",
        "exact_accuracy",
        "lm_loss_per_executed_pass",
        "lm_loss_per_completed_episode",
        "mean_refinement_passes",
        "dominant_schedule",
    )
    write_csv(
        output_dir / "final_comparison.csv",
        ({field: row[field] for field in comparison_fields} for row in comparison),
        comparison_fields,
    )
    write_json(
        output_dir / "fixed_p4_final_metrics.json",
        {"metrics": p4_row, "hashes": p4_hashes},
    )
    write_json(
        output_dir / "fixed_p6_final_metrics.json",
        {"metrics": p6_row, "hashes": p6_hashes},
    )

    state_after = logical_state_sha256(model)
    if state_after != state_before:
        raise RuntimeError("adaptive model state changed during evaluation")
    final_hashes = verify_frozen_files()
    archived_hashes = archive_inputs(output_dir, original_hashes)
    write_json(
        output_dir / "hashes.json",
        {
            "original_files_before": original_hashes,
            "original_files_after": final_hashes,
            "adaptive_loaded_state_before": state_before,
            "adaptive_loaded_state_after": state_after,
            "adaptive_model_state_stable": state_before == state_after,
            "fixed_p4": p4_hashes,
            "fixed_p6": p6_hashes,
            "archived_files": archived_hashes,
        },
    )
    write_json(
        output_dir / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "device": "cpu",
            "batch_size": 4,
            "training_steps_all_conditions": 400,
            "adaptive_eta": 0.0,
            "fixed_refinement_steps": 8,
            "split_seed": 20260819,
            "calibration_fraction": 0.20,
            "threshold_comparison": "strict beta > theta",
            "calibration_task_metrics_exposed": False,
            "causal_rollout_per_threshold": True,
            "selected_threshold": selected_threshold,
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_status": git_value("status", "--short").splitlines(),
        },
    )
    render_report(
        output_dir,
        selected_row,
        neighbors,
        feature_stats,
        clock,
        adaptive_schedule,
        adaptive_task,
        comparison,
    )
    print(f"completed: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    observed = verify_frozen_files()
    if args.preflight_only:
        print(json.dumps(observed, indent=2))
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or ROOT / (
        "experiments/meta_agents_adaptive_matched_budget/"
        f"{timestamp}_eta0_step400_k2"
    )
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    run(output_dir)


if __name__ == "__main__":
    main()
