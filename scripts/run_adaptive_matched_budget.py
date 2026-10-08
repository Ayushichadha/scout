#!/usr/bin/env python3
"""Calibrate frozen adaptive replanning to K~=2 without task-metric leakage.

Calibration and final evaluation are intentionally separate.  Calibration
only sees :class:`BudgetSummary`; the final evaluator is the only code path
that computes task metrics.  Every threshold candidate is evaluated by a new
causal rollout from pass one.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
HRM_ROOT = REPO_ROOT / "HRM"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HRM_ROOT))

from models.losses import accumulate_episode_metrics  # noqa: E402
from puzzle_dataset import IGNORE_LABEL_ID, PuzzleDatasetMetadata  # noqa: E402
from scripts.eval_checkpoints_local import (  # noqa: E402
    build_model,
    dataset_path_from_config,
    load_config,
    load_weights,
)


SCHEMA_VERSION = "meta_agents_adaptive_matched_budget_k2_v1"
DEFAULT_CHECKPOINT = REPO_ROOT / (
    "experiments/meta_agents_adaptive_sanity/"
    "20260816T144122Z_adaptive_eta0_seed0_96step_sanity/checkpoints/step_96"
)
DEFAULT_CONFIG = DEFAULT_CHECKPOINT.parents[1] / "resolved_config.yaml"
AUDITED_CHECKPOINT_SHA256 = (
    "12a581d53efd1dc9335e6eca9e986186750528442759fc26ef6a7a5b7d1e0e28"
)
AUDITED_CONFIG_SHA256 = (
    "f226fc0875777eafaf20851bcabc279a01caffa286cba864b9597ffaa2374b17"
)
AUDITED_STATE_SHA256 = (
    "a4971ef9fc45739acd99870bcdd70559842d79890488ae27e596140050b518a3"
)
FEATURE_NAMES = ("c", "d", "rho", "dwell", "q", "gate")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_state_sha256(module: torch.nn.Module) -> str:
    """Stable logical state hash, including names, dtypes, shapes, and values.

    NumPy cannot serialize bfloat16 directly.  Its values are converted to
    float32 only for byte serialization; the original dtype remains in the
    metadata, so bfloat16 and float32 states cannot collide by construction.
    """

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(cpu.dtype),
                "shape": list(cpu.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        array = cpu.float().numpy() if cpu.dtype == torch.bfloat16 else cpu.numpy()
        payload = array.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class Episode:
    episode_id: str
    set_name: str
    original_index: int
    inputs: np.ndarray
    labels: np.ndarray
    puzzle_identifier: int
    digest: str


def episode_digest(
    set_name: str,
    original_index: int,
    inputs: np.ndarray,
    labels: np.ndarray,
    puzzle_identifier: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(set_name.encode())
    digest.update(original_index.to_bytes(8, "little"))
    digest.update(np.ascontiguousarray(inputs).tobytes())
    digest.update(np.ascontiguousarray(labels).tobytes())
    digest.update(int(puzzle_identifier).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def load_held_out_episodes(
    dataset_path: Path,
) -> tuple[PuzzleDatasetMetadata, list[Episode]]:
    metadata_path = dataset_path / "test/dataset.json"
    metadata = PuzzleDatasetMetadata(**json.loads(metadata_path.read_text()))
    episodes: list[Episode] = []
    global_index = 0
    for set_name in metadata.sets:
        prefix = dataset_path / "test" / set_name
        inputs = np.load(f"{prefix}__inputs.npy", mmap_mode="r")
        labels = np.load(f"{prefix}__labels.npy", mmap_mode="r")
        puzzle_ids = np.load(f"{prefix}__puzzle_identifiers.npy", mmap_mode="r")
        puzzle_indices = np.load(f"{prefix}__puzzle_indices.npy")
        for local_index in range(len(inputs)):
            puzzle_index = int(
                np.searchsorted(puzzle_indices, local_index, side="right") - 1
            )
            puzzle_identifier = int(puzzle_ids[puzzle_index])
            episode_id = f"{set_name}:{global_index:08d}"
            digest = episode_digest(
                set_name,
                global_index,
                inputs[local_index],
                labels[local_index],
                puzzle_identifier,
            )
            episodes.append(
                Episode(
                    episode_id=episode_id,
                    set_name=set_name,
                    original_index=global_index,
                    inputs=inputs[local_index],
                    labels=labels[local_index],
                    puzzle_identifier=puzzle_identifier,
                    digest=digest,
                )
            )
            global_index += 1
    return metadata, episodes


@dataclass(frozen=True)
class Split:
    calibration: tuple[Episode, ...]
    final: tuple[Episode, ...]
    seed: int
    calibration_fraction: float


def deterministic_split(
    episodes: Sequence[Episode], *, seed: int, calibration_fraction: float
) -> Split:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be strictly between 0 and 1")
    if len(episodes) < 2:
        raise ValueError("at least two episodes are required")
    calibration_size = int(round(len(episodes) * calibration_fraction))
    calibration_size = min(max(calibration_size, 1), len(episodes) - 1)
    permutation = np.random.Generator(np.random.Philox(seed)).permutation(len(episodes))
    calibration_members = set(int(value) for value in permutation[:calibration_size])
    calibration = tuple(
        episode
        for index, episode in enumerate(episodes)
        if index in calibration_members
    )
    final = tuple(
        episode
        for index, episode in enumerate(episodes)
        if index not in calibration_members
    )
    calibration_ids = {episode.episode_id for episode in calibration}
    final_ids = {episode.episode_id for episode in final}
    if calibration_ids & final_ids:
        raise RuntimeError("calibration/final episode overlap")
    return Split(calibration, final, seed, calibration_fraction)


def ordered_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass
class EpisodeTrace:
    episode_id: str
    positions: list[int] = field(default_factory=list)
    betas: list[float] = field(default_factory=list)
    eligible_passes: list[int] = field(default_factory=list)
    hard: list[bool] = field(default_factory=list)
    features: list[list[float]] = field(default_factory=list)
    dwells: list[int] = field(default_factory=list)
    replacement_goal_cosines: list[float] = field(default_factory=list)

    def validate(self) -> None:
        if not self.positions or self.positions[0] != 1:
            raise RuntimeError(f"{self.episode_id}: forced pass-1 emission missing")
        if any(position >= 8 for position in self.positions):
            raise RuntimeError(
                f"{self.episode_id}: terminal/post-terminal intervention"
            )
        if self.eligible_passes != [2, 3, 4, 5, 6, 7]:
            raise RuntimeError(
                f"{self.episode_id}: eligible decisions are not passes 2-7"
            )
        if sum(self.dwells) != 7:
            raise RuntimeError(f"{self.episode_id}: dwell sum is not seven")
        if len(self.dwells) != len(self.positions):
            raise RuntimeError(
                f"{self.episode_id}: dwell segments != total interventions"
            )


@dataclass(frozen=True)
class BudgetSummary:
    threshold: float
    episodes: int
    eligible_decisions: int
    total_interventions: int
    adaptive_interventions: int
    beta_sum: float
    traces: tuple[EpisodeTrace, ...] = field(compare=False, repr=False)

    @property
    def mean_total_interventions(self) -> float:
        return self.total_interventions / self.episodes

    @property
    def mean_adaptive_interventions(self) -> float:
        return self.adaptive_interventions / self.episodes

    @property
    def hard_intervention_rate(self) -> float:
        return self.adaptive_interventions / self.eligible_decisions

    @property
    def mean_beta(self) -> float:
        return self.beta_sum / self.eligible_decisions

    def search_row(self, target_k: float) -> "BudgetSearchRow":
        return BudgetSearchRow(
            threshold=self.threshold,
            episodes=self.episodes,
            eligible_decisions=self.eligible_decisions,
            mean_total_interventions=self.mean_total_interventions,
            mean_adaptive_interventions=self.mean_adaptive_interventions,
            hard_intervention_rate=self.hard_intervention_rate,
            mean_beta=self.mean_beta,
            absolute_budget_error=abs(self.mean_total_interventions - target_k),
        )


@dataclass(frozen=True)
class BudgetSearchRow:
    threshold: float
    episodes: int
    eligible_decisions: int
    mean_total_interventions: float
    mean_adaptive_interventions: float
    hard_intervention_rate: float
    mean_beta: float
    absolute_budget_error: float


class BudgetEvaluator(Protocol):
    def __call__(self, threshold: float) -> BudgetSummary: ...


def decimal_grid(low: float, high: float, step: float) -> list[float]:
    scale = 10 ** max(0, len(f"{step:.12f}".rstrip("0").split(".")[-1]))
    start = round(low * scale)
    stop = round(high * scale)
    stride = round(step * scale)
    return [value / scale for value in range(start, stop + 1, stride)]


def calibrate_threshold(
    evaluator: BudgetEvaluator,
    *,
    target_k: float = 2.0,
    coarse_low: float = 0.4984,
    coarse_high: float = 0.4992,
    steps: Sequence[float] = (1e-4, 1e-5),
) -> tuple[float, list[BudgetSearchRow]]:
    """Select theta using intervention-budget fields only.

    Each unique threshold invokes ``evaluator`` exactly once.  Refinement uses
    only neighboring evaluated thresholds and never reuses beta trajectories.
    """

    evaluated: dict[float, BudgetSearchRow] = {}
    low, high = coarse_low, coarse_high
    for step in steps:
        for threshold in decimal_grid(low, high, step):
            if threshold not in evaluated:
                summary = evaluator(threshold)
                if summary.threshold != threshold:
                    raise RuntimeError(
                        "budget evaluator returned a different threshold"
                    )
                evaluated[threshold] = summary.search_row(target_k)
        ordered = sorted(evaluated)
        best = min(
            evaluated.values(),
            key=lambda row: (row.absolute_budget_error, row.threshold),
        )
        index = ordered.index(best.threshold)
        low = ordered[max(0, index - 1)]
        high = ordered[min(len(ordered) - 1, index + 1)]
    rows = [evaluated[threshold] for threshold in sorted(evaluated)]
    selected = min(rows, key=lambda row: (row.absolute_budget_error, row.threshold))
    return selected.threshold, rows


def _batch_episodes(
    episodes: Sequence[Episode],
    *,
    batch_size: int,
    metadata: PuzzleDatasetMetadata,
) -> Iterable[tuple[dict[str, torch.Tensor], Sequence[Episode]]]:
    for start in range(0, len(episodes), batch_size):
        valid = episodes[start : start + batch_size]
        inputs = np.full(
            (batch_size, metadata.seq_len), metadata.pad_id, dtype=np.int32
        )
        labels = np.full(
            (batch_size, metadata.seq_len), IGNORE_LABEL_ID, dtype=np.int32
        )
        puzzle_identifiers = np.full(
            batch_size, metadata.blank_identifier_id, dtype=np.int32
        )
        for index, episode in enumerate(valid):
            inputs[index] = episode.inputs.astype(np.int32, copy=False)
            raw_labels = episode.labels.astype(np.int32, copy=True)
            if metadata.ignore_label_id is not None:
                raw_labels[raw_labels == metadata.ignore_label_id] = IGNORE_LABEL_ID
            labels[index] = raw_labels
            puzzle_identifiers[index] = episode.puzzle_identifier
        yield {
            "inputs": torch.from_numpy(inputs),
            "labels": torch.from_numpy(labels),
            "puzzle_identifiers": torch.from_numpy(puzzle_identifiers),
        }, valid


def _set_threshold(loss_model: torch.nn.Module, threshold: float) -> None:
    core = getattr(loss_model, "model", loss_model)
    if core.adaptive_trigger is None:
        raise RuntimeError("adaptive checkpoint reconstructed without adaptive trigger")
    core.config.subgoal_head.trigger_threshold = float(threshold)


def evaluate_budget_only(
    loss_model: torch.nn.Module,
    episodes: Sequence[Episode],
    metadata: PuzzleDatasetMetadata,
    *,
    threshold: float,
    batch_size: int,
    device: str,
) -> BudgetSummary:
    """Run causal trajectories without computing task metrics."""

    _set_threshold(loss_model, threshold)
    core = getattr(loss_model, "model", loss_model)
    core.eval()
    traces: list[EpisodeTrace] = []
    beta_sum = 0.0
    eligible_count = 0
    adaptive_count = 0
    with torch.inference_mode():
        for batch, valid in _batch_episodes(
            episodes, batch_size=batch_size, metadata=metadata
        ):
            batch = {name: value.to(device) for name, value in batch.items()}
            carry = core.initial_carry(batch)
            local = [EpisodeTrace(episode.episode_id) for episode in valid]
            for pass_index in range(1, 9):
                carry, outputs = core(carry, batch)
                updated = outputs["subgoal_updated"].bool().cpu()
                eligible = outputs["adaptive_trigger_eligible"].bool().cpu()
                hard = outputs["adaptive_trigger_hard"].bool().cpu()
                beta = outputs["adaptive_trigger_probability"].float().cpu()
                feature = outputs["adaptive_trigger_feature"].float().cpu()
                dwell = outputs["adaptive_completed_dwell"].long().cpu()
                dwell_mask = outputs["adaptive_completed_dwell_mask"].bool().cpu()
                cosine = outputs["adaptive_old_new_goal_cosine"].float().cpu()
                for index, trace in enumerate(local):
                    if updated[index]:
                        trace.positions.append(pass_index)
                    if eligible[index]:
                        trace.eligible_passes.append(pass_index)
                        trace.betas.append(float(beta[index]))
                        trace.hard.append(bool(hard[index]))
                        trace.features.append(feature[index].tolist())
                        beta_sum += float(beta[index])
                        eligible_count += 1
                        adaptive_count += int(hard[index])
                        if hard[index]:
                            trace.replacement_goal_cosines.append(float(cosine[index]))
                    if dwell_mask[index]:
                        trace.dwells.append(int(dwell[index]))
            for trace in local:
                trace.validate()
            traces.extend(local)
    if eligible_count != 6 * len(episodes):
        raise RuntimeError("eligible-decision count invariant failed")
    return BudgetSummary(
        threshold=float(threshold),
        episodes=len(episodes),
        eligible_decisions=eligible_count,
        total_interventions=len(episodes) + adaptive_count,
        adaptive_interventions=adaptive_count,
        beta_sum=beta_sum,
        traces=tuple(traces),
    )


def summarize_schedules(summary: BudgetSummary) -> dict[str, object]:
    position_histogram = Counter(
        position for trace in summary.traces for position in trace.positions
    )
    dwell_histogram = Counter(
        dwell for trace in summary.traces for dwell in trace.dwells
    )
    total_k_histogram = Counter(len(trace.positions) for trace in summary.traces)
    schedules = Counter(tuple(trace.positions) for trace in summary.traces)
    total = sum(schedules.values())
    entropy = -sum(
        (count / total) * math.log2(count / total) for count in schedules.values()
    )
    all_dwells = [dwell for trace in summary.traces for dwell in trace.dwells]
    features: dict[str, dict[str, object]] = {}
    for action_name, desired in (("intervene", True), ("retain", False)):
        rows = [
            feature
            for trace in summary.traces
            for feature, hard in zip(trace.features, trace.hard)
            if hard is desired
        ]
        features[action_name] = {
            "count": len(rows),
            "means": (
                dict(zip(FEATURE_NAMES, np.asarray(rows).mean(axis=0).tolist()))
                if rows
                else None
            ),
        }
    beta_by_pass = {}
    for position in range(2, 8):
        values = [
            beta
            for trace in summary.traces
            for pass_index, beta in zip(trace.eligible_passes, trace.betas)
            if pass_index == position
        ]
        beta_by_pass[str(position)] = {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    goal_cosines = [
        value for trace in summary.traces for value in trace.replacement_goal_cosines
    ]
    return {
        "threshold": summary.threshold,
        "episodes": summary.episodes,
        "eligible_decisions": summary.eligible_decisions,
        "mean_total_interventions": summary.mean_total_interventions,
        "mean_adaptive_interventions": summary.mean_adaptive_interventions,
        "hard_intervention_rate": summary.hard_intervention_rate,
        "mean_beta": summary.mean_beta,
        "intervention_position_histogram": {
            str(position): position_histogram[position] for position in range(1, 8)
        },
        "adaptive_intervention_frequency_by_position": {
            str(position): position_histogram[position] / summary.episodes
            for position in range(2, 8)
        },
        "total_intervention_count_distribution": dict(
            sorted(total_k_histogram.items())
        ),
        "mean_dwell": float(np.mean(all_dwells)),
        "median_dwell": float(np.median(all_dwells)),
        "dwell_histogram": dict(sorted(dwell_histogram.items())),
        "unique_replanning_schedules": len(schedules),
        "schedule_entropy_bits": entropy,
        "most_common_schedules": [
            {"positions": list(schedule), "count": count, "fraction": count / total}
            for schedule, count in schedules.most_common(10)
        ],
        "beta_by_eligible_pass": beta_by_pass,
        "conditional_feature_means": features,
        "hard_replacement_goal_cosine": {
            "count": len(goal_cosines),
            "mean": float(np.mean(goal_cosines)) if goal_cosines else None,
        },
    }


def evaluate_final(
    loss_model: torch.nn.Module,
    episodes: Sequence[Episode],
    metadata: PuzzleDatasetMetadata,
    *,
    frozen_threshold: float,
    batch_size: int,
    device: str,
) -> tuple[BudgetSummary, dict[str, float]]:
    """Evaluate task metrics once after theta has been frozen."""

    _set_threshold(loss_model, frozen_threshold)
    loss_model.eval()
    metric_totals: dict[str, float] = defaultdict(float)
    with torch.inference_mode():
        for batch, _valid in _batch_episodes(
            episodes, batch_size=batch_size, metadata=metadata
        ):
            batch = {name: value.to(device) for name, value in batch.items()}
            carry = loss_model.initial_carry(batch)
            episode_metrics = None
            for _pass_index in range(1, 9):
                carry, _loss, metrics, _preds, _done = loss_model(
                    carry=carry, batch=batch, return_keys=[]
                )
                episode_metrics = accumulate_episode_metrics(episode_metrics, metrics)
            assert episode_metrics is not None
            for name, value in episode_metrics.items():
                metric_totals[name] += float(value.cpu())
    count = metric_totals["count"]
    if count != len(episodes):
        raise RuntimeError("final task-metric denominator invariant failed")
    # The model reports physical batch width for executed passes, including
    # padding in the final batch.  Scientific aggregation uses valid completed
    # episodes, so the corrected denominator is exactly count * M.
    executed = count * 8
    task = {
        "token_accuracy": metric_totals["accuracy"] / count,
        "exact_accuracy": metric_totals["exact_accuracy"] / count,
        "lm_loss_per_executed_pass": metric_totals["lm_loss"] / executed,
        "lm_loss_per_completed_episode": metric_totals["lm_loss"] / count,
        "mean_refinement_passes": 8.0,
    }
    # Re-run only the accounting path so task metrics cannot contaminate its
    # implementation or threshold.  This is a verification rollout at frozen
    # theta, not threshold selection.
    budget = evaluate_budget_only(
        loss_model,
        episodes,
        metadata,
        threshold=frozen_threshold,
        batch_size=batch_size,
        device=device,
    )
    return budget, task


def evaluate_fixed_final(
    checkpoint: Path,
    config_path: Path,
    episodes: Sequence[Episode],
    metadata: PuzzleDatasetMetadata,
    *,
    expected_period: int,
    batch_size: int,
    device: str,
) -> tuple[dict[str, object], dict[str, str]]:
    """Evaluate an existing frozen fixed policy on the exact final subset."""

    config = load_config(config_path)
    arch = config["arch"]
    subgoal = arch.get("subgoal_head", {})
    if arch.get("fixed_refinement_steps") != 8:
        raise RuntimeError(f"fixed P={expected_period} checkpoint does not use M=8")
    if subgoal.get("replan_mode", "fixed") != "fixed":
        raise RuntimeError(f"P={expected_period} checkpoint is not a fixed policy")
    if int(subgoal.get("manager_period")) != expected_period:
        raise RuntimeError(
            f"fixed checkpoint/config period mismatch for P={expected_period}"
        )
    model = build_model(config, metadata, batch_size, device)
    load_weights(model, checkpoint, device)
    model.eval()
    before = logical_state_sha256(model)
    metric_totals: dict[str, float] = defaultdict(float)
    position_histogram: Counter[int] = Counter()
    with torch.inference_mode():
        for batch, valid in _batch_episodes(
            episodes, batch_size=batch_size, metadata=metadata
        ):
            batch = {name: value.to(device) for name, value in batch.items()}
            carry = model.initial_carry(batch)
            episode_metrics = None
            local_positions = [[] for _ in valid]
            for pass_index in range(1, 9):
                carry, _loss, metrics, outputs, _done = model(
                    carry=carry, batch=batch, return_keys=["subgoal_updated"]
                )
                episode_metrics = accumulate_episode_metrics(episode_metrics, metrics)
                updated = outputs["subgoal_updated"].bool().cpu()
                for index in range(len(valid)):
                    if updated[index]:
                        local_positions[index].append(pass_index)
            assert episode_metrics is not None
            for name, value in episode_metrics.items():
                metric_totals[name] += float(value.cpu())
            expected_positions = [1, expected_period]
            for positions in local_positions:
                if positions != expected_positions:
                    raise RuntimeError(
                        f"fixed P={expected_period} schedule changed: {positions}"
                    )
                position_histogram.update(positions)
    after = logical_state_sha256(model)
    if before != after:
        raise RuntimeError(f"fixed P={expected_period} model-state hash changed")
    count = metric_totals["count"]
    if count != len(episodes):
        raise RuntimeError(f"fixed P={expected_period} task denominator mismatch")
    return (
        {
            "condition": f"fixed_p{expected_period}",
            "actual_mean_k": sum(position_histogram.values()) / count,
            "token_accuracy": metric_totals["accuracy"] / count,
            "exact_accuracy": metric_totals["exact_accuracy"] / count,
            "lm_loss_per_executed_pass": metric_totals["lm_loss"] / (count * 8),
            "lm_loss_per_completed_episode": metric_totals["lm_loss"] / count,
            "mean_refinement_passes": 8.0,
            "intervention_positions": json.dumps(
                {
                    str(position): position_histogram[position]
                    for position in range(1, 8)
                },
                sort_keys=True,
            ),
        },
        {
            "checkpoint_file_sha256": sha256_file(checkpoint),
            "resolved_config_sha256": sha256_file(config_path),
            "initial_loaded_state_sha256": before,
            "final_loaded_state_sha256": after,
        },
    )


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def split_provenance(split: Split) -> dict[str, object]:
    calibration_ids = [episode.episode_id for episode in split.calibration]
    final_ids = [episode.episode_id for episode in split.final]
    return {
        "strategy": "Philox permutation assigns membership; evaluation retains original order",
        "seed": split.seed,
        "calibration_fraction": split.calibration_fraction,
        "calibration_size": len(split.calibration),
        "final_size": len(split.final),
        "overlap_count": len(set(calibration_ids) & set(final_ids)),
        "calibration_ordered_digest": ordered_digest(
            f"{episode.episode_id}:{episode.digest}" for episode in split.calibration
        ),
        "final_ordered_digest": ordered_digest(
            f"{episode.episode_id}:{episode.digest}" for episode in split.final
        ),
        "calibration_episodes": [
            {"id": episode.episode_id, "digest": episode.digest}
            for episode in split.calibration
        ],
        "final_episodes": [
            {"id": episode.episode_id, "digest": episode.digest}
            for episode in split.final
        ],
    }


def require_fixed_inputs(paths: Sequence[Path | None]) -> None:
    labels = ("P=4 checkpoint", "P=4 config", "P=6 checkpoint", "P=6 config")
    missing = [
        label for label, path in zip(labels, paths) if path is None or not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "apples-to-apples final comparison requires the exact frozen fixed inputs; "
            "missing: " + ", ".join(missing)
        )


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--adaptive-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fixed-p4-checkpoint", type=Path)
    parser.add_argument("--fixed-p4-config", type=Path)
    parser.add_argument("--fixed-p6-checkpoint", type=Path)
    parser.add_argument("--fixed-p6-config", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260819)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--adaptive-only",
        action="store_true",
        help="Produce calibration/adaptive diagnostics without claiming a fixed comparison.",
    )
    args = parser.parse_args()

    adaptive_checkpoint = args.adaptive_checkpoint.resolve()
    adaptive_config = args.adaptive_config.resolve()
    if sha256_file(adaptive_checkpoint) != AUDITED_CHECKPOINT_SHA256:
        raise RuntimeError("adaptive checkpoint hash differs from audited step_96")
    if sha256_file(adaptive_config) != AUDITED_CONFIG_SHA256:
        raise RuntimeError("adaptive resolved-config hash differs from threshold audit")
    if not args.adaptive_only:
        require_fixed_inputs(
            (
                args.fixed_p4_checkpoint,
                args.fixed_p4_config,
                args.fixed_p6_checkpoint,
                args.fixed_p6_config,
            )
        )

    config = load_config(adaptive_config)
    if config["arch"].get("fixed_refinement_steps") != 8:
        raise RuntimeError("matched-budget experiment requires M=8")
    subgoal = config["arch"].get("subgoal_head", {})
    if subgoal.get("replan_mode") != "adaptive":
        raise RuntimeError("audited checkpoint config is not adaptive")
    dataset_path = dataset_path_from_config(config)
    metadata, episodes = load_held_out_episodes(dataset_path)
    split = deterministic_split(
        episodes,
        seed=args.split_seed,
        calibration_fraction=args.calibration_fraction,
    )
    loss_model = build_model(config, metadata, args.batch_size, args.device)
    load_weights(loss_model, adaptive_checkpoint, args.device)
    loss_model.eval()
    initial_state_hash = logical_state_sha256(loss_model)

    evaluation_serial = 0

    def budget_evaluator(threshold: float) -> BudgetSummary:
        nonlocal evaluation_serial
        evaluation_serial += 1
        print(
            f"calibration candidate {evaluation_serial}: theta={threshold:.6f}",
            flush=True,
        )
        before = logical_state_sha256(loss_model)
        summary = evaluate_budget_only(
            loss_model,
            split.calibration,
            metadata,
            threshold=threshold,
            batch_size=args.batch_size,
            device=args.device,
        )
        after = logical_state_sha256(loss_model)
        if before != initial_state_hash or after != initial_state_hash:
            raise RuntimeError("model-state hash changed during calibration")
        print(
            "  "
            f"mean_K={summary.mean_total_interventions:.6f} "
            f"error={abs(summary.mean_total_interventions - 2.0):.6f}",
            flush=True,
        )
        return summary

    selected_threshold, search_rows = calibrate_threshold(budget_evaluator)
    selected_index = next(
        index
        for index, row in enumerate(search_rows)
        if row.threshold == selected_threshold
    )
    neighboring_rows = search_rows[
        max(0, selected_index - 1) : min(len(search_rows), selected_index + 2)
    ]
    selected_calibration = next(
        budget_evaluator(row.threshold)
        for row in search_rows
        if row.threshold == selected_threshold
    )
    # Freeze theta before the only final task-metric call.
    final_budget, final_task = evaluate_final(
        loss_model,
        split.final,
        metadata,
        frozen_threshold=selected_threshold,
        batch_size=args.batch_size,
        device=args.device,
    )
    final_state_hash = logical_state_sha256(loss_model)
    if final_state_hash != initial_state_hash:
        raise RuntimeError("model-state hash changed during final evaluation")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or REPO_ROOT / (
        "experiments/meta_agents_adaptive_matched_budget/" f"{timestamp}_eta0_step96_k2"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "split_provenance.json", split_provenance(split))
    write_json(
        output_dir / "hashes.json",
        {
            "checkpoint_file_sha256": sha256_file(adaptive_checkpoint),
            "resolved_config_sha256": sha256_file(adaptive_config),
            "audit_loaded_model_state_sha256_reference": AUDITED_STATE_SHA256,
            "matched_budget_initial_loaded_state_sha256": initial_state_hash,
            "matched_budget_final_loaded_state_sha256": final_state_hash,
            "model_state_stable": initial_state_hash == final_state_hash,
        },
    )
    with (output_dir / "calibration_threshold_search.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(search_rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in search_rows)
    write_json(
        output_dir / "calibration_summary.json",
        {
            "selected_threshold": selected_threshold,
            "target_mean_total_interventions": 2.0,
            "selection_fields": ["absolute_budget_error", "threshold"],
            "task_metrics_available_to_selector": False,
            "selected_threshold_neighbors": [asdict(row) for row in neighboring_rows],
            **summarize_schedules(selected_calibration),
        },
    )
    adaptive_schedule = summarize_schedules(final_budget)
    write_json(output_dir / "adaptive_schedule_summary.json", adaptive_schedule)
    adaptive_row = {
        "condition": "adaptive_calibrated_ranking",
        "actual_mean_k": final_budget.mean_total_interventions,
        **final_task,
        "intervention_positions": json.dumps(
            adaptive_schedule["intervention_position_histogram"], sort_keys=True
        ),
    }
    comparison_rows: list[dict[str, object]] = []
    fixed_hashes: dict[str, object] = {}
    if not args.adaptive_only:
        for period, checkpoint, fixed_config in (
            (4, args.fixed_p4_checkpoint, args.fixed_p4_config),
            (6, args.fixed_p6_checkpoint, args.fixed_p6_config),
        ):
            assert checkpoint is not None and fixed_config is not None
            row, condition_hashes = evaluate_fixed_final(
                checkpoint.resolve(),
                fixed_config.resolve(),
                split.final,
                metadata,
                expected_period=period,
                batch_size=args.batch_size,
                device=args.device,
            )
            comparison_rows.append(row)
            fixed_hashes[f"fixed_p{period}"] = condition_hashes
    comparison_rows.append(adaptive_row)
    with (output_dir / "final_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(adaptive_row))
        writer.writeheader()
        writer.writerows(comparison_rows)
    hashes_path = output_dir / "hashes.json"
    hashes_payload = json.loads(hashes_path.read_text())
    hashes_payload["fixed_conditions"] = fixed_hashes
    write_json(hashes_path, hashes_payload)
    write_json(
        output_dir / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "adaptive_only": args.adaptive_only,
            "checkpoint": str(adaptive_checkpoint),
            "config": str(adaptive_config),
            "device": args.device,
            "batch_size": args.batch_size,
            "threshold_comparison": "strict beta > theta",
            "threshold_search": {
                "coarse_interval": [0.4984, 0.4992],
                "steps": [0.0001, 0.00001],
                "selection_key": ["absolute_budget_error", "threshold"],
            },
            "causal_candidate_rollouts": len(search_rows) + 1,
            "final_task_evaluations": 1,
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_status": _git_value("status", "--short").splitlines(),
            "fixed_comparison_status": (
                "not_run_adaptive_only"
                if args.adaptive_only
                else "evaluated_same_final_ids"
            ),
        },
    )
    calibration_neighbors_md = "\n".join(
        f"| {row.threshold:.6f} | {row.mean_total_interventions:.6f} | "
        f"{row.absolute_budget_error:.6f} |"
        for row in neighboring_rows
    )
    common_schedule = adaptive_schedule["most_common_schedules"][0]
    report = f"""# Adaptive Matched-Budget K=2 Diagnostic

## A. Calibration result

- Calibration episodes: {len(split.calibration)}
- Selected threshold: `{selected_threshold:.6f}`
- Calibration mean total interventions: {selected_calibration.mean_total_interventions:.6f}
- Selection used intervention budget only; task metrics were unavailable to the selector.

| Threshold | Mean K | Absolute error from 2 |
|---:|---:|---:|
{calibration_neighbors_md}

## B. Final held-out result

- Final episodes: {len(split.final)}
- Adaptive actual mean K: {final_budget.mean_total_interventions:.6f}
- Token accuracy: {final_task['token_accuracy']:.6f}
- Exact accuracy: {final_task['exact_accuracy']:.6f}
- LM loss per executed pass: {final_task['lm_loss_per_executed_pass']:.6f}
- Mean/median dwell: {adaptive_schedule['mean_dwell']:.6f} / {adaptive_schedule['median_dwell']:.1f}
- Unique schedules: {adaptive_schedule['unique_replanning_schedules']}
- Schedule entropy: {adaptive_schedule['schedule_entropy_bits']:.6f} bits
- Most common schedule: {common_schedule['positions']} ({common_schedule['fraction']:.2%})
- Intervention positions: {adaptive_schedule['intervention_position_histogram']}
- Total-K distribution: {adaptive_schedule['total_intervention_count_distribution']}

{('The fixed P=4/P=6 rows are intentionally absent in adaptive-only mode.  A scientifically valid three-way comparison requires their exact frozen checkpoints evaluated on these same final episode IDs.' if args.adaptive_only else 'Frozen fixed P=4 and P=6 checkpoints were evaluated on the identical final episode IDs and are recorded in `final_comparison.csv`.')}

## C. Scientific interpretation

Calibration transfers in budget (final K differs from target by
{abs(final_budget.mean_total_interventions - 2.0):.6f}), but the primary
fixed-policy comparison is incomplete because the frozen P=4/P=6 checkpoints
are unavailable.  No claim about useful task-performance timing signal can be
made.  This remains a diagnostic of the learned replanning-score ranking, not
a final autonomously calibrated replanning policy.
"""
    (output_dir / "META_AGENTS_ADAPTIVE_MATCHED_BUDGET_K2.md").write_text(report)
    print(output_dir)


if __name__ == "__main__":
    main()
