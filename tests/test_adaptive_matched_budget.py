from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
HRM_ROOT = ROOT / "HRM"
for path in (ROOT, HRM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_adaptive_matched_budget import (  # noqa: E402
    BudgetSearchRow,
    BudgetSummary,
    Episode,
    calibrate_threshold,
    deterministic_split,
    evaluate_budget_only,
    logical_state_sha256,
)

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1  # noqa: E402


def _episode(index: int) -> Episode:
    inputs = np.asarray([index % 8, 1, 2, 3], dtype=np.int32)
    labels = np.asarray([1, 2, 3, 4], dtype=np.int32)
    return Episode(
        episode_id=f"all:{index:08d}",
        set_name="all",
        original_index=index,
        inputs=inputs,
        labels=labels,
        puzzle_identifier=index % 4,
        digest=f"digest-{index}",
    )


def _model(batch_size: int = 2) -> HierarchicalReasoningModel_ACTV1:
    torch.manual_seed(31)
    model = HierarchicalReasoningModel_ACTV1(
        {
            "batch_size": batch_size,
            "seq_len": 4,
            "vocab_size": 16,
            "num_puzzle_identifiers": 4,
            "H_cycles": 1,
            "L_cycles": 1,
            "H_layers": 1,
            "L_layers": 1,
            "hidden_size": 8,
            "expansion": 2,
            "num_heads": 2,
            "pos_encodings": "rope",
            "halt_max_steps": 100,
            "halt_exploration_prob": 0.0,
            "fixed_refinement_steps": 8,
            "forward_dtype": "float32",
            "subgoal_head": {
                "hidden_size": 8,
                "goal_dim": 8,
                "manager_period": 4,
                "gating": True,
                "detach_goals": True,
                "inject_subgoal": True,
                "use_alignment_loss": True,
                "directional_displacement": True,
                "initial_goal_only": False,
                "replan_mode": "adaptive",
                "trigger_threshold": 0.5,
                "trigger_stochastic_train": False,
            },
        }
    )
    model.eval()
    return model


def _metadata():
    return SimpleNamespace(
        seq_len=4,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
    )


def test_split_is_deterministic_order_preserving_and_disjoint():
    episodes = [_episode(index) for index in range(101)]
    first = deterministic_split(episodes, seed=20260819, calibration_fraction=0.2)
    second = deterministic_split(episodes, seed=20260819, calibration_fraction=0.2)
    assert [row.episode_id for row in first.calibration] == [
        row.episode_id for row in second.calibration
    ]
    assert [row.episode_id for row in first.final] == [
        row.episode_id for row in second.final
    ]
    assert not (
        {row.episode_id for row in first.calibration}
        & {row.episode_id for row in first.final}
    )
    assert [row.original_index for row in first.calibration] == sorted(
        row.original_index for row in first.calibration
    )
    assert [row.original_index for row in first.final] == sorted(
        row.original_index for row in first.final
    )


def test_calibration_schema_cannot_carry_task_performance_metrics():
    names = {field.name for field in fields(BudgetSearchRow)}
    assert not names & {"accuracy", "token_accuracy", "exact_accuracy", "lm_loss"}


def test_every_threshold_candidate_gets_a_distinct_causal_rerun():
    calls: list[tuple[float, int]] = []
    rollout_serial = 0

    def evaluate(threshold: float) -> BudgetSummary:
        nonlocal rollout_serial
        rollout_serial += 1
        calls.append((threshold, rollout_serial))
        mean_k = 2.0 + (threshold - 0.4988) * 1000
        episodes = 1000
        adaptive = round((mean_k - 1.0) * episodes)
        return BudgetSummary(
            threshold=threshold,
            episodes=episodes,
            eligible_decisions=episodes * 6,
            total_interventions=episodes + adaptive,
            adaptive_interventions=adaptive,
            beta_sum=0.5 * episodes * 6,
            traces=(),
        )

    selected, rows = calibrate_threshold(evaluate)
    thresholds = [threshold for threshold, _serial in calls]
    assert len(thresholds) == len(set(thresholds))
    assert len(rows) == len(calls)
    assert selected == pytest.approx(0.4988)


def test_budget_selection_is_deterministic_and_budget_only():
    def evaluate(threshold: float) -> BudgetSummary:
        episodes = 100
        adaptive = int(round((0.4992 - threshold) * 2500))
        return BudgetSummary(
            threshold=threshold,
            episodes=episodes,
            eligible_decisions=600,
            total_interventions=episodes + adaptive,
            adaptive_interventions=adaptive,
            beta_sum=299.0,
            traces=(),
        )

    first = calibrate_threshold(evaluate)
    second = calibrate_threshold(evaluate)
    assert first == second


def test_real_online_rollout_preserves_hash_timing_and_dwell_invariants():
    model = _model()
    before = logical_state_sha256(model)
    summary = evaluate_budget_only(
        model,
        [_episode(1), _episode(2), _episode(3)],
        _metadata(),
        threshold=0.5,
        batch_size=2,
        device="cpu",
    )
    after = logical_state_sha256(model)
    assert before == after
    assert summary.eligible_decisions == 18
    assert summary.total_interventions == 3
    assert summary.adaptive_interventions == 0
    for trace in summary.traces:
        assert trace.positions == [1]
        assert trace.eligible_passes == [2, 3, 4, 5, 6, 7]
        assert trace.dwells == [7]
        assert sum(trace.dwells) == 7
        assert len(trace.dwells) == len(trace.positions)


def test_threshold_is_frozen_in_model_for_each_complete_rollout():
    model = _model(batch_size=1)
    low = evaluate_budget_only(
        model,
        [_episode(1)],
        _metadata(),
        threshold=0.0,
        batch_size=1,
        device="cpu",
    )
    assert model.config.subgoal_head.trigger_threshold == pytest.approx(0.0)
    high = evaluate_budget_only(
        model,
        [_episode(1)],
        _metadata(),
        threshold=1.0,
        batch_size=1,
        device="cpu",
    )
    assert model.config.subgoal_head.trigger_threshold == pytest.approx(1.0)
    assert low.total_interventions == 7
    assert high.total_interventions == 1
