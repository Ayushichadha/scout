from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
HRM_ROOT = ROOT / "HRM"
for path in (ROOT, HRM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_frozen_schedule_control import threshold_for_pass  # noqa: E402
from scripts.run_schedule_sweep_k2 import (  # noqa: E402
    SCHEDULES,
    aggregate_rows,
    analyze,
    dwell_segments,
    schedule_positions,
)


@pytest.mark.parametrize("k", SCHEDULES)
def test_schedule_emits_only_at_one_and_k(k):
    assert schedule_positions(k) == (1, k)
    forced = [
        position
        for position in range(1, 9)
        if position == 1 or threshold_for_pass(position, k) < 0
    ]
    assert forced == [1, k]


@pytest.mark.parametrize("k", SCHEDULES)
def test_dwell_segments_cover_all_seven_transitions(k):
    assert dwell_segments(k) == (k - 1, 8 - k)
    assert sum(dwell_segments(k)) == 7


def fixture_rows():
    rows = []
    for k in SCHEDULES:
        for episode_id, correct, valid in (("a", k, 10), ("b", 8 - k, 20)):
            rows.append(
                {
                    "episode_id": episode_id,
                    "schedule_k": k,
                    "dwell_first": k - 1,
                    "dwell_second": 8 - k,
                    "tokens_correct": correct,
                    "tokens_valid": valid,
                    "token_acc_episode": correct / valid,
                    "lm_loss_sum": 8.0,
                    "passes_executed": 8,
                    "lm_loss_per_pass": 1.0,
                    "exact_correct": 0,
                }
            )
    return rows


def test_episode_rows_reaggregate_to_reported_micro():
    rows = fixture_rows()[:2]
    assert aggregate_rows(rows)["micro"] == pytest.approx(
        sum(row["tokens_correct"] for row in rows)
        / sum(row["tokens_valid"] for row in rows)
    )


def test_emitted_episode_matrix_reaggregates_to_reported_micro(require_archive_files):
    output_dir = ROOT / "experiments/schedule_sweep_k2"
    require_archive_files(
        output_dir / "per_episode_matrix.csv", output_dir / "analysis_summary.json"
    )
    with (output_dir / "per_episode_matrix.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((output_dir / "analysis_summary.json").read_text())
    for k in SCHEDULES:
        scheduled = [row for row in rows if int(row["schedule_k"]) == k]
        assert len(scheduled) == 3686
        assert aggregate_rows(scheduled)["micro"] == pytest.approx(
            summary["aggregates"][str(k)]["micro"], abs=1e-15
        )


def test_ceiling_and_floor_bound_best_fixed(monkeypatch):
    monkeypatch.setattr(
        "scripts.run_schedule_sweep_k2.bootstrap_analysis",
        lambda *_args, **_kwargs: (
            (0.0, 1.0),
            {
                pair: (0.0, 1.0)
                for pair in __import__("itertools").combinations(SCHEDULES, 2)
            },
        ),
    )
    _report, summary = analyze(fixture_rows())
    assert summary["oracle_ceiling_micro"] >= summary["best_fixed_micro"]
    assert summary["floor_micro"] <= summary["best_fixed_micro"]
