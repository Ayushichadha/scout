from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_adaptive_matched_budget import (  # noqa: E402
    BudgetSummary,
    EpisodeTrace,
)
from scripts.run_adaptive_step400_audit import (  # noqa: E402
    average_ranks,
    build_reference_audit,
    decision_rows,
    outward_search_bounds,
    safe_correlations,
    verify_frozen_files,
)


def summary() -> BudgetSummary:
    traces = []
    for episode_index in range(2):
        trace = EpisodeTrace(
            episode_id=f"episode-{episode_index}",
            positions=[1],
            betas=[0.49791 + 0.00002 * index for index in range(6)],
            eligible_passes=[2, 3, 4, 5, 6, 7],
            hard=[False] * 6,
            features=[
                [
                    0.1 * episode_index,
                    0.01 * index,
                    -0.02 * index,
                    (index + 1) / 8,
                    0.001 * index,
                    0.4 + 0.01 * episode_index,
                ]
                for index in range(6)
            ],
            dwells=[7],
        )
        traces.append(trace)
    return BudgetSummary(
        threshold=0.5,
        episodes=2,
        eligible_decisions=12,
        total_interventions=2,
        adaptive_interventions=0,
        beta_sum=sum(beta for trace in traces for beta in trace.betas),
        traces=tuple(traces),
    )


def test_step400_frozen_input_hashes_match():
    observed = verify_frozen_files()
    assert set(observed) == {
        "adaptive_checkpoint",
        "p4_checkpoint",
        "p6_checkpoint",
        "p4_config",
        "p6_config",
    }


def test_reference_audit_closes_variance_and_preserves_rows():
    source = summary()
    statistics, clock, rows = build_reference_audit(
        source, np.asarray([0.01, -0.02, 0.03, 0.04, 0.0, -0.01]), -0.002
    )

    variance = statistics["variance_decomposition"]
    assert np.isclose(
        variance["total_beta_variance"],
        variance["between_position_variance"]
        + variance["within_position_residual_variance"],
    )
    assert len(rows) == 12
    assert len(decision_rows(source)) == 12
    assert clock["definition"]["time_only"].startswith("w_dwell")


def test_search_bounds_round_outward_from_observed_beta():
    low, high = outward_search_bounds(summary())
    assert low == 0.4979
    assert high == 0.4981


def test_correlations_are_stable_and_average_ties():
    values = np.asarray([3.0, 1.0, 1.0, 2.0])
    assert average_ranks(values).tolist() == [4.0, 1.5, 1.5, 3.0]
    correlations = safe_correlations(values, values)
    assert correlations == {"pearson": 1.0, "spearman": 1.0}
