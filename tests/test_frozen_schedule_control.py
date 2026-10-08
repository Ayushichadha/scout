from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
HRM_ROOT = ROOT / "HRM"
for path in (ROOT, HRM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.eval_checkpoints_local import (  # noqa: E402
    build_model,
    dataset_path_from_config,
    load_config,
    load_weights,
)
from scripts.run_adaptive_matched_budget import (  # noqa: E402
    deterministic_split,
    load_held_out_episodes,
    sha256_file,
    split_provenance,
)
from scripts.run_frozen_schedule_control import (  # noqa: E402
    CHECKPOINT,
    CONFIG,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_SPLIT_DIGEST,
    evaluate_forced_schedule,
    threshold_for_pass,
)


def test_threshold_switch_forces_only_predeclared_second_pass():
    assert [threshold_for_pass(index, 4) for index in range(1, 9)] == [
        2.0,
        2.0,
        2.0,
        -1.0,
        2.0,
        2.0,
        2.0,
        2.0,
    ]
    with pytest.raises(ValueError, match="unsupported"):
        threshold_for_pass(3, 8)


def test_frozen_checkpoint_and_final_split_match_source_audit():
    assert sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256
    config = load_config(CONFIG)
    metadata, episodes = load_held_out_episodes(dataset_path_from_config(config))
    split = deterministic_split(episodes, seed=20260819, calibration_fraction=0.20)
    assert split_provenance(split)["final_ordered_digest"] == EXPECTED_SPLIT_DIGEST


def test_real_model_enforces_forced_schedule_without_state_change():
    config = load_config(CONFIG)
    metadata, episodes = load_held_out_episodes(dataset_path_from_config(config))
    model = build_model(config, metadata, batch_size=4, device="cpu")
    load_weights(model, CHECKPOINT, "cpu")

    row, diagnostics = evaluate_forced_schedule(
        model, episodes[:4], metadata, second_pass=4, batch_size=4, device="cpu"
    )

    assert row["actual_mean_k"] == 2.0
    assert diagnostics["expected_schedule"] == [1, 4]
    assert diagnostics["position_histogram"]["1"] == 4
    assert diagnostics["position_histogram"]["4"] == 4
    assert diagnostics["model_state_stable"] is True
