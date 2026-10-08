from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_adaptive_400step import (  # noqa: E402
    TRAINING_STEPS,
    build_overrides,
    preflight_against_96step,
    resolve_config,
)


def candidate_config(tmp_path: Path):
    return resolve_config(build_overrides(tmp_path / "run"))


def test_400step_config_matches_audited_96step_run(tmp_path: Path):
    config = candidate_config(tmp_path)

    differences = preflight_against_96step(config)

    assert set(differences) == {
        "checkpoint_path",
        "max_steps",
        "project_name",
        "run_name",
        "run_summary_path",
    }
    assert config["max_steps"] == TRAINING_STEPS
    assert config["device"] == "cpu"
    assert config["arch"]["fixed_refinement_steps"] == 8
    assert config["arch"]["loss"]["intervention_weight"] == 0.0


def test_preflight_rejects_unreviewed_scientific_change(tmp_path: Path):
    config = candidate_config(tmp_path)
    config["arch"]["subgoal_head"]["trigger_threshold"] = 0.6

    with pytest.raises(RuntimeError, match="unexpected fields"):
        preflight_against_96step(config)


def test_preflight_allows_declared_replication_seed(tmp_path: Path):
    config = resolve_config(build_overrides(tmp_path / "seed1", seed=1))

    differences = preflight_against_96step(config, seed=1)

    assert differences["seed"] == {"reference": 0, "candidate": 1}
    assert config["seed"] == 1
    assert config["run_name"] == "adaptive_eta0_seed1_400step"
