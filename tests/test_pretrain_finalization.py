"""Regression checks for final metrics and complete fixed-compute episodes."""

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HRM"))

import pretrain  # noqa: E402


@pytest.mark.parametrize("max_steps", [None, 16])
def test_epoch_budget_cannot_cut_fixed_episode_short(monkeypatch, max_steps):
    config = SimpleNamespace(
        epochs=1,
        global_batch_size=4,
        max_steps=max_steps,
        arch=SimpleNamespace(__pydantic_extra__={"fixed_refinement_steps": 8}),
    )
    metadata = SimpleNamespace(total_groups=40, mean_puzzle_examples=1)
    create_model = Mock()
    monkeypatch.setattr(pretrain, "create_model", create_model)
    with pytest.raises(ValueError, match="Effective total_steps=10"):
        pretrain.init_train_state(config, metadata, world_size=1)
    create_model.assert_not_called()


def test_effective_budget_is_valid_even_if_unused_cap_is_not(monkeypatch):
    config = SimpleNamespace(
        epochs=1,
        global_batch_size=4,
        max_steps=17,
        arch=SimpleNamespace(__pydantic_extra__={"fixed_refinement_steps": 8}),
    )
    metadata = SimpleNamespace(total_groups=64, mean_puzzle_examples=1)
    monkeypatch.setattr(
        pretrain, "create_model", lambda *_args, **_kwargs: (Mock(), [], [])
    )
    state = pretrain.init_train_state(config, metadata, world_size=1)
    assert state.total_steps == 16


@pytest.mark.parametrize("rank", [0, 1])
def test_final_evaluation_runs_after_stale_intermediate_metrics(monkeypatch, rank):
    model = torch.nn.Linear(1, 1, bias=False)
    state = SimpleNamespace(model=model, step=16, latest_eval_metrics={"old": 0.1})
    config = SimpleNamespace(final_eval=True)
    model.weight.data.fill_(2.0)
    calls = []

    def evaluate(_config, observed_state, *_args, **kwargs):
        calls.append(
            (observed_state.step, observed_state.model.weight.item(), kwargs["rank"])
        )
        assert not observed_state.model.training
        return {"test": {"accuracy": 0.9}} if rank == 0 else None

    monkeypatch.setattr(pretrain, "evaluate", evaluate)
    pretrain.evaluate_final_weights(config, state, None, None, rank=rank, world_size=2)
    assert calls == [(16, 2.0, rank)]
    if rank == 0:
        assert state.latest_eval_metrics == {"test": {"accuracy": 0.9}}


def test_disabled_final_evaluation_does_not_run(monkeypatch):
    evaluate = Mock()
    monkeypatch.setattr(pretrain, "evaluate", evaluate)
    pretrain.evaluate_final_weights(
        SimpleNamespace(final_eval=False), Mock(), None, None, rank=0, world_size=1
    )
    evaluate.assert_not_called()
