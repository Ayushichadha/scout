from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from HRM.models.subgoal_head import SubgoalHead, SubgoalHeadConfig  # noqa: E402


def test_initial_state_shapes():
    cfg = SubgoalHeadConfig(hidden_size=8, goal_dim=8)
    head = SubgoalHead(cfg)
    state = head.initial_state(batch_size=3, device=torch.device("cpu"))

    assert state.step.shape == (3,)
    assert state.goal.shape == (3, 8)
    assert state.gate is not None and state.gate.shape == (3, 1)


def test_periodic_goal_updates():
    cfg = SubgoalHeadConfig(hidden_size=8, goal_dim=8, manager_period=2)
    head = SubgoalHead(cfg)
    state = head.initial_state(batch_size=1, device=torch.device("cpu"))
    z = torch.randn(1, 8)

    state, output = head(z, state)
    # First step should not trigger an update (step=1 -> not divisible by 2)
    assert output.updated.item() == 0
    first_goal = state.goal.clone()

    state, output = head(z, state)
    assert output.updated.item() == 1
    assert not torch.allclose(first_goal, state.goal)
    # Stored goal is detached from graph
    assert state.goal.requires_grad is False
