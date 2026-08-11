from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
HRM_ROOT = ROOT / "HRM"
if str(HRM_ROOT) not in sys.path:
    sys.path.insert(0, str(HRM_ROOT))

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1  # noqa: E402
from models.losses import feudal_loss  # noqa: E402


def _config(*, batch_size: int = 2, manager_period: int = 3) -> dict:
    return {
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
        "forward_dtype": "float32",
        "subgoal_head": {
            "hidden_size": 8,
            "goal_dim": 8,
            "manager_period": manager_period,
            "gating": True,
            "detach_goals": True,
            "inject_subgoal": True,
            "use_alignment_loss": True,
            "directional_displacement": True,
        },
    }


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    inputs = torch.arange(batch_size * 4).view(batch_size, 4) % 16
    return {
        "inputs": inputs,
        "labels": (inputs + 1) % 16,
        "puzzle_identifiers": torch.arange(batch_size) % 4,
    }


def _model(*, batch_size: int = 2, manager_period: int = 3):
    torch.manual_seed(7)
    model = HierarchicalReasoningModel_ACTV1(
        _config(batch_size=batch_size, manager_period=manager_period)
    )
    model.eval()
    return model


def test_1_episode_reset_is_per_sample_and_forces_emission():
    model = _model()
    batch = _batch()
    carry = model.initial_carry(batch)

    with torch.no_grad():
        carry, first = model(carry, batch)
    assert first["subgoal_updated"].bool().tolist() == [True, True]
    assert torch.all(carry.subgoal_state.goal.norm(dim=-1) > 0)
    assert torch.equal(carry.subgoal_state.anchor, first["worker_hidden"].mean(dim=1))
    old = carry.subgoal_state.clone()

    # Only sample 0 begins a new episode on the next outer call.
    carry.halted = torch.tensor([True, False])
    with torch.no_grad():
        carry, outputs = model(carry, batch)

    state = carry.subgoal_state
    assert outputs["subgoal_fresh_episode"].bool().tolist() == [True, False]
    assert outputs["subgoal_updated"].bool().tolist() == [True, False]
    assert state.step.tolist() == [1, 2]

    # The commitment consumed by fresh sample 0 was fully reset before work.
    assert torch.equal(outputs["subgoal_goal"][0], torch.zeros(8))
    assert torch.equal(outputs["subgoal_gate"][0], torch.zeros(1))
    assert torch.equal(outputs["subgoal_anchor"][0], torch.zeros(8))
    assert not outputs["subgoal_active"][0].bool()

    # Sample 1 did not reset and retains its complete commitment.
    assert torch.equal(state.goal[1], old.goal[1])
    assert torch.equal(state.gate[1], old.gate[1])
    assert torch.equal(state.anchor[1], old.anchor[1])


def test_2_period_three_timing_probe():
    model = _model(batch_size=1)
    batch = _batch(batch_size=1)
    carry = model.initial_carry(batch)
    rows = []
    previous_goal = None

    with torch.no_grad():
        for outer_call in range(1, 8):
            carry, outputs = model(carry, batch)
            goal = carry.subgoal_state.goal
            cosine = None
            if previous_goal is not None:
                cosine = torch.nn.functional.cosine_similarity(
                    goal, previous_goal, dim=-1
                ).item()
            row = (
                outer_call,
                bool(outputs["subgoal_fresh_episode"].item()),
                bool(outputs["subgoal_updated"].item()),
                int(carry.subgoal_state.step.item()),
                cosine,
            )
            rows.append(row)
            print(row)
            previous_goal = goal.clone()

    assert [row[2] for row in rows] == [True, False, True, False, False, True, False]
    assert [row[3] for row in rows] == [1, 2, 3, 4, 5, 6, 7]
    assert rows[0][1] is True and all(not row[1] for row in rows[1:])


def test_3_goal_gate_and_anchor_persist_between_emissions():
    model = _model(batch_size=1)
    batch = _batch(batch_size=1)
    carry = model.initial_carry(batch)

    with torch.no_grad():
        carry, _ = model(carry, batch)  # call 1: forced emission
        emitted = carry.subgoal_state.clone()
        carry, outputs = model(carry, batch)  # call 2: no emission

    assert not outputs["subgoal_updated"].bool().item()
    assert torch.equal(carry.subgoal_state.goal, emitted.goal)
    assert torch.equal(carry.subgoal_state.gate, emitted.gate)
    assert torch.equal(carry.subgoal_state.anchor, emitted.anchor)


def test_4_directional_loss_uses_displacement_and_zero_is_finite():
    goal = torch.tensor([[1.0, 0.0]])
    anchor = torch.tensor([[0.0, 10.0]])
    worker = torch.tensor([[2.0, 10.0]])

    directional = feudal_loss(worker, goal, anchor=anchor, reduction="none")
    absolute = feudal_loss(worker, goal, reduction="none")
    assert directional.item() == pytest.approx(0.0, abs=1e-6)
    assert absolute.item() > 0.5

    zero_delta = feudal_loss(anchor, goal, anchor=anchor, reduction="none")
    assert zero_delta.item() == pytest.approx(1.0)
    assert torch.isfinite(zero_delta).all()


def test_5_v_l_is_bias_free_injected_and_receives_task_gradient():
    model = _model(batch_size=1)
    batch = _batch(batch_size=1)
    carry = model.initial_carry(batch)
    with torch.no_grad():
        carry, _ = model(carry, batch)  # populate the first causal commitment

    assert model.inner.V_L is not None
    assert model.inner.V_L.bias is None
    assert model.inner.V_L.out_features == model.config.hidden_size

    carry_on, outputs_on = model(carry, batch)
    state = carry.subgoal_state
    expected = model.inner.V_L(state.goal) * state.gate
    assert torch.allclose(outputs_on["subgoal_goal_bias"], expected, atol=1e-6)
    assert outputs_on["subgoal_goal_bias"].shape == (1, model.config.hidden_size)

    task_loss = outputs_on["logits"].square().mean()
    (v_l_grad,) = torch.autograd.grad(task_loss, (model.inner.V_L.weight,))
    assert torch.isfinite(v_l_grad).all() and v_l_grad.norm() > 0

    model.config.subgoal_head.inject_subgoal = False
    with torch.no_grad():
        _, outputs_off = model(carry, batch)
    assert "subgoal_goal_bias" not in outputs_off
    assert not torch.allclose(outputs_on["worker_hidden"], outputs_off["worker_hidden"])


def _group_grad_norm(loss, parameters) -> float | None:
    parameters = tuple(parameters)
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    used = [grad for grad in grads if grad is not None]
    if not used:
        return None
    return torch.sqrt(sum(grad.float().square().sum() for grad in used)).item()


def _alignment_from_outputs(outputs) -> torch.Tensor:
    return feudal_loss(
        outputs["worker_hidden"],
        outputs["subgoal_goal"],
        gate=outputs["subgoal_gate"],
        anchor=outputs["subgoal_anchor"],
        active_mask=outputs["subgoal_active"],
        reduction="sum",
    )


def test_6_directional_gradient_sanity_emission_and_non_emission():
    model = _model(batch_size=1)
    batch = _batch(batch_size=1)
    carry = model.initial_carry(batch)

    with torch.no_grad():
        carry, _ = model(carry, batch)  # call 1: forced emission and anchor

    records = []
    for label in ("non_emission", "emission"):
        carry, outputs = model(carry, batch)
        assert bool(outputs["subgoal_updated"].item()) == (label == "emission")
        # Recompute per group because autograd.grad frees the graph by default.
        # The forward is tiny, and this keeps the probe explicit and isolated.
        recomputed = []
        for group_index in range(3):
            _, fresh_outputs = model(
                _rewind_for_probe(model, batch, label),
                batch,
            )
            fresh_loss = _alignment_from_outputs(fresh_outputs)
            fresh_groups = (
                tuple(model.inner.L_level.parameters()),
                tuple(model.subgoal_head.goal_proj.parameters()),
                (model.inner.V_L.weight,),
            )
            recomputed.append(_group_grad_norm(fresh_loss, fresh_groups[group_index]))
        records.append((label, recomputed))
        print(label, dict(zip(("L_level", "goal_proj", "V_L"), recomputed)))

    for _, (l_norm, goal_norm, v_l_norm) in records:
        assert l_norm is not None and l_norm > 0
        assert goal_norm is None
        assert v_l_norm is not None and v_l_norm > 0


def _rewind_for_probe(model, batch, target: str):
    """Build a detached carry immediately before call 2 or call 3."""
    with torch.no_grad():
        carry = model.initial_carry(batch)
        carry, _ = model(carry, batch)
        if target == "emission":
            carry, _ = model(carry, batch)
    return carry
