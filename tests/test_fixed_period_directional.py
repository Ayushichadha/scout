from __future__ import annotations

from dataclasses import replace
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


def _config(
    *,
    batch_size: int = 2,
    manager_period: int = 3,
    fixed_refinement_steps: int | None = None,
    initial_goal_only: bool = False,
) -> dict:
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
        "fixed_refinement_steps": fixed_refinement_steps,
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
            "initial_goal_only": initial_goal_only,
        },
    }


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    inputs = torch.arange(batch_size * 4).view(batch_size, 4) % 16
    return {
        "inputs": inputs,
        "labels": (inputs + 1) % 16,
        "puzzle_identifiers": torch.arange(batch_size) % 4,
    }


def _model(
    *,
    batch_size: int = 2,
    manager_period: int = 3,
    fixed_refinement_steps: int | None = None,
    initial_goal_only: bool = False,
):
    torch.manual_seed(7)
    model = HierarchicalReasoningModel_ACTV1(
        _config(
            batch_size=batch_size,
            manager_period=manager_period,
            fixed_refinement_steps=fixed_refinement_steps,
            initial_goal_only=initial_goal_only,
        )
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
    assert carry.subgoal_credit_state.pending.tolist() == [True, True]
    assert not carry.subgoal_credit_state.manager_repr.requires_grad
    old = carry.subgoal_state.clone()

    # Only sample 0 begins a new episode on the next outer call.
    carry.halted = torch.tensor([True, False])
    with torch.no_grad():
        carry, outputs = model(carry, batch)

    state = carry.subgoal_state
    assert outputs["subgoal_fresh_episode"].bool().tolist() == [True, False]
    assert outputs["subgoal_updated"].bool().tolist() == [True, False]
    assert outputs["subgoal_credit_pending"].bool().tolist() == [False, True]
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


def _gradient_norms(loss, groups) -> dict[str, float | None]:
    flat_parameters = []
    spans = {}
    for name, parameters in groups.items():
        start = len(flat_parameters)
        flat_parameters.extend(tuple(parameters))
        spans[name] = (start, len(flat_parameters))
    grads = torch.autograd.grad(loss, tuple(flat_parameters), allow_unused=True)

    norms = {}
    for name, (start, end) in spans.items():
        used = [grad for grad in grads[start:end] if grad is not None]
        norms[name] = (
            None
            if not used
            else torch.sqrt(sum(grad.float().square().sum() for grad in used)).item()
        )
    return norms


def _alignment_from_outputs(outputs) -> torch.Tensor:
    return feudal_loss(
        outputs["worker_hidden"],
        outputs["subgoal_goal"],
        gate=outputs["subgoal_gate"],
        anchor=outputs["subgoal_anchor"],
        active_mask=outputs["subgoal_active"],
        reduction="sum",
    )


def test_6_one_step_manager_credit_and_truncation_probe():
    model = _model(batch_size=1)
    batch = _batch(batch_size=1)
    carry = model.initial_carry(batch)

    # Match the real lifecycle: call 1 emits, then backward and optimizer step
    # complete before its returned carry is consumed by call 2.
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    carry, bootstrap_outputs = model(carry, batch)
    bootstrap_outputs["logits"].square().mean().backward()
    optimizer.step()
    optimizer.zero_grad()

    emitted_goal = carry.subgoal_state.goal.clone()
    old_credit_repr = carry.subgoal_credit_state.manager_repr.clone()
    assert carry.subgoal_credit_state.pending.item()
    assert not carry.inner_carry.z_H.requires_grad
    assert not carry.inner_carry.z_L.requires_grad
    assert not carry.subgoal_state.goal.requires_grad
    assert not carry.subgoal_state.anchor.requires_grad
    assert not carry.subgoal_credit_state.manager_repr.requires_grad

    # Before-patch equivalent: suppress the separate credit record while
    # retaining the same detached recurrent commitment.
    legacy_credit = replace(
        carry.subgoal_credit_state,
        pending=torch.zeros_like(carry.subgoal_credit_state.pending),
    )
    legacy_carry = replace(carry, subgoal_credit_state=legacy_credit)
    _, legacy_outputs = model(legacy_carry, batch)
    legacy_loss = _alignment_from_outputs(legacy_outputs)
    legacy_norms = _gradient_norms(
        legacy_loss,
        {
            "L_level": model.inner.L_level.parameters(),
            "goal_proj": model.subgoal_head.goal_proj.parameters(),
            "V_L": (model.inner.V_L.weight,),
            "gate_proj": model.subgoal_head.logit_proj.parameters(),
        },
    )

    # Patched path: call 2 consumes exactly the goal emitted after call 1.
    next_carry, outputs = model(carry, batch)
    assert not outputs["subgoal_updated"].bool().item()
    assert outputs["subgoal_credit_pending"].bool().item()
    assert torch.equal(outputs["subgoal_goal"], emitted_goal)
    assert outputs["subgoal_goal"].requires_grad
    assert outputs["worker_hidden"].requires_grad
    assert not outputs["subgoal_anchor"].requires_grad
    assert not outputs["subgoal_credit_manager_repr"].requires_grad
    assert not next_carry.inner_carry.z_H.requires_grad
    assert not next_carry.inner_carry.z_L.requires_grad
    assert not next_carry.subgoal_state.anchor.requires_grad
    assert not next_carry.subgoal_credit_state.pending.item()

    loss = _alignment_from_outputs(outputs)
    assert torch.isfinite(loss)
    patched_norms = _gradient_norms(
        loss,
        {
            "L_level": model.inner.L_level.parameters(),
            "goal_proj": model.subgoal_head.goal_proj.parameters(),
            "V_L": (model.inner.V_L.weight,),
            "gate_proj": model.subgoal_head.logit_proj.parameters(),
        },
    )

    print("before", legacy_norms)
    print("after", patched_norms)
    assert legacy_norms["goal_proj"] in (None, 0.0)
    assert legacy_norms["gate_proj"] in (None, 0.0)
    for name in ("L_level", "goal_proj", "V_L", "gate_proj"):
        assert patched_norms[name] is not None and patched_norms[name] > 0

    # Call 3 emits a replacement commitment. No live graph from the older
    # commitment survives, and the detached credit representation is replaced.
    with torch.no_grad():
        newer_carry, emission_outputs = model(next_carry, batch)
    assert emission_outputs["subgoal_updated"].bool().item()
    assert newer_carry.subgoal_credit_state.pending.item()
    assert not newer_carry.subgoal_credit_state.manager_repr.requires_grad
    assert not emitted_goal.requires_grad
    assert not torch.equal(
        newer_carry.subgoal_credit_state.manager_repr, old_credit_repr
    )


@pytest.mark.parametrize(
    ("period", "initial_only", "expected_emissions", "expected_dwell"),
    [
        (1, False, [1, 2, 3, 4, 5, 6, 7], [1, 1, 1, 1, 1, 1, 1]),
        (3, False, [1, 3, 6], [2, 3, 2]),
        (4, False, [1, 4], [3, 4]),
        (6, False, [1, 6], [5, 2]),
        (1, True, [1], [7]),
    ],
)
def test_7_fixed_compute_schedules(
    period, initial_only, expected_emissions, expected_dwell
):
    fixed_steps = 8
    model = _model(
        batch_size=1,
        manager_period=period,
        fixed_refinement_steps=fixed_steps,
        initial_goal_only=initial_only,
    )
    model.train()
    # Even an overwhelming learned halt preference must not alter execution.
    with torch.no_grad():
        model.inner.q_head.bias.copy_(torch.tensor([100.0, -100.0]))

    batch = _batch(batch_size=1)
    carry = model.initial_carry(batch)
    emissions = []
    active_goal_ids = []
    dwell = []
    current_goal = None
    current_length = 0

    with torch.no_grad():
        for pass_index in range(1, fixed_steps + 1):
            carry, outputs = model(carry, batch)
            assert torch.isfinite(outputs["logits"]).all()
            if outputs["subgoal_active"].bool().item():
                active_goal = outputs["subgoal_goal"].clone()
                active_goal_ids.append(active_goal)
                if current_goal is None or torch.equal(active_goal, current_goal):
                    current_length += 1
                else:
                    dwell.append(current_length)
                    current_length = 1
                current_goal = active_goal
            if outputs["subgoal_updated"].bool().item():
                emissions.append(pass_index)
                assert outputs["subgoal_has_next_worker_pass"].bool().item()

            assert carry.steps.item() == pass_index
            assert carry.halted.item() == (pass_index == fixed_steps)

    if current_length:
        dwell.append(current_length)
    assert emissions == expected_emissions
    assert dwell == expected_dwell
    assert len(active_goal_ids) == fixed_steps - 1
    assert not outputs["subgoal_updated"].bool().item()
    assert not outputs["subgoal_has_next_worker_pass"].bool().item()

    if initial_only:
        assert all(
            torch.equal(active_goal_ids[0], goal) for goal in active_goal_ids[1:]
        )

    # The next call is a fresh episode and schedules a new initial emission.
    with torch.no_grad():
        reset_carry, reset_outputs = model(carry, batch)
    assert reset_outputs["subgoal_fresh_episode"].bool().item()
    assert reset_outputs["subgoal_updated"].bool().item()
    assert reset_carry.steps.item() == 1
