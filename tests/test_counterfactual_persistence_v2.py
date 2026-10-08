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
from models.losses import (  # noqa: E402
    ACTLossHead,
    counterfactual_local_objective,
)
from models.subgoal_head import (  # noqa: E402
    CounterfactualPersistenceCritic,
    counterfactual_critic_features,
)


def _config(batch_size: int = 1) -> dict:
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
        "fixed_refinement_steps": 8,
        "forward_dtype": "float32",
        "subgoal_head": {
            "hidden_size": 8,
            "goal_dim": 8,
            "manager_period": 3,
            "gating": True,
            "detach_goals": True,
            "inject_subgoal": True,
            "use_alignment_loss": True,
            "directional_displacement": True,
            "initial_goal_only": False,
            "replan_mode": "counterfactual_v2",
            "counterfactual_replan_cost": 0.01,
        },
    }


def _batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
    inputs = torch.arange(batch_size * 4).view(batch_size, 4) % 16
    return {
        "inputs": inputs,
        "labels": (inputs + 1) % 16,
        "puzzle_identifiers": torch.arange(batch_size) % 4,
    }


def _model(batch_size: int = 1) -> HierarchicalReasoningModel_ACTV1:
    torch.manual_seed(29)
    model = HierarchicalReasoningModel_ACTV1(_config(batch_size))
    model.eval()
    return model


def test_two_features_are_geometric_finite_and_detached():
    worker = torch.tensor([[2.0, 1.0]], requires_grad=True)
    anchor = torch.tensor([[1.0, 1.0]], requires_grad=True)
    old = torch.tensor([[1.0, 0.0]], requires_grad=True)
    new = torch.tensor([[0.0, 1.0]], requires_grad=True)
    features = counterfactual_critic_features(
        worker_repr=worker,
        active_goal=old,
        active_anchor=anchor,
        candidate_goal=new,
    )
    assert features.shape == (1, 2)
    assert features[0].tolist() == pytest.approx([1.0, 1.0])
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()
    assert not features.requires_grad

    critic = CounterfactualPersistenceCritic()
    assert critic(features).shape == (1,)
    with pytest.raises(ValueError, match="expects"):
        critic(torch.zeros(1, 3))


def test_counterfactual_mode_has_six_eligible_decisions_and_no_terminal_emission():
    model = _model()
    batch = _batch()
    carry = model.initial_carry(batch)
    positions = []
    eligible_positions = []
    with torch.no_grad():
        for pass_position in range(1, 9):
            carry, outputs = model(carry, batch)
            if outputs["subgoal_updated"].bool().item():
                positions.append(pass_position)
            if outputs["counterfactual_eligible"].bool().item():
                eligible_positions.append(pass_position)
                assert outputs["counterfactual_critic_feature"].shape[-1] == 2
                assert "counterfactual_persist_logits" in outputs
                assert "counterfactual_replan_logits" in outputs
                assert not outputs["counterfactual_persist_logits"].requires_grad
                assert not outputs["counterfactual_replan_logits"].requires_grad
            assert torch.isfinite(outputs["counterfactual_critic_score"]).all()
    assert carry.halted.item()
    assert positions == [1]  # neutral score is below the positive cost
    assert eligible_positions == [2, 3, 4, 5, 6, 7]


def test_matched_branches_share_starting_state_and_differ_only_by_commitment():
    model = _model()
    batch = _batch()
    carry = model.initial_carry(batch)
    with torch.no_grad():
        carry, _ = model(carry, batch)

    calls = []

    def capture(_module, args, kwargs):
        inner_carry, inner_batch = args
        calls.append(
            (
                inner_carry.z_H.detach().clone(),
                inner_carry.z_L.detach().clone(),
                inner_batch["inputs"].detach().clone(),
                kwargs.get("goal").detach().clone(),
                kwargs.get("gate").detach().clone(),
            )
        )

    handle = model.inner.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.no_grad():
            _carry, outputs = model(carry, batch)
    finally:
        handle.remove()

    # actual, persist, persist lookahead, replan, replan lookahead
    assert len(calls) == 5
    persist_start = calls[1]
    replan_start = calls[3]
    for index in range(3):
        assert torch.equal(persist_start[index], replan_start[index])
    assert not torch.equal(persist_start[3], replan_start[3])
    assert outputs["counterfactual_eligible"].bool().item()


def test_detached_target_updates_critic_only():
    model = _model()
    batch = _batch()
    carry = model.initial_carry(batch)
    with torch.no_grad():
        carry, _ = model(carry, batch)
    carry, outputs = model(carry, batch)

    # Use the production loss helper for the exact target computation.
    loss_head = ACTLossHead(model, loss_type="softmax_cross_entropy")
    persist = counterfactual_local_objective(
        logits=outputs["counterfactual_persist_logits"],
        q_halt_logits=outputs["counterfactual_persist_q_halt_logits"],
        q_continue_logits=outputs["counterfactual_persist_q_continue_logits"],
        continue_target=outputs["counterfactual_persist_continue_target"],
        worker_hidden=outputs["counterfactual_persist_worker_hidden"],
        goal=outputs["counterfactual_persist_goal"],
        gate=outputs["counterfactual_persist_gate"],
        anchor=outputs["counterfactual_persist_anchor"],
        labels=batch["labels"],
        loss_fn=loss_head.loss_fn,
        feudal_loss_weight=0.05,
    )
    replan = counterfactual_local_objective(
        logits=outputs["counterfactual_replan_logits"],
        q_halt_logits=outputs["counterfactual_replan_q_halt_logits"],
        q_continue_logits=outputs["counterfactual_replan_q_continue_logits"],
        continue_target=outputs["counterfactual_replan_continue_target"],
        worker_hidden=outputs["counterfactual_replan_worker_hidden"],
        goal=outputs["counterfactual_replan_goal"],
        gate=outputs["counterfactual_replan_gate"],
        anchor=outputs["counterfactual_replan_anchor"],
        labels=batch["labels"],
        loss_fn=loss_head.loss_fn,
        feudal_loss_weight=0.05,
    )
    target = (persist - replan).detach()
    assert not target.requires_grad
    torch.nn.functional.mse_loss(
        outputs["counterfactual_critic_score"], target
    ).backward()
    critic_parameters = set(model.counterfactual_critic.parameters())
    assert any(parameter.grad is not None for parameter in critic_parameters)
    assert all(
        parameter.grad is None
        for parameter in model.parameters()
        if parameter not in critic_parameters
    )


def test_loss_head_counterfactual_metrics_are_finite_and_schema_stable():
    model = _model()
    loss_model = ACTLossHead(
        model,
        loss_type="softmax_cross_entropy",
        feudal_loss_weight=0.05,
        counterfactual_critic_weight=1.0,
    )
    loss_model.eval()
    batch = _batch()
    carry = loss_model.initial_carry(batch)
    metric_keys = None
    eligible = 0
    for _ in range(8):
        carry, loss, metrics, _, _ = loss_model(
            carry=carry, batch=batch, return_keys=[]
        )
        assert torch.isfinite(loss)
        assert all(torch.isfinite(value).all() for value in metrics.values())
        if metric_keys is None:
            metric_keys = metrics.keys()
        else:
            assert metrics.keys() == metric_keys
        eligible += int(metrics["counterfactual_eligible_decisions"])
    assert eligible == 6
