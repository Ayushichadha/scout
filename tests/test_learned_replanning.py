from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
HRM_ROOT = ROOT / "HRM"
if str(HRM_ROOT) not in sys.path:
    sys.path.insert(0, str(HRM_ROOT))

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1  # noqa: E402
from models.losses import (  # noqa: E402
    ACTLossHead,
    accumulate_episode_metrics,
    feudal_loss,
)
from models.subgoal_head import (  # noqa: E402
    AdaptiveReplanTrigger,
    adaptive_trigger_features,
)
from scripts.eval_checkpoints_local import (  # noqa: E402
    build_model as build_local_eval_model,
    load_config as load_local_eval_config,
    load_weights as load_local_eval_weights,
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
            "replan_mode": "adaptive",
            "trigger_threshold": 0.5,
            "trigger_stochastic_train": False,
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
    torch.manual_seed(17)
    model = HierarchicalReasoningModel_ACTV1(_config(batch_size))
    model.eval()
    return model


def test_six_features_are_exact_float32_finite_and_detached():
    worker = torch.tensor([[2.0, 2.0]], requires_grad=True)
    previous = torch.tensor([[1.0, 2.0]], requires_grad=True)
    anchor = torch.tensor([[1.0, 1.0]], requires_grad=True)
    goal = torch.tensor([[1.0, 0.0]], requires_grad=True)
    gate = torch.tensor([[0.25]], requires_grad=True)
    features = adaptive_trigger_features(
        worker_repr=worker,
        previous_worker_repr=previous,
        previous_cumulative=torch.tensor([0.2]),
        dwell=torch.tensor([2]),
        q_halt_logits=torch.tensor([2.0]),
        q_continue_logits=torch.tensor([-2.0]),
        active_goal=goal,
        active_gate=gate,
        active_anchor=anchor,
        max_passes=8,
    )
    expected_c = 1.0 / (2.0**0.5)
    expected_q = torch.sigmoid(torch.tensor(2.0)) - torch.sigmoid(torch.tensor(-2.0))
    assert features.shape == (1, 6)
    assert features.dtype == torch.float32
    assert not features.requires_grad
    assert features[0, 0].item() == pytest.approx(expected_c)
    assert features[0, 1].item() == pytest.approx(1.0)
    assert features[0, 2].item() == pytest.approx(expected_c - 0.2)
    assert features[0, 3].item() == pytest.approx(0.25)
    assert features[0, 4].item() == pytest.approx(expected_q.item())
    assert features[0, 5].item() == pytest.approx(0.25)

    zero = adaptive_trigger_features(
        worker_repr=anchor,
        previous_worker_repr=anchor,
        previous_cumulative=torch.zeros(1),
        dwell=torch.ones(1, dtype=torch.long),
        q_halt_logits=torch.zeros(1),
        q_continue_logits=torch.zeros(1),
        active_goal=goal,
        active_gate=gate,
        active_anchor=anchor,
        max_passes=8,
    )
    assert torch.isfinite(zero).all()
    assert zero[0, :3].tolist() == [0.0, 0.0, 0.0]


def test_trigger_initialization_and_policy_modes():
    trigger = AdaptiveReplanTrigger()
    features = torch.randn(32, 6)
    probability = trigger(features)
    assert torch.equal(probability, torch.full_like(probability, 0.5))
    _, eval_hard, _ = trigger.decide(
        features, training=False, stochastic_train=True, threshold=0.5
    )
    assert not eval_hard.any()

    torch.manual_seed(123)
    first = trigger.decide(
        features, training=True, stochastic_train=True, threshold=0.5
    )[1]
    torch.manual_seed(123)
    second = trigger.decide(
        features, training=True, stochastic_train=True, threshold=0.5
    )[1]
    assert torch.equal(first, second)


def _run_episode(model):
    batch = _batch()
    carry = model.initial_carry(batch)
    rows = []
    with torch.no_grad():
        for pass_index in range(1, 9):
            carry, outputs = model(carry, batch)
            rows.append((pass_index, carry, outputs))
    return rows


def _account(rows):
    positions = []
    adaptive = 0
    dwells = []
    for pass_index, _, outputs in rows:
        if outputs["subgoal_updated"].bool().item():
            positions.append(pass_index)
        adaptive += int(outputs.get("adaptive_trigger_hard", torch.zeros(1)).item())
        if outputs.get("adaptive_completed_dwell_mask", torch.zeros(1)).bool().item():
            dwells.append(int(outputs["adaptive_completed_dwell"].item()))
    return positions, adaptive, dwells


@pytest.mark.parametrize(
    ("weight", "bias", "positions", "dwells"),
    [
        (0.0, -1.0, [1], [7]),
        (100.0, -56.0, [1, 6], [5, 2]),
        (0.0, 1.0, [1, 2, 3, 4, 5, 6, 7], [1, 1, 1, 1, 1, 1, 1]),
    ],
)
def test_scripted_policy_accounting(weight, bias, positions, dwells):
    model = _model()
    with torch.no_grad():
        model.adaptive_trigger.linear.weight.zero_()
        model.adaptive_trigger.linear.weight[0, 3] = weight
        model.adaptive_trigger.linear.bias.fill_(bias)
    rows = _run_episode(model)
    actual_positions, adaptive, actual_dwells = _account(rows)
    assert actual_positions == positions
    assert adaptive == len(positions) - 1
    assert actual_dwells == dwells
    assert sum(actual_dwells) == 7
    assert not rows[-1][2]["adaptive_trigger_eligible"].bool().item()
    assert not rows[-1][2]["subgoal_updated"].bool().item()


def test_causality_persistence_and_adaptive_state_convention():
    model = _model()
    batch = _batch()
    carry = model.initial_carry(batch)
    with torch.no_grad():
        carry, bootstrap = model(carry, batch)
        emitted = carry.subgoal_state.clone()
        carry, retained = model(carry, batch)
    assert not retained["adaptive_trigger_hard"].bool().item()
    assert torch.equal(retained["subgoal_goal"], emitted.goal)
    assert torch.equal(retained["subgoal_gate"], emitted.gate)
    assert torch.equal(retained["subgoal_anchor"], emitted.anchor)
    assert carry.adaptive_trigger_state.dwell.item() == 1
    assert carry.adaptive_trigger_state.previous_cumulative.item() == pytest.approx(
        retained["adaptive_trigger_feature"][0, 0].item()
    )
    # On the first consuming pass previous_cumulative was zero, so rho == c.
    assert retained["adaptive_trigger_feature"][0, 2].item() == pytest.approx(
        retained["adaptive_trigger_feature"][0, 0].item()
    )
    assert not bootstrap["adaptive_trigger_eligible"].bool().item()

    replacing = _model()
    with torch.no_grad():
        replacing.adaptive_trigger.linear.bias.fill_(1.0)
        replacement_carry = replacing.initial_carry(batch)
        replacement_carry, _ = replacing(replacement_carry, batch)
        old_anchor = replacement_carry.subgoal_state.anchor.clone()
        replacement_carry, replacement = replacing(replacement_carry, batch)
    assert replacement["adaptive_trigger_hard"].bool().item()
    assert torch.equal(
        replacement_carry.subgoal_state.anchor,
        replacement["worker_hidden"].mean(dim=1),
    )
    assert not torch.equal(replacement_carry.subgoal_state.anchor, old_anchor)
    assert replacement_carry.adaptive_trigger_state.dwell.item() == 0


def test_mixed_sample_reset_clears_all_adaptive_state_and_credit():
    model = _model(batch_size=2)
    batch = _batch(batch_size=2)
    carry = model.initial_carry(batch)
    with torch.no_grad():
        carry, _ = model(carry, batch)
        carry, _ = model(carry, batch)
    assert carry.adaptive_decision_credit.pending.tolist() == [True, True]
    carry.halted = torch.tensor([True, False])
    with torch.no_grad():
        next_carry, outputs = model(carry, batch)
    state = next_carry.adaptive_trigger_state
    assert outputs["subgoal_fresh_episode"].bool().tolist() == [True, False]
    assert state.dwell[0].item() == 0
    assert state.previous_cumulative[0].item() == 0
    assert state.has_previous_worker_repr[0].item()
    assert not outputs["adaptive_credit_pending"][0].bool().item()
    for tensor in (
        state.previous_worker_repr,
        state.previous_cumulative,
        state.dwell,
        next_carry.adaptive_decision_credit.feature,
        next_carry.adaptive_decision_credit.manager_repr,
        next_carry.adaptive_decision_credit.old_goal,
        next_carry.adaptive_decision_credit.old_gate,
        next_carry.adaptive_decision_credit.old_anchor,
        next_carry.adaptive_decision_credit.candidate_anchor,
    ):
        assert not tensor.requires_grad


def _optimizer_boundary_loss(kind: str):
    model = _model()
    model.train()
    batch = _batch()
    carry = model.initial_carry(batch)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(2):
        carry, outputs = model(carry, batch)
        outputs["logits"].square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
    assert carry.adaptive_decision_credit.pending.item()
    assert not carry.adaptive_decision_credit.feature.requires_grad
    prior_feature = carry.adaptive_decision_credit.feature.clone()
    next_carry, outputs = model(carry, batch)
    if kind == "task":
        loss = outputs["logits"].square().mean()
    else:
        loss = feudal_loss(
            outputs["worker_hidden"],
            outputs["subgoal_goal"],
            gate=outputs["subgoal_gate"],
            anchor=outputs["subgoal_anchor"],
            active_mask=outputs["subgoal_active"],
            reduction="sum",
        )
    loss.backward()
    grad = model.adaptive_trigger.linear.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    # The incoming record was consumed; because pass 3 is also eligible it is
    # atomically replaced by the new decision record rather than left stale.
    assert outputs["adaptive_credit_pending"].bool().item()
    assert next_carry.adaptive_decision_credit.pending.item()
    assert torch.equal(
        next_carry.adaptive_decision_credit.feature,
        outputs["adaptive_trigger_feature"],
    )
    assert not torch.equal(next_carry.adaptive_decision_credit.feature, prior_feature)


@pytest.mark.parametrize("kind", ["task", "alignment"])
def test_real_optimizer_boundary_downstream_trigger_credit(kind):
    _optimizer_boundary_loss(kind)


def test_intervention_objective_normalization_and_gradient():
    model = _model(batch_size=2)
    loss_model = ACTLossHead(
        model,
        loss_type="softmax_cross_entropy",
        feudal_loss_weight=0.0,
        intervention_weight=1.0,
    )
    loss_model.eval()
    batch = _batch(batch_size=2)
    carry = loss_model.initial_carry(batch)
    intervention_sum = torch.zeros(())
    for _ in range(8):
        carry, _loss, metrics, _preds, _done = loss_model(
            carry=carry, batch=batch, return_keys=[]
        )
        intervention_sum = intervention_sum + metrics.get("intervention_loss", 0.0)
    effective_episode_objective = intervention_sum / 2.0
    assert effective_episode_objective.item() == pytest.approx(0.5)

    model.zero_grad()
    model.train()
    direct_carry = model.initial_carry(batch)
    direct_carry, _ = model(direct_carry, batch)
    direct_carry, eligible_outputs = model(direct_carry, batch)
    probability = eligible_outputs["adaptive_trigger_probability"]
    eligible = eligible_outputs["adaptive_trigger_eligible"]
    ((probability * eligible).sum() / 6.0).backward()
    assert model.adaptive_trigger.linear.weight.grad.norm() > 0


def test_per_pass_metric_accounting_matches_scripted_all_retain():
    model = _model()
    with torch.no_grad():
        model.adaptive_trigger.linear.bias.fill_(-1.0)
    loss_model = ACTLossHead(
        model, loss_type="softmax_cross_entropy", feudal_loss_weight=0.0
    )
    loss_model.eval()
    batch = _batch()
    carry = loss_model.initial_carry(batch)
    totals = {}
    with torch.no_grad():
        for _ in range(8):
            carry, _loss, metrics, _preds, _done = loss_model(
                carry=carry, batch=batch, return_keys=[]
            )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
    assert totals["manager_interventions"] == 1
    assert totals["adaptive_interventions"] == 0
    assert totals["eligible_decisions"] == 6
    assert totals["intervention_position_1"] == 1
    assert sum(totals[f"intervention_position_{i}"] for i in range(1, 8)) == 1
    assert totals["completed_dwell_length_7"] == 1
    assert totals["completed_dwell_sum"] == 7
    assert totals["completed_dwell_count"] == 1


def test_invalid_adaptive_configuration_is_rejected():
    config = _config()
    config["subgoal_head"]["gating"] = False
    with pytest.raises(ValueError, match="gating=true"):
        HierarchicalReasoningModel_ACTV1(config)

    config = _config()
    config["subgoal_head"]["detach_goals"] = False
    with pytest.raises(ValueError, match="detach_goals=true"):
        HierarchicalReasoningModel_ACTV1(config)


def test_fixed_mode_instantiates_no_adaptive_parameters_or_state():
    config = _config()
    config["subgoal_head"]["replan_mode"] = "fixed"
    model = HierarchicalReasoningModel_ACTV1(config)
    carry = model.initial_carry(_batch())
    assert model.adaptive_trigger is None
    assert carry.adaptive_trigger_state is None
    assert carry.adaptive_decision_credit is None
    assert not any("adaptive_trigger" in name for name, _ in model.named_parameters())


def test_replacement_preserves_one_consuming_pass_goal_and_gate_credit():
    model = _model()
    model.train()
    with torch.no_grad():
        model.adaptive_trigger.linear.bias.fill_(1.0)
    batch = _batch()
    carry = model.initial_carry(batch)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(2):
        carry, outputs = model(carry, batch)
        outputs["logits"].square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
    assert carry.adaptive_decision_credit.hard_decision.item()
    _carry, outputs = model(carry, batch)
    loss = feudal_loss(
        outputs["worker_hidden"],
        outputs["subgoal_goal"],
        gate=outputs["subgoal_gate"],
        anchor=outputs["subgoal_anchor"],
        active_mask=outputs["subgoal_active"],
        reduction="sum",
    )
    loss.backward()
    assert model.subgoal_head.goal_proj.weight.grad is not None
    assert model.subgoal_head.goal_proj.weight.grad.norm() > 0
    assert model.subgoal_head.logit_proj.weight.grad is not None
    assert model.subgoal_head.logit_proj.weight.grad.norm() > 0


def _aggregate_eight_pass_episode(model):
    loss_model = ACTLossHead(
        model, loss_type="softmax_cross_entropy", feudal_loss_weight=0.0
    )
    loss_model.eval()
    batch = _batch()
    carry = loss_model.initial_carry(batch)
    totals = None
    positions = []
    dwells = []
    return_keys = [
        "subgoal_updated",
        "adaptive_completed_dwell",
        "adaptive_completed_dwell_mask",
    ]
    with torch.no_grad():
        for pass_index in range(1, 9):
            carry, _loss, metrics, outputs, done = loss_model(
                carry=carry, batch=batch, return_keys=return_keys
            )
            assert all(torch.isfinite(value).all() for value in metrics.values())
            totals = accumulate_episode_metrics(totals, metrics)
            if outputs["subgoal_updated"].bool().item():
                positions.append(pass_index)
            if (
                outputs.get("adaptive_completed_dwell_mask", torch.zeros(1))
                .bool()
                .item()
            ):
                dwells.append(int(outputs["adaptive_completed_dwell"].item()))
            assert done == (pass_index == 8)
    assert totals is not None
    return {key: float(value) for key, value in totals.items()}, positions, dwells


@pytest.mark.parametrize(
    ("dwell_weight", "bias", "positions", "dwells", "adaptive_count"),
    [
        (0.0, -1.0, [1], [7], 0.0),
        (100.0, -56.0, [1, 6], [5, 2], 1.0),
    ],
)
def test_complete_adaptive_evaluator_accounting(
    dwell_weight, bias, positions, dwells, adaptive_count
):
    model = _model()
    with torch.no_grad():
        model.adaptive_trigger.linear.weight.zero_()
        model.adaptive_trigger.linear.weight[0, 3] = dwell_weight
        model.adaptive_trigger.linear.bias.fill_(bias)
    totals, actual_positions, actual_dwells = _aggregate_eight_pass_episode(model)
    assert actual_positions == positions
    assert actual_dwells == dwells
    assert totals["executed_refinement_passes"] == 8
    assert totals["eligible_decisions"] == 6
    assert totals["manager_interventions"] == len(positions)
    assert totals["adaptive_interventions"] == adaptive_count
    assert totals["unconsumed_terminal_emissions"] == 0
    assert totals["directional_cosine_count"] == 7


def test_complete_fixed_evaluator_schema_smoke():
    config = _config()
    config["subgoal_head"]["replan_mode"] = "fixed"
    model = HierarchicalReasoningModel_ACTV1(config)
    totals, positions, dwells = _aggregate_eight_pass_episode(model)
    assert totals["executed_refinement_passes"] == 8
    assert totals["unconsumed_terminal_emissions"] == 0
    assert totals["directional_cosine_count"] == 7
    assert positions == [1, 3, 6]
    assert dwells == []


def test_ineligible_rows_do_not_consume_or_perturb_trigger_rng():
    trigger = AdaptiveReplanTrigger()
    features = torch.zeros(3, 6)

    torch.manual_seed(321)
    state_before = torch.get_rng_state().clone()
    _, ineligible_hard, _ = trigger.decide(
        features,
        training=True,
        stochastic_train=True,
        threshold=0.5,
        eligible=torch.zeros(3, dtype=torch.bool),
    )
    assert not ineligible_hard.any()
    assert torch.equal(torch.get_rng_state(), state_before)

    _, mixed_hard, _ = trigger.decide(
        features,
        training=True,
        stochastic_train=True,
        threshold=0.5,
        eligible=torch.tensor([True, False, False]),
    )
    mixed_state = torch.get_rng_state().clone()

    torch.manual_seed(321)
    _, single_hard, _ = trigger.decide(
        features[:1],
        training=True,
        stochastic_train=True,
        threshold=0.5,
        eligible=torch.ones(1, dtype=torch.bool),
    )
    assert mixed_hard[0].item() == single_hard[0].item()
    assert torch.equal(mixed_state, torch.get_rng_state())


def test_bootstrap_and_terminal_passes_consume_no_trigger_rng():
    config = _config()
    config["subgoal_head"]["trigger_stochastic_train"] = True
    torch.manual_seed(17)
    model = HierarchicalReasoningModel_ACTV1(config)
    model.train()
    batch = _batch()
    carry = model.initial_carry(batch)

    before_bootstrap = torch.get_rng_state().clone()
    carry, _ = model(carry, batch)
    assert torch.equal(before_bootstrap, torch.get_rng_state())
    for _ in range(6):
        carry, _ = model(carry, batch)
    before_terminal = torch.get_rng_state().clone()
    carry, outputs = model(carry, batch)
    assert carry.halted.item()
    assert not outputs["adaptive_trigger_eligible"].bool().item()
    assert torch.equal(before_terminal, torch.get_rng_state())


def test_project_seed_reproduces_eligible_decision_sequence():
    def sequence(seed):
        config = _config()
        config["subgoal_head"]["trigger_stochastic_train"] = True
        torch.manual_seed(seed)
        model = HierarchicalReasoningModel_ACTV1(config)
        model.train()
        batch = _batch()
        carry = model.initial_carry(batch)
        decisions = []
        for _ in range(8):
            carry, outputs = model(carry, batch)
            if outputs["adaptive_trigger_eligible"].item():
                decisions.append(bool(outputs["adaptive_trigger_hard"].item()))
        return decisions

    assert sequence(1234) == sequence(1234)
    torch.manual_seed(1)
    first_seed_state = torch.get_rng_state().clone()
    torch.manual_seed(2)
    assert not torch.equal(first_seed_state, torch.get_rng_state())


def test_production_intervention_total_loss_and_gradient():
    config = _config(batch_size=2)
    config["subgoal_head"]["inject_subgoal"] = False
    config["subgoal_head"]["use_alignment_loss"] = False
    model = HierarchicalReasoningModel_ACTV1(config)
    loss_model = ACTLossHead(
        model,
        loss_type="softmax_cross_entropy",
        feudal_loss_weight=0.0,
        intervention_weight=1.0,
    )
    loss_model.eval()
    batch = _batch(batch_size=2)
    carry = loss_model.initial_carry(batch)
    intervention_sum = 0.0
    for _ in range(8):
        carry, loss, metrics, _outputs, _done = loss_model(
            carry=carry, batch=batch, return_keys=[]
        )
        base_loss = metrics["lm_loss"] + 0.5 * (
            metrics["q_halt_loss"] + metrics.get("q_continue_loss", 0.0)
        )
        assert (loss.detach() - base_loss).item() == pytest.approx(
            metrics["intervention_loss"].item(), abs=1e-6
        )
        (loss / 2.0).backward()
        intervention_sum += metrics["intervention_loss"].item()

    assert intervention_sum / 2.0 == pytest.approx(0.5)
    trigger = model.adaptive_trigger.linear
    assert trigger.weight.grad is not None and trigger.weight.grad.norm() > 0
    assert trigger.bias.grad is not None
    assert trigger.bias.grad.item() == pytest.approx(0.25, abs=1e-6)


def _saved_eval_config(*, adaptive: bool) -> dict:
    model_config = _config(batch_size=2)
    for key in ("batch_size", "seq_len", "vocab_size", "num_puzzle_identifiers"):
        model_config.pop(key)
    model_config["subgoal_head"]["replan_mode"] = "adaptive" if adaptive else "fixed"
    model_config["subgoal_head"]["initial_goal_only"] = not adaptive
    model_config["subgoal_head"]["trigger_threshold"] = 0.37
    model_config["subgoal_head"]["trigger_stochastic_train"] = True
    return {
        "arch": {
            "name": "hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1",
            "loss": {
                "name": "losses@ACTLossHead",
                "loss_type": "softmax_cross_entropy",
                "feudal_loss_weight": 0.1,
                "intervention_weight": 0.23 if adaptive else 0.0,
            },
            **model_config,
        },
        "data_path": "data/conceptarc-mini",
        "global_batch_size": 2,
        "seed": 19,
    }


@pytest.mark.parametrize("adaptive", [True, False])
def test_local_evaluator_reconstructs_complete_config_and_loads_strictly(
    tmp_path, adaptive
):
    saved = _saved_eval_config(adaptive=adaptive)
    config_path = tmp_path / "all_config.yaml"
    config_path.write_text(yaml.safe_dump(saved))
    reconstructed = load_local_eval_config(config_path)
    arch = reconstructed["arch"]
    subgoal = arch["subgoal_head"]
    assert arch["fixed_refinement_steps"] == 8
    assert arch["loss"]["intervention_weight"] == pytest.approx(
        0.23 if adaptive else 0.0
    )
    assert subgoal["directional_displacement"] is True
    assert subgoal["initial_goal_only"] is (not adaptive)
    assert subgoal["replan_mode"] == ("adaptive" if adaptive else "fixed")
    assert subgoal["trigger_threshold"] == pytest.approx(0.37)
    assert subgoal["trigger_stochastic_train"] is True

    metadata = SimpleNamespace(vocab_size=16, seq_len=4, num_puzzle_identifiers=4)
    source = build_local_eval_model(reconstructed, metadata, 2, "cpu")
    target = build_local_eval_model(reconstructed, metadata, 2, "cpu")
    checkpoint = tmp_path / "step_1"
    torch.save(source.state_dict(), checkpoint)
    load_local_eval_weights(target, checkpoint, "cpu")
    assert (target.model.adaptive_trigger is not None) is adaptive
    assert target.model.config.fixed_refinement_steps == 8
    assert target.model.config.subgoal_head.initial_goal_only is (not adaptive)
