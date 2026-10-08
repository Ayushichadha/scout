from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn


class SubgoalHeadConfig(BaseModel):
    """Configuration for the latent subgoal head.

    Attributes
    ----------
    hidden_size:
        Dimension of the manager (high-level) hidden state that feeds the head.
    goal_dim:
        Dimension of the latent goal vector produced for the worker.
    manager_period:
        Number of outer ACT refinement calls between scheduled manager goal
        updates. It does not count inner H/L cycle micro-steps.
    temperature:
        Temperature used when converting logits to a distribution (if needed).
    projection_bias:
        Whether to include bias terms in the linear projections.
    normalize_goal:
        If True, project goals onto the unit hypersphere (optionally scaled by
        goal_scale) to ensure directional semantics similar to FuN.
    goal_scale:
        Optional scaling factor applied after normalization. Ignored when
        ``normalize_goal`` is False.
    gating:
        If True, also emit a scalar gate (sigmoid) that can be interpreted as
        an intrinsic value / commitment strength for the proposed goal.
    detach_goals:
        If True, returned goals are detached from the computation graph before
        being stored in the head state (maintaining the deep supervision
        contract of HRM).
    """

    hidden_size: int
    goal_dim: int
    manager_period: int = Field(default=4, ge=1)
    temperature: float = 1.0
    projection_bias: bool = True
    normalize_goal: bool = True
    goal_scale: float = 1.0
    gating: bool = True
    detach_goals: bool = True

    inject_subgoal: bool = True
    use_alignment_loss: bool = True
    random_directions: bool = False
    directional_displacement: bool = False
    initial_goal_only: bool = False
    replan_mode: Literal["fixed", "adaptive", "counterfactual_v2"] = "fixed"
    trigger_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    trigger_stochastic_train: bool = True
    counterfactual_replan_cost: float = Field(default=0.01, ge=0.0)


@dataclass
class SubgoalHeadState:
    """State tracked across deep-supervision segments."""

    step: Tensor
    goal: Tensor
    gate: Optional[Tensor]
    anchor: Tensor

    def clone(self) -> "SubgoalHeadState":
        gate = None if self.gate is None else self.gate.clone()
        return SubgoalHeadState(
            step=self.step.clone(),
            goal=self.goal.clone(),
            gate=gate,
            anchor=self.anchor.clone(),
        )

    def detach(self) -> "SubgoalHeadState":
        gate = None if self.gate is None else self.gate.detach()
        return SubgoalHeadState(
            step=self.step.detach(),
            goal=self.goal.detach(),
            gate=gate,
            anchor=self.anchor.detach(),
        )


@dataclass
class SubgoalHeadOutput:
    """Outputs emitted by the subgoal head at each manager update."""

    goal: Tensor
    gate: Optional[Tensor]
    logits: Optional[Tensor]
    probs: Optional[Tensor]
    updated: Tensor


@dataclass
class AdaptiveTriggerState:
    """Minimal detached episode-scoped observations for adaptive re-planning."""

    previous_worker_repr: Tensor
    previous_cumulative: Tensor
    dwell: Tensor
    has_previous_worker_repr: Tensor

    def detach(self) -> "AdaptiveTriggerState":
        return AdaptiveTriggerState(
            previous_worker_repr=self.previous_worker_repr.detach(),
            previous_cumulative=self.previous_cumulative.detach(),
            dwell=self.dwell.detach(),
            has_previous_worker_repr=self.has_previous_worker_repr.detach(),
        )


def _directional_cosine(displacement: Tensor, goal: Tensor) -> Tensor:
    displacement = displacement.to(torch.float32)
    goal = goal.to(torch.float32)
    return (
        F.normalize(displacement, dim=-1, eps=1e-8)
        * F.normalize(goal, dim=-1, eps=1e-8)
    ).sum(dim=-1)


def adaptive_trigger_features(
    *,
    worker_repr: Tensor,
    previous_worker_repr: Tensor,
    previous_cumulative: Tensor,
    dwell: Tensor,
    q_halt_logits: Tensor,
    q_continue_logits: Tensor,
    active_goal: Tensor,
    active_gate: Tensor,
    active_anchor: Tensor,
    max_passes: int,
) -> Tensor:
    """Build the exact detached six-feature adaptive trigger observation."""
    worker = worker_repr.to(torch.float32)
    goal = active_goal.to(torch.float32)
    cumulative = _directional_cosine(worker - active_anchor, goal)
    latest = _directional_cosine(worker - previous_worker_repr, goal)
    trend = cumulative - previous_cumulative.to(torch.float32)
    normalized_dwell = dwell.to(torch.float32) / float(max_passes)
    halt_confidence = torch.sigmoid(q_halt_logits.to(torch.float32)) - torch.sigmoid(
        q_continue_logits.to(torch.float32)
    )
    gate = active_gate.to(torch.float32).squeeze(-1)
    return torch.stack(
        (cumulative, latest, trend, normalized_dwell, halt_confidence, gate), dim=-1
    ).detach()


class AdaptiveReplanTrigger(nn.Module):
    """Interpretable affine Bernoulli policy over six detached features."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(6, 1, bias=True)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != 6:
            raise ValueError(f"adaptive trigger expects [..., 6], got {features.shape}")
        return torch.sigmoid(self.linear(features.to(torch.float32))).squeeze(-1)

    def decide(
        self,
        features: Tensor,
        *,
        training: bool,
        stochastic_train: bool,
        threshold: float,
        eligible: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        probability = self(features)
        if training and stochastic_train:
            if eligible is None:
                hard = torch.bernoulli(probability).to(torch.bool)
            else:
                eligible = eligible.to(device=probability.device, dtype=torch.bool)
                if eligible.shape != probability.shape:
                    raise ValueError(
                        "eligible mask must match trigger probability shape, got "
                        f"{tuple(eligible.shape)} and {tuple(probability.shape)}"
                    )
                eligible_hard = torch.bernoulli(probability.masked_select(eligible)).to(
                    torch.bool
                )
                hard = torch.zeros_like(eligible).masked_scatter(
                    eligible, eligible_hard
                )
        else:
            hard = probability > threshold
            if eligible is not None:
                hard = hard & eligible.to(device=hard.device, dtype=torch.bool)
        straight_through = (
            probability + (hard.to(probability.dtype) - probability).detach()
        )
        return probability, hard, straight_through


def counterfactual_critic_features(
    *,
    worker_repr: Tensor,
    active_goal: Tensor,
    active_anchor: Tensor,
    candidate_goal: Tensor,
) -> Tensor:
    """Return the two decision-time signals used by persistence v2.

    The input path is deliberately closed over only geometrically meaningful
    state available after the current pass: cumulative progress under the
    committed goal and cosine disagreement with the candidate goal.  No pass,
    dwell, or other clock variable is accepted by this function.  Detaching
    here prevents either the critic loss or its decision features from shaping
    the reasoning model or manager projections.
    """

    progress = _directional_cosine(worker_repr - active_anchor, active_goal)
    similarity = F.cosine_similarity(
        active_goal.to(torch.float32),
        candidate_goal.to(torch.float32),
        dim=-1,
        eps=1e-8,
    )
    return torch.stack((progress, 1.0 - similarity), dim=-1).detach()


class CounterfactualPersistenceCritic(nn.Module):
    """Tiny regressor for the detached replanning advantage target."""

    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        # Start with neutral evidence, so the positive deliberation cost makes
        # the initial deterministic policy persist.
        with torch.no_grad():
            self.network[-1].weight.zero_()
            self.network[-1].bias.zero_()

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != 2:
            raise ValueError(
                f"counterfactual critic expects [..., 2], got {features.shape}"
            )
        return self.network(features.to(torch.float32)).squeeze(-1)


class SubgoalHead(nn.Module):
    """Manager head that proposes directional latent subgoals.

    This module follows the FuN-style manager. Every ``manager_period`` outer
    ACT calls it proposes a normalized goal vector derived from the high-level
    hidden state. Between updates, the previously committed goal and gate are
    reused.

    The head maintains per-sample state consisting of the current commitment
    (goal, gate, and directional anchor) and outer-call counter. External
    callers are responsible for passing the state back on subsequent calls.
    """

    def __init__(self, config: SubgoalHeadConfig):
        super().__init__()
        self.cfg = config

        self.goal_proj = nn.Linear(
            config.hidden_size,
            config.goal_dim,
            bias=config.projection_bias,
        )

        self.logit_proj: Optional[nn.Linear]
        if config.gating:
            # Outputs a scalar gate / value per sample.
            self.logit_proj = nn.Linear(
                config.hidden_size,
                1,
                bias=config.projection_bias,
            )
        else:
            self.logit_proj = None

    def initial_state(self, batch_size: int, device: torch.device) -> SubgoalHeadState:
        step = torch.zeros(batch_size, dtype=torch.long, device=device)
        goal = torch.zeros(batch_size, self.cfg.goal_dim, device=device)
        gate: Optional[Tensor]
        if self.logit_proj is not None:
            gate = torch.zeros(batch_size, 1, device=device)
        else:
            gate = None
        anchor = torch.zeros(batch_size, self.cfg.hidden_size, device=device)
        return SubgoalHeadState(step=step, goal=goal, gate=gate, anchor=anchor)

    def reset_state(
        self, state: SubgoalHeadState, reset_mask: Tensor
    ) -> SubgoalHeadState:
        """Reset manager commitment state for selected ACT episodes."""
        if reset_mask.shape != state.step.shape:
            raise ValueError(
                f"reset_mask must have shape {tuple(state.step.shape)}, "
                f"got {tuple(reset_mask.shape)}"
            )
        reset_mask = reset_mask.to(device=state.step.device, dtype=torch.bool)
        reset_col = reset_mask.unsqueeze(-1)
        gate = state.gate
        if gate is not None:
            gate = torch.where(reset_col, torch.zeros_like(gate), gate)
        return SubgoalHeadState(
            step=torch.where(reset_mask, torch.zeros_like(state.step), state.step),
            goal=torch.where(reset_col, torch.zeros_like(state.goal), state.goal),
            gate=gate,
            anchor=torch.where(reset_col, torch.zeros_like(state.anchor), state.anchor),
        )

    def _compute_goal(self, z_h: Tensor) -> Tensor:
        goal = self.goal_proj(z_h)
        if self.cfg.normalize_goal:
            goal = F.normalize(goal, dim=-1, eps=1e-8)
            if self.cfg.goal_scale != 1.0:
                goal = goal * self.cfg.goal_scale
        if self.cfg.random_directions:
            goal = F.normalize(torch.randn_like(goal).detach(), dim=-1, eps=1e-8)
            if self.cfg.goal_scale != 1.0:
                goal = goal * self.cfg.goal_scale
        return goal

    def _compute_gate(self, z_h: Tensor) -> Optional[Tensor]:
        if self.logit_proj is None:
            return None
        logits = self.logit_proj(z_h)
        # Temperature controlled logistic gate.
        temperature = max(float(self.cfg.temperature), 1e-6)
        return torch.sigmoid(logits / temperature)

    def forward(
        self,
        z_h: Tensor,
        state: Optional[SubgoalHeadState],
        *,
        update_mask: Optional[Tensor] = None,
        anchor: Optional[Tensor] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[SubgoalHeadState, SubgoalHeadOutput]:
        """Run one manager step.

        Parameters
        ----------
        z_h:
            High-level hidden state of shape ``[B, D]``.
        state:
            Previous ``SubgoalHeadState``. If ``None``, an initial state is
            created (goals start at zero).
        update_mask:
            Optional boolean mask ``[B]`` indicating which batch elements
            should refresh their subgoal this step. When ``None`` a periodic
            schedule derived from ``manager_period`` is used.
        anchor:
            Post-worker representation ``[B, hidden_size]`` to store for each
            newly emitted commitment. The current anchor is retained when this
            is omitted, which supports legacy direct uses of the head.
        temperature:
            Optional override for the gating temperature.

        Returns
        -------
        new_state, output:
            Updated state (detached if ``detach_goals`` is True) and output
            structure containing the active goal and optional gating signals.
        """

        if z_h.dim() != 2:
            raise ValueError(f"Expected z_h with shape [B, D], got {tuple(z_h.shape)}")

        batch_size, *_ = z_h.shape
        device = z_h.device

        if state is None:
            state = self.initial_state(batch_size=batch_size, device=device)
        else:
            # Ensure state tensors live on the correct device.
            if state.step.device != device:
                state = SubgoalHeadState(
                    step=state.step.to(device),
                    goal=state.goal.to(device),
                    gate=None if state.gate is None else state.gate.to(device),
                    anchor=state.anchor.to(device),
                )

        step = state.step + 1

        if update_mask is None:
            update_mask = (step % self.cfg.manager_period) == 0
        else:
            update_mask = update_mask.to(device=device, dtype=torch.bool)
            if update_mask.shape != (batch_size,):
                raise ValueError(
                    f"update_mask must have shape [B], got {tuple(update_mask.shape)}"
                )

        candidate_goal = self._compute_goal(z_h)

        if anchor is not None and anchor.shape != (batch_size, self.cfg.hidden_size):
            raise ValueError(
                "anchor must have shape "
                f"[{batch_size}, {self.cfg.hidden_size}], got {tuple(anchor.shape)}"
            )

        logits: Optional[Tensor] = None
        candidate_gate: Optional[Tensor] = None
        if self.logit_proj is not None:
            logit_values = self.logit_proj(z_h)
            temp = max(
                float(temperature if temperature is not None else self.cfg.temperature),
                1e-6,
            )
            candidate_gate = torch.sigmoid(logit_values / temp)
            logits = logit_values

        # Select between previous and candidate goal.
        expanded_mask = update_mask.unsqueeze(-1)
        goal = torch.where(expanded_mask, candidate_goal, state.goal)
        if candidate_gate is None:
            gate = None
        else:
            assert state.gate is not None
            gate = torch.where(expanded_mask, candidate_gate, state.gate)
        new_anchor = (
            torch.where(expanded_mask, anchor, state.anchor)
            if anchor is not None
            else state.anchor
        )

        if self.cfg.detach_goals:
            goal_to_store = goal.detach()
        else:
            goal_to_store = goal

        gate_to_store: Optional[Tensor]
        if gate is None:
            gate_to_store = None
        elif self.cfg.detach_goals:
            gate_to_store = gate.detach()
        else:
            gate_to_store = gate

        new_state = SubgoalHeadState(
            step=step,
            goal=goal_to_store,
            gate=gate_to_store,
            anchor=new_anchor.detach(),
        )

        output = SubgoalHeadOutput(
            goal=goal,
            gate=gate,
            logits=logits,
            probs=gate,
            updated=update_mask,
        )

        return new_state, output
