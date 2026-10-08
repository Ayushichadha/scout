from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel, Field

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm,
    SwiGLU,
    Attention,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
)
from models.sparse_embedding import CastedSparseEmbedding
from models.subgoal_head import (
    AdaptiveReplanTrigger,
    AdaptiveTriggerState,
    CounterfactualPersistenceCritic,
    SubgoalHead,
    SubgoalHeadConfig,
    SubgoalHeadState,
    adaptive_trigger_features,
    counterfactual_critic_features,
)


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class SubgoalCreditState:
    """Detached data used to rebuild one consuming-pass manager Jacobian."""

    manager_repr: torch.Tensor
    pending: torch.Tensor


@dataclass
class AdaptiveDecisionCredit:
    """Detached record used for one-consuming-pass trigger reconstruction."""

    feature: torch.Tensor
    manager_repr: torch.Tensor
    old_goal: torch.Tensor
    old_gate: torch.Tensor
    old_anchor: torch.Tensor
    candidate_anchor: torch.Tensor
    hard_decision: torch.Tensor
    pending: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]
    subgoal_state: Optional[SubgoalHeadState] = None
    subgoal_credit_state: Optional[SubgoalCreditState] = None
    adaptive_trigger_state: Optional[AdaptiveTriggerState] = None
    adaptive_decision_credit: Optional[AdaptiveDecisionCredit] = None


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float
    fixed_refinement_steps: Optional[int] = Field(default=None, ge=1)

    forward_dtype: str = "bfloat16"

    subgoal_head: Optional[SubgoalHeadConfig] = None


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(
            hidden_states
            + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        # Fully Connected
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps
        )
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(
            self.config.puzzle_emb_ndim // -self.config.hidden_size
        )  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[
                HierarchicalReasoningModel_ACTV1Block(self.config)
                for _i in range(self.config.H_layers)
            ]
        )
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[
                HierarchicalReasoningModel_ACTV1Block(self.config)
                for _i in range(self.config.L_layers)
            ]
        )

        # Learned map from manager goal space to the shared HRM hidden space.
        self.V_L: Optional[nn.Linear] = None
        if self.config.subgoal_head is not None:
            self.V_L = nn.Linear(
                self.config.subgoal_head.goal_dim,
                self.config.hidden_size,
                bias=False,
            )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1
            ),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1
            ),
            persistent=True,
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = (
                self.puzzle_emb_len * self.config.hidden_size
                - puzzle_embedding.shape[-1]
            )
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (
                    puzzle_embedding.view(
                        -1, self.puzzle_emb_len, self.config.hidden_size
                    ),
                    embedding,
                ),
                dim=-2,
            )

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
            )

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        device = self.H_init.device
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
        )

    def reset_carry(
        self,
        reset_flag: torch.Tensor,
        carry: HierarchicalReasoningModel_ACTV1InnerCarry,
    ):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
        *,
        goal: Optional[torch.Tensor] = None,
        gate: Optional[torch.Tensor] = None,
    ) -> Tuple[
        HierarchicalReasoningModel_ACTV1InnerCarry,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"]
        )

        if (
            self.config.subgoal_head is not None
            and not self.config.subgoal_head.inject_subgoal
        ):
            goal = None

        goal_bias: Optional[torch.Tensor] = None
        if goal is not None:
            if self.V_L is None:
                raise RuntimeError("A subgoal was supplied without a V_L projection")
            goal_bias = self.V_L(goal.to(self.V_L.weight.dtype)).to(self.forward_dtype)
            if gate is not None:
                goal_bias = goal_bias * gate.to(goal_bias.dtype)
            goal_bias = goal_bias.unsqueeze(1)

        def add_goal(tensor: torch.Tensor) -> torch.Tensor:
            if goal_bias is None:
                return tensor
            return tensor + goal_bias.expand(-1, tensor.shape[1], -1)

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not (
                        (_H_step == self.config.H_cycles - 1)
                        and (_L_step == self.config.L_cycles - 1)
                    ):
                        z_L = self.L_level(
                            z_L, add_goal(z_H + input_embeddings), **seq_info
                        )

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, add_goal(z_L), **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, add_goal(z_H + input_embeddings), **seq_info)
        z_H = self.H_level(z_H, add_goal(z_L), **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(), z_L=z_L.detach()
        )  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        extras = {
            "manager_hidden": z_H,
            "worker_hidden": z_L,
        }
        if goal_bias is not None:
            extras["goal_bias"] = goal_bias.squeeze(1)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), extras


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)
        self.subgoal_head: Optional[SubgoalHead]
        self.adaptive_trigger: Optional[AdaptiveReplanTrigger] = None
        self.counterfactual_critic: Optional[CounterfactualPersistenceCritic] = None
        if self.config.subgoal_head is not None:
            if (
                self.config.subgoal_head.directional_displacement
                and self.config.subgoal_head.goal_dim != self.config.hidden_size
            ):
                raise ValueError(
                    "directional_displacement requires goal_dim == hidden_size "
                    "until a separate worker-state projection is introduced"
                )
            self.subgoal_head = SubgoalHead(self.config.subgoal_head)
            if self.config.subgoal_head.replan_mode == "adaptive":
                if not self.config.subgoal_head.detach_goals:
                    raise ValueError(
                        "adaptive re-planning v1 requires detach_goals=true"
                    )
                if not self.config.subgoal_head.gating:
                    raise ValueError("adaptive re-planning v1 requires gating=true")
                if not self.config.subgoal_head.directional_displacement:
                    raise ValueError(
                        "adaptive re-planning v1 requires directional_displacement=true"
                    )
                if self.config.fixed_refinement_steps != 8:
                    raise ValueError(
                        "adaptive re-planning v1 requires fixed_refinement_steps=8"
                    )
                if self.config.subgoal_head.initial_goal_only:
                    raise ValueError(
                        "adaptive re-planning is incompatible with initial_goal_only"
                    )
                self.adaptive_trigger = AdaptiveReplanTrigger()
            elif self.config.subgoal_head.replan_mode == "counterfactual_v2":
                if not self.config.subgoal_head.detach_goals:
                    raise ValueError(
                        "counterfactual persistence v2 requires detach_goals=true"
                    )
                if not self.config.subgoal_head.gating:
                    raise ValueError(
                        "counterfactual persistence v2 requires gating=true"
                    )
                if not self.config.subgoal_head.directional_displacement:
                    raise ValueError(
                        "counterfactual persistence v2 requires "
                        "directional_displacement=true"
                    )
                if self.config.fixed_refinement_steps != 8:
                    raise ValueError(
                        "counterfactual persistence v2 requires "
                        "fixed_refinement_steps=8"
                    )
                if self.config.subgoal_head.initial_goal_only:
                    raise ValueError(
                        "counterfactual persistence v2 is incompatible with "
                        "initial_goal_only"
                    )
                self.counterfactual_critic = CounterfactualPersistenceCritic()
        else:
            self.subgoal_head = None

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        subgoal_state = None
        subgoal_credit_state = None
        adaptive_trigger_state = None
        adaptive_decision_credit = None
        if self.subgoal_head is not None:
            subgoal_state = self.subgoal_head.initial_state(
                batch_size=batch_size, device=device
            )
            subgoal_credit_state = SubgoalCreditState(
                manager_repr=torch.zeros(
                    batch_size,
                    self.config.hidden_size,
                    dtype=torch.float32,
                    device=device,
                ),
                pending=torch.zeros(batch_size, dtype=torch.bool, device=device),
            )
            if self.adaptive_trigger is not None:
                adaptive_trigger_state = AdaptiveTriggerState(
                    previous_worker_repr=torch.zeros(
                        batch_size,
                        self.config.hidden_size,
                        dtype=torch.float32,
                        device=device,
                    ),
                    previous_cumulative=torch.zeros(
                        batch_size, dtype=torch.float32, device=device
                    ),
                    dwell=torch.zeros(batch_size, dtype=torch.long, device=device),
                    has_previous_worker_repr=torch.zeros(
                        batch_size, dtype=torch.bool, device=device
                    ),
                )
                goal_dim = self.config.subgoal_head.goal_dim
                adaptive_decision_credit = AdaptiveDecisionCredit(
                    feature=torch.zeros(
                        batch_size, 6, dtype=torch.float32, device=device
                    ),
                    manager_repr=torch.zeros(
                        batch_size,
                        self.config.hidden_size,
                        dtype=torch.float32,
                        device=device,
                    ),
                    old_goal=torch.zeros(
                        batch_size, goal_dim, dtype=torch.float32, device=device
                    ),
                    old_gate=torch.zeros(
                        batch_size, 1, dtype=torch.float32, device=device
                    ),
                    old_anchor=torch.zeros(
                        batch_size,
                        self.config.hidden_size,
                        dtype=torch.float32,
                        device=device,
                    ),
                    candidate_anchor=torch.zeros(
                        batch_size,
                        self.config.hidden_size,
                        dtype=torch.float32,
                        device=device,
                    ),
                    hard_decision=torch.zeros(
                        batch_size, dtype=torch.bool, device=device
                    ),
                    pending=torch.zeros(batch_size, dtype=torch.bool, device=device),
                )

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(
                batch_size
            ),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones(
                (batch_size,), dtype=torch.bool, device=device
            ),  # Default to halted
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
            subgoal_state=subgoal_state,
            subgoal_credit_state=subgoal_credit_state,
            adaptive_trigger_state=adaptive_trigger_state,
            adaptive_decision_credit=adaptive_decision_credit,
        )

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v
            )
            for k, v in carry.current_data.items()
        }

        # Subgoal state and controls. A halted sample begins a fresh ACT episode,
        # so its complete manager commitment is reset before this worker pass.
        subgoal_state = carry.subgoal_state
        if self.subgoal_head is not None and subgoal_state is None:
            subgoal_state = self.subgoal_head.initial_state(
                batch_size=batch["inputs"].shape[0], device=batch["inputs"].device
            )
        if self.subgoal_head is not None and subgoal_state is not None:
            subgoal_state = self.subgoal_head.reset_state(subgoal_state, carry.halted)

        fresh_episode = carry.halted

        credit_state = carry.subgoal_credit_state
        if self.subgoal_head is not None and credit_state is None:
            credit_state = SubgoalCreditState(
                manager_repr=torch.zeros(
                    batch["inputs"].shape[0],
                    self.config.hidden_size,
                    dtype=torch.float32,
                    device=batch["inputs"].device,
                ),
                pending=torch.zeros_like(carry.halted),
            )
        if credit_state is not None:
            reset_col = fresh_episode.unsqueeze(-1)
            credit_state = SubgoalCreditState(
                manager_repr=torch.where(
                    reset_col,
                    torch.zeros_like(credit_state.manager_repr),
                    credit_state.manager_repr,
                ).detach(),
                pending=torch.where(
                    fresh_episode,
                    torch.zeros_like(credit_state.pending),
                    credit_state.pending,
                ),
            )

        adaptive_state = carry.adaptive_trigger_state
        decision_credit = carry.adaptive_decision_credit
        if self.adaptive_trigger is not None:
            if adaptive_state is None or decision_credit is None:
                raise RuntimeError("adaptive carry state was not initialized")
            reset_col = fresh_episode.unsqueeze(-1)
            adaptive_state = AdaptiveTriggerState(
                previous_worker_repr=torch.where(
                    reset_col,
                    torch.zeros_like(adaptive_state.previous_worker_repr),
                    adaptive_state.previous_worker_repr,
                ).detach(),
                previous_cumulative=torch.where(
                    fresh_episode,
                    torch.zeros_like(adaptive_state.previous_cumulative),
                    adaptive_state.previous_cumulative,
                ).detach(),
                dwell=torch.where(
                    fresh_episode,
                    torch.zeros_like(adaptive_state.dwell),
                    adaptive_state.dwell,
                ).detach(),
                has_previous_worker_repr=torch.where(
                    fresh_episode,
                    torch.zeros_like(adaptive_state.has_previous_worker_repr),
                    adaptive_state.has_previous_worker_repr,
                ).detach(),
            )
            decision_credit = AdaptiveDecisionCredit(
                feature=torch.where(
                    reset_col,
                    torch.zeros_like(decision_credit.feature),
                    decision_credit.feature,
                ).detach(),
                manager_repr=torch.where(
                    reset_col,
                    torch.zeros_like(decision_credit.manager_repr),
                    decision_credit.manager_repr,
                ).detach(),
                old_goal=torch.where(
                    reset_col,
                    torch.zeros_like(decision_credit.old_goal),
                    decision_credit.old_goal,
                ).detach(),
                old_gate=torch.where(
                    reset_col,
                    torch.zeros_like(decision_credit.old_gate),
                    decision_credit.old_gate,
                ).detach(),
                old_anchor=torch.where(
                    reset_col,
                    torch.zeros_like(decision_credit.old_anchor),
                    decision_credit.old_anchor,
                ).detach(),
                candidate_anchor=torch.where(
                    reset_col,
                    torch.zeros_like(decision_credit.candidate_anchor),
                    decision_credit.candidate_anchor,
                ).detach(),
                hard_decision=torch.where(
                    fresh_episode,
                    torch.zeros_like(decision_credit.hard_decision),
                    decision_credit.hard_decision,
                ).detach(),
                pending=torch.where(
                    fresh_episode,
                    torch.zeros_like(decision_credit.pending),
                    decision_credit.pending,
                ).detach(),
            )

        goal_tensor: Optional[torch.Tensor] = None
        gate_tensor: Optional[torch.Tensor] = None
        if subgoal_state is not None:
            goal_tensor = subgoal_state.goal
            gate_tensor = subgoal_state.gate

        anchor_tensor = None if subgoal_state is None else subgoal_state.anchor

        # Rebuild the previous adaptive selection locally. This record contains
        # values only; no graph crosses the optimizer step between outer calls.
        if (
            self.adaptive_trigger is not None
            and decision_credit is not None
            and goal_tensor is not None
            and gate_tensor is not None
            and anchor_tensor is not None
        ):
            pending_col = decision_credit.pending.unsqueeze(-1)
            beta_hat = self.adaptive_trigger(decision_credit.feature)
            hard = decision_credit.hard_decision.to(beta_hat.dtype)
            h_hat = hard + beta_hat - beta_hat.detach()
            h_col = h_hat.unsqueeze(-1)
            candidate_goal = self.subgoal_head._compute_goal(
                decision_credit.manager_repr
            )
            candidate_gate = self.subgoal_head._compute_gate(
                decision_credit.manager_repr
            )
            assert candidate_gate is not None
            selected_goal = (
                h_col * candidate_goal + (1.0 - h_col) * decision_credit.old_goal
            )
            selected_gate = (
                h_col * candidate_gate + (1.0 - h_col) * decision_credit.old_gate
            )
            selected_anchor = (
                h_col * decision_credit.candidate_anchor
                + (1.0 - h_col) * decision_credit.old_anchor
            )
            goal_tensor = goal_tensor + pending_col * (
                selected_goal - selected_goal.detach()
            )
            gate_tensor = gate_tensor + pending_col * (
                selected_gate - selected_gate.detach()
            )
            anchor_tensor = anchor_tensor + pending_col * (
                selected_anchor - selected_anchor.detach()
            )

        # The optimizer steps between outer calls, so a raw graph from the
        # emission pass cannot safely cross carry. Rebuild only the local
        # manager-head Jacobian from a detached emission representation. The
        # value-preserving form keeps the exact cached commitment in forward
        # while routing this consuming pass's gradient to the current head.
        if (
            self.subgoal_head is not None
            and self.config.subgoal_head.directional_displacement
            and credit_state is not None
            and goal_tensor is not None
        ):
            manager_credit_pending = credit_state.pending
            if decision_credit is not None:
                manager_credit_pending = (
                    manager_credit_pending & ~decision_credit.pending
                )
            credit_mask = manager_credit_pending.unsqueeze(-1)
            recomputed_goal = self.subgoal_head._compute_goal(credit_state.manager_repr)
            goal_tensor = goal_tensor + credit_mask * (
                recomputed_goal - recomputed_goal.detach()
            )
            if gate_tensor is not None:
                recomputed_gate = self.subgoal_head._compute_gate(
                    credit_state.manager_repr
                )
                assert recomputed_gate is not None
                gate_tensor = gate_tensor + credit_mask * (
                    recomputed_gate - recomputed_gate.detach()
                )

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), extras = (
            self.inner(
                new_inner_carry,
                new_current_data,
                goal=goal_tensor,
                gate=gate_tensor,
            )
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        # Add hidden states to outputs for feudal loss computation
        outputs["worker_hidden"] = extras["worker_hidden"]
        outputs["manager_hidden"] = extras["manager_hidden"]
        if "goal_bias" in extras:
            outputs["subgoal_goal_bias"] = extras["goal_bias"]

        new_subgoal_state: Optional[SubgoalHeadState] = None
        new_credit_state: Optional[SubgoalCreditState] = None
        new_adaptive_state = adaptive_state
        new_decision_credit = decision_credit
        subgoal_output = None
        if self.subgoal_head is not None:
            manager_hidden = extras["manager_hidden"]
            manager_repr = manager_hidden[:, 0].to(torch.float32)
            worker_repr = extras["worker_hidden"].mean(dim=1).to(torch.float32)
            active_goal = goal_tensor
            active_gate = gate_tensor
            active_anchor = anchor_tensor
            active_mask = ~fresh_episode
            next_step = subgoal_state.step + 1
            # Preserve the historical episode-step phase: for P=3 the forced
            # initial emission occurs after call 1, followed by periodic
            # emissions after calls 3, 6, 9, ... . Because emission follows
            # worker computation, each emitted commitment is first consumed
            # on the next outer call.
            periodic_update = (
                (next_step % self.config.subgoal_head.manager_period) == 0
            ) & (not self.config.subgoal_head.initial_goal_only)
            # Emission follows worker computation and is first consumed on the
            # next outer call. In fixed-compute mode, suppress any scheduled
            # intervention after final pass M because pass M+1 does not exist.
            has_next_worker_pass = (
                torch.ones_like(fresh_episode)
                if self.config.fixed_refinement_steps is None
                else next_step < self.config.fixed_refinement_steps
            )
            adaptive_eligible = torch.zeros_like(fresh_episode)
            trigger_probability = torch.zeros_like(next_step, dtype=torch.float32)
            trigger_hard = torch.zeros_like(fresh_episode)
            trigger_feature = torch.zeros(
                worker_repr.shape[0], 6, dtype=torch.float32, device=worker_repr.device
            )
            counterfactual_feature = torch.zeros(
                worker_repr.shape[0], 2, dtype=torch.float32, device=worker_repr.device
            )
            counterfactual_score = torch.zeros_like(next_step, dtype=torch.float32)
            counterfactual_hard = torch.zeros_like(fresh_episode)
            counterfactual_eligible = torch.zeros_like(fresh_episode)
            counterfactual_branches: Dict[str, torch.Tensor] = {}
            dwell_after_pass = torch.zeros_like(next_step, dtype=torch.long)
            if self.counterfactual_critic is not None:
                assert active_goal is not None and active_gate is not None
                assert active_anchor is not None
                counterfactual_eligible = (~fresh_episode) & has_next_worker_pass
                candidate_goal = self.subgoal_head._compute_goal(manager_repr)
                candidate_gate = self.subgoal_head._compute_gate(manager_repr)
                assert candidate_gate is not None
                counterfactual_feature = counterfactual_critic_features(
                    worker_repr=worker_repr,
                    active_goal=active_goal,
                    active_anchor=active_anchor,
                    candidate_goal=candidate_goal,
                )
                counterfactual_score = self.counterfactual_critic(
                    counterfactual_feature
                )
                counterfactual_hard = counterfactual_eligible & (
                    counterfactual_score
                    > self.config.subgoal_head.counterfactual_replan_cost
                )
                update_mask = fresh_episode | counterfactual_hard

                # Supervision-only futures. Both alternatives start at the
                # exact same detached latent state. The first transition and
                # Q lookahead differ only in the selected commitment. No graph
                # is retained, so this target cannot update the reasoner.
                if bool(counterfactual_eligible.any()):
                    branch_specs = {
                        "persist": (active_goal, active_gate, active_anchor),
                        "replan": (candidate_goal, candidate_gate, worker_repr),
                    }
                    with torch.no_grad():
                        for branch_name, (
                            branch_goal,
                            branch_gate,
                            branch_anchor,
                        ) in branch_specs.items():
                            branch_carry, branch_logits, branch_q, branch_extras = (
                                self.inner(
                                    new_inner_carry,
                                    new_current_data,
                                    goal=branch_goal.detach(),
                                    gate=branch_gate.detach(),
                                )
                            )
                            _, _, branch_next_q, _ = self.inner(
                                branch_carry,
                                new_current_data,
                                goal=branch_goal.detach(),
                                gate=branch_gate.detach(),
                            )
                            consuming_is_terminal = (
                                next_step + 1 >= self.config.fixed_refinement_steps
                            )
                            branch_continue_target = torch.sigmoid(
                                torch.where(
                                    consuming_is_terminal,
                                    branch_next_q[0],
                                    torch.maximum(branch_next_q[0], branch_next_q[1]),
                                )
                            )
                            prefix = f"counterfactual_{branch_name}"
                            counterfactual_branches[f"{prefix}_logits"] = (
                                branch_logits.detach()
                            )
                            counterfactual_branches[f"{prefix}_q_halt_logits"] = (
                                branch_q[0].detach()
                            )
                            counterfactual_branches[f"{prefix}_q_continue_logits"] = (
                                branch_q[1].detach()
                            )
                            counterfactual_branches[f"{prefix}_continue_target"] = (
                                branch_continue_target.detach()
                            )
                            counterfactual_branches[f"{prefix}_worker_hidden"] = (
                                branch_extras["worker_hidden"].detach()
                            )
                            counterfactual_branches[f"{prefix}_goal"] = (
                                branch_goal.detach()
                            )
                            counterfactual_branches[f"{prefix}_gate"] = (
                                branch_gate.detach()
                            )
                            counterfactual_branches[f"{prefix}_anchor"] = (
                                branch_anchor.detach()
                            )
            elif self.adaptive_trigger is None:
                scheduled_update = fresh_episode | periodic_update
                update_mask = scheduled_update & has_next_worker_pass
            else:
                assert adaptive_state is not None
                assert active_goal is not None and active_gate is not None
                assert active_anchor is not None
                adaptive_eligible = (~fresh_episode) & has_next_worker_pass
                torch._assert(
                    (
                        ~adaptive_eligible | adaptive_state.has_previous_worker_repr
                    ).all(),
                    "eligible adaptive decision lacks a previous worker representation",
                )
                dwell_after_pass = adaptive_state.dwell + active_mask.to(torch.long)
                trigger_feature = adaptive_trigger_features(
                    worker_repr=worker_repr,
                    previous_worker_repr=adaptive_state.previous_worker_repr,
                    previous_cumulative=adaptive_state.previous_cumulative,
                    dwell=dwell_after_pass,
                    q_halt_logits=q_halt_logits,
                    q_continue_logits=q_continue_logits,
                    active_goal=active_goal,
                    active_gate=active_gate,
                    active_anchor=active_anchor,
                    max_passes=self.config.fixed_refinement_steps,
                )
                trigger_probability, sampled_hard, _ = self.adaptive_trigger.decide(
                    trigger_feature,
                    training=self.training,
                    stochastic_train=self.config.subgoal_head.trigger_stochastic_train,
                    threshold=self.config.subgoal_head.trigger_threshold,
                    eligible=adaptive_eligible,
                )
                trigger_hard = sampled_hard
                update_mask = fresh_episode | trigger_hard
            new_subgoal_state, subgoal_output = self.subgoal_head(
                manager_repr,
                subgoal_state,
                update_mask=update_mask,
                anchor=worker_repr,
            )
            assert credit_state is not None
            update_col = update_mask.unsqueeze(-1)
            new_credit_state = SubgoalCreditState(
                manager_repr=torch.where(
                    update_col,
                    manager_repr.detach(),
                    credit_state.manager_repr,
                ).detach(),
                # Credit is valid for exactly the first pass after emission.
                # Any previous pending credit was consumed by this call.
                pending=(
                    update_mask if self.adaptive_trigger is None else fresh_episode
                ).detach(),
            )

            new_adaptive_state = adaptive_state
            new_decision_credit = decision_credit
            completed_dwell = torch.zeros_like(next_step, dtype=torch.long)
            completed_dwell_mask = torch.zeros_like(fresh_episode)
            old_new_goal_cosine = torch.zeros_like(trigger_probability)
            if self.adaptive_trigger is not None:
                assert adaptive_state is not None and decision_credit is not None
                assert active_goal is not None and active_gate is not None
                assert active_anchor is not None
                cumulative = trigger_feature[:, 0]
                replace = trigger_hard
                # Bootstrap initializes observation history after the forced
                # emission. A replacement starts a new, not-yet-consumed dwell.
                next_previous_cumulative = torch.where(
                    replace | fresh_episode,
                    torch.zeros_like(cumulative),
                    cumulative,
                )
                next_dwell = torch.where(
                    replace | fresh_episode,
                    torch.zeros_like(dwell_after_pass),
                    dwell_after_pass,
                )
                new_adaptive_state = AdaptiveTriggerState(
                    previous_worker_repr=worker_repr.detach(),
                    previous_cumulative=next_previous_cumulative.detach(),
                    dwell=next_dwell.detach(),
                    has_previous_worker_repr=torch.ones_like(fresh_episode).detach(),
                )
                eligible_col = adaptive_eligible.unsqueeze(-1)
                old_gate = active_gate.to(torch.float32)
                new_decision_credit = AdaptiveDecisionCredit(
                    feature=torch.where(
                        eligible_col, trigger_feature, decision_credit.feature
                    ).detach(),
                    manager_repr=torch.where(
                        eligible_col,
                        manager_repr.detach(),
                        decision_credit.manager_repr,
                    ).detach(),
                    old_goal=torch.where(
                        eligible_col, active_goal.detach(), decision_credit.old_goal
                    ).detach(),
                    old_gate=torch.where(
                        eligible_col, old_gate.detach(), decision_credit.old_gate
                    ).detach(),
                    old_anchor=torch.where(
                        eligible_col, active_anchor.detach(), decision_credit.old_anchor
                    ).detach(),
                    candidate_anchor=torch.where(
                        eligible_col,
                        worker_repr.detach(),
                        decision_credit.candidate_anchor,
                    ).detach(),
                    hard_decision=torch.where(
                        adaptive_eligible, trigger_hard, decision_credit.hard_decision
                    ).detach(),
                    pending=adaptive_eligible.detach(),
                )
                completed_dwell_mask = replace | (active_mask & ~has_next_worker_pass)
                completed_dwell = torch.where(
                    completed_dwell_mask,
                    dwell_after_pass,
                    torch.zeros_like(dwell_after_pass),
                )
                candidate_goal_metric = self.subgoal_head._compute_goal(
                    manager_repr
                ).detach()
                old_new_goal_cosine = F.cosine_similarity(
                    active_goal.detach().to(torch.float32),
                    candidate_goal_metric.to(torch.float32),
                    dim=-1,
                    eps=1e-8,
                )

            if self.config.subgoal_head.directional_displacement:
                # Alignment describes the commitment consumed by this worker
                # pass, never the goal emitted after it. The first pass of an
                # episode is inactive because causal ordering only makes its
                # forced goal available to the next outer call.
                outputs["subgoal_goal"] = active_goal
                outputs["subgoal_anchor"] = active_anchor
                outputs["subgoal_active"] = active_mask.to(torch.float32)
                if active_gate is not None:
                    outputs["subgoal_gate"] = active_gate
            else:
                # Compatibility path for historical absolute-state alignment.
                outputs["subgoal_goal"] = subgoal_output.goal
                if subgoal_output.gate is not None:
                    outputs["subgoal_gate"] = subgoal_output.gate
            outputs["subgoal_fresh_episode"] = fresh_episode.to(torch.float32)
            outputs["subgoal_step"] = new_subgoal_state.step
            outputs["subgoal_updated"] = subgoal_output.updated.to(torch.float32)
            outputs["subgoal_has_next_worker_pass"] = has_next_worker_pass.to(
                torch.float32
            )
            outputs["subgoal_credit_pending"] = credit_state.pending.to(torch.float32)
            outputs["subgoal_credit_manager_repr"] = credit_state.manager_repr
            if self.counterfactual_critic is not None:
                outputs["counterfactual_critic_feature"] = counterfactual_feature
                outputs["counterfactual_critic_score"] = counterfactual_score
                outputs["counterfactual_effective_advantage"] = (
                    counterfactual_score
                    - self.config.subgoal_head.counterfactual_replan_cost
                )
                outputs["counterfactual_replan_hard"] = counterfactual_hard.to(
                    torch.float32
                )
                outputs["counterfactual_eligible"] = counterfactual_eligible.to(
                    torch.float32
                )
                outputs.update(counterfactual_branches)
            if self.adaptive_trigger is not None:
                outputs["adaptive_trigger_feature"] = trigger_feature
                outputs["adaptive_trigger_probability"] = trigger_probability
                outputs["adaptive_trigger_hard"] = trigger_hard.to(torch.float32)
                outputs["adaptive_trigger_eligible"] = adaptive_eligible.to(
                    torch.float32
                )
                outputs["adaptive_dwell_after_pass"] = dwell_after_pass
                outputs["adaptive_completed_dwell"] = completed_dwell
                outputs["adaptive_completed_dwell_mask"] = completed_dwell_mask.to(
                    torch.float32
                )
                outputs["adaptive_old_new_goal_cosine"] = old_new_goal_cosine
                outputs["adaptive_credit_pending"] = decision_credit.pending.to(
                    torch.float32
                )

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            execution_limit = (
                self.config.fixed_refinement_steps
                if self.config.fixed_refinement_steps is not None
                else self.config.halt_max_steps
            )
            is_last_step = new_steps >= execution_limit

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                if self.config.fixed_refinement_steps is None:
                    halted = halted | (q_halt_logits > q_continue_logits)

                    # Exploration
                    min_halt_steps = (
                        torch.rand_like(q_halt_logits)
                        < self.config.halt_exploration_prob
                    ) * torch.randint_like(
                        new_steps, low=2, high=self.config.halt_max_steps + 1
                    )

                    halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                next_goal = (
                    None if new_subgoal_state is None else new_subgoal_state.goal
                )
                next_gate = (
                    None if new_subgoal_state is None else new_subgoal_state.gate
                )
                _, _, (next_q_halt_logits, next_q_continue_logits), _ = self.inner(
                    new_inner_carry,
                    new_current_data,
                    goal=next_goal,
                    gate=next_gate,
                )

                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt_logits,
                        torch.maximum(next_q_halt_logits, next_q_continue_logits),
                    )
                )

            if self.config.fixed_refinement_steps is not None and halted.any():
                completed_steps = new_steps[halted]
                if not torch.all(completed_steps == self.config.fixed_refinement_steps):
                    raise AssertionError(
                        "fixed-compute invariant violated: completed episode "
                        f"steps={completed_steps.tolist()}, expected "
                        f"{self.config.fixed_refinement_steps}"
                    )

        return (
            HierarchicalReasoningModel_ACTV1Carry(
                inner_carry=new_inner_carry,
                steps=new_steps,
                halted=halted,
                current_data=new_current_data,
                subgoal_state=new_subgoal_state,
                subgoal_credit_state=new_credit_state,
                adaptive_trigger_state=new_adaptive_state,
                adaptive_decision_credit=new_decision_credit,
            ),
            outputs,
        )
