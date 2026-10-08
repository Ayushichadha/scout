from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


def accumulate_episode_metrics(
    accumulated: Optional[Dict[str, torch.Tensor]],
    current: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Accumulate one outer pass while enforcing a stable metric schema."""
    if accumulated is None:
        return {key: value.clone() for key, value in current.items()}
    if accumulated.keys() != current.keys():
        missing = sorted(accumulated.keys() - current.keys())
        added = sorted(current.keys() - accumulated.keys())
        raise RuntimeError(
            "evaluation metric schema changed between outer passes: "
            f"missing={missing}, added={added}"
        )
    for key, value in current.items():
        accumulated[key] += value
    return accumulated


def s(x, epsilon=1e-30):
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1
    ).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]),
        labels.to(torch.long).view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(labels.shape)


def feudal_loss(
    worker_state: torch.Tensor,
    manager_goal: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    anchor: Optional[torch.Tensor] = None,
    active_mask: Optional[torch.Tensor] = None,
    reduction: str = "sum",
) -> torch.Tensor:
    """Compute feudal (intrinsic reward) loss based on worker progress toward manager goal.

    This implements the FuN-style intrinsic reward mechanism where the worker
    is rewarded for making progress toward the manager's directional subgoal.

    Parameters
    ----------
    worker_state:
        Worker (low-level) hidden state of shape [B, T, D] or [B, D].
        If [B, T, D], we pool over time dimension (mean) to get [B, D].
    manager_goal:
        Manager (high-level) goal vector of shape [B, D].
    gate:
        Optional gating signal of shape [B, 1] or [B] to modulate the reward.
        Higher gate values indicate stronger commitment to the goal.
    anchor:
        Optional pooled worker state at goal emission. When supplied, alignment
        uses ``worker_repr - anchor``; when omitted, the historical absolute
        worker representation is used.
    active_mask:
        Optional per-sample mask for passes on which a goal has been consumed.
    reduction:
        Reduction mode: "sum", "mean", or "none".

    Returns
    -------
    loss:
        Feudal loss (negative intrinsic reward). Higher cosine similarity
        between worker state and goal yields lower loss.
    """
    # Pool worker state if it has time dimension
    if worker_state.dim() == 3:
        worker_repr = worker_state.mean(dim=1)  # [B, T, D] -> [B, D]
    else:
        worker_repr = worker_state  # [B, D]

    # Directional mode aligns displacement since goal emission. Omitting an
    # anchor preserves the historical absolute-state behavior.
    alignment_vector = worker_repr if anchor is None else worker_repr - anchor

    # Normalize for cosine similarity. F.normalize maps an exactly zero
    # displacement to zero, yielding a finite per-sample loss of one.
    worker_norm = F.normalize(alignment_vector, p=2, dim=-1, eps=1e-8)
    goal_norm = F.normalize(manager_goal, p=2, dim=-1, eps=1e-8)

    # Cosine similarity: higher = better alignment
    cosine_sim = (worker_norm * goal_norm).sum(dim=-1)  # [B]

    # Convert to loss: negative reward (we want to maximize similarity)
    # Loss = 1 - cosine_sim, so perfect alignment (cosine_sim=1) gives loss=0
    feudal_loss_per_sample = 1.0 - cosine_sim  # [B]

    # Apply gating if provided
    if gate is not None:
        if gate.dim() > 1:
            gate = gate.squeeze(-1)  # [B, 1] -> [B]
        feudal_loss_per_sample = feudal_loss_per_sample * gate

    if active_mask is not None:
        feudal_loss_per_sample = feudal_loss_per_sample * active_mask.to(
            feudal_loss_per_sample.dtype
        )

    # Reduce
    if reduction == "sum":
        return feudal_loss_per_sample.sum()
    elif reduction == "mean":
        return feudal_loss_per_sample.mean()
    elif reduction == "none":
        return feudal_loss_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def counterfactual_local_objective(
    *,
    logits: torch.Tensor,
    q_halt_logits: torch.Tensor,
    q_continue_logits: torch.Tensor,
    continue_target: torch.Tensor,
    worker_hidden: torch.Tensor,
    goal: torch.Tensor,
    gate: torch.Tensor,
    anchor: torch.Tensor,
    labels: torch.Tensor,
    loss_fn,
    feudal_loss_weight: float,
) -> torch.Tensor:
    """Per-sample objective for a matched counterfactual consuming pass.

    This mirrors the ordinary LM, halt, continue, and directional terms.  All
    inputs are supervision-only detached branch outcomes; callers must not use
    this value as a decision-time feature.
    """

    valid = labels != IGNORE_LABEL_ID
    counts = valid.sum(-1).clamp_min(1)
    lm = loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID).sum(-1) / counts
    correct = valid & (torch.argmax(logits, dim=-1) == labels)
    sequence_correct = correct.sum(-1) == valid.sum(-1)
    halt = F.binary_cross_entropy_with_logits(
        q_halt_logits,
        sequence_correct.to(q_halt_logits.dtype),
        reduction="none",
    )
    cont = F.binary_cross_entropy_with_logits(
        q_continue_logits,
        continue_target.to(q_continue_logits.dtype),
        reduction="none",
    )
    directional = feudal_loss(
        worker_state=worker_hidden,
        manager_goal=goal,
        gate=gate,
        anchor=anchor,
        reduction="none",
    )
    return (lm + 0.5 * (halt + cont) + feudal_loss_weight * directional).to(
        torch.float32
    )


class ACTLossHead(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        loss_type: str,
        feudal_loss_weight: float = 0.1,
        intervention_weight: float = 0.0,
        counterfactual_critic_weight: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.feudal_loss_weight = feudal_loss_weight
        if intervention_weight < 0:
            raise ValueError("intervention_weight must be non-negative")
        if intervention_weight > 0 and getattr(model, "adaptive_trigger", None) is None:
            raise ValueError(
                "intervention_weight > 0 requires subgoal_head.replan_mode=adaptive"
            )
        self.intervention_weight = intervention_weight
        if counterfactual_critic_weight < 0:
            raise ValueError("counterfactual_critic_weight must be non-negative")
        self.counterfactual_critic_weight = counterfactual_critic_weight
        subgoal_cfg = getattr(getattr(model, "config", None), "subgoal_head", None)
        self.use_alignment_loss: bool = (
            subgoal_cfg.use_alignment_loss if subgoal_cfg is not None else True
        )

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[
        Any,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
        torch.Tensor,
    ]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Correctness
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(
                -1
            )  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    0,
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
                "completed_episodes": new_carry.halted.sum(),
                "completed_episode_refinement_steps": torch.where(
                    new_carry.halted, new_carry.steps, 0
                ).sum(),
                "executed_refinement_passes": torch.tensor(
                    new_carry.steps.shape[0],
                    dtype=torch.float32,
                    device=new_carry.steps.device,
                ),
                "fixed_compute_violations": torch.tensor(
                    0.0,
                    dtype=torch.float32,
                    device=new_carry.steps.device,
                ),
            }

        # Losses
        # FIXME: Assuming the batch is always full
        lm_loss = (
            self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID)
            / loss_divisor
        ).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )

        metrics.update(
            {
                "lm_loss": lm_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            }
        )

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum",
            )

            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Feudal loss (intrinsic reward for worker progress toward manager goal)
        feudal_loss_value = 0
        if (
            self.use_alignment_loss
            and "subgoal_goal" in outputs
            and "worker_hidden" in outputs
        ):
            worker_state = outputs["worker_hidden"]
            manager_goal = outputs["subgoal_goal"]
            gate = outputs.get("subgoal_gate")
            anchor = (
                outputs.get("subgoal_anchor")
                if getattr(
                    getattr(self.model.config, "subgoal_head", None),
                    "directional_displacement",
                    False,
                )
                else None
            )
            active_mask = outputs.get("subgoal_active")
            feudal_loss_value = feudal_loss(
                worker_state=worker_state,
                manager_goal=manager_goal,
                gate=gate,
                anchor=anchor,
                active_mask=active_mask,
                reduction="sum",
            )
            metrics["feudal_loss"] = feudal_loss_value.detach()

            # Subgoal metrics (as per NEXT_STEPS.md)
            with torch.no_grad():
                # Subgoal update frequency
                subgoal_updated = outputs.get("subgoal_updated")
                if subgoal_updated is not None:
                    metrics["subgoal_update_frequency"] = subgoal_updated.sum()
                    metrics["manager_interventions"] = subgoal_updated.sum()
                    has_next_worker = outputs.get("subgoal_has_next_worker_pass")
                    if has_next_worker is not None:
                        metrics["unconsumed_terminal_emissions"] = (
                            subgoal_updated
                            * (1.0 - has_next_worker.to(subgoal_updated.dtype))
                        ).sum()
                active_mask_metric = outputs.get("subgoal_active")
                if active_mask_metric is not None:
                    metrics["active_goal_consumptions"] = active_mask_metric.sum()

                # Subgoal gate mean (commitment strength)
                if gate is not None:
                    if gate.dim() > 1:
                        gate_flat = gate.squeeze(-1)
                    else:
                        gate_flat = gate
                    metrics["subgoal_gate_mean"] = gate_flat.mean()
                    metrics["subgoal_gate_std"] = gate_flat.std(unbiased=False)
                    active_for_gate = outputs.get("subgoal_active")
                    if active_for_gate is not None:
                        active_for_gate = active_for_gate.to(torch.bool)
                        metrics["active_gate_sum"] = gate_flat[active_for_gate].sum()
                        metrics["active_gate_count"] = active_for_gate.sum()

                # Subgoal goal norm (magnitude)
                goal_norm = torch.norm(manager_goal, p=2, dim=-1)
                metrics["subgoal_goal_norm_mean"] = goal_norm.mean()
                metrics["subgoal_goal_norm_std"] = goal_norm.std(unbiased=False)

                # Worker displacement-goal alignment in directional mode.
                if worker_state.dim() == 3:
                    worker_repr = worker_state.mean(dim=1)  # [B, T, D] -> [B, D]
                else:
                    worker_repr = worker_state  # [B, D]
                alignment_vector = (
                    worker_repr if anchor is None else worker_repr - anchor
                )
                worker_norm = F.normalize(alignment_vector, p=2, dim=-1, eps=1e-8)
                goal_norm_vec = F.normalize(manager_goal, p=2, dim=-1, eps=1e-8)
                alignment = (worker_norm * goal_norm_vec).sum(dim=-1)
                if active_mask is not None:
                    active = active_mask.to(torch.bool)
                    alignment = alignment[active]
                # Keep the per-pass metric schema stable. Bootstrap has no
                # active directional observations, so it contributes zero
                # numerator/count rather than fabricating an observation.
                zero = torch.zeros((), dtype=torch.float32, device=worker_repr.device)
                metrics["subgoal_worker_alignment_mean"] = zero
                metrics["subgoal_worker_alignment_std"] = zero
                metrics["directional_cosine_sum"] = zero
                metrics["directional_cosine_count"] = zero
                if alignment.numel() > 0:
                    metrics["subgoal_worker_alignment_mean"] = alignment.mean()
                    metrics["subgoal_worker_alignment_std"] = alignment.std(
                        unbiased=False
                    )
                    metrics["directional_cosine_sum"] = alignment.sum()
                    metrics["directional_cosine_count"] = torch.tensor(
                        alignment.numel(),
                        dtype=torch.float32,
                        device=alignment.device,
                    )

        intervention_loss = torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device)
        trigger_probability = outputs.get("adaptive_trigger_probability")
        trigger_eligible = outputs.get("adaptive_trigger_eligible")
        trigger_hard = outputs.get("adaptive_trigger_hard")
        if trigger_probability is not None and trigger_eligible is not None:
            # There are exactly K=M-2=6 eligible decisions per completed
            # fixed-M=8 episode. The training loop applies the batch divisor.
            intervention_loss = (
                trigger_probability * trigger_eligible.to(trigger_probability.dtype)
            ).sum() / 6.0
            metrics["intervention_loss"] = intervention_loss.detach()
            with torch.no_grad():
                eligible = trigger_eligible.to(torch.bool)
                hard = (
                    trigger_hard.to(torch.bool)
                    if trigger_hard is not None
                    else eligible & False
                )
                metrics["adaptive_interventions"] = (hard & eligible).sum()
                metrics["eligible_decisions"] = eligible.sum()
                metrics["trigger_probability_sum"] = trigger_probability[eligible].sum()
                metrics["trigger_probability_squared_sum"] = (
                    trigger_probability[eligible].square().sum()
                )
                metrics["trigger_probability_count"] = eligible.sum()
                metrics["trigger_probability_near_half_count"] = (
                    eligible
                    & (trigger_probability >= 0.45)
                    & (trigger_probability <= 0.55)
                ).sum()
                for bin_index in range(10):
                    lower = bin_index / 10.0
                    upper = (bin_index + 1) / 10.0
                    in_bin = eligible & (trigger_probability >= lower)
                    in_bin = in_bin & (
                        (trigger_probability <= upper)
                        if bin_index == 9
                        else (trigger_probability < upper)
                    )
                    metrics[f"trigger_probability_bin_{bin_index}"] = in_bin.sum()
                if "adaptive_trigger_feature" in outputs:
                    feature = outputs["adaptive_trigger_feature"]
                    for index, name in enumerate(
                        ("c", "d", "rho", "dwell", "q", "gate")
                    ):
                        metrics[f"trigger_feature_{name}_intervene_sum"] = feature[
                            hard & eligible, index
                        ].sum()
                        metrics[f"trigger_feature_{name}_retain_sum"] = feature[
                            (~hard) & eligible, index
                        ].sum()
                    metrics["trigger_feature_intervene_count"] = (hard & eligible).sum()
                    metrics["trigger_feature_retain_count"] = ((~hard) & eligible).sum()
                    gate = feature[:, 5]
                    metrics["trigger_beta_gate_product_sum"] = (
                        trigger_probability[eligible] * gate[eligible]
                    ).sum()
                    metrics["trigger_gate_sum"] = gate[eligible].sum()
                    metrics["trigger_gate_squared_sum"] = gate[eligible].square().sum()
                step = outputs.get("subgoal_step")
                updated = outputs.get("subgoal_updated")
                if step is not None and updated is not None:
                    for position in range(1, 8):
                        metrics[f"intervention_position_{position}"] = (
                            updated.to(torch.bool) & (step == position)
                        ).sum()
                completed_dwell = outputs.get("adaptive_completed_dwell")
                completed_mask = outputs.get("adaptive_completed_dwell_mask")
                if completed_dwell is not None and completed_mask is not None:
                    completed = completed_mask.to(torch.bool)
                    metrics["completed_dwell_sum"] = completed_dwell[completed].sum()
                    metrics["completed_dwell_count"] = completed.sum()
                    for dwell in range(1, 8):
                        metrics[f"completed_dwell_length_{dwell}"] = (
                            completed & (completed_dwell == dwell)
                        ).sum()
                dwell_after_pass = outputs.get("adaptive_dwell_after_pass")
                if dwell_after_pass is not None:
                    for dwell in range(1, 8):
                        at_dwell = eligible & (dwell_after_pass == dwell)
                        metrics[f"eligible_at_dwell_{dwell}"] = at_dwell.sum()
                        metrics[f"intervene_at_dwell_{dwell}"] = (at_dwell & hard).sum()
                cosine = outputs.get("adaptive_old_new_goal_cosine")
                if cosine is not None:
                    replacements = hard & eligible
                    metrics["old_new_goal_cosine_sum"] = cosine[replacements].sum()
                    metrics["old_new_goal_cosine_count"] = replacements.sum()
                trigger_module = getattr(self.model, "adaptive_trigger", None)
                if trigger_module is not None:
                    weights = trigger_module.linear.weight.detach().squeeze(0)
                    has_next = outputs.get("subgoal_has_next_worker_pass")
                    report_parameters = (
                        torch.ones((), device=weights.device, dtype=weights.dtype)
                        if has_next is None
                        else (~has_next.to(torch.bool)).any().to(weights.dtype)
                    )
                    for index, name in enumerate(
                        ("c", "d", "rho", "dwell", "q", "gate")
                    ):
                        metrics[f"trigger_weight_{name}"] = (
                            weights[index] * report_parameters
                        )
                    metrics["trigger_bias"] = (
                        trigger_module.linear.bias.detach().squeeze(0)
                        * report_parameters
                    )

        critic_loss = torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device)
        critic_score = outputs.get("counterfactual_critic_score")
        critic_eligible = outputs.get("counterfactual_eligible")
        if critic_score is not None and critic_eligible is not None:
            eligible = critic_eligible.to(torch.bool)
            persist_logits = outputs.get("counterfactual_persist_logits")
            if persist_logits is None:
                advantage = torch.zeros_like(critic_score)
                persist_objective = torch.zeros_like(critic_score)
                replan_objective = torch.zeros_like(critic_score)
            else:
                persist_objective = counterfactual_local_objective(
                    logits=persist_logits,
                    q_halt_logits=outputs["counterfactual_persist_q_halt_logits"],
                    q_continue_logits=outputs[
                        "counterfactual_persist_q_continue_logits"
                    ],
                    continue_target=outputs["counterfactual_persist_continue_target"],
                    worker_hidden=outputs["counterfactual_persist_worker_hidden"],
                    goal=outputs["counterfactual_persist_goal"],
                    gate=outputs["counterfactual_persist_gate"],
                    anchor=outputs["counterfactual_persist_anchor"],
                    labels=labels,
                    loss_fn=self.loss_fn,
                    feudal_loss_weight=self.feudal_loss_weight,
                )
                replan_objective = counterfactual_local_objective(
                    logits=outputs["counterfactual_replan_logits"],
                    q_halt_logits=outputs["counterfactual_replan_q_halt_logits"],
                    q_continue_logits=outputs[
                        "counterfactual_replan_q_continue_logits"
                    ],
                    continue_target=outputs["counterfactual_replan_continue_target"],
                    worker_hidden=outputs["counterfactual_replan_worker_hidden"],
                    goal=outputs["counterfactual_replan_goal"],
                    gate=outputs["counterfactual_replan_gate"],
                    anchor=outputs["counterfactual_replan_anchor"],
                    labels=labels,
                    loss_fn=self.loss_fn,
                    feudal_loss_weight=self.feudal_loss_weight,
                )
                advantage = (persist_objective - replan_objective).detach()
            if eligible.any():
                critic_loss = F.mse_loss(
                    critic_score[eligible], advantage[eligible], reduction="sum"
                )
            with torch.no_grad():
                hard = outputs["counterfactual_replan_hard"].to(torch.bool)
                feature = outputs["counterfactual_critic_feature"]
                effective = outputs["counterfactual_effective_advantage"]
                metrics["counterfactual_critic_loss"] = critic_loss.detach()
                metrics["counterfactual_eligible_decisions"] = eligible.sum()
                metrics["counterfactual_replans"] = (hard & eligible).sum()
                metrics["counterfactual_score_sum"] = critic_score[eligible].sum()
                metrics["counterfactual_effective_advantage_sum"] = effective[
                    eligible
                ].sum()
                metrics["counterfactual_target_advantage_sum"] = advantage[
                    eligible
                ].sum()
                metrics["counterfactual_target_advantage_sq_sum"] = (
                    advantage[eligible].square().sum()
                )
                metrics["counterfactual_persist_objective_sum"] = persist_objective[
                    eligible
                ].sum()
                metrics["counterfactual_replan_objective_sum"] = replan_objective[
                    eligible
                ].sum()
                metrics["counterfactual_progress_sum"] = feature[eligible, 0].sum()
                metrics["counterfactual_disagreement_sum"] = feature[eligible, 1].sum()
                metrics["counterfactual_oracle_replan_count"] = (
                    eligible & (advantage > 0)
                ).sum()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        total_loss = (
            lm_loss
            + 0.5 * (q_halt_loss + q_continue_loss)
            + self.feudal_loss_weight * feudal_loss_value
            + self.intervention_weight * intervention_loss
            + self.counterfactual_critic_weight * critic_loss
        )

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
