from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


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

    # Normalize for cosine similarity
    worker_norm = F.normalize(worker_repr, p=2, dim=-1, eps=1e-8)
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

    # Reduce
    if reduction == "sum":
        return feudal_loss_per_sample.sum()
    elif reduction == "mean":
        return feudal_loss_per_sample.mean()
    elif reduction == "none":
        return feudal_loss_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class ACTLossHead(nn.Module):
    def __init__(
        self, model: nn.Module, loss_type: str, feudal_loss_weight: float = 0.1
    ):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.feudal_loss_weight = feudal_loss_weight

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
        if "subgoal_goal" in outputs and "worker_hidden" in outputs:
            worker_state = outputs["worker_hidden"]
            manager_goal = outputs["subgoal_goal"]
            gate = outputs.get("subgoal_gate")
            feudal_loss_value = feudal_loss(
                worker_state=worker_state,
                manager_goal=manager_goal,
                gate=gate,
                reduction="sum",
            )
            metrics["feudal_loss"] = feudal_loss_value.detach()

            # Subgoal metrics (as per NEXT_STEPS.md)
            with torch.no_grad():
                # Subgoal update frequency
                subgoal_updated = outputs.get("subgoal_updated")
                if subgoal_updated is not None:
                    metrics["subgoal_update_frequency"] = subgoal_updated.sum()

                # Subgoal gate mean (commitment strength)
                if gate is not None:
                    if gate.dim() > 1:
                        gate_flat = gate.squeeze(-1)
                    else:
                        gate_flat = gate
                    metrics["subgoal_gate_mean"] = gate_flat.mean()
                    metrics["subgoal_gate_std"] = gate_flat.std()

                # Subgoal goal norm (magnitude)
                goal_norm = torch.norm(manager_goal, p=2, dim=-1)
                metrics["subgoal_goal_norm_mean"] = goal_norm.mean()
                metrics["subgoal_goal_norm_std"] = goal_norm.std()

                # Worker-goal alignment (cosine similarity)
                if worker_state.dim() == 3:
                    worker_repr = worker_state.mean(dim=1)  # [B, T, D] -> [B, D]
                else:
                    worker_repr = worker_state  # [B, D]
                worker_norm = F.normalize(worker_repr, p=2, dim=-1, eps=1e-8)
                goal_norm_vec = F.normalize(manager_goal, p=2, dim=-1, eps=1e-8)
                alignment = (worker_norm * goal_norm_vec).sum(dim=-1)
                metrics["subgoal_worker_alignment_mean"] = alignment.mean()
                metrics["subgoal_worker_alignment_std"] = alignment.std()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        total_loss = (
            lm_loss
            + 0.5 * (q_halt_loss + q_continue_loss)
            + self.feudal_loss_weight * feudal_loss_value
        )

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
