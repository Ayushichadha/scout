#!/usr/bin/env python3
"""Dump per-example hidden states for CompLearn workshop ablation.

Usage:
    python scripts/dump_hidden_states.py \\
        --checkpoint "checkpoints/<project>/<run>/step_<N>" \\
        --config-name cfg_pretrain \\
        --output dumps/<run>/step_<N>.pt \\
        [hydra overrides...]

Hydra overrides are forwarded verbatim to the config, allowing flag combinations
to be reproduced exactly (e.g. arch.subgoal_head.inject_subgoal=false for a
cell-D ablation checkpoint).

Output .pt structure:
    worker_hidden    [200, hidden_size]   float32 — final z_L, mean-pooled over seq
    subgoal_goal     [200, goal_dim]      float32 — manager goal at final ACT step
    solved           [200]                bool    — exact-match correctness
    puzzle_identifier [200]               long    — stable puzzle ID from dataset
    example_index    [200]                long    — 0..199 position in subset
    metadata         dict                 — checkpoint path, config, param count, etc.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Belt-and-suspenders seed before any random ops (dropout off in eval mode, but be safe)
torch.manual_seed(42)

# Add HRM/ to sys.path so HRM modules resolve — matches convention from CLAUDE.md
HRM_ROOT = Path(__file__).resolve().parent.parent / "HRM"
sys.path.insert(0, str(HRM_ROOT))

# Skip torch.compile: not needed for single-pass eval, saves ~30 s startup
os.environ.setdefault("DISABLE_COMPILE", "1")

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from pretrain import PretrainConfig, create_dataloader  # noqa: E402
from utils.functions import load_model_class  # noqa: E402
from models.losses import IGNORE_LABEL_ID  # noqa: E402

SUBSET_SIZE = 200


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Dump HRM hidden states for ablation analysis"
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint state-dict file (e.g. checkpoints/.../step_1000)",
    )
    p.add_argument(
        "--config-name",
        required=True,
        dest="config_name",
        help="Hydra config name (e.g. cfg_pretrain)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output .pt file path (parent dirs created automatically)",
    )
    p.add_argument(
        "--device", default=None, help="Device (default: cuda if available, else cpu)"
    )
    # Everything after the known flags is treated as Hydra overrides
    args, overrides = p.parse_known_args()
    return args, overrides


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(config_name: str, overrides: list[str], device: str) -> PretrainConfig:
    GlobalHydra.instance().clear()
    all_overrides = [*overrides, f"device={device}", "enable_wandb=false"]
    with initialize_config_dir(config_dir=str(HRM_ROOT / "config"), version_base=None):
        hydra_cfg = compose(config_name=config_name, overrides=all_overrides)
        # resolve=True expands OmegaConf interpolations (${..hidden_size} etc.)
        # before Pydantic sees them, so no manual interpolation resolution is needed.
        container = OmegaConf.to_container(hydra_cfg, resolve=True)
    config = PretrainConfig(**container)  # type: ignore[arg-type]

    # Hydra compose does not chdir the way @hydra.main does; resolve any relative
    # data_path against HRM_ROOT so the script can be run from the repo root.
    if not os.path.isabs(config.data_path):
        config.data_path = str(HRM_ROOT / config.data_path)

    return config


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(config: PretrainConfig, eval_metadata, device: str):
    """Replicate create_model() from pretrain.py — model + loss head, no optimizers or compile."""
    # arch.__pydantic_extra__ holds all fields beyond {name, loss} — already resolved
    arch_extra = dict(config.arch.__pydantic_extra__ or {})
    loss_extra = dict(config.arch.loss.__pydantic_extra__ or {})

    model_cfg = {
        **arch_extra,
        # batch_size must match training so the CastedSparseEmbedding state dict
        # shape (local_weights/local_ids) aligns with the checkpoint.
        # In eval mode CastedSparseEmbedding ignores these buffers, so we can
        # safely pass smaller batches at forward time.
        "batch_size": config.global_batch_size,
        "vocab_size": eval_metadata.vocab_size,
        "seq_len": eval_metadata.seq_len,
        "num_puzzle_identifiers": eval_metadata.num_puzzle_identifiers,
    }

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    model = loss_head_cls(model_cls(model_cfg), **loss_extra)

    if device == "cuda":
        model = model.cuda()
    return model


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # torch.compile wraps the model in OptimizedModule; .state_dict() may include
    # an _orig_mod. prefix in some PyTorch builds — strip it if present.
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # CastedSparseEmbedding.local_weights / local_ids are sized to training batch_size.
    # In eval mode forward() bypasses them entirely (returns self.weights[inputs]).
    # A shape mismatch here is safe to tolerate — it only means the model was built
    # with a different batch_size than training, which is intentional for dump scripts.
    eval_safe_skip = {
        k for k in missing_keys if k.endswith((".local_weights", ".local_ids"))
    }
    real_missing = [k for k in missing_keys if k not in eval_safe_skip]
    if real_missing:
        raise RuntimeError(f"Checkpoint is missing required keys: {real_missing}")
    if unexpected_keys:
        raise RuntimeError(f"Checkpoint has unexpected keys: {unexpected_keys}")
    if eval_safe_skip:
        print(
            f"  (skipped {len(eval_safe_skip)} batch-local puzzle-emb buffers — eval-safe)"
        )


# ---------------------------------------------------------------------------
# Forward pass (full ACT loop)
# ---------------------------------------------------------------------------


def run_act_forward(model: torch.nn.Module, batch: dict) -> dict:
    """Run the ACT loop to completion, returning the final step's outputs.

    With model.eval(), the Q-halt noise is suppressed and all examples run for
    exactly halt_max_steps iterations — deterministic across calls.
    """
    with torch.no_grad():
        carry = model.initial_carry(batch)
        final_outputs: dict | None = None
        while True:
            carry, _, _, final_outputs, all_finish = model(
                carry=carry,
                batch=batch,
                return_keys=["worker_hidden", "subgoal_goal", "logits"],
            )
            if all_finish:
                break
    return final_outputs  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args, overrides = parse_args()
    device: str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device  : {device}")
    print(f"Config  : {args.config_name}  overrides: {overrides or '(none)'}")
    print(f"Ckpt    : {args.checkpoint}")

    config = load_config(args.config_name, overrides, device)

    # Build eval dataloader — metadata (vocab_size, seq_len, …) required before model init
    eval_loader, eval_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )

    model = build_model(config, eval_metadata, device)
    load_checkpoint(model, args.checkpoint, device)
    # eval mode: ACT halts at exactly halt_max_steps steps — no Q-exploration noise
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Params  : {param_count:,}")
    print(f"Collecting first {SUBSET_SIZE} examples (deterministic file order)…")

    worker_hiddens: list[torch.Tensor] = []
    subgoal_goals: list[torch.Tensor] = []
    solved_labels: list[torch.Tensor] = []
    puzzle_ids: list[torch.Tensor] = []
    total_collected = 0

    for _set_name, batch, _global_bs in eval_loader:
        if total_collected >= SUBSET_SIZE:
            break

        # deterministic first-200 subset; eval split is in arbitrary file order, not sorted
        n = min(batch["inputs"].shape[0], SUBSET_SIZE - total_collected)
        batch = {k: v[:n].to(device) for k, v in batch.items()}

        outputs = run_act_forward(model, batch)

        # Worker hidden: mean-pool over seq dim if 3D — matches feudal_loss() in losses.py:75
        # z_L shape is [B, seq_len + puzzle_emb_len, hidden_size]
        worker_h = outputs["worker_hidden"].to(torch.float32)
        if worker_h.dim() == 3:
            worker_h = worker_h.mean(dim=1)  # [B, T, D] -> [B, D]

        # Subgoal: [B, goal_dim] from outputs dict; zeros if no subgoal head configured
        if "subgoal_goal" in outputs:
            subgoal_g = outputs["subgoal_goal"].to(torch.float32)
        else:
            subgoal_g = torch.zeros(
                n, worker_h.shape[-1], dtype=torch.float32, device=device
            )

        # seq_is_correct: exact-match per example — replicates losses.py:140-147 exactly
        logits = outputs["logits"].to(torch.float32)
        labels = batch["labels"]
        mask = labels != IGNORE_LABEL_ID
        is_correct = mask & (logits.argmax(-1) == labels)
        seq_is_correct = is_correct.sum(-1) == mask.sum(-1)

        worker_hiddens.append(worker_h.cpu())
        subgoal_goals.append(subgoal_g.cpu())
        solved_labels.append(seq_is_correct.cpu())
        puzzle_ids.append(batch["puzzle_identifiers"].cpu())
        total_collected += n

    assert (
        total_collected == SUBSET_SIZE
    ), f"Only collected {total_collected} examples; dataset may be too small"

    worker_hidden_all = torch.cat(worker_hiddens, dim=0)  # [200, hidden_size]
    subgoal_goal_all = torch.cat(subgoal_goals, dim=0)  # [200, goal_dim]
    solved_all = torch.cat(solved_labels, dim=0).bool()  # [200]
    puzzle_id_all = torch.cat(puzzle_ids, dim=0).long()  # [200]
    example_index_all = torch.arange(SUBSET_SIZE, dtype=torch.long)  # [200]

    payload = {
        "worker_hidden": worker_hidden_all,
        "subgoal_goal": subgoal_goal_all,
        "solved": solved_all,
        "puzzle_identifier": puzzle_id_all,
        "example_index": example_index_all,
        "metadata": {
            "checkpoint_path": str(args.checkpoint),
            "config_name": args.config_name,
            "hydra_overrides": overrides,
            "model_param_count": param_count,
            "eval_data_path": config.data_path,
            "subset_size": SUBSET_SIZE,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    n_solved = solved_all.sum().item()
    hidden_norms = worker_hidden_all.norm(dim=-1)
    subgoal_norms = subgoal_goal_all.norm(dim=-1)
    print(f"Dumped {SUBSET_SIZE} examples from {args.checkpoint} to {args.output}")
    print(f"Solved: {n_solved}/{SUBSET_SIZE} ({100 * n_solved / SUBSET_SIZE:.1f}%)")
    print(
        f"Hidden state norm range: [{hidden_norms.min():.3f}, {hidden_norms.max():.3f}]"
    )
    print(f"Subgoal norm range: [{subgoal_norms.min():.3f}, {subgoal_norms.max():.3f}]")


if __name__ == "__main__":
    main()
