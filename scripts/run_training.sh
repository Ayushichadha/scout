#!/bin/bash
# Quick training script for feudal loss experiments

cd "$(dirname "$0")/../HRM" || exit 1

# Quick CPU test with feudal loss enabled
python pretrain.py \
    device=cpu \
    max_steps=10 \
    enable_wandb=false \
    global_batch_size=4 \
    epochs=1 \
    lr_warmup_steps=0 \
    eval_interval=null \
    final_eval=false \
    arch.halt_max_steps=4 \
    arch.H_cycles=1 \
    arch.L_cycles=1 \
    arch.H_layers=1 \
    arch.L_layers=1 \
    arch.hidden_size=32 \
    arch.num_heads=2 \
    arch.expansion=2 \
    arch.puzzle_emb_ndim=32 \
    loss.feudal_loss_weight=0.1 \
    arch.subgoal_head.hidden_size=32 \
    arch.subgoal_head.goal_dim=32 \
    arch.subgoal_head.manager_period=2

