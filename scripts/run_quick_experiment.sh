#!/bin/bash
# Quick experiment runner - single experiment with feudal loss

cd "$(dirname "$0")/../HRM" || exit 1

# Default values
FEUDAL_WEIGHT=${1:-0.1}
MANAGER_PERIOD=${2:-4}
MAX_STEPS=${3:-50}
DEVICE=${4:-cpu}

echo "Running quick experiment:"
echo "  Feudal loss weight: $FEUDAL_WEIGHT"
echo "  Manager period: $MANAGER_PERIOD"
echo "  Max steps: $MAX_STEPS"
echo "  Device: $DEVICE"
echo ""

python pretrain.py \
    device=$DEVICE \
    max_steps=$MAX_STEPS \
    enable_wandb=false \
    global_batch_size=4 \
    loss.feudal_loss_weight=$FEUDAL_WEIGHT \
    arch.subgoal_head.manager_period=$MANAGER_PERIOD \
    arch.halt_max_steps=4 \
    arch.H_cycles=1 \
    arch.L_cycles=1 \
    arch.H_layers=1 \
    arch.L_layers=1 \
    arch.hidden_size=32 \
    arch.num_heads=2 \
    arch.expansion=2 \
    arch.puzzle_emb_ndim=32 \
    arch.subgoal_head.hidden_size=32 \
    arch.subgoal_head.goal_dim=32 \
    epochs=1 \
    lr_warmup_steps=0 \
    eval_interval=null \
    final_eval=false

