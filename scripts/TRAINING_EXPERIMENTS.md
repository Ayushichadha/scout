# Training Experiments Guide

This guide explains how to run training experiments with the feudal loss integration.

## Quick Start

### Option 1: Using the shell script (easiest)

```bash
cd scripts
./run_training.sh
```

This runs a quick CPU test with:
- Small model (32 hidden size, 1 layer)
- 10 training steps
- Feudal loss enabled (weight: 0.1)
- WandB disabled

### Option 2: Using Hydra directly

```bash
cd HRM
python pretrain.py device=cpu max_steps=10 enable_wandb=false global_batch_size=4
```

### Option 3: Using config overrides

```bash
cd HRM
python pretrain.py --config-name=cfg_pretrain \
    device=cpu \
    max_steps=10 \
    enable_wandb=false \
    global_batch_size=4 \
    loss.feudal_loss_weight=0.1
```

## Key Configuration Parameters

### Feudal Loss
- `loss.feudal_loss_weight`: Weight for feudal loss (default: 0.1)
  - Higher values = stronger intrinsic reward signal
  - Lower values = more focus on task loss

### Subgoal Head
- `arch.subgoal_head.manager_period`: Steps between manager goal updates (default: 4)
  - Lower = more frequent goal updates
  - Higher = more stable goals

- `arch.subgoal_head.goal_dim`: Dimension of goal vectors (default: hidden_size)
  - Should match or be smaller than hidden_size

- `arch.subgoal_head.gating`: Enable gating signal (default: true)
  - Controls commitment strength to goals

### Model Architecture
- `arch.halt_max_steps`: Maximum ACT steps (default: 16)
- `arch.H_cycles`: High-level reasoning cycles (default: 2)
- `arch.L_cycles`: Low-level reasoning cycles (default: 2)

## Monitoring Training

### Metrics to Watch

1. **Task Loss**: `train/lm_loss` - Main task loss
2. **Feudal Loss**: `train/feudal_loss` - Intrinsic reward loss
3. **Total Loss**: Combined task + feudal loss
4. **Accuracy**: `train/accuracy` - Task accuracy
5. **Steps**: `train/steps` - Average ACT steps taken

### WandB Integration

Enable WandB logging:
```bash
python pretrain.py enable_wandb=true project_name=scout run_name=feudal_exp_1
```

## Experiment Ideas

### 1. Feudal Loss Weight Sweep
Test different feudal loss weights:
```bash
for weight in 0.01 0.05 0.1 0.2 0.5; do
    python pretrain.py loss.feudal_loss_weight=$weight run_name=feudal_weight_$weight
done
```

### 2. Manager Period Sweep
Test different manager update frequencies:
```bash
for period in 1 2 4 8; do
    python pretrain.py arch.subgoal_head.manager_period=$period run_name=period_$period
done
```

### 3. Ablation Studies
- Disable feudal loss: `loss.feudal_loss_weight=0`
- Disable subgoal head: Remove `arch.subgoal_head` from config
- Disable gating: `arch.subgoal_head.gating=false`

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `global_batch_size` or `arch.halt_max_steps`
2. **Slow Training**: Use `device=cuda` if available, or reduce model size
3. **Feudal Loss Not Appearing**: Check that `arch.subgoal_head` is in config

### Debug Mode

Run with minimal steps for debugging:
```bash
python pretrain.py device=cpu max_steps=5 enable_wandb=false global_batch_size=2
```

