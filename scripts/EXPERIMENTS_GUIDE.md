# Training Experiments and Hyperparameter Tuning Guide

This guide explains how to run training experiments and hyperparameter sweeps for the feudal loss integration.

## Quick Start

### 1. Baseline Comparison (With vs Without Feudal Loss)

Compare HRM baseline (no feudal loss) vs HRM with feudal loss:

```bash
cd scripts
python run_baseline_comparison.py \
    --feudal-weight 0.1 \
    --manager-period 4 \
    --max-steps 100 \
    --device cpu
```

### 2. Feudal Loss Weight Sweep

Test different feudal loss weights:

```bash
python run_hyperparameter_sweep.py \
    --sweep-type feudal_weight \
    --feudal-weights 0.0 0.01 0.05 0.1 0.2 0.5 \
    --manager-period 4 \
    --max-steps 100 \
    --device cpu
```

### 3. Manager Period Sweep

Test different manager update frequencies:

```bash
python run_hyperparameter_sweep.py \
    --sweep-type manager_period \
    --manager-periods 1 2 4 8 \
    --feudal-weight 0.1 \
    --max-steps 100 \
    --device cpu
```

### 4. Grid Search

Test all combinations of feudal loss weights and manager periods:

```bash
python run_hyperparameter_sweep.py \
    --sweep-type grid \
    --feudal-weights 0.01 0.05 0.1 0.2 \
    --manager-periods 2 4 8 \
    --max-steps 100 \
    --device cpu
```

## Analyzing Results

### Analyze Experiment Results

```bash
python analyze_experiments.py \
    experiments/sweep_feudal_weight_*.json \
    experiments/baseline_comparison_*.json \
    --format table
```

### With WandB

Enable WandB logging for better visualization:

```bash
python run_hyperparameter_sweep.py \
    --sweep-type feudal_weight \
    --feudal-weights 0.0 0.05 0.1 0.2 \
    --enable-wandb \
    --device cuda
```

Then view results in WandB dashboard.

## Experiment Configuration

### Key Hyperparameters

1. **Feudal Loss Weight** (`loss.feudal_loss_weight`)
   - Range: 0.0 (disabled) to 0.5+
   - Default: 0.1
   - Effect: Controls strength of intrinsic reward signal

2. **Manager Period** (`arch.subgoal_head.manager_period`)
   - Range: 1 (every step) to 8+ (less frequent)
   - Default: 4
   - Effect: How often manager updates subgoals

3. **Goal Dimension** (`arch.subgoal_head.goal_dim`)
   - Default: `hidden_size`
   - Effect: Dimensionality of goal vectors

4. **Gating** (`arch.subgoal_head.gating`)
   - Default: `true`
   - Effect: Whether to use gating signal for goal commitment

### Training Parameters

- `max_steps`: Number of training steps (default: 100 for quick tests)
- `global_batch_size`: Batch size (default: 768 for full training)
- `device`: `cpu` or `cuda`
- `enable_wandb`: Enable WandB logging

## Experiment Workflow

### Step 1: Quick Validation

Run a quick test to ensure everything works:

```bash
cd HRM
python pretrain.py \
    device=cpu \
    max_steps=10 \
    enable_wandb=false \
    global_batch_size=2 \
    loss.feudal_loss_weight=0.1
```

### Step 2: Baseline Comparison

Compare with and without feudal loss:

```bash
cd scripts
python run_baseline_comparison.py \
    --max-steps 100 \
    --device cpu
```

### Step 3: Hyperparameter Sweep

Find optimal hyperparameters:

```bash
python run_hyperparameter_sweep.py \
    --sweep-type feudal_weight \
    --feudal-weights 0.0 0.01 0.05 0.1 0.2 \
    --max-steps 200 \
    --device cuda \
    --enable-wandb
```

### Step 4: Full Training

Run full training with best hyperparameters:

```bash
cd HRM
python pretrain.py \
    device=cuda \
    global_batch_size=768 \
    epochs=100000 \
    loss.feudal_loss_weight=0.1 \
    arch.subgoal_head.manager_period=4 \
    enable_wandb=true
```

### Step 5: Analysis

Analyze results:

```bash
cd scripts
python analyze_experiments.py \
    experiments/*.json \
    --format table
```

## Expected Results

### Metrics to Monitor

1. **Task Loss** (`train/lm_loss`)
   - Should decrease over training
   - Compare baseline vs feudal

2. **Feudal Loss** (`train/feudal_loss`)
   - Should decrease (worker getting closer to goals)
   - Monitor trend over training

3. **Accuracy** (`train/accuracy`, `train/exact_accuracy`)
   - Primary metric for task performance
   - Compare baseline vs feudal

4. **Steps** (`train/steps`)
   - Average ACT steps taken
   - May change with feudal loss

### What to Look For

- **Feudal loss decreasing**: Worker is making progress toward goals
- **Task accuracy improving**: Feudal loss is helping task performance
- **Optimal weight**: Balance between task loss and feudal loss
- **Optimal period**: Manager updates at right frequency

## Troubleshooting

### Experiments Failing

1. Check device availability: `python -c "import torch; print(torch.cuda.is_available())"`
2. Reduce batch size: `global_batch_size=2`
3. Reduce max steps: `max_steps=10`
4. Check memory: Use CPU if GPU OOM

### Results Not Saving

1. Check `experiments/` directory exists
2. Check write permissions
3. Verify JSON output format

### WandB Not Logging

1. Check WandB login: `wandb login`
2. Verify `enable_wandb=true`
3. Check network connection

## Next Steps

After finding optimal hyperparameters:

1. Run full training with best config
2. Evaluate on test sets (ARC, Sudoku, Maze)
3. Compare with baseline HRM
4. Analyze feudal loss impact
5. Visualize subgoal learning

