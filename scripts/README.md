# Training Experiments Scripts

This directory contains scripts for running training experiments and hyperparameter tuning for the feudal loss integration.

## Quick Start

### 1. Run a Single Quick Experiment

```bash
./run_quick_experiment.sh [feudal_weight] [manager_period] [max_steps] [device]
```

Example:
```bash
./run_quick_experiment.sh 0.1 4 50 cpu
```

### 2. Compare Baseline vs Feudal Loss

```bash
python run_baseline_comparison.py \
    --feudal-weight 0.1 \
    --manager-period 4 \
    --max-steps 100 \
    --device cpu
```

### 3. Sweep Feudal Loss Weights

```bash
python run_hyperparameter_sweep.py \
    --sweep-type feudal_weight \
    --feudal-weights 0.0 0.01 0.05 0.1 0.2 0.5 \
    --manager-period 4 \
    --max-steps 100 \
    --device cpu
```

### 4. Sweep Manager Periods

```bash
python run_hyperparameter_sweep.py \
    --sweep-type manager_period \
    --manager-periods 1 2 4 8 \
    --feudal-weight 0.1 \
    --max-steps 100 \
    --device cpu
```

### 5. Grid Search

```bash
python run_hyperparameter_sweep.py \
    --sweep-type grid \
    --feudal-weights 0.01 0.05 0.1 0.2 \
    --manager-periods 2 4 8 \
    --max-steps 100 \
    --device cpu
```

## Scripts Overview

### `run_quick_experiment.sh`
Quick single experiment runner with minimal config.

### `run_baseline_comparison.py`
Compare HRM baseline (no feudal loss) vs HRM with feudal loss.

### `run_hyperparameter_sweep.py`
Hyperparameter sweep script for feudal loss weight and manager period.

### `analyze_experiments.py`
Analyze and compare experiment results from JSON files.

### `validate_feudal_loss.py`
Validation script for feudal loss integration (unit tests).

## Output

Results are saved to `experiments/` directory:
- `baseline_comparison_*.json`: Baseline comparison results
- `sweep_*.json`: Hyperparameter sweep results

## Documentation

See `EXPERIMENTS_GUIDE.md` for detailed documentation.

