# Confirmation Experiments Guide

This guide explains how to run confirmation experiments to validate findings and explore different manager periods.

## Overview

The `run_confirmation_experiments.py` script runs two types of experiments:

1. **Replication Experiments**: Multiple runs of the same configuration to confirm reproducibility
2. **Manager Period Sweep**: Test different manager periods with a fixed feudal weight

## Quick Start

### Run Both Experiments (Default)

```bash
cd scripts
python run_confirmation_experiments.py \
    --feudal-weight 0.05 \
    --manager-period 4 \
    --num-replications 5 \
    --manager-periods 1 2 3 4 5 6 8 \
    --max-steps 200 \
    --device cpu
```

Or use the convenience script:

```bash
cd scripts
./run_confirmation.sh
```

### Run Only Replication Experiments

```bash
python run_confirmation_experiments.py \
    --skip-manager-sweep \
    --feudal-weight 0.05 \
    --manager-period 4 \
    --num-replications 5 \
    --max-steps 200
```

### Run Only Manager Period Sweep

```bash
python run_confirmation_experiments.py \
    --skip-replications \
    --feudal-weight 0.05 \
    --manager-periods 1 2 3 4 5 6 8 \
    --max-steps 200
```

## Parameters

### Replication Experiments
- `--feudal-weight`: Feudal loss weight (default: 0.05)
- `--manager-period`: Manager period to replicate (default: 4)
- `--num-replications`: Number of replication runs (default: 5)

### Manager Period Sweep
- `--feudal-weight`: Fixed feudal loss weight (default: 0.05)
- `--manager-periods`: List of manager periods to test (default: 1 2 3 4 5 6 8)

### Common Parameters
- `--max-steps`: Maximum training steps (default: 200)
- `--device`: Device to use (`cpu` or `cuda`, default: `cpu`)
- `--enable-wandb`: Enable WandB logging
- `--no-wandb`: Disable WandB logging (default)
- `--output`: Custom output file path (default: auto-generated timestamp)

## Output

Results are saved to `experiments/confirmation_experiments_YYYYMMDD_HHMMSS.json` with:

1. **Replication Results**: Statistics including mean, min, max, std for:
   - LM Loss
   - Accuracy
   - Feudal Loss

2. **Manager Period Sweep Results**: Results for each manager period tested

The script also prints a summary table showing:
- Replication statistics
- Manager period comparison
- Best configuration found

## Example Output

```
EXPERIMENT SUMMARY
================================================================================

📊 REPLICATION EXPERIMENTS
--------------------------------------------------------------------------------
Successful: 5/5

LM Loss:
  Mean: 1.5686
  Min:  1.5600
  Max:  1.5800
  Std:  0.0080

Accuracy:
  Mean: 0.7104
  Min:  0.7050
  Max:  0.7150
  Std:  0.0040

📈 MANAGER PERIOD SWEEP RESULTS
--------------------------------------------------------------------------------
Results by Manager Period:
Period     LM Loss      Accuracy     Feudal Loss  
--------------------------------------------------
1          1.6200       0.6800       0.2500
2          1.5900       0.6950       0.2300
3          1.5750       0.7050       0.2100
4          1.5686       0.7104       0.2239
5          1.5700       0.7080       0.2000
6          1.5800       0.7020       0.1950
8          1.5900       0.6980       0.1900

🏆 Best Configuration:
  Manager Period: 4
  Feudal Weight: 0.05
  Accuracy: 0.7104
```

## Use Cases

### 1. Confirm Best Configuration
Run 5+ replications to verify that `feudal_weight=0.05, manager_period=4` consistently performs well:

```bash
python run_confirmation_experiments.py \
    --skip-manager-sweep \
    --num-replications 10 \
    --max-steps 200
```

### 2. Find Optimal Manager Period
Test different manager periods with the best feudal weight:

```bash
python run_confirmation_experiments.py \
    --skip-replications \
    --feudal-weight 0.05 \
    --manager-periods 1 2 3 4 5 6 8 10 12 \
    --max-steps 200
```

### 3. Full Confirmation Suite
Run both to get comprehensive results:

```bash
python run_confirmation_experiments.py \
    --feudal-weight 0.05 \
    --manager-period 4 \
    --num-replications 5 \
    --manager-periods 1 2 3 4 5 6 8 \
    --max-steps 200 \
    --device cuda \
    --enable-wandb
```

## Analyzing Results

After running experiments, analyze the JSON results:

```bash
python analyze_experiments.py \
    experiments/confirmation_experiments_*.json \
    --format table
```

Or load in Python:

```python
import json
from pathlib import Path

with open("experiments/confirmation_experiments_YYYYMMDD_HHMMSS.json") as f:
    results = json.load(f)

# Replication statistics
reps = results["replications"]
lm_losses = [r["metrics"]["final_lm_loss"] for r in reps if r.get("success")]

# Manager period comparison
mps = results["manager_period_sweep"]
best = max(mps, key=lambda x: x["metrics"].get("final_accuracy", 0))
```

