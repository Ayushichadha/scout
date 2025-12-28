# Visualization Scripts

This directory contains scripts for generating publication-quality plots for the Feudal HRM experiments.

## Main Plots (4 Total)

The experiment analysis produces **4 main plots**:

1. **`manager_period_sweep_lm_loss.png`** - Manager period sweep showing the optimal period
2. **`feudal_weight_sweep_lm_loss.png`** - Feudal weight sweep (loss metric)
3. **`feudal_weight_sweep_accuracy.png`** - Feudal weight sweep (accuracy metric)
4. **`replication_distribution_lm_loss.png`** - Statistical comparison of baseline vs best-feudal

---

## Shared Style Module

All plots use a shared style configuration for consistency:

### `plot_style.py`

Provides:
- Consistent figure size (10×6 inches)
- 300 DPI export
- Professional font sizes (title: 17, axis labels: 13, ticks: 11)
- Clean white background with subtle grid
- Unified color palette
- Standard label formatting for baseline and λ=0 points

---

## Manager Period Sweep Signature Plot

### Script: `plot_manager_period_sweep.py`

Creates the signature plot showing the non-monotonic "sweet spot" at manager period ≈ 3.

### Usage

```bash
python3 scripts/plot_manager_period_sweep.py \
    --sweep-file experiments/confirmation_experiments_20251119_183610.json \
    --baseline-files experiments/baseline_comparison_20251119_173527.json \
                     experiments/baseline_comparison_20251119_173745.json \
    --output-dir experiments/plots
```

### Features

- **X-axis**: Manager Period (1, 2, 3, 4, 5, 6, 8)
- **Y-axis**: Language Model Loss (lower is better)
- **Baseline reference**: Dashed line labeled "HRM baseline (no subgoal head)"
- **Feudal curve**: Line plot with error bars
- **Optimal point**: Star marker with annotation (no legend entry)
- **Footnote**: Sample sizes and single-run periods noted

### Output

- `experiments/plots/manager_period_sweep_lm_loss.png`

---

## Feudal Weight Sweep Plot

### Script: `plot_feudal_weight_sweep.py`

Shows the "light helps, heavy hurts" pattern for feudal loss weight.

### Usage

```bash
python3 scripts/plot_feudal_weight_sweep.py \
    --sweep-files experiments/confirmation_experiments_20251119_183610.json \
    --baseline-files experiments/baseline_comparison_20251119_173527.json \
                     experiments/baseline_comparison_20251119_173745.json \
    --primary-period 3 \
    --output-dir experiments/plots
```

### Features

- **X-axis**: Feudal Loss Weight (λ)
- **Y-axis**: Loss or Accuracy
- **Baseline reference**: Dashed line labeled "HRM baseline (no subgoal head)"
- **λ=0 point**: Square marker labeled "Subgoal head only (λ=0)"
- **Feudal curve**: Line plot for λ > 0
- **Optimal point**: Star marker with annotation (no legend entry)
- **Footnote**: Sample sizes noted

### Output

- `experiments/plots/feudal_weight_sweep_lm_loss.png`
- `experiments/plots/feudal_weight_sweep_accuracy.png`

---

## Replication Distribution Plot

### Script: `plot_replication_distribution.py`

Shows statistical comparison between baseline and best-feudal configurations.

### Usage

```bash
python3 scripts/plot_replication_distribution.py \
    --baseline-files experiments/baseline_comparison_20251119_173527.json \
                     experiments/baseline_comparison_20251119_173745.json \
    --feudal-replication-file experiments/confirmation_experiments_20251119_183610.json \
    --feudal-period 3 \
    --feudal-weight 0.05 \
    --output-dir experiments/plots
```

### Features

- **Groups**: HRM Baseline vs Best-Feudal (P=3, λ=0.05)
- **Violin + box plot**: Shows distribution shape
- **Swarm points**: Individual run values
- **Statistics**: Compact annotation with n, mean±std, p-value

### Output

- `experiments/plots/replication_distribution_lm_loss.png`

---

## Style Conventions

All plots follow these conventions:

| Element | Specification |
|---------|---------------|
| Figure size | 10×6 inches |
| DPI | 300 |
| Title font | 17pt, bold |
| Axis label font | 13pt, bold |
| Tick font | 11pt |
| Line width | 2.2 (data), 2.0 (reference) |
| Marker size | 8 (data), 14 (highlight) |
| Background | White |
| Grid | Subtle gray (#E5E7EB), 0.5pt |

### Label Standards

- **Baseline**: "HRM baseline (no subgoal head)"
- **λ=0 point**: "Subgoal head only (λ=0)"
- **Optimal**: Star marker with annotation (excluded from legend)

---

## Quick Regeneration

To regenerate all 4 plots:

```bash
cd /Users/ayushi/Documents/hrm_research

# Manager period sweep
python3 scripts/plot_manager_period_sweep.py \
    --sweep-file experiments/confirmation_experiments_20251119_183610.json \
    --baseline-files experiments/baseline_comparison_*.json \
    --output-dir experiments/plots

# Feudal weight sweep (generates both loss and accuracy)
python3 scripts/plot_feudal_weight_sweep.py \
    --sweep-files experiments/confirmation_experiments_20251119_183610.json \
    --baseline-files experiments/baseline_comparison_*.json \
    --output-dir experiments/plots

# Replication distribution
python3 scripts/plot_replication_distribution.py \
    --baseline-files experiments/baseline_comparison_*.json \
    --feudal-replication-file experiments/confirmation_experiments_20251119_183610.json \
    --output-dir experiments/plots
```

### Expected Output Files

```
experiments/plots/
├── manager_period_sweep_lm_loss.png
├── feudal_weight_sweep_lm_loss.png
├── feudal_weight_sweep_accuracy.png
└── replication_distribution_lm_loss.png
```
