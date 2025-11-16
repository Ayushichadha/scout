# Visualization Scripts

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
- **Y-axis**: Primary metric (Language Model Loss or Accuracy)
- **Baseline reference**: Horizontal dashed line showing baseline (no feudal) performance
- **Feudal curve**: Line plot with error bars (std or 95% CI if multiple seeds available)
- **Sweet spot highlight**: Star marker at period=3 with annotation
- **Separate plots**: Generates separate plots for Loss and Accuracy (if data available)

### Output

- `experiments/plots/manager_period_sweep_lm_loss.png` - Loss plot (main signature plot)
- `experiments/plots/manager_period_sweep_accuracy.png` - Accuracy plot (if accuracy data available)

### Key Visual Elements

1. **Baseline reference line**: Gray dashed horizontal line showing performance without feudal loss
2. **Feudal curve**: Purple line with markers showing performance across manager periods
3. **Error bars**: Standard deviation or 95% confidence intervals (if multiple seeds)
4. **Sweet spot marker**: Gold star at period=3 with annotation showing best performance
5. **Error bands**: Shaded region around baseline if multiple baseline runs available

### Data Requirements

The script expects:
- Manager period sweep results with `manager_period` and `metrics.final_lm_loss` (and optionally `final_accuracy`)
- Baseline results with `use_feudal=false` or `feudal_loss_weight=0.0` and corresponding metrics

### Notes

- The script automatically handles single-seed vs multi-seed data
- If accuracy data is not available in the sweep, the accuracy plot is skipped
- The script uses scipy for 95% CI calculation if available, otherwise falls back to std

