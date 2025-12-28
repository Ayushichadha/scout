#!/bin/bash
# Quick script to run confirmation experiments

cd "$(dirname "$0")"

# Default: run both replications and manager period sweep
# Use --skip-replications or --skip-manager-sweep to skip either

python run_confirmation_experiments.py \
    --feudal-weight 0.05 \
    --manager-period 4 \
    --num-replications 5 \
    --manager-periods 1 2 3 4 5 6 8 \
    --max-steps 200 \
    --device cpu \
    "$@"

