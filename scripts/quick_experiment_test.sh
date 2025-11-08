#!/bin/bash
# Quick test script for experiment infrastructure

cd "$(dirname "$0")/../HRM" || exit 1

echo "Testing baseline comparison (quick test)..."
echo ""

# Test baseline comparison with minimal steps
python ../scripts/run_baseline_comparison.py \
    --feudal-weight 0.1 \
    --manager-period 2 \
    --max-steps 5 \
    --device cpu \
    --output ../experiments/test_baseline.json

echo ""
echo "✅ Baseline comparison test completed!"
echo ""

# Test feudal weight sweep with minimal steps
echo "Testing feudal weight sweep (quick test)..."
echo ""

python ../scripts/run_hyperparameter_sweep.py \
    --sweep-type feudal_weight \
    --feudal-weights 0.0 0.1 \
    --manager-period 2 \
    --max-steps 5 \
    --device cpu \
    --output ../experiments/test_sweep.json

echo ""
echo "✅ Hyperparameter sweep test completed!"
echo ""

# Analyze results
echo "Analyzing test results..."
echo ""

python ../scripts/analyze_experiments.py \
    ../experiments/test_baseline.json \
    ../experiments/test_sweep.json \
    --format table

echo ""
echo "✅ All tests completed!"

