#!/bin/bash
# Run Experiment E: Parameter Capacity Scaling (XS -> XXL, 6 tiers)
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON=".pixi/envs/default/bin/python"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="results/experiment_e_${TIMESTAMP}.log"

echo "============================================================"
echo "  EXPERIMENT E: EXTENDED PARAMETER SCALING (6 Tiers, 5 Seeds)"
echo "  Starting at $(date)"
echo "  Log: $LOG_FILE"
echo "============================================================"

$PYTHON training/run_capacity_scaling_6tiers.py --force "$@" 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "  Regenerating Table and Figure..."
echo "============================================================"
$PYTHON visualization/plot_figure_s9_capacity_scaling.py

echo "All done! Results saved to results/experiment_e_param_scaling.json and table/figure updated."
