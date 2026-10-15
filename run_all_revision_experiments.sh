#!/bin/bash
# Master runner for revision experiments.

set -e

PYTHON=".pixi/envs/default/bin/python"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "  REVISION EXPERIMENTS — $(date)"
echo "============================================================"
echo ""

# Experiment 1: 20-Seed Entanglement Ablation
echo "------------------------------------------------------------"
echo "  EXPERIMENT 1: 20-Seed Entanglement Ablation"
echo "  Output: results/entanglement_ablation_20seeds.json"
echo "  Est. time: ~7.5 hours"
echo "------------------------------------------------------------"
$PYTHON training/run_table2_entanglement_ablation.py \
    --targets homo lumo gap \
    --epochs 15 \
    2>&1 | tee "results/revision_entanglement_${TIMESTAMP}.log"

echo ""

# Experiment 2: Multi-Seed Large-Scale Scaling
echo "------------------------------------------------------------"
echo "  EXPERIMENT 2: Multi-Seed Large-Scale Scaling (Table 5)"
echo "  Output: results/p21_large_scale_eval_multiseed.json"
echo "  Est. time: ~25 min (classical only)"
echo "------------------------------------------------------------"
$PYTHON training/run_p21_large_scale_eval.py \
    --n-values 5000 10000 \
    --targets homo lumo gap \
    --seeds 42 43 44 \
    --epochs 20 \
    2>&1 | tee "results/revision_scaling_${TIMESTAMP}.log"

echo ""

# Experiment 3: Learnable ω₀
echo "------------------------------------------------------------"
echo "  EXPERIMENT 3: Learnable ω₀ for Periodic-EGNN"
echo "  Output: results/revision_learnable_omega.json"
echo "  Est. time: ~5 min"
echo "------------------------------------------------------------"
$PYTHON training/run_learnable_omega_study.py \
    --targets homo lumo gap \
    --seeds 42 43 44 \
    --epochs 15 \
    2>&1 | tee "results/revision_omega_${TIMESTAMP}.log"

echo ""
echo "============================================================"
echo "  ALL REVISION EXPERIMENTS COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  1. results/entanglement_ablation_20seeds.json"
echo "  2. results/p21_large_scale_eval_multiseed.json"
echo "  3. results/revision_learnable_omega.json"
echo ""
echo "Logs saved to:"
echo "  results/revision_entanglement_${TIMESTAMP}.log"
echo "  results/revision_scaling_${TIMESTAMP}.log"
echo "  results/revision_omega_${TIMESTAMP}.log"
