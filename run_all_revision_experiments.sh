#!/bin/bash
# =============================================================================
# Master Revision Runner — All Reviewer-Requested Experiments
# =============================================================================
# Run from: /Users/kkyn/Documents/Work/Quantum/qgnn
#
# This script runs all 3 revision experiments sequentially.
# Each experiment checkpoints progress, so you can Ctrl+C and resume safely.
#
# Estimated total runtime:
#   Experiment 1 (20-seed entanglement):  ~7.5 hours
#   Experiment 2 (multi-seed scaling):    ~25 min (without EQ-GNN)
#   Experiment 3 (learnable ω₀):          ~5 min
#   ─────────────────────────────────────────────
#   Total:                                ~8 hours
#
# You can run experiments individually — see instructions below.
# =============================================================================

set -e

PYTHON=".pixi/envs/default/bin/python"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "  REVISION EXPERIMENTS — $(date)"
echo "============================================================"
echo ""

# ── Experiment 1: 20-Seed Entanglement Ablation ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EXPERIMENT 1: 20-Seed Entanglement Ablation"
echo "  Addresses: R1 W-1, R3 W-3, DA MAJOR-3"
echo "  Output: results/entanglement_ablation_20seeds.json"
echo "  Est. time: ~7.5 hours"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON training/run_revision_entanglement_20seeds.py \
    --targets homo lumo gap \
    --epochs 15 \
    2>&1 | tee "results/revision_entanglement_${TIMESTAMP}.log"

echo ""

# ── Experiment 2: Multi-Seed Large-Scale Scaling ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EXPERIMENT 2: Multi-Seed Large-Scale Scaling (Table 5)"
echo "  Addresses: DA MAJOR-4, R3"
echo "  Output: results/p21_large_scale_eval_multiseed.json"
echo "  Est. time: ~25 min (classical only)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON training/run_revision_multiseed_scaling.py \
    --n-values 5000 10000 \
    --targets homo lumo gap \
    --seeds 42 43 44 \
    --epochs 20 \
    2>&1 | tee "results/revision_scaling_${TIMESTAMP}.log"

echo ""

# ── Experiment 3: Learnable ω₀ ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EXPERIMENT 3: Learnable ω₀ for Periodic-EGNN"
echo "  Addresses: R1 Q-1, R3 W-2, EIC Q-1"
echo "  Output: results/revision_learnable_omega.json"
echo "  Est. time: ~5 min"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON training/run_revision_learnable_omega.py \
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
