#!/usr/bin/env bash
# run_all.sh — full experiment suite (5 seeds)
# Sequential: ~20-30h on RTX 4090.
# Re-running with --resume will skip already-completed entries.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEEDS="11 22 33 44 55"

COMMON="--seeds $SEEDS --attack-per-class 300 --search-steps 20
        --candidates-per-state 48 --beam-size 5 --epochs 8 --batch-size 128
        --max-tokens 256 --attack-search-group-size 128
        --allow-download --device cuda --resume"

# ── Exp1: clean model baselines ───────────────────────────────────────────────
echo "=== Exp1-A: clean models vs official_wafamole ==="
PYTHONUNBUFFERED=1 python -u experiments/run_exp1.py \
  --backbones word_svc textcnn bilstm codebert \
  --operator-set official_wafamole \
  --output experiments/results_exp1_official.json \
  $COMMON

echo "=== Exp1-B: clean models vs advsqli ==="
PYTHONUNBUFFERED=1 python -u experiments/run_exp1.py \
  --backbones word_svc textcnn bilstm codebert \
  --operator-set advsqli \
  --output experiments/results_exp1_advsqli.json \
  $COMMON

# ── Exp2: paired consistency training ─────────────────────────────────────────
echo "=== Exp2-A: pair training vs official_wafamole ==="
PYTHONUNBUFFERED=1 python -u experiments/run_exp2.py \
  --backbones textcnn bilstm codebert \
  --methods clean_ce pair_ce pair_canonical \
  --require-pairs \
  --train-operator-set official_wafamole \
  --attack-operator-set official_wafamole \
  --codebert-lr 1e-4 \
  --output experiments/results_exp2_official.json \
  $COMMON

echo "=== Exp2-B: pair training vs advsqli (cross-operator) ==="
PYTHONUNBUFFERED=1 python -u experiments/run_exp2.py \
  --backbones textcnn bilstm codebert \
  --methods clean_ce pair_ce pair_canonical \
  --require-pairs \
  --train-operator-set official_wafamole \
  --attack-operator-set advsqli \
  --codebert-lr 1e-4 \
  --output experiments/results_exp2_advsqli.json \
  $COMMON

# ── Ablation: consistency_weight sweep ────────────────────────────────────────
ABL_COMMON="--seeds $SEEDS --attack-per-class 100 --search-steps 20
            --candidates-per-state 48 --beam-size 5
            --attack-search-group-size 128
            --allow-download --device cuda --resume"

echo "=== Ablation: BiLSTM ==="
PYTHONUNBUFFERED=1 python -u experiments/sweep_ablation.py \
  --backbone bilstm \
  --output experiments/results_ablation.json \
  $ABL_COMMON

echo "=== Ablation: TextCNN ==="
PYTHONUNBUFFERED=1 python -u experiments/sweep_ablation.py \
  --backbone textcnn \
  --output experiments/results_ablation_textcnn.json \
  $ABL_COMMON

echo "=== Ablation: CodeBERT ==="
PYTHONUNBUFFERED=1 python -u experiments/sweep_ablation.py \
  --backbone codebert \
  --output experiments/results_ablation_codebert.json \
  $ABL_COMMON

echo "=== All experiments complete ==="
