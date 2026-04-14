#!/usr/bin/env bash
# run_all.sh — full experiment suite
# Sequential: ~12-20h on RTX 4090. For parallel runs see RUNS.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

COMMON="--seeds 11 22 33 --attack-per-class 300 --search-steps 20
        --candidates-per-state 48 --beam-size 5 --epochs 8 --batch-size 128
        --max-tokens 256 --attack-search-group-size 128
        --allow-download --device cuda --resume"

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

echo "=== Ablation: consistency_weight sweep ==="
PYTHONUNBUFFERED=1 python -u experiments/sweep_ablation.py \
  --backbone bilstm \
  --seeds 11 22 33 \
  --attack-per-class 100 \
  --search-steps 20 \
  --candidates-per-state 48 \
  --beam-size 5 \
  --attack-search-group-size 128 \
  --allow-download --device cuda \
  --output experiments/results_ablation.json

echo "=== All experiments complete ==="
