#!/usr/bin/env bash
# smoke.sh — quick end-to-end check (< 10 min on GPU)
# Runs each experiment with 1 seed and minimal attack budget to catch import/data errors.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

COMMON="--seeds 11 --attack-per-class 5 --search-steps 3 --candidates-per-state 8
        --beam-size 2 --epochs 2 --batch-size 32 --max-tokens 256
        --attack-search-group-size 32 --allow-download --device cuda"

echo "=== Smoke: Exp1 official_wafamole ==="
PYTHONUNBUFFERED=1 python -u experiments/run_exp1.py \
  --backbones word_svc textcnn bilstm codebert \
  --operator-set official_wafamole \
  --output experiments/smoke_exp1_official.json \
  $COMMON

echo "=== Smoke: Exp1 advsqli ==="
PYTHONUNBUFFERED=1 python -u experiments/run_exp1.py \
  --backbones word_svc textcnn bilstm codebert \
  --operator-set advsqli \
  --output experiments/smoke_exp1_advsqli.json \
  $COMMON

echo "=== Smoke: Exp2 official_wafamole ==="
PYTHONUNBUFFERED=1 python -u experiments/run_exp2.py \
  --backbones textcnn bilstm codebert \
  --methods clean_ce pair_ce pair_canonical \
  --require-pairs \
  --train-operator-set official_wafamole \
  --attack-operator-set official_wafamole \
  --codebert-lr 1e-4 \
  --output experiments/smoke_exp2_official.json \
  $COMMON

echo "=== Smoke: Exp2 advsqli ==="
PYTHONUNBUFFERED=1 python -u experiments/run_exp2.py \
  --backbones textcnn bilstm codebert \
  --methods clean_ce pair_ce pair_canonical \
  --require-pairs \
  --train-operator-set official_wafamole \
  --attack-operator-set advsqli \
  --codebert-lr 1e-4 \
  --output experiments/smoke_exp2_advsqli.json \
  $COMMON

echo "=== Smoke: Ablation ==="
PYTHONUNBUFFERED=1 python -u experiments/sweep_ablation.py \
  --backbone bilstm \
  --seeds 11 \
  --attack-per-class 5 \
  --search-steps 3 \
  --candidates-per-state 8 \
  --beam-size 2 \
  --attack-search-group-size 32 \
  --allow-download --device cuda \
  --output experiments/smoke_ablation.json

echo "=== All smoke tests passed ==="
