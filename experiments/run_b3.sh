#!/usr/bin/env bash
# B3 ablation: pair_proj_ce on TextCNN + BiLSTM × {WAF-A-MoLE, AdvSQLi} × 5 seeds.
# Purpose: quantify "projection-layer contribution" vs "alignment-loss contribution"
# in pair_canonical, by running an ablation method that keeps the projection head
# but disables the alignment loss (attack-per-class=300, matching Exp2 main table).
#
# Usage (from repo root):
#   bash experiments/run_b3.sh 2>&1 | tee b3.log
# or detached under tmux:
#   tmux new -s b3 -d "bash experiments/run_b3.sh 2>&1 | tee b3.log"

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

COMMON=(
  --seeds 11 22 33 44 55
  --require-pairs
  --train-operator-set wafamole
  --attack-per-class 300
  --search-steps 20
  --candidates-per-state 48
  --beam-size 5
  --attack-search-group-size 128
  --epochs 8
  --batch-size 128
  --max-tokens 256
  --device cuda
  --resume
)

run_one () {
  local backbone="$1"
  local attack_set="$2"
  local out="experiments/results_b3_${backbone}_${attack_set}.json"
  echo "=== [$(date +'%F %T')] backbone=${backbone} attack=${attack_set} ==="
  PYTHONUNBUFFERED=1 python -u experiments/run_exp2.py \
    --backbones "${backbone}" \
    --methods pair_proj_ce \
    --attack-operator-set "${attack_set}" \
    "${COMMON[@]}" \
    --output "${out}"
}

run_one textcnn wafamole
run_one textcnn advsqli
run_one bilstm  wafamole
run_one bilstm  advsqli

echo "=== [$(date +'%F %T')] B3 done. ==="
ls -lah experiments/results_b3_*.json
