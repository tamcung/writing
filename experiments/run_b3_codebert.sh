#!/usr/bin/env bash
# B3 ablation (CodeBERT extension): pair_proj_ce on CodeBERT × {WAF-A-MoLE, AdvSQLi} × 5 seeds.
# Safe to run in parallel with experiments/run_b3.sh on the same GPU — CodeBERT uses
# ~5-6GB VRAM and shallow backbones <1GB, so a 12GB+ card fits both.
#
# Usage (from repo root):
#   tmux new -s b3cb -d "bash experiments/run_b3_codebert.sh 2>&1 | tee b3cb.log"

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

COMMON=(
  --backbones codebert
  --methods pair_proj_ce
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
  --codebert-lr 1e-4
  --device cuda
  --resume
)

run_one () {
  local attack_set="$1"
  local out="experiments/results_b3_codebert_${attack_set}.json"
  echo "=== [$(date +'%F %T')] codebert × ${attack_set} ==="
  PYTHONUNBUFFERED=1 python -u experiments/run_exp2.py \
    "${COMMON[@]}" \
    --attack-operator-set "${attack_set}" \
    --output "${out}"
}

run_one wafamole
run_one advsqli

echo "=== [$(date +'%F %T')] B3 codebert done. ==="
ls -lah experiments/results_b3_codebert_*.json
