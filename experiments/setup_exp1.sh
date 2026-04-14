#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f experiment1_inputs.tar.gz ]]; then
  echo "Missing experiment1_inputs.tar.gz in $ROOT_DIR" >&2
  exit 1
fi

tar -xzf experiment1_inputs.tar.gz

if [[ ! -f external/WAF-A-MoLE/wafamole/payloadfuzzer/sqlfuzzer.py ]]; then
  mkdir -p external
  git clone --depth 1 https://github.com/AvalZ/WAF-A-MoLE.git external/WAF-A-MoLE
fi

echo "Experiment 1 inputs are ready."
