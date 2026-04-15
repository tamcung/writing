#!/usr/bin/env python3
"""Merge partial-seed result JSONs into a single combined file.

Usage:
    # Merge ablation results (rows-only, no summary recompute needed)
    python experiments/merge_results.py \
        --type ablation \
        --inputs experiments/results_ablation.json \
                 experiments/results_ablation_s4455.json \
        --output experiments/results_ablation.json

    python experiments/merge_results.py \
        --type ablation \
        --inputs experiments/results_ablation_textcnn.json \
                 experiments/results_ablation_textcnn_s4455.json \
        --output experiments/results_ablation_textcnn.json

    # Exp1 and Exp2 use --resume so they update in-place; no merge needed.
    # This script is only needed for ablation files that lack resume support.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_ablation(inputs: list[Path], output: Path) -> None:
    all_rows: list[dict] = []
    base_config: dict = {}
    total_elapsed: float = 0.0

    seen: set[tuple] = set()
    for path in inputs:
        d = json.loads(path.read_text(encoding="utf-8"))
        if not base_config:
            base_config = dict(d.get("config", {}))
        total_elapsed += float(d.get("elapsed_seconds", 0.0))
        for row in d.get("rows", []):
            key = (row["seed"], row["backbone"], row["consistency_weight"])
            if key not in seen:
                seen.add(key)
                all_rows.append(row)

    # Update seeds list in config
    all_seeds = sorted({r["seed"] for r in all_rows})
    base_config["seeds"] = all_seeds

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"config": base_config, "elapsed_seconds": round(total_elapsed, 1), "rows": all_rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Merged {len(all_rows)} rows (seeds {all_seeds}) → {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["ablation"], required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.type == "ablation":
        merge_ablation(args.inputs, args.output)


if __name__ == "__main__":
    main()
