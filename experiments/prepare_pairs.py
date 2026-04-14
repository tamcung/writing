#!/usr/bin/env python3
"""Materialize paired training data for experiment 2."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pair_data import build_pair_rows  # noqa: E402
from experiments.model_utils import load_seed_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--output-dir", default="data/pairs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--operator-set", choices=["official_wafamole"], default="official_wafamole")
    parser.add_argument("--sqli-pairs-per-sample", type=int, default=1)
    parser.add_argument("--benign-pairs-per-sample", type=int, default=1)
    parser.add_argument("--mutation-rounds", type=int, default=7)
    parser.add_argument("--mutation-retries", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=896)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seed_manifests: dict[str, dict] = {}

    for seed in args.seeds:
        train_rows = load_seed_split(splits_dir, seed, "train")
        pair_rows, stats = build_pair_rows(
            train_rows=train_rows,
            seed=seed,
            operator_set=args.operator_set,
            sqli_pairs_per_sample=args.sqli_pairs_per_sample,
            benign_pairs_per_sample=args.benign_pairs_per_sample,
            mutation_rounds=args.mutation_rounds,
            mutation_retries=args.mutation_retries,
            max_chars=args.max_chars,
        )
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        pairs_path = seed_dir / "train_pairs.json"
        manifest_path = seed_dir / "manifest.json"
        pairs_path.write_text(json.dumps(pair_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        seed_manifest = {
            "seed": seed,
            "train_split": str((splits_dir / f"seed_{seed}" / "train.json").resolve()),
            "pairs": str(pairs_path.resolve()),
            "stats": stats,
        }
        manifest_path.write_text(json.dumps(seed_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        seed_manifests[str(seed)] = seed_manifest
        print(
            f"seed={seed} pairs={stats['total_pairs']} "
            f"sqli_changed={stats['sqli_changed_rate']:.4f} "
            f"benign_changed={stats['benign_changed_rate']:.4f} "
            f"mean_chain={stats['mean_sqli_chain_len']:.3f}"
        )

    manifest = {
        "experiment": "experiment2_pair_training_targeted",
        "source_splits_dir": str(splits_dir.resolve()),
        "operator_set": args.operator_set,
        "seeds": args.seeds,
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "seeds_manifest": seed_manifests,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote pair manifest to {manifest_path}")


if __name__ == "__main__":
    main()
