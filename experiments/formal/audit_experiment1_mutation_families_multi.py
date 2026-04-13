#!/usr/bin/env python3
"""Aggregate experiment-1 mutation family audits across multiple seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from experiments.formal.audit_experiment1_mutation_families import audit_family, load_rows  # noqa: E402
from experiments.formal.raw_processing import write_json  # noqa: E402
from experiments.formal.semantic_mutation import ALL_FAMILIES  # noqa: E402


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/derived/formal_v3/experiment1/splits")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    parser.add_argument("--families", nargs="+", default=ALL_FAMILIES)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--preview-limit", type=int, default=8)
    parser.add_argument("--output", default="data/derived/formal_v3/experiment1/audit/family_audit_multi_seed.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits_dir = Path(args.splits_dir)

    report: dict[str, object] = {
        "splits_dir": str(splits_dir.resolve()),
        "seeds": args.seeds,
        "families": {},
    }

    for family in args.families:
        seed_rows = []
        coverages: list[float] = []
        usable_counts: list[int] = []
        failed_counts: list[int] = []
        for seed in args.seeds:
            clean_test_path = splits_dir / f"seed_{seed}" / "clean_test.json"
            rows = load_rows(clean_test_path)
            audit = audit_family(
                rows,
                family=family,
                seed=seed,
                max_attempts=args.max_attempts,
                preview_limit=args.preview_limit,
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "coverage": audit["coverage"],
                    "usable_sqli": audit["usable_sqli"],
                    "failed_sqli": audit["failed_sqli"],
                    "failure_feature_counts": audit["failure_feature_counts"],
                    "success_preview": audit["success_preview"],
                    "failure_preview": audit["failure_preview"],
                }
            )
            coverages.append(float(audit["coverage"]))
            usable_counts.append(int(audit["usable_sqli"]))
            failed_counts.append(int(audit["failed_sqli"]))

        report["families"][family] = {
            "coverage": summarize(coverages),
            "usable_sqli": summarize([float(x) for x in usable_counts]),
            "failed_sqli": summarize([float(x) for x in failed_counts]),
            "per_seed": seed_rows,
        }
        print(
            f"{family}: coverage_mean={report['families'][family]['coverage']['mean']:.3f} "
            f"min={report['families'][family]['coverage']['min']:.3f} "
            f"max={report['families'][family]['coverage']['max']:.3f}"
        )

    write_json(Path(args.output), report)
    print(f"Wrote multi-seed family audit to {args.output}")


if __name__ == "__main__":
    main()
