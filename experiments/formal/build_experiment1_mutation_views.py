#!/usr/bin/env python3
"""Build experiment-1 mutation test views from fixed clean test splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from experiments.formal.raw_processing import write_json  # noqa: E402
from experiments.formal.semantic_mutation import (  # noqa: E402
    NUMERIC_REPR,
    PRIMARY_FAMILIES,
    STRING_CONSTRUCTION,
    SURFACE_OBFUSCATION,
    mutate_with_forced_surface_mixed,
    mutate_with_family,
)


FAMILY_ORDER = [SURFACE_OBFUSCATION, NUMERIC_REPR, STRING_CONSTRUCTION]


def load_split_rows(splits_dir: Path, seed: int, split_name: str) -> list[dict]:
    path = splits_dir / f"seed_{seed}" / f"{split_name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def attempt_family_mutation(row: dict, family: str, seed: int, max_attempts: int = 12, rounds: int = 1) -> dict | None:
    for attempt in range(max_attempts):
        rec = mutate_with_family(row["text"], family=family, seed=seed + attempt, rounds=rounds)
        if rec.mutated_text != row["text"]:
            out = dict(row)
            out["text"] = rec.mutated_text
            out["mutation_family"] = family
            out["mutation_rounds"] = rec.rounds
            out["source_text"] = row["text"]
            out["mutation_seed"] = seed + attempt
            out["mutated"] = True
            return out
    return None


def pick_benign_subset(benign_rows: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = benign_rows[:]
    rng.shuffle(pool)
    chosen = [dict(row, mutated=False, mutation_family=None, mutation_rounds=0, source_text=row["text"]) for row in pool[:count]]
    return chosen


def build_family_view(clean_test_rows: list[dict], family: str, seed: int, rounds: int = 1) -> tuple[list[dict], list[dict], dict]:
    benign_rows = [row for row in clean_test_rows if row["label"] == 0]
    sqli_rows = [row for row in clean_test_rows if row["label"] == 1]

    clean_sqli_subset: list[dict] = []
    mutated_sqli_subset: list[dict] = []
    failures = 0
    for i, row in enumerate(sqli_rows):
        mutated = attempt_family_mutation(
            row,
            family=family,
            seed=seed * 100000 + i * 97 + len(family),
            rounds=rounds,
        )
        if mutated is None:
            failures += 1
            continue
        clean_sqli_subset.append(
            dict(row, mutated=False, mutation_family=None, mutation_rounds=0, source_text=row["text"])
        )
        mutated_sqli_subset.append(mutated)

    n = len(mutated_sqli_subset)
    benign_subset = pick_benign_subset(benign_rows, n, seed=seed * 17 + len(family))
    clean_view = benign_subset + clean_sqli_subset
    mutated_view = benign_subset + mutated_sqli_subset
    random.Random(seed + 7).shuffle(clean_view)
    random.Random(seed + 13).shuffle(mutated_view)

    stats = {
        "family": family,
        "seed": seed,
        "rounds": rounds,
        "usable_sqli": n,
        "failed_sqli": failures,
        "benign_sampled": len(benign_subset),
        "clean_total": len(clean_view),
        "mutated_total": len(mutated_view),
    }
    return clean_view, mutated_view, stats


def build_mixed_view(clean_test_rows: list[dict], seed: int) -> tuple[list[dict], list[dict], dict]:
    benign_rows = [row for row in clean_test_rows if row["label"] == 0]
    sqli_rows = [row for row in clean_test_rows if row["label"] == 1]

    clean_sqli_subset: list[dict] = []
    mutated_sqli_subset: list[dict] = []
    chosen_counts = {family: 0 for family in PRIMARY_FAMILIES}
    failures = 0

    for i, row in enumerate(sqli_rows):
        rng = random.Random(seed * 100003 + i)
        family_order = FAMILY_ORDER[:]
        rng.shuffle(family_order)
        mutated = None
        chosen_family = None
        for family in family_order:
            candidate = attempt_family_mutation(row, family=family, seed=seed * 100000 + i * 97 + len(family))
            if candidate is not None:
                mutated = candidate
                chosen_family = family
                break
        if mutated is None:
            failures += 1
            continue
        chosen_counts[chosen_family] += 1
        clean_sqli_subset.append(
            dict(row, mutated=False, mutation_family=None, mutation_rounds=0, source_text=row["text"])
        )
        mutated_sqli_subset.append(mutated)

    n = len(mutated_sqli_subset)
    benign_subset = pick_benign_subset(benign_rows, n, seed=seed * 19 + 3)
    clean_view = benign_subset + clean_sqli_subset
    mutated_view = benign_subset + mutated_sqli_subset
    random.Random(seed + 23).shuffle(clean_view)
    random.Random(seed + 29).shuffle(mutated_view)

    stats = {
        "family": "mixed_primary",
        "seed": seed,
        "rounds": 1,
        "usable_sqli": n,
        "failed_sqli": failures,
        "benign_sampled": len(benign_subset),
        "clean_total": len(clean_view),
        "mutated_total": len(mutated_view),
        "chosen_family_counts": chosen_counts,
    }
    return clean_view, mutated_view, stats


def build_mixed_hard_view(clean_test_rows: list[dict], seed: int, rounds: int = 3) -> tuple[list[dict], list[dict], dict]:
    benign_rows = [row for row in clean_test_rows if row["label"] == 0]
    sqli_rows = [row for row in clean_test_rows if row["label"] == 1]

    clean_sqli_subset: list[dict] = []
    mutated_sqli_subset: list[dict] = []
    failures = 0

    for i, row in enumerate(sqli_rows):
        base_seed = seed * 200000 + i * 131 + rounds
        mutated = None
        for attempt in range(12):
            rec = mutate_with_forced_surface_mixed(row["text"], seed=base_seed + attempt, rounds=rounds)
            if rec.mutated_text != row["text"]:
                mutated = dict(row)
                mutated["text"] = rec.mutated_text
                mutated["mutation_family"] = rec.family
                mutated["mutation_rounds"] = rec.rounds
                mutated["source_text"] = row["text"]
                mutated["mutation_seed"] = base_seed + attempt
                mutated["mutated"] = True
                break
        if mutated is None:
            failures += 1
            continue
        clean_sqli_subset.append(
            dict(row, mutated=False, mutation_family=None, mutation_rounds=0, source_text=row["text"])
        )
        mutated_sqli_subset.append(mutated)

    n = len(mutated_sqli_subset)
    benign_subset = pick_benign_subset(benign_rows, n, seed=seed * 23 + rounds)
    clean_view = benign_subset + clean_sqli_subset
    mutated_view = benign_subset + mutated_sqli_subset
    random.Random(seed + 41).shuffle(clean_view)
    random.Random(seed + 43).shuffle(mutated_view)

    stats = {
        "family": "mixed_primary_hard_forced_surface",
        "seed": seed,
        "rounds": rounds,
        "protocol": "value-level rewrites first; surface obfuscation forced as final pass",
        "usable_sqli": n,
        "failed_sqli": failures,
        "benign_sampled": len(benign_subset),
        "clean_total": len(clean_view),
        "mutated_total": len(mutated_view),
    }
    return clean_view, mutated_view, stats


def build_audit_preview(rows: list[dict], limit: int = 12) -> list[dict]:
    return rows[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/derived/formal_v3/experiment1/splits")
    parser.add_argument("--output-dir", default="data/derived/formal_v3/experiment1/views")
    parser.add_argument("--audit-dir", default="data/derived/formal_v3/experiment1/audit")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    audit_dir = Path(args.audit_dir)

    global_manifest = {
        "experiment": "experiment1_problem_existence",
        "splits_dir": str(splits_dir),
        "views_dir": str(output_dir),
        "audit_dir": str(audit_dir),
        "families": FAMILY_ORDER,
        "seeds": args.seeds,
        "seeds_manifest": {},
    }

    for seed in args.seeds:
        clean_test_rows = load_split_rows(splits_dir, seed, "clean_test")
        seed_dir = output_dir / f"seed_{seed}"
        seed_audit_dir = audit_dir / f"seed_{seed}"

        clean_full = [dict(row, mutated=False, mutation_family=None, mutation_rounds=0, source_text=row["text"]) for row in clean_test_rows]
        write_json(seed_dir / "clean_test_full.json", clean_full)

        seed_manifest = {
            "seed": seed,
            "views": {
                "clean_test_full": str((seed_dir / "clean_test_full.json").resolve()),
            },
            "stats": {
                "clean_test_full": {
                    "total": len(clean_full),
                    "benign": sum(1 for row in clean_full if row["label"] == 0),
                    "sqli": sum(1 for row in clean_full if row["label"] == 1),
                }
            },
        }

        clean_mixed, mutated_mixed, mixed_stats = build_mixed_view(clean_test_rows, seed)
        write_json(seed_dir / "clean_test_mixed_matched.json", clean_mixed)
        write_json(seed_dir / "mutated_test_mixed.json", mutated_mixed)
        write_json(seed_audit_dir / "mixed_preview.json", build_audit_preview(mutated_mixed))
        seed_manifest["views"]["clean_test_mixed_matched"] = str((seed_dir / "clean_test_mixed_matched.json").resolve())
        seed_manifest["views"]["mutated_test_mixed"] = str((seed_dir / "mutated_test_mixed.json").resolve())
        seed_manifest["stats"]["mixed"] = mixed_stats

        clean_mixed_hard, mutated_mixed_hard, mixed_hard_stats = build_mixed_hard_view(clean_test_rows, seed, rounds=3)
        write_json(seed_dir / "clean_test_mixed_hard_matched.json", clean_mixed_hard)
        write_json(seed_dir / "mutated_test_mixed_hard.json", mutated_mixed_hard)
        write_json(seed_audit_dir / "mixed_hard_preview.json", build_audit_preview(mutated_mixed_hard))
        seed_manifest["views"]["clean_test_mixed_hard_matched"] = str((seed_dir / "clean_test_mixed_hard_matched.json").resolve())
        seed_manifest["views"]["mutated_test_mixed_hard"] = str((seed_dir / "mutated_test_mixed_hard.json").resolve())
        seed_manifest["stats"]["mixed_hard"] = mixed_hard_stats

        for family in FAMILY_ORDER:
            clean_view, mutated_view, family_stats = build_family_view(clean_test_rows, family, seed)
            clean_name = f"clean_test_{family}_matched.json"
            mutated_name = f"mutated_test_{family}.json"
            write_json(seed_dir / clean_name, clean_view)
            write_json(seed_dir / mutated_name, mutated_view)
            write_json(seed_audit_dir / f"{family}_preview.json", build_audit_preview(mutated_view))
            seed_manifest["views"][clean_name[:-5]] = str((seed_dir / clean_name).resolve())
            seed_manifest["views"][mutated_name[:-5]] = str((seed_dir / mutated_name).resolve())
            seed_manifest["stats"][family] = family_stats

        write_json(seed_dir / "manifest.json", seed_manifest)
        global_manifest["seeds_manifest"][str(seed)] = str((seed_dir / "manifest.json").resolve())
        print(f"seed={seed} mixed={seed_manifest['stats']['mixed']}")

    write_json(output_dir / "manifest.json", global_manifest)
    print(f"Wrote mutation views to {output_dir}")


if __name__ == "__main__":
    main()
