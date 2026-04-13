#!/usr/bin/env python3
"""Audit mutation family coverage and sample previews for experiment 1."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from experiments.formal.raw_processing import write_json  # noqa: E402
from experiments.formal.semantic_mutation import ALL_FAMILIES, mutate_with_family  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_flags(text: str) -> list[str]:
    flags: list[str] = []
    if re.search(r"'[^']*'", text):
        flags.append("single_quote")
    if '"' in text:
        flags.append("double_quote")
    if "0x" in text.lower():
        flags.append("hex_literal")
    if re.search(r"\b\d+\b", text):
        flags.append("numeric_literal")
    if re.search(r"\b(or|and)\b", text, flags=re.IGNORECASE):
        flags.append("boolean_op")
    if re.search(r"\bunion\b", text, flags=re.IGNORECASE):
        flags.append("union")
    if re.search(r"\bselect\b", text, flags=re.IGNORECASE):
        flags.append("select")
    if re.search(r"\b(sleep|benchmark|pg_sleep|waitfor)\b", text, flags=re.IGNORECASE):
        flags.append("time_based")
    return flags


def try_mutate(text: str, family: str, base_seed: int, max_attempts: int) -> dict | None:
    for attempt in range(max_attempts):
        rec = mutate_with_family(text, family=family, seed=base_seed + attempt, rounds=1)
        if rec.mutated_text != text:
            return {
                "source_text": text,
                "mutated_text": rec.mutated_text,
                "family": family,
                "mutation_seed": base_seed + attempt,
                "rounds": rec.rounds,
            }
    return None


def audit_family(rows: list[dict], family: str, seed: int, max_attempts: int, preview_limit: int) -> dict:
    sqli_rows = [row for row in rows if int(row["label"]) == 1]
    benign_rows = [row for row in rows if int(row["label"]) == 0]

    successes: list[dict] = []
    failed_examples: list[dict] = []
    failure_flags = Counter()

    for i, row in enumerate(sqli_rows):
        result = try_mutate(row["text"], family, seed * 100000 + i * 97 + len(family), max_attempts)
        if result is None:
            flags = feature_flags(row["text"])
            failure_flags.update(flags)
            if len(failed_examples) < preview_limit:
                failed_examples.append(
                    {
                        "text": row["text"],
                        "flags": flags,
                    }
                )
            continue
        if len(successes) < preview_limit:
            result["flags"] = feature_flags(row["text"])
            successes.append(result)

    usable = len(sqli_rows) - len(failed_examples) - max(0, 0)
    success_count = len(sqli_rows) - (sum(1 for _ in failed_examples) if len(failed_examples) == len(sqli_rows) else 0)
    # recompute exactly without relying on previews
    exact_success = 0
    for i, row in enumerate(sqli_rows):
        result = try_mutate(row["text"], family, seed * 100000 + i * 97 + len(family), max_attempts)
        if result is not None:
            exact_success += 1

    return {
        "family": family,
        "seed": seed,
        "sqli_total": len(sqli_rows),
        "benign_total": len(benign_rows),
        "usable_sqli": exact_success,
        "failed_sqli": len(sqli_rows) - exact_success,
        "coverage": exact_success / len(sqli_rows) if sqli_rows else 0.0,
        "success_preview": successes,
        "failure_preview": failed_examples,
        "failure_feature_counts": dict(failure_flags),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-test", default="data/derived/formal_v3/experiment1/splits/seed_11/clean_test.json")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--families", nargs="+", default=ALL_FAMILIES)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--preview-limit", type=int, default=12)
    parser.add_argument("--output", default="data/derived/formal_v3/experiment1/audit/seed_11/family_audit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.clean_test))
    report = {
        "seed": args.seed,
        "clean_test_path": str(Path(args.clean_test).resolve()),
        "families": {},
    }
    for family in args.families:
        audit = audit_family(rows, family=family, seed=args.seed, max_attempts=args.max_attempts, preview_limit=args.preview_limit)
        report["families"][family] = audit
        print(
            f"{family}: usable={audit['usable_sqli']}/{audit['sqli_total']} "
            f"coverage={audit['coverage']:.3f}"
        )
    write_json(Path(args.output), report)
    print(f"Wrote family audit to {args.output}")


if __name__ == "__main__":
    main()
