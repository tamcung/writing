#!/usr/bin/env python3
"""Build and load formal paired training data for experiment 2."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.formal.semantic_mutation import benign_nuisance_transform, benign_nuisance_transform_values
from experiments.formal.targeted_sql_mutation import get_operator_set, random_operator_chain


def build_pair_rows(
    train_rows: list[dict],
    seed: int,
    operator_set: str,
    sqli_pairs_per_sample: int,
    benign_pairs_per_sample: int,
    mutation_rounds: int,
    mutation_retries: int,
    max_chars: int,
) -> tuple[list[dict], dict]:
    operators = get_operator_set(operator_set)
    pair_rows: list[dict] = []

    for i, row in enumerate(train_rows):
        text = str(row["text"])
        label = int(row["label"])
        repeats = sqli_pairs_per_sample if label == 1 else benign_pairs_per_sample
        for p in range(max(1, repeats)):
            pair_seed = seed * 1_000_003 + i * 10_007 + p * 1_009
            if label == 1:
                result = random_operator_chain(
                    source_text=text,
                    seed=pair_seed,
                    operators=operators,
                    rounds=mutation_rounds,
                    retries=mutation_retries,
                    max_chars=max_chars,
                    ensure_changed=True,
                )
                pair_rows.append(
                    {
                        "pair_id": f"seed{seed}_row{i}_pair{p}",
                        "source_index": i,
                        "label": label,
                        "x_canon": text,
                        "x_raw_mut": result.mutated_text,
                        "changed": result.changed,
                        "pair_kind": "sqli_official_wafamole",
                        "operator_set": operator_set,
                        "mutation_chain": list(result.chain),
                        "source": row.get("source"),
                        "origin": row.get("origin"),
                    }
                )
            else:
                value_parts = row.get("value_parts")
                if isinstance(value_parts, list) and value_parts:
                    mutated_values = benign_nuisance_transform_values(
                        [str(value) for value in value_parts],
                        pair_seed,
                    )
                    mutated = " ".join(mutated_values)
                else:
                    mutated = benign_nuisance_transform(text, pair_seed)
                pair_rows.append(
                    {
                        "pair_id": f"seed{seed}_row{i}_pair{p}",
                        "source_index": i,
                        "label": label,
                        "x_canon": text,
                        "x_raw_mut": mutated,
                        "changed": mutated != text,
                        "pair_kind": "benign_nuisance",
                        "operator_set": None,
                        "mutation_chain": [],
                        "source": row.get("source"),
                        "origin": row.get("origin"),
                    }
                )

    return pair_rows, summarize_pair_rows(
        pair_rows,
        operator_set=operator_set,
        mutation_rounds=mutation_rounds,
        mutation_retries=mutation_retries,
        sqli_pairs_per_sample=sqli_pairs_per_sample,
        benign_pairs_per_sample=benign_pairs_per_sample,
        max_chars=max_chars,
    )


def summarize_pair_rows(
    pair_rows: list[dict],
    operator_set: str | None = None,
    mutation_rounds: int | None = None,
    mutation_retries: int | None = None,
    sqli_pairs_per_sample: int | None = None,
    benign_pairs_per_sample: int | None = None,
    max_chars: int | None = None,
) -> dict:
    totals = Counter(int(row["label"]) for row in pair_rows)
    changed = Counter(int(row["label"]) for row in pair_rows if row.get("changed"))
    sqli_chain_lengths = [
        len(row.get("mutation_chain", []))
        for row in pair_rows
        if int(row["label"]) == 1
    ]
    operator_counts: Counter[str] = Counter()
    for row in pair_rows:
        operator_counts.update(str(op) for op in row.get("mutation_chain", []))

    return {
        "operator_set": operator_set,
        "total_pairs": len(pair_rows),
        "benign_pairs": totals[0],
        "sqli_pairs": totals[1],
        "benign_changed_rate": changed[0] / totals[0] if totals[0] else 0.0,
        "sqli_changed_rate": changed[1] / totals[1] if totals[1] else 0.0,
        "mean_sqli_chain_len": float(np.mean(sqli_chain_lengths)) if sqli_chain_lengths else 0.0,
        "min_sqli_chain_len": int(np.min(sqli_chain_lengths)) if sqli_chain_lengths else 0,
        "max_sqli_chain_len": int(np.max(sqli_chain_lengths)) if sqli_chain_lengths else 0,
        "operator_counts": dict(sorted(operator_counts.items())),
        "mutation_rounds": mutation_rounds,
        "mutation_retries": mutation_retries,
        "sqli_pairs_per_sample": sqli_pairs_per_sample,
        "benign_pairs_per_sample": benign_pairs_per_sample,
        "max_chars": max_chars,
    }


def pair_rows_to_training_arrays(pair_rows: list[dict]) -> tuple[list[str], list[str], list[int]]:
    return (
        [str(row["x_canon"]) for row in pair_rows],
        [str(row["x_raw_mut"]) for row in pair_rows],
        [int(row["label"]) for row in pair_rows],
    )


def load_pair_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))
