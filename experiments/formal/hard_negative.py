#!/usr/bin/env python3
"""Real benign hard-negative mining for formal SQLi experiments."""

from __future__ import annotations

import random
import re
from collections.abc import Iterable


KEYWORD_WEIGHTS = {
    "select": 3.0,
    "union": 3.0,
    "from": 2.0,
    "where": 2.0,
    "and": 1.0,
    "or": 1.0,
    "order": 1.0,
    "by": 0.5,
    "like": 1.0,
    "sleep": 4.0,
    "benchmark": 4.0,
    "concat": 2.0,
    "char": 2.0,
    "null": 1.0,
}

SYMBOL_WEIGHTS = {
    "=": 0.8,
    "(": 0.4,
    ")": 0.4,
    ",": 0.4,
    "'": 0.5,
    '"': 0.5,
    "--": 1.0,
    "#": 1.0,
    "/*": 1.0,
}


def benign_hard_negative_score(text: str) -> float:
    lowered = text.lower()
    score = 0.0
    for keyword, weight in KEYWORD_WEIGHTS.items():
        if re.search(r"\b" + re.escape(keyword) + r"\b", lowered):
            score += weight
    for symbol, weight in SYMBOL_WEIGHTS.items():
        score += min(lowered.count(symbol), 3) * weight
    return score


def _repeat_to_count(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    if not rows or count <= 0:
        return []
    out: list[dict] = []
    shuffled = rows[:]
    while len(out) < count:
        rng.shuffle(shuffled)
        out.extend(shuffled[: count - len(out)])
    return out[:count]


def augment_with_real_hard_negatives(
    rows: list[dict],
    extra_benign: int,
    seed: int,
    min_score: float = 1.0,
    balance_sqli: bool = True,
) -> tuple[list[dict], dict]:
    """Oversample existing benign hard negatives without changing labels."""

    if extra_benign <= 0:
        return rows, {
            "enabled": False,
            "extra_benign": 0,
            "extra_sqli": 0,
            "eligible_benign": 0,
        }

    benign_rows = [row for row in rows if int(row["label"]) == 0]
    sqli_rows = [row for row in rows if int(row["label"]) == 1]
    scored = [
        (benign_hard_negative_score(str(row["text"])), idx, row)
        for idx, row in enumerate(benign_rows)
    ]
    eligible = [(score, idx, row) for score, idx, row in scored if score >= min_score]
    eligible.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    rng = random.Random(seed)

    hard_pool = [row for score, _, row in eligible]
    selected_hard = _repeat_to_count(hard_pool, extra_benign, rng)
    hard_augmented = [
        {
            **row,
            "hard_negative_oversampled": True,
            "hard_negative_score": benign_hard_negative_score(str(row["text"])),
        }
        for row in selected_hard
    ]

    sqli_augmented: list[dict] = []
    if balance_sqli:
        selected_sqli = _repeat_to_count(sqli_rows, len(hard_augmented), rng)
        sqli_augmented = [{**row, "sqli_balance_oversampled": True} for row in selected_sqli]

    augmented = rows[:] + hard_augmented + sqli_augmented
    rng.shuffle(augmented)
    top_examples = [
        {"score": score, "text": str(row["text"])[:240]}
        for score, _, row in eligible[:10]
    ]
    return augmented, {
        "enabled": True,
        "extra_benign": len(hard_augmented),
        "extra_sqli": len(sqli_augmented),
        "eligible_benign": len(eligible),
        "min_score": min_score,
        "balance_sqli": balance_sqli,
        "top_examples": top_examples,
    }


def summarize_scores(rows: Iterable[dict]) -> dict:
    scores_by_label = {0: [], 1: []}
    for row in rows:
        label = int(row["label"])
        if label in scores_by_label:
            scores_by_label[label].append(benign_hard_negative_score(str(row["text"])))
    summary = {}
    for label, scores in scores_by_label.items():
        if not scores:
            summary[str(label)] = {"count": 0}
            continue
        summary[str(label)] = {
            "count": len(scores),
            "mean": sum(scores) / len(scores),
            "max": max(scores),
            "ge_1": sum(score >= 1 for score in scores) / len(scores),
            "ge_3": sum(score >= 3 for score in scores) / len(scores),
            "ge_5": sum(score >= 5 for score in scores) / len(scores),
        }
    return summary
