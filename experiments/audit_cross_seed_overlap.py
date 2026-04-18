#!/usr/bin/env python3
"""Audit cross-seed overlap across train/valid/clean_test splits.

Quantifies the extent to which the independent per-seed reshuffles of the same
master pool (70,544 samples) leave train_i and test_j (i != j) disjoint.

Prints three matrices:
  - |train_i ∩ train_j|
  - |train_i ∩ clean_test_j|  <-- the structural contamination of interest
  - |clean_test_i ∩ clean_test_j|

Each row is grouped by label (benign / sqli) to reveal asymmetry from the
smaller SQLi pool being sampled more heavily per seed.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
SEEDS = [11, 22, 33, 44, 55]


def load_indices(seed: int, split: str) -> dict[int, set[tuple[str, int]]]:
    """Return {label -> set of (source_dataset, source_index)} for a split."""
    path = SPLITS_DIR / f"seed_{seed}" / f"{split}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, set[tuple[str, int]]] = {0: set(), 1: set()}
    for row in rows:
        key = (row["source_dataset"], int(row["source_index"]))
        out[int(row["label"])].add(key)
    return out


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def main() -> None:
    splits = {}  # seed -> split -> label -> set
    for seed in SEEDS:
        splits[seed] = {}
        for split in ("train", "clean_test"):
            splits[seed][split] = load_indices(seed, split)

    print(f"Seeds audited: {SEEDS}")
    print(f"Per-seed per-class counts (train / clean_test):")
    for seed in SEEDS:
        b_tr = len(splits[seed]["train"][0])
        s_tr = len(splits[seed]["train"][1])
        b_te = len(splits[seed]["clean_test"][0])
        s_te = len(splits[seed]["clean_test"][1])
        print(f"  seed={seed}: benign={b_tr}/{b_te}  sqli={s_tr}/{s_te}")
    print()

    for label, name in [(0, "benign"), (1, "sqli")]:
        print(f"=== label={name} ===")
        # train_i ∩ train_j
        print("train_i ∩ train_j (count | Jaccard):")
        for i in SEEDS:
            row = []
            for j in SEEDS:
                a = splits[i]["train"][label]
                b = splits[j]["train"][label]
                if i == j:
                    row.append("     —    ")
                else:
                    inter = len(a & b)
                    jac = jaccard(a, b)
                    row.append(f"{inter:>5} | {jac:.3f}")
            print(f"  seed {i}: " + "   ".join(row))
        # train_i ∩ clean_test_j
        print("train_i ∩ clean_test_j (count | ratio over clean_test_j):")
        for i in SEEDS:
            row = []
            for j in SEEDS:
                a = splits[i]["train"][label]
                b = splits[j]["clean_test"][label]
                if i == j:
                    row.append("     —    ")
                else:
                    inter = len(a & b)
                    ratio = inter / max(1, len(b))
                    row.append(f"{inter:>5} | {ratio:.3f}")
            print(f"  seed {i}: " + "   ".join(row))
        # clean_test_i ∩ clean_test_j
        print("clean_test_i ∩ clean_test_j (count | Jaccard):")
        for i in SEEDS:
            row = []
            for j in SEEDS:
                a = splits[i]["clean_test"][label]
                b = splits[j]["clean_test"][label]
                if i == j:
                    row.append("     —    ")
                else:
                    inter = len(a & b)
                    jac = jaccard(a, b)
                    row.append(f"{inter:>5} | {jac:.3f}")
            print(f"  seed {i}: " + "   ".join(row))
        print()

    print("Summary")
    print("-------")
    for label, name in [(0, "benign"), (1, "sqli")]:
        tr_cross_te_ratios = []
        tr_cross_tr_jacs = []
        te_cross_te_jacs = []
        for i in SEEDS:
            for j in SEEDS:
                if i == j:
                    continue
                a_tr = splits[i]["train"][label]
                b_tr = splits[j]["train"][label]
                b_te = splits[j]["clean_test"][label]
                a_te = splits[i]["clean_test"][label]
                tr_cross_te_ratios.append(len(a_tr & b_te) / max(1, len(b_te)))
                tr_cross_tr_jacs.append(jaccard(a_tr, b_tr))
                te_cross_te_jacs.append(jaccard(a_te, b_te))

        def avg(xs):
            return sum(xs) / max(1, len(xs))

        print(f"  {name}:")
        print(f"    mean |train_i ∩ train_j| Jaccard   = {avg(tr_cross_tr_jacs):.3f}")
        print(f"    mean |train_i ∩ test_j| / |test_j| = {avg(tr_cross_te_ratios):.3f}"
              f"  (contamination fraction of test_j from train_i)")
        print(f"    mean |test_i ∩ test_j| Jaccard     = {avg(te_cross_te_jacs):.3f}")


if __name__ == "__main__":
    main()
