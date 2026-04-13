#!/usr/bin/env python3
"""Audit SQLiV5 usage under the thesis protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.consistency_sqli_experiment import load_payload_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/raw/SQLiV3_clean.json")
    parser.add_argument("--sqliv5-data", default="data/raw/SQLiV5.json")
    parser.add_argument("--output", default="data/raw/SQLiV5_audit.json")
    parser.add_argument("--max-len", type=int, default=260)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_texts, train_labels = load_payload_data(Path(args.train_data), max_len=args.max_len)
    v5_texts, v5_labels = load_payload_data(Path(args.sqliv5_data), max_len=args.max_len)

    v3_all = set(train_texts)
    v5_all = set(v5_texts)

    v5_benign = {text for text, y in zip(v5_texts, v5_labels) if y == 0}
    v5_sqli = {text for text, y in zip(v5_texts, v5_labels) if y == 1}

    overlap = v3_all & v5_all
    new_sqli = v5_sqli - v3_all
    new_benign = v5_benign - v3_all

    audit = {
        "train_data": args.train_data,
        "sqliv5_data": args.sqliv5_data,
        "max_len": args.max_len,
        "sqliv3_total": len(train_texts),
        "sqliv3_benign": train_labels.count(0),
        "sqliv3_sqli": train_labels.count(1),
        "sqliv5_total": len(v5_texts),
        "sqliv5_benign": v5_labels.count(0),
        "sqliv5_sqli": v5_labels.count(1),
        "sqliv5_exact_overlap_with_sqliv3": len(overlap),
        "sqliv5_overlap_ratio_unique": len(overlap) / len(v5_all) if v5_all else 0.0,
        "sqliv5_new_sqli": len(new_sqli),
        "sqliv5_new_benign": len(new_benign),
        "recommended_eval_view": "use SQLiV5 as a supplementary real-mutation benchmark; prioritize unseen-SQLi subset not present in SQLiV3",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
