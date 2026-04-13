#!/usr/bin/env python3
"""Build fixed experiment-1 train/valid/test splits from the formal processed dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_rows(processed_dir: Path, dataset_name: str) -> list[dict]:
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    rel_path = manifest["datasets"][dataset_name]["path"]
    rows = json.loads((ROOT / rel_path).read_text(encoding="utf-8"))
    enriched = []
    for idx, row in enumerate(rows):
        enriched.append(
            {
                "source_dataset": dataset_name,
                "source_index": idx,
                "text": str(row["text"]),
                "label": int(row["label"]),
                "source": str(row.get("source", dataset_name)),
                "origin": str(row.get("origin", row.get("source", dataset_name))),
            }
        )
    return enriched


def build_split(
    rows: list[dict],
    seed: int,
    train_per_class: int,
    valid_per_class: int,
    test_per_class: int,
) -> dict[str, list[dict]]:
    by_label = {0: [], 1: []}
    for row in rows:
        by_label[row["label"]].append(row)

    rng = random.Random(seed)
    out: dict[str, list[dict]] = {"train": [], "valid": [], "clean_test": []}
    for label in [0, 1]:
        pool = by_label[label][:]
        rng.shuffle(pool)
        need = train_per_class + valid_per_class + test_per_class
        if len(pool) < need:
            raise ValueError(f"class={label} requires {need}, but only {len(pool)} available")
        out["train"].extend(pool[:train_per_class])
        out["valid"].extend(pool[train_per_class : train_per_class + valid_per_class])
        out["clean_test"].extend(pool[train_per_class + valid_per_class : need])

    for split_name in out:
        rng.shuffle(out[split_name])
    return out


def split_stats(rows: list[dict]) -> dict[str, int]:
    benign = sum(1 for row in rows if row["label"] == 0)
    sqli = sum(1 for row in rows if row["label"] == 1)
    return {"total": len(rows), "benign": benign, "sqli": sqli}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/formal_v3")
    parser.add_argument("--dataset", default="sqliv3_clean")
    parser.add_argument("--output-dir", default="data/derived/formal_v3/experiment1/splits")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--valid-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    rows = load_rows(processed_dir, args.dataset)

    manifest = {
        "experiment": "experiment1_problem_existence",
        "processed_dir": str(processed_dir),
        "dataset": args.dataset,
        "seeds": args.seeds,
        "train_per_class": args.train_per_class,
        "valid_per_class": args.valid_per_class,
        "test_per_class": args.test_per_class,
        "splits": {},
    }

    for seed in args.seeds:
        split_rows = build_split(
            rows,
            seed=seed,
            train_per_class=args.train_per_class,
            valid_per_class=args.valid_per_class,
            test_per_class=args.test_per_class,
        )
        seed_dir = output_dir / f"seed_{seed}"
        for split_name, split_data in split_rows.items():
            write_json(seed_dir / f"{split_name}.json", split_data)

        seed_manifest = {
            "seed": seed,
            "files": {
                split_name: str((seed_dir / f"{split_name}.json").resolve())
                for split_name in ["train", "valid", "clean_test"]
            },
            "stats": {split_name: split_stats(split_data) for split_name, split_data in split_rows.items()},
        }
        write_json(seed_dir / "manifest.json", seed_manifest)
        manifest["splits"][str(seed)] = seed_manifest
        print(
            f"seed={seed} train={seed_manifest['stats']['train']} "
            f"valid={seed_manifest['stats']['valid']} clean_test={seed_manifest['stats']['clean_test']}"
        )

    write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote split manifests to {output_dir}")


if __name__ == "__main__":
    main()
