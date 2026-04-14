#!/usr/bin/env python3
"""Sensitivity analysis for experiment-1 attack budget selection."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.formal.metrics import summarize
from experiments.formal.run_experiment1_clean_ce import (
    build_model,
    load_seed_split,
    resolve_device,
    set_seed,
)
from experiments.formal.run_experiment1_targeted_attack import (
    attack_sqli_rows,
    evaluate_view,
    pick_attack_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/derived/formal_modsec_decoded/experiment1/splits")
    parser.add_argument("--backbones", nargs="+", default=["word_svc", "textcnn", "bilstm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--attack-per-class", type=int, default=100)
    parser.add_argument("--operator-set", choices=["conservative", "wafamole_style", "official_wafamole"], default="official_wafamole")
    parser.add_argument("--max-chars", type=int, default=896)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")

    parser.add_argument(
        "--budget-specs",
        nargs="+",
        default=["light:10:24:3", "default:20:48:5", "heavy:30:64:8"],
        help="name:steps:candidates:beam",
    )

    parser.add_argument("--word-ngram-max", type=int, default=2)
    parser.add_argument("--word-min-df", type=int, default=1)
    parser.add_argument("--word-c", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)

    parser.add_argument("--output", default="experiments/formal/results_experiment1_attack_budget_analysis.json")
    return parser.parse_args()


def parse_budget_spec(spec: str) -> dict:
    name, steps, candidates, beam = spec.split(":")
    return {
        "name": name,
        "steps": int(steps),
        "candidates_per_state": int(candidates),
        "beam_size": int(beam),
    }


def wilson_half_width(n: int, p: float = 0.5, z: float = 1.96) -> float:
    if n <= 0:
        return float("nan")
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    del center
    return float(radius)


def main() -> None:
    args = parse_args()
    budgets = [parse_budget_spec(spec) for spec in args.budget_specs]
    device = resolve_device(args.device)
    splits_dir = Path(args.splits_dir)
    started = time.time()
    rows: list[dict] = []

    for seed in args.seeds:
        set_seed(seed)
        train_rows = load_seed_split(splits_dir, seed, "train")
        clean_test_rows = load_seed_split(splits_dir, seed, "clean_test")
        train_texts = [str(row["text"]) for row in train_rows]
        train_labels = [int(row["label"]) for row in train_rows]
        benign_rows, sqli_rows = pick_attack_rows(clean_test_rows, args.attack_per_class, seed)
        clean_attack_view = [
            {**row, "mutated": False, "source_text": row["text"], "mutation_family": None}
            for row in benign_rows + sqli_rows
        ]

        for backbone in args.backbones:
            print(f"seed={seed} training {backbone}")
            model = build_model(backbone, args, device)
            model.fit(train_texts, train_labels)
            clean_eval = evaluate_view(model, clean_attack_view, seed, backbone, "clean_attack_matched", "clean")
            for budget in budgets:
                print(
                    f"  budget={budget['name']} steps={budget['steps']} "
                    f"candidates={budget['candidates_per_state']} beam={budget['beam_size']}"
                )
                adv_sqli_rows = attack_sqli_rows(
                    model=model,
                    sqli_rows=sqli_rows,
                    seed=seed,
                    operator_set=args.operator_set,
                    steps=budget["steps"],
                    candidates_per_state=budget["candidates_per_state"],
                    beam_size=budget["beam_size"],
                    threshold=args.threshold,
                    max_chars=args.max_chars,
                    early_stop=True,
                )
                adv_view = [
                    {**row, "mutated": False, "source_text": row["text"], "mutation_family": None}
                    for row in benign_rows
                ] + adv_sqli_rows
                adv_eval = evaluate_view(
                    model,
                    adv_view,
                    seed,
                    backbone,
                    f"targeted_{budget['name']}",
                    "targeted_mutated",
                )
                success_rate = sum(1 for row in adv_sqli_rows if row["attack_success"]) / max(1, len(adv_sqli_rows))
                mean_drop = float(np.mean([row["prob_drop"] for row in adv_sqli_rows])) if adv_sqli_rows else 0.0
                mean_queries = float(np.mean([row["attack_queries"] for row in adv_sqli_rows])) if adv_sqli_rows else 0.0
                rows.append(
                    {
                        "seed": seed,
                        "backbone": backbone,
                        "budget": budget,
                        "clean_metrics": clean_eval["metrics"],
                        "targeted_metrics": adv_eval["metrics"],
                        "attack_success": success_rate,
                        "mean_prob_drop": mean_drop,
                        "mean_queries": mean_queries,
                    }
                )
                print(
                    f"    recall={adv_eval['metrics']['recall']:.4f} "
                    f"success={success_rate:.4f} "
                    f"drop={mean_drop:.4f} "
                    f"queries={mean_queries:.1f}"
                )

    summary: dict[str, dict] = {}
    for budget in budgets:
        budget_name = budget["name"]
        summary[budget_name] = {}
        for backbone in args.backbones:
            subset = [row for row in rows if row["budget"]["name"] == budget_name and row["backbone"] == backbone]
            summary[budget_name][backbone] = {
                "targeted_recall": summarize([row["targeted_metrics"]["recall"] for row in subset]),
                "targeted_f1": summarize([row["targeted_metrics"]["f1"] for row in subset]),
                "targeted_p10": summarize([row["targeted_metrics"]["p10_sqli_prob"] for row in subset]),
                "attack_success": summarize([row["attack_success"] for row in subset]),
                "mean_prob_drop": summarize([row["mean_prob_drop"] for row in subset]),
                "mean_queries": summarize([row["mean_queries"] for row in subset]),
            }

    sample_precision = {
        str(n): {
            "worst_case_half_width_95": wilson_half_width(n=n, p=0.5),
            "half_width_p10_95": wilson_half_width(n=n, p=0.1),
            "half_width_p20_95": wilson_half_width(n=n, p=0.2),
        }
        for n in [50, 100, 200, 300, 500]
    }

    payload = {
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "summary": summary,
        "sample_precision": sample_precision,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote analysis to {out_path}")


if __name__ == "__main__":
    main()
