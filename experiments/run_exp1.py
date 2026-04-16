#!/usr/bin/env python3
"""Experiment 1: targeted SQL evasion attack against clean-trained WAF models."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.metrics import metrics_from_probs, summarize  # noqa: E402
from experiments.model_utils import (  # noqa: E402
    build_model,
    load_seed_split,
    resolve_device,
    rows_to_xy,
    set_seed,
    summarize_rows,
)
from experiments.attack_operators import (  # noqa: E402
    get_operator_set,
    targeted_evasion_search_many,
)


def pick_attack_rows(clean_test_rows: list[dict], per_class: int | None, seed: int) -> tuple[list[dict], list[dict]]:
    benign = [row for row in clean_test_rows if int(row["label"]) == 0]
    sqli = [row for row in clean_test_rows if int(row["label"]) == 1]
    rng = random.Random(seed)
    rng.shuffle(benign)
    rng.shuffle(sqli)
    n = min(len(benign), len(sqli), per_class if per_class is not None else len(sqli))
    return benign[:n], sqli[:n]


def attack_sqli_rows(
    model,
    sqli_rows: list[dict],
    seed: int,
    operator_set: str,
    steps: int,
    candidates_per_state: int,
    beam_size: int,
    threshold: float,
    max_chars: int,
    early_stop: bool,
    group_size: int = 128,
) -> list[dict]:
    operators = get_operator_set(operator_set)
    grad_fn = getattr(model, "token_importances", None)
    adv_rows: list[dict] = []

    for start in range(0, len(sqli_rows), max(1, group_size)):
        batch_rows = sqli_rows[start : start + group_size]
        batch_texts = [str(row["text"]) for row in batch_rows]
        batch_seeds = [
            seed * 1_000_003 + (start + i) * 9_176 + len(t)
            for i, t in enumerate(batch_texts)
        ]
        results = targeted_evasion_search_many(
            source_texts=batch_texts,
            score_fn=lambda batch: model.predict_proba(batch),
            seeds=batch_seeds,
            operators=operators,
            steps=steps,
            candidates_per_state=candidates_per_state,
            beam_size=beam_size,
            success_threshold=threshold,
            max_chars=max_chars,
            early_stop=early_stop,
            grad_fn=grad_fn,
        )
        for row, result in zip(batch_rows, results):
            adv_rows.append({
                **row,
                "text": result.adversarial_text,
                "source_text": result.source_text,
                "mutated": result.changed,
                "attack_success": result.success,
                "source_sqli_prob": result.source_prob,
                "adversarial_sqli_prob": result.adversarial_prob,
                "prob_drop": result.source_prob - result.adversarial_prob,
                "attack_steps": result.steps,
                "attack_queries": result.queries,
                "mutation_chain": list(result.chain),
                "mutation_family": "targeted_" + operator_set,
            })
            i = len(adv_rows)
            if i % 50 == 0 or i == len(sqli_rows):
                recent = adv_rows[max(0, i - 50):]
                success = sum(1 for r in recent if r["attack_success"]) / len(recent)
                drop = sum(float(r["prob_drop"]) for r in recent) / len(recent)
                print(f"    attacked {i}/{len(sqli_rows)} recent_success={success:.3f} recent_drop={drop:.4f}")

    return adv_rows


def summarize_attack_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        "count": len(rows),
        "success_rate": sum(1 for r in rows if r["attack_success"]) / len(rows),
        "mean_prob_drop": float(np.mean([r["prob_drop"] for r in rows])),
        "mean_adversarial_sqli_prob": float(np.mean([r["adversarial_sqli_prob"] for r in rows])),
        "mean_queries": float(np.mean([r["attack_queries"] for r in rows])),
    }


def evaluate_view(model, rows: list[dict], seed: int, backbone: str, view: str, view_kind: str) -> dict:
    texts, labels = rows_to_xy(rows)
    probs = model.predict_proba(texts)
    return {
        "seed": seed,
        "backbone": backbone,
        "method": "clean_ce",
        "view": view,
        "view_kind": view_kind,
        "metrics": metrics_from_probs(probs, labels),
        "stats": summarize_rows(rows),
    }


def write_partial_output(args, started, result_rows, attack_rows, completed) -> None:
    partial_path = Path(str(args.output) + ".partial.json")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text(json.dumps({
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "completed": completed,
        "rows": result_rows,
        "attack_rows": attack_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def load_resume_state(args) -> tuple[list[dict], list[dict], set[tuple[int, str]], float]:
    for path in [Path(str(args.output) + ".partial.json"), Path(args.output)]:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result_rows = list(payload.get("rows", []))
        attack_rows = list(payload.get("attack_rows", []))
        completed = {
            (int(r["seed"]), str(r["backbone"]))
            for r in attack_rows
            if "seed" in r and "backbone" in r
        }
        elapsed = float(payload.get("elapsed_seconds", 0.0))
        print(f"Resuming from {path}: completed={len(completed)} elapsed={elapsed:.0f}s")
        return result_rows, attack_rows, completed, elapsed
    print("No resume file found, starting fresh.")
    return [], [], set(), 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--backbones", nargs="+", default=["word_svc", "textcnn", "bilstm", "codebert"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--output", default="experiments/formal/results_experiment1_targeted_attack.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--operator-set", choices=["wafamole", "advsqli"], default="wafamole")
    parser.add_argument("--attack-per-class", type=int, default=300)
    parser.add_argument("--search-steps", type=int, default=20)
    parser.add_argument("--candidates-per-state", type=int, default=48)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-chars", type=int, default=896)
    parser.add_argument("--attack-search-group-size", type=int, default=128)
    parser.add_argument("--no-early-stop", action="store_true")

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

    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--codebert-epochs", type=int, default=2)
    parser.add_argument("--codebert-batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--codebert-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--codebert-dropout", type=float, default=0.1)
    parser.add_argument("--freeze-encoder", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local-files-only", dest="local_files_only", action="store_true", default=True)
    group.add_argument("--allow-download", dest="local_files_only", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits_dir = Path(args.splits_dir)
    device = resolve_device(args.device)
    started = time.time()
    result_rows: list[dict] = []
    attack_rows: list[dict] = []
    completed: set[tuple[int, str]] = set()

    if args.resume:
        result_rows, attack_rows, completed, prev_elapsed = load_resume_state(args)
        started -= prev_elapsed

    for seed in args.seeds:
        set_seed(seed)
        train_rows = load_seed_split(splits_dir, seed, "train")
        valid_rows = load_seed_split(splits_dir, seed, "valid")
        clean_test_rows = load_seed_split(splits_dir, seed, "clean_test")
        train_texts, train_labels = rows_to_xy(train_rows)
        benign_rows, sqli_rows = pick_attack_rows(clean_test_rows, args.attack_per_class, seed)
        clean_attack_view = [
            {**r, "mutated": False, "source_text": r["text"], "mutation_family": None}
            for r in benign_rows + sqli_rows
        ]

        for backbone in args.backbones:
            if (seed, backbone) in completed:
                print(f"seed={seed} backbone={backbone} already completed; skipping")
                continue
            print(f"seed={seed} training {backbone}")
            model = build_model(backbone, args, device)
            model.fit(train_texts, train_labels)

            result_rows.append(evaluate_view(model, valid_rows, seed, backbone, "valid", "validation"))

            clean_eval = evaluate_view(model, clean_attack_view, seed, backbone, "clean_attack_matched", "clean")
            result_rows.append(clean_eval)
            print(
                f"  clean_attack_matched f1={clean_eval['metrics']['f1']:.4f} "
                f"recall={clean_eval['metrics']['recall']:.4f} "
                f"p10={clean_eval['metrics']['p10_sqli_prob']:.4f}"
            )

            print(
                f"  targeted search operator_set={args.operator_set} "
                f"sqli={len(sqli_rows)} steps={args.search_steps} "
                f"candidates={args.candidates_per_state} beam={args.beam_size}"
            )
            adv_sqli_rows = attack_sqli_rows(
                model=model,
                sqli_rows=sqli_rows,
                seed=seed,
                operator_set=args.operator_set,
                steps=args.search_steps,
                candidates_per_state=args.candidates_per_state,
                beam_size=args.beam_size,
                threshold=args.threshold,
                max_chars=args.max_chars,
                early_stop=not args.no_early_stop,
                group_size=args.attack_search_group_size,
            )
            adv_view = [
                {**r, "mutated": False, "source_text": r["text"], "mutation_family": None}
                for r in benign_rows
            ] + adv_sqli_rows
            adv_eval = evaluate_view(model, adv_view, seed, backbone, f"targeted_{args.operator_set}", "targeted_mutated")
            result_rows.append(adv_eval)

            atk_summary = summarize_attack_rows(adv_sqli_rows)
            attack_rows.append({
                "seed": seed,
                "backbone": backbone,
                "operator_set": args.operator_set,
                "attack_summary": atk_summary,
                "examples": adv_sqli_rows[:10],
            })
            print(
                f"  targeted_{args.operator_set} f1={adv_eval['metrics']['f1']:.4f} "
                f"recall={adv_eval['metrics']['recall']:.4f} "
                f"p10={adv_eval['metrics']['p10_sqli_prob']:.4f} "
                f"success={atk_summary.get('success_rate', 0):.4f} "
                f"drop={atk_summary.get('mean_prob_drop', 0):.4f}"
            )
            write_partial_output(args, started, result_rows, attack_rows, {"seed": seed, "backbone": backbone})

    summary: dict[str, dict] = {}
    for view in sorted({r["view"] for r in result_rows}):
        summary[view] = {}
        for backbone in args.backbones:
            vals = [r["metrics"] for r in result_rows if r["view"] == view and r["backbone"] == backbone]
            summary[view][backbone] = {
                "f1": summarize([v["f1"] for v in vals]),
                "recall": summarize([v["recall"] for v in vals]),
                "p10_sqli_prob": summarize([v["p10_sqli_prob"] for v in vals]),
            }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "elapsed_seconds": round(time.time() - started, 1),
        "rows": result_rows,
        "attack_rows": attack_rows,
        "summary": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
