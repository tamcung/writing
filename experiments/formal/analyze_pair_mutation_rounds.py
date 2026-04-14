#!/usr/bin/env python3
"""Audit and validate mutation-round choices for paired SQLi training."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.formal.pair_data import build_pair_rows
from experiments.formal.run_experiment1_clean_ce import (
    build_model,
    load_seed_split,
    resolve_device,
    rows_to_xy,
    set_seed,
)
from experiments.formal.run_experiment1_targeted_attack import (
    attack_sqli_rows,
    pick_attack_rows,
    summarize_attack_rows,
)
from experiments.formal.run_experiment2_pair_training_targeted import (
    build_pair_model,
    evaluate_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/derived/formal_modsec_decoded/experiment1/splits")
    parser.add_argument("--rounds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--audit-seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--audit-sqli-limit", type=int, default=1000)
    parser.add_argument("--operator-set", choices=["conservative", "wafamole_style", "official_wafamole"], default="official_wafamole")
    parser.add_argument("--mutation-retries", type=int, default=8)
    parser.add_argument("--pair-max-chars", type=int, default=896)

    parser.add_argument("--backbone", choices=["textcnn", "bilstm"], default="textcnn")
    parser.add_argument("--method", choices=["pair_ce", "pair_proj_ce", "pair_canonical"], default="pair_canonical")
    parser.add_argument("--attack-per-class", type=int, default=100)
    parser.add_argument("--search-steps", type=int, default=12)
    parser.add_argument("--candidates-per-state", type=int, default=32)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-chars", type=int, default=896)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")

    parser.add_argument("--epochs", type=int, default=6)
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
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--canonical-logit-weight", type=float, default=0.0)
    parser.add_argument("--hard-align-gamma", type=float, default=0.5)
    parser.add_argument("--align-benign-pairs", action="store_true")

    parser.add_argument("--output", default="experiments/formal/results_pair_rounds_analysis.json")
    return parser.parse_args()


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return float(values[idx])


def summarize_lengths(pair_rows: list[dict]) -> dict:
    sqli_rows = [row for row in pair_rows if int(row["label"]) == 1]
    source_lens = sorted(len(str(row["x_canon"])) for row in sqli_rows)
    mutated_lens = sorted(len(str(row["x_raw_mut"])) for row in sqli_rows)
    ratios = sorted(
        len(str(row["x_raw_mut"])) / max(1, len(str(row["x_canon"])))
        for row in sqli_rows
    )
    changed_rows = [row for row in sqli_rows if row.get("changed")]
    return {
        "sqli_pairs": len(sqli_rows),
        "changed_pairs": len(changed_rows),
        "source_len_mean": float(np.mean(source_lens)) if source_lens else 0.0,
        "source_len_p90": _pct(source_lens, 0.9),
        "source_len_p95": _pct(source_lens, 0.95),
        "mutated_len_mean": float(np.mean(mutated_lens)) if mutated_lens else 0.0,
        "mutated_len_p90": _pct(mutated_lens, 0.9),
        "mutated_len_p95": _pct(mutated_lens, 0.95),
        "ratio_mean": float(np.mean(ratios)) if ratios else 0.0,
        "ratio_p90": _pct(ratios, 0.9),
        "ratio_p95": _pct(ratios, 0.95),
        "ratio_gt_1_25": float(np.mean([r > 1.25 for r in ratios])) if ratios else 0.0,
        "ratio_gt_1_50": float(np.mean([r > 1.50 for r in ratios])) if ratios else 0.0,
        "ratio_gt_2_00": float(np.mean([r > 2.00 for r in ratios])) if ratios else 0.0,
    }


def sample_train_rows(train_rows: list[dict], sqli_limit: int) -> list[dict]:
    benign = [row for row in train_rows if int(row["label"]) == 0]
    sqli = [row for row in train_rows if int(row["label"]) == 1][:sqli_limit]
    return benign[: len(sqli)] + sqli


def audit_rounds(args: argparse.Namespace) -> list[dict]:
    audit_rows: list[dict] = []
    splits_dir = Path(args.splits_dir)
    for rounds in args.rounds:
        for seed in args.audit_seeds:
            train_rows = load_seed_split(splits_dir, seed, "train")
            sampled_rows = sample_train_rows(train_rows, args.audit_sqli_limit)
            pair_rows, pair_stats = build_pair_rows(
                train_rows=sampled_rows,
                seed=seed,
                operator_set=args.operator_set,
                sqli_pairs_per_sample=1,
                benign_pairs_per_sample=1,
                mutation_rounds=rounds,
                mutation_retries=args.mutation_retries,
                max_chars=args.pair_max_chars,
            )
            audit_rows.append(
                {
                    "rounds": rounds,
                    "seed": seed,
                    "pair_stats": pair_stats,
                    "length_stats": summarize_lengths(pair_rows),
                }
            )
            print(
                f"audit rounds={rounds} seed={seed} "
                f"sqli_changed={pair_stats['sqli_changed_rate']:.4f} "
                f"mean_chain={pair_stats['mean_sqli_chain_len']:.3f} "
                f"ratio_mean={summarize_lengths(pair_rows)['ratio_mean']:.3f}"
            )
    return audit_rows


def _base_train_args(args: argparse.Namespace, rounds: int) -> Namespace:
    return Namespace(
        current_seed=0,
        train_operator_set=args.operator_set,
        attack_operator_set=args.operator_set,
        sqli_pairs_per_sample=1,
        benign_pairs_per_sample=1,
        mutation_rounds=rounds,
        mutation_retries=args.mutation_retries,
        pair_max_chars=args.pair_max_chars,
        attack_per_class=args.attack_per_class,
        search_steps=args.search_steps,
        candidates_per_state=args.candidates_per_state,
        beam_size=args.beam_size,
        threshold=args.threshold,
        max_chars=args.max_chars,
        no_early_stop=False,
        word_ngram_max=2,
        word_min_df=1,
        word_c=1.0,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        max_vocab=args.max_vocab,
        min_freq=args.min_freq,
        lowercase=args.lowercase,
        lr=args.lr,
        emb_dim=args.emb_dim,
        channels=args.channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        consistency_weight=args.consistency_weight,
        canonical_logit_weight=args.canonical_logit_weight,
        hard_align_gamma=args.hard_align_gamma,
        align_benign_pairs=args.align_benign_pairs,
        model_name="microsoft/codebert-base",
        codebert_epochs=2,
        codebert_batch_size=8,
        max_len=512,
        codebert_lr=1e-3,
        encoder_lr=2e-5,
        codebert_dropout=0.1,
        freeze_encoder=False,
        grad_clip=1.0,
        local_files_only=True,
        pairs_dir="",
        require_pairs=False,
    )


def eval_rounds(args: argparse.Namespace) -> list[dict]:
    device = resolve_device(args.device)
    splits_dir = Path(args.splits_dir)
    results: list[dict] = []

    for rounds in args.rounds:
        for seed in args.eval_seeds:
            set_seed(seed)
            train_rows = load_seed_split(splits_dir, seed, "train")
            clean_test_rows = load_seed_split(splits_dir, seed, "clean_test")
            benign_rows, sqli_rows = pick_attack_rows(clean_test_rows, args.attack_per_class, seed)
            clean_attack_view = [
                {**row, "mutated": False, "source_text": row["text"], "mutation_family": None}
                for row in benign_rows + sqli_rows
            ]

            train_args = _base_train_args(args, rounds)
            train_args.current_seed = seed
            clean_texts, clean_labels = rows_to_xy(train_rows)
            pair_rows, pair_stats = build_pair_rows(
                train_rows=train_rows,
                seed=seed,
                operator_set=args.operator_set,
                sqli_pairs_per_sample=1,
                benign_pairs_per_sample=1,
                mutation_rounds=rounds,
                mutation_retries=args.mutation_retries,
                max_chars=args.pair_max_chars,
            )
            pair_canon = [str(row["x_canon"]) for row in pair_rows]
            pair_mutated = [str(row["x_raw_mut"]) for row in pair_rows]
            pair_labels = [int(row["label"]) for row in pair_rows]
            train_bundle = {
                "clean_texts": clean_texts,
                "clean_labels": clean_labels,
                "pair_canon": pair_canon,
                "pair_mutated": pair_mutated,
                "pair_labels": pair_labels,
                "pair_stats": pair_stats,
            }

            model = build_pair_model(args.backbone, args.method, train_args, device)
            started = time.time()
            model.fit_pairs(pair_canon, pair_mutated, pair_labels)
            train_seconds = time.time() - started

            clean_eval = evaluate_rows(
                model,
                clean_attack_view,
                seed,
                args.backbone,
                args.method,
                "clean_attack_matched",
                "clean",
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
                early_stop=True,
            )
            adv_view = [
                {**row, "mutated": False, "source_text": row["text"], "mutation_family": None}
                for row in benign_rows
            ] + adv_sqli_rows
            adv_eval = evaluate_rows(
                model,
                adv_view,
                seed,
                args.backbone,
                args.method,
                f"targeted_{args.operator_set}",
                "targeted_mutated",
            )
            attack_summary = summarize_attack_rows(adv_sqli_rows)
            results.append(
                {
                    "rounds": rounds,
                    "seed": seed,
                    "backbone": args.backbone,
                    "method": args.method,
                    "pair_stats": pair_stats,
                    "train_seconds": train_seconds,
                    "clean_metrics": clean_eval["metrics"],
                    "targeted_metrics": adv_eval["metrics"],
                    "attack_summary": attack_summary,
                }
            )
            print(
                f"eval rounds={rounds} seed={seed} "
                f"clean_recall={clean_eval['metrics']['recall']:.4f} "
                f"targeted_recall={adv_eval['metrics']['recall']:.4f} "
                f"success={attack_summary.get('success_rate', 0.0):.4f} "
                f"train_s={train_seconds:.1f}"
            )
    return results


def aggregate_by_round(rows: list[dict], key: str) -> dict[str, dict]:
    out: dict[str, list[float]] = {}
    for row in rows:
        rounds = str(row["rounds"])
        value = row
        for part in key.split("."):
            value = value[part]
        out.setdefault(rounds, []).append(float(value))
    return {
        rounds: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n": len(values),
        }
        for rounds, values in out.items()
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    audit_rows = audit_rounds(args)
    eval_rows = eval_rounds(args)
    payload = {
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "audit_rows": audit_rows,
        "eval_rows": eval_rows,
        "summary": {
            "audit_sqli_changed_rate": aggregate_by_round(audit_rows, "pair_stats.sqli_changed_rate"),
            "audit_mean_chain_len": aggregate_by_round(audit_rows, "pair_stats.mean_sqli_chain_len"),
            "audit_ratio_mean": aggregate_by_round(audit_rows, "length_stats.ratio_mean"),
            "audit_ratio_p95": aggregate_by_round(audit_rows, "length_stats.ratio_p95"),
            "eval_clean_recall": aggregate_by_round(eval_rows, "clean_metrics.recall"),
            "eval_targeted_recall": aggregate_by_round(eval_rows, "targeted_metrics.recall"),
            "eval_attack_success": aggregate_by_round(eval_rows, "attack_summary.success_rate"),
            "eval_prob_drop": aggregate_by_round(eval_rows, "attack_summary.mean_prob_drop"),
            "eval_train_seconds": aggregate_by_round(eval_rows, "train_seconds"),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote analysis to {output_path}")


if __name__ == "__main__":
    main()
