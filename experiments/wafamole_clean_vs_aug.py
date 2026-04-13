#!/usr/bin/env python3
"""Compare clean-only vs augmentation training on the WAF-A-MoLE SQL dataset."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import sqlparse
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.consistency_sqli_experiment import (
    TrainConfig,
    build_vocab,
    embedding_similarity,
    make_split,
    make_test_view,
    metrics_from_probs,
    mutate_payload_with_retries,
    paired_summary,
    predict_proba,
    random_search_evasion,
    summarize,
    train_model,
)


def resolve_device(device: str) -> str:
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps":
        try:
            _ = torch.zeros(1, device="mps")
        except RuntimeError as exc:
            print(f"MPS unavailable at runtime ({exc}); falling back to CPU.")
            device = "cpu"
    return device


def _cache_path(dataset_dir: Path, stem: str) -> Path:
    return dataset_dir / f"{stem}.statements.jsonl"


def split_sql_chunks(dataset_dir: Path, stem: str) -> list[str]:
    cache_path = _cache_path(dataset_dir, stem)
    if cache_path.exists():
        return [json.loads(line)["text"] for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    chunk_paths = sorted(dataset_dir.glob(f"{stem}.*"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files found for {stem} under {dataset_dir}")

    parts = [path.read_text(encoding="utf-8", errors="ignore") for path in chunk_paths]
    statements = [stmt.strip() for stmt in sqlparse.split("".join(parts)) if stmt.strip()]
    cache_path.write_text(
        "".join(json.dumps({"text": stmt}, ensure_ascii=False) + "\n" for stmt in statements),
        encoding="utf-8",
    )
    return statements


def load_wafamole_sql_data(dataset_dir: Path, max_len: int) -> tuple[list[str], list[int], dict[str, int]]:
    attacks = split_sql_chunks(dataset_dir, "attacks.sql")
    sane = split_sql_chunks(dataset_dir, "sane.sql")

    by_text: dict[str, set[int]] = {}
    raw_counts = {"attacks": len(attacks), "sane": len(sane)}
    for stmt in sane:
        if len(stmt) <= max_len:
            by_text.setdefault(stmt, set()).add(0)
    for stmt in attacks:
        if len(stmt) <= max_len:
            by_text.setdefault(stmt, set()).add(1)

    texts: list[str] = []
    labels: list[int] = []
    conflict = 0
    for text, label_set in by_text.items():
        if len(label_set) != 1:
            conflict += 1
            continue
        texts.append(text)
        labels.append(next(iter(label_set)))

    meta = raw_counts | {
        "usable_total": len(texts),
        "usable_sane": int(sum(1 for y in labels if y == 0)),
        "usable_attacks": int(sum(1 for y in labels if y == 1)),
        "conflicts_dropped": conflict,
    }
    return texts, labels, meta


def build_train_aug_views(
    train_texts: list[str],
    train_labels: list[int],
    seed: int,
    mutation_source: str,
    wafamole_repo: str,
    num_views: int,
    rounds: int,
    ensure_changed: bool,
    retries: int,
) -> list[list[str]]:
    views: list[list[str]] = []
    for i, (text, y) in enumerate(zip(train_texts, train_labels)):
        if y == 1:
            sample_views = [
                mutate_payload_with_retries(
                    text,
                    seed=seed * 1_000_003 + i * 10_007 + v * 1_009,
                    hard=False,
                    rounds=rounds,
                    mutation_source=mutation_source,
                    wafamole_repo=wafamole_repo,
                    ensure_changed=ensure_changed,
                    retries=retries,
                )
                for v in range(num_views)
            ]
        else:
            sample_views = [text] * num_views
        views.append(sample_views)
    return views


def summarize_train_aug_views(
    train_texts: list[str],
    train_labels: list[int],
    train_aug_views: list[list[str]],
) -> dict[str, float]:
    total_attacks = 0
    changed_samples = 0
    changed_view_count = 0
    for text, y, views in zip(train_texts, train_labels, train_aug_views):
        if y != 1:
            continue
        total_attacks += 1
        changed_this = sum(1 for v in views if v != text)
        changed_view_count += changed_this
        if changed_this:
            changed_samples += 1
    return {
        "changed_samples": float(changed_samples),
        "total_attacks": float(total_attacks),
        "avg_changed_views_per_attack": float(changed_view_count / total_attacks) if total_attacks else math.nan,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="external/wafamole_dataset")
    parser.add_argument("--output", default="experiments/wafamole_clean_vs_aug_results.json")
    parser.add_argument("--wafamole-repo", default="external/WAF-A-MoLE")
    parser.add_argument("--mutation-source", choices=["wafamole"], default="wafamole")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--methods", nargs="+", default=["clean_only", "aug_ce"])
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--num-aug-views", type=int, default=1)
    parser.add_argument("--train-aug-rounds", type=int, default=1)
    parser.add_argument("--ensure-train-aug-changed", action="store_true")
    parser.add_argument("--train-aug-retries", type=int, default=8)
    parser.add_argument("--evasion-seeds", type=int, default=80)
    parser.add_argument("--evasion-rounds", type=int, default=18)
    parser.add_argument("--evasion-candidates", type=int, default=8)
    parser.add_argument("--consistency-weight", type=float, default=0.5)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)

    dataset_dir = Path(args.dataset_dir)
    texts, labels, data_meta = load_wafamole_sql_data(dataset_dir, max_len=args.max_len)
    print(
        "Loaded WAF-A-MoLE SQL statements: usable_total={}, sane={}, attacks={}, raw_attacks={}, raw_sane={}".format(
            data_meta["usable_total"],
            data_meta["usable_sane"],
            data_meta["usable_attacks"],
            data_meta["attacks"],
            data_meta["sane"],
        )
    )

    rows: list[dict] = []
    started = time.time()
    for seed in args.seeds:
        train_texts, train_labels, test_texts, test_labels = make_split(
            texts,
            labels,
            seed=seed,
            train_per_class=args.train_per_class,
            test_per_class=args.test_per_class,
        )
        train_aug = build_train_aug_views(
            train_texts=train_texts,
            train_labels=train_labels,
            seed=seed,
            mutation_source=args.mutation_source,
            wafamole_repo=args.wafamole_repo,
            num_views=args.num_aug_views,
            rounds=args.train_aug_rounds,
            ensure_changed=args.ensure_train_aug_changed,
            retries=args.train_aug_retries,
        )
        train_aug_meta = summarize_train_aug_views(train_texts, train_labels, train_aug)
        print(
            "Seed {}: changed {}/{} attack samples; avg changed views per attack={:.2f}".format(
                seed,
                int(train_aug_meta["changed_samples"]),
                int(train_aug_meta["total_attacks"]),
                train_aug_meta["avg_changed_views_per_attack"],
            )
        )

        vocab = build_vocab(train_texts + test_texts)
        test_mut, test_mut_labels, test_base_sqli, test_aug_sqli = make_test_view(
            test_texts,
            test_labels,
            seed=seed + 10_000,
            hard=False,
            rounds=2,
            mutation_source=args.mutation_source,
            wafamole_repo=args.wafamole_repo,
        )
        test_hard, test_hard_labels, hard_base_sqli, hard_aug_sqli = make_test_view(
            test_texts,
            test_labels,
            seed=seed + 20_000,
            hard=True,
            rounds=3,
            mutation_source=args.mutation_source,
            wafamole_repo=args.wafamole_repo,
        )
        evasion_seeds = [text for text, y in zip(test_texts, test_labels) if y == 1][: args.evasion_seeds]

        for method in args.methods:
            print(f"Seed {seed} | training {method} on {device}")
            cfg = TrainConfig(
                method=method,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_len=args.max_len,
                lr=args.lr,
                consistency_weight=args.consistency_weight,
                device=device,
            )
            model = train_model(cfg, train_texts, train_labels, train_aug, vocab)

            clean_probs, clean_y = predict_proba(
                model, test_texts, test_labels, vocab, args.max_len, device=device, batch_size=args.batch_size
            )
            mut_probs, mut_y = predict_proba(
                model, test_mut, test_mut_labels, vocab, args.max_len, device=device, batch_size=args.batch_size
            )
            hard_probs, hard_y = predict_proba(
                model, test_hard, test_hard_labels, vocab, args.max_len, device=device, batch_size=args.batch_size
            )
            evasion = random_search_evasion(
                model=model,
                seeds=evasion_seeds,
                vocab=vocab,
                max_len=args.max_len,
                device=device,
                rng_seed=seed + 30_000,
                rounds=args.evasion_rounds,
                candidates=args.evasion_candidates,
                threshold=0.5,
                mutation_source=args.mutation_source,
                wafamole_repo=args.wafamole_repo,
            )

            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "train_aug": train_aug_meta,
                    "clean_metrics": metrics_from_probs(clean_probs, clean_y),
                    "mutated_metrics": metrics_from_probs(mut_probs, mut_y),
                    "hard_mutated_metrics": metrics_from_probs(hard_probs, hard_y),
                    "embedding_cosine_hard_mutated": embedding_similarity(
                        model,
                        hard_base_sqli,
                        hard_aug_sqli,
                        vocab=vocab,
                        max_len=args.max_len,
                        device=device,
                        batch_size=args.batch_size,
                    ),
                    "evasion": evasion,
                }
            )

    summary = {}
    for method in args.methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {
            "clean_f1": summarize([row["clean_metrics"]["f1"] for row in method_rows]),
            "mutated_recall": summarize([row["mutated_metrics"]["recall"] for row in method_rows]),
            "hard_mutated_recall": summarize([row["hard_mutated_metrics"]["recall"] for row in method_rows]),
            "hard_mutated_p10_sqli_prob": summarize(
                [row["hard_mutated_metrics"]["p10_sqli_prob"] for row in method_rows]
            ),
            "embedding_cosine_hard_mutated": summarize(
                [row["embedding_cosine_hard_mutated"] for row in method_rows]
            ),
            "evasion_success_rate": summarize([row["evasion"]["success_rate"] for row in method_rows]),
            "evasion_final_prob_mean": summarize([row["evasion"]["final_prob_mean"] for row in method_rows]),
        }

    comparisons = {}
    if "clean_only" in args.methods and "aug_ce" in args.methods:
        comparisons["aug_ce_minus_clean_only"] = {
            "clean_f1": paired_summary(rows, ("clean_metrics", "f1"), "clean_only", "aug_ce"),
            "mutated_recall": paired_summary(rows, ("mutated_metrics", "recall"), "clean_only", "aug_ce"),
            "hard_mutated_recall": paired_summary(rows, ("hard_mutated_metrics", "recall"), "clean_only", "aug_ce"),
            "hard_mutated_p10_sqli_prob": paired_summary(
                rows, ("hard_mutated_metrics", "p10_sqli_prob"), "clean_only", "aug_ce"
            ),
            "embedding_cosine_hard_mutated": paired_summary(
                rows, ("embedding_cosine_hard_mutated",), "clean_only", "aug_ce"
            ),
            "evasion_success_rate": paired_summary(rows, ("evasion", "success_rate"), "clean_only", "aug_ce"),
            "evasion_final_prob_mean": paired_summary(
                rows, ("evasion", "final_prob_mean"), "clean_only", "aug_ce"
            ),
        }

    result = {
        "config": vars(args) | {"device_resolved": device},
        "data_meta": data_meta,
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "summary": summary,
        "comparisons": comparisons,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    return result


if __name__ == "__main__":
    run(parse_args())
