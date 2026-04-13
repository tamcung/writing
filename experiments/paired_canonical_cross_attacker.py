#!/usr/bin/env python3
"""Explicit paired-data canonical-anchor experiment for cross-attacker generalization."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.consistency_sqli_experiment import (  # noqa: E402
    build_vocab,
    embedding_similarity,
    ensure_dataset,
    load_payload_data,
    make_split,
    make_test_view,
    mutate_payload_with_retries,
    metrics_from_probs,
    predict_proba,
    set_seed,
)
from experiments.paired_canonical_family_holdout import (  # noqa: E402
    PairTrainConfig,
    benign_nuisance_mutation,
    paired_summary,
    summarize,
    summarize_pairs,
    train_pair_model,
)


def build_train_pairs_from_source(
    train_texts: list[str],
    train_labels: list[int],
    seed: int,
    pairs_per_sample: int,
    rounds: int,
    hard: bool,
    retries: int,
    mutation_source: str,
    wafamole_repo: str,
    benign_pairs: str,
    benign_rounds: int,
    benign_retries: int,
) -> tuple[list[str], list[str], list[int]]:
    canon_texts: list[str] = []
    mut_texts: list[str] = []
    labels: list[int] = []

    for i, (text, y) in enumerate(zip(train_texts, train_labels)):
        repeats = pairs_per_sample if y == 1 else 1
        for p in range(repeats):
            canon_texts.append(text)
            labels.append(y)
            if y == 1:
                mut_texts.append(
                    mutate_payload_with_retries(
                        text=text,
                        seed=seed * 1_000_003 + i * 10_007 + p * 1_009,
                        hard=hard,
                        rounds=rounds,
                        mutation_source=mutation_source,
                        wafamole_repo=wafamole_repo,
                        ensure_changed=True,
                        retries=retries,
                    )
                )
            else:
                if benign_pairs == "nuisance":
                    mut_texts.append(
                        benign_nuisance_mutation(
                            text,
                            seed=seed * 1_000_003 + i * 10_007 + p * 1_009,
                            rounds=benign_rounds,
                            retries=benign_retries,
                        )
                    )
                else:
                    mut_texts.append(text)
    return canon_texts, mut_texts, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/SQLiV3_clean.json")
    parser.add_argument("--output", default="experiments/paired_canonical_cross_attacker_results.json")
    parser.add_argument("--wafamole-repo", default="external/WAF-A-MoLE")
    parser.add_argument("--methods", nargs="+", default=["pair_ce", "pair_proj_ce", "pair_canonical"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--canonical-logit-weight", type=float, default=0.0)
    parser.add_argument("--pairs-per-sample", type=int, default=1)
    parser.add_argument("--benign-pairs", choices=["identity", "nuisance"], default="identity")
    parser.add_argument("--benign-rounds", type=int, default=2)
    parser.add_argument("--benign-retries", type=int, default=4)
    parser.add_argument("--train-mutation-source", choices=["local", "wafamole", "advsqli"], default="wafamole")
    parser.add_argument("--test-mutation-source", choices=["local", "wafamole", "advsqli"], default="advsqli")
    parser.add_argument("--train-rounds", type=int, default=3)
    parser.add_argument("--test-rounds", type=int, default=3)
    parser.add_argument("--train-hard", action="store_true")
    parser.add_argument("--test-hard", action="store_true")
    parser.add_argument("--train-retries", type=int, default=8)
    parser.add_argument("--test-retries", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps":
        try:
            _ = torch.zeros(1, device="mps")
        except RuntimeError as exc:
            print(f"MPS unavailable at runtime ({exc}); falling back to CPU.")
            device = "cpu"

    data_path = ensure_dataset(Path(args.data))
    texts, labels = load_payload_data(data_path, max_len=args.max_len)
    print(f"Loaded payload-level records: total={len(texts)}, benign={labels.count(0)}, sqli={labels.count(1)}")
    set_seed(1234)
    rows = []
    started = time.time()

    for seed in args.seeds:
        train_texts, train_labels, test_texts, test_labels = make_split(
            texts,
            labels,
            seed=seed,
            train_per_class=args.train_per_class,
            test_per_class=args.test_per_class,
        )
        train_canon, train_mut, pair_labels = build_train_pairs_from_source(
            train_texts=train_texts,
            train_labels=train_labels,
            seed=seed,
            pairs_per_sample=args.pairs_per_sample,
            rounds=args.train_rounds,
            hard=args.train_hard,
            retries=args.train_retries,
            mutation_source=args.train_mutation_source,
            wafamole_repo=args.wafamole_repo,
            benign_pairs=args.benign_pairs,
            benign_rounds=args.benign_rounds,
            benign_retries=args.benign_retries,
        )
        pair_stats = summarize_pairs(train_canon, train_mut, pair_labels)
        print(
            "seed={} train_source={} changed {}/{} SQLi pairs, {}/{} benign pairs".format(
                seed,
                args.train_mutation_source,
                pair_stats["changed_sqli_pairs"],
                pair_stats["total_sqli_pairs"],
                pair_stats["changed_benign_pairs"],
                pair_stats["total_benign_pairs"],
            )
        )
        test_mut, test_mut_labels, test_base_sqli, test_aug_sqli = make_test_view(
            test_texts,
            test_labels,
            seed=seed + 10_000,
            hard=False,
            rounds=args.test_rounds,
            mutation_source=args.test_mutation_source,
            wafamole_repo=args.wafamole_repo,
        )
        test_hard, test_hard_labels, hard_base_sqli, hard_aug_sqli = make_test_view(
            test_texts,
            test_labels,
            seed=seed + 20_000,
            hard=True if args.test_hard else False,
            rounds=args.test_rounds,
            mutation_source=args.test_mutation_source,
            wafamole_repo=args.wafamole_repo,
        )
        vocab = build_vocab(train_canon + train_mut)

        for method in args.methods:
            cfg = PairTrainConfig(
                method=method,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_len=args.max_len,
                lr=args.lr,
                consistency_weight=args.consistency_weight,
                canonical_logit_weight=args.canonical_logit_weight,
                device=device,
            )
            print(f"  Training {method} seed={seed}")
            model = train_pair_model(cfg, train_canon, train_mut, pair_labels, vocab)
            clean_probs, clean_y = predict_proba(model, test_texts, test_labels, vocab, args.max_len, device, args.batch_size)
            mut_probs, mut_y = predict_proba(model, test_mut, test_mut_labels, vocab, args.max_len, device, args.batch_size)
            hard_probs, hard_y = predict_proba(
                model, test_hard, test_hard_labels, vocab, args.max_len, device, args.batch_size
            )
            sim_mut = embedding_similarity(model, test_base_sqli, test_aug_sqli, vocab, args.max_len, device, args.batch_size)
            sim_hard = embedding_similarity(model, hard_base_sqli, hard_aug_sqli, vocab, args.max_len, device, args.batch_size)
            row = {
                "seed": seed,
                "method": method,
                "clean": metrics_from_probs(clean_probs, clean_y),
                "mutated": metrics_from_probs(mut_probs, mut_y),
                "hard_mutated": metrics_from_probs(hard_probs, hard_y),
                "embedding_cosine_mutated": sim_mut,
                "embedding_cosine_hard_mutated": sim_hard,
            }
            rows.append(row)
            print(
                "    clean_f1={:.4f} mut_recall={:.4f} hard_recall={:.4f} hard_p10={:.4f} sim_hard={:.4f}".format(
                    row["clean"]["f1"],
                    row["mutated"]["recall"],
                    row["hard_mutated"]["recall"],
                    row["hard_mutated"]["p10_sqli_prob"],
                    row["embedding_cosine_hard_mutated"],
                )
            )

    summary = {}
    for method in args.methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {
            "clean_f1": summarize([row["clean"]["f1"] for row in method_rows]),
            "mutated_recall": summarize([row["mutated"]["recall"] for row in method_rows]),
            "hard_mutated_recall": summarize([row["hard_mutated"]["recall"] for row in method_rows]),
            "hard_mutated_p10_sqli_prob": summarize([row["hard_mutated"]["p10_sqli_prob"] for row in method_rows]),
            "embedding_cosine_hard_mutated": summarize(
                [row["embedding_cosine_hard_mutated"] for row in method_rows]
            ),
        }

    comparisons = {}
    if "pair_ce" in args.methods and "pair_proj_ce" in args.methods:
        comparisons["pair_proj_ce_minus_pair_ce"] = {
            "hard_mutated_recall": paired_summary(rows, ("hard_mutated", "recall"), "pair_ce", "pair_proj_ce"),
            "hard_mutated_p10_sqli_prob": paired_summary(
                rows, ("hard_mutated", "p10_sqli_prob"), "pair_ce", "pair_proj_ce"
            ),
            "clean_f1": paired_summary(rows, ("clean", "f1"), "pair_ce", "pair_proj_ce"),
        }
    if "pair_ce" in args.methods and "pair_canonical" in args.methods:
        comparisons["pair_canonical_minus_pair_ce"] = {
            "hard_mutated_recall": paired_summary(rows, ("hard_mutated", "recall"), "pair_ce", "pair_canonical"),
            "hard_mutated_p10_sqli_prob": paired_summary(
                rows, ("hard_mutated", "p10_sqli_prob"), "pair_ce", "pair_canonical"
            ),
            "clean_f1": paired_summary(rows, ("clean", "f1"), "pair_ce", "pair_canonical"),
        }
    if "pair_proj_ce" in args.methods and "pair_canonical" in args.methods:
        comparisons["pair_canonical_minus_pair_proj_ce"] = {
            "hard_mutated_recall": paired_summary(
                rows, ("hard_mutated", "recall"), "pair_proj_ce", "pair_canonical"
            ),
            "hard_mutated_p10_sqli_prob": paired_summary(
                rows, ("hard_mutated", "p10_sqli_prob"), "pair_proj_ce", "pair_canonical"
            ),
            "clean_f1": paired_summary(rows, ("clean", "f1"), "pair_proj_ce", "pair_canonical"),
        }

    result = {
        "config": vars(args) | {"device_resolved": device},
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
