#!/usr/bin/env python3
"""Paired canonical-anchor experiment with custom semantic-preserving mutation families."""

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

from experiments.consistency_sqli_experiment import (
    build_vocab,
    embedding_similarity,
    ensure_dataset,
    load_payload_data,
    make_split,
    metrics_from_probs,
    predict_proba,
    set_seed,
)
from experiments.paired_canonical_family_holdout import (
    PairTrainConfig,
    benign_nuisance_mutation,
    paired_summary,
    summarize,
    summarize_pairs,
    train_pair_model,
)
from experiments.semantic_mutation_v2 import (
    SEMANTIC_FAMILIES,
    apply_semantic_strategy_rounds,
    build_semantic_test_family_view,
)


def build_semantic_train_pairs(
    train_texts: list[str],
    train_labels: list[int],
    seed: int,
    strategy_names: list[str],
    pairs_per_sample: int,
    rounds: int,
    retries: int,
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
            sample_seed = seed * 1_000_003 + i * 10_007 + p * 1_009
            if y == 1:
                mut_texts.append(
                    apply_semantic_strategy_rounds(
                        text=text,
                        strategy_names=strategy_names,
                        seed=sample_seed,
                        rounds=rounds,
                        ensure_changed=True,
                        retries=retries,
                    )
                )
            else:
                if benign_pairs == "nuisance":
                    mut_texts.append(
                        benign_nuisance_mutation(
                            text=text,
                            seed=sample_seed,
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
    parser.add_argument("--output", default="experiments/paired_canonical_semantic_holdout_results.json")
    parser.add_argument("--families", nargs="+", default=["numeric_repr", "string_construction"])
    parser.add_argument("--methods", nargs="+", default=["pair_proj_ce", "pair_canonical"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--canonical-logit-weight", type=float, default=0.0)
    parser.add_argument("--hard-align-gamma", type=float, default=0.5)
    parser.add_argument("--pairs-per-sample", type=int, default=1)
    parser.add_argument("--benign-pairs", choices=["identity", "nuisance"], default="nuisance")
    parser.add_argument("--benign-rounds", type=int, default=2)
    parser.add_argument("--benign-retries", type=int, default=4)
    parser.add_argument("--train-rounds", type=int, default=3)
    parser.add_argument("--test-rounds", type=int, default=3)
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
    started = time.time()
    family_results = {}

    for holdout_family in args.families:
        heldout_names = SEMANTIC_FAMILIES[holdout_family]
        train_names = [name for fam, names in SEMANTIC_FAMILIES.items() if fam != holdout_family for name in names]
        print(f"\n=== Semantic Holdout family: {holdout_family} ===")
        rows = []

        for seed in args.seeds:
            train_texts, train_labels, test_texts, test_labels = make_split(
                texts,
                labels,
                seed=seed,
                train_per_class=args.train_per_class,
                test_per_class=args.test_per_class,
            )
            train_canon, train_mut, pair_labels = build_semantic_train_pairs(
                train_texts=train_texts,
                train_labels=train_labels,
                seed=seed,
                strategy_names=train_names,
                pairs_per_sample=args.pairs_per_sample,
                rounds=args.train_rounds,
                retries=args.train_retries,
                benign_pairs=args.benign_pairs,
                benign_rounds=args.benign_rounds,
                benign_retries=args.benign_retries,
            )
            pair_stats = summarize_pairs(train_canon, train_mut, pair_labels)
            print(
                "seed={} semantic paired train changed {}/{} SQLi pairs, {}/{} benign pairs".format(
                    seed,
                    pair_stats["changed_sqli_pairs"],
                    pair_stats["total_sqli_pairs"],
                    pair_stats["changed_benign_pairs"],
                    pair_stats["total_benign_pairs"],
                )
            )
            heldout_texts, heldout_labels, heldout_base, heldout_aug = build_semantic_test_family_view(
                test_texts=test_texts,
                test_labels=test_labels,
                seed=seed + 20_000,
                strategy_names=heldout_names,
                rounds=args.test_rounds,
                retries=args.test_retries,
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
                    hard_align_gamma=args.hard_align_gamma,
                )
                print(f"  Training {method} seed={seed}")
                model = train_pair_model(cfg, train_canon, train_mut, pair_labels, vocab)
                clean_probs, clean_y = predict_proba(
                    model, test_texts, test_labels, vocab, args.max_len, device, args.batch_size
                )
                held_probs, held_y = predict_proba(
                    model, heldout_texts, heldout_labels, vocab, args.max_len, device, args.batch_size
                )
                held_sim = embedding_similarity(
                    model, heldout_base, heldout_aug, vocab, args.max_len, device, args.batch_size
                )
                row = {
                    "seed": seed,
                    "method": method,
                    "holdout_family": holdout_family,
                    "clean": metrics_from_probs(clean_probs, clean_y),
                    "heldout_family": metrics_from_probs(held_probs, held_y),
                    "embedding_cosine_heldout_family": held_sim,
                }
                rows.append(row)
                print(
                    "    clean_f1={:.4f} heldout_recall={:.4f} heldout_p10={:.4f} heldout_sim={:.4f}".format(
                        row["clean"]["f1"],
                        row["heldout_family"]["recall"],
                        row["heldout_family"]["p10_sqli_prob"],
                        row["embedding_cosine_heldout_family"],
                    )
                )

        summary = {}
        for method in args.methods:
            method_rows = [row for row in rows if row["method"] == method]
            summary[method] = {
                "clean_f1": summarize([row["clean"]["f1"] for row in method_rows]),
                "heldout_family_recall": summarize([row["heldout_family"]["recall"] for row in method_rows]),
                "heldout_family_p10_sqli_prob": summarize(
                    [row["heldout_family"]["p10_sqli_prob"] for row in method_rows]
                ),
                "embedding_cosine_heldout_family": summarize(
                    [row["embedding_cosine_heldout_family"] for row in method_rows]
                ),
            }

        comparisons = {}
        if "pair_proj_ce" in args.methods and "pair_canonical" in args.methods:
            comparisons["pair_canonical_minus_pair_proj_ce"] = {
                "heldout_family_recall": paired_summary(
                    rows, ("heldout_family", "recall"), "pair_proj_ce", "pair_canonical"
                ),
                "heldout_family_p10_sqli_prob": paired_summary(
                    rows, ("heldout_family", "p10_sqli_prob"), "pair_proj_ce", "pair_canonical"
                ),
                "clean_f1": paired_summary(rows, ("clean", "f1"), "pair_proj_ce", "pair_canonical"),
            }

        family_results[holdout_family] = {
            "rows": rows,
            "summary": summary,
            "comparisons": comparisons,
            "heldout_strategies": heldout_names,
            "train_strategies": train_names,
        }

    result = {
        "config": vars(args) | {"device_resolved": device},
        "elapsed_seconds": time.time() - started,
        "families": family_results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    return result


if __name__ == "__main__":
    run(parse_args())
