#!/usr/bin/env python3
"""Find where consistency helps by testing unseen official WAF-A-MoLE mutation families."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.consistency_sqli_experiment import (  # noqa: E402
    build_vocab,
    embedding_similarity,
    ensure_dataset,
    load_payload_data,
    metrics_from_probs,
    predict_proba,
    set_seed,
    train_model,
    make_split,
)


ALL_FAMILIES = {
    "whitespace": [
        "spaces_to_comments",
        "spaces_to_whitespaces_alternatives",
        "comment_rewriting",
        "reset_inline_comments",
    ],
    "lexical": [
        "random_case",
        "swap_keywords",
    ],
    "numeric": [
        "swap_int_repr",
    ],
    "tautology": [
        "change_tautologies",
        "logical_invariant",
    ],
}


def get_sqlfuzzer_module(wafamole_repo: str):
    repo = Path(wafamole_repo)
    if not repo.exists():
        raise FileNotFoundError(f"WAF-A-MoLE repo not found: {repo}")
    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return importlib.import_module("wafamole.payloadfuzzer.sqlfuzzer")


def apply_strategy_rounds(
    text: str,
    strategy_names: list[str],
    seed: int,
    rounds: int,
    ensure_changed: bool,
    retries: int,
    wafamole_repo: str,
) -> str:
    module = get_sqlfuzzer_module(wafamole_repo)
    old_state = random.getstate()
    try:
        last = text
        for attempt in range(max(1, retries)):
            random.seed(seed + attempt * 104_729)
            out = text
            for _ in range(max(1, rounds)):
                fn_name = random.choice(strategy_names)
                fn = getattr(module, fn_name)
                out = fn(out)
            last = out
            if not ensure_changed or out != text:
                return out[:260]
        return last[:260]
    finally:
        random.setstate(old_state)


def build_multiview_train(
    train_texts: list[str],
    train_labels: list[int],
    seed: int,
    strategy_names: list[str],
    num_views: int,
    rounds: int,
    retries: int,
    wafamole_repo: str,
) -> list[list[str]]:
    views: list[list[str]] = []
    for i, (text, y) in enumerate(zip(train_texts, train_labels)):
        if y == 1:
            sample_views = [
                apply_strategy_rounds(
                    text=text,
                    strategy_names=strategy_names,
                    seed=seed * 1_000_003 + i * 10_007 + v * 1_009,
                    rounds=rounds,
                    ensure_changed=True,
                    retries=retries,
                    wafamole_repo=wafamole_repo,
                )
                for v in range(num_views)
            ]
        else:
            sample_views = [text] * num_views
        views.append(sample_views)
    return views


def summarize_train_views(train_texts: list[str], train_labels: list[int], train_views: list[list[str]]) -> dict:
    total_sqli = 0
    samples_with_change = 0
    changed_view_count = 0
    for text, y, views in zip(train_texts, train_labels, train_views):
        if y != 1:
            continue
        total_sqli += 1
        changed_this = sum(1 for view in views if view != text)
        changed_view_count += changed_this
        if changed_this:
            samples_with_change += 1
    return {
        "changed_samples": samples_with_change,
        "total_sqli": total_sqli,
        "avg_changed_views_per_sqli": changed_view_count / total_sqli if total_sqli else 0.0,
    }


def build_test_family_view(
    test_texts: list[str],
    test_labels: list[int],
    seed: int,
    strategy_names: list[str],
    rounds: int,
    retries: int,
    wafamole_repo: str,
) -> tuple[list[str], list[int], list[str], list[str]]:
    texts = []
    base_sqli = []
    aug_sqli = []
    for i, (text, y) in enumerate(zip(test_texts, test_labels)):
        if y == 1:
            aug = apply_strategy_rounds(
                text=text,
                strategy_names=strategy_names,
                seed=seed * 100_003 + i,
                rounds=rounds,
                ensure_changed=True,
                retries=retries,
                wafamole_repo=wafamole_repo,
            )
            texts.append(aug)
            base_sqli.append(text)
            aug_sqli.append(aug)
        else:
            texts.append(text)
    return texts, list(test_labels), base_sqli, aug_sqli


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def paired_summary(rows: list[dict], metric_path: tuple[str, ...], a: str, b: str) -> dict[str, float]:
    by_seed = {}
    for row in rows:
        value = row
        for key in metric_path:
            value = value[key]
        by_seed.setdefault(row["seed"], {})[row["method"]] = float(value)

    diffs = []
    for seed, methods in by_seed.items():
        if a in methods and b in methods:
            diffs.append(methods[b] - methods[a])

    result = {
        "n": float(len(diffs)),
        "mean_diff": float(np.mean(diffs)) if diffs else math.nan,
        "std_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
    }
    if len(diffs) >= 3 and np.std(diffs, ddof=1) > 0:
        result["paired_t_p"] = float(stats.ttest_1samp(diffs, 0.0).pvalue)
    else:
        result["paired_t_p"] = math.nan
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/SQLiV3_clean.json")
    parser.add_argument("--output", default="experiments/consistency_family_holdout_results.json")
    parser.add_argument("--wafamole-repo", default="external/WAF-A-MoLE")
    parser.add_argument("--families", nargs="+", default=list(ALL_FAMILIES.keys()))
    parser.add_argument("--methods", nargs="+", default=["aug_ce", "aug_consistency", "aug_infonce"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--train-per-class", type=int, default=300)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--consistency-weight", type=float, default=0.5)
    parser.add_argument("--canonical-logit-weight", type=float, default=0.25)
    parser.add_argument("--num-aug-views", type=int, default=3)
    parser.add_argument("--train-rounds", type=int, default=3)
    parser.add_argument("--test-rounds", type=int, default=3)
    parser.add_argument("--train-retries", type=int, default=8)
    parser.add_argument("--test-retries", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    import torch
    from experiments.consistency_sqli_experiment import TrainConfig

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

    all_strategy_names = {k: list(v) for k, v in ALL_FAMILIES.items()}
    family_results = {}
    started = time.time()

    for holdout_family in args.families:
        heldout_names = all_strategy_names[holdout_family]
        train_names = [
            name
            for fam, names in all_strategy_names.items()
            if fam != holdout_family
            for name in names
        ]
        print(f"\n=== Holdout family: {holdout_family} ===")
        rows = []

        for seed in args.seeds:
            train_texts, train_labels, test_texts, test_labels = make_split(
                texts,
                labels,
                seed=seed,
                train_per_class=args.train_per_class,
                test_per_class=args.test_per_class,
            )
            train_views = build_multiview_train(
                train_texts,
                train_labels,
                seed=seed,
                strategy_names=train_names,
                num_views=args.num_aug_views,
                rounds=args.train_rounds,
                retries=args.train_retries,
                wafamole_repo=args.wafamole_repo,
            )
            train_view_stats = summarize_train_views(train_texts, train_labels, train_views)
            print(
                "seed={} train changed {}/{} SQLi, avg changed views {:.2f}".format(
                    seed,
                    train_view_stats["changed_samples"],
                    train_view_stats["total_sqli"],
                    train_view_stats["avg_changed_views_per_sqli"],
                )
            )

            heldout_texts, heldout_labels, heldout_base, heldout_aug = build_test_family_view(
                test_texts,
                test_labels,
                seed=seed + 20_000,
                strategy_names=heldout_names,
                rounds=args.test_rounds,
                retries=args.test_retries,
                wafamole_repo=args.wafamole_repo,
            )
            vocab = build_vocab(train_texts + [view for views in train_views for view in views])

            for method in args.methods:
                cfg = TrainConfig(
                    method=method,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_len=args.max_len,
                    lr=args.lr,
                    consistency_weight=args.consistency_weight,
                    device=device,
                    canonical_logit_weight=args.canonical_logit_weight,
                )
                print(f"  Training {method} seed={seed}")
                model = train_model(cfg, train_texts, train_labels, train_views, vocab)
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
        if "aug_ce" in args.methods and "aug_consistency" in args.methods:
            comparisons["aug_consistency_minus_aug_ce"] = {
                "heldout_family_recall": paired_summary(
                    rows, ("heldout_family", "recall"), "aug_ce", "aug_consistency"
                ),
                "heldout_family_p10_sqli_prob": paired_summary(
                    rows, ("heldout_family", "p10_sqli_prob"), "aug_ce", "aug_consistency"
                ),
                "embedding_cosine_heldout_family": paired_summary(
                    rows, ("embedding_cosine_heldout_family",), "aug_ce", "aug_consistency"
                ),
            }
        if "aug_ce" in args.methods and "aug_infonce" in args.methods:
            comparisons["aug_infonce_minus_aug_ce"] = {
                "heldout_family_recall": paired_summary(
                    rows, ("heldout_family", "recall"), "aug_ce", "aug_infonce"
                ),
                "heldout_family_p10_sqli_prob": paired_summary(
                    rows, ("heldout_family", "p10_sqli_prob"), "aug_ce", "aug_infonce"
                ),
                "embedding_cosine_heldout_family": paired_summary(
                    rows, ("embedding_cosine_heldout_family",), "aug_ce", "aug_infonce"
                ),
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
