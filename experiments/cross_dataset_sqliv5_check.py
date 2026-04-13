#!/usr/bin/env python3
"""Cross-dataset check: train on SQLiV3_clean, test on SQLiV5."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.clean_backbone_semantic_holdout import (  # noqa: E402
    predict_charcnn,
    predict_textcnn,
    train_charcnn,
    train_textcnn,
)
from experiments.consistency_sqli_experiment import (  # noqa: E402
    build_vocab,
    load_payload_data,
    make_split,
    metrics_from_probs,
    set_seed,
)
from experiments.paired_canonical_family_holdout import (  # noqa: E402
    PairTrainConfig,
    train_pair_model as train_char_pair_model,
)
from experiments.paired_canonical_semantic_holdout import build_semantic_train_pairs  # noqa: E402
from experiments.paired_canonical_textcnn_check import (  # noqa: E402
    TextCNNConfig,
    build_token_vocab,
    predict_proba as predict_text_pair,
    train_pair_model as train_text_pair_model,
)
from experiments.semantic_mutation_v2 import SEMANTIC_FAMILIES  # noqa: E402


SQLIV5_URL = "https://raw.githubusercontent.com/nidnogg/sqliv5-dataset/main/SQLiV5.json"


def ensure_file(path: Path, url: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {url} -> {path}")
        urlretrieve(url, path)
    return path


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


def balanced_sample(
    texts: list[str],
    labels: list[int],
    seed: int,
    per_class: int,
) -> tuple[list[str], list[int]]:
    idx0 = [i for i, y in enumerate(labels) if y == 0]
    idx1 = [i for i, y in enumerate(labels) if y == 1]
    rng = random.Random(seed)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n = min(per_class, len(idx0), len(idx1))
    if n <= 0:
        raise ValueError(f"Cannot build balanced sample: benign={len(idx0)}, sqli={len(idx1)}")
    idx = idx0[:n] + idx1[:n]
    rng.shuffle(idx)
    return [texts[i] for i in idx], [labels[i] for i in idx]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def paired_summary(rows: list[dict], view: str, metric_path: tuple[str, ...], a: tuple[str, str], b: tuple[str, str]) -> dict:
    by_seed = {}
    for row in rows:
        if row["view"] != view:
            continue
        value = row
        for key in metric_path:
            value = value[key]
        by_seed.setdefault(row["seed"], {})[(row["backbone"], row["method"])] = float(value)
    diffs = []
    for methods in by_seed.values():
        if a in methods and b in methods:
            diffs.append(methods[b] - methods[a])
    return {
        "n": float(len(diffs)),
        "mean_diff": float(np.mean(diffs)) if diffs else float("nan"),
        "std_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
        "paired_t_p": float(stats.ttest_1samp(diffs, popmean=0.0).pvalue) if len(diffs) > 1 else float("nan"),
    }


def train_and_eval_char(
    method: str,
    train_texts: list[str],
    train_labels: list[int],
    pair_canon: list[str],
    pair_mut: list[str],
    pair_labels: list[int],
    eval_views: dict[str, tuple[list[str], list[int]]],
    args: argparse.Namespace,
    seed: int,
    device: str,
) -> list[dict]:
    if method == "clean_ce":
        model, vocab = train_charcnn(
            train_texts,
            train_labels,
            seed,
            args.epochs,
            args.batch_size,
            args.max_chars,
            args.lr,
            device,
        )
    else:
        vocab = build_vocab(pair_canon + pair_mut)
        cfg = PairTrainConfig(
            method=method,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_len=args.max_chars,
            lr=args.lr,
            consistency_weight=args.consistency_weight,
            canonical_logit_weight=args.canonical_logit_weight,
            device=device,
            hard_align_gamma=args.hard_align_gamma if method == "pair_canonical" else 0.0,
        )
        model = train_char_pair_model(cfg, pair_canon, pair_mut, pair_labels, vocab)

    rows = []
    for view, (texts, labels) in eval_views.items():
        probs, y = predict_charcnn(model, texts, labels, vocab, args.max_chars, device, args.batch_size)
        rows.append(
            {
                "seed": seed,
                "backbone": "charcnn",
                "method": method,
                "view": view,
                "metrics": metrics_from_probs(probs, y),
            }
        )
    return rows


def train_and_eval_text(
    method: str,
    train_texts: list[str],
    train_labels: list[int],
    pair_canon: list[str],
    pair_mut: list[str],
    pair_labels: list[int],
    eval_views: dict[str, tuple[list[str], list[int]]],
    args: argparse.Namespace,
    seed: int,
    device: str,
) -> list[dict]:
    if method == "clean_ce":
        model, vocab = train_textcnn(
            train_texts,
            train_labels,
            seed,
            args.epochs,
            args.batch_size,
            args.max_tokens,
            args.max_vocab,
            args.min_freq,
            args.lowercase,
            args.lr,
            args.emb_dim,
            args.channels,
            args.dropout,
            device,
        )
        predict = lambda texts, labels: predict_textcnn(
            model, texts, labels, vocab, args.max_tokens, args.lowercase, device, args.batch_size
        )
    else:
        vocab = build_token_vocab(
            pair_canon + pair_mut,
            lowercase=args.lowercase,
            max_vocab=args.max_vocab,
            min_freq=args.min_freq,
        )
        cfg = TextCNNConfig(
            method=method,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            lr=args.lr,
            consistency_weight=args.consistency_weight,
            canonical_logit_weight=args.canonical_logit_weight,
            hard_align_gamma=args.hard_align_gamma if method == "pair_canonical" else 0.0,
            device=device,
            emb_dim=args.emb_dim,
            channels=args.channels,
            dropout=args.dropout,
            lowercase=args.lowercase,
        )
        model = train_text_pair_model(cfg, pair_canon, pair_mut, pair_labels, vocab)
        predict = lambda texts, labels: predict_text_pair(
            model, texts, labels, vocab, args.max_tokens, args.lowercase, device, args.batch_size
        )

    rows = []
    for view, (texts, labels) in eval_views.items():
        probs, y = predict(texts, labels)
        rows.append(
            {
                "seed": seed,
                "backbone": "textcnn",
                "method": method,
                "view": view,
                "metrics": metrics_from_probs(probs, y),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/raw/SQLiV3_clean.json")
    parser.add_argument("--test-data", default="data/raw/SQLiV5.json")
    parser.add_argument("--output", default="experiments/cross_dataset_sqliv5_results.json")
    parser.add_argument("--backbones", nargs="+", default=["charcnn", "textcnn"])
    parser.add_argument("--methods", nargs="+", default=["clean_ce", "pair_ce", "pair_proj_ce", "pair_canonical"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-chars", type=int, default=260)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--canonical-logit-weight", type=float, default=0.0)
    parser.add_argument("--hard-align-gamma", type=float, default=0.5)
    parser.add_argument("--pairs-per-sample", type=int, default=1)
    parser.add_argument("--train-rounds", type=int, default=3)
    parser.add_argument("--train-retries", type=int, default=8)
    parser.add_argument("--benign-rounds", type=int, default=2)
    parser.add_argument("--benign-retries", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    train_path = Path(args.train_data)
    test_path = ensure_file(Path(args.test_data), SQLIV5_URL)
    train_all_texts, train_all_labels = load_payload_data(train_path, max_len=args.max_chars)
    v5_texts, v5_labels = load_payload_data(test_path, max_len=args.max_chars)
    v3_all = set(train_all_texts)
    print(
        "Loaded train={} benign={} sqli={} | test={} benign={} sqli={}".format(
            len(train_all_texts),
            train_all_labels.count(0),
            train_all_labels.count(1),
            len(v5_texts),
            v5_labels.count(0),
            v5_labels.count(1),
        )
    )
    print(f"Exact V3/V5 overlap after filtering: {len(v3_all & set(v5_texts))}/{len(set(v5_texts))}")
    strategy_names = [name for names in SEMANTIC_FAMILIES.values() for name in names]
    rows: list[dict] = []
    started = time.time()

    for seed in args.seeds:
        train_texts, train_labels, v3_test_texts, v3_test_labels = make_split(
            train_all_texts,
            train_all_labels,
            seed=seed,
            train_per_class=args.train_per_class,
            test_per_class=args.test_per_class,
        )
        train_set = set(train_texts)
        v5_no_train_texts = [text for text in v5_texts if text not in train_set]
        v5_no_train_labels = [y for text, y in zip(v5_texts, v5_labels) if text not in train_set]

        new_sqli = [text for text, y in zip(v5_texts, v5_labels) if y == 1 and text not in v3_all]
        benign_pool = [text for text, y in zip(v5_texts, v5_labels) if y == 0 and text not in train_set]
        strict_texts = benign_pool + new_sqli
        strict_labels = [0] * len(benign_pool) + [1] * len(new_sqli)
        strict_texts, strict_labels = balanced_sample(strict_texts, strict_labels, seed + 9_999, args.test_per_class)

        eval_views = {
            "v3_holdout": (v3_test_texts, v3_test_labels),
            "v5_no_train_overlap": (v5_no_train_texts, v5_no_train_labels),
            "v5_new_sqli_balanced": (strict_texts, strict_labels),
        }
        print(
            "seed={} train={} v5_no_train={} strict_balanced={} strict_sqli_pool={} benign_pool={}".format(
                seed,
                len(train_texts),
                len(v5_no_train_texts),
                len(strict_texts),
                len(new_sqli),
                len(benign_pool),
            )
        )

        pair_canon: list[str] = []
        pair_mut: list[str] = []
        pair_labels: list[int] = []
        if any(method != "clean_ce" for method in args.methods):
            pair_canon, pair_mut, pair_labels = build_semantic_train_pairs(
                train_texts=train_texts,
                train_labels=train_labels,
                seed=seed,
                strategy_names=strategy_names,
                pairs_per_sample=args.pairs_per_sample,
                rounds=args.train_rounds,
                retries=args.train_retries,
                benign_pairs="nuisance",
                benign_rounds=args.benign_rounds,
                benign_retries=args.benign_retries,
            )

        for backbone in args.backbones:
            for method in args.methods:
                print(f"  Training {backbone}/{method} seed={seed}")
                if backbone == "charcnn":
                    method_rows = train_and_eval_char(
                        method,
                        train_texts,
                        train_labels,
                        pair_canon,
                        pair_mut,
                        pair_labels,
                        eval_views,
                        args,
                        seed,
                        device,
                    )
                elif backbone == "textcnn":
                    method_rows = train_and_eval_text(
                        method,
                        train_texts,
                        train_labels,
                        pair_canon,
                        pair_mut,
                        pair_labels,
                        eval_views,
                        args,
                        seed,
                        device,
                    )
                else:
                    raise ValueError(f"Unknown backbone: {backbone}")
                rows.extend(method_rows)
                for row in method_rows:
                    print(
                        "    {} f1={:.4f} recall={:.4f} p10={:.4f}".format(
                            row["view"],
                            row["metrics"]["f1"],
                            row["metrics"]["recall"],
                            row["metrics"]["p10_sqli_prob"],
                        )
                    )

    summary: dict[str, dict] = {}
    for view in sorted({row["view"] for row in rows}):
        summary[view] = {}
        for backbone in args.backbones:
            summary[view][backbone] = {}
            for method in args.methods:
                method_rows = [
                    row for row in rows if row["view"] == view and row["backbone"] == backbone and row["method"] == method
                ]
                summary[view][backbone][method] = {
                    "f1": summarize([row["metrics"]["f1"] for row in method_rows]),
                    "recall": summarize([row["metrics"]["recall"] for row in method_rows]),
                    "p10_sqli_prob": summarize([row["metrics"]["p10_sqli_prob"] for row in method_rows]),
                    "mean_sqli_prob": summarize([row["metrics"]["mean_sqli_prob"] for row in method_rows]),
                }

    comparisons: dict[str, dict] = {}
    for view in sorted({row["view"] for row in rows}):
        comparisons[view] = {}
        for backbone in args.backbones:
            for a, b in [
                ("clean_ce", "pair_ce"),
                ("pair_ce", "pair_proj_ce"),
                ("pair_proj_ce", "pair_canonical"),
                ("pair_ce", "pair_canonical"),
            ]:
                if a in args.methods and b in args.methods:
                    comparisons[view][f"{backbone}_{b}_minus_{a}"] = {
                        "f1": paired_summary(rows, view, ("metrics", "f1"), (backbone, a), (backbone, b)),
                        "recall": paired_summary(rows, view, ("metrics", "recall"), (backbone, a), (backbone, b)),
                        "p10_sqli_prob": paired_summary(
                            rows, view, ("metrics", "p10_sqli_prob"), (backbone, a), (backbone, b)
                        ),
                    }

    result = {
        "config": vars(args) | {"device_resolved": device, "sqliv5_url": SQLIV5_URL},
        "elapsed_seconds": time.time() - started,
        "data_audit": {
            "train_records": len(train_all_texts),
            "train_benign": train_all_labels.count(0),
            "train_sqli": train_all_labels.count(1),
            "test_records": len(v5_texts),
            "test_benign": v5_labels.count(0),
            "test_sqli": v5_labels.count(1),
            "exact_v3_v5_overlap": len(v3_all & set(v5_texts)),
            "v5_unique_texts": len(set(v5_texts)),
            "v5_new_sqli_vs_v3": int(sum(1 for text, y in zip(v5_texts, v5_labels) if y == 1 and text not in v3_all)),
        },
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
