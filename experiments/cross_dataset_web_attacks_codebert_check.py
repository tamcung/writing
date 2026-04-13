#!/usr/bin/env python3
"""CodeBERT external generalization check on web-attacks-long."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.clean_backbone_semantic_holdout import (  # noqa: E402
    predict_codebert,
    train_codebert,
)
from experiments.consistency_sqli_experiment import (  # noqa: E402
    load_payload_data,
    make_split,
    metrics_from_probs,
    set_seed,
)
from experiments.cross_dataset_http_params_check import (  # noqa: E402
    balanced_sample_all,
    ensure_web_attacks_dataset,
    load_web_attacks_sqli_norm,
)
from experiments.cross_dataset_sqliv5_check import resolve_device  # noqa: E402
from experiments.paired_canonical_codebert_check import (  # noqa: E402
    CodeBERTConfig,
    predict_proba as predict_pair_codebert,
    train_pair_model,
)
from experiments.paired_canonical_semantic_holdout import build_semantic_train_pairs  # noqa: E402
from experiments.semantic_mutation_v2 import SEMANTIC_FAMILIES  # noqa: E402


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def paired_summary(rows: list[dict], view: str, metric: str, a: str, b: str) -> dict[str, float]:
    by_seed: dict[int, dict[str, float]] = {}
    for row in rows:
        if row["view"] != view:
            continue
        by_seed.setdefault(row["seed"], {})[row["method"]] = float(row["metrics"][metric])
    diffs = [methods[b] - methods[a] for methods in by_seed.values() if a in methods and b in methods]
    return {
        "n": float(len(diffs)),
        "mean_diff": float(np.mean(diffs)) if diffs else float("nan"),
        "std_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/raw/SQLiV3_clean.json")
    parser.add_argument("--web-attacks-dir", default="data/raw/web_attacks_long")
    parser.add_argument("--web-attacks-file", default="test.csv")
    parser.add_argument("--output", default="experiments/cross_dataset_web_attacks_long_codebert_results.json")
    parser.add_argument("--methods", nargs="+", default=["clean_ce", "pair_ce", "pair_proj_ce", "pair_canonical"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--train-per-class", type=int, default=120)
    parser.add_argument("--test-per-class", type=int, default=300)
    parser.add_argument("--external-per-class", type=int, default=300, help="0 means use all available balanced classes.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
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
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local-files-only", dest="local_files_only", action="store_true", default=True)
    group.add_argument("--allow-download", dest="local_files_only", action="store_false")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    set_seed(1234)
    started = time.time()

    web_attacks_dir = Path(args.web_attacks_dir)
    ensure_web_attacks_dataset(web_attacks_dir)
    external_path = web_attacks_dir / args.web_attacks_file
    train_texts_all, train_labels_all = load_payload_data(Path(args.train_data), max_len=args.max_len)
    external_texts_all, external_labels_all, external_audit = load_web_attacks_sqli_norm(
        external_path, args.max_len
    )
    v3_all = set(train_texts_all)
    exact_overlap_all = len(v3_all & set(external_texts_all))
    external_no_v3 = [(text, y) for text, y in zip(external_texts_all, external_labels_all) if text not in v3_all]
    external_texts_no_v3 = [text for text, _ in external_no_v3]
    external_labels_no_v3 = [y for _, y in external_no_v3]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    strategy_names = [name for names in SEMANTIC_FAMILIES.values() for name in names]
    rows: list[dict] = []

    print(
        "Loaded SQLiV3 total={} benign={} sqli={} | web-attacks-long {} total={} benign={} sqli={} overlap_with_v3={}".format(
            len(train_texts_all),
            train_labels_all.count(0),
            train_labels_all.count(1),
            args.web_attacks_file,
            len(external_texts_all),
            external_labels_all.count(0),
            external_labels_all.count(1),
            exact_overlap_all,
        )
    )
    print(f"Using CodeBERT model={args.model_name} device={device} freeze_encoder={args.freeze_encoder}")

    for seed in args.seeds:
        train_texts, train_labels, v3_test_texts, v3_test_labels = make_split(
            train_texts_all,
            train_labels_all,
            seed=seed,
            train_per_class=args.train_per_class,
            test_per_class=args.test_per_class,
        )
        train_set = set(train_texts)
        external_seed_pool = [(text, y) for text, y in external_no_v3 if text not in train_set]
        external_seed_texts = [text for text, _ in external_seed_pool]
        external_seed_labels = [y for _, y in external_seed_pool]
        external_bal_texts, external_bal_labels = balanced_sample_all(
            external_seed_texts,
            external_seed_labels,
            seed + 31_337,
            args.external_per_class,
        )
        eval_views = {
            "v3_holdout": (v3_test_texts, v3_test_labels),
            "web_attacks_long_sqli_normal_balanced": (external_bal_texts, external_bal_labels),
        }
        print(
            "seed={} train={} external_pool={} external_balanced={} external_sqli={} external_benign={}".format(
                seed,
                len(train_texts),
                len(external_seed_texts),
                len(external_bal_texts),
                external_bal_labels.count(1),
                external_bal_labels.count(0),
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

        for method in args.methods:
            print(f"  Training codebert/{method} seed={seed}")
            if method == "clean_ce":
                model = train_codebert(
                    train_texts,
                    train_labels,
                    tokenizer,
                    seed,
                    args.epochs,
                    args.batch_size,
                    args.max_len,
                    args.lr,
                    args.encoder_lr,
                    args.model_name,
                    args.local_files_only,
                    args.freeze_encoder,
                    args.dropout,
                    args.grad_clip,
                    device,
                )
                predict = lambda texts, labels: predict_codebert(
                    model, texts, labels, tokenizer, args.max_len, device, args.batch_size
                )
            else:
                cfg = CodeBERTConfig(
                    method=method,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_len=args.max_len,
                    lr=args.lr,
                    encoder_lr=args.encoder_lr,
                    consistency_weight=args.consistency_weight,
                    canonical_logit_weight=args.canonical_logit_weight,
                    hard_align_gamma=args.hard_align_gamma if method == "pair_canonical" else 0.0,
                    device=device,
                    model_name=args.model_name,
                    local_files_only=args.local_files_only,
                    freeze_encoder=args.freeze_encoder,
                    dropout=args.dropout,
                    grad_clip=args.grad_clip,
                )
                model = train_pair_model(cfg, pair_canon, pair_mut, pair_labels, tokenizer)
                predict = lambda texts, labels: predict_pair_codebert(
                    model, texts, labels, tokenizer, args.max_len, device, args.batch_size
                )

            for view, (texts, labels) in eval_views.items():
                probs, y = predict(texts, labels)
                row = {
                    "seed": seed,
                    "backbone": "codebert",
                    "method": method,
                    "view": view,
                    "metrics": metrics_from_probs(probs, y),
                }
                rows.append(row)
                print(
                    "    {} f1={:.4f} recall={:.4f} p10={:.4f} mean_sqli_prob={:.4f}".format(
                        view,
                        row["metrics"]["f1"],
                        row["metrics"]["recall"],
                        row["metrics"]["p10_sqli_prob"],
                        row["metrics"]["mean_sqli_prob"],
                    )
                )

    summary: dict[str, dict] = {}
    for view in sorted({row["view"] for row in rows}):
        summary[view] = {"codebert": {}}
        for method in args.methods:
            method_rows = [row for row in rows if row["view"] == view and row["method"] == method]
            summary[view]["codebert"][method] = {
                "f1": summarize([row["metrics"]["f1"] for row in method_rows]),
                "recall": summarize([row["metrics"]["recall"] for row in method_rows]),
                "p10_sqli_prob": summarize([row["metrics"]["p10_sqli_prob"] for row in method_rows]),
                "mean_sqli_prob": summarize([row["metrics"]["mean_sqli_prob"] for row in method_rows]),
            }

    comparisons: dict[str, dict] = {}
    for view in sorted({row["view"] for row in rows}):
        comparisons[view] = {}
        for a, b in [
            ("clean_ce", "pair_ce"),
            ("pair_ce", "pair_proj_ce"),
            ("pair_proj_ce", "pair_canonical"),
            ("pair_ce", "pair_canonical"),
        ]:
            if a in args.methods and b in args.methods:
                comparisons[view][f"codebert_{b}_minus_{a}"] = {
                    "f1": paired_summary(rows, view, "f1", a, b),
                    "recall": paired_summary(rows, view, "recall", a, b),
                    "p10_sqli_prob": paired_summary(rows, view, "p10_sqli_prob", a, b),
                }

    result = {
        "config": vars(args)
        | {
            "device_resolved": device,
            "external_source": "https://huggingface.co/datasets/shengqin/web-attacks-long",
        },
        "elapsed_seconds": time.time() - started,
        "data_audit": {
            "train_records": len(train_texts_all),
            "train_benign": train_labels_all.count(0),
            "train_sqli": train_labels_all.count(1),
            "external_file": args.web_attacks_file,
            "external": external_audit,
            "external_exact_overlap_with_sqliv3": exact_overlap_all,
            "external_no_v3_overlap_records": len(external_texts_no_v3),
            "external_no_v3_overlap_benign": external_labels_no_v3.count(0),
            "external_no_v3_overlap_sqli": external_labels_no_v3.count(1),
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
