#!/usr/bin/env python3
"""External generalization check: train on SQLiV3_clean, test on external SQLi datasets."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.consistency_sqli_experiment import load_payload_data, make_split, set_seed  # noqa: E402
from experiments.cross_dataset_sqliv5_check import (  # noqa: E402
    balanced_sample,
    paired_summary,
    resolve_device,
    summarize,
    train_and_eval_text,
)
from experiments.paired_canonical_semantic_holdout import build_semantic_train_pairs  # noqa: E402
from experiments.semantic_mutation_v2 import SEMANTIC_FAMILIES  # noqa: E402


HTTP_PARAMS_BASE_URL = "https://raw.githubusercontent.com/Morzeux/HttpParamsDataset/master/"
HTTP_PARAMS_FILES = ["README.md", "payload_full.csv", "payload_train.csv", "payload_test.csv", "payload_test_lexical.csv"]
WEB_ATTACKS_BASE_URL = "https://huggingface.co/datasets/shengqin/web-attacks-long/resolve/main/"
WEB_ATTACKS_FILES = ["train.csv", "validate.csv", "test.csv", "all_1000.csv", "train_2000.csv", "test_1000.csv"]


def ensure_http_params_dataset(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in HTTP_PARAMS_FILES:
        path = data_dir / filename
        if not path.exists():
            url = HTTP_PARAMS_BASE_URL + filename
            print(f"Downloading {url} -> {path}")
            urlretrieve(url, path)


def ensure_web_attacks_dataset(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in WEB_ATTACKS_FILES:
        path = data_dir / filename
        if not path.exists():
            url = WEB_ATTACKS_BASE_URL + filename
            print(f"Downloading {url} -> {path}")
            urlretrieve(url, path)


def load_http_params_sqli_norm(path: Path, max_len: int) -> tuple[list[str], list[int], dict[str, int]]:
    df = pd.read_csv(path, keep_default_na=False)
    df = df[df["attack_type"].isin(["norm", "sqli"])].copy()
    df["payload"] = df["payload"].astype(str).str.strip()
    df = df[(df["payload"] != "") & (df["payload"].str.len() <= max_len)]
    labels = (df["attack_type"] == "sqli").astype(int).tolist()
    texts = df["payload"].tolist()
    audit = {
        "records_after_filter": int(len(df)),
        "benign": int(sum(1 for y in labels if y == 0)),
        "sqli": int(sum(1 for y in labels if y == 1)),
        "unique_texts": int(len(set(texts))),
    }
    return texts, labels, audit


def load_web_attacks_sqli_norm(path: Path, max_len: int) -> tuple[list[str], list[int], dict[str, int]]:
    df = pd.read_csv(path, keep_default_na=False)
    df = df[df["text_label"].isin(["normal", "SQLi"])].copy()
    df["Payload"] = df["Payload"].astype(str).str.strip()
    df = df[(df["Payload"] != "") & (df["Payload"].str.len() <= max_len)]
    labels = (df["text_label"] == "SQLi").astype(int).tolist()
    texts = df["Payload"].tolist()
    audit = {
        "records_after_filter": int(len(df)),
        "benign": int(sum(1 for y in labels if y == 0)),
        "sqli": int(sum(1 for y in labels if y == 1)),
        "unique_texts": int(len(set(texts))),
    }
    return texts, labels, audit


def balanced_sample_all(
    texts: list[str],
    labels: list[int],
    seed: int,
    per_class: int,
) -> tuple[list[str], list[int]]:
    if per_class > 0:
        return balanced_sample(texts, labels, seed, per_class)

    idx0 = [i for i, y in enumerate(labels) if y == 0]
    idx1 = [i for i, y in enumerate(labels) if y == 1]
    rng = random.Random(seed)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n = min(len(idx0), len(idx1))
    idx = idx0[:n] + idx1[:n]
    rng.shuffle(idx)
    return [texts[i] for i in idx], [labels[i] for i in idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/raw/SQLiV3_clean.json")
    parser.add_argument("--external-kind", choices=["http_params", "web_attacks_long"], default="http_params")
    parser.add_argument("--http-params-dir", default="data/raw/http_params_dataset")
    parser.add_argument("--http-params-file", default="payload_test.csv")
    parser.add_argument("--web-attacks-dir", default="data/raw/web_attacks_long")
    parser.add_argument("--web-attacks-file", default="test.csv")
    parser.add_argument("--output", default="experiments/cross_dataset_http_params_textcnn_results.json")
    parser.add_argument("--methods", nargs="+", default=["clean_ce", "pair_ce", "pair_proj_ce", "pair_canonical"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--external-per-class", type=int, default=0, help="0 means use all available balanced classes.")
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
    set_seed(1234)
    started = time.time()

    train_texts_all, train_labels_all = load_payload_data(Path(args.train_data), max_len=args.max_chars)
    if args.external_kind == "http_params":
        external_dir = Path(args.http_params_dir)
        ensure_http_params_dataset(external_dir)
        external_path = external_dir / args.http_params_file
        external_texts_all, external_labels_all, external_audit = load_http_params_sqli_norm(
            external_path, args.max_chars
        )
        external_view_name = "http_params_sqli_norm_balanced"
        external_source = "https://github.com/Morzeux/HttpParamsDataset"
        external_file = args.http_params_file
    elif args.external_kind == "web_attacks_long":
        external_dir = Path(args.web_attacks_dir)
        ensure_web_attacks_dataset(external_dir)
        external_path = external_dir / args.web_attacks_file
        external_texts_all, external_labels_all, external_audit = load_web_attacks_sqli_norm(
            external_path, args.max_chars
        )
        external_view_name = "web_attacks_long_sqli_normal_balanced"
        external_source = "https://huggingface.co/datasets/shengqin/web-attacks-long"
        external_file = args.web_attacks_file
    else:
        raise ValueError(f"Unsupported external kind: {args.external_kind}")

    v3_all = set(train_texts_all)
    exact_overlap_all = len(v3_all & set(external_texts_all))
    external_no_v3 = [(text, y) for text, y in zip(external_texts_all, external_labels_all) if text not in v3_all]
    external_texts_no_v3 = [text for text, _ in external_no_v3]
    external_labels_no_v3 = [y for _, y in external_no_v3]

    print(
        "Loaded SQLiV3 train source total={} benign={} sqli={} | {} {} total={} benign={} sqli={} overlap_with_v3={}".format(
            len(train_texts_all),
            train_labels_all.count(0),
            train_labels_all.count(1),
            args.external_kind,
            external_file,
            len(external_texts_all),
            external_labels_all.count(0),
            external_labels_all.count(1),
            exact_overlap_all,
        )
    )
    print(f"Using TextCNN device={device}; external_per_class={args.external_per_class or 'all-balanced'}")

    strategy_names = [name for names in SEMANTIC_FAMILIES.values() for name in names]
    rows: list[dict] = []

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
            external_view_name: (external_bal_texts, external_bal_labels),
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
            print(f"  Training textcnn/{method} seed={seed}")
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
            rows.extend(method_rows)
            for row in method_rows:
                print(
                    "    {} f1={:.4f} recall={:.4f} p10={:.4f} mean_sqli_prob={:.4f}".format(
                        row["view"],
                        row["metrics"]["f1"],
                        row["metrics"]["recall"],
                        row["metrics"]["p10_sqli_prob"],
                        row["metrics"]["mean_sqli_prob"],
                    )
                )

    summary: dict[str, dict] = {}
    for view in sorted({row["view"] for row in rows}):
        summary[view] = {"textcnn": {}}
        for method in args.methods:
            method_rows = [row for row in rows if row["view"] == view and row["method"] == method]
            summary[view]["textcnn"][method] = {
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
                comparisons[view][f"textcnn_{b}_minus_{a}"] = {
                    "f1": paired_summary(rows, view, ("metrics", "f1"), ("textcnn", a), ("textcnn", b)),
                    "recall": paired_summary(rows, view, ("metrics", "recall"), ("textcnn", a), ("textcnn", b)),
                    "p10_sqli_prob": paired_summary(
                        rows, view, ("metrics", "p10_sqli_prob"), ("textcnn", a), ("textcnn", b)
                    ),
                }

    result = {
        "config": vars(args)
        | {
            "device_resolved": device,
            "http_params_base_url": HTTP_PARAMS_BASE_URL,
            "web_attacks_base_url": WEB_ATTACKS_BASE_URL,
            "external_source": external_source,
        },
        "elapsed_seconds": time.time() - started,
        "data_audit": {
            "train_records": len(train_texts_all),
            "train_benign": train_labels_all.count(0),
            "train_sqli": train_labels_all.count(1),
            "external_kind": args.external_kind,
            "external_file": external_file,
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
