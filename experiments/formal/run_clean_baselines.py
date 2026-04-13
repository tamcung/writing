#!/usr/bin/env python3
"""Unified clean_ce runner on processed formal datasets."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.formal.clean_models import (  # noqa: E402
    BiLSTMModel,
    CodeBERTConfig,
    CodeBERTModel,
    SeqConfig,
    TextCNNModel,
    WordSVCModel,
)
from experiments.formal.data import (  # noqa: E402
    balanced_sample,
    filter_dataset,
    load_dataset_by_name,
    make_stratified_split,
)
from experiments.formal.metrics import metrics_from_probs, summarize  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def resolve_device(device: str) -> str:
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps":
        try:
            _ = torch.zeros(1, device="mps")
        except RuntimeError:
            device = "cpu"
    return device


def build_model(backbone: str, args: argparse.Namespace, device: str):
    if backbone == "word_svc":
        return WordSVCModel(ngram_max=args.word_ngram_max, min_df=args.word_min_df, c=args.word_c)
    if backbone == "textcnn":
        cfg = SeqConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            max_vocab=args.max_vocab,
            min_freq=args.min_freq,
            lowercase=args.lowercase,
            lr=args.lr,
            emb_dim=args.emb_dim,
            channels=args.channels,
            dropout=args.dropout,
            device=device,
        )
        return TextCNNModel(cfg)
    if backbone == "bilstm":
        cfg = SeqConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            max_vocab=args.max_vocab,
            min_freq=args.min_freq,
            lowercase=args.lowercase,
            lr=args.lr,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            device=device,
        )
        return BiLSTMModel(cfg)
    if backbone == "codebert":
        cfg = CodeBERTConfig(
            model_name=args.model_name,
            local_files_only=args.local_files_only,
            freeze_encoder=args.freeze_encoder,
            epochs=args.codebert_epochs,
            batch_size=args.codebert_batch_size,
            max_len=args.max_len,
            lr=args.codebert_lr,
            encoder_lr=args.encoder_lr,
            dropout=args.codebert_dropout,
            device=device,
        )
        return CodeBERTModel(cfg)
    raise ValueError(f"Unsupported backbone: {backbone}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/formal_v3")
    parser.add_argument("--train-dataset", default="sqliv3_clean")
    parser.add_argument("--external-datasets", nargs="+", default=["modsec_learn_cleaned", "web_attacks_long_test"])
    parser.add_argument("--backbones", nargs="+", default=["word_svc", "textcnn", "bilstm", "codebert"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--external-per-class", type=int, default=0, help="0 means all-balanced")
    parser.add_argument("--output", default="experiments/formal/results_clean_baselines.json")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")

    parser.add_argument("--word-ngram-max", type=int, default=2)
    parser.add_argument("--word-min-df", type=int, default=1)
    parser.add_argument("--word-c", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
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
    parser.add_argument("--max-len", type=int, default=320)
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
    processed_dir = Path(args.processed_dir)
    device = resolve_device(args.device)
    started = time.time()

    train_bundle = load_dataset_by_name(processed_dir, args.train_dataset)
    full_train_source_texts = set(train_bundle.texts)
    external_bundles = {name: load_dataset_by_name(processed_dir, name) for name in args.external_datasets}

    rows: list[dict] = []

    for seed in args.seeds:
        set_seed(seed)
        train_texts, train_labels, holdout_texts, holdout_labels = make_stratified_split(
            train_bundle,
            seed=seed,
            train_per_class=args.train_per_class,
            test_per_class=args.test_per_class,
        )
        forbidden_texts = full_train_source_texts | set(train_texts)

        eval_views = {
            f"{args.train_dataset}_holdout": (holdout_texts, holdout_labels),
        }
        for name, bundle in external_bundles.items():
            filtered = filter_dataset(bundle, forbidden_texts)
            balanced = balanced_sample(filtered, seed + 31337, args.external_per_class)
            eval_views[name] = (balanced.texts, balanced.labels)

        for backbone in args.backbones:
            print(f"seed={seed} training clean {backbone}")
            model = build_model(backbone, args, device)
            model.fit(train_texts, train_labels)
            for view, (texts, labels) in eval_views.items():
                probs = model.predict_proba(texts)
                metric = metrics_from_probs(probs, labels)
                rows.append(
                    {
                        "seed": seed,
                        "backbone": backbone,
                        "method": "clean_ce",
                        "view": view,
                        "metrics": metric,
                    }
                )
                print(
                    f"  {view} f1={metric['f1']:.4f} recall={metric['recall']:.4f} p10={metric['p10_sqli_prob']:.4f}"
                )

    summary: dict[str, dict] = {}
    views = sorted({row["view"] for row in rows})
    for view in views:
        summary[view] = {}
        for backbone in args.backbones:
            vals = [row["metrics"] for row in rows if row["view"] == view and row["backbone"] == backbone]
            summary[view][backbone] = {
                "f1": summarize([v["f1"] for v in vals]),
                "recall": summarize([v["recall"] for v in vals]),
                "precision": summarize([v["precision"] for v in vals]),
                "p10_sqli_prob": summarize([v["p10_sqli_prob"] for v in vals]),
            }

    output = {
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "summary": summary,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
