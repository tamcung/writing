#!/usr/bin/env python3
"""Clean baseline training helpers and standalone clean-eval runner for Experiment 1."""

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
from experiments.formal.metrics import metrics_from_probs, summarize  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
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


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_to_xy(rows: list[dict]) -> tuple[list[str], list[int]]:
    return [str(row["text"]) for row in rows], [int(row["label"]) for row in rows]


def load_seed_split(splits_dir: Path, seed: int, split_name: str) -> list[dict]:
    return load_rows(splits_dir / f"seed_{seed}" / f"{split_name}.json")


def summarize_rows(rows: list[dict]) -> dict[str, int]:
    return {
        "total": len(rows),
        "benign": sum(1 for row in rows if int(row["label"]) == 0),
        "sqli": sum(1 for row in rows if int(row["label"]) == 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/derived/formal_modsec_decoded/experiment1/splits")
    parser.add_argument("--backbones", nargs="+", default=["word_svc", "textcnn", "bilstm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--output", default="experiments/formal/results_experiment1_clean_ce.json")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")

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
    rows: list[dict] = []

    for seed in args.seeds:
        set_seed(seed)
        train_rows = load_seed_split(splits_dir, seed, "train")
        valid_rows = load_seed_split(splits_dir, seed, "valid")
        clean_test_rows = load_seed_split(splits_dir, seed, "clean_test")
        train_texts, train_labels = rows_to_xy(train_rows)

        for backbone in args.backbones:
            print(f"seed={seed} training {backbone}")
            model = build_model(backbone, args, device)
            model.fit(train_texts, train_labels)

            for split_name, split_rows in [("valid", valid_rows), ("clean_test", clean_test_rows)]:
                texts, labels = rows_to_xy(split_rows)
                probs = model.predict_proba(texts)
                metrics = metrics_from_probs(probs, labels)
                rows.append({
                    "seed": seed,
                    "backbone": backbone,
                    "split": split_name,
                    "metrics": metrics,
                })
                print(f"  {split_name} f1={metrics['f1']:.4f} recall={metrics['recall']:.4f}")

    summary: dict[str, dict] = {}
    for split_name in ["valid", "clean_test"]:
        summary[split_name] = {}
        for backbone in args.backbones:
            vals = [r["metrics"] for r in rows if r["split"] == split_name and r["backbone"] == backbone]
            summary[split_name][backbone] = {
                "f1": summarize([v["f1"] for v in vals]),
                "recall": summarize([v["recall"] for v in vals]),
            }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "elapsed_seconds": round(time.time() - started, 1),
        "rows": rows,
        "summary": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
