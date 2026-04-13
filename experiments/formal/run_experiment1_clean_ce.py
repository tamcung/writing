#!/usr/bin/env python3
"""Run experiment-1 clean_ce baselines on fixed clean/mutated views."""

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


DEFAULT_VIEW_PAIRS = [
    ("clean_test_full", None),
    ("clean_test_mixed_matched", "mutated_test_mixed"),
    ("clean_test_mixed_hard_matched", "mutated_test_mixed_hard"),
    ("clean_test_surface_obfuscation_matched", "mutated_test_surface_obfuscation"),
    ("clean_test_numeric_repr_matched", "mutated_test_numeric_repr"),
    ("clean_test_string_construction_matched", "mutated_test_string_construction"),
]


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


def load_seed_views(views_dir: Path, seed: int) -> dict[str, list[dict]]:
    manifest = load_rows(views_dir / f"seed_{seed}" / "manifest.json")
    views: dict[str, list[dict]] = {}
    for name, abs_path in manifest["views"].items():
        views[name] = load_rows(Path(abs_path))
    return views


def summarize_rows(rows: list[dict]) -> dict[str, int]:
    return {
        "total": len(rows),
        "benign": sum(1 for row in rows if int(row["label"]) == 0),
        "sqli": sum(1 for row in rows if int(row["label"]) == 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/derived/formal_v3/experiment1/splits")
    parser.add_argument("--views-dir", default="data/derived/formal_v3/experiment1/views")
    parser.add_argument("--backbones", nargs="+", default=["word_svc", "textcnn", "bilstm", "codebert"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    parser.add_argument("--output", default="experiments/formal/results_experiment1_clean_ce.json")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")

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
    splits_dir = Path(args.splits_dir)
    views_dir = Path(args.views_dir)
    device = resolve_device(args.device)
    started = time.time()

    rows: list[dict] = []

    for seed in args.seeds:
        set_seed(seed)
        train_rows = load_seed_split(splits_dir, seed, "train")
        valid_rows = load_seed_split(splits_dir, seed, "valid")
        train_texts, train_labels = rows_to_xy(train_rows)
        valid_texts, valid_labels = rows_to_xy(valid_rows)
        seed_views = load_seed_views(views_dir, seed)

        for backbone in args.backbones:
            print(f"seed={seed} training {backbone}")
            model = build_model(backbone, args, device)
            model.fit(train_texts, train_labels)

            valid_probs = model.predict_proba(valid_texts)
            valid_metric = metrics_from_probs(valid_probs, valid_labels)
            rows.append(
                {
                    "seed": seed,
                    "backbone": backbone,
                    "method": "clean_ce",
                    "view": "valid",
                    "view_kind": "validation",
                    "metrics": valid_metric,
                    "stats": summarize_rows(valid_rows),
                }
            )

            for clean_view, mutated_view in DEFAULT_VIEW_PAIRS:
                clean_rows = seed_views[clean_view]
                clean_texts, clean_labels = rows_to_xy(clean_rows)
                clean_probs = model.predict_proba(clean_texts)
                clean_metric = metrics_from_probs(clean_probs, clean_labels)
                rows.append(
                    {
                        "seed": seed,
                        "backbone": backbone,
                        "method": "clean_ce",
                        "view": clean_view,
                        "view_kind": "clean",
                        "metrics": clean_metric,
                        "stats": summarize_rows(clean_rows),
                    }
                )
                print(
                    f"  {clean_view} f1={clean_metric['f1']:.4f} "
                    f"recall={clean_metric['recall']:.4f} p10={clean_metric['p10_sqli_prob']:.4f}"
                )

                if mutated_view is None:
                    continue

                mutated_rows = seed_views[mutated_view]
                mutated_texts, mutated_labels = rows_to_xy(mutated_rows)
                mutated_probs = model.predict_proba(mutated_texts)
                mutated_metric = metrics_from_probs(mutated_probs, mutated_labels)
                rows.append(
                    {
                        "seed": seed,
                        "backbone": backbone,
                        "method": "clean_ce",
                        "view": mutated_view,
                        "view_kind": "mutated",
                        "metrics": mutated_metric,
                        "stats": summarize_rows(mutated_rows),
                    }
                )
                print(
                    f"  {mutated_view} f1={mutated_metric['f1']:.4f} "
                    f"recall={mutated_metric['recall']:.4f} p10={mutated_metric['p10_sqli_prob']:.4f}"
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

    delta_pairs = [
        ("mixed", "clean_test_mixed_matched", "mutated_test_mixed"),
        ("mixed_hard", "clean_test_mixed_hard_matched", "mutated_test_mixed_hard"),
        ("surface_obfuscation", "clean_test_surface_obfuscation_matched", "mutated_test_surface_obfuscation"),
        ("numeric_repr", "clean_test_numeric_repr_matched", "mutated_test_numeric_repr"),
        ("string_construction", "clean_test_string_construction_matched", "mutated_test_string_construction"),
    ]
    delta_summary: dict[str, dict] = {}
    for tag, clean_view, mutated_view in delta_pairs:
        delta_summary[tag] = {}
        for backbone in args.backbones:
            clean_vals = [row["metrics"] for row in rows if row["view"] == clean_view and row["backbone"] == backbone]
            mutated_vals = [row["metrics"] for row in rows if row["view"] == mutated_view and row["backbone"] == backbone]
            diffs = {
                "f1": [m["f1"] - c["f1"] for c, m in zip(clean_vals, mutated_vals)],
                "recall": [m["recall"] - c["recall"] for c, m in zip(clean_vals, mutated_vals)],
                "precision": [m["precision"] - c["precision"] for c, m in zip(clean_vals, mutated_vals)],
                "p10_sqli_prob": [m["p10_sqli_prob"] - c["p10_sqli_prob"] for c, m in zip(clean_vals, mutated_vals)],
            }
            delta_summary[tag][backbone] = {metric: summarize(values) for metric, values in diffs.items()}

    output = {
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "summary": summary,
        "delta_summary": delta_summary,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
