#!/usr/bin/env python3
"""Clean-only training robustness check across backbones on semantic holdout mutations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.consistency_sqli_experiment import (  # noqa: E402
    CharCNN,
    build_vocab,
    encode,
    ensure_dataset,
    load_payload_data,
    make_split,
    metrics_from_probs,
    set_seed,
)
from experiments.paired_canonical_codebert_check import CodeBERTPairModel  # noqa: E402
from experiments.paired_canonical_textcnn_check import (  # noqa: E402
    TextCNN,
    build_token_vocab,
    encode_tokens,
)
from experiments.semantic_mutation_v2 import (  # noqa: E402
    SEMANTIC_FAMILIES,
    build_semantic_test_family_view,
)


class CodeBERTCleanDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int) -> None:
        self.encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoded["input_ids"][idx], self.encoded["attention_mask"][idx], self.labels[idx]


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


def train_charcnn(
    train_texts: list[str],
    train_labels: list[int],
    seed: int,
    epochs: int,
    batch_size: int,
    max_chars: int,
    lr: float,
    device: str,
) -> tuple[nn.Module, dict[str, int]]:
    set_seed(seed)
    vocab = build_vocab(train_texts)
    x = torch.tensor([encode(text, vocab, max_chars) for text in train_texts], dtype=torch.long)
    y = torch.tensor(train_labels, dtype=torch.float32)
    dataset = TensorDataset(x, y)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = CharCNN(vocab_size=max(vocab.values(), default=1) + 1, projected_classifier=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
    return model, vocab


def train_textcnn(
    train_texts: list[str],
    train_labels: list[int],
    seed: int,
    epochs: int,
    batch_size: int,
    max_tokens: int,
    max_vocab: int,
    min_freq: int,
    lowercase: bool,
    lr: float,
    emb_dim: int,
    channels: int,
    dropout: float,
    device: str,
) -> tuple[nn.Module, dict[str, int]]:
    set_seed(seed)
    vocab = build_token_vocab(train_texts, lowercase=lowercase, max_vocab=max_vocab, min_freq=min_freq)
    x = torch.tensor(
        [encode_tokens(text, vocab, max_tokens, lowercase) for text in train_texts],
        dtype=torch.long,
    )
    y = torch.tensor(train_labels, dtype=torch.float32)
    dataset = TensorDataset(x, y)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = TextCNN(
        vocab_size=max(vocab.values(), default=1) + 1,
        emb_dim=emb_dim,
        channels=channels,
        dropout=dropout,
        projected_classifier=False,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
    return model, vocab


def train_codebert(
    train_texts: list[str],
    train_labels: list[int],
    tokenizer,
    seed: int,
    epochs: int,
    batch_size: int,
    max_len: int,
    lr: float,
    encoder_lr: float,
    model_name: str,
    local_files_only: bool,
    freeze_encoder: bool,
    dropout: float,
    grad_clip: float,
    device: str,
) -> nn.Module:
    set_seed(seed)
    model = CodeBERTPairModel(
        model_name=model_name,
        local_files_only=local_files_only,
        projected_classifier=False,
        freeze_encoder=freeze_encoder,
        dropout=dropout,
    ).to(device)
    dataset = CodeBERTCleanDataset(train_texts, train_labels, tokenizer, max_len)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    if freeze_encoder:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    else:
        encoder_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder."):
                encoder_params.append(param)
            else:
                head_params.append(param)
        opt = torch.optim.AdamW(
            [{"params": encoder_params, "lr": encoder_lr}, {"params": head_params, "lr": lr}],
            weight_decay=1e-4,
        )
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for input_ids, attention_mask, y in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(input_ids, attention_mask)
            loss = bce(logits, y)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
    return model


@torch.no_grad()
def predict_charcnn(
    model: nn.Module,
    texts: list[str],
    labels: list[int],
    vocab: dict[str, int],
    max_chars: int,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        x = torch.tensor([encode(text, vocab, max_chars) for text in batch], dtype=torch.long, device=device)
        logits, _ = model(x)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        ys.extend(labels[start : start + batch_size])
    return np.asarray(probs, dtype=float), np.asarray(ys, dtype=int)


@torch.no_grad()
def predict_textcnn(
    model: nn.Module,
    texts: list[str],
    labels: list[int],
    vocab: dict[str, int],
    max_tokens: int,
    lowercase: bool,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        x = torch.tensor(
            [encode_tokens(text, vocab, max_tokens, lowercase) for text in batch],
            dtype=torch.long,
            device=device,
        )
        logits, _ = model(x)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        ys.extend(labels[start : start + batch_size])
    return np.asarray(probs, dtype=float), np.asarray(ys, dtype=int)


@torch.no_grad()
def predict_codebert(
    model: nn.Module,
    texts: list[str],
    labels: list[int],
    tokenizer,
    max_len: int,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        logits, _ = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        ys.extend(labels[start : start + batch_size])
    return np.asarray(probs, dtype=float), np.asarray(ys, dtype=int)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/SQLiV3_clean.json")
    parser.add_argument("--output", default="experiments/clean_backbone_semantic_holdout_results.json")
    parser.add_argument("--families", nargs="+", default=["numeric_repr", "string_construction"])
    parser.add_argument("--backbones", nargs="+", default=["charcnn", "textcnn"])
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
    parser.add_argument("--test-rounds", type=int, default=3)
    parser.add_argument("--test-retries", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--freeze-codebert", action="store_true")
    parser.add_argument("--codebert-dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local-files-only", dest="local_files_only", action="store_true", default=True)
    group.add_argument("--allow-download", dest="local_files_only", action="store_false")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    data_path = ensure_dataset(Path(args.data))
    texts, labels = load_payload_data(data_path, max_len=args.max_chars)
    tokenizer = None
    if "codebert" in args.backbones:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)

    print(f"Loaded payload-level records: total={len(texts)}, benign={labels.count(0)}, sqli={labels.count(1)}")
    print(f"Clean-only backbones={args.backbones} device={device}")
    started = time.time()
    family_results = {}

    for holdout_family in args.families:
        heldout_names = SEMANTIC_FAMILIES[holdout_family]
        print(f"\n=== Clean-only Holdout family: {holdout_family} ===")
        rows = []

        for seed in args.seeds:
            train_texts, train_labels, test_texts, test_labels = make_split(
                texts,
                labels,
                seed=seed,
                train_per_class=args.train_per_class,
                test_per_class=args.test_per_class,
            )
            heldout_texts, heldout_labels, _, _ = build_semantic_test_family_view(
                test_texts=test_texts,
                test_labels=test_labels,
                seed=seed + 20_000,
                strategy_names=heldout_names,
                rounds=args.test_rounds,
                retries=args.test_retries,
            )

            for backbone in args.backbones:
                print(f"  Training clean-only {backbone} seed={seed}")
                if backbone == "charcnn":
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
                    clean_probs, clean_y = predict_charcnn(
                        model, test_texts, test_labels, vocab, args.max_chars, device, args.batch_size
                    )
                    held_probs, held_y = predict_charcnn(
                        model, heldout_texts, heldout_labels, vocab, args.max_chars, device, args.batch_size
                    )
                elif backbone == "textcnn":
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
                    clean_probs, clean_y = predict_textcnn(
                        model,
                        test_texts,
                        test_labels,
                        vocab,
                        args.max_tokens,
                        args.lowercase,
                        device,
                        args.batch_size,
                    )
                    held_probs, held_y = predict_textcnn(
                        model,
                        heldout_texts,
                        heldout_labels,
                        vocab,
                        args.max_tokens,
                        args.lowercase,
                        device,
                        args.batch_size,
                    )
                elif backbone == "codebert":
                    if tokenizer is None:
                        raise RuntimeError("CodeBERT tokenizer was not initialized.")
                    model = train_codebert(
                        train_texts,
                        train_labels,
                        tokenizer,
                        seed,
                        args.epochs,
                        args.batch_size,
                        args.max_tokens,
                        args.lr,
                        args.encoder_lr,
                        args.model_name,
                        args.local_files_only,
                        args.freeze_codebert,
                        args.codebert_dropout,
                        args.grad_clip,
                        device,
                    )
                    clean_probs, clean_y = predict_codebert(
                        model, test_texts, test_labels, tokenizer, args.max_tokens, device, args.batch_size
                    )
                    held_probs, held_y = predict_codebert(
                        model, heldout_texts, heldout_labels, tokenizer, args.max_tokens, device, args.batch_size
                    )
                else:
                    raise ValueError(f"Unknown backbone: {backbone}")

                row = {
                    "seed": seed,
                    "backbone": backbone,
                    "holdout_family": holdout_family,
                    "clean": metrics_from_probs(clean_probs, clean_y),
                    "heldout_family": metrics_from_probs(held_probs, held_y),
                }
                rows.append(row)
                print(
                    "    clean_f1={:.4f} heldout_recall={:.4f} heldout_p10={:.4f}".format(
                        row["clean"]["f1"],
                        row["heldout_family"]["recall"],
                        row["heldout_family"]["p10_sqli_prob"],
                    )
                )

        summary = {}
        for backbone in args.backbones:
            backbone_rows = [row for row in rows if row["backbone"] == backbone]
            summary[backbone] = {
                "clean_f1": summarize([row["clean"]["f1"] for row in backbone_rows]),
                "heldout_family_recall": summarize([row["heldout_family"]["recall"] for row in backbone_rows]),
                "heldout_family_p10_sqli_prob": summarize(
                    [row["heldout_family"]["p10_sqli_prob"] for row in backbone_rows]
                ),
            }

        family_results[holdout_family] = {
            "rows": rows,
            "summary": summary,
            "heldout_strategies": heldout_names,
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
