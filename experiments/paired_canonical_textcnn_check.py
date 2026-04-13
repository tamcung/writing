#!/usr/bin/env python3
"""Token-level TextCNN check for paired canonical-anchor SQLi robustness."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.consistency_sqli_experiment import (  # noqa: E402
    ensure_dataset,
    load_payload_data,
    make_split,
    metrics_from_probs,
    set_seed,
)
from experiments.paired_canonical_family_holdout import weighted_mean  # noqa: E402
from experiments.paired_canonical_semantic_holdout import build_semantic_train_pairs  # noqa: E402
from experiments.semantic_mutation_v2 import (  # noqa: E402
    SEMANTIC_FAMILIES,
    build_semantic_test_family_view,
)


TOKEN_RE = re.compile(
    r"""
    /\*.*?\*/              | # block comments
    --[^\r\n]*             | # line comments
    %[0-9a-fA-F]{2}        | # url-encoded byte
    0x[0-9a-fA-F]+         | # hex literal
    [A-Za-z_][A-Za-z_0-9]* | # identifiers / keywords
    \d+(?:\.\d+)?         | # numeric literals
    <>|!=|<=|>=|==|\|\||&& | # multi-character operators
    \S                       # fallback: any non-space character
    """,
    re.S | re.X,
)


@dataclass
class TextCNNConfig:
    method: str
    seed: int
    epochs: int
    batch_size: int
    max_tokens: int
    lr: float
    consistency_weight: float
    canonical_logit_weight: float
    hard_align_gamma: float
    device: str
    emb_dim: int
    channels: int
    dropout: float
    lowercase: bool


def tokenize_sql(text: str, lowercase: bool) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens or ["<EMPTY>"]


def build_token_vocab(
    texts: list[str],
    lowercase: bool,
    max_vocab: int,
    min_freq: int,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize_sql(text, lowercase))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in counts.most_common(max(0, max_vocab - len(vocab))):
        if count < min_freq:
            continue
        vocab[token] = len(vocab)
    return vocab


def encode_tokens(text: str, vocab: dict[str, int], max_tokens: int, lowercase: bool) -> list[int]:
    ids = [vocab.get(token, 1) for token in tokenize_sql(text, lowercase)[:max_tokens]]
    if len(ids) < max_tokens:
        ids.extend([0] * (max_tokens - len(ids)))
    return ids


class TextPairDataset(Dataset):
    def __init__(
        self,
        canon_texts: list[str],
        mut_texts: list[str],
        labels: list[int],
        vocab: dict[str, int],
        max_tokens: int,
        lowercase: bool,
    ) -> None:
        self.canon_texts = canon_texts
        self.mut_texts = mut_texts
        self.labels = labels
        self.vocab = vocab
        self.max_tokens = max_tokens
        self.lowercase = lowercase

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        canon = torch.tensor(
            encode_tokens(self.canon_texts[idx], self.vocab, self.max_tokens, self.lowercase),
            dtype=torch.long,
        )
        mutated = torch.tensor(
            encode_tokens(self.mut_texts[idx], self.vocab, self.max_tokens, self.lowercase),
            dtype=torch.long,
        )
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return canon, mutated, y


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        channels: int,
        dropout: float,
        projected_classifier: bool,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, channels, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )
        self.hidden_dim = channels * len(self.convs)
        self.dropout = nn.Dropout(dropout)
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.classifier = nn.Linear(self.hidden_dim, 1)
        self.canonical_classifier = nn.Linear(self.hidden_dim, 1)
        self.projected_classifier = projected_classifier

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(emb))
            pooled.append(torch.max(h, dim=-1).values)
        return self.dropout(torch.cat(pooled, dim=1))

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.projector(h)

    def classify(self, h: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        if self.projected_classifier:
            if z is None:
                z = self.project(h)
            return self.canonical_classifier(z).squeeze(1)
        return self.classifier(h).squeeze(1)

    def embed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        z = self.project(h)
        logits = self.classify(h, z)
        rep = z if self.projected_classifier else h
        return logits, rep, h, z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, rep, _, _ = self.embed(x)
        return logits, rep


def train_pair_model(
    cfg: TextCNNConfig,
    canon_texts: list[str],
    mut_texts: list[str],
    labels: list[int],
    vocab: dict[str, int],
) -> TextCNN:
    set_seed(cfg.seed)
    projected_classifier = cfg.method in {"pair_proj_ce", "pair_canonical"}
    model = TextCNN(
        vocab_size=max(vocab.values(), default=1) + 1,
        emb_dim=cfg.emb_dim,
        channels=cfg.channels,
        dropout=cfg.dropout,
        projected_classifier=projected_classifier,
    ).to(cfg.device)
    dataset = TextPairDataset(canon_texts, mut_texts, labels, vocab, cfg.max_tokens, cfg.lowercase)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(cfg.epochs):
        for canon_x, mut_x, y in loader:
            canon_x = canon_x.to(cfg.device)
            mut_x = mut_x.to(cfg.device)
            y = y.to(cfg.device)
            opt.zero_grad(set_to_none=True)

            canon_logits, _, _, canon_z = model.embed(canon_x)
            mut_logits, _, _, mut_z = model.embed(mut_x)
            loss = 0.5 * (bce(canon_logits, y) + bce(mut_logits, y))

            if cfg.method == "pair_canonical":
                mask = y > 0.5
                if mask.any():
                    pair_weights = torch.ones_like(y)
                    if cfg.hard_align_gamma > 0:
                        with torch.no_grad():
                            mut_probs = torch.sigmoid(mut_logits)
                            difficulty = (1.0 - mut_probs).clamp_min(1e-4).pow(cfg.hard_align_gamma)
                        pair_weights = pair_weights * difficulty
                    align_loss = 1.0 - F.cosine_similarity(mut_z[mask], canon_z[mask].detach(), dim=1)
                    loss = loss + cfg.consistency_weight * weighted_mean(align_loss, pair_weights[mask])
                    if cfg.canonical_logit_weight > 0:
                        logit_loss = (mut_logits[mask] - canon_logits[mask].detach()).pow(2)
                        loss = loss + cfg.canonical_logit_weight * weighted_mean(logit_loss, pair_weights[mask])

            loss.backward()
            opt.step()

    return model


@torch.no_grad()
def predict_proba(
    model: TextCNN,
    texts: list[str],
    labels: list[int],
    vocab: dict[str, int],
    max_tokens: int,
    lowercase: bool,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    ys = []
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
def embedding_similarity(
    model: TextCNN,
    base_texts: list[str],
    aug_texts: list[str],
    vocab: dict[str, int],
    max_tokens: int,
    lowercase: bool,
    device: str,
    batch_size: int,
) -> float:
    model.eval()
    sims: list[float] = []
    for start in range(0, len(base_texts), batch_size):
        base = base_texts[start : start + batch_size]
        aug = aug_texts[start : start + batch_size]
        x1 = torch.tensor(
            [encode_tokens(text, vocab, max_tokens, lowercase) for text in base],
            dtype=torch.long,
            device=device,
        )
        x2 = torch.tensor(
            [encode_tokens(text, vocab, max_tokens, lowercase) for text in aug],
            dtype=torch.long,
            device=device,
        )
        _, h1 = model(x1)
        _, h2 = model(x2)
        sims.extend(F.cosine_similarity(h1, h2, dim=1).cpu().numpy().tolist())
    return float(np.mean(sims)) if sims else float("nan")


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
    for methods in by_seed.values():
        if a in methods and b in methods:
            diffs.append(methods[b] - methods[a])

    return {
        "n": float(len(diffs)),
        "mean_diff": float(np.mean(diffs)) if diffs else float("nan"),
        "std_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
        "paired_t_p": float(stats.ttest_1samp(diffs, popmean=0.0).pvalue) if len(diffs) > 1 else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/SQLiV3_clean.json")
    parser.add_argument("--output", default="experiments/paired_canonical_textcnn_check_results.json")
    parser.add_argument("--families", nargs="+", default=["numeric_repr", "string_construction"])
    parser.add_argument("--methods", nargs="+", default=["pair_ce", "pair_proj_ce", "pair_canonical"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-chars", type=int, default=260)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--canonical-logit-weight", type=float, default=0.0)
    parser.add_argument("--hard-align-gamma", type=float, default=0.5)
    parser.add_argument("--pairs-per-sample", type=int, default=1)
    parser.add_argument("--benign-rounds", type=int, default=2)
    parser.add_argument("--benign-retries", type=int, default=4)
    parser.add_argument("--train-rounds", type=int, default=3)
    parser.add_argument("--test-rounds", type=int, default=3)
    parser.add_argument("--train-retries", type=int, default=8)
    parser.add_argument("--test-retries", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


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


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    data_path = ensure_dataset(Path(args.data))
    texts, labels = load_payload_data(data_path, max_len=args.max_chars)

    print(f"Loaded payload-level records: total={len(texts)}, benign={labels.count(0)}, sqli={labels.count(1)}")
    print(f"Using token-level TextCNN device={device} lowercase={args.lowercase}")
    set_seed(1234)
    started = time.time()
    family_results = {}

    for holdout_family in args.families:
        heldout_names = SEMANTIC_FAMILIES[holdout_family]
        train_names = [name for fam, names in SEMANTIC_FAMILIES.items() if fam != holdout_family for name in names]
        print(f"\n=== TextCNN Holdout family: {holdout_family} ===")
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
                benign_pairs="nuisance",
                benign_rounds=args.benign_rounds,
                benign_retries=args.benign_retries,
            )
            heldout_texts, heldout_labels, heldout_base, heldout_aug = build_semantic_test_family_view(
                test_texts=test_texts,
                test_labels=test_labels,
                seed=seed + 20_000,
                strategy_names=heldout_names,
                rounds=args.test_rounds,
                retries=args.test_retries,
            )
            vocab = build_token_vocab(
                train_canon + train_mut,
                lowercase=args.lowercase,
                max_vocab=args.max_vocab,
                min_freq=args.min_freq,
            )
            print(f"seed={seed} vocab_size={len(vocab)}")

            for method in args.methods:
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
                print(f"  Training textcnn/{method} seed={seed}")
                model = train_pair_model(cfg, train_canon, train_mut, pair_labels, vocab)
                clean_probs, clean_y = predict_proba(
                    model, test_texts, test_labels, vocab, args.max_tokens, args.lowercase, device, args.batch_size
                )
                held_probs, held_y = predict_proba(
                    model,
                    heldout_texts,
                    heldout_labels,
                    vocab,
                    args.max_tokens,
                    args.lowercase,
                    device,
                    args.batch_size,
                )
                held_sim = embedding_similarity(
                    model,
                    heldout_base,
                    heldout_aug,
                    vocab,
                    args.max_tokens,
                    args.lowercase,
                    device,
                    args.batch_size,
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
        for a, b in [
            ("pair_ce", "pair_proj_ce"),
            ("pair_ce", "pair_canonical"),
            ("pair_proj_ce", "pair_canonical"),
        ]:
            if a in args.methods and b in args.methods:
                comparisons[f"{b}_minus_{a}"] = {
                    "heldout_family_recall": paired_summary(rows, ("heldout_family", "recall"), a, b),
                    "heldout_family_p10_sqli_prob": paired_summary(
                        rows, ("heldout_family", "p10_sqli_prob"), a, b
                    ),
                    "clean_f1": paired_summary(rows, ("clean", "f1"), a, b),
                    "embedding_cosine_heldout_family": paired_summary(
                        rows, ("embedding_cosine_heldout_family",), a, b
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
