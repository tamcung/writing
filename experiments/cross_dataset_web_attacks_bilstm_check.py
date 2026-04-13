#!/usr/bin/env python3
"""BiLSTM external generalization check on web-attacks-long."""

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
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from experiments.paired_canonical_semantic_holdout import build_semantic_train_pairs  # noqa: E402
from experiments.semantic_mutation_v2 import SEMANTIC_FAMILIES  # noqa: E402


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
class BiLSTMConfig:
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
    hidden_dim: int
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


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum().clamp_min(1e-8)
    return (values * weights).sum() / denom


class BiLSTMCleanDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: dict[str, int],
        max_tokens: int,
        lowercase: bool,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_tokens = max_tokens
        self.lowercase = lowercase

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(
            encode_tokens(self.texts[idx], self.vocab, self.max_tokens, self.lowercase),
            dtype=torch.long,
        )
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class BiLSTMPairDataset(Dataset):
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


class BiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        projected_classifier: bool,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            emb_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.hidden_dim = hidden_dim * 2
        self.dropout = nn.Dropout(dropout)
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.classifier = nn.Linear(self.hidden_dim, 1)
        self.canonical_classifier = nn.Linear(self.hidden_dim, 1)
        self.projected_classifier = projected_classifier

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        lengths = x.ne(0).sum(dim=1).clamp_min(1)
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.encoder(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.dropout(h)

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


def train_clean_model(
    cfg: BiLSTMConfig,
    texts: list[str],
    labels: list[int],
    vocab: dict[str, int],
) -> BiLSTM:
    set_seed(cfg.seed)
    model = BiLSTM(
        vocab_size=max(vocab.values(), default=1) + 1,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        projected_classifier=False,
    ).to(cfg.device)
    dataset = BiLSTMCleanDataset(texts, labels, vocab, cfg.max_tokens, cfg.lowercase)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(cfg.epochs):
        for x, y in loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = bce(logits, y)
            loss.backward()
            opt.step()
    return model


def train_pair_model(
    cfg: BiLSTMConfig,
    canon_texts: list[str],
    mut_texts: list[str],
    labels: list[int],
    vocab: dict[str, int],
) -> BiLSTM:
    set_seed(cfg.seed)
    model = BiLSTM(
        vocab_size=max(vocab.values(), default=1) + 1,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        projected_classifier=cfg.method in {"pair_proj_ce", "pair_canonical"},
    ).to(cfg.device)
    dataset = BiLSTMPairDataset(canon_texts, mut_texts, labels, vocab, cfg.max_tokens, cfg.lowercase)
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
    model: BiLSTM,
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
    parser.add_argument("--output", default="experiments/cross_dataset_web_attacks_bilstm_results.json")
    parser.add_argument("--methods", nargs="+", default=["clean_ce", "pair_ce", "pair_proj_ce", "pair_canonical"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--train-per-class", type=int, default=120)
    parser.add_argument("--test-per-class", type=int, default=300)
    parser.add_argument("--external-per-class", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
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

    web_attacks_dir = Path(args.web_attacks_dir)
    ensure_web_attacks_dataset(web_attacks_dir)
    external_path = web_attacks_dir / args.web_attacks_file
    train_texts_all, train_labels_all = load_payload_data(Path(args.train_data), max_len=260)
    external_texts_all, external_labels_all, external_audit = load_web_attacks_sqli_norm(external_path, 260)
    v3_all = set(train_texts_all)
    exact_overlap_all = len(v3_all & set(external_texts_all))
    external_no_v3 = [(text, y) for text, y in zip(external_texts_all, external_labels_all) if text not in v3_all]
    external_texts_no_v3 = [text for text, _ in external_no_v3]
    external_labels_no_v3 = [y for _, y in external_no_v3]
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
    print(f"Using BiLSTM device={device} emb_dim={args.emb_dim} hidden_dim={args.hidden_dim}")

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
            print(f"  Training bilstm/{method} seed={seed}")
            if method == "clean_ce":
                vocab = build_token_vocab(
                    train_texts,
                    lowercase=args.lowercase,
                    max_vocab=args.max_vocab,
                    min_freq=args.min_freq,
                )
                cfg = BiLSTMConfig(
                    method=method,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_tokens=args.max_tokens,
                    lr=args.lr,
                    consistency_weight=args.consistency_weight,
                    canonical_logit_weight=args.canonical_logit_weight,
                    hard_align_gamma=0.0,
                    device=device,
                    emb_dim=args.emb_dim,
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                    lowercase=args.lowercase,
                )
                model = train_clean_model(cfg, train_texts, train_labels, vocab)
            else:
                vocab = build_token_vocab(
                    pair_canon + pair_mut,
                    lowercase=args.lowercase,
                    max_vocab=args.max_vocab,
                    min_freq=args.min_freq,
                )
                cfg = BiLSTMConfig(
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
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                    lowercase=args.lowercase,
                )
                model = train_pair_model(cfg, pair_canon, pair_mut, pair_labels, vocab)

            for view, (texts, labels) in eval_views.items():
                probs, y = predict_proba(
                    model,
                    texts,
                    labels,
                    vocab,
                    args.max_tokens,
                    args.lowercase,
                    device,
                    args.batch_size,
                )
                row = {
                    "seed": seed,
                    "backbone": "bilstm",
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
        summary[view] = {"bilstm": {}}
        for method in args.methods:
            method_rows = [row for row in rows if row["view"] == view and row["method"] == method]
            summary[view]["bilstm"][method] = {
                "f1": summarize([row["metrics"]["f1"] for row in method_rows]),
                "recall": summarize([row["metrics"]["recall"] for row in method_rows]),
                "p10_sqli_prob": summarize([row["metrics"]["p10_sqli_prob"] for row in method_rows]),
                "mean_sqli_prob": summarize([row["metrics"]["mean_sqli_prob"] for row in method_rows]),
                "precision": summarize([row["metrics"]["precision"] for row in method_rows]),
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
                comparisons[view][f"bilstm_{b}_minus_{a}"] = {
                    "f1": paired_summary(rows, view, "f1", a, b),
                    "recall": paired_summary(rows, view, "recall", a, b),
                    "p10_sqli_prob": paired_summary(rows, view, "p10_sqli_prob", a, b),
                }

    result = {
        "config": vars(args) | {"device_resolved": device},
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
