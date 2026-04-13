#!/usr/bin/env python3
"""Check whether a stronger sequence backbone improves canonical-anchor gains."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.consistency_sqli_experiment import (  # noqa: E402
    CharCNN,
    build_vocab,
    ensure_dataset,
    load_payload_data,
    make_split,
    metrics_from_probs,
    set_seed,
)
from experiments.paired_canonical_family_holdout import (  # noqa: E402
    PairDataset,
    encode,
    weighted_mean,
)
from experiments.paired_canonical_semantic_holdout import (  # noqa: E402
    build_semantic_train_pairs,
)
from experiments.semantic_mutation_v2 import (  # noqa: E402
    SEMANTIC_FAMILIES,
    build_semantic_test_family_view,
)


@dataclass
class BackboneConfig:
    backbone: str
    method: str
    seed: int
    epochs: int
    batch_size: int
    max_len: int
    lr: float
    consistency_weight: float
    canonical_logit_weight: float
    hard_align_gamma: float
    device: str


class CharTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15,
        projected_classifier: bool = False,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.hidden_dim = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder_layers = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.classifier = nn.Linear(d_model, 1)
        self.canonical_classifier = nn.Linear(d_model, 1)
        self.projected_classifier = projected_classifier

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
        h = self.embedding(x) + self.position(pos)
        pad_mask = x.eq(0)
        h = self.encoder_layers(h, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        valid = (~pad_mask).unsqueeze(-1)
        summed = (h * valid).sum(dim=1)
        denom = valid.sum(dim=1).clamp_min(1)
        pooled = summed / denom
        return self.dropout(pooled)

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


def build_model(backbone: str, vocab_size: int, max_len: int, projected_classifier: bool) -> nn.Module:
    if backbone == "cnn":
        return CharCNN(vocab_size=vocab_size, projected_classifier=projected_classifier)
    if backbone == "transformer":
        return CharTransformer(vocab_size=vocab_size, max_len=max_len, projected_classifier=projected_classifier)
    raise ValueError(backbone)


def train_pair_model(cfg: BackboneConfig, canon_texts: list[str], mut_texts: list[str], labels: list[int], vocab: dict[str, int]) -> nn.Module:
    set_seed(cfg.seed)
    projected_classifier = cfg.method in {"pair_proj_ce", "pair_canonical"}
    model = build_model(
        backbone=cfg.backbone,
        vocab_size=max(vocab.values(), default=1) + 1,
        max_len=cfg.max_len,
        projected_classifier=projected_classifier,
    ).to(cfg.device)
    dataset = PairDataset(canon_texts, mut_texts, labels, vocab, cfg.max_len)
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
                pair_weights = torch.ones_like(y)
                if mask.any():
                    if cfg.hard_align_gamma > 0:
                        with torch.no_grad():
                            mut_probs = torch.sigmoid(mut_logits)
                            difficulty = (1.0 - mut_probs).clamp_min(1e-4).pow(cfg.hard_align_gamma)
                        pair_weights = pair_weights * difficulty
                    sim = torch.cosine_similarity(mut_z[mask], canon_z[mask].detach(), dim=1)
                    align_loss = 1.0 - sim
                    loss = loss + cfg.consistency_weight * weighted_mean(align_loss, pair_weights[mask])
                    if cfg.canonical_logit_weight > 0:
                        logit_loss = (mut_logits[mask] - canon_logits[mask].detach()).pow(2)
                        loss = loss + cfg.canonical_logit_weight * weighted_mean(logit_loss, pair_weights[mask])

            loss.backward()
            opt.step()

    return model


@torch.no_grad()
def predict_proba(model: nn.Module, texts: list[str], labels: list[int], vocab: dict[str, int], max_len: int, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    ys = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        x = torch.tensor([encode(t, vocab, max_len) for t in batch], dtype=torch.long, device=device)
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


def paired_summary(rows: list[dict], metric_path: tuple[str, ...], a: tuple[str, str], b: tuple[str, str]) -> dict[str, float]:
    by_seed = {}
    for row in rows:
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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/SQLiV3_clean.json")
    parser.add_argument("--output", default="experiments/paired_canonical_backbone_check_results.json")
    parser.add_argument("--families", nargs="+", default=["numeric_repr", "string_construction"])
    parser.add_argument("--backbones", nargs="+", default=["cnn", "transformer"])
    parser.add_argument("--methods", nargs="+", default=["pair_proj_ce", "pair_canonical"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-3)
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


def run(args: argparse.Namespace) -> dict:
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
    started = time.time()
    family_results = {}

    for holdout_family in args.families:
        heldout_names = SEMANTIC_FAMILIES[holdout_family]
        train_names = [name for fam, names in SEMANTIC_FAMILIES.items() if fam != holdout_family for name in names]
        print(f"\n=== Backbone Check family: {holdout_family} ===")
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
            vocab = build_vocab(train_canon + train_mut)

            for backbone in args.backbones:
                for method in args.methods:
                    cfg = BackboneConfig(
                        backbone=backbone,
                        method=method,
                        seed=seed,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        max_len=args.max_len,
                        lr=args.lr,
                        consistency_weight=args.consistency_weight,
                        canonical_logit_weight=args.canonical_logit_weight,
                        hard_align_gamma=args.hard_align_gamma if method == "pair_canonical" else 0.0,
                        device=device,
                    )
                    print(f"  Training {backbone}/{method} seed={seed}")
                    model = train_pair_model(cfg, train_canon, train_mut, pair_labels, vocab)
                    clean_probs, clean_y = predict_proba(
                        model, test_texts, test_labels, vocab, args.max_len, device, args.batch_size
                    )
                    held_probs, held_y = predict_proba(
                        model, heldout_texts, heldout_labels, vocab, args.max_len, device, args.batch_size
                    )
                    row = {
                        "seed": seed,
                        "backbone": backbone,
                        "method": method,
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
            summary[backbone] = {}
            for method in args.methods:
                method_rows = [row for row in rows if row["backbone"] == backbone and row["method"] == method]
                summary[backbone][method] = {
                    "clean_f1": summarize([row["clean"]["f1"] for row in method_rows]),
                    "heldout_family_recall": summarize([row["heldout_family"]["recall"] for row in method_rows]),
                    "heldout_family_p10_sqli_prob": summarize(
                        [row["heldout_family"]["p10_sqli_prob"] for row in method_rows]
                    ),
                }

        comparisons = {}
        for backbone in args.backbones:
            comparisons[f"{backbone}_pair_canonical_minus_pair_proj_ce"] = {
                "heldout_family_recall": paired_summary(
                    rows, ("heldout_family", "recall"), (backbone, "pair_proj_ce"), (backbone, "pair_canonical")
                ),
                "heldout_family_p10_sqli_prob": paired_summary(
                    rows,
                    ("heldout_family", "p10_sqli_prob"),
                    (backbone, "pair_proj_ce"),
                    (backbone, "pair_canonical"),
                ),
                "clean_f1": paired_summary(
                    rows, ("clean", "f1"), (backbone, "pair_proj_ce"), (backbone, "pair_canonical")
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
