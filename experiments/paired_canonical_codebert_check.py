#!/usr/bin/env python3
"""CodeBERT backbone check for paired canonical-anchor SQLi robustness."""

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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

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


@dataclass
class CodeBERTConfig:
    method: str
    seed: int
    epochs: int
    batch_size: int
    max_len: int
    lr: float
    encoder_lr: float
    consistency_weight: float
    canonical_logit_weight: float
    hard_align_gamma: float
    device: str
    model_name: str
    local_files_only: bool
    freeze_encoder: bool
    dropout: float
    grad_clip: float


class CodeBERTPairDataset(Dataset):
    def __init__(
        self,
        canon_texts: list[str],
        mut_texts: list[str],
        labels: list[int],
        tokenizer,
        max_len: int,
    ) -> None:
        self.canon = tokenizer(
            canon_texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.mut = tokenizer(
            mut_texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.canon["input_ids"][idx],
            self.canon["attention_mask"][idx],
            self.mut["input_ids"][idx],
            self.mut["attention_mask"][idx],
            self.labels[idx],
        )


class CodeBERTPairModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        local_files_only: bool,
        projected_classifier: bool,
        freeze_encoder: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden_dim = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        self.canonical_classifier = nn.Linear(hidden_dim, 1)
        self.projected_classifier = projected_classifier

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # CodeBERT is RoBERTa-like; the first token is the sequence representation.
        return self.dropout(outputs.last_hidden_state[:, 0])

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.projector(h)

    def classify(self, h: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        if self.projected_classifier:
            if z is None:
                z = self.project(h)
            return self.canonical_classifier(z).squeeze(1)
        return self.classifier(h).squeeze(1)

    def embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(input_ids, attention_mask)
        z = self.project(h)
        logits = self.classify(h, z)
        rep = z if self.projected_classifier else h
        return logits, rep, h, z

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, rep, _, _ = self.embed(input_ids, attention_mask)
        return logits, rep


def build_optimizer(model: CodeBERTPairModel, cfg: CodeBERTConfig) -> torch.optim.Optimizer:
    if cfg.freeze_encoder:
        trainable = [param for param in model.parameters() if param.requires_grad]
        return torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=1e-4)

    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.encoder_lr},
            {"params": head_params, "lr": cfg.lr},
        ],
        weight_decay=1e-4,
    )


def train_pair_model(
    cfg: CodeBERTConfig,
    canon_texts: list[str],
    mut_texts: list[str],
    labels: list[int],
    tokenizer,
) -> CodeBERTPairModel:
    set_seed(cfg.seed)
    projected_classifier = cfg.method in {"pair_proj_ce", "pair_canonical"}
    model = CodeBERTPairModel(
        model_name=cfg.model_name,
        local_files_only=cfg.local_files_only,
        projected_classifier=projected_classifier,
        freeze_encoder=cfg.freeze_encoder,
        dropout=cfg.dropout,
    ).to(cfg.device)
    dataset = CodeBERTPairDataset(canon_texts, mut_texts, labels, tokenizer, cfg.max_len)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    opt = build_optimizer(model, cfg)
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        for canon_ids, canon_mask, mut_ids, mut_mask, y in loader:
            canon_ids = canon_ids.to(cfg.device)
            canon_mask = canon_mask.to(cfg.device)
            mut_ids = mut_ids.to(cfg.device)
            mut_mask = mut_mask.to(cfg.device)
            y = y.to(cfg.device)
            opt.zero_grad(set_to_none=True)

            canon_logits, _, _, canon_z = model.embed(canon_ids, canon_mask)
            mut_logits, _, _, mut_z = model.embed(mut_ids, mut_mask)
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
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total_loss += float(loss.detach().cpu())
        print(f"    epoch={epoch + 1}/{cfg.epochs} loss={total_loss / max(1, len(loader)):.4f}")

    return model


@torch.no_grad()
def predict_proba(
    model: CodeBERTPairModel,
    texts: list[str],
    labels: list[int],
    tokenizer,
    max_len: int,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    ys = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        logits, _ = model(input_ids, attention_mask)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        ys.extend(labels[start : start + batch_size])
    return np.asarray(probs, dtype=float), np.asarray(ys, dtype=int)


@torch.no_grad()
def embedding_similarity(
    model: CodeBERTPairModel,
    base_texts: list[str],
    aug_texts: list[str],
    tokenizer,
    max_len: int,
    device: str,
    batch_size: int,
) -> float:
    model.eval()
    sims: list[float] = []
    for start in range(0, len(base_texts), batch_size):
        base = base_texts[start : start + batch_size]
        aug = aug_texts[start : start + batch_size]
        base_enc = tokenizer(base, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        aug_enc = tokenizer(aug, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        _, h1 = model(base_enc["input_ids"].to(device), base_enc["attention_mask"].to(device))
        _, h2 = model(aug_enc["input_ids"].to(device), aug_enc["attention_mask"].to(device))
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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/SQLiV3_clean.json")
    parser.add_argument("--output", default="experiments/paired_canonical_codebert_check_results.json")
    parser.add_argument("--families", nargs="+", default=["numeric_repr", "string_construction"])
    parser.add_argument("--methods", nargs="+", default=["pair_proj_ce", "pair_canonical"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
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
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local-files-only", dest="local_files_only", action="store_true", default=True)
    group.add_argument("--allow-download", dest="local_files_only", action="store_false")
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
    texts, labels = load_payload_data(data_path, max_len=args.max_len)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)

    print(f"Loaded payload-level records: total={len(texts)}, benign={labels.count(0)}, sqli={labels.count(1)}")
    print(f"Using model={args.model_name} device={device} freeze_encoder={args.freeze_encoder}")
    set_seed(1234)
    started = time.time()
    family_results = {}

    for holdout_family in args.families:
        heldout_names = SEMANTIC_FAMILIES[holdout_family]
        train_names = [name for fam, names in SEMANTIC_FAMILIES.items() if fam != holdout_family for name in names]
        print(f"\n=== CodeBERT Holdout family: {holdout_family} ===")
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

            for method in args.methods:
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
                print(f"  Training codebert/{method} seed={seed}")
                model = train_pair_model(cfg, train_canon, train_mut, pair_labels, tokenizer)
                clean_probs, clean_y = predict_proba(
                    model, test_texts, test_labels, tokenizer, args.max_len, device, args.batch_size
                )
                held_probs, held_y = predict_proba(
                    model, heldout_texts, heldout_labels, tokenizer, args.max_len, device, args.batch_size
                )
                held_sim = embedding_similarity(
                    model, heldout_base, heldout_aug, tokenizer, args.max_len, device, args.batch_size
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
        if "pair_proj_ce" in args.methods and "pair_canonical" in args.methods:
            comparisons["pair_canonical_minus_pair_proj_ce"] = {
                "heldout_family_recall": paired_summary(
                    rows, ("heldout_family", "recall"), "pair_proj_ce", "pair_canonical"
                ),
                "heldout_family_p10_sqli_prob": paired_summary(
                    rows, ("heldout_family", "p10_sqli_prob"), "pair_proj_ce", "pair_canonical"
                ),
                "clean_f1": paired_summary(rows, ("clean", "f1"), "pair_proj_ce", "pair_canonical"),
                "embedding_cosine_heldout_family": paired_summary(
                    rows, ("embedding_cosine_heldout_family",), "pair_proj_ce", "pair_canonical"
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
