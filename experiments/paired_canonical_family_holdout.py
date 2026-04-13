#!/usr/bin/env python3
"""Explicit paired-data canonical-anchor experiment with unseen mutation-family holdout."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.consistency_family_holdout import (  # noqa: E402
    ALL_FAMILIES,
    apply_strategy_rounds,
    build_test_family_view,
)
from experiments.consistency_sqli_experiment import (  # noqa: E402
    CharCNN,
    build_vocab,
    embedding_similarity,
    ensure_dataset,
    load_payload_data,
    make_split,
    metrics_from_probs,
    predict_proba,
    score_texts,
    set_seed,
)


@dataclass
class PairTrainConfig:
    method: str
    seed: int
    epochs: int
    batch_size: int
    max_len: int
    lr: float
    consistency_weight: float
    canonical_logit_weight: float
    device: str
    align_benign_pairs: bool = False
    benign_consistency_weight: float = 1.0
    hard_align_gamma: float = 0.0


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(ch, 1) for ch in text[:max_len]]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids


class PairDataset(Dataset):
    def __init__(
        self,
        canon_texts: list[str],
        mut_texts: list[str],
        labels: list[int],
        vocab: dict[str, int],
        max_len: int,
    ) -> None:
        self.canon_texts = canon_texts
        self.mut_texts = mut_texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        canon = torch.tensor(encode(self.canon_texts[idx], self.vocab, self.max_len), dtype=torch.long)
        mutated = torch.tensor(encode(self.mut_texts[idx], self.vocab, self.max_len), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return canon, mutated, y


def benign_random_case(text: str, rng: random.Random) -> str:
    chars = list(text)
    changed = False
    for i, ch in enumerate(chars):
        if ch.isalpha() and rng.random() < 0.35:
            chars[i] = ch.upper() if rng.random() < 0.5 else ch.lower()
            changed = changed or chars[i] != ch
    return "".join(chars) if changed else text


def benign_url_encode(text: str, rng: random.Random) -> str:
    chars = list(text)
    eligible = [i for i, ch in enumerate(chars) if not ch.isalnum() and ch not in {"%", "\n", "\r"}]
    if not eligible:
        eligible = [i for i, ch in enumerate(chars) if ch.isalnum()]
    if not eligible:
        return text
    chosen = set(rng.sample(eligible, k=min(len(eligible), max(1, rng.randint(1, 3)))))
    out = []
    changed = False
    for i, ch in enumerate(chars):
        if i in chosen:
            enc = quote(ch, safe="")
            out.append(enc)
            changed = changed or enc != ch
        else:
            out.append(ch)
    return "".join(out) if changed else text


def benign_space_variant(text: str, rng: random.Random) -> str:
    if " " not in text:
        return text
    replacement = rng.choice(["%20", "+"])
    return text.replace(" ", replacement)


def benign_nuisance_mutation(text: str, seed: int, rounds: int, retries: int) -> str:
    rng = random.Random(seed)
    ops = [benign_random_case, benign_url_encode, benign_space_variant]
    last = text
    for attempt in range(max(1, retries)):
        out = text
        local_rng = random.Random(seed + attempt * 104_729)
        for _ in range(max(1, rounds)):
            op = local_rng.choice(ops)
            out = op(out, local_rng)
        last = out
        if out != text:
            return out[:260]
    return last[:260]


def build_train_pairs(
    train_texts: list[str],
    train_labels: list[int],
    seed: int,
    strategy_names: list[str],
    pairs_per_sample: int,
    rounds: int,
    retries: int,
    wafamole_repo: str,
    benign_pairs: str,
    benign_rounds: int,
    benign_retries: int,
) -> tuple[list[str], list[str], list[int]]:
    canon_texts: list[str] = []
    mut_texts: list[str] = []
    labels: list[int] = []

    for i, (text, y) in enumerate(zip(train_texts, train_labels)):
        repeats = pairs_per_sample if y == 1 else 1
        for p in range(repeats):
            canon_texts.append(text)
            labels.append(y)
            if y == 1:
                mut_texts.append(
                    apply_strategy_rounds(
                        text=text,
                        strategy_names=strategy_names,
                        seed=seed * 1_000_003 + i * 10_007 + p * 1_009,
                        rounds=rounds,
                        ensure_changed=True,
                        retries=retries,
                        wafamole_repo=wafamole_repo,
                    )
                )
            else:
                if benign_pairs == "nuisance":
                    mut_texts.append(
                        benign_nuisance_mutation(
                            text,
                            seed=seed * 1_000_003 + i * 10_007 + p * 1_009,
                            rounds=benign_rounds,
                            retries=benign_retries,
                        )
                    )
                else:
                    mut_texts.append(text)
    return canon_texts, mut_texts, labels


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    if weights is None:
        return values.mean()
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def build_malicious_candidate_pool(
    text: str,
    seed: int,
    strategy_names: list[str],
    num_candidates: int,
    rounds: int,
    retries: int,
    wafamole_repo: str,
) -> list[str]:
    candidates: list[str] = []
    seen = {text}
    attempts = max(num_candidates * 3, num_candidates)
    for offset in range(attempts):
        mutated = apply_strategy_rounds(
            text=text,
            strategy_names=strategy_names,
            seed=seed + offset * 1_009,
            rounds=rounds,
            ensure_changed=True,
            retries=retries,
            wafamole_repo=wafamole_repo,
        )
        if mutated not in seen:
            seen.add(mutated)
            candidates.append(mutated)
        if len(candidates) >= num_candidates:
            break
    if not candidates:
        candidates.append(text)
    return candidates


def build_benign_candidate_pool(
    text: str,
    seed: int,
    num_candidates: int,
    rounds: int,
    retries: int,
) -> list[str]:
    candidates: list[str] = []
    seen = {text}
    attempts = max(num_candidates * 3, num_candidates)
    for offset in range(attempts):
        mutated = benign_nuisance_mutation(
            text,
            seed=seed + offset * 1_009,
            rounds=rounds,
            retries=retries,
        )
        if mutated not in seen:
            seen.add(mutated)
            candidates.append(mutated)
        if len(candidates) >= num_candidates:
            break
    if not candidates:
        candidates.append(text)
    return candidates


def build_hard_mined_pairs(
    train_texts: list[str],
    train_labels: list[int],
    seed: int,
    strategy_names: list[str],
    pairs_per_sample: int,
    rounds: int,
    retries: int,
    wafamole_repo: str,
    benign_pairs: str,
    benign_rounds: int,
    benign_retries: int,
    warmup_method: str,
    warmup_epochs: int,
    mine_candidates: int,
    mine_benign_candidates: int,
    mine_rounds: int,
    device: str,
    batch_size: int,
    max_len: int,
    lr: float,
    consistency_weight: float,
    canonical_logit_weight: float,
    align_benign_pairs: bool,
    benign_consistency_weight: float,
) -> tuple[list[str], list[str], list[int], dict[str, float]]:
    warmup_canon, warmup_mut, warmup_labels = build_train_pairs(
        train_texts=train_texts,
        train_labels=train_labels,
        seed=seed,
        strategy_names=strategy_names,
        pairs_per_sample=pairs_per_sample,
        rounds=rounds,
        retries=retries,
        wafamole_repo=wafamole_repo,
        benign_pairs=benign_pairs,
        benign_rounds=benign_rounds,
        benign_retries=benign_retries,
    )
    warmup_vocab = build_vocab(warmup_canon + warmup_mut)
    warmup_cfg = PairTrainConfig(
        method=warmup_method,
        seed=seed,
        epochs=warmup_epochs,
        batch_size=batch_size,
        max_len=max_len,
        lr=lr,
        consistency_weight=consistency_weight,
        canonical_logit_weight=canonical_logit_weight,
        device=device,
        align_benign_pairs=align_benign_pairs,
        benign_consistency_weight=benign_consistency_weight,
    )
    warmup_model = train_pair_model(warmup_cfg, warmup_canon, warmup_mut, warmup_labels, warmup_vocab)

    canon_texts: list[str] = []
    mut_texts: list[str] = []
    labels: list[int] = []
    malicious_scores: list[float] = []
    benign_scores: list[float] = []

    for i, (text, y) in enumerate(zip(train_texts, train_labels)):
        repeats = pairs_per_sample if y == 1 else 1
        for p in range(repeats):
            canon_texts.append(text)
            labels.append(y)
            sample_seed = seed * 1_000_003 + i * 10_007 + p * 1_009
            if y == 1:
                candidates = build_malicious_candidate_pool(
                    text=text,
                    seed=sample_seed,
                    strategy_names=strategy_names,
                    num_candidates=max(1, mine_candidates),
                    rounds=mine_rounds,
                    retries=retries,
                    wafamole_repo=wafamole_repo,
                )
                probs = score_texts(
                    warmup_model,
                    candidates,
                    warmup_vocab,
                    max_len,
                    device,
                    min(batch_size, max(1, len(candidates))),
                )
                best_idx = int(np.argmin(probs))
                mut_texts.append(candidates[best_idx])
                malicious_scores.append(float(probs[best_idx]))
            else:
                if benign_pairs == "nuisance":
                    candidates = build_benign_candidate_pool(
                        text=text,
                        seed=sample_seed,
                        num_candidates=max(1, mine_benign_candidates),
                        rounds=benign_rounds,
                        retries=benign_retries,
                    )
                    probs = score_texts(
                        warmup_model,
                        candidates,
                        warmup_vocab,
                        max_len,
                        device,
                        min(batch_size, max(1, len(candidates))),
                    )
                    best_idx = int(np.argmax(probs))
                    mut_texts.append(candidates[best_idx])
                    benign_scores.append(float(probs[best_idx]))
                else:
                    mut_texts.append(text)

    mining_stats = {
        "mean_selected_malicious_prob": float(np.mean(malicious_scores)) if malicious_scores else math.nan,
        "mean_selected_benign_prob": float(np.mean(benign_scores)) if benign_scores else math.nan,
        "num_malicious_pairs_scored": float(len(malicious_scores)),
        "num_benign_pairs_scored": float(len(benign_scores)),
    }
    return canon_texts, mut_texts, labels, mining_stats


def summarize_pairs(canon_texts: list[str], mut_texts: list[str], labels: list[int]) -> dict[str, float]:
    total_sqli = 0
    changed_pairs = 0
    total_benign = 0
    changed_benign = 0
    for canon, mutated, y in zip(canon_texts, mut_texts, labels):
        if y == 1:
            total_sqli += 1
            if canon != mutated:
                changed_pairs += 1
        else:
            total_benign += 1
            if canon != mutated:
                changed_benign += 1
    return {
        "total_pairs": len(labels),
        "total_sqli_pairs": total_sqli,
        "changed_sqli_pairs": changed_pairs,
        "changed_ratio": changed_pairs / total_sqli if total_sqli else 0.0,
        "total_benign_pairs": total_benign,
        "changed_benign_pairs": changed_benign,
        "changed_benign_ratio": changed_benign / total_benign if total_benign else 0.0,
    }


def train_pair_model(
    cfg: PairTrainConfig,
    canon_texts: list[str],
    mut_texts: list[str],
    labels: list[int],
    vocab: dict[str, int],
) -> CharCNN:
    set_seed(cfg.seed)
    projected_classifier = cfg.method in {"pair_proj_ce", "pair_canonical"}
    model = CharCNN(
        vocab_size=max(vocab.values(), default=1) + 1,
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
                if cfg.align_benign_pairs:
                    mask = torch.ones_like(y, dtype=torch.bool)
                    pair_weights = torch.where(
                        y > 0.5,
                        torch.ones_like(y),
                        torch.full_like(y, float(cfg.benign_consistency_weight)),
                    )
                else:
                    mask = y > 0.5
                    pair_weights = torch.ones_like(y)
                if mask.any():
                    if cfg.hard_align_gamma > 0:
                        with torch.no_grad():
                            mut_probs = torch.sigmoid(mut_logits)
                            difficulty = torch.where(
                                y > 0.5,
                                (1.0 - mut_probs).clamp_min(1e-4).pow(cfg.hard_align_gamma),
                                mut_probs.clamp_min(1e-4).pow(cfg.hard_align_gamma),
                            )
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
    for _, methods in by_seed.items():
        if a in methods and b in methods:
            diffs.append(methods[b] - methods[a])

    result = {
        "n": float(len(diffs)),
        "mean_diff": float(np.mean(diffs)) if diffs else math.nan,
        "std_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
    }
    if len(diffs) >= 3 and np.std(diffs, ddof=1) > 0:
        result["paired_t_p"] = float(stats.ttest_1samp(diffs, 0.0).pvalue)
    else:
        result["paired_t_p"] = math.nan
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/SQLiV3_clean.json")
    parser.add_argument("--output", default="experiments/paired_canonical_family_holdout_results.json")
    parser.add_argument("--wafamole-repo", default="external/WAF-A-MoLE")
    parser.add_argument("--families", nargs="+", default=["numeric"])
    parser.add_argument("--methods", nargs="+", default=["pair_ce", "pair_proj_ce", "pair_canonical"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--canonical-logit-weight", type=float, default=0.0)
    parser.add_argument("--pairs-per-sample", type=int, default=1)
    parser.add_argument("--pair-builder", choices=["random", "hard_mined"], default="random")
    parser.add_argument("--mine-warmup-method", choices=["pair_ce", "pair_proj_ce", "pair_canonical"], default="pair_proj_ce")
    parser.add_argument("--mine-warmup-epochs", type=int, default=4)
    parser.add_argument("--mine-candidates", type=int, default=8)
    parser.add_argument("--mine-benign-candidates", type=int, default=6)
    parser.add_argument("--mine-rounds", type=int, default=4)
    parser.add_argument("--benign-pairs", choices=["identity", "nuisance"], default="identity")
    parser.add_argument("--benign-rounds", type=int, default=2)
    parser.add_argument("--benign-retries", type=int, default=4)
    parser.add_argument("--align-benign-pairs", action="store_true")
    parser.add_argument("--benign-consistency-weight", type=float, default=1.0)
    parser.add_argument("--hard-align-gamma", type=float, default=0.0)
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

    family_results = {}
    started = time.time()

    for holdout_family in args.families:
        heldout_names = ALL_FAMILIES[holdout_family]
        train_names = [name for fam, names in ALL_FAMILIES.items() if fam != holdout_family for name in names]
        print(f"\n=== Paired Holdout family: {holdout_family} ===")
        rows = []

        for seed in args.seeds:
            train_texts, train_labels, test_texts, test_labels = make_split(
                texts,
                labels,
                seed=seed,
                train_per_class=args.train_per_class,
                test_per_class=args.test_per_class,
            )
            if args.pair_builder == "hard_mined":
                train_canon, train_mut, pair_labels, mining_stats = build_hard_mined_pairs(
                    train_texts=train_texts,
                    train_labels=train_labels,
                    seed=seed,
                    strategy_names=train_names,
                    pairs_per_sample=args.pairs_per_sample,
                    rounds=args.train_rounds,
                    retries=args.train_retries,
                    wafamole_repo=args.wafamole_repo,
                    benign_pairs=args.benign_pairs,
                    benign_rounds=args.benign_rounds,
                    benign_retries=args.benign_retries,
                    warmup_method=args.mine_warmup_method,
                    warmup_epochs=args.mine_warmup_epochs,
                    mine_candidates=args.mine_candidates,
                    mine_benign_candidates=args.mine_benign_candidates,
                    mine_rounds=args.mine_rounds,
                    device=device,
                    batch_size=args.batch_size,
                    max_len=args.max_len,
                    lr=args.lr,
                    consistency_weight=args.consistency_weight,
                    canonical_logit_weight=args.canonical_logit_weight,
                    align_benign_pairs=args.align_benign_pairs,
                    benign_consistency_weight=args.benign_consistency_weight,
                )
            else:
                train_canon, train_mut, pair_labels = build_train_pairs(
                    train_texts=train_texts,
                    train_labels=train_labels,
                    seed=seed,
                    strategy_names=train_names,
                    pairs_per_sample=args.pairs_per_sample,
                    rounds=args.train_rounds,
                    retries=args.train_retries,
                    wafamole_repo=args.wafamole_repo,
                    benign_pairs=args.benign_pairs,
                    benign_rounds=args.benign_rounds,
                    benign_retries=args.benign_retries,
                )
                mining_stats = None
            pair_stats = summarize_pairs(train_canon, train_mut, pair_labels)
            print(
                "seed={} paired train changed {}/{} SQLi pairs, {}/{} benign pairs".format(
                    seed,
                    pair_stats["changed_sqli_pairs"],
                    pair_stats["total_sqli_pairs"],
                    pair_stats["changed_benign_pairs"],
                    pair_stats["total_benign_pairs"],
                )
            )
            if mining_stats is not None:
                print(
                    "      mined stats: mean sel mal prob={:.4f}, mean sel benign prob={:.4f}".format(
                        mining_stats["mean_selected_malicious_prob"],
                        mining_stats["mean_selected_benign_prob"],
                    )
                )
            heldout_texts, heldout_labels, heldout_base, heldout_aug = build_test_family_view(
                test_texts,
                test_labels,
                seed=seed + 20_000,
                strategy_names=heldout_names,
                rounds=args.test_rounds,
                retries=args.test_retries,
                wafamole_repo=args.wafamole_repo,
            )
            vocab = build_vocab(train_canon + train_mut)

            for method in args.methods:
                cfg = PairTrainConfig(
                    method=method,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_len=args.max_len,
                    lr=args.lr,
                    consistency_weight=args.consistency_weight,
                    canonical_logit_weight=args.canonical_logit_weight,
                    device=device,
                    align_benign_pairs=args.align_benign_pairs,
                    benign_consistency_weight=args.benign_consistency_weight,
                    hard_align_gamma=args.hard_align_gamma,
                )
                print(f"  Training {method} seed={seed}")
                model = train_pair_model(cfg, train_canon, train_mut, pair_labels, vocab)
                clean_probs, clean_y = predict_proba(
                    model, test_texts, test_labels, vocab, args.max_len, device, args.batch_size
                )
                held_probs, held_y = predict_proba(
                    model, heldout_texts, heldout_labels, vocab, args.max_len, device, args.batch_size
                )
                held_sim = embedding_similarity(
                    model, heldout_base, heldout_aug, vocab, args.max_len, device, args.batch_size
                )
                row = {
                    "seed": seed,
                    "method": method,
                    "holdout_family": holdout_family,
                    "pair_stats": pair_stats,
                    "mining_stats": mining_stats,
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
        if "pair_ce" in args.methods and "pair_proj_ce" in args.methods:
            comparisons["pair_proj_ce_minus_pair_ce"] = {
                "heldout_family_recall": paired_summary(rows, ("heldout_family", "recall"), "pair_ce", "pair_proj_ce"),
                "heldout_family_p10_sqli_prob": paired_summary(
                    rows, ("heldout_family", "p10_sqli_prob"), "pair_ce", "pair_proj_ce"
                ),
                "clean_f1": paired_summary(rows, ("clean", "f1"), "pair_ce", "pair_proj_ce"),
            }
        if "pair_ce" in args.methods and "pair_canonical" in args.methods:
            comparisons["pair_canonical_minus_pair_ce"] = {
                "heldout_family_recall": paired_summary(rows, ("heldout_family", "recall"), "pair_ce", "pair_canonical"),
                "heldout_family_p10_sqli_prob": paired_summary(
                    rows, ("heldout_family", "p10_sqli_prob"), "pair_ce", "pair_canonical"
                ),
                "clean_f1": paired_summary(rows, ("clean", "f1"), "pair_ce", "pair_canonical"),
            }
        if "pair_proj_ce" in args.methods and "pair_canonical" in args.methods:
            comparisons["pair_canonical_minus_pair_proj_ce"] = {
                "heldout_family_recall": paired_summary(
                    rows, ("heldout_family", "recall"), "pair_proj_ce", "pair_canonical"
                ),
                "heldout_family_p10_sqli_prob": paired_summary(
                    rows, ("heldout_family", "p10_sqli_prob"), "pair_proj_ce", "pair_canonical"
                ),
                "clean_f1": paired_summary(rows, ("clean", "f1"), "pair_proj_ce", "pair_canonical"),
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
