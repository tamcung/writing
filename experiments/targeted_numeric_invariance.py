#!/usr/bin/env python3
"""Targeted numeric invariance experiment.

Compare:
- base_ce: official non-numeric augmentation only
- numeric_ce: same + custom numeric-equivalent views with CE
- numeric_consistency: same views + targeted numeric consistency loss
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
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
    build_vocab,
    encode,
    ensure_dataset,
    load_payload_data,
    make_split,
    metrics_from_probs,
    predict_proba,
    set_seed,
)
from experiments.consistency_family_holdout import (  # noqa: E402
    apply_strategy_rounds,
    build_test_family_view,
)


NON_NUMERIC_STRATEGIES = [
    "spaces_to_comments",
    "spaces_to_whitespaces_alternatives",
    "comment_rewriting",
    "reset_inline_comments",
    "random_case",
    "swap_keywords",
    "change_tautologies",
    "logical_invariant",
]
NUM_RE = re.compile(r"\b\d+\b")


def numeric_equiv_transform(text: str, seed: int, variant_idx: int) -> str:
    """Generate semantic-preserving numeric representation views."""
    rng = random.Random(seed)
    matches = list(NUM_RE.finditer(text))
    if not matches:
        return text

    indices = list(range(len(matches)))
    rng.shuffle(indices)
    use_count = max(1, min(len(matches), 1 + (variant_idx % 2)))
    chosen = sorted(indices[:use_count], reverse=True)

    def repl_num(value: str, mode: str) -> str:
        n = int(value)
        if mode == "hex":
            return hex(n)
        if mode == "select":
            return f"(SELECT {n})"
        if mode == "plus_zero":
            return f"({n}+0)"
        if mode == "double_neg":
            return f"(-(-{n}))"
        return value

    modes = ["hex", "select", "plus_zero", "double_neg"]
    out = text
    for pos, idx in enumerate(sorted(chosen, reverse=True)):
        match = matches[idx]
        mode = modes[(seed + variant_idx + pos) % len(modes)]
        replacement = repl_num(match.group(0), mode)
        out = out[: match.start()] + replacement + out[match.end() :]
    return out[:260]


def build_train_views(
    train_texts: list[str],
    train_labels: list[int],
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[list[str]], list[list[int]]]:
    """Return views and mask where 1 denotes numeric-targeted views."""
    all_views = []
    all_numeric_masks = []
    for i, (text, y) in enumerate(zip(train_texts, train_labels)):
        views = [text]
        numeric_mask = [0]
        if y == 1:
            for v in range(args.num_base_views):
                aug = apply_strategy_rounds(
                    text=text,
                    strategy_names=NON_NUMERIC_STRATEGIES,
                    seed=seed * 1_000_003 + i * 10_007 + v * 1_009,
                    rounds=args.base_rounds,
                    ensure_changed=True,
                    retries=args.retries,
                    wafamole_repo=args.wafamole_repo,
                )
                views.append(aug)
                numeric_mask.append(0)
            for v in range(args.num_numeric_views):
                aug = numeric_equiv_transform(
                    text=text,
                    seed=seed * 2_000_003 + i * 20_011 + v * 2_021,
                    variant_idx=v,
                )
                views.append(aug)
                numeric_mask.append(1 if aug != text else 0)
        else:
            for _ in range(args.num_base_views + args.num_numeric_views):
                views.append(text)
                numeric_mask.append(0)
        all_views.append(views)
        all_numeric_masks.append(numeric_mask)
    return all_views, all_numeric_masks


class MultiViewNumericDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        view_texts: list[list[str]],
        numeric_masks: list[list[int]],
        vocab: dict[str, int],
        max_len: int,
        mode: str,
        base_view_count: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.view_texts = view_texts
        self.numeric_masks = numeric_masks
        self.vocab = vocab
        self.max_len = max_len
        self.mode = mode
        self.base_view_count = base_view_count

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        y = self.labels[idx]
        views = self.view_texts[idx]
        numeric_mask = self.numeric_masks[idx]
        if self.mode == "base_ce":
            keep = 1 + self.base_view_count
            views = views[:keep]
            numeric_mask = numeric_mask[:keep]
        x = torch.tensor([encode(v, self.vocab, self.max_len) for v in views], dtype=torch.long)
        return x, torch.tensor(y, dtype=torch.float32), torch.tensor(numeric_mask, dtype=torch.float32)


class ProjectedCharCNN(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 32, channels: int = 48, proj_dim: int = 48) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, channels, kernel_size=k, padding=k // 2) for k in (3, 5, 7)]
        )
        self.dropout = nn.Dropout(0.25)
        feat_dim = channels * len(self.convs)
        self.classifier = nn.Linear(feat_dim, 1)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(emb))
            pooled.append(torch.max(h, dim=-1).values)
        return self.dropout(torch.cat(pooled, dim=1))

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        return self.classifier(h).squeeze(1), h, self.projector(h)


@torch.no_grad()
def model_predict_proba(
    model: ProjectedCharCNN,
    texts: list[str],
    labels: list[int],
    vocab: dict[str, int],
    max_len: int,
    device: str,
    batch_size: int,
):
    from experiments.consistency_sqli_experiment import EvalDataset

    model.eval()
    loader = DataLoader(EvalDataset(texts, labels, vocab, max_len), batch_size=batch_size, shuffle=False)
    probs = []
    ys = []
    for x, y in loader:
        logits, _, _ = model(x.to(device))
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        ys.extend(y.numpy().tolist())
    return np.asarray(probs), np.asarray(ys, dtype=int)


@torch.no_grad()
def model_embedding_similarity(
    model: ProjectedCharCNN,
    base_texts: list[str],
    aug_texts: list[str],
    vocab: dict[str, int],
    max_len: int,
    device: str,
    batch_size: int,
) -> float:
    model.eval()
    sims = []
    for start in range(0, len(base_texts), batch_size):
        base = base_texts[start : start + batch_size]
        aug = aug_texts[start : start + batch_size]
        x1 = torch.tensor([encode(x, vocab, max_len) for x in base], dtype=torch.long, device=device)
        x2 = torch.tensor([encode(x, vocab, max_len) for x in aug], dtype=torch.long, device=device)
        _, _, z1 = model(x1)
        _, _, z2 = model(x2)
        sims.extend(torch.cosine_similarity(z1, z2, dim=1).cpu().numpy().tolist())
    return float(np.mean(sims))


def train_numeric_model(
    mode: str,
    train_texts: list[str],
    train_labels: list[int],
    view_texts: list[list[str]],
    numeric_masks: list[list[int]],
    vocab: dict[str, int],
    args: argparse.Namespace,
    seed: int,
    device: str,
) -> ProjectedCharCNN:
    set_seed(seed)
    model = ProjectedCharCNN(vocab_size=max(vocab.values(), default=1) + 1).to(device)
    dataset = MultiViewNumericDataset(
        train_texts,
        train_labels,
        view_texts,
        numeric_masks,
        vocab,
        args.max_len,
        mode=mode,
        base_view_count=args.num_base_views,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(args.epochs):
        model.train()
        for x_views, y, numeric_mask in loader:
            x_views = x_views.to(device)
            y = y.to(device)
            numeric_mask = numeric_mask.to(device)
            bs, nv, seq_len = x_views.shape
            logits, h, z = model(x_views.reshape(bs * nv, seq_len))
            logits = logits.view(bs, nv)
            z = z.view(bs, nv, -1)

            y_expand = y.unsqueeze(1).expand(bs, nv)
            loss = bce(logits.reshape(-1), y_expand.reshape(-1))

            if mode == "numeric_consistency":
                pos_rows = (y > 0.5) & (numeric_mask.sum(dim=1) > 0)
                if pos_rows.any():
                    z_pos = z[pos_rows]
                    mask_pos = numeric_mask[pos_rows].bool()
                    anchor = z_pos[:, 0]
                    diffs = []
                    logit_diffs = []
                    for i in range(z_pos.shape[0]):
                        pos_views = z_pos[i][mask_pos[i]]
                        if pos_views.shape[0] == 0:
                            continue
                        anchor_i = anchor[i].unsqueeze(0).expand_as(pos_views)
                        diffs.append(1.0 - torch.cosine_similarity(anchor_i, pos_views, dim=1))
                        anchor_logit = logits[pos_rows][i, 0].expand(pos_views.shape[0])
                        pos_logits = logits[pos_rows][i][mask_pos[i]]
                        logit_diffs.append((anchor_logit - pos_logits).pow(2))
                    if diffs:
                        loss = loss + args.consistency_weight * torch.cat(diffs).mean()
                    if logit_diffs:
                        loss = loss + args.logit_weight * torch.cat(logit_diffs).mean()

            opt.zero_grad(set_to_none=True)
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
    diffs = [methods[b] - methods[a] for methods in by_seed.values() if a in methods and b in methods]
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
    parser.add_argument("--output", default="experiments/targeted_numeric_invariance_results.json")
    parser.add_argument("--wafamole-repo", default="external/WAF-A-MoLE")
    parser.add_argument("--methods", nargs="+", default=["base_ce", "numeric_ce", "numeric_consistency"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--consistency-weight", type=float, default=0.5)
    parser.add_argument("--logit-weight", type=float, default=0.1)
    parser.add_argument("--num-base-views", type=int, default=2)
    parser.add_argument("--num-numeric-views", type=int, default=2)
    parser.add_argument("--base-rounds", type=int, default=3)
    parser.add_argument("--retries", type=int, default=8)
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

    rows = []
    started = time.time()
    for seed in args.seeds:
        train_texts, train_labels, test_texts, test_labels = make_split(
            texts,
            labels,
            seed=seed,
            train_per_class=args.train_per_class,
            test_per_class=args.test_per_class,
        )
        train_views, numeric_masks = build_train_views(train_texts, train_labels, seed, args)
        changed_numeric = sum(
            1 for text, y, views, masks in zip(train_texts, train_labels, train_views, numeric_masks)
            if y == 1 and any(m == 1 and v != text for v, m in zip(views, masks))
        )
        total_sqli = sum(1 for y in train_labels if y == 1)
        print(
            f"seed={seed} targeted numeric views changed {changed_numeric}/{total_sqli} SQLi samples"
        )

        numeric_holdout_texts, numeric_holdout_labels, numeric_base, numeric_aug = build_test_family_view(
            test_texts,
            test_labels,
            seed=seed + 20_000,
            strategy_names=["swap_int_repr"],
            rounds=3,
            retries=12,
            wafamole_repo=args.wafamole_repo,
        )
        vocab = build_vocab(train_texts + [v for views in train_views for v in views])

        for method in args.methods:
            print(f"  Training {method} seed={seed}")
            model = train_numeric_model(
                mode=method,
                train_texts=train_texts,
                train_labels=train_labels,
                view_texts=train_views,
                numeric_masks=numeric_masks,
                vocab=vocab,
                args=args,
                seed=seed,
                device=device,
            )
            clean_probs, clean_y = model_predict_proba(
                model, test_texts, test_labels, vocab, args.max_len, device, args.batch_size
            )
            num_probs, num_y = model_predict_proba(
                model, numeric_holdout_texts, numeric_holdout_labels, vocab, args.max_len, device, args.batch_size
            )
            num_sim = model_embedding_similarity(
                model, numeric_base, numeric_aug, vocab, args.max_len, device, args.batch_size
            )
            row = {
                "seed": seed,
                "method": method,
                "clean": metrics_from_probs(clean_probs, clean_y),
                "numeric_holdout": metrics_from_probs(num_probs, num_y),
                "embedding_cosine_numeric_holdout": num_sim,
            }
            rows.append(row)
            print(
                "    clean_f1={:.4f} num_recall={:.4f} num_p10={:.4f} num_sim={:.4f}".format(
                    row["clean"]["f1"],
                    row["numeric_holdout"]["recall"],
                    row["numeric_holdout"]["p10_sqli_prob"],
                    row["embedding_cosine_numeric_holdout"],
                )
            )

    summary = {}
    for method in args.methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {
            "clean_f1": summarize([row["clean"]["f1"] for row in method_rows]),
            "numeric_holdout_recall": summarize([row["numeric_holdout"]["recall"] for row in method_rows]),
            "numeric_holdout_p10_sqli_prob": summarize(
                [row["numeric_holdout"]["p10_sqli_prob"] for row in method_rows]
            ),
            "embedding_cosine_numeric_holdout": summarize(
                [row["embedding_cosine_numeric_holdout"] for row in method_rows]
            ),
        }

    comparisons = {}
    if "base_ce" in args.methods and "numeric_ce" in args.methods:
        comparisons["numeric_ce_minus_base_ce"] = {
            "numeric_holdout_recall": paired_summary(rows, ("numeric_holdout", "recall"), "base_ce", "numeric_ce"),
            "numeric_holdout_p10_sqli_prob": paired_summary(
                rows, ("numeric_holdout", "p10_sqli_prob"), "base_ce", "numeric_ce"
            ),
        }
    if "numeric_ce" in args.methods and "numeric_consistency" in args.methods:
        comparisons["numeric_consistency_minus_numeric_ce"] = {
            "numeric_holdout_recall": paired_summary(
                rows, ("numeric_holdout", "recall"), "numeric_ce", "numeric_consistency"
            ),
            "numeric_holdout_p10_sqli_prob": paired_summary(
                rows, ("numeric_holdout", "p10_sqli_prob"), "numeric_ce", "numeric_consistency"
            ),
            "embedding_cosine_numeric_holdout": paired_summary(
                rows, ("embedding_cosine_numeric_holdout",), "numeric_ce", "numeric_consistency"
            ),
        }

    result = {
        "config": vars(args) | {"device_resolved": device},
        "elapsed_seconds": time.time() - started,
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
