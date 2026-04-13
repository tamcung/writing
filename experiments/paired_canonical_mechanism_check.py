#!/usr/bin/env python3
"""Mechanism check for whether canonical-anchor helps on unseen mutation families."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.consistency_family_holdout import (  # noqa: E402
    ALL_FAMILIES,
    build_test_family_view,
)
from experiments.consistency_sqli_experiment import (  # noqa: E402
    build_vocab,
    ensure_dataset,
    load_payload_data,
    make_split,
    predict_proba,
    set_seed,
)
from experiments.paired_canonical_family_holdout import (  # noqa: E402
    PairTrainConfig,
    build_train_pairs,
    train_pair_model,
)


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
    parser.add_argument("--output", default="experiments/paired_canonical_mechanism_check.json")
    parser.add_argument("--families", nargs="+", default=["numeric", "whitespace"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
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
    parser.add_argument("--wafamole-repo", default="external/WAF-A-MoLE")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--example-seed", type=int, default=55)
    parser.add_argument("--num-examples", type=int, default=5)
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
    family_results: dict[str, dict] = {}

    for holdout_family in args.families:
        heldout_names = ALL_FAMILIES[holdout_family]
        train_names = [name for fam, names in ALL_FAMILIES.items() if fam != holdout_family for name in names]
        print(f"\n=== Mechanism family: {holdout_family} ===")
        rows = []
        example_rows = []

        for seed in args.seeds:
            train_texts, train_labels, test_texts, test_labels = make_split(
                texts,
                labels,
                seed=seed,
                train_per_class=args.train_per_class,
                test_per_class=args.test_per_class,
            )
            train_canon, train_mut, pair_labels = build_train_pairs(
                train_texts=train_texts,
                train_labels=train_labels,
                seed=seed,
                strategy_names=train_names,
                pairs_per_sample=args.pairs_per_sample,
                rounds=args.train_rounds,
                retries=args.train_retries,
                wafamole_repo=args.wafamole_repo,
                benign_pairs="nuisance",
                benign_rounds=args.benign_rounds,
                benign_retries=args.benign_retries,
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

            model_outputs = {}
            for method in ("pair_proj_ce", "pair_canonical"):
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
                    hard_align_gamma=args.hard_align_gamma if method == "pair_canonical" else 0.0,
                )
                print(f"  Training {method} seed={seed}")
                model = train_pair_model(cfg, train_canon, train_mut, pair_labels, vocab)

                malicious_clean = [t for t, y in zip(test_texts, test_labels) if y == 1]
                malicious_hold = [t for t, y in zip(heldout_texts, heldout_labels) if y == 1]
                clean_probs, _ = predict_proba(
                    model,
                    malicious_clean,
                    [1] * len(malicious_clean),
                    vocab,
                    args.max_len,
                    device,
                    args.batch_size,
                )
                hold_probs, _ = predict_proba(
                    model,
                    malicious_hold,
                    [1] * len(malicious_hold),
                    vocab,
                    args.max_len,
                    device,
                    args.batch_size,
                )
                model_outputs[method] = {
                    "clean_probs": clean_probs,
                    "hold_probs": hold_probs,
                }

            proj_clean = model_outputs["pair_proj_ce"]["clean_probs"]
            proj_hold = model_outputs["pair_proj_ce"]["hold_probs"]
            can_clean = model_outputs["pair_canonical"]["clean_probs"]
            can_hold = model_outputs["pair_canonical"]["hold_probs"]
            proj_drop = proj_clean - proj_hold
            can_drop = can_clean - can_hold
            rescue = int(np.sum((proj_hold < 0.5) & (can_hold >= 0.5)))
            harm = int(np.sum((proj_hold >= 0.5) & (can_hold < 0.5)))

            row = {
                "seed": seed,
                "family": holdout_family,
                "proj_clean_prob_mean": float(np.mean(proj_clean)),
                "can_clean_prob_mean": float(np.mean(can_clean)),
                "proj_hold_prob_mean": float(np.mean(proj_hold)),
                "can_hold_prob_mean": float(np.mean(can_hold)),
                "proj_drop_mean": float(np.mean(proj_drop)),
                "can_drop_mean": float(np.mean(can_drop)),
                "proj_drop_p90": float(np.quantile(proj_drop, 0.90)),
                "can_drop_p90": float(np.quantile(can_drop, 0.90)),
                "rescue_count": rescue,
                "harm_count": harm,
            }
            rows.append(row)
            print(
                "    proj_hold={:.4f} can_hold={:.4f} proj_drop={:.4f} can_drop={:.4f} rescue={} harm={}".format(
                    row["proj_hold_prob_mean"],
                    row["can_hold_prob_mean"],
                    row["proj_drop_mean"],
                    row["can_drop_mean"],
                    rescue,
                    harm,
                )
            )

            if seed == args.example_seed:
                gains = can_hold - proj_hold
                top_idx = np.argsort(gains)[::-1][: args.num_examples]
                bottom_idx = np.argsort(gains)[: args.num_examples]
                example_rows = {
                    "top_rescued": [
                        {
                            "base": heldout_base[int(i)],
                            "mutated": heldout_aug[int(i)],
                            "proj_hold_prob": float(proj_hold[int(i)]),
                            "can_hold_prob": float(can_hold[int(i)]),
                            "gain": float(gains[int(i)]),
                        }
                        for i in top_idx
                    ],
                    "top_hurt": [
                        {
                            "base": heldout_base[int(i)],
                            "mutated": heldout_aug[int(i)],
                            "proj_hold_prob": float(proj_hold[int(i)]),
                            "can_hold_prob": float(can_hold[int(i)]),
                            "gain": float(gains[int(i)]),
                        }
                        for i in bottom_idx
                    ],
                }

        family_results[holdout_family] = {
            "rows": rows,
            "summary": {
                "proj_hold_prob_mean": summarize([r["proj_hold_prob_mean"] for r in rows]),
                "can_hold_prob_mean": summarize([r["can_hold_prob_mean"] for r in rows]),
                "proj_drop_mean": summarize([r["proj_drop_mean"] for r in rows]),
                "can_drop_mean": summarize([r["can_drop_mean"] for r in rows]),
                "proj_drop_p90": summarize([r["proj_drop_p90"] for r in rows]),
                "can_drop_p90": summarize([r["can_drop_p90"] for r in rows]),
                "rescue_count": summarize([r["rescue_count"] for r in rows]),
                "harm_count": summarize([r["harm_count"] for r in rows]),
            },
            "examples": example_rows,
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
