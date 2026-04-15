#!/usr/bin/env python3
"""Ablation: sweep consistency_weight for pair_canonical to isolate the effect of canonical alignment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.metrics import summarize  # noqa: E402
from experiments.model_utils import (  # noqa: E402
    load_seed_split,
    resolve_device,
    rows_to_xy,
    set_seed,
)
from experiments.run_exp1 import attack_sqli_rows, pick_attack_rows  # noqa: E402
from experiments.run_exp2 import load_or_build_training_pairs  # noqa: E402
from experiments.paired_models import PairSeqConfig, PairSequenceModel, PairCodeBERTConfig, PairCodeBERTModel  # noqa: E402
from experiments.metrics import metrics_from_probs  # noqa: E402

CONSISTENCY_WEIGHTS = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
ATTACK_OPERATOR_SETS = ["official_wafamole", "advsqli"]


def train_pair_canonical_codebert(consistency_weight: float, args: argparse.Namespace, device: str, train_bundle: dict):
    cfg = PairCodeBERTConfig(
        method="pair_canonical",
        seed=args.current_seed,
        model_name=args.model_name,
        local_files_only=args.local_files_only,
        epochs=args.codebert_epochs,
        batch_size=args.codebert_batch_size,
        max_len=args.max_len,
        lr=args.codebert_lr,
        encoder_lr=args.encoder_lr,
        dropout=args.codebert_dropout,
        consistency_weight=consistency_weight,
        canonical_logit_weight=0.0,
        hard_align_gamma=args.hard_align_gamma,
        align_benign_pairs=False,
        device=device,
    )
    model = PairCodeBERTModel(cfg)
    model.fit_pairs(train_bundle["pair_canon"], train_bundle["pair_mutated"], train_bundle["pair_labels"])
    return model


def train_pair_canonical(backbone: str, consistency_weight: float, args: argparse.Namespace, device: str, train_bundle: dict):
    cfg = PairSeqConfig(
        backbone=backbone,
        method="pair_canonical",
        seed=args.current_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        max_vocab=args.max_vocab,
        min_freq=args.min_freq,
        lowercase=args.lowercase,
        lr=args.lr,
        emb_dim=args.emb_dim,
        channels=args.channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        consistency_weight=consistency_weight,
        canonical_logit_weight=0.0,
        hard_align_gamma=args.hard_align_gamma,
        align_benign_pairs=False,
        device=device,
    )
    model = PairSequenceModel(cfg)
    model.fit_pairs(train_bundle["pair_canon"], train_bundle["pair_mutated"], train_bundle["pair_labels"])
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--pairs-dir", default="data/pairs")
    parser.add_argument("--backbone", default="bilstm", choices=["textcnn", "bilstm", "codebert"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--consistency-weights", nargs="+", type=float, default=CONSISTENCY_WEIGHTS)
    parser.add_argument("--attack-per-class", type=int, default=100)
    parser.add_argument("--search-steps", type=int, default=20)
    parser.add_argument("--candidates-per-state", type=int, default=48)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-chars", type=int, default=896)
    parser.add_argument("--attack-search-group-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output", default="experiments/formal/results_ablation_consistency_weight.json")

    parser.add_argument("--sqli-pairs-per-sample", type=int, default=1)
    parser.add_argument("--benign-pairs-per-sample", type=int, default=1)
    parser.add_argument("--mutation-rounds", type=int, default=7)
    parser.add_argument("--mutation-retries", type=int, default=8)
    parser.add_argument("--pair-max-chars", type=int, default=896)
    parser.add_argument("--train-operator-set", default="official_wafamole")
    parser.add_argument("--require-pairs", action="store_true")
    parser.add_argument("--hard-align-gamma", type=float, default=0.5)

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
        args.current_seed = seed
        set_seed(seed)
        train_rows = load_seed_split(splits_dir, seed, "train")
        clean_test_rows = load_seed_split(splits_dir, seed, "clean_test")
        benign_rows, sqli_rows = pick_attack_rows(clean_test_rows, args.attack_per_class, seed)
        clean_view = [
            {**r, "mutated": False, "source_text": r["text"], "mutation_family": None}
            for r in benign_rows + sqli_rows
        ]
        pair_canon, pair_mutated, pair_labels, pair_stats = load_or_build_training_pairs(train_rows, seed, args)
        train_bundle = {
            "pair_canon": pair_canon,
            "pair_mutated": pair_mutated,
            "pair_labels": pair_labels,
        }

        for cw in args.consistency_weights:
            print(f"\nseed={seed} backbone={args.backbone} consistency_weight={cw}  training…", flush=True)
            if args.backbone == "codebert":
                model = train_pair_canonical_codebert(cw, args, device, train_bundle)
            else:
                model = train_pair_canonical(args.backbone, cw, args, device, train_bundle)

            # Clean eval
            clean_texts = [str(r["text"]) for r in clean_view]
            clean_labels = [int(r["label"]) for r in clean_view]
            clean_probs = model.predict_proba(clean_texts)
            clean_metrics = metrics_from_probs(clean_probs, clean_labels)
            print(f"  clean recall={clean_metrics['recall']:.4f} f1={clean_metrics['f1']:.4f}")

            row: dict = {
                "seed": seed,
                "backbone": args.backbone,
                "consistency_weight": cw,
                "clean_recall": clean_metrics["recall"],
                "clean_f1": clean_metrics["f1"],
            }

            # Attack eval for each operator set
            for op_set in ATTACK_OPERATOR_SETS:
                adv_sqli = attack_sqli_rows(
                    model=model,
                    sqli_rows=sqli_rows,
                    seed=seed,
                    operator_set=op_set,
                    steps=args.search_steps,
                    candidates_per_state=args.candidates_per_state,
                    beam_size=args.beam_size,
                    threshold=args.threshold,
                    max_chars=args.max_chars,
                    early_stop=True,
                    group_size=args.attack_search_group_size,
                )
                adv_view = [
                    {**r, "mutated": False, "source_text": r["text"], "mutation_family": None}
                    for r in benign_rows
                ] + adv_sqli
                adv_texts = [str(r["text"]) for r in adv_view]
                adv_labels = [int(r["label"]) for r in adv_view]
                adv_probs = model.predict_proba(adv_texts)
                adv_metrics = metrics_from_probs(adv_probs, adv_labels)
                asr = sum(1 for r in adv_sqli if r["attack_success"]) / max(1, len(adv_sqli))
                row[f"recall_{op_set}"] = adv_metrics["recall"]
                row[f"asr_{op_set}"] = asr
                print(f"  {op_set}: recall={adv_metrics['recall']:.4f} asr={asr:.4f}")

            rows.append(row)

    # Summary table
    print("\n" + "=" * 70)
    print(f"ABLATION SUMMARY  backbone={args.backbone}  (mean over seeds)")
    print("=" * 70)
    header = f"{'cw':>6}  {'clean':>7}  {'recall_official':>15}  {'recall_advsqli':>14}  {'asr_official':>12}  {'asr_advsqli':>11}"
    print(header)
    print("-" * len(header))
    for cw in args.consistency_weights:
        subset = [r for r in rows if r["consistency_weight"] == cw]
        if not subset:
            continue
        print(
            f"{cw:6.3f}  "
            f"{np.mean([r['clean_recall'] for r in subset]):7.4f}  "
            f"{np.mean([r['recall_official_wafamole'] for r in subset]):15.4f}  "
            f"{np.mean([r['recall_advsqli'] for r in subset]):14.4f}  "
            f"{np.mean([r['asr_official_wafamole'] for r in subset]):12.4f}  "
            f"{np.mean([r['asr_advsqli'] for r in subset]):11.4f}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "elapsed_seconds": round(time.time() - started, 1),
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
