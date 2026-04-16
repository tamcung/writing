#!/usr/bin/env python3
"""Run paired robust training under official WAF-A-MoLE targeted evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.metrics import metrics_from_probs, summarize  # noqa: E402
from experiments.pair_data import (  # noqa: E402
    build_pair_rows,
    load_pair_rows,
    pair_rows_to_training_arrays,
    summarize_pair_rows,
)
from experiments.paired_models import (  # noqa: E402
    PairCodeBERTConfig,
    PairCodeBERTModel,
    PairSeqConfig,
    PairSequenceModel,
)
from experiments.model_utils import (  # noqa: E402
    build_model,
    load_seed_split,
    resolve_device,
    rows_to_xy,
    set_seed,
    summarize_rows,
)
from experiments.run_exp1 import (  # noqa: E402
    attack_sqli_rows,
    pick_attack_rows,
    summarize_attack_rows,
)


def build_pair_model(backbone: str, method: str, args: argparse.Namespace, device: str):
    if backbone in {"textcnn", "bilstm"}:
        cfg = PairSeqConfig(
            backbone=backbone,
            method=method,
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
            consistency_weight=args.consistency_weight,
            canonical_logit_weight=args.canonical_logit_weight,
            hard_align_gamma=args.hard_align_gamma if method == "pair_canonical" else 0.0,
            align_benign_pairs=args.align_benign_pairs,
            device=device,
        )
        return PairSequenceModel(cfg)
    if backbone == "codebert":
        cfg = PairCodeBERTConfig(
            method=method,
            seed=args.current_seed,
            model_name=args.model_name,
            local_files_only=args.local_files_only,
            freeze_encoder=args.freeze_encoder,
            epochs=args.codebert_epochs,
            batch_size=args.codebert_batch_size,
            max_len=args.max_len,
            lr=args.codebert_lr,
            encoder_lr=args.encoder_lr,
            dropout=args.codebert_dropout,
            consistency_weight=args.consistency_weight,
            canonical_logit_weight=args.canonical_logit_weight,
            hard_align_gamma=args.hard_align_gamma if method == "pair_canonical" else 0.0,
            align_benign_pairs=args.align_benign_pairs,
            grad_clip=args.grad_clip,
            device=device,
        )
        return PairCodeBERTModel(cfg)
    raise ValueError(f"Paired methods do not support backbone={backbone}")


def train_method(backbone: str, method: str, args: argparse.Namespace, device: str, train_bundle: dict):
    if method == "clean_ce":
        model = build_model(backbone, args, device)
        model.fit(train_bundle["clean_texts"], train_bundle["clean_labels"])
        return model, {"method": method, "uses_pairs": False}

    if backbone == "word_svc":
        raise ValueError("word_svc is only supported for clean_ce in experiment 2.")
    model = build_pair_model(backbone, method, args, device)
    model.fit_pairs(train_bundle["pair_canon"], train_bundle["pair_mutated"], train_bundle["pair_labels"])
    return model, {"method": method, "uses_pairs": True, **train_bundle["pair_stats"]}


def load_or_build_training_pairs(
    train_rows: list[dict],
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[str], list[str], list[int], dict]:
    pairs_path = Path(args.pairs_dir) / f"seed_{seed}" / "train_pairs.json"
    if pairs_path.exists():
        rows = load_pair_rows(pairs_path)
        manifest_path = pairs_path.with_name("manifest.json")
        if manifest_path.exists():
            stats = dict(json.loads(manifest_path.read_text(encoding="utf-8")).get("stats", {}))
        else:
            stats = summarize_pair_rows(rows)
        stats["pairs_path"] = str(pairs_path.resolve())
        stats["source"] = "materialized"
        return (*pair_rows_to_training_arrays(rows), stats)
    if args.require_pairs:
        raise FileNotFoundError(f"Required pair file not found: {pairs_path}")

    pair_rows, stats = build_pair_rows(
        train_rows=train_rows,
        seed=seed,
        operator_set=args.train_operator_set,
        sqli_pairs_per_sample=args.sqli_pairs_per_sample,
        benign_pairs_per_sample=args.benign_pairs_per_sample,
        mutation_rounds=args.mutation_rounds,
        mutation_retries=args.mutation_retries,
        max_chars=args.pair_max_chars,
    )
    stats["source"] = "built_at_runtime"
    return (*pair_rows_to_training_arrays(pair_rows), stats)


def evaluate_rows(model, rows: list[dict], seed: int, backbone: str, method: str, view: str, view_kind: str) -> dict:
    texts, labels = rows_to_xy(rows)
    probs = model.predict_proba(texts)
    return {
        "seed": seed,
        "backbone": backbone,
        "method": method,
        "view": view,
        "view_kind": view_kind,
        "metrics": metrics_from_probs(probs, labels),
        "stats": summarize_rows(rows),
    }


def write_partial_output(
    args: argparse.Namespace,
    started: float,
    rows: list[dict],
    attack_rows: list[dict],
    pair_rows: list[dict],
    completed: dict,
) -> None:
    partial_path = Path(str(args.output) + ".partial.json")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "completed": completed,
        "rows": rows,
        "attack_rows": attack_rows,
        "pair_rows": pair_rows,
    }
    partial_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_resume_state(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict], set[tuple[int, str, str]], float]:
    for path in [Path(str(args.output) + ".partial.json"), Path(args.output)]:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = list(payload.get("rows", []))
        attack_rows = list(payload.get("attack_rows", []))
        pair_rows = list(payload.get("pair_rows", []))
        completed = {
            (int(row["seed"]), str(row["backbone"]), str(row["method"]))
            for row in attack_rows
            if "seed" in row and "backbone" in row and "method" in row
        }
        elapsed = float(payload.get("elapsed_seconds", 0.0))
        print(f"Resuming from {path}: completed={len(completed)} elapsed_seconds={elapsed:.2f}")
        return rows, attack_rows, pair_rows, completed, elapsed
    print("Resume requested, but no output or partial file was found. Starting from scratch.")
    return [], [], [], set(), 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--backbones", nargs="+", default=["textcnn", "bilstm", "codebert"])
    parser.add_argument("--methods", nargs="+", default=["clean_ce", "pair_ce", "pair_canonical"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--pairs-dir", default="data/pairs")
    parser.add_argument("--require-pairs", action="store_true")
    parser.add_argument("--output", default="experiments/formal/results_experiment2_pair_training_targeted.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")

    parser.add_argument("--train-operator-set", choices=["wafamole"], default="wafamole")
    parser.add_argument("--attack-operator-set", choices=["wafamole", "advsqli"], default="wafamole")
    parser.add_argument("--sqli-pairs-per-sample", type=int, default=1)
    parser.add_argument("--benign-pairs-per-sample", type=int, default=1)
    parser.add_argument("--mutation-rounds", type=int, default=7)
    parser.add_argument("--mutation-retries", type=int, default=8)
    parser.add_argument("--pair-max-chars", type=int, default=896)

    parser.add_argument("--attack-per-class", type=int, default=300)
    parser.add_argument("--search-steps", type=int, default=20)
    parser.add_argument("--candidates-per-state", type=int, default=48)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-chars", type=int, default=896)
    parser.add_argument("--attack-search-group-size", type=int, default=128)
    parser.add_argument("--no-early-stop", action="store_true")

    parser.add_argument("--word-ngram-max", type=int, default=2)
    parser.add_argument("--word-min-df", type=int, default=1)
    parser.add_argument("--word-c", type=float, default=1.0)

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
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--canonical-logit-weight", type=float, default=0.0)
    parser.add_argument("--hard-align-gamma", type=float, default=0.5)
    parser.add_argument("--align-benign-pairs", action="store_true")

    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--codebert-epochs", type=int, default=2)
    parser.add_argument("--codebert-batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--codebert-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--codebert-dropout", type=float, default=0.1)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
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
    attack_rows: list[dict] = []
    pair_rows: list[dict] = []
    completed: set[tuple[int, str, str]] = set()
    if args.resume:
        rows, attack_rows, pair_rows, completed, previous_elapsed = load_resume_state(args)
        started -= previous_elapsed

    for seed in args.seeds:
        args.current_seed = seed
        set_seed(seed)
        train_rows = load_seed_split(splits_dir, seed, "train")
        valid_rows = load_seed_split(splits_dir, seed, "valid")
        clean_test_rows = load_seed_split(splits_dir, seed, "clean_test")
        clean_texts, clean_labels = rows_to_xy(train_rows)
        pair_canon, pair_mutated, pair_labels, pair_stats = load_or_build_training_pairs(train_rows, seed, args)
        benign_rows, sqli_rows = pick_attack_rows(clean_test_rows, args.attack_per_class, seed)
        clean_attack_view = [
            {**row, "mutated": False, "source_text": row["text"], "mutation_family": None}
            for row in benign_rows + sqli_rows
        ]
        train_bundle = {
            "clean_texts": clean_texts,
            "clean_labels": clean_labels,
            "pair_canon": pair_canon,
            "pair_mutated": pair_mutated,
            "pair_labels": pair_labels,
            "pair_stats": pair_stats,
        }
        print(
            f"seed={seed} pair_stats sqli_changed={pair_stats['sqli_changed_rate']:.4f} "
            f"benign_changed={pair_stats['benign_changed_rate']:.4f} "
            f"pairs={pair_stats['total_pairs']} source={pair_stats.get('source')}"
        )

        for backbone in args.backbones:
            for method in args.methods:
                if method != "clean_ce" and backbone == "word_svc":
                    print(f"seed={seed} backbone=word_svc method={method} unsupported; skipping")
                    continue
                if (seed, backbone, method) in completed:
                    print(f"seed={seed} backbone={backbone} method={method} already completed; skipping")
                    continue

                print(f"seed={seed} training {backbone}/{method}")
                model, train_stats = train_method(backbone, method, args, device, train_bundle)
                pair_rows.append({"seed": seed, "backbone": backbone, "method": method, "train_stats": train_stats})

                valid_eval = evaluate_rows(model, valid_rows, seed, backbone, method, "valid", "validation")
                rows.append(valid_eval)

                clean_eval = evaluate_rows(
                    model,
                    clean_attack_view,
                    seed,
                    backbone,
                    method,
                    "clean_attack_matched",
                    "clean",
                )
                rows.append(clean_eval)
                print(
                    f"  clean_attack_matched f1={clean_eval['metrics']['f1']:.4f} "
                    f"recall={clean_eval['metrics']['recall']:.4f} "
                    f"p10={clean_eval['metrics']['p10_sqli_prob']:.4f}"
                )

                print(
                    f"  targeted search operator_set={args.attack_operator_set} "
                    f"sqli={len(sqli_rows)} steps={args.search_steps} "
                    f"candidates={args.candidates_per_state} beam={args.beam_size}"
                )
                adv_sqli_rows = attack_sqli_rows(
                    model=model,
                    sqli_rows=sqli_rows,
                    seed=seed,
                    operator_set=args.attack_operator_set,
                    steps=args.search_steps,
                    candidates_per_state=args.candidates_per_state,
                    beam_size=args.beam_size,
                    threshold=args.threshold,
                    max_chars=args.max_chars,
                    early_stop=not args.no_early_stop,
                    group_size=args.attack_search_group_size,
                )
                adv_view = [
                    {**row, "mutated": False, "source_text": row["text"], "mutation_family": None}
                    for row in benign_rows
                ] + adv_sqli_rows
                adv_eval = evaluate_rows(
                    model,
                    adv_view,
                    seed,
                    backbone,
                    method,
                    f"targeted_{args.attack_operator_set}",
                    "targeted_mutated",
                )
                rows.append(adv_eval)
                attack_summary = summarize_attack_rows(adv_sqli_rows)
                attack_rows.append(
                    {
                        "seed": seed,
                        "backbone": backbone,
                        "method": method,
                        "operator_set": args.attack_operator_set,
                        "attack_summary": attack_summary,
                        "examples": adv_sqli_rows[:10],
                    }
                )
                print(
                    f"  targeted_{args.attack_operator_set} f1={adv_eval['metrics']['f1']:.4f} "
                    f"recall={adv_eval['metrics']['recall']:.4f} "
                    f"p10={adv_eval['metrics']['p10_sqli_prob']:.4f} "
                    f"success={attack_summary.get('success_rate', 0):.4f} "
                    f"drop={attack_summary.get('mean_prob_drop', 0):.4f}"
                )
                write_partial_output(
                    args=args,
                    started=started,
                    rows=rows,
                    attack_rows=attack_rows,
                    pair_rows=pair_rows,
                    completed={"seed": seed, "backbone": backbone, "method": method},
                )

    summary: dict[str, dict] = {}
    for view in sorted({row["view"] for row in rows}):
        summary[view] = {}
        for backbone in args.backbones:
            summary[view][backbone] = {}
            for method in args.methods:
                vals = [
                    row["metrics"]
                    for row in rows
                    if row["view"] == view and row["backbone"] == backbone and row["method"] == method
                ]
                if not vals:
                    continue
                summary[view][backbone][method] = {
                    "f1": summarize([v["f1"] for v in vals]),
                    "recall": summarize([v["recall"] for v in vals]),
                    "precision": summarize([v["precision"] for v in vals]),
                    "p10_sqli_prob": summarize([v["p10_sqli_prob"] for v in vals]),
                }

    attack_summary_by_method: dict[str, dict] = {}
    for backbone in args.backbones:
        attack_summary_by_method[backbone] = {}
        for method in args.methods:
            vals = [
                row["attack_summary"]
                for row in attack_rows
                if row["backbone"] == backbone and row["method"] == method
            ]
            if not vals:
                continue
            attack_summary_by_method[backbone][method] = {
                "success_rate": summarize([v["success_rate"] for v in vals]),
                "mean_prob_drop": summarize([v["mean_prob_drop"] for v in vals]),
                "mean_adversarial_sqli_prob": summarize([v["mean_adversarial_sqli_prob"] for v in vals]),
                "mean_queries": summarize([v["mean_queries"] for v in vals]),
            }

    output = {
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "attack_rows": attack_rows,
        "pair_rows": pair_rows,
        "summary": summary,
        "attack_summary_by_method": attack_summary_by_method,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
