#!/usr/bin/env python3
"""Run external generalization evaluation for paired robust SQLi detectors."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.formal.data import (  # noqa: E402
    DatasetBundle,
    balanced_sample,
    filter_dataset,
    load_dataset_by_name,
    load_manifest,
    load_processed_dataset,
)
from experiments.formal.metrics import metrics_from_probs, summarize  # noqa: E402
from experiments.formal.run_experiment1_clean_ce import (  # noqa: E402
    load_seed_split,
    resolve_device,
    rows_to_xy,
    set_seed,
    summarize_rows,
)
from experiments.formal.run_experiment2_pair_training_targeted import (  # noqa: E402
    load_or_build_training_pairs,
    train_method,
)


def sample_positive_only(bundle: DatasetBundle, seed: int, n_samples: int) -> list[dict]:
    rows = [
        {"text": text, "label": label, "source": source, "origin": origin}
        for text, label, source, origin in zip(bundle.texts, bundle.labels, bundle.sources, bundle.origins)
        if int(label) == 1
    ]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if n_samples > 0:
        rows = rows[: min(n_samples, len(rows))]
    return rows


def bundle_to_rows(bundle: DatasetBundle) -> list[dict]:
    return [
        {"text": text, "label": label, "source": source, "origin": origin}
        for text, label, source, origin in zip(bundle.texts, bundle.labels, bundle.sources, bundle.origins)
    ]


def load_sqliv5_new_sqli_only(processed_dir: Path) -> DatasetBundle:
    manifest = load_manifest(processed_dir)
    rel_path = manifest["datasets"]["sqliv5"]["new_sqli_only_path"]
    return load_processed_dataset((ROOT / rel_path).resolve(), "sqliv5_new_sqli_only")


def build_eval_views(args: argparse.Namespace, seed: int, clean_test_rows: list[dict]) -> dict[str, tuple[list[dict], str]]:
    processed_dir = Path(args.processed_dir)
    sqliv3_bundle = load_dataset_by_name(processed_dir, "sqliv3_clean")
    forbidden_texts = set(sqliv3_bundle.texts)

    views: dict[str, tuple[list[dict], str]] = {
        "sqliv3_clean_holdout": (clean_test_rows, "in_distribution"),
    }

    sqliv5_bundle = filter_dataset(load_sqliv5_new_sqli_only(processed_dir), forbidden_texts)
    views["sqliv5_new_sqli_only"] = (
        sample_positive_only(sqliv5_bundle, seed + args.sqliv5_seed_offset, args.sqliv5_sqli_samples),
        "positive_only_external",
    )

    modsec_bundle = filter_dataset(load_dataset_by_name(processed_dir, "modsec_learn_cleaned"), forbidden_texts)
    modsec_balanced = balanced_sample(modsec_bundle, seed + args.modsec_seed_offset, args.modsec_per_class)
    views["modsec_learn_cleaned_balanced"] = (bundle_to_rows(modsec_balanced), "balanced_external")

    web_bundle = filter_dataset(load_dataset_by_name(processed_dir, "web_attacks_long_test"), forbidden_texts)
    web_balanced = balanced_sample(web_bundle, seed + args.web_seed_offset, args.web_per_class)
    views["web_attacks_long_test_balanced"] = (bundle_to_rows(web_balanced), "balanced_external")

    return views


def evaluate_view(model, rows: list[dict], seed: int, backbone: str, method: str, view: str, view_kind: str) -> dict:
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


def write_partial_output(args: argparse.Namespace, started: float, rows: list[dict], train_rows: list[dict]) -> None:
    partial_path = Path(str(args.output) + ".partial.json")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "train_rows": train_rows,
    }
    partial_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_resume_state(args: argparse.Namespace) -> tuple[list[dict], list[dict], set[tuple[int, str, str]], float]:
    for path in [Path(str(args.output) + ".partial.json"), Path(args.output)]:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = list(payload.get("rows", []))
        train_rows = list(payload.get("train_rows", []))
        completed = {
            (int(row["seed"]), str(row["backbone"]), str(row["method"]))
            for row in train_rows
            if "seed" in row and "backbone" in row and "method" in row
        }
        elapsed = float(payload.get("elapsed_seconds", 0.0))
        print(f"Resuming from {path}: completed={len(completed)} elapsed_seconds={elapsed:.2f}")
        return rows, train_rows, completed, elapsed
    print("Resume requested, but no output or partial file was found. Starting from scratch.")
    return [], [], set(), 0.0


def summarize_results(rows: list[dict], backbones: list[str], methods: list[str]) -> dict:
    summary: dict[str, dict] = {}
    for view in sorted({row["view"] for row in rows}):
        summary[view] = {}
        for backbone in backbones:
            summary[view][backbone] = {}
            for method in methods:
                vals = [
                    row["metrics"]
                    for row in rows
                    if row["view"] == view and row["backbone"] == backbone and row["method"] == method
                ]
                if not vals:
                    continue
                summary[view][backbone][method] = {
                    "f1": summarize([v["f1"] for v in vals]),
                    "precision": summarize([v["precision"] for v in vals]),
                    "recall": summarize([v["recall"] for v in vals]),
                    "mean_sqli_prob": summarize([v["mean_sqli_prob"] for v in vals]),
                    "mean_benign_prob": summarize([v["mean_benign_prob"] for v in vals]),
                    "p10_sqli_prob": summarize([v["p10_sqli_prob"] for v in vals]),
                }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/formal_v3")
    parser.add_argument("--splits-dir", default="data/derived/formal_v3/experiment1/splits")
    parser.add_argument("--pairs-dir", default="data/derived/formal_v3/experiment2/pairs")
    parser.add_argument("--require-pairs", action="store_true")
    parser.add_argument("--backbones", nargs="+", default=["textcnn", "bilstm"])
    parser.add_argument("--methods", nargs="+", default=["clean_ce", "pair_ce", "pair_canonical"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    parser.add_argument("--output", default="experiments/formal/results_experiment3_external_generalization.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")

    parser.add_argument("--modsec-per-class", type=int, default=3000)
    parser.add_argument("--web-per-class", type=int, default=0, help="0 means all-balanced")
    parser.add_argument("--sqliv5-sqli-samples", type=int, default=3000, help="0 means all SQLiV5 new SQLi rows")
    parser.add_argument("--modsec-seed-offset", type=int, default=31337)
    parser.add_argument("--web-seed-offset", type=int, default=41337)
    parser.add_argument("--sqliv5-seed-offset", type=int, default=51337)

    parser.add_argument("--train-operator-set", choices=["conservative", "wafamole_style", "official_wafamole"], default="official_wafamole")
    parser.add_argument("--sqli-pairs-per-sample", type=int, default=1)
    parser.add_argument("--benign-pairs-per-sample", type=int, default=1)
    parser.add_argument("--mutation-rounds", type=int, default=5)
    parser.add_argument("--mutation-retries", type=int, default=8)
    parser.add_argument("--pair-max-chars", type=int, default=640)

    parser.add_argument("--word-ngram-max", type=int, default=2)
    parser.add_argument("--word-min-df", type=int, default=1)
    parser.add_argument("--word-c", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
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
    parser.add_argument("--max-len", type=int, default=320)
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
    device = resolve_device(args.device)
    started = time.time()
    rows: list[dict] = []
    train_summary_rows: list[dict] = []
    completed: set[tuple[int, str, str]] = set()

    if args.resume:
        rows, train_summary_rows, completed, previous_elapsed = load_resume_state(args)
        started -= previous_elapsed

    for seed in args.seeds:
        args.current_seed = seed
        set_seed(seed)
        train_rows = load_seed_split(Path(args.splits_dir), seed, "train")
        valid_rows = load_seed_split(Path(args.splits_dir), seed, "valid")
        clean_test_rows = load_seed_split(Path(args.splits_dir), seed, "clean_test")
        clean_texts, clean_labels = rows_to_xy(train_rows)
        pair_canon, pair_mutated, pair_labels, pair_stats = load_or_build_training_pairs(train_rows, seed, args)
        eval_views = build_eval_views(args, seed, clean_test_rows)
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
        print(
            "  eval sizes "
            + " ".join(f"{name}={len(view_rows)}" for name, (view_rows, _) in eval_views.items())
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
                train_summary_rows.append({"seed": seed, "backbone": backbone, "method": method, "train_stats": train_stats})

                valid_eval = evaluate_view(model, valid_rows, seed, backbone, method, "valid", "validation")
                rows.append(valid_eval)
                print(
                    f"  valid f1={valid_eval['metrics']['f1']:.4f} "
                    f"recall={valid_eval['metrics']['recall']:.4f} "
                    f"p10={valid_eval['metrics']['p10_sqli_prob']:.4f}"
                )

                for view, (view_rows, view_kind) in eval_views.items():
                    eval_row = evaluate_view(model, view_rows, seed, backbone, method, view, view_kind)
                    rows.append(eval_row)
                    metrics = eval_row["metrics"]
                    if view_kind == "positive_only_external":
                        print(
                            f"  {view} recall={metrics['recall']:.4f} "
                            f"mean_sqli_prob={metrics['mean_sqli_prob']:.4f} "
                            f"p10={metrics['p10_sqli_prob']:.4f}"
                        )
                    else:
                        print(
                            f"  {view} f1={metrics['f1']:.4f} "
                            f"precision={metrics['precision']:.4f} "
                            f"recall={metrics['recall']:.4f} "
                            f"p10={metrics['p10_sqli_prob']:.4f}"
                        )

                write_partial_output(args, started, rows, train_summary_rows)

    output = {
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "train_rows": train_summary_rows,
        "summary": summarize_results(rows, args.backbones, args.methods),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
