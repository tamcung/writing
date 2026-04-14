#!/usr/bin/env python3
"""Parameter sweep for experiment 1: attack budget and model training sensitivity.

Two phases, each run independently with --phase 1 or --phase 2.

Phase 1 — Attack sensitivity (model params fixed at defaults):
  Sweep operator_set × (steps, candidates_per_state, beam_size).
  Train each backbone once per seed, then re-use the trained model
  across all attack configs to amortise training cost.

Phase 2 — Model sensitivity (attack config fixed at best from phase 1):
  Sweep (epochs, max_tokens) for TextCNN and BiLSTM.
  Measure clean F1 and robustness (recall under attack).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.formal.model_utils import (  # noqa: E402
    build_model,
    load_seed_split,
    resolve_device,
    rows_to_xy,
    set_seed,
)
from experiments.formal.run_exp1 import (  # noqa: E402
    attack_sqli_rows,
    evaluate_view,
    pick_attack_rows,
)
from experiments.formal.metrics import summarize  # noqa: E402


# ── Attack budget grid ───────────────────────────────────────────────────────

ATTACK_BUDGETS = [
    {"tag": "micro",   "steps":  6, "candidates": 12, "beam": 1},
    {"tag": "light",   "steps":  6, "candidates": 24, "beam": 3},
    {"tag": "default", "steps": 12, "candidates": 24, "beam": 3},
    {"tag": "wide",    "steps": 12, "candidates": 48, "beam": 5},
    {"tag": "heavy",   "steps": 20, "candidates": 48, "beam": 5},
    {"tag": "max",     "steps": 30, "candidates": 64, "beam": 8},
]

OPERATOR_SETS_P1 = ["wafamole_style", "advsqli", "official_wafamole"]

# ── Model parameter grid (Phase 2) ──────────────────────────────────────────

MODEL_GRID = list(product(
    [4, 8, 12],   # epochs
    [128, 256],   # max_tokens
    [64, 128],    # emb_dim (also channels / hidden_dim)
))  # 3 × 2 × 2 = 12 configs


# ── Helpers ──────────────────────────────────────────────────────────────────

def run_attack(model, sqli_rows, benign_rows, seed, budget, operator_set, args):
    adv_sqli = attack_sqli_rows(
        model=model,
        sqli_rows=sqli_rows,
        seed=seed,
        operator_set=operator_set,
        steps=budget["steps"],
        candidates_per_state=budget["candidates"],
        beam_size=budget["beam"],
        threshold=args.threshold,
        max_chars=args.max_chars,
        early_stop=True,
        group_size=args.group_size,
    )
    adv_view = [
        {**r, "mutated": False, "source_text": r["text"], "mutation_family": None}
        for r in benign_rows
    ] + adv_sqli
    eval_result = evaluate_view(model, adv_view, seed, "?", "adv", "adv")
    asr  = sum(1 for r in adv_sqli if r["attack_success"]) / max(1, len(adv_sqli))
    drop = float(np.mean([r["prob_drop"] for r in adv_sqli])) if adv_sqli else 0.0
    qry  = float(np.mean([r["attack_queries"] for r in adv_sqli])) if adv_sqli else 0.0
    return {
        "recall_under_attack": eval_result["metrics"]["recall"],
        "f1_under_attack":     eval_result["metrics"]["f1"],
        "asr":                 asr,
        "mean_prob_drop":      drop,
        "mean_queries":        qry,
    }


# ── Phase 1 ──────────────────────────────────────────────────────────────────

def phase1(args) -> list[dict]:
    splits_dir = Path(args.splits_dir)
    device = resolve_device(args.device)
    rows: list[dict] = []

    for seed in args.seeds:
        set_seed(seed)
        train_rows   = load_seed_split(splits_dir, seed, "train")
        test_rows    = load_seed_split(splits_dir, seed, "clean_test")
        train_texts, train_labels = rows_to_xy(train_rows)
        benign_rows, sqli_rows = pick_attack_rows(test_rows, args.attack_per_class, seed)
        clean_view = [
            {**r, "mutated": False, "source_text": r["text"], "mutation_family": None}
            for r in benign_rows + sqli_rows
        ]

        for backbone in args.backbones:
            print(f"\n[phase1] seed={seed} backbone={backbone}  training…", flush=True)
            model = build_model(backbone, args, device)
            model.fit(train_texts, train_labels)
            clean_eval = evaluate_view(model, clean_view, seed, backbone, "clean", "clean")
            clean_f1     = clean_eval["metrics"]["f1"]
            clean_recall = clean_eval["metrics"]["recall"]
            print(f"  clean  f1={clean_f1:.4f} recall={clean_recall:.4f}")

            for op_set in OPERATOR_SETS_P1:
                for budget in ATTACK_BUDGETS:
                    t0 = time.time()
                    metrics = run_attack(model, sqli_rows, benign_rows, seed, budget, op_set, args)
                    elapsed = time.time() - t0
                    row = {
                        "phase": 1,
                        "seed": seed,
                        "backbone": backbone,
                        "operator_set": op_set,
                        **budget,
                        "clean_f1": clean_f1,
                        "clean_recall": clean_recall,
                        **metrics,
                        "elapsed_s": round(elapsed, 1),
                    }
                    rows.append(row)
                    print(
                        f"  {op_set:20s} {budget['tag']:8s}"
                        f"  asr={metrics['asr']:.3f}"
                        f"  drop={metrics['mean_prob_drop']:.3f}"
                        f"  qry={metrics['mean_queries']:.0f}"
                        f"  recall_atk={metrics['recall_under_attack']:.3f}"
                        f"  ({elapsed:.0f}s)",
                        flush=True,
                    )
    return rows


# ── Phase 2 ──────────────────────────────────────────────────────────────────

def phase2(args) -> list[dict]:
    splits_dir = Path(args.splits_dir)
    device = resolve_device(args.device)
    rows: list[dict] = []
    fixed_budget = {
        "tag": "default", "steps": args.fixed_steps,
        "candidates": args.fixed_candidates, "beam": args.fixed_beam,
    }

    for seed in args.seeds:
        set_seed(seed)
        train_rows   = load_seed_split(splits_dir, seed, "train")
        test_rows    = load_seed_split(splits_dir, seed, "clean_test")
        benign_rows, sqli_rows = pick_attack_rows(test_rows, args.attack_per_class, seed)

        for backbone in ["textcnn", "bilstm"]:
            for epochs, max_tokens, emb_dim in MODEL_GRID:
                import argparse as _ap
                model_args = _ap.Namespace(**vars(args))
                model_args.epochs     = epochs
                model_args.max_tokens = max_tokens
                model_args.emb_dim    = emb_dim
                model_args.channels   = emb_dim
                model_args.hidden_dim = emb_dim

                train_texts, train_labels = rows_to_xy(train_rows)
                print(
                    f"\n[phase2] seed={seed} {backbone} epochs={epochs}"
                    f" max_tokens={max_tokens} emb_dim={emb_dim}  training…",
                    flush=True,
                )
                model = build_model(backbone, model_args, device)
                model.fit(train_texts, train_labels)

                clean_view = [
                    {**r, "mutated": False, "source_text": r["text"], "mutation_family": None}
                    for r in benign_rows + sqli_rows
                ]
                clean_eval = evaluate_view(model, clean_view, seed, backbone, "clean", "clean")
                clean_f1 = clean_eval["metrics"]["f1"]

                t0 = time.time()
                metrics = run_attack(model, sqli_rows, benign_rows, seed,
                                     fixed_budget, args.fixed_operator_set, args)
                elapsed = time.time() - t0
                row = {
                    "phase": 2,
                    "seed": seed,
                    "backbone": backbone,
                    "epochs": epochs,
                    "max_tokens": max_tokens,
                    "emb_dim": emb_dim,
                    "clean_f1": clean_f1,
                    **metrics,
                    "elapsed_s": round(elapsed, 1),
                }
                rows.append(row)
                print(
                    f"  clean_f1={clean_f1:.4f}"
                    f"  asr={metrics['asr']:.3f}"
                    f"  recall_atk={metrics['recall_under_attack']:.3f}"
                    f"  ({elapsed:.0f}s)",
                    flush=True,
                )
    return rows


# ── Summary printers ─────────────────────────────────────────────────────────

def print_phase1_summary(rows: list[dict]) -> None:
    from collections import defaultdict
    print("\n" + "=" * 80)
    print("PHASE 1 SUMMARY  (mean over seeds)")
    print("=" * 80)
    # Group by backbone / operator_set / budget_tag
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["backbone"], r["operator_set"], r["tag"])].append(r)

    backbones  = sorted({r["backbone"]   for r in rows})
    op_sets    = sorted({r["operator_set"] for r in rows})
    budget_tags = [b["tag"] for b in ATTACK_BUDGETS]

    for backbone in backbones:
        print(f"\n── {backbone} ──")
        header = f"{'op_set':22s} {'budget':8s}  {'asr':>6}  {'drop':>6}  {'qry':>6}  {'recall_atk':>10}"
        print(header)
        print("-" * len(header))
        for op_set in op_sets:
            for tag in budget_tags:
                subset = buckets[(backbone, op_set, tag)]
                if not subset:
                    continue
                asr  = np.mean([r["asr"] for r in subset])
                drop = np.mean([r["mean_prob_drop"] for r in subset])
                qry  = np.mean([r["mean_queries"] for r in subset])
                rec  = np.mean([r["recall_under_attack"] for r in subset])
                print(f"{op_set:22s} {tag:8s}  {asr:6.3f}  {drop:6.3f}  {qry:6.0f}  {rec:10.3f}")


def print_phase2_summary(rows: list[dict]) -> None:
    from collections import defaultdict
    print("\n" + "=" * 80)
    print("PHASE 2 SUMMARY  (mean over seeds)")
    print("=" * 80)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["backbone"], r["epochs"], r["max_tokens"], r["emb_dim"])].append(r)

    for backbone in ["textcnn", "bilstm"]:
        print(f"\n── {backbone} ──")
        header = f"{'epochs':>6}  {'max_tok':>7}  {'emb_dim':>7}  {'clean_f1':>8}  {'asr':>6}  {'recall_atk':>10}"
        print(header)
        print("-" * len(header))
        for (bb, ep, mt, ed), subset in sorted(buckets.items()):
            if bb != backbone:
                continue
            f1  = np.mean([r["clean_f1"] for r in subset])
            asr = np.mean([r["asr"]      for r in subset])
            rec = np.mean([r["recall_under_attack"] for r in subset])
            print(f"{ep:6d}  {mt:7d}  {ed:7d}  {f1:8.4f}  {asr:6.3f}  {rec:10.3f}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, choices=[1, 2], required=True)
    p.add_argument("--splits-dir", default="data/derived/formal_modsec_decoded/experiment1/splits")
    p.add_argument("--seeds",  nargs="+", type=int, default=[11, 22, 33])
    p.add_argument("--backbones", nargs="+", default=["word_svc", "textcnn", "bilstm"])
    p.add_argument("--attack-per-class", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-chars", type=int, default=896)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--output", default="experiments/formal/results_tune_experiment1.json")

    # Phase 2: fixed attack config (fill in best from phase 1)
    p.add_argument("--fixed-steps",        type=int, default=12)
    p.add_argument("--fixed-candidates",   type=int, default=24)
    p.add_argument("--fixed-beam",         type=int, default=3)
    p.add_argument("--fixed-operator-set", default="wafamole_style",
                   choices=["wafamole_style", "advsqli", "official_wafamole"])

    # Model defaults (used in phase 1; phase 2 sweeps these)
    p.add_argument("--word-ngram-max", type=int, default=2)
    p.add_argument("--word-min-df",    type=int, default=1)
    p.add_argument("--word-c",         type=float, default=1.0)
    p.add_argument("--epochs",         type=int, default=8)
    p.add_argument("--batch-size",     type=int, default=128)
    p.add_argument("--max-tokens",     type=int, default=256)
    p.add_argument("--max-vocab",      type=int, default=20000)
    p.add_argument("--min-freq",       type=int, default=1)
    p.add_argument("--lowercase",      action="store_true")
    p.add_argument("--lr",             type=float, default=2e-3)
    p.add_argument("--emb-dim",        type=int, default=128)
    p.add_argument("--channels",       type=int, default=128)
    p.add_argument("--hidden-dim",     type=int, default=128)
    p.add_argument("--dropout",        type=float, default=0.25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    if args.phase == 1:
        rows = phase1(args)
        print_phase1_summary(rows)
    else:
        rows = phase2(args)
        print_phase2_summary(rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8")).get("rows", [])
        except Exception:
            pass
    # Merge: keep rows from the other phase if present
    other_phase = 2 if args.phase == 1 else 1
    merged = [r for r in existing if r.get("phase") == other_phase] + rows
    payload = {
        "config": vars(args),
        "elapsed_seconds": round(time.time() - started, 1),
        "rows": merged,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
