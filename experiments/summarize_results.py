#!/usr/bin/env python3
"""Summarize experiment results into thesis-ready tables."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments"


def fmt(mean: float, std: float, decimals: int = 3) -> str:
    m = round(mean, decimals)
    s = round(std, decimals)
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ─── Exp1 ────────────────────────────────────────────────────────────────────

def summarize_exp1(path: Path, attack_key: str, label: str) -> None:
    with open(path) as f:
        d = json.load(f)
    cfg = d["config"]
    print(f"\n{'='*70}")
    print(f"Exp1  {label}")
    print(f"  seeds={cfg.get('seeds')}  attack_per_class={cfg.get('attack_per_class')}")
    print(f"  steps={cfg.get('steps','?')}  candidates={cfg.get('candidates_per_state')}  beam={cfg.get('beam_size')}")
    print(f"{'='*70}")

    clean = d["summary"]["clean_attack_matched"]
    attack = d["summary"][attack_key]
    models = list(clean.keys())

    print(f"{'Model':<12} {'Clean F1':>12} {'Clean Rec':>12} {'Mut Rec':>12} {'ASR':>12} {'MeanDrop(p10)':>15}")
    print("-" * 75)
    for m in models:
        cf1_m = clean[m]["f1"]["mean"]
        cf1_s = clean[m]["f1"]["std"]
        cr_m  = clean[m]["recall"]["mean"]
        cr_s  = clean[m]["recall"]["std"]
        ar_m  = attack[m]["recall"]["mean"]
        ar_s  = attack[m]["recall"]["std"]
        asr_m = 1 - ar_m
        # p10 drop as proxy for mean prob drop
        cp10  = clean[m]["p10_sqli_prob"]["mean"]
        ap10  = attack[m]["p10_sqli_prob"]["mean"]
        drop  = cp10 - ap10
        print(f"{m:<12} {fmt(cf1_m,cf1_s):>12} {fmt(cr_m,cr_s):>12} {fmt(ar_m,ar_s):>12} {asr_m:>12.3f} {drop:>15.3f}")


# ─── Exp2 ────────────────────────────────────────────────────────────────────

def summarize_exp2(path: Path, attack_key: str, label: str) -> None:
    with open(path) as f:
        d = json.load(f)
    cfg = d["config"]
    print(f"\n{'='*70}")
    print(f"Exp2  {label}")
    print(f"  seeds={cfg.get('seeds')}  methods={cfg.get('methods')}  backbones={list(d['summary']['clean_attack_matched'].keys())}")
    print(f"{'='*70}")

    clean = d["summary"]["clean_attack_matched"]
    attack = d["summary"][attack_key]
    methods = cfg.get("methods", [])
    backbones = list(clean.keys())

    print(f"{'Backbone':<10} {'Method':<18} {'Clean F1':>12} {'Clean Rec':>12} {'Mut Rec':>12} {'ASR':>8}")
    print("-" * 75)
    for bb in backbones:
        for method in methods:
            if method not in clean.get(bb, {}):
                continue
            cf1_m = clean[bb][method]["f1"]["mean"]
            cf1_s = clean[bb][method]["f1"]["std"]
            cr_m  = clean[bb][method]["recall"]["mean"]
            cr_s  = clean[bb][method]["recall"]["std"]
            ar_m  = attack[bb][method]["recall"]["mean"]
            ar_s  = attack[bb][method]["recall"]["std"]
            asr   = 1 - ar_m
            print(f"{bb:<10} {method:<18} {fmt(cf1_m,cf1_s):>12} {fmt(cr_m,cr_s):>12} {fmt(ar_m,ar_s):>12} {asr:>8.3f}")
        print()


# ─── Ablation ────────────────────────────────────────────────────────────────

def summarize_ablation(path: Path, label: str) -> None:
    with open(path) as f:
        d = json.load(f)
    rows = d["rows"]
    cfg = d["config"]
    print(f"\n{'='*70}")
    print(f"Ablation  {label}")
    print(f"  backbone={cfg.get('backbone')}  seeds={cfg.get('seeds')}")
    print(f"{'='*70}")

    groups: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["consistency_weight"]].append(r)

    print(f"{'CW':<6} {'Clean Rec':>12} {'Clean F1':>12} {'Rec WAF':>12} {'ASR WAF':>9} {'Rec ADV':>12} {'ASR ADV':>9}")
    print("-" * 75)
    for cw in sorted(groups.keys()):
        rs = groups[cw]
        cr_vals  = [r["clean_recall"] for r in rs]
        cf_vals  = [r["clean_f1"] for r in rs]
        rw_vals  = [r["recall_official_wafamole"] for r in rs]
        ra_vals  = [r["recall_advsqli"] for r in rs]
        aw_vals  = [1 - r["recall_official_wafamole"] for r in rs]
        aa_vals  = [1 - r["recall_advsqli"] for r in rs]
        print(
            f"{cw:<6} "
            f"{fmt(mean(cr_vals), std(cr_vals)):>12} "
            f"{fmt(mean(cf_vals), std(cf_vals)):>12} "
            f"{fmt(mean(rw_vals), std(rw_vals)):>12} "
            f"{mean(aw_vals):>9.3f} "
            f"{fmt(mean(ra_vals), std(ra_vals)):>12} "
            f"{mean(aa_vals):>9.3f}"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Exp1
    exp1_official = EXP_DIR / "results_exp1_official.json"
    exp1_advsqli  = EXP_DIR / "results_exp1_advsqli.json"
    if exp1_official.exists():
        summarize_exp1(exp1_official, "targeted_official_wafamole", "Official WAF-A-MoLE")
    if exp1_advsqli.exists():
        summarize_exp1(exp1_advsqli, "targeted_advsqli", "AdvSQLi")

    # Exp2
    exp2_official = EXP_DIR / "results_exp2_official.json"
    exp2_advsqli  = EXP_DIR / "results_exp2_advsqli.json"
    if exp2_official.exists():
        summarize_exp2(exp2_official, "targeted_official_wafamole", "Official WAF-A-MoLE")
    if exp2_advsqli.exists():
        summarize_exp2(exp2_advsqli, "targeted_advsqli", "AdvSQLi")

    # Ablation
    abl_bilstm   = EXP_DIR / "results_ablation.json"
    abl_textcnn  = EXP_DIR / "results_ablation_textcnn.json"
    abl_codebert = EXP_DIR / "results_ablation_codebert.json"
    if abl_bilstm.exists():
        summarize_ablation(abl_bilstm, "BiLSTM")
    if abl_textcnn.exists():
        summarize_ablation(abl_textcnn, "TextCNN")
    if abl_codebert.exists():
        summarize_ablation(abl_codebert, "CodeBERT")
