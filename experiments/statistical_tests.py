#!/usr/bin/env python3
"""Wilcoxon signed-rank tests for pairwise method comparisons (Exp2)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments"


def load_exp2(path: Path, op_set: str) -> dict:
    """Returns {backbone: {method: {'clean_recall': [], 'mut_recall': [], 'asr': []}}}"""
    with open(path) as f:
        d = json.load(f)

    # clean metrics per seed
    clean: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in d["rows"]:
        if row["view_kind"] != "clean_attack_matched":
            continue
        bb, method = row["backbone"], row["method"]
        clean[bb][method]["clean_recall"].append(row["metrics"]["recall"])
        clean[bb][method]["clean_f1"].append(row["metrics"]["f1"])

    # attack metrics per seed
    for arow in d["attack_rows"]:
        if arow["operator_set"] != op_set:
            continue
        bb, method = arow["backbone"], arow["method"]
        asr = arow["attack_summary"]["success_rate"]
        # reconstruct mut_recall from attack_summary
        # mut_recall = fraction of sqli that were NOT successfully attacked
        # We need per-example data; use success_rate as proxy
        clean[bb][method]["asr"].append(asr)
        clean[bb][method]["mut_recall"].append(1 - asr)

    return {bb: dict(methods) for bb, methods in clean.items()}


def wilcoxon_pair(a: list[float], b: list[float], label: str, higher_is_better: bool = True) -> str:
    """One-sided Wilcoxon: test if a > b (higher_is_better=True) or a < b."""
    if len(a) < 2 or len(a) != len(b):
        return f"  {label}: insufficient data"
    diff = [x - y for x, y in zip(a, b)]
    if all(abs(d) < 1e-10 for d in diff):
        return f"  {label}: Δ=0.000 (identical)"
    try:
        alt = "greater" if higher_is_better else "less"
        stat, p = wilcoxon(a, b, alternative=alt)
        mean_diff = np.mean(diff)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        return f"  {label}: Δ={mean_diff:+.4f}  p={p:.4f}  {sig}"
    except Exception as e:
        return f"  {label}: ERROR {e}"


def run_exp2_tests(path: Path, op_set: str, label: str) -> None:
    data = load_exp2(path, op_set)
    methods = ["clean_ce", "pair_ce", "pair_canonical"]
    backbones = ["textcnn", "bilstm", "codebert"]

    print(f"\n{'='*65}")
    print(f"Exp2 Wilcoxon Tests — {label}")
    print(f"{'='*65}")

    comparisons = [
        ("pair_ce", "clean_ce",        "pair_ce > clean_ce"),
        ("pair_canonical", "clean_ce", "pair_canonical > clean_ce"),
        ("pair_canonical", "pair_ce",  "pair_canonical > pair_ce"),
    ]

    for bb in backbones:
        if bb not in data:
            continue
        print(f"\n[{bb}]  攻击下召回率 (mut_recall)")
        for m_a, m_b, desc in comparisons:
            if m_a not in data[bb] or m_b not in data[bb]:
                continue
            a = data[bb][m_a]["mut_recall"]
            b = data[bb][m_b]["mut_recall"]
            print(wilcoxon_pair(a, b, desc))

        print(f"[{bb}]  干净 F1 (clean_f1，检验是否显著下降)")
        for m_a, m_b, desc in [
            ("pair_ce",        "clean_ce", "pair_ce vs clean_ce"),
            ("pair_canonical", "clean_ce", "pair_canonical vs clean_ce"),
        ]:
            if m_a not in data[bb] or m_b not in data[bb]:
                continue
            a = data[bb][m_a]["clean_f1"]
            b = data[bb][m_b]["clean_f1"]
            # 检验 a < b（干净性能是否下降）
            diff = [x - y for x, y in zip(a, b)]
            mean_diff = np.mean(diff)
            try:
                _, p_less = wilcoxon(a, b, alternative="less")
                sig = "***" if p_less < 0.001 else ("**" if p_less < 0.01 else ("*" if p_less < 0.05 else "ns"))
                print(f"  {desc}: Δ={mean_diff:+.4f}  p_less={p_less:.4f}  {sig}")
            except Exception as e:
                print(f"  {desc}: Δ={mean_diff:+.4f}  ERROR {e}")


def run_ablation_tests(path: Path, label: str) -> None:
    with open(path) as f:
        d = json.load(f)
    rows = d["rows"]

    groups: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["consistency_weight"]].append(r)

    cw_list = sorted(groups.keys())
    baseline_cw = 0.0

    print(f"\n{'='*65}")
    print(f"Ablation Wilcoxon Tests — {label}")
    print(f"  baseline: CW={baseline_cw}")
    print(f"{'='*65}")

    baseline = groups[baseline_cw]
    b_waf = [r["recall_official_wafamole"] for r in baseline]
    b_adv = [r["recall_advsqli"] for r in baseline]

    for cw in cw_list:
        if cw == baseline_cw:
            continue
        rs = groups[cw]
        a_waf = [r["recall_official_wafamole"] for r in rs]
        a_adv = [r["recall_advsqli"] for r in rs]

        results = []
        for metric_name, a_vals, b_vals in [("WAF recall", a_waf, b_waf), ("ADV recall", a_adv, b_adv)]:
            diff = [x - y for x, y in zip(a_vals, b_vals)]
            mean_diff = np.mean(diff)
            try:
                _, p = wilcoxon(a_vals, b_vals, alternative="greater")
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                results.append(f"{metric_name}: Δ={mean_diff:+.4f} p={p:.4f} {sig}")
            except Exception as e:
                results.append(f"{metric_name}: Δ={mean_diff:+.4f} ERROR {e}")
        print(f"  CW={cw:<5}  {results[0]}  |  {results[1]}")


if __name__ == "__main__":
    exp2_official = EXP_DIR / "results_exp2_official.json"
    exp2_advsqli  = EXP_DIR / "results_exp2_advsqli.json"

    if exp2_official.exists():
        run_exp2_tests(exp2_official, "official_wafamole", "Official WAF-A-MoLE")
    if exp2_advsqli.exists():
        run_exp2_tests(exp2_advsqli, "advsqli", "AdvSQLi")

    abl_bilstm  = EXP_DIR / "results_ablation.json"
    abl_textcnn = EXP_DIR / "results_ablation_textcnn.json"
    if abl_bilstm.exists():
        run_ablation_tests(abl_bilstm, "BiLSTM")
    if abl_textcnn.exists():
        run_ablation_tests(abl_textcnn, "TextCNN")
