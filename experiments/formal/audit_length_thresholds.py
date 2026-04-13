#!/usr/bin/env python3
"""Audit candidate max_len thresholds on raw datasets for the formal protocol."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path


MODSEC_REPO_URL = "https://github.com/pralab/modsec-learn-dataset"


def canonicalize(text: str) -> str:
    return str(text).strip()


def ensure_modsec_repo(repo_dir: Path) -> None:
    if repo_dir.exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", MODSEC_REPO_URL, str(repo_dir)], check=True)


def summarize_lengths(texts: list[str], thresholds: list[int]) -> dict:
    lengths = [len(t) for t in texts]
    lengths_sorted = sorted(lengths)
    total = len(lengths)

    def pct(q: float) -> int:
        if not lengths_sorted:
            return 0
        idx = min(len(lengths_sorted) - 1, max(0, int(round(q * (len(lengths_sorted) - 1)))))
        return lengths_sorted[idx]

    retained = {}
    for th in thresholds:
        kept = sum(1 for x in lengths if x <= th)
        retained[str(th)] = {
            "kept": kept,
            "dropped": total - kept,
            "keep_ratio": kept / total if total else 0.0,
        }

    return {
        "count": total,
        "min": min(lengths) if lengths else 0,
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": max(lengths) if lengths else 0,
        "thresholds": retained,
    }


def load_sqliv_texts(path: Path) -> tuple[list[str], dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    texts = []
    stats = Counter()
    for item in raw:
        text = canonicalize(item.get("pattern", ""))
        typ = canonicalize(item.get("type", "")).lower()
        if typ not in {"valid", "sqli"}:
            stats["removed_other_type"] += 1
            continue
        if not text:
            stats["removed_empty"] += 1
            continue
        texts.append(text)
    return texts, dict(stats)


def load_http_params_texts(path: Path) -> tuple[list[str], dict]:
    texts = []
    stats = Counter()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            attack_type = canonicalize(row.get("attack_type", "")).lower()
            if attack_type not in {"norm", "sqli"}:
                stats["removed_other_type"] += 1
                continue
            text = canonicalize(row.get("payload", ""))
            if not text:
                stats["removed_empty"] += 1
                continue
            texts.append(text)
    return texts, dict(stats)


def load_web_attacks_texts(path: Path) -> tuple[list[str], dict]:
    texts = []
    stats = Counter()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_text = canonicalize(row.get("text_label", ""))
            if label_text not in {"normal", "SQLi"}:
                stats["removed_other_type"] += 1
                continue
            text = canonicalize(row.get("Payload", ""))
            if not text:
                stats["removed_empty"] += 1
                continue
            texts.append(text)
    return texts, dict(stats)


def load_modsec_texts(repo_dir: Path) -> tuple[list[str], dict]:
    ensure_modsec_repo(repo_dir)
    files = [
        repo_dir / "legitimate/openappsec/legitimate_1.json",
        repo_dir / "legitimate/openappsec/legitimate_2.json",
        repo_dir / "legitimate/openappsec/legitimate_3.json",
        repo_dir / "legitimate/openappsec/legitimate_4.json",
        repo_dir / "legitimate/openappsec/legitimate_5.json",
        repo_dir / "legitimate/openappsec/legitimate_6.json",
        repo_dir / "malicious/openappsec/sqli_parsed.json",
        repo_dir / "malicious/httpparams/sqli_parsed.json",
        repo_dir / "malicious/sqlmap/sqli_parsed.json",
    ]
    texts = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        texts.extend(canonicalize(item) for item in data if canonicalize(item))
    return texts, {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--external-dir", default="external")
    parser.add_argument("--thresholds", nargs="+", type=int, default=[128, 192, 260, 320, 512])
    parser.add_argument("--output", default="data/processed/formal_v3/length_audit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    external_dir = Path(args.external_dir)

    datasets = {}

    texts, meta = load_sqliv_texts(raw_dir / "SQLiV3_clean.json")
    datasets["sqliv3_clean_raw_filtered"] = {"pre_filter": meta, **summarize_lengths(texts, args.thresholds)}

    texts, meta = load_sqliv_texts(raw_dir / "SQLiV5.json")
    datasets["sqliv5_raw_filtered"] = {"pre_filter": meta, **summarize_lengths(texts, args.thresholds)}

    texts, meta = load_web_attacks_texts(raw_dir / "web_attacks_long/test.csv")
    datasets["web_attacks_long_test_raw_filtered"] = {"pre_filter": meta, **summarize_lengths(texts, args.thresholds)}

    texts, meta = load_http_params_texts(raw_dir / "http_params_dataset/payload_train.csv")
    datasets["http_params_train_raw_filtered"] = {"pre_filter": meta, **summarize_lengths(texts, args.thresholds)}

    texts, meta = load_http_params_texts(raw_dir / "http_params_dataset/payload_test.csv")
    datasets["http_params_test_raw_filtered"] = {"pre_filter": meta, **summarize_lengths(texts, args.thresholds)}

    texts, meta = load_http_params_texts(raw_dir / "http_params_dataset/payload_test_lexical.csv")
    datasets["http_params_test_lexical_raw_filtered"] = {"pre_filter": meta, **summarize_lengths(texts, args.thresholds)}

    texts, meta = load_modsec_texts(external_dir / "modsec_learn_dataset")
    datasets["modsec_learn_raw_kept_sources"] = {"pre_filter": meta, **summarize_lengths(texts, args.thresholds)}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"thresholds": args.thresholds, "datasets": datasets}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"thresholds": args.thresholds, "datasets": datasets}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
