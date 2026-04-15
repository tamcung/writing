#!/usr/bin/env python3
"""Build the cleaned ModSec-Learn external validation dataset.

The cleaned variant follows the thesis protocol:
- keep legitimate samples from openappsec;
- keep malicious samples from openappsec / httpparams / sqlmap;
- drop the sqli_kaggle branch to reduce source overlap with SQLiV3_clean;
- remove empty strings and overlong samples;
- remove exact overlaps with SQLiV3_clean after SQLiV3 filtering;
- drop exact-text label conflicts;
- emit a JSON file compatible with load_payload_data().
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.consistency_sqli_experiment import load_payload_data  # noqa: E402


MODSEC_REPO_URL = "https://github.com/pralab/modsec-learn-dataset"


def run_git_clone(repo_url: str, repo_dir: Path) -> None:
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        return
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
        check=True,
        cwd=ROOT,
    )


def load_json_list(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON list")
    return [str(item) for item in data]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-dir",
        default="external/modsec_learn_dataset",
        help="Local clone of the public ModSec-Learn dataset repository.",
    )
    parser.add_argument(
        "--train-data",
        default="data/raw/SQLiV3_clean.json",
        help="Primary training source used to remove exact overlaps.",
    )
    parser.add_argument(
        "--output-data",
        default="data/raw/ModSec-Learn-cleaned.json",
        help="Cleaned dataset in SQLiV3-style JSON format.",
    )
    parser.add_argument(
        "--output-audit",
        default="data/raw/ModSec-Learn-cleaned_audit.json",
        help="Audit report with source counts and filtering statistics.",
    )
    parser.add_argument("--max-len", type=int, default=260)
    return parser.parse_args()


def build_records(repo_dir: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    source_to_path = {
        "benign_openappsec_1": repo_dir / "legitimate/openappsec/legitimate_1.json",
        "benign_openappsec_2": repo_dir / "legitimate/openappsec/legitimate_2.json",
        "benign_openappsec_3": repo_dir / "legitimate/openappsec/legitimate_3.json",
        "benign_openappsec_4": repo_dir / "legitimate/openappsec/legitimate_4.json",
        "benign_openappsec_5": repo_dir / "legitimate/openappsec/legitimate_5.json",
        "benign_openappsec_6": repo_dir / "legitimate/openappsec/legitimate_6.json",
        "sqli_openappsec": repo_dir / "malicious/openappsec/sqli_parsed.json",
        "sqli_httpparams": repo_dir / "malicious/httpparams/sqli_parsed.json",
        "sqli_sqlmap": repo_dir / "malicious/sqlmap/sqli_parsed.json",
    }
    loaded_counts: dict[str, int] = {}
    records: list[dict[str, str]] = []

    for source, path in source_to_path.items():
        rows = load_json_list(path)
        loaded_counts[source] = len(rows)
        label = "valid" if source.startswith("benign_") else "sqli"
        for text in rows:
            records.append({"pattern": text, "type": label, "source": source})

    return records, loaded_counts


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir)
    output_data = Path(args.output_data)
    output_audit = Path(args.output_audit)

    run_git_clone(MODSEC_REPO_URL, repo_dir)

    train_texts, train_labels = load_payload_data(Path(args.train_data), max_len=args.max_len)
    del train_labels
    v3_texts = set(train_texts)

    raw_records, loaded_counts = build_records(repo_dir)

    stage_counts = defaultdict(Counter)
    by_text: dict[str, list[dict[str, str]]] = defaultdict(list)

    for record in raw_records:
        source = record["source"]
        stage_counts["loaded"][source] += 1
        text = record["pattern"].strip()
        if not text:
            stage_counts["removed_empty"][source] += 1
            continue
        if len(text) > args.max_len:
            stage_counts["removed_too_long"][source] += 1
            continue
        if text in v3_texts:
            stage_counts["removed_overlap_v3"][source] += 1
            continue
        stage_counts["after_basic_filter"][source] += 1
        by_text[text].append({"pattern": text, "type": record["type"], "source": source})

    cleaned_records: list[dict[str, str]] = []
    conflict_count = 0
    duplicate_same_label = 0
    source_kept = Counter()
    overlap_after_clean = 0

    for text, rows in by_text.items():
        labels = {row["type"] for row in rows}
        if text in v3_texts:
            overlap_after_clean += 1
            continue
        if len(labels) > 1:
            conflict_count += 1
            continue
        if len(rows) > 1:
            duplicate_same_label += len(rows) - 1
        first = rows[0]
        cleaned_records.append(
            {
                "pattern": text,
                "type": first["type"],
                "source": first["source"],
            }
        )
        source_kept[first["source"]] += 1

    cleaned_records.sort(key=lambda row: (row["type"], row["pattern"]))
    output_data.parent.mkdir(parents=True, exist_ok=True)
    output_data.write_text(json.dumps(cleaned_records, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "repo_url": MODSEC_REPO_URL,
        "repo_dir": str(repo_dir),
        "train_data": args.train_data,
        "max_len": args.max_len,
        "loaded_source_counts": dict(loaded_counts),
        "stage_counts": {stage: dict(counter) for stage, counter in stage_counts.items()},
        "final_source_counts": dict(source_kept),
        "final_total": len(cleaned_records),
        "final_benign": sum(1 for row in cleaned_records if row["type"] == "valid"),
        "final_sqli": sum(1 for row in cleaned_records if row["type"] == "sqli"),
        "dropped_label_conflicts": conflict_count,
        "collapsed_duplicate_same_label_rows": duplicate_same_label,
        "exact_overlap_with_sqliv3_after_cleaning": overlap_after_clean,
    }
    output_audit.parent.mkdir(parents=True, exist_ok=True)
    output_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "Built ModSec-Learn-cleaned total={} benign={} sqli={} overlap_after_clean={}".format(
            audit["final_total"],
            audit["final_benign"],
            audit["final_sqli"],
            audit["exact_overlap_with_sqliv3_after_cleaning"],
        )
    )
    print("Final source counts:", json.dumps(audit["final_source_counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
