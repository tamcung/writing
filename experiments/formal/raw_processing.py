#!/usr/bin/env python3
"""Raw-to-processed dataset pipeline for the formal thesis experiments."""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


MODSEC_REPO_URL = "https://github.com/pralab/modsec-learn-dataset"


@dataclass
class ProcessedRow:
    text: str
    label: int
    source: str
    origin: str


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def canonicalize_text(text: str) -> str:
    return str(text).strip()


def dedup_rows(rows: Iterable[ProcessedRow]) -> tuple[list[ProcessedRow], dict[str, int]]:
    by_text: dict[str, set[int]] = defaultdict(set)
    first_seen: dict[str, ProcessedRow] = {}
    duplicate_same_label = 0
    for row in rows:
        if row.text in by_text and row.label in by_text[row.text]:
            duplicate_same_label += 1
        by_text[row.text].add(row.label)
        first_seen.setdefault(row.text, row)

    kept: list[ProcessedRow] = []
    conflicts = 0
    for text, labels in by_text.items():
        if len(labels) > 1:
            conflicts += 1
            continue
        kept.append(first_seen[text])
    kept.sort(key=lambda row: (row.label, row.text))
    return kept, {
        "duplicate_same_label_rows": duplicate_same_label,
        "label_conflicts": conflicts,
    }


def filter_length_and_empty(rows: Iterable[ProcessedRow], max_len: int) -> tuple[list[ProcessedRow], Counter]:
    counts = Counter()
    kept: list[ProcessedRow] = []
    for row in rows:
        if not row.text:
            counts["removed_empty"] += 1
            continue
        if len(row.text) > max_len:
            counts["removed_too_long"] += 1
            continue
        kept.append(row)
    counts["kept_after_basic_filter"] = len(kept)
    return kept, counts


def load_sqliv_json(
    path: Path,
    source_name: str,
    max_len: int,
) -> tuple[list[ProcessedRow], dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[ProcessedRow] = []
    stage = Counter()
    for item in raw:
        text = canonicalize_text(item.get("pattern", ""))
        typ = canonicalize_text(item.get("type", "")).lower()
        if typ not in {"valid", "sqli"}:
            stage["removed_other_type"] += 1
            continue
        label = 1 if typ == "sqli" else 0
        rows.append(ProcessedRow(text=text, label=label, source=source_name, origin=source_name))
    rows, basic = filter_length_and_empty(rows, max_len)
    stage.update(basic)
    rows, dedup = dedup_rows(rows)
    audit = {
        "source": source_name,
        "loaded": len(raw),
        "after_filter": len(rows),
        **dict(stage),
        **dedup,
        "benign": sum(1 for row in rows if row.label == 0),
        "sqli": sum(1 for row in rows if row.label == 1),
    }
    return rows, audit


def load_http_params_csv(path: Path, split_name: str, max_len: int) -> tuple[list[ProcessedRow], dict]:
    rows: list[ProcessedRow] = []
    loaded = 0
    removed_other = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for item in reader:
            loaded += 1
            attack_type = canonicalize_text(item.get("attack_type", "")).lower()
            if attack_type not in {"norm", "sqli"}:
                removed_other += 1
                continue
            rows.append(
                ProcessedRow(
                    text=canonicalize_text(item.get("payload", "")),
                    label=1 if attack_type == "sqli" else 0,
                    source=f"http_params_{split_name}",
                    origin=f"http_params_{split_name}",
                )
            )
    rows, basic = filter_length_and_empty(rows, max_len)
    rows, dedup = dedup_rows(rows)
    audit = {
        "source": f"http_params_{split_name}",
        "loaded": loaded,
        "after_filter": len(rows),
        "removed_other_type": removed_other,
        **dict(basic),
        **dedup,
        "benign": sum(1 for row in rows if row.label == 0),
        "sqli": sum(1 for row in rows if row.label == 1),
    }
    return rows, audit


def load_web_attacks_csv(path: Path, split_name: str, max_len: int) -> tuple[list[ProcessedRow], dict]:
    rows: list[ProcessedRow] = []
    loaded = 0
    removed_other = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for item in reader:
            loaded += 1
            label_text = canonicalize_text(item.get("text_label", ""))
            if label_text not in {"normal", "SQLi"}:
                removed_other += 1
                continue
            rows.append(
                ProcessedRow(
                    text=canonicalize_text(item.get("Payload", "")),
                    label=1 if label_text == "SQLi" else 0,
                    source=f"web_attacks_long_{split_name}",
                    origin=f"web_attacks_long_{split_name}",
                )
            )
    rows, basic = filter_length_and_empty(rows, max_len)
    rows, dedup = dedup_rows(rows)
    audit = {
        "source": f"web_attacks_long_{split_name}",
        "loaded": loaded,
        "after_filter": len(rows),
        "removed_other_type": removed_other,
        **dict(basic),
        **dedup,
        "benign": sum(1 for row in rows if row.label == 0),
        "sqli": sum(1 for row in rows if row.label == 1),
    }
    return rows, audit


def ensure_modsec_repo(repo_dir: Path) -> None:
    if repo_dir.exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", MODSEC_REPO_URL, str(repo_dir)], check=True)


def load_json_list(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [canonicalize_text(item) for item in data]


def load_modsec_learn_cleaned(repo_dir: Path, v3_texts: set[str], max_len: int) -> tuple[list[ProcessedRow], dict]:
    ensure_modsec_repo(repo_dir)
    source_map = {
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
    raw_rows: list[ProcessedRow] = []
    loaded_counts: dict[str, int] = {}
    removed_overlap = Counter()
    for source, path in source_map.items():
        texts = load_json_list(path)
        loaded_counts[source] = len(texts)
        label = 0 if source.startswith("benign_") else 1
        for text in texts:
            raw_rows.append(ProcessedRow(text=text, label=label, source="modsec_learn_cleaned", origin=source))

    raw_rows, basic = filter_length_and_empty(raw_rows, max_len)
    overlap_filtered: list[ProcessedRow] = []
    for row in raw_rows:
        if row.text in v3_texts:
            removed_overlap[row.origin] += 1
            continue
        overlap_filtered.append(row)

    rows, dedup = dedup_rows(overlap_filtered)
    audit = {
        "source": "modsec_learn_cleaned",
        "repo_url": MODSEC_REPO_URL,
        "loaded_source_counts": loaded_counts,
        "after_filter": len(rows),
        **dict(basic),
        **dedup,
        "removed_overlap_with_sqliv3": dict(removed_overlap),
        "exact_overlap_with_sqliv3_after_cleaning": sum(1 for row in rows if row.text in v3_texts),
        "benign": sum(1 for row in rows if row.label == 0),
        "sqli": sum(1 for row in rows if row.label == 1),
        "kept_origin_counts": dict(Counter(row.origin for row in rows)),
    }
    return rows, audit


def overlap_audit(base_rows: list[ProcessedRow], other_rows: list[ProcessedRow], name: str) -> dict:
    base_texts = {row.text for row in base_rows}
    other_texts = {row.text for row in other_rows}
    overlap = base_texts & other_texts
    other_sqli = {row.text for row in other_rows if row.label == 1}
    other_benign = {row.text for row in other_rows if row.label == 0}
    return {
        "name": name,
        "other_total_unique": len(other_texts),
        "exact_overlap_with_sqliv3": len(overlap),
        "overlap_ratio": len(overlap) / len(other_texts) if other_texts else 0.0,
        "new_sqli_vs_sqliv3": len(other_sqli - base_texts),
        "new_benign_vs_sqliv3": len(other_benign - base_texts),
    }


def serialize_rows(rows: list[ProcessedRow]) -> list[dict]:
    return [asdict(row) for row in rows]
