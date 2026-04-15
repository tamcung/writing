#!/usr/bin/env python3
"""Prepare a decoded ModSec-Learn parameter-level dataset as a new formal source."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

MODSEC_REPO_URL = "https://github.com/pralab/modsec-learn-dataset"
PARAM_SEPARATOR = " "


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_modsec_repo(repo_dir: Path) -> None:
    if repo_dir.exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", MODSEC_REPO_URL, str(repo_dir)], check=True)


def load_json_list(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [str(item).strip() for item in data]


def decode_query_text(text: str, passes: int) -> str:
    out = str(text).strip()
    for _ in range(max(0, passes)):
        decoded = urllib.parse.unquote_plus(out)
        if decoded == out:
            break
        out = decoded.strip()
    return out


def dedup_dict_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    by_text: dict[str, set[int]] = defaultdict(set)
    first_seen: dict[str, dict] = {}
    duplicate_same_label = 0
    for row in rows:
        text = str(row["text"])
        label = int(row["label"])
        if text in by_text and label in by_text[text]:
            duplicate_same_label += 1
        by_text[text].add(label)
        first_seen.setdefault(text, row)

    kept: list[dict] = []
    conflicts = 0
    for text, labels in by_text.items():
        if len(labels) > 1:
            conflicts += 1
            continue
        kept.append(first_seen[text])
    kept.sort(key=lambda row: (int(row["label"]), str(row["text"])))
    return kept, {
        "duplicate_same_label_rows": duplicate_same_label,
        "label_conflicts": conflicts,
    }


def modsec_source_map(repo_dir: Path, include_kaggle: bool) -> dict[str, Path]:
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
    if include_kaggle:
        source_map["sqli_kaggle"] = repo_dir / "malicious/sqli_kaggle/sqli_parsed.json"
    return source_map


def prepare_modsec_decoded(repo_dir: Path, max_len: int, decode_passes: int, include_kaggle: bool) -> tuple[list[dict], dict]:
    ensure_modsec_repo(repo_dir)
    source_map = modsec_source_map(repo_dir, include_kaggle)

    loaded_counts: dict[str, int] = {}
    decode_changed_counts = Counter()
    stage = Counter()
    filter_counts_by_origin: dict[str, Counter] = defaultdict(Counter)
    filter_counts_by_label: dict[str, Counter] = defaultdict(Counter)
    raw_rows: list[dict] = []

    for origin, path in source_map.items():
        texts = load_json_list(path)
        loaded_counts[origin] = len(texts)
        label = 0 if origin.startswith("benign_") else 1
        label_name = "benign" if label == 0 else "sqli"
        for raw_text in texts:
            filter_counts_by_origin[origin]["loaded"] += 1
            filter_counts_by_label[label_name]["loaded"] += 1
            decoded_text = decode_query_text(raw_text, decode_passes)
            if decoded_text != raw_text.strip():
                decode_changed_counts[origin] += 1
                filter_counts_by_origin[origin]["decode_changed"] += 1
                filter_counts_by_label[label_name]["decode_changed"] += 1
            if not decoded_text:
                stage["removed_empty_after_decode"] += 1
                filter_counts_by_origin[origin]["removed_empty_after_decode"] += 1
                filter_counts_by_label[label_name]["removed_empty_after_decode"] += 1
                continue
            if len(decoded_text) > max_len:
                stage["removed_too_long_after_decode"] += 1
                filter_counts_by_origin[origin]["removed_too_long_after_decode"] += 1
                filter_counts_by_label[label_name]["removed_too_long_after_decode"] += 1
                continue
            filter_counts_by_origin[origin]["kept_after_basic_filter"] += 1
            filter_counts_by_label[label_name]["kept_after_basic_filter"] += 1
            raw_rows.append(
                {
                    "text": decoded_text,
                    "label": label,
                    "source": "modsec_learn_decoded",
                    "origin": origin,
                    "decode_passes": decode_passes,
                    "decoded_changed": decoded_text != raw_text.strip(),
                }
            )

    stage["kept_after_basic_filter"] = len(raw_rows)
    rows, dedup = dedup_dict_rows(raw_rows)
    audit = {
        "source": "modsec_learn_decoded",
        "repo_url": MODSEC_REPO_URL,
        "max_len": max_len,
        "decode_passes": decode_passes,
        "include_kaggle": include_kaggle,
        "loaded_source_counts": loaded_counts,
        "decode_changed_counts": dict(decode_changed_counts),
        "after_filter": len(rows),
        **dict(stage),
        **dedup,
        "benign": sum(1 for row in rows if int(row["label"]) == 0),
        "sqli": sum(1 for row in rows if int(row["label"]) == 1),
        "kept_origin_counts": dict(Counter(str(row["origin"]) for row in rows)),
        "filter_counts_by_origin": {
            origin: dict(counts) for origin, counts in filter_counts_by_origin.items()
        },
        "filter_counts_by_label": {
            label_name: dict(counts) for label_name, counts in filter_counts_by_label.items()
        },
    }
    return rows, audit


def _stable_offset(raw_text: str, size: int) -> int:
    if size <= 0:
        return 0
    digest = hashlib.sha256(raw_text.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:4], "big") % size


def extract_sqli_payload_value(raw_text: str, decode_passes: int) -> str:
    raw = str(raw_text).strip()
    value = raw.split("=", 1)[1] if "=" in raw else raw
    return decode_query_text(value, decode_passes)


def extract_benign_query_values(raw_text: str, decode_passes: int) -> list[str]:
    raw = str(raw_text).strip()
    pairs = urllib.parse.parse_qsl(raw, keep_blank_values=True, strict_parsing=False)
    values = [
        decode_query_text(value, max(0, decode_passes - 1))
        for _, value in pairs
    ]
    values = [value for value in values if value]
    if values:
        return values

    decoded = decode_query_text(raw, decode_passes)
    if "=" in decoded:
        decoded = decoded.split("=", 1)[1].strip()
    return [decoded] if decoded else []


def _join_values(values: list[str]) -> str:
    return PARAM_SEPARATOR.join(value for value in values if value)


def build_benign_value_windows(
    raw_text: str,
    values: list[str],
    min_len: int,
    target_len: int,
    max_len: int,
    max_windows: int,
) -> list[tuple[str, list[str]]]:
    if not values:
        return []

    starts = [_stable_offset(raw_text, len(values)), 0]
    seen_starts: set[int] = set()
    windows: list[tuple[str, list[str]]] = []
    for start in starts:
        if start in seen_starts:
            continue
        seen_starts.add(start)
        pieces: list[str] = []
        for value in values[start:]:
            candidate = _join_values([*pieces, value])
            if len(candidate) > max_len:
                if not pieces:
                    continue
                break
            pieces.append(value)
            if len(candidate) >= target_len:
                break
        window = _join_values(pieces)
        if min_len <= len(window) <= max_len and all(window != existing for existing, _ in windows):
            windows.append((window, list(pieces)))
        if len(windows) >= max_windows:
            break
    return windows


def prepare_modsec_value_windows(
    repo_dir: Path,
    max_len: int,
    decode_passes: int,
    include_kaggle: bool,
    benign_min_len: int,
    benign_target_len: int,
    benign_windows_per_request: int,
) -> tuple[list[dict], dict]:
    ensure_modsec_repo(repo_dir)
    source_map = modsec_source_map(repo_dir, include_kaggle)
    loaded_counts: dict[str, int] = {}
    stage = Counter()
    counts_by_origin: dict[str, Counter] = defaultdict(Counter)
    counts_by_label: dict[str, Counter] = defaultdict(Counter)
    raw_rows: list[dict] = []

    for origin, path in source_map.items():
        texts = load_json_list(path)
        loaded_counts[origin] = len(texts)
        label = 0 if origin.startswith("benign_") else 1
        label_name = "benign" if label == 0 else "sqli"
        for raw_i, raw_text in enumerate(texts):
            counts_by_origin[origin]["loaded"] += 1
            counts_by_label[label_name]["loaded"] += 1

            if label == 1:
                payload = extract_sqli_payload_value(raw_text, decode_passes)
                if not payload:
                    stage["removed_empty_value"] += 1
                    counts_by_origin[origin]["removed_empty_value"] += 1
                    counts_by_label[label_name]["removed_empty_value"] += 1
                    continue
                if len(payload) > max_len:
                    stage["removed_too_long_value"] += 1
                    counts_by_origin[origin]["removed_too_long_value"] += 1
                    counts_by_label[label_name]["removed_too_long_value"] += 1
                    continue
                raw_rows.append(
                    {
                        "text": payload,
                        "label": label,
                        "source": "modsec_learn_value_windows",
                        "origin": origin,
                        "input_view": "sqli_payload_value",
                        "raw_item_index": raw_i,
                        "value_count": 1,
                        "value_parts": [payload],
                        "decode_passes": decode_passes,
                    }
                )
                counts_by_origin[origin]["kept_value_rows"] += 1
                counts_by_label[label_name]["kept_value_rows"] += 1
                continue

            values = extract_benign_query_values(raw_text, decode_passes)
            windows = build_benign_value_windows(
                raw_text=raw_text,
                values=values,
                min_len=benign_min_len,
                target_len=benign_target_len,
                max_len=max_len,
                max_windows=max(1, benign_windows_per_request),
            )
            if not windows:
                stage["removed_no_benign_window"] += 1
                counts_by_origin[origin]["removed_no_benign_window"] += 1
                counts_by_label[label_name]["removed_no_benign_window"] += 1
                continue
            for window_i, (window, window_values) in enumerate(windows):
                raw_rows.append(
                    {
                        "text": window,
                        "label": label,
                        "source": "modsec_learn_value_windows",
                        "origin": origin,
                        "input_view": "benign_value_window",
                        "raw_item_index": raw_i,
                        "window_index": window_i,
                        "value_count": len(window_values),
                        "value_parts": window_values,
                        "decode_passes": decode_passes,
                    }
                )
                counts_by_origin[origin]["kept_value_rows"] += 1
                counts_by_label[label_name]["kept_value_rows"] += 1

    stage["kept_before_dedup"] = len(raw_rows)
    rows, dedup = dedup_dict_rows(raw_rows)
    audit = {
        "source": "modsec_learn_value_windows",
        "repo_url": MODSEC_REPO_URL,
        "input_policy": "value_only_length_matched_windows",
        "max_len": max_len,
        "decode_passes": decode_passes,
        "include_kaggle": include_kaggle,
        "param_separator": PARAM_SEPARATOR,
        "benign_min_len": benign_min_len,
        "benign_target_len": benign_target_len,
        "benign_windows_per_request": benign_windows_per_request,
        "loaded_source_counts": loaded_counts,
        "after_filter": len(rows),
        **dict(stage),
        **dedup,
        "benign": sum(1 for row in rows if int(row["label"]) == 0),
        "sqli": sum(1 for row in rows if int(row["label"]) == 1),
        "kept_origin_counts": dict(Counter(str(row["origin"]) for row in rows)),
        "filter_counts_by_origin": {
            origin: dict(counts) for origin, counts in counts_by_origin.items()
        },
        "filter_counts_by_label": {
            label_name: dict(counts) for label_name, counts in counts_by_label.items()
        },
    }
    return rows, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-dir", default="external")
    parser.add_argument("--output-dir", default="data/processed/formal_modsec_decoded")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--decode-passes", type=int, default=1)
    parser.add_argument("--include-kaggle", action="store_true")
    parser.add_argument("--benign-min-len", type=int, default=32)
    parser.add_argument("--benign-target-len", type=int, default=96)
    parser.add_argument("--benign-windows-per-request", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    datasets_dir = out_dir / "datasets"
    audits_dir = out_dir / "audits"
    rows, audit = prepare_modsec_decoded(
        repo_dir=Path(args.external_dir) / "modsec_learn_dataset",
        max_len=args.max_len,
        decode_passes=args.decode_passes,
        include_kaggle=args.include_kaggle,
    )
    value_rows, value_audit = prepare_modsec_value_windows(
        repo_dir=Path(args.external_dir) / "modsec_learn_dataset",
        max_len=args.max_len,
        decode_passes=args.decode_passes,
        include_kaggle=args.include_kaggle,
        benign_min_len=args.benign_min_len,
        benign_target_len=args.benign_target_len,
        benign_windows_per_request=args.benign_windows_per_request,
    )

    dataset_path = datasets_dir / "modsec_learn_decoded.json"
    value_dataset_path = datasets_dir / "modsec_learn_value_windows.json"
    write_json(dataset_path, rows)
    write_json(value_dataset_path, value_rows)
    write_json(audits_dir / "modsec_learn_decoded_audit.json", audit)
    write_json(audits_dir / "modsec_learn_value_windows_audit.json", value_audit)
    manifest = {
        "protocol": "formal_modsec_decoded",
        "max_len": args.max_len,
        "decode_passes": args.decode_passes,
        "primary_dataset": "modsec_learn_value_windows",
        "datasets": {
            "modsec_learn_decoded": {
                "path": str(dataset_path),
                "audit": audit,
            },
            "modsec_learn_value_windows": {
                "path": str(value_dataset_path),
                "audit": value_audit,
            },
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    print("Prepared decoded ModSec-Learn dataset under", out_dir)
    print(
        "ModSec-Learn-decoded:",
        audit["after_filter"],
        "rows; benign",
        audit["benign"],
        "sqli",
        audit["sqli"],
    )
    print("decode_changed_counts:", audit["decode_changed_counts"])
    print(
        "ModSec-Learn-value-windows:",
        value_audit["after_filter"],
        "rows; benign",
        value_audit["benign"],
        "sqli",
        value_audit["sqli"],
    )


if __name__ == "__main__":
    main()
