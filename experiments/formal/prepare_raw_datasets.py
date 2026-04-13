#!/usr/bin/env python3
"""Build processed datasets for the formal experiment suite from raw sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.formal.raw_processing import (
    load_http_params_csv,
    load_modsec_learn_cleaned,
    load_sqliv_json,
    load_web_attacks_csv,
    overlap_audit,
    serialize_rows,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--external-dir", default="external")
    parser.add_argument("--output-dir", default="data/processed/formal_v3")
    parser.add_argument("--max-len", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    external_dir = Path(args.external_dir)
    out_dir = Path(args.output_dir)
    audits_dir = out_dir / "audits"
    datasets_dir = out_dir / "datasets"

    sqliv3_rows, sqliv3_audit = load_sqliv_json(raw_dir / "SQLiV3_clean.json", "sqliv3_clean", args.max_len)
    sqliv5_rows, sqliv5_audit = load_sqliv_json(raw_dir / "SQLiV5.json", "sqliv5", args.max_len)
    web_rows, web_audit = load_web_attacks_csv(raw_dir / "web_attacks_long/test.csv", "test", args.max_len)
    http_train_rows, http_train_audit = load_http_params_csv(
        raw_dir / "http_params_dataset/payload_train.csv", "train", args.max_len
    )
    http_test_rows, http_test_audit = load_http_params_csv(
        raw_dir / "http_params_dataset/payload_test.csv", "test", args.max_len
    )
    http_lex_rows, http_lex_audit = load_http_params_csv(
        raw_dir / "http_params_dataset/payload_test_lexical.csv", "test_lexical", args.max_len
    )
    modsec_rows, modsec_audit = load_modsec_learn_cleaned(
        external_dir / "modsec_learn_dataset",
        {row.text for row in sqliv3_rows},
        args.max_len,
    )

    write_json(datasets_dir / "sqliv3_clean.json", serialize_rows(sqliv3_rows))
    write_json(datasets_dir / "sqliv5.json", serialize_rows(sqliv5_rows))
    write_json(datasets_dir / "modsec_learn_cleaned.json", serialize_rows(modsec_rows))
    write_json(datasets_dir / "web_attacks_long_test.json", serialize_rows(web_rows))
    write_json(datasets_dir / "http_params_train.json", serialize_rows(http_train_rows))
    write_json(datasets_dir / "http_params_test.json", serialize_rows(http_test_rows))
    write_json(datasets_dir / "http_params_test_lexical.json", serialize_rows(http_lex_rows))

    sqliv3_texts = {row.text for row in sqliv3_rows}
    sqliv5_new_sqli_rows = [row for row in sqliv5_rows if row.label == 1 and row.text not in sqliv3_texts]
    write_json(datasets_dir / "sqliv5_new_sqli_only.json", serialize_rows(sqliv5_new_sqli_rows))

    write_json(audits_dir / "sqliv3_clean_audit.json", sqliv3_audit)
    write_json(audits_dir / "sqliv5_audit.json", sqliv5_audit)
    write_json(audits_dir / "modsec_learn_cleaned_audit.json", modsec_audit)
    write_json(audits_dir / "web_attacks_long_test_audit.json", web_audit)
    write_json(audits_dir / "http_params_train_audit.json", http_train_audit)
    write_json(audits_dir / "http_params_test_audit.json", http_test_audit)
    write_json(audits_dir / "http_params_test_lexical_audit.json", http_lex_audit)

    manifest = {
        "protocol": "formal_v3",
        "max_len": args.max_len,
        "datasets": {
            "sqliv3_clean": {
                "path": str(datasets_dir / "sqliv3_clean.json"),
                "audit": sqliv3_audit,
            },
            "sqliv5": {
                "path": str(datasets_dir / "sqliv5.json"),
                "audit": sqliv5_audit,
                "overlap_vs_sqliv3": overlap_audit(sqliv3_rows, sqliv5_rows, "sqliv5"),
                "new_sqli_only_path": str(datasets_dir / "sqliv5_new_sqli_only.json"),
                "new_sqli_only_count": len(sqliv5_new_sqli_rows),
            },
            "modsec_learn_cleaned": {
                "path": str(datasets_dir / "modsec_learn_cleaned.json"),
                "audit": modsec_audit,
                "overlap_vs_sqliv3": overlap_audit(sqliv3_rows, modsec_rows, "modsec_learn_cleaned"),
            },
            "web_attacks_long_test": {
                "path": str(datasets_dir / "web_attacks_long_test.json"),
                "audit": web_audit,
                "overlap_vs_sqliv3": overlap_audit(sqliv3_rows, web_rows, "web_attacks_long_test"),
            },
            "http_params_train": {"path": str(datasets_dir / "http_params_train.json"), "audit": http_train_audit},
            "http_params_test": {"path": str(datasets_dir / "http_params_test.json"), "audit": http_test_audit},
            "http_params_test_lexical": {
                "path": str(datasets_dir / "http_params_test_lexical.json"),
                "audit": http_lex_audit,
            },
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    print("Prepared processed datasets under", out_dir)
    print("SQLiV3:", sqliv3_audit["after_filter"], "rows")
    print("SQLiV5:", sqliv5_audit["after_filter"], "rows;", len(sqliv5_new_sqli_rows), "new SQLi rows")
    print("ModSec-Learn-cleaned:", modsec_audit["after_filter"], "rows")
    print("web-attacks-long:", web_audit["after_filter"], "rows")


if __name__ == "__main__":
    main()
