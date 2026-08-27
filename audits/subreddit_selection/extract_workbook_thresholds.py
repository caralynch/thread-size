#!/usr/bin/env python3
"""Extract aggregate author-threshold values from the recovered XLSX safely."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from analyze_prepared_politics import clean, workbook_sheet_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = workbook_sheet_rows(args.workbook, "Author thresholds")
    output_rows = []
    for row in rows[1:]:
        if len(row) < 2 or not clean(row[0]):
            continue
        percent_single = float(row[1])
        output_rows.append(
            {
                "subreddit": clean(row[0]).lower(),
                "workbook_percent_accounts_with_one_activity": percent_single,
                "workbook_implied_percent_accounts_with_at_least_two_activities": 100.0 - percent_single,
                "sheet_first_recovered_in_git": "2023-08-01T17:24:21+01:00",
                "calculation_code_recovered": False,
                "conclusion_class": "unresolved",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
