#!/usr/bin/env python3
"""Join aggregate repeat-account results to recovered workbook percentages."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--prepared-politics", required=True, type=Path)
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    workbook = {
        row["subreddit"]: float(row["workbook_implied_percent_accounts_with_at_least_two_activities"])
        for row in read_rows(args.workbook)
    }
    output_rows = []
    sources = (
        ("raw_export", read_rows(args.raw)),
        ("prepared_study_csv", read_rows(args.prepared_politics)),
    )
    for source_population, rows in sources:
        for row in rows:
            subreddit = row.get("subreddit") or "politics"
            observed_text = row.get("percent_repeat_accounts", "")
            observed = float(observed_text) if observed_text else None
            recovered = workbook[subreddit]
            difference = observed - recovered if observed is not None else None
            output_rows.append(
                {
                    "subreddit": subreddit,
                    "source_population": source_population,
                    "variant": row["variant"],
                    "observed_percent_repeat_accounts": observed_text,
                    "workbook_implied_percent_repeat_accounts": recovered,
                    "difference_percentage_points": difference if difference is not None else "",
                    "reproduces_workbook_within_0_01pp": abs(difference) <= 0.01 if difference is not None else "unresolved",
                    "published_over_50_percent_holds": observed > 50 if observed is not None else "unresolved",
                    "conclusion_class": row["conclusion_class"],
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
