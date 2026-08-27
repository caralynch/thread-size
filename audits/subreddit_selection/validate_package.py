#!/usr/bin/env python3
"""Validate the aggregate audit package without reopening source data."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    checks: list[dict[str, str]] = []

    def check(name: str, passed: bool, observed: object, expected: object) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "observed": str(observed),
                "expected": str(expected),
            }
        )

    source_checks = read_csv(OUTPUTS / "validation_checks.csv")
    check(
        "all_source_level_checks_pass",
        all(row["status"] == "PASS" for row in source_checks),
        sum(row["status"] == "PASS" for row in source_checks),
        len(source_checks),
    )

    monthly = read_csv(OUTPUTS / "monthly_submissions.csv")
    coverage = {row["subreddit"]: row for row in read_csv(OUTPUTS / "dataset_coverage.csv")}
    monthly_sums: dict[str, int] = {}
    for row in monthly:
        monthly_sums[row["subreddit"]] = monthly_sums.get(row["subreddit"], 0) + int(row["root_submissions"])
    check(
        "monthly_root_sums_match_coverage",
        all(monthly_sums[key] == int(value["strict_root_submissions"]) for key, value in coverage.items()),
        monthly_sums,
        {key: int(value["strict_root_submissions"]) for key, value in coverage.items()},
    )

    format_claims = read_csv(OUTPUTS / "submission_format_claims.csv")
    check(
        "all_format_rules_support_nonmedia_majority",
        all(row["predominantly_not_image_or_video_even_if_unknown_is_media"] == "True" for row in format_claims),
        sum(row["predominantly_not_image_or_video_even_if_unknown_is_media"] == "True" for row in format_claims),
        len(format_claims),
    )

    repeats = read_csv(OUTPUTS / "repeat_account_comparison.csv")
    observed_repeats = [row for row in repeats if row["observed_percent_repeat_accounts"]]
    check(
        "all_observed_repeat_variants_exceed_50_percent",
        all(float(row["observed_percent_repeat_accounts"]) > 50 for row in observed_repeats),
        min(float(row["observed_percent_repeat_accounts"]) for row in observed_repeats),
        "> 50",
    )
    check(
        "no_observed_repeat_variant_claims_workbook_reproduction",
        all(row["reproduces_workbook_within_0_01pp"] == "False" for row in observed_repeats),
        sum(row["reproduces_workbook_within_0_01pp"] == "False" for row in observed_repeats),
        len(observed_repeats),
    )

    inventory = read_csv(ROOT / "source_inventory.csv")
    check("source_inventory_has_candidate_coverage", len(inventory) >= 20, len(inventory), ">= 20")
    check(
        "source_inventory_rows_have_consistent_schema",
        all(len(row) == len(inventory[0]) for row in inventory),
        len(inventory),
        len(inventory),
    )

    prohibited_headers = {"id", "thread_id", "parent", "body", "subject", "author", "url", "domain"}
    metric_files = [
        path
        for path in OUTPUTS.glob("*.csv")
        if path.name not in {"source_inventory_computed.csv", "package_validation.csv"}
    ]
    exposed = set()
    for path in metric_files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            exposed.update((csv.DictReader(handle).fieldnames or []))
    check(
        "aggregate_outputs_have_no_raw_record_columns",
        not (exposed & prohibited_headers),
        sorted(exposed & prohibited_headers),
        [],
    )

    artifact = json.loads((ROOT / "artifact.json").read_text(encoding="utf-8"))
    artifact_text = json.dumps(artifact)
    check("artifact_uses_relative_source_paths", "C:\\\\Users\\\\" not in artifact_text and "Z:\\\\" not in artifact_text, "relative", "relative")
    check("portable_report_exists", (ROOT / "report.html").is_file(), (ROOT / "report.html").is_file(), True)

    output = OUTPUTS / "package_validation.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(checks)
    failed = [row for row in checks if row["status"] == "FAIL"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
