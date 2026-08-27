#!/usr/bin/env python3
"""Independent structural validation of the aggregate endpoint-audit package."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
HORIZONS = [1, 6, 12, 24, 48, 72, 168]
SENSITIVITY = [None, 24, 48, 72, 168]


def rows(name: str) -> list[dict[str, str]]:
    with (OUTPUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: str) -> float:
    return float(value) if value not in ("", None) else math.nan


checks: list[dict[str, object]] = []


def check(name: str, passed: bool, observed: object, expected: object, severity: str = "critical") -> None:
    checks.append(
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "expected": expected,
            "severity": severity,
        }
    )


def main() -> int:
    primary_checks = rows("validation_checks.csv")
    check(
        "primary audit has no non-PASS validation checks",
        all(row["status"] == "PASS" for row in primary_checks),
        sum(row["status"] != "PASS" for row in primary_checks),
        0,
    )

    inventory = rows("source_inventory.csv")
    file_sources = [
        row
        for row in inventory
        if row["exists"] == "True" and not row["path_or_url"].startswith("http")
    ]
    check(
        "all existing file sources have SHA-256 checksums",
        all(len(row["sha256"]) == 64 for row in file_sources),
        sum(len(row["sha256"]) == 64 for row in file_sources),
        len(file_sources),
    )
    run_record = json.loads((OUTPUT / "run_record.json").read_text(encoding="utf-8"))
    check("final run did not skip hashes", run_record["hashes_skipped"] is False, run_record["hashes_skipped"], False)
    check("no model fitting occurred", run_record["models_loaded_or_fitted"] is False, run_record["models_loaded_or_fitted"], False)
    check("predictions were not regenerated", run_record["predictions_regenerated"] is False, run_record["predictions_regenerated"], False)

    populations = rows("population_reconciliation.csv")
    for row in populations:
        final_n = int(row["final_cleaned_roots"])
        train_n = int(row["reconstructed_train_roots"])
        heldout_n = int(row["reconstructed_heldout_roots"])
        check(
            f"{row['subreddit']}: population partitions exactly",
            train_n + heldout_n == final_n and train_n == int(final_n * 0.8),
            f"{train_n}+{heldout_n}={train_n + heldout_n}",
            f"floor(0.8*{final_n}) plus remainder",
        )

    exposure = rows("endpoint_exposure.csv")
    exposure_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in exposure:
        exposure_groups[(row["subreddit"], row["classification_stage"], row["observed_class"])].append(row)
    for key, group in exposure_groups.items():
        group.sort(key=lambda row: int(row["horizon_hours"]))
        hs = [int(row["horizon_hours"]) for row in group]
        limited = [int(row["limited_followup_roots"]) for row in group]
        totals = [int(row["heldout_roots"]) for row in group]
        pcts_ok = all(
            abs(number(row["limited_followup_pct"]) - 100 * int(row["limited_followup_roots"]) / int(row["heldout_roots"])) < 1e-9
            for row in group
            if int(row["heldout_roots"])
        )
        check(f"exposure horizons complete for {key}", hs == HORIZONS, hs, HORIZONS)
        check(f"exposure counts monotone for {key}", limited == sorted(limited), limited, "nondecreasing")
        check(f"exposure denominators fixed for {key}", len(set(totals)) == 1, totals, "one denominator")
        check(f"exposure percentages reconcile for {key}", pcts_ok, pcts_ok, True)

    performance = rows("sensitivity_performance.csv")
    perf_key = {(r["subreddit"], r["stage"], r["exclusion"]): r for r in performance}
    perf_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in performance:
        perf_groups[(row["subreddit"], row["stage"])].append(row)
    exclusion_order = {"full_heldout": 0, "at_least_24h": 1, "at_least_48h": 2, "at_least_72h": 3, "at_least_168h": 4}
    for key, group in perf_groups.items():
        group.sort(key=lambda row: exclusion_order[row["exclusion"]])
        retained = [int(row["retained_n"]) for row in group]
        excluded = [int(row["excluded_n"]) for row in group]
        full_n = retained[0]
        check(f"sensitivity subsets complete for {key}", len(group) == 5, len(group), 5)
        check(f"sensitivity retained counts monotone for {key}", retained == sorted(retained, reverse=True), retained, "nonincreasing")
        check(f"sensitivity counts partition for {key}", all(r + e == full_n for r, e in zip(retained, excluded)), list(zip(retained, excluded)), full_n)

    cm_rows = rows("sensitivity_confusion_matrices.csv")
    cm_sums: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in cm_rows:
        cm_sums[(row["subreddit"], row["stage"], row["exclusion"])] += int(row["count"])
    check(
        "all confusion-matrix counts sum to retained N",
        all(total == int(perf_key[key]["retained_n"]) for key, total in cm_sums.items()),
        sum(total != int(perf_key[key]["retained_n"]) for key, total in cm_sums.items()),
        0,
    )

    class_rows = rows("sensitivity_class_metrics.csv")
    class_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in class_rows:
        class_groups[(row["subreddit"], row["stage"], row["exclusion"])].append(row)
    class_ok = True
    for key, group in class_groups.items():
        retained = int(perf_key[key]["retained_n"])
        class_ok &= sum(int(row["true_count"]) for row in group) == retained
        if retained:
            class_ok &= abs(sum(number(row["prevalence_pct"]) for row in group) - 100) < 1e-8
    check("class counts and prevalence reconcile", class_ok, class_ok, True)

    timing = rows("reply_timing_comment_cumulative.csv")
    timing_ok = all(
        int(row["comments_within_horizon"]) <= int(row["observed_descendant_comments"])
        and 0 <= number(row["cumulative_comment_pct"]) <= 100
        for row in timing
    )
    check("reply-timing numerators and percentages are bounded", timing_ok, timing_ok, True)

    raw = rows("raw_archive_diagnostics.csv")
    raw_ok = all(int(row["raw_roots"]) + int(row["raw_comments"]) == int(row["raw_rows"]) for row in raw)
    check("raw row types partition exactly", raw_ok, raw_ok, True)
    raw_checks = rows("raw_archive_validation.csv")
    check(
        "raw validation has no FAIL status",
        all(row["status"] != "FAIL" for row in raw_checks),
        sum(row["status"] == "FAIL" for row in raw_checks),
        0,
    )

    forbidden = {"id", "thread_id", "author", "body", "subject", "text", "account"}
    privacy_ok = True
    inspected = 0
    for path in OUTPUT.glob("*.csv"):
        if path.name == "package_validation.csv":
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        inspected += 1
        privacy_ok &= not bool(forbidden.intersection(header))
    check("aggregate CSV headers contain no Reddit identifier/text fields", privacy_ok, inspected, "all aggregate CSV files")

    with (OUTPUT / "package_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    failures = [row for row in checks if row["status"] == "FAIL"]
    print(f"{len(checks)} checks; {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
