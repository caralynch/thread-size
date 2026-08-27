#!/usr/bin/env python3
"""Aggregate endpoint diagnostics for the three published raw Reddit archives.

The program streams each ZIP member without extracting it and writes no Reddit
text, account name, or post/comment identifier.  It is deliberately separate
from the final-cleaned-population audit because raw-archive diagnostics and
final-analysis estimates answer different provenance questions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


CONFIG = {
    "conspiracy": {
        "endpoint": "2022-10-31T00:00:00Z",
        "filename": "conspiracy_oct2022.zip",
        "member": "conspiracy_oct2022.csv",
        "md5": "0fc82cae91d2c78b6d1581f073dbfc04",
        "preliminary_root_pct": 3.29,
        "preliminary_comment_pct": 94.0,
    },
    "crypto": {
        "endpoint": "2022-10-31T00:00:00Z",
        "filename": "cryptocurrency_oct2022.zip",
        "member": "cryptocurrency_oct2022.csv",
        "md5": "7128dcd7784077e77716b835cc760bbd",
        "preliminary_root_pct": 2.88,
        "preliminary_comment_pct": 97.8,
    },
    "politics": {
        "endpoint": "2020-11-20T00:00:00Z",
        "filename": "politics_nov2020.zip",
        "member": "politics_nov2020.csv",
        "md5": "f7f8545542840b5353c4b1da230289cb",
        "preliminary_root_pct": 2.20,
        "preliminary_comment_pct": 98.8,
    },
}


def file_hash(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def endpoint_epoch(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def parse_archive_spec(value: str) -> tuple[str, Path]:
    try:
        subreddit, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use subreddit=path") from exc
    if subreddit not in CONFIG:
        raise argparse.ArgumentTypeError(f"Unknown subreddit: {subreddit}")
    return subreddit, Path(path)


def inspect_archive(subreddit: str, path: Path) -> tuple[dict, list[dict]]:
    config = CONFIG[subreddit]
    observed_md5 = file_hash(path)
    checks: list[dict] = [
        {
            "subreddit": subreddit,
            "check": "published_archive_md5",
            "status": "PASS" if observed_md5 == config["md5"] else "FAIL",
            "observed": observed_md5,
            "expected": config["md5"],
        }
    ]
    if observed_md5 != config["md5"]:
        raise RuntimeError(f"{subreddit}: raw archive MD5 does not match Zenodo")

    endpoint = endpoint_epoch(config["endpoint"])
    one_day = 24 * 60 * 60
    roots: dict[str, int] = {}
    root_count = roots_last_24h = duplicate_roots = 0
    comment_count = matched_comments = comments_without_matched_root = 0
    eligible_matched_comments = eligible_comments_within_24h = 0
    negative_delays = timestamp_text_mismatches = nonmonotonic_rows = 0
    total_rows = 0
    min_epoch: int | None = None
    max_epoch: int | None = None
    previous_epoch: int | None = None

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        checks.append(
            {
                "subreddit": subreddit,
                "check": "expected_csv_member",
                "status": "PASS" if config["member"] in names else "FAIL",
                "observed": "|".join(names),
                "expected": config["member"],
            }
        )
        if config["member"] not in names:
            raise RuntimeError(f"{subreddit}: expected CSV member is absent")
        with archive.open(config["member"]) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
                reader = csv.DictReader(text)
                required = {"thread_id", "id", "timestamp", "unix_timestamp"}
                missing = required.difference(reader.fieldnames or [])
                if missing:
                    raise RuntimeError(f"{subreddit}: missing columns {sorted(missing)}")
                for row in reader:
                    total_rows += 1
                    epoch = int(row["unix_timestamp"])
                    min_epoch = epoch if min_epoch is None else min(min_epoch, epoch)
                    max_epoch = epoch if max_epoch is None else max(max_epoch, epoch)
                    if previous_epoch is not None and epoch < previous_epoch:
                        nonmonotonic_rows += 1
                    previous_epoch = epoch

                    text_epoch = int(
                        datetime.fromisoformat(row["timestamp"])
                        .replace(tzinfo=timezone.utc)
                        .timestamp()
                    )
                    if text_epoch != epoch:
                        timestamp_text_mismatches += 1

                    thread_id = row["thread_id"]
                    item_id = row["id"]
                    if item_id == thread_id:
                        root_count += 1
                        if thread_id in roots:
                            duplicate_roots += 1
                        roots[thread_id] = epoch
                        if 0 <= endpoint - epoch < one_day:
                            roots_last_24h += 1
                    else:
                        comment_count += 1

        # A second pass makes the identifier join exact even if the archive has
        # an ordering reversal or a comment precedes its root in file order.
        with archive.open(config["member"]) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
                reader = csv.DictReader(text)
                for row in reader:
                    if row["id"] == row["thread_id"]:
                        continue
                    root_epoch = roots.get(row["thread_id"])
                    if root_epoch is None:
                        comments_without_matched_root += 1
                        continue
                    matched_comments += 1
                    delay = int(row["unix_timestamp"]) - root_epoch
                    if delay < 0:
                        negative_delays += 1
                    if endpoint - root_epoch >= one_day:
                        eligible_matched_comments += 1
                        if 0 <= delay <= one_day:
                            eligible_comments_within_24h += 1

    root_pct = 100 * roots_last_24h / root_count if root_count else None
    comment_pct = (
        100 * eligible_comments_within_24h / eligible_matched_comments
        if eligible_matched_comments
        else None
    )
    checks.extend(
        [
            {
                "subreddit": subreddit,
                "check": "unique_root_identifiers",
                "status": "PASS" if duplicate_roots == 0 else "FAIL",
                "observed": duplicate_roots,
                "expected": 0,
            },
            {
                "subreddit": subreddit,
                "check": "timestamp_text_equals_unix_utc",
                "status": "PASS" if timestamp_text_mismatches == 0 else "FAIL",
                "observed": timestamp_text_mismatches,
                "expected": 0,
            },
            {
                "subreddit": subreddit,
                "check": "raw_rows_timestamp_sorted",
                "status": "PASS" if nonmonotonic_rows == 0 else "WARN",
                "observed": nonmonotonic_rows,
                "expected": 0,
            },
            {
                "subreddit": subreddit,
                "check": "nonnegative_matched_comment_delays",
                "status": "PASS" if negative_delays == 0 else "FAIL",
                "observed": negative_delays,
                "expected": 0,
            },
        ]
    )
    result = {
        "subreddit": subreddit,
        "archive_filename": config["filename"],
        "zenodo_record": "10.5281/zenodo.17079717",
        "archive_md5": observed_md5,
        "coverage_boundary_utc": config["endpoint"],
        "endpoint_kind": "documented_coverage_boundary_proxy",
        "raw_rows": total_rows,
        "raw_roots": root_count,
        "raw_comments": comment_count,
        "raw_min_timestamp_utc": datetime.fromtimestamp(min_epoch, timezone.utc).isoformat(),
        "raw_max_timestamp_utc": datetime.fromtimestamp(max_epoch, timezone.utc).isoformat(),
        "raw_roots_final_24h": roots_last_24h,
        "raw_roots_final_24h_pct": root_pct,
        "preliminary_raw_roots_final_24h_pct": config["preliminary_root_pct"],
        "preliminary_root_lead_status_0_1pp": (
            "reproduced"
            if abs(root_pct - config["preliminary_root_pct"]) <= 0.1
            else "rejected"
        ),
        "matched_comments": matched_comments,
        "comments_without_matched_root": comments_without_matched_root,
        "eligible_matched_comments_24h": eligible_matched_comments,
        "eligible_comments_arriving_within_24h": eligible_comments_within_24h,
        "eligible_comments_arriving_within_24h_pct": comment_pct,
        "preliminary_comments_within_24h_pct": config["preliminary_comment_pct"],
        "preliminary_comment_lead_status_0_1pp": (
            "reproduced"
            if abs(comment_pct - config["preliminary_comment_pct"]) <= 0.1
            else "rejected"
        ),
    }
    return result, checks


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        action="append",
        type=parse_archive_spec,
        required=True,
        help="Repeat as subreddit=path; valid subreddits: conspiracy, crypto, politics",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    supplied = dict(args.archive)
    if set(supplied) != set(CONFIG):
        parser.error("Exactly one archive for each of conspiracy, crypto, and politics is required")

    results: list[dict] = []
    checks: list[dict] = []
    for subreddit in CONFIG:
        result, subreddit_checks = inspect_archive(subreddit, supplied[subreddit])
        results.append(result)
        checks.extend(subreddit_checks)
        print(f"completed {subreddit}", flush=True)

    write_csv(args.output_dir / "raw_archive_diagnostics.csv", results)
    write_csv(args.output_dir / "raw_archive_validation.csv", checks)
    run = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archives": {key: str(value) for key, value in supplied.items()},
        "outputs_contain_identifiers_or_text": False,
    }
    (args.output_dir / "raw_archive_run_record.json").write_text(
        json.dumps(run, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
