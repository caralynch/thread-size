#!/usr/bin/env python3
"""Aggregate-only audit of historical subreddit-selection source exports.

The script streams 4CAT CSV files directly from ZIP archives. It never writes
Reddit records, text, URLs, domains, or account labels. Pickle files are neither
opened nor imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import subprocess
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import urlparse


UTC = timezone.utc
ROOT_REQUIRED_COLUMNS = {
    "thread_id",
    "id",
    "timestamp",
    "author",
    "domain",
    "url",
    "subreddit",
    "parent",
    "unix_timestamp",
}
OPTIONAL_COLUMNS = {"body", "subject", "image_file", "image_md5"}
LITERAL_MISSING = {"", "na", "n/a", "nan", "none", "null"}
LITERAL_ACCOUNT_PLACEHOLDERS = {"[deleted]", "[removed]", "deleted", "removed"}

IMAGE_EXTENSIONS = {
    ".apng",
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".jpeg",
    ".jpg",
    ".png",
    ".svg",
    ".tif",
    ".tiff",
    ".webp",
}
VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ogv",
    ".webm",
    ".wmv",
}
IMAGE_HOSTS = {
    "flic.kr",
    "flickr.com",
    "giphy.com",
    "gyazo.com",
    "i.imgur.com",
    "i.redd.it",
    "ibb.co",
    "imgur.com",
    "imageshack.com",
    "postimg.cc",
    "preview.redd.it",
}
VIDEO_HOSTS = {
    "bitchute.com",
    "dailymotion.com",
    "gfycat.com",
    "odysee.com",
    "rumble.com",
    "streamable.com",
    "tiktok.com",
    "v.redd.it",
    "vimeo.com",
    "youtu.be",
    "youtube.com",
}
REDDIT_HOSTS = {"reddit.com", "redd.it"}


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in LITERAL_MISSING else text


def canonical_host(value: str) -> str:
    value = clean(value).lower()
    if not value:
        return ""
    if "://" in value:
        value = urlparse(value).hostname or ""
    else:
        value = value.split("/", 1)[0].split(":", 1)[0]
    if value.startswith("www."):
        value = value[4:]
    return value.rstrip(".")


def host_matches(host: str, candidates: set[str]) -> bool:
    return any(host == candidate or host.endswith("." + candidate) for candidate in candidates)


def url_extension(url: str) -> str:
    path = urlparse(clean(url)).path.lower()
    suffix = Path(path).suffix
    return suffix if len(suffix) <= 8 else ""


def parse_epoch(value: str) -> datetime | None:
    value = clean(value)
    if not value:
        return None
    try:
        number = float(value)
        if not math.isfinite(number):
            return None
        return datetime.fromtimestamp(number, tz=UTC)
    except (ValueError, OverflowError, OSError):
        return None


def parse_display_timestamp(value: str) -> datetime | None:
    value = clean(value)
    if not value:
        return None
    value = value.replace("Z", "+00:00")
    for candidate in (value, value.replace(" ", "T", 1)):
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(UTC).replace(tzinfo=None)
            return parsed
        except ValueError:
            pass
    return None


def is_root(row: Mapping[str, str]) -> bool:
    return not clean(row.get("parent")) and clean(row.get("id")) == clean(row.get("thread_id"))


def month_start(value: datetime) -> datetime:
    return datetime(value.year, value.month, 1, tzinfo=UTC)


def next_month(value: datetime) -> datetime:
    if value.month == 12:
        return datetime(value.year + 1, 1, 1, tzinfo=UTC)
    return datetime(value.year, value.month + 1, 1, tzinfo=UTC)


def iter_months(start: datetime, end: datetime) -> Iterable[datetime]:
    current = month_start(start)
    final = month_start(end)
    while current <= final:
        yield current
        current = next_month(current)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def classify_format(row: Mapping[str, str], rule: str) -> tuple[str, str]:
    domain = canonical_host(row.get("domain", ""))
    url = clean(row.get("url"))
    url_host = canonical_host(url)
    host = url_host or domain
    extension = url_extension(url)
    self_domain = domain.startswith("self.")
    image_by_url = extension in IMAGE_EXTENSIONS or host_matches(host, IMAGE_HOSTS)
    video_by_url = extension in VIDEO_EXTENSIONS or host_matches(host, VIDEO_HOSTS)
    image_metadata = bool(clean(row.get("image_file")) or clean(row.get("image_md5")))

    if video_by_url:
        return "video", "video_url_or_host"
    if image_by_url:
        return "image", "image_url_or_host"
    if rule == "metadata_inclusive" and image_metadata:
        return "image", "fourcat_image_metadata"
    if self_domain:
        return "self_text", "self_domain"
    if rule == "metadata_inclusive" and not url and clean(row.get("body")):
        return "self_text", "body_without_url"
    if host_matches(host, REDDIT_HOSTS):
        return "unknown", "internal_reddit_nonmedia_unresolved"
    if host or url:
        return "external_article_or_link", "external_nonmedia_host"
    return "unknown", "insufficient_metadata"


def account_summary(counts: Counter[str]) -> tuple[int, int, float]:
    denominator = len(counts)
    numerator = sum(value >= 2 for value in counts.values())
    percent = (100.0 * numerator / denominator) if denominator else math.nan
    return denominator, numerator, percent


def git_blob(repo: Path, revision_path: str) -> bytes:
    process = subprocess.run(
        ["git", "-C", str(repo), "show", revision_path],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process.stdout


def load_documented_exclusions(repo: Path, commit: str) -> tuple[dict[str, dict[str, set[str]]], dict[str, object]]:
    blob = git_blob(repo, f"{commit}:authors_to_remove.csv")
    text = blob.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    rules: dict[str, dict[str, set[str]]] = {}
    rows = list(reader)
    subreddit_columns = [
        column
        for column in (reader.fieldnames or [])
        if column not in {"author", "to_remove", "note"}
    ]
    for subreddit in subreddit_columns:
        rules[subreddit] = {
            "all": set(),
            "comments": set(),
            "deleted_placeholder_all": set(),
            "deleted_placeholder_comments": set(),
        }
    for row in rows:
        author = clean(row.get("author"))
        mode = clean(row.get("to_remove")).lower()
        note = clean(row.get("note")).lower()
        if not author or mode not in {"all", "comments"}:
            continue
        for subreddit in subreddit_columns:
            if clean(row.get(subreddit)) == "1":
                rules[subreddit][mode].add(author)
                if "dummy author label" in note or "removed comment" in note:
                    rules[subreddit][f"deleted_placeholder_{mode}"].add(author)
    provenance = {
        "commit": commit,
        "git_blob_sha1": subprocess.run(
            ["git", "-C", str(repo), "rev-parse", f"{commit}:authors_to_remove.csv"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "row_count": len(rows),
        "subreddit_columns": "|".join(subreddit_columns),
    }
    return rules, provenance


@dataclass
class DatasetAggregate:
    subreddit: str
    source_path: Path
    archive_entry: str = ""
    archive_entry_bytes: int = 0
    archive_entry_mtime: str = ""
    schema: list[str] = field(default_factory=list)
    missing_required_columns: list[str] = field(default_factory=list)
    total_rows: int = 0
    invalid_epoch_rows: int = 0
    missing_author_rows: int = 0
    literal_placeholder_author_rows: int = 0
    parent_blank_rows: int = 0
    self_id_rows: int = 0
    strict_root_rows: int = 0
    parent_blank_not_self_id: int = 0
    self_id_with_parent: int = 0
    earliest_utc: datetime | None = None
    latest_utc: datetime | None = None
    earliest_root_utc: datetime | None = None
    latest_root_utc: datetime | None = None
    monthly_roots: Counter[str] = field(default_factory=Counter)
    format_counts: dict[str, Counter[tuple[str, str]]] = field(
        default_factory=lambda: {"url_domain_only": Counter(), "metadata_inclusive": Counter()}
    )
    account_counts: Counter[str] = field(default_factory=Counter)
    literal_filtered_counts: Counter[str] = field(default_factory=Counter)
    deleted_placeholder_filtered_counts: Counter[str] = field(default_factory=Counter)
    documented_cleaning_counts: Counter[str] = field(default_factory=Counter)
    timestamp_offsets_seconds: Counter[int] = field(default_factory=Counter)
    display_timestamp_unparseable: int = 0


def analyze_archive(
    subreddit: str,
    archive_path: Path,
    exclusion_rules: dict[str, dict[str, set[str]]],
) -> DatasetAggregate:
    aggregate = DatasetAggregate(subreddit=subreddit, source_path=archive_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        csv_entries = [entry for entry in archive.infolist() if entry.filename.lower().endswith(".csv")]
        if len(csv_entries) != 1:
            raise ValueError(f"{archive_path}: expected exactly one CSV entry, found {len(csv_entries)}")
        entry = csv_entries[0]
        aggregate.archive_entry = entry.filename
        aggregate.archive_entry_bytes = entry.file_size
        aggregate.archive_entry_mtime = datetime(*entry.date_time).isoformat()
        with archive.open(entry, "r") as binary, io.TextIOWrapper(
            binary, encoding="utf-8-sig", errors="replace", newline=""
        ) as text:
            reader = csv.DictReader(text)
            aggregate.schema = reader.fieldnames or []
            aggregate.missing_required_columns = sorted(ROOT_REQUIRED_COLUMNS - set(aggregate.schema))
            if aggregate.missing_required_columns:
                raise ValueError(
                    f"{archive_path}: missing required columns {aggregate.missing_required_columns}"
                )
            rules = exclusion_rules.get(subreddit, {})
            remove_all = rules.get("all", set())
            remove_comments = rules.get("comments", set())
            deleted_all = rules.get("deleted_placeholder_all", set())
            deleted_comments = rules.get("deleted_placeholder_comments", set())

            for row in reader:
                aggregate.total_rows += 1
                root = is_root(row)
                parent_blank = not clean(row.get("parent"))
                self_id = clean(row.get("id")) == clean(row.get("thread_id"))
                aggregate.parent_blank_rows += int(parent_blank)
                aggregate.self_id_rows += int(self_id)
                aggregate.strict_root_rows += int(root)
                aggregate.parent_blank_not_self_id += int(parent_blank and not self_id)
                aggregate.self_id_with_parent += int(self_id and not parent_blank)

                observed = parse_epoch(row.get("unix_timestamp", ""))
                if observed is None:
                    aggregate.invalid_epoch_rows += 1
                else:
                    aggregate.earliest_utc = min(aggregate.earliest_utc or observed, observed)
                    aggregate.latest_utc = max(aggregate.latest_utc or observed, observed)
                    displayed = parse_display_timestamp(row.get("timestamp", ""))
                    if displayed is None:
                        aggregate.display_timestamp_unparseable += 1
                    else:
                        offset = round((displayed - observed.replace(tzinfo=None)).total_seconds())
                        aggregate.timestamp_offsets_seconds[offset] += 1
                    if root:
                        aggregate.earliest_root_utc = min(aggregate.earliest_root_utc or observed, observed)
                        aggregate.latest_root_utc = max(aggregate.latest_root_utc or observed, observed)
                        aggregate.monthly_roots[observed.strftime("%Y-%m")] += 1

                if root:
                    for rule in aggregate.format_counts:
                        category, reason = classify_format(row, rule)
                        aggregate.format_counts[rule][(category, reason)] += 1

                author = clean(row.get("author"))
                if not author:
                    aggregate.missing_author_rows += 1
                    continue
                author_lower = author.lower()
                if author_lower in LITERAL_ACCOUNT_PLACEHOLDERS:
                    aggregate.literal_placeholder_author_rows += 1
                aggregate.account_counts[author] += 1
                if author_lower not in LITERAL_ACCOUNT_PLACEHOLDERS:
                    aggregate.literal_filtered_counts[author] += 1
                if author not in deleted_all and not (not root and author in deleted_comments):
                    aggregate.deleted_placeholder_filtered_counts[author] += 1
                if author not in remove_all and not (not root and author in remove_comments):
                    aggregate.documented_cleaning_counts[author] += 1
    return aggregate


def coverage_rows(aggregates: list[DatasetAggregate]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in aggregates:
        mode_offset, mode_count = (item.timestamp_offsets_seconds.most_common(1)[0]
                                   if item.timestamp_offsets_seconds else ("", 0))
        rows.append(
            {
                "subreddit": item.subreddit,
                "total_contributions": item.total_rows,
                "strict_root_submissions": item.strict_root_rows,
                "earliest_contribution_utc": item.earliest_utc.isoformat() if item.earliest_utc else "",
                "latest_contribution_utc": item.latest_utc.isoformat() if item.latest_utc else "",
                "earliest_root_utc": item.earliest_root_utc.isoformat() if item.earliest_root_utc else "",
                "latest_root_utc": item.latest_root_utc.isoformat() if item.latest_root_utc else "",
                "invalid_unix_timestamp_rows": item.invalid_epoch_rows,
                "display_timestamp_mode_offset_seconds_from_utc": mode_offset,
                "rows_with_mode_offset": mode_count,
                "display_timestamp_unparseable_rows": item.display_timestamp_unparseable,
                "missing_author_rows": item.missing_author_rows,
                "literal_placeholder_author_rows": item.literal_placeholder_author_rows,
            }
        )
    return rows


def monthly_rows(aggregates: list[DatasetAggregate]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    details: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for item in aggregates:
        if item.earliest_utc is None or item.latest_utc is None:
            continue
        observed_counts: list[int] = []
        complete_counts: list[int] = []
        for start in iter_months(item.earliest_utc, item.latest_utc):
            end = next_month(start)
            key = start.strftime("%Y-%m")
            count = item.monthly_roots.get(key, 0)
            inferred_complete = (
                item.earliest_utc <= start + timedelta(minutes=5)
                and item.latest_utc >= end - timedelta(minutes=5)
            )
            observed_counts.append(count)
            if inferred_complete:
                complete_counts.append(count)
            details.append(
                {
                    "subreddit": item.subreddit,
                    "calendar_month_utc": key,
                    "root_submissions": count,
                    "coverage_status": "complete_inferred_from_timestamps" if inferred_complete else "partial",
                    "month_start_utc": start.isoformat(),
                    "month_end_exclusive_utc": end.isoformat(),
                }
            )
        complete_result = (
            "holds" if complete_counts and all(value > 1000 for value in complete_counts)
            else "does_not_hold" if complete_counts
            else "unresolved_no_complete_month"
        )
        summaries.append(
            {
                "subreddit": item.subreddit,
                "observed_months": len(observed_counts),
                "complete_months_inferred": len(complete_counts),
                "minimum_observed_month": min(observed_counts),
                "median_observed_month": statistics.median(observed_counts),
                "mean_observed_month": statistics.mean(observed_counts),
                "minimum_complete_month": min(complete_counts) if complete_counts else "",
                "median_complete_month": statistics.median(complete_counts) if complete_counts else "",
                "mean_complete_month": statistics.mean(complete_counts) if complete_counts else "",
                "over_1000_every_observed_month": all(value > 1000 for value in observed_counts),
                "over_1000_typical_month_median": statistics.median(observed_counts) > 1000,
                "over_1000_mean_observed_month": statistics.mean(observed_counts) > 1000,
                "over_1000_every_complete_month": complete_result,
                "conclusion_class": "retrospective validation using later study data",
            }
        )
    return details, summaries


def format_rows(aggregates: list[DatasetAggregate]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    details: list[dict[str, object]] = []
    claims: list[dict[str, object]] = []
    categories = ["self_text", "external_article_or_link", "image", "video", "unknown"]
    for item in aggregates:
        for rule, counts in item.format_counts.items():
            category_counts = Counter()
            for (category, reason), count in counts.items():
                category_counts[category] += count
                details.append(
                    {
                        "subreddit": item.subreddit,
                        "classification_rule": rule,
                        "category": category,
                        "reason": reason,
                        "root_submissions": count,
                        "percent_of_roots": 100.0 * count / item.strict_root_rows if item.strict_root_rows else "",
                    }
                )
            for category in categories:
                if category not in category_counts:
                    details.append(
                        {
                            "subreddit": item.subreddit,
                            "classification_rule": rule,
                            "category": category,
                            "reason": "no_classified_rows",
                            "root_submissions": 0,
                            "percent_of_roots": 0.0,
                        }
                    )
            total = item.strict_root_rows
            definite_media = category_counts["image"] + category_counts["video"]
            definite_nonmedia = category_counts["self_text"] + category_counts["external_article_or_link"]
            unknown = category_counts["unknown"]
            claims.append(
                {
                    "subreddit": item.subreddit,
                    "classification_rule": rule,
                    "root_submissions": total,
                    "definite_media_percent_lower_bound": 100.0 * definite_media / total if total else "",
                    "unknown_percent": 100.0 * unknown / total if total else "",
                    "media_percent_upper_bound_if_all_unknown_are_media": 100.0 * (definite_media + unknown) / total if total else "",
                    "definite_nonmedia_percent_lower_bound": 100.0 * definite_nonmedia / total if total else "",
                    "predominantly_not_image_or_video_even_if_unknown_is_media": definite_nonmedia > total / 2,
                    "conclusion_class": "sensitivity analysis under a newly defined rule",
                }
            )
    details.sort(key=lambda row: (str(row["subreddit"]), str(row["classification_rule"]), str(row["category"]), str(row["reason"])))
    return details, claims


def repeat_rows(aggregates: list[DatasetAggregate], exclusions_available: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    variants = (
        ("observable_nonblank_unfiltered", "Includes pseudonymous labels for deleted/service accounts when present", "account_counts"),
        ("literal_placeholders_excluded", "Excludes only literal deleted/removed account labels", "literal_filtered_counts"),
        ("documented_deleted_placeholder_excluded", "Uses the 2023 documented dummy-label rule", "deleted_placeholder_filtered_counts"),
        ("documented_2023_cleaning_rules", "Uses all applicable 2023 author-removal rules", "documented_cleaning_counts"),
    )
    for item in aggregates:
        for variant, treatment, attribute in variants:
            documented = variant.startswith("documented_")
            if documented and item.subreddit not in exclusions_available:
                rows.append(
                    {
                        "subreddit": item.subreddit,
                        "variant": variant,
                        "denominator_unique_observable_accounts": "",
                        "numerator_accounts_with_at_least_two_contributions": "",
                        "percent_repeat_accounts": "",
                        "over_50_percent": "unresolved_no_documented_rule_for_subreddit",
                        "treatment": treatment,
                        "conclusion_class": "unresolved",
                    }
                )
                continue
            denominator, numerator, percent = account_summary(getattr(item, attribute))
            rows.append(
                {
                    "subreddit": item.subreddit,
                    "variant": variant,
                    "denominator_unique_observable_accounts": denominator,
                    "numerator_accounts_with_at_least_two_contributions": numerator,
                    "percent_repeat_accounts": percent,
                    "over_50_percent": percent > 50.0,
                    "treatment": treatment,
                    "conclusion_class": "sensitivity analysis under a newly defined rule",
                }
            )
    return rows


def source_inventory_rows(
    aggregates: list[DatasetAggregate],
    hashes: dict[Path, str],
    exclusion_provenance: Mapping[str, object],
    workbook: Path | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in aggregates:
        stat = item.source_path.stat()
        rows.append(
            {
                "source_id": f"raw_{item.subreddit}_zip",
                "path": str(item.source_path),
                "filesystem_modified_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
                "strongest_dated_evidence": item.archive_entry_mtime,
                "checksum_type": "SHA-256" if item.source_path in hashes else "not_computed",
                "checksum": hashes.get(item.source_path, ""),
                "schema": "|".join(item.schema),
                "period_utc": f"{item.earliest_utc.isoformat() if item.earliest_utc else ''}/{item.latest_utc.isoformat() if item.latest_utc else ''}",
                "processing_status": "raw collected 4CAT/Pushshift export in ZIP",
                "audit_role": "retrospective validation source",
                "loaded": "streamed_csv_only",
                "notes": "ZIP/entry copy times are recorded but are not assumed to equal collection time",
            }
        )
    rows.append(
        {
            "source_id": "historical_author_exclusions",
            "path": "thread_size_prediction git blob: authors_to_remove.csv",
            "filesystem_modified_local": "",
            "strongest_dated_evidence": "2023-01-16T17:39:21+00:00",
            "checksum_type": "SHA-256 and Git blob SHA-1",
            "checksum": f"sha256:{exclusion_provenance['sha256']};git-blob:{exclusion_provenance['git_blob_sha1']}",
            "schema": "author|to_remove|note|books|conspiracy|crypto|thedonald",
            "period_utc": "not_applicable",
            "processing_status": "historical cleaning rule record",
            "audit_role": "repeat-account sensitivity for conspiracy/crypto only",
            "loaded": "parsed_as_text_from_git_blob",
            "notes": "Account labels retained in memory only and never written to audit outputs; no politics column",
        }
    )
    if workbook is not None and workbook.exists():
        stat = workbook.stat()
        rows.append(
            {
                "source_id": "dataset_summaries_workbook",
                "path": str(workbook),
                "filesystem_modified_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
                "strongest_dated_evidence": "Author thresholds sheet first present in Git commit 2023-08-01T17:24:21+01:00",
                "checksum_type": "SHA-256",
                "checksum": sha256_file(workbook),
                "schema": "workbook; Author thresholds columns: subreddit|% authors with 1 activity",
                "period_utc": "study periods; not stated on threshold sheet",
                "processing_status": "summary workbook derived from collected/cleaned study data",
                "audit_role": "recovered original summary artifact; calculation code unresolved",
                "loaded": "safe_xlsx_inspection_only",
                "notes": "Sheet stores values, not formulas; workbook predates first located prose claim but postdates collection",
            }
        )
    return rows


def validation_rows(aggregates: list[DatasetAggregate]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in aggregates:
        tests = {
            "required_schema_present": not item.missing_required_columns,
            "all_rows_have_valid_unix_timestamp": item.invalid_epoch_rows == 0,
            "root_definition_parent_and_id_agree": item.parent_blank_not_self_id == 0 and item.self_id_with_parent == 0,
            "format_totals_equal_root_total_url_domain_only": sum(item.format_counts["url_domain_only"].values()) == item.strict_root_rows,
            "format_totals_equal_root_total_metadata_inclusive": sum(item.format_counts["metadata_inclusive"].values()) == item.strict_root_rows,
            "root_month_counts_equal_root_total": sum(item.monthly_roots.values()) == item.strict_root_rows,
            "account_denominator_not_greater_than_contributions": len(item.account_counts) <= item.total_rows,
        }
        for name, passed in tests.items():
            rows.append(
                {
                    "subreddit": item.subreddit,
                    "check": name,
                    "status": "PASS" if passed else "FAIL",
                    "observed": {
                        "required_schema_present": "|".join(item.missing_required_columns),
                        "all_rows_have_valid_unix_timestamp": item.invalid_epoch_rows,
                        "root_definition_parent_and_id_agree": f"parent_blank_not_self_id={item.parent_blank_not_self_id};self_id_with_parent={item.self_id_with_parent}",
                        "format_totals_equal_root_total_url_domain_only": sum(item.format_counts["url_domain_only"].values()),
                        "format_totals_equal_root_total_metadata_inclusive": sum(item.format_counts["metadata_inclusive"].values()),
                        "root_month_counts_equal_root_total": sum(item.monthly_roots.values()),
                        "account_denominator_not_greater_than_contributions": len(item.account_counts),
                    }[name],
                    "expected": {
                        "required_schema_present": "no missing columns",
                        "all_rows_have_valid_unix_timestamp": 0,
                        "root_definition_parent_and_id_agree": "both mismatch counts 0",
                        "format_totals_equal_root_total_url_domain_only": item.strict_root_rows,
                        "format_totals_equal_root_total_metadata_inclusive": item.strict_root_rows,
                        "root_month_counts_equal_root_total": item.strict_root_rows,
                        "account_denominator_not_greater_than_contributions": f"<= {item.total_rows}",
                    }[name],
                }
            )
    return rows


def parse_archive_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("archive must be SUBREDDIT=PATH")
    subreddit, path = value.split("=", 1)
    subreddit = subreddit.strip().lower()
    if not subreddit or not path.strip():
        raise argparse.ArgumentTypeError("archive must be SUBREDDIT=PATH")
    return subreddit, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", required=True, type=parse_archive_argument)
    parser.add_argument("--historical-repo", required=True, type=Path)
    parser.add_argument("--exclusions-commit", default="f44ea082d097f4f23974b0eef9a41ed63de5bc7b")
    parser.add_argument("--workbook", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hash-inputs", action="store_true")
    args = parser.parse_args()

    archive_map = dict(args.archive)
    expected = {"politics", "crypto", "conspiracy"}
    if set(archive_map) != expected:
        parser.error(f"exactly these subreddit keys are required: {sorted(expected)}")
    for path in archive_map.values():
        if path.suffix.lower() in {".p", ".pickle", ".pkl"}:
            parser.error("pickle inputs are prohibited")
        if not path.is_file():
            parser.error(f"archive does not exist: {path}")
    if not args.historical_repo.is_dir():
        parser.error(f"historical repository does not exist: {args.historical_repo}")

    exclusions, exclusion_provenance = load_documented_exclusions(
        args.historical_repo, args.exclusions_commit
    )
    aggregates = []
    for subreddit in sorted(archive_map):
        print(f"Streaming aggregate audit for r/{subreddit}...", file=sys.stderr, flush=True)
        aggregates.append(analyze_archive(subreddit, archive_map[subreddit], exclusions))

    hashes: dict[Path, str] = {}
    if args.hash_inputs:
        for subreddit in sorted(archive_map):
            path = archive_map[subreddit]
            print(f"Computing SHA-256 for r/{subreddit} archive...", file=sys.stderr, flush=True)
            hashes[path] = sha256_file(path)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    coverage = coverage_rows(aggregates)
    monthly_detail, monthly_summary = monthly_rows(aggregates)
    format_detail, format_claims = format_rows(aggregates)
    repeats = repeat_rows(aggregates, set(exclusions))
    inventory = source_inventory_rows(aggregates, hashes, exclusion_provenance, args.workbook)
    validations = validation_rows(aggregates)

    write_csv(output / "dataset_coverage.csv", list(coverage[0]), coverage)
    write_csv(output / "monthly_submissions.csv", list(monthly_detail[0]), monthly_detail)
    write_csv(output / "monthly_submission_summary.csv", list(monthly_summary[0]), monthly_summary)
    write_csv(output / "submission_format_counts.csv", list(format_detail[0]), format_detail)
    write_csv(output / "submission_format_claims.csv", list(format_claims[0]), format_claims)
    write_csv(output / "repeat_account_summary.csv", list(repeats[0]), repeats)
    write_csv(output / "source_inventory_computed.csv", list(inventory[0]), inventory)
    write_csv(output / "validation_checks.csv", list(validations[0]), validations)

    run_record = {
        "analysis_version": "1.0.0",
        "python": sys.version.split()[0],
        "timezone_for_metrics": "UTC",
        "root_definition": "parent empty AND id equals thread_id",
        "archives": {key: str(archive_map[key]) for key in sorted(archive_map)},
        "historical_repo": str(args.historical_repo),
        "exclusions_commit": args.exclusions_commit,
        "workbook": str(args.workbook) if args.workbook else None,
        "hash_inputs": args.hash_inputs,
        "outputs": sorted(path.name for path in output.glob("*.csv")),
    }
    (output / "run_record.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failed = [row for row in validations if row["status"] == "FAIL"]
    print(f"Completed with {len(failed)} failed validation checks.", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
