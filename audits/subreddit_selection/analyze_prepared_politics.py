#!/usr/bin/env python3
"""Aggregate-only comparison of a prepared r/politics CSV with the workbook.

This script exists because the recovered 2023 author-threshold workbook may
have been calculated from the earlier, separately prepared r/politics data.
It reads the CSV and XLSX safely as text/XML containers and never opens pickle
files or emits account labels or Reddit text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path


MISSING = {"", "na", "n/a", "nan", "none", "null"}
PLACEHOLDERS = {"[deleted]", "[removed]", "deleted", "removed"}
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def clean(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in MISSING else text


def is_root(row: dict[str, str]) -> bool:
    return not clean(row.get("parent")) and clean(row.get("id")) == clean(row.get("thread_id"))


class HashingReader(io.RawIOBase):
    def __init__(self, raw: io.BufferedReader, digest: "hashlib._Hash") -> None:
        self.raw = raw
        self.digest = digest

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        count = self.raw.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count


def shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t"))
            for item in root.findall(f"{{{MAIN_NS}}}si")]


def workbook_sheet_rows(path: Path, sheet_name: str) -> list[list[object]]:
    with zipfile.ZipFile(path, "r") as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationship_id = None
        for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
            if sheet.get("name") == sheet_name:
                relationship_id = sheet.get(f"{{{REL_NS}}}id")
                break
        if not relationship_id:
            raise ValueError(f"Workbook sheet not found: {sheet_name}")
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in relationships.findall(f"{{{PKG_REL_NS}}}Relationship"):
            if rel.get("Id") == relationship_id:
                target = rel.get("Target")
                break
        if not target:
            raise ValueError(f"Workbook relationship not found for: {sheet_name}")
        sheet_path = posixpath.normpath(posixpath.join("xl", target.lstrip("/")))
        strings = shared_strings(archive)
        root = ET.fromstring(archive.read(sheet_path))
        rows: list[list[object]] = []
        for row_node in root.findall(f".//{{{MAIN_NS}}}row"):
            row: list[object] = []
            for cell in row_node.findall(f"{{{MAIN_NS}}}c"):
                cell_type = cell.get("t")
                value_node = cell.find(f"{{{MAIN_NS}}}v")
                if cell_type == "inlineStr":
                    value: object = "".join(
                        node.text or "" for node in cell.iter(f"{{{MAIN_NS}}}t")
                    )
                elif value_node is None:
                    value = ""
                elif cell_type == "s":
                    value = strings[int(value_node.text or "0")]
                else:
                    raw_value = value_node.text or ""
                    try:
                        value = float(raw_value)
                    except ValueError:
                        value = raw_value
                row.append(value)
            rows.append(row)
        return rows


def workbook_repeat_percent(path: Path, subreddit: str) -> float:
    rows = workbook_sheet_rows(path, "Author thresholds")
    for row in rows[1:]:
        if len(row) >= 2 and clean(row[0]).lower() == subreddit.lower():
            return 100.0 - float(row[1])
    raise ValueError(f"No {subreddit!r} row on Author thresholds sheet")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.csv.suffix.lower() in {".p", ".pkl", ".pickle"}:
        parser.error("pickle inputs are prohibited")
    if not args.csv.is_file() or not args.workbook.is_file():
        parser.error("CSV and workbook must exist")

    digest = hashlib.sha256()
    total_rows = 0
    strict_roots = 0
    missing_author_rows = 0
    accounts: Counter[str] = Counter()
    accounts_without_literals: Counter[str] = Counter()
    schema: list[str] = []
    with args.csv.open("rb") as raw:
        hashing = HashingReader(raw, digest)
        with io.BufferedReader(hashing, buffer_size=8 * 1024 * 1024) as buffered:
            with io.TextIOWrapper(buffered, encoding="utf-8-sig", errors="replace", newline="") as text:
                reader = csv.DictReader(text)
                schema = reader.fieldnames or []
                required = {"thread_id", "id", "author", "parent"}
                missing = sorted(required - set(schema))
                if missing:
                    raise ValueError(f"Prepared CSV missing required columns: {missing}")
                for row in reader:
                    total_rows += 1
                    strict_roots += int(is_root(row))
                    author = clean(row.get("author"))
                    if not author:
                        missing_author_rows += 1
                        continue
                    accounts[author] += 1
                    if author.lower() not in PLACEHOLDERS:
                        accounts_without_literals[author] += 1

    workbook_percent = workbook_repeat_percent(args.workbook, "politics")
    rows = []
    for variant, counts in (
        ("observable_nonblank_unfiltered", accounts),
        ("literal_placeholders_excluded", accounts_without_literals),
    ):
        denominator = len(counts)
        numerator = sum(value >= 2 for value in counts.values())
        percent = 100.0 * numerator / denominator if denominator else math.nan
        rows.append(
            {
                "source": "prepared_politics_csv",
                "variant": variant,
                "total_contributions": total_rows,
                "strict_root_submissions": strict_roots,
                "denominator_unique_observable_accounts": denominator,
                "numerator_accounts_with_at_least_two_contributions": numerator,
                "percent_repeat_accounts": percent,
                "workbook_percent_repeat_accounts": workbook_percent,
                "difference_percentage_points": percent - workbook_percent,
                "over_50_percent": percent > 50.0,
                "conclusion_class": "retrospective validation using later study data",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    record = {
        "source_path": str(args.csv),
        "source_bytes": args.csv.stat().st_size,
        "source_modified_local": datetime.fromtimestamp(args.csv.stat().st_mtime).astimezone().isoformat(),
        "source_sha256": digest.hexdigest(),
        "schema": schema,
        "missing_author_rows": missing_author_rows,
        "workbook_path": str(args.workbook),
        "workbook_sha256": hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
        "account_labels_or_text_written": False,
    }
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
