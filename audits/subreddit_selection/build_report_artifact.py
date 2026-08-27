#!/usr/bin/env python3
"""Build the bounded, source-backed portable report manifest from audit CSVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def rows(name: str) -> list[dict[str, str]]:
    with (OUTPUTS / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str) -> float | int | None:
    if value == "":
        return None
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def source(source_id: str, label: str, path: str, description: str, filters: list[str] | None = None) -> dict:
    if path.endswith(".csv"):
        query = {
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM read_csv_auto({path!r}, header=true)",
            "description": description,
            "tables_used": [path],
            "filters": filters or [],
        }
    else:
        query = None
    result = {
        "id": source_id,
        "label": label,
        "path": path,
    }
    if query is not None:
        result["query"] = query
    return result


def write_report_csv(name: str, data: list[dict]) -> None:
    if not data:
        raise ValueError(f"Cannot write empty report dataset: {name}")
    with (OUTPUTS / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def main() -> int:
    monthly_detail_raw = rows("monthly_submissions.csv")
    monthly_summary_raw = rows("monthly_submission_summary.csv")
    format_raw = [row for row in rows("submission_format_claims.csv")
                  if row["classification_rule"] == "metadata_inclusive"]
    repeat_raw = rows("repeat_account_comparison.csv")
    with (ROOT / "source_inventory.csv").open("r", encoding="utf-8", newline="") as handle:
        inventory_raw = list(csv.DictReader(handle))

    monthly_detail = [
        {
            "subreddit": row["subreddit"],
            "month_utc": row["calendar_month_utc"],
            "root_submissions": int(row["root_submissions"]),
            "coverage": row["coverage_status"],
        }
        for row in monthly_detail_raw
    ]
    monthly_summary = [
        {
            "subreddit": row["subreddit"],
            "observed_months": int(row["observed_months"]),
            "complete_months": int(row["complete_months_inferred"]),
            "minimum_observed": number(row["minimum_observed_month"]),
            "median_observed": number(row["median_observed_month"]),
            "mean_observed": number(row["mean_observed_month"]),
            "every_complete_month_over_1000": row["over_1000_every_complete_month"],
        }
        for row in monthly_summary_raw
    ]
    format_summary = [
        {
            "subreddit": row["subreddit"],
            "definite_media_lower_pct": round(float(row["definite_media_percent_lower_bound"]), 3),
            "unknown_pct": round(float(row["unknown_percent"]), 3),
            "media_upper_pct": round(float(row["media_percent_upper_bound_if_all_unknown_are_media"]), 3),
            "definite_nonmedia_lower_pct": round(float(row["definite_nonmedia_percent_lower_bound"]), 3),
            "robust_nonmedia_majority": row["predominantly_not_image_or_video_even_if_unknown_is_media"],
        }
        for row in format_raw
    ]
    format_composition = []
    for row in format_summary:
        context = {
            "subreddit": row["subreddit"],
            "root_submissions": next(
                int(raw["root_submissions"])
                for raw in format_raw
                if raw["subreddit"] == row["subreddit"]
            ),
            "unknown_pct": row["unknown_pct"],
            "classification_rule": "metadata_inclusive",
        }
        format_composition.extend(
            [
                {
                    **context,
                    "component": "Definite non-media lower bound",
                    "percent_of_roots": row["definite_nonmedia_lower_pct"],
                    "bound_interpretation": "Self-text plus external article/link",
                },
                {
                    **context,
                    "component": "Media upper bound",
                    "percent_of_roots": row["media_upper_pct"],
                    "bound_interpretation": "Definite image/video plus every unknown",
                },
            ]
        )
    repeat_summary = []
    for row in repeat_raw:
        keep = (
            row["source_population"] == "raw_export"
            and row["variant"] in {"observable_nonblank_unfiltered", "documented_2023_cleaning_rules"}
        ) or (
            row["source_population"] == "prepared_study_csv"
            and row["variant"] == "observable_nonblank_unfiltered"
        )
        if keep:
            repeat_summary.append(
                {
                    "subreddit": row["subreddit"],
                    "source_population": row["source_population"],
                    "variant": row["variant"],
                    "observed_repeat_pct": number(row["observed_percent_repeat_accounts"]),
                    "workbook_repeat_pct": number(row["workbook_implied_percent_repeat_accounts"]),
                    "difference_pp": number(row["difference_percentage_points"]),
                    "over_50": row["published_over_50_percent_holds"],
                    "reproduces_workbook": row["reproduces_workbook_within_0_01pp"],
                }
            )

    by_subreddit = {row["subreddit"]: row for row in format_summary}
    raw_repeat = {
        row["subreddit"]: row
        for row in repeat_summary
        if row["source_population"] == "raw_export"
        and row["variant"] == "observable_nonblank_unfiltered"
    }
    monthly_by = {row["subreddit"]: row for row in monthly_summary}
    claim_summary = []
    for subreddit in ("conspiracy", "crypto", "politics"):
        claim_summary.append(
            {
                "subreddit": subreddit,
                "complete_months": monthly_by[subreddit]["complete_months"],
                "complete_month_test": monthly_by[subreddit]["every_complete_month_over_1000"],
                "definite_nonmedia_lower_pct": by_subreddit[subreddit]["definite_nonmedia_lower_pct"],
                "media_upper_pct": by_subreddit[subreddit]["media_upper_pct"],
                "repeat_accounts_raw_pct": round(float(raw_repeat[subreddit]["observed_repeat_pct"]), 3),
                "classification": "retrospective/sensitivity; not original selection reproduction",
            }
        )

    chronology = [
        {"date": "2021-11-18", "evidence": "r/politics raw ZIP entry", "implication": "Earlier separate politics collection supported"},
        {"date": "2022-04-07", "evidence": "Cleaned r/politics CSV", "implication": "Prepared politics data existed before October 2022"},
        {"date": "2022-05-09", "evidence": "Executed r/politics activity notebook", "implication": "Preliminary account-activity calculation recovered; selection use unresolved"},
        {"date": "2022-06-15", "evidence": "r/politics account aggregate CSVs", "implication": "Saved aggregates reproduce the 59.14% full-period result"},
        {"date": "2022-10-13/18", "evidence": "4CAT source tree and installation note", "implication": "Collection capability established; job date not recovered"},
        {"date": "2022-10-01/30", "evidence": "Conspiracy and Crypto export periods", "implication": "Observation window only; retrieval date unresolved"},
        {"date": "2022-12-07", "evidence": "Crypto ZIP entry and Docker archive", "implication": "Crypto export existed; archived Docker state has no surviving job metadata"},
        {"date": "2023-01-16", "evidence": "First comparative Git summary", "implication": "Four candidates; politics absent"},
        {"date": "2023-02-07", "evidence": "Git: added r/politics to datasets", "implication": "First recovered combined state"},
        {"date": "2023-08-01", "evidence": "First workbook with Author thresholds", "implication": "Threshold artifact postdates collection"},
        {"date": "2023-09-25", "evidence": "Research note asks which thresholds were used", "implication": "Original handling remained uncertain"},
        {"date": "2024-09-17", "evidence": "First located prose containing all criteria", "implication": "Claim appears well after collection"},
    ]
    classifications = [
        {"claim": "Earlier separate r/politics collection and preparation", "classification": "Recovered chronology evidence", "finding": "Supported"},
        {"claim": "Preliminary r/politics repeat-account calculation by May 2022", "classification": "Recovered original preliminary calculation", "finding": "59.14%; not documented as a selection calculation"},
        {"claim": "All three criteria applied before study-specific collection", "classification": "Unresolved", "finding": "Not supported by recovered dates"},
        {"claim": "Preliminary r/politics analysis used for final subreddit selection", "classification": "Unresolved", "finding": "Analysis recovered; decision link absent"},
        {"claim": "External statistics used for screening", "classification": "Unresolved", "finding": "No positive evidence recovered"},
        {"claim": "Workbook complements exceed 50% for selected three", "classification": "Recovered original summary artifact", "finding": "Values recovered; calculation not recovered"},
        {"claim": "Full-period repeat-account rule exceeds 50%", "classification": "Sensitivity analysis under a newly defined rule", "finding": "Holds in all three raw exports"},
        {"claim": "Predominantly not image/video", "classification": "Sensitivity analysis under a newly defined rule", "finding": "Holds under lower/upper-bound rules"},
        {"claim": "Over 1,000 in every complete month", "classification": "Retrospective validation using later study data", "finding": "Holds for politics; unresolved for the two October exports"},
    ]
    inventory = [
        {
            "source_id": row["source_id"],
            "kind": row["kind"],
            "strongest_date": row["strongest_date"],
            "processing_status": row["processing_status"],
            "audit_use": row["audit_use"],
            "access": row["access_or_safety"],
        }
        for row in inventory_raw
    ]

    write_report_csv("report_chronology.csv", chronology)
    write_report_csv("report_claim_summary.csv", claim_summary)
    write_report_csv("report_classifications.csv", classifications)

    sources = [
        source("provenance_record", "Dated provenance record", "provenance.md", "Git, file, workbook, and collection chronology assembled without executing notebooks or models."),
        source("chronology_output", "Recovered chronology table", "outputs/report_chronology.csv", "Bounded chronology rows assembled from the dated provenance record."),
        source("claim_summary_output", "Claim audit summary", "outputs/report_claim_summary.csv", "Bounded join of monthly, format, and repeat-account aggregate findings."),
        source("classification_output", "Conclusion classification table", "outputs/report_classifications.csv", "Required provenance classification for each substantive conclusion."),
        source("monthly_output", "UTC monthly submission counts", "outputs/monthly_submissions.csv", "Root submissions counted by UTC calendar month from historical ZIP exports.", ["Root: parent empty and id equals thread_id", "No current Reddit data", "Partial months retained and labelled"]),
        source("format_output", "Submission-format sensitivity", "outputs/submission_format_claims.csv", "URL/domain and metadata-inclusive classification of historical root submissions.", ["External links are not equated with media", "Unknowns retained", "No i.redd.it/v.redd.it-only shortcut"]),
        source("repeat_output", "Repeat-account sensitivity", "outputs/repeat_account_comparison.csv", "Full-period account contribution counts compared with the recovered workbook.", ["Unique observable non-empty account labels", "At least two total contributions", "No account labels emitted"]),
        source("inventory_output", "Comprehensive source inventory", "source_inventory.csv", "Candidate sources, dates, checksums, schemas/status, and safe-handling decisions."),
        source("definitions", "Metric definitions", "metric_definitions.md", "Audit metric definitions and conclusion labels."),
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Study 1 subreddit-selection provenance audit",
            "description": "A provenance-first audit of the selection chronology and three published subreddit criteria.",
            "generatedAt": "2026-08-27T00:00:00Z",
            "cards": [],
            "charts": [
                {
                    "id": "format_composition_chart",
                    "title": "Worst-case submission-format composition",
                    "subtitle": "Share of historical root submissions; the media upper bound treats every unknown as image/video.",
                    "type": "bar",
                    "dataset": "format_composition",
                    "source": {
                        "id": "format_chart_source",
                        "label": "Submission-format chart transformation",
                        "path": "outputs/submission_format_claims.csv",
                        "query": {
                            "engine": "duckdb",
                            "language": "sql",
                            "description": "Materialise two complementary worst-case format components per subreddit.",
                            "sql": "SELECT subreddit, 'Definite non-media lower bound' AS component, definite_nonmedia_percent_lower_bound AS percent_of_roots, root_submissions, unknown_percent, classification_rule, 'Self-text plus external article/link' AS bound_interpretation FROM read_csv_auto('outputs/submission_format_claims.csv', header=true) WHERE classification_rule = 'metadata_inclusive' UNION ALL SELECT subreddit, 'Media upper bound' AS component, media_percent_upper_bound_if_all_unknown_are_media AS percent_of_roots, root_submissions, unknown_percent, classification_rule, 'Definite image/video plus every unknown' AS bound_interpretation FROM read_csv_auto('outputs/submission_format_claims.csv', header=true) WHERE classification_rule = 'metadata_inclusive'",
                            "tables_used": ["outputs/submission_format_claims.csv"],
                            "filters": ["classification_rule = metadata_inclusive", "No raw Reddit records"],
                        },
                    },
                    "encodings": {
                        "x": {"field": "subreddit", "type": "nominal", "label": "Subreddit"},
                        "y": {"field": "percent_of_roots", "type": "quantitative", "label": "Share of roots (%)", "format": "number"},
                        "color": {"field": "component", "type": "nominal", "label": "Bound"},
                    },
                    "options": {"grouping": "stacked100"},
                    "yAxisTitle": "Share of root submissions (%)",
                    "valueFormat": "number",
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "chronology_table", "title": "Recovered chronology", "dataset": "chronology", "sourceId": "chronology_output",
                    "defaultSort": {"field": "date", "direction": "asc"},
                    "columns": [
                        {"field": "date", "label": "Date", "type": "text"},
                        {"field": "evidence", "label": "Evidence", "type": "text"},
                        {"field": "implication", "label": "Implication", "type": "text"},
                    ],
                },
                {
                    "id": "claim_summary_table", "title": "Claim audit at a glance", "dataset": "claim_summary", "sourceId": "claim_summary_output",
                    "defaultSort": {"field": "subreddit", "direction": "asc"},
                    "columns": [
                        {"field": "subreddit", "label": "Subreddit", "type": "text"},
                        {"field": "complete_months", "label": "Complete UTC months", "format": "number"},
                        {"field": "complete_month_test", "label": ">1,000 every complete month", "type": "text"},
                        {"field": "definite_nonmedia_lower_pct", "label": "Non-media lower bound (%)", "format": "number"},
                        {"field": "media_upper_pct", "label": "Media upper bound (%)", "format": "number"},
                        {"field": "repeat_accounts_raw_pct", "label": "Repeat accounts, raw (%)", "format": "number"},
                    ],
                },
                {
                    "id": "monthly_table", "title": "Every observed UTC month", "dataset": "monthly_detail", "sourceId": "monthly_output",
                    "defaultSort": {"field": "month_utc", "direction": "asc"},
                    "columns": [
                        {"field": "subreddit", "label": "Subreddit", "type": "text"},
                        {"field": "month_utc", "label": "UTC month", "type": "text"},
                        {"field": "root_submissions", "label": "Root submissions", "format": "number"},
                        {"field": "coverage", "label": "Coverage", "type": "text"},
                    ],
                },
                {
                    "id": "monthly_summary_table", "title": "Monthly interpretation checks", "dataset": "monthly_summary", "sourceId": "monthly_output",
                    "defaultSort": {"field": "subreddit", "direction": "asc"},
                    "columns": [
                        {"field": "subreddit", "label": "Subreddit", "type": "text"},
                        {"field": "minimum_observed", "label": "Minimum", "format": "number"},
                        {"field": "median_observed", "label": "Median", "format": "number"},
                        {"field": "mean_observed", "label": "Mean", "format": "number"},
                        {"field": "complete_months", "label": "Complete months", "format": "number"},
                        {"field": "every_complete_month_over_1000", "label": "Every complete month >1,000", "type": "text"},
                    ],
                },
                {
                    "id": "format_table", "title": "Format bounds under metadata-inclusive rule", "dataset": "format_summary", "sourceId": "format_output",
                    "defaultSort": {"field": "subreddit", "direction": "asc"},
                    "columns": [
                        {"field": "subreddit", "label": "Subreddit", "type": "text"},
                        {"field": "definite_media_lower_pct", "label": "Definite media (%)", "format": "number"},
                        {"field": "unknown_pct", "label": "Unknown (%)", "format": "number"},
                        {"field": "media_upper_pct", "label": "Media upper bound (%)", "format": "number"},
                        {"field": "definite_nonmedia_lower_pct", "label": "Definite non-media (%)", "format": "number"},
                        {"field": "robust_nonmedia_majority", "label": "Non-media majority robust", "type": "text"},
                    ],
                },
                {
                    "id": "repeat_table", "title": "Repeat-account comparison", "dataset": "repeat_summary", "sourceId": "repeat_output",
                    "defaultSort": {"field": "subreddit", "direction": "asc"},
                    "columns": [
                        {"field": "subreddit", "label": "Subreddit", "type": "text"},
                        {"field": "source_population", "label": "Population", "type": "text"},
                        {"field": "variant", "label": "Rule", "type": "text"},
                        {"field": "observed_repeat_pct", "label": "Observed repeat (%)", "format": "number"},
                        {"field": "workbook_repeat_pct", "label": "Workbook implied (%)", "format": "number"},
                        {"field": "difference_pp", "label": "Difference (pp)", "format": "number"},
                        {"field": "over_50", "label": ">50%", "type": "text"},
                        {"field": "reproduces_workbook", "label": "Reproduces workbook", "type": "text"},
                    ],
                },
                {
                    "id": "classification_table", "title": "Required conclusion labels", "dataset": "classifications", "sourceId": "classification_output",
                    "defaultSort": {"field": "claim", "direction": "asc"},
                    "columns": [
                        {"field": "claim", "label": "Claim", "type": "text"},
                        {"field": "classification", "label": "Classification", "type": "text"},
                        {"field": "finding", "label": "Finding", "type": "text"},
                    ],
                },
                {
                    "id": "inventory_table", "title": "Source inventory summary", "dataset": "inventory", "sourceId": "inventory_output",
                    "defaultSort": {"field": "strongest_date", "direction": "asc"},
                    "columns": [
                        {"field": "source_id", "label": "Source", "type": "text"},
                        {"field": "kind", "label": "Kind", "type": "text"},
                        {"field": "strongest_date", "label": "Strongest date", "type": "text"},
                        {"field": "processing_status", "label": "Processing status", "type": "text"},
                        {"field": "audit_use", "label": "Audit use", "type": "text"},
                        {"field": "access", "label": "Access/safety", "type": "text"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Study 1 subreddit-selection provenance audit"},
                {"id": "answer", "type": "markdown", "body": "## Answer\n\nThe evidence supports an earlier, separate r/politics collection and preliminary analysis. An executed May 2022 notebook recovers a repeat-account calculation for r/politics, but it does **not** document a three-subreddit selection exercise. No pre-collection artifact applying all three criteria to all three selected subreddits, and no external-statistics source, was recovered.\n\nThe historical exports do retrospectively support high activity, a robust non-image/video majority, and more than 50% repeat accounts under the audit's newly defined full-period rule. Those results are validation and sensitivity findings—not reproduction of the original selection calculation."},
                {"id": "chronology_heading", "type": "markdown", "sourceId": "provenance_record", "body": "## Preliminary politics analysis predates the later 4CAT setup\n\nr/politics was raw by November 2021, cleaned by April 2022, and analysed for account activity by May 2022. A 4CAT installation note records successful setup on 18 October 2022. The Conspiracy and Crypto exports cover 1–30 October, but the exact retrieval/job dates were not recovered; the Crypto export existed by 7 December. The first combined summary is January 2023, the first threshold sheet August 2023, and the first located paper wording September 2024."},
                {"id": "chronology", "type": "table", "tableId": "chronology_table", "layout": "full"},
                {"id": "claims_heading", "type": "markdown", "body": "## Retrospective metrics support the concepts, with material qualifications\n\nOnly r/politics contains an inferred-complete UTC calendar month. Format classification remains robust even if every unknown is treated as media. Every full-period raw-export repeat-account estimate exceeds 50%, but none reproduces the workbook percentage exactly."},
                {"id": "claims", "type": "table", "tableId": "claim_summary_table", "layout": "full"},
                {"id": "monthly_heading", "type": "markdown", "sourceId": "monthly_output", "body": "## Monthly submissions\n\nThe r/Conspiracy and r/CryptoCurrency exports stop on 30 October 2022 and therefore contain no complete UTC calendar month. r/politics has two partial months and one inferred-complete month; its complete October count is 37,724. The partial September count of 57 is why an 'every observed month' interpretation fails for politics while the median and mean interpretations pass."},
                {"id": "monthly_detail", "type": "table", "tableId": "monthly_table", "layout": "full"},
                {"id": "monthly_summary", "type": "table", "tableId": "monthly_summary_table", "layout": "full"},
                {"id": "format_heading", "type": "markdown", "sourceId": "format_output", "body": "## Submission format\n\nThe intended concept is better described as **not image- or video-based**, not simply 'text-based'. External article/link submissions remain non-media links rather than being counted as images or videos. Under the metadata-inclusive sensitivity, definite non-media lower bounds are 62.48% for r/Conspiracy, 87.48% for r/CryptoCurrency, and 93.13% for r/politics."},
                {"id": "format_chart", "type": "chart", "chartId": "format_composition_chart", "layout": "full"},
                {"id": "format", "type": "table", "tableId": "format_table", "layout": "full"},
                {"id": "repeat_heading", "type": "markdown", "sourceId": "repeat_output", "body": "## Repeat-account participation\n\nUsing unique observable accounts over each full export period, raw repeat-account shares are 57.30% (Conspiracy), 53.35% (Crypto), and 59.14% (politics). The May 2022 politics notebook and June aggregate file also yield 59.14%, establishing an early preliminary calculation. The later workbook implies 56.69%, so its activity/window/denominator rule remains unrecovered. Politics-specific documented bot/moderator handling for this metric is unavailable."},
                {"id": "repeat", "type": "table", "tableId": "repeat_table", "layout": "full"},
                {"id": "classification_heading", "type": "markdown", "sourceId": "provenance_record", "body": "## What each conclusion is—and is not\n\nThe May 2022 notebook is a recovered original preliminary r/politics calculation, but no result qualifies as a recovered original subreddit-selection calculation. The workbook is a recovered later summary artifact, while the audit's all-subreddit raw-data calculations are retrospective validation or newly defined sensitivities."},
                {"id": "classification", "type": "table", "tableId": "classification_table", "layout": "full"},
                {"id": "safe_claims", "type": "markdown", "body": "## Recommended thesis-safe claims\n\n- Historical study records show that r/politics was collected, prepared and analysed for account activity before the later 4CAT setup used for the other study exports.\n- Retrospective checks of the archived study exports show high submission volumes during the observed windows, although only r/politics contains a complete UTC calendar month.\n- Under transparent URL/domain and media-metadata rules, all three datasets are predominantly not image- or video-based; external links are treated separately from media.\n- Under a newly defined full-period account rule, more than 50% of observable accounts contributed at least twice in each dataset.\n- A 2023 workbook records similar above-50% complements for the selected subreddits, but its original calculation and pre-collection use could not be recovered.\n\nClaims that the three criteria were applied to all three subreddits before collection, that the preliminary politics analysis determined final selection, that external statistics were used, or that 'monthly' meant every complete month should remain qualified as unresolved."},
                {"id": "viva", "type": "markdown", "body": "## Viva questions arising from unresolved provenance\n\n1. Was there a pre-October-2022 spreadsheet, screenshot, SubredditStats page, Pushshift query, or handwritten candidate list?\n2. Did 'over 1,000 monthly posts' mean a single observed window, every month, a median month, or a mean?\n3. Why does the May 2022 politics calculation yield 59.14% repeat accounts while the later workbook implies 56.69%?\n4. Which deleted, bot, moderator, or service accounts were excluded for r/politics?\n5. Did 'text-based' mean self-text only, or the broader concept 'not image/video', including external article links?\n6. Were r/books and r/TheDonald collected for screening, and when was the five-subreddit candidate pool first decided?\n7. Why do some later methods drafts describe the October datasets as 2021 when the exports contain October 2022 timestamps?\n8. Were external statistics consulted informally even though no citation, screenshot or query record survives?"},
                {"id": "inventory_heading", "type": "markdown", "sourceId": "inventory_output", "body": "## Source inventory and missing evidence\n\nThe package inventories raw exports, cleaned data, historical Git blobs, workbooks, notebooks, paper drafts, thesis notes, collection setup records, recovered L-drive preliminary-analysis files, the bounded Docker archive inspection, and untrusted pickle families. Exact paths, checksums, schemas/contents, periods, and handling decisions are retained in the CSV inventory."},
                {"id": "inventory", "type": "table", "tableId": "inventory_table", "layout": "full"},
                {"id": "methods", "type": "markdown", "sourceId": "definitions", "body": "## Methods, caveats, and reproducibility\n\nAll month assignment uses Unix epoch timestamps in UTC. A root requires empty `parent` and `id == thread_id`; these indicators agreed for every row. The three ZIP exports were streamed without copying records, and all 21 deterministic validation checks passed. Pickles were not opened. The Linux-side notebook was inspected as JSON without execution, and only aggregate account counts were retained. The archived Docker image was inspected through a temporary read-only-derived sparse partition copy; no Reddit records were extracted.\n\nThe package contains the deterministic scripts, SHA-256 archive identities, aggregate CSVs, run record, and exact missing-evidence list."},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": "2026-08-27T00:00:00Z",
            "status": "ready",
            "datasets": {
                "chronology": chronology,
                "claim_summary": claim_summary,
                "monthly_detail": monthly_detail,
                "monthly_summary": monthly_summary,
                "format_summary": format_summary,
                "format_composition": format_composition,
                "repeat_summary": repeat_summary,
                "classifications": classifications,
                "inventory": inventory,
            },
        },
        "sources": sources,
    }
    (ROOT / "artifact.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
