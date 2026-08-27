# Subreddit-selection provenance audit

This folder is an aggregate-only, reproducible audit of the subreddit-
selection claims used in the Study 1 thread-initiation and thread-size paper.
It does not alter thesis prose, rerun models, or open pickle files.

The primary human-readable deliverable is `report.html` once packaged. Exact
results are in `outputs/`; definitions are in `metric_definitions.md`; source
and chronology findings are in `provenance.md` and `source_inventory.csv`.

The provenance record also includes a bounded read-only follow-up on the Linux
`L:` source. That follow-up recovered a May 2022 r/politics account-activity
notebook and June 2022 aggregate files, but no three-subreddit screening record
or surviving 4CAT job receipt. No raw Reddit rows, text, URLs, domains, or
account labels were copied into this package.

## Reproduction

Run `analyze_selection_sources.py` with one historical ZIP archive for each of
`politics`, `crypto`, and `conspiracy`, the historical
`thread_size_prediction` Git repository, and an output folder. The audit run
used the `--hash-inputs` option so that exact ZIP SHA-256 checksums are recorded.

The script accepts only ZIP/CSV sources for the Reddit calculations and
explicitly rejects pickle extensions. It writes aggregate counts and metadata
only; no Reddit record, text, URL, domain, or account label is emitted.

## Output guide

- `outputs/dataset_coverage.csv`: row totals, UTC periods, and timestamp checks;
- `outputs/monthly_submissions.csv`: every UTC calendar-month root count;
- `outputs/monthly_submission_summary.csv`: minimum/median/mean and separate
  threshold interpretations;
- `outputs/submission_format_counts.csv`: format counts and classification
  reasons for both rules;
- `outputs/submission_format_claims.csv`: media lower/upper bounds;
- `outputs/repeat_account_summary.csv`: account-participation sensitivities;
- `outputs/repeat_account_comparison.csv`: raw/prepared estimates joined to the
  recovered workbook percentages;
- `outputs/prepared_politics_account_comparison.csv`: aggregate-only check of
  the dated cleaned politics CSV;
- `outputs/recovered_workbook_thresholds.csv`: safe extraction of the workbook
  constants and their complements;
- `outputs/source_inventory_computed.csv`: checksums and schemas for sources
  actually processed;
- `outputs/validation_checks.csv`: deterministic integrity checks; and
- `outputs/package_validation.csv`: package-level cross-checks;
- `outputs/run_record.json`: exact run configuration.

`package_manifest.csv` gives a SHA-256 checksum for each package file. The
portable report builder passed artifact validation and structural verification.
Browser-level QA was unavailable because no compatible headless Chromium was
installed; the self-contained semantic HTML fallback remains part of
`report.html`.
