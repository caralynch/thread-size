# Endpoint follow-up audit package

This directory is a self-contained, aggregate-only audit of endpoint follow-up
and possible right-censoring in the final Study 1 thread-initiation and
thread-size analyses.

Start with [`report.md`](report.md). Source lineage and endpoint provenance are
documented in [`provenance.md`](provenance.md), and exact definitions are in
[`metric_definitions.md`](metric_definitions.md).

## Guardrails

- The frozen modelling and raw-data mirrors are read only.
- No thesis prose is read or edited by the audit code.
- No model is loaded, fitted, tuned, ranked, or selected.
- No prediction is generated or changed; saved hard predictions are used.
- Outputs contain aggregate counts, rates, metrics, paths, schemas, and
  checksums only. They contain no Reddit text, account names, or post/comment
  identifiers.
- The original observed-size outcome is retained. No fixed-horizon target is
  substituted.

## Code

- [`audit_endpoint_followup.py`](audit_endpoint_followup.py): reconstructs the
  exact final population and split, validates saved prediction linkage, measures
  exposure and reply timing, and recomputes sensitivity metrics from frozen
  decisions.
- [`raw_archive_diagnostics.py`](raw_archive_diagnostics.py): streams the three
  checksum-matched published raw ZIPs and emits aggregate raw endpoint checks.
  It uses a two-pass identifier join and does not extract the ZIPs.
- [`validate_outputs.py`](validate_outputs.py): independently checks structural
  consistency, count/rate reconciliation, source hashes, and output privacy.

Python 3.12 with pandas, NumPy, and PyArrow is required for the main audit. The
raw diagnostic and independent validator use the Python standard library.

## Reproduction

From the repository root on Windows, with the extracted frozen mirror available
at `L:\Documents\reddit_analyses\thread-size`:

```powershell
python audits\endpoint_followup\audit_endpoint_followup.py
python audits\endpoint_followup\validate_outputs.py
```

The raw scan is substantially faster when run natively on `linuxbox` rather
than through SSHFS:

```bash
python3 raw_archive_diagnostics.py \
  --archive conspiracy=/home/cara/Documents/reddit_analyses/conspiracy_oct2022.zip \
  --archive crypto=/home/cara/Documents/reddit_analyses/cryptocurrency_oct2022.zip \
  --archive politics=/home/cara/Documents/reddit_analyses/politics_nov2020.zip \
  --output-dir outputs
```

For the recorded run, the script was copied temporarily to `/tmp` on
`linuxbox`, only the three aggregate outputs were copied back, and the remote
temporary script/output directory was removed. The raw archives were not
copied or extracted.

## Output map

- Source/endpoints: `source_inventory.csv`, `raw_source_inventory.csv`,
  `collection_endpoints_used.csv`, `data_coverage.csv`, and the two run records.
- Population/configuration: `population_reconciliation.csv`,
  `model_configuration.csv`, and `target_class_definitions.csv`.
- Direct exposure: `endpoint_exposure.csv`.
- Observed timing: `reply_timing_comment_cumulative.csv` and
  `reply_timing_first_comment.csv`.
- Sensitivity: `sensitivity_performance.csv`,
  `sensitivity_class_metrics.csv`, and
  `sensitivity_confusion_matrices.csv`.
- Preliminary checks: `preliminary_lead_comparison.csv` and
  `raw_archive_diagnostics.csv`.
- Validation: `validation_checks.csv`, `raw_archive_validation.csv`, and
  `package_validation.csv`.

`outputs/package_manifest.csv` records package checksums after final assembly.
