# Provenance record: Study 1 endpoint follow-up audit

## Audit boundary

This record covers the provenance needed to audit follow-up opportunity and
possible endpoint censoring in the final Study 1 thread-initiation and
thread-size analyses. The completed subreddit-selection audit was inspected
only as a locator for historical sources. None of its substantive conclusions
or aggregate values were imported as endpoint findings.

No thesis file, raw or prepared dataset, trained model, model metadata file, or
saved prediction file was edited. The audit loaded no trained model and did not
fit, tune, rank, select, or regenerate any model or prediction.

## Repository and deposit versions

- Working repository HEAD at the final audit run:
  `1051e881a2d0fe4a9ae7886d2d3b2985d0da610a` (`main`).
- The current repository is a later publication/modelling repository. Its first
  located history is later than the original data collection.
- The sibling historical `thread_size_prediction` repository begins at
  `f44ea082d097f4f23974b0eef9a41ed63de5bc7b` on 16 January 2023; commit
  `30deca2d6971c333f98b0a4df42b3eade338c62c` records politics being added to
  the combined datasets on 7 February 2023.
- Frozen modelling artefacts: [Zenodo 10.5281/zenodo.17831100](https://doi.org/10.5281/zenodo.17831100),
  version 1.0, extracted read-only mirror at
  `L:\Documents\reddit_analyses\thread-size`.
- Published raw archives and coverage descriptions:
  [Zenodo 10.5281/zenodo.17079717](https://doi.org/10.5281/zenodo.17079717),
  version 1.0, checksum-matched local mirror at
  `L:\Documents\reddit_analyses`.

The path-level final-artefact inventory, schemas, row counts, timestamps,
versions, provenance labels, file sizes, and SHA-256 checksums are in
[`outputs/source_inventory.csv`](outputs/source_inventory.csv). The raw ZIP
inventory is in
[`outputs/raw_source_inventory.csv`](outputs/raw_source_inventory.csv).

## Raw archive verification

| Subreddit | Published file | Rows (roots; comments) | MD5 matched to record | UTC content range |
|---|---|---:|---|---|
| conspiracy | `conspiracy_oct2022.zip` | 455,437 (11,436; 444,001) | `0fc82cae91d2c78b6d1581f073dbfc04` | 2022-10-01 00:00:02 to 2022-10-30 23:59:58 |
| crypto | `cryptocurrency_oct2022.zip` | 476,948 (14,842; 462,106) | `7128dcd7784077e77716b835cc760bbd` | 2022-10-01 00:00:04 to 2022-10-30 23:59:50 |
| politics | `politics_nov2020.zip` | 7,124,674 (67,920; 7,056,754) | `f7f8545542840b5353c4b1da230289cb` | 2020-09-30 23:00:02 to 2020-11-19 23:59:58 |

The raw `timestamp` strings agree with `unix_timestamp` under a UTC
interpretation for every row. Each raw CSV has one timestamp-order reversal.
Consequently, the raw diagnostic uses a two-pass identifier join and does not
depend on file order. This is recorded as a source warning, not a failed join.

The October schemas contain `thread_id`, `id`, `timestamp`, `body`, `subject`,
`author`, `author_flair`, `post_flair`, `image_file`, `domain`, `url`,
`image_md5`, `subreddit`, `parent`, `score`, and `unix_timestamp`. The politics
schema contains the same core fields without `author_flair` and `post_flair`.
No values from identifier, text, account, URL, or image fields are emitted by
the audit.

## Endpoint determination

No acquisition-job receipt, query completion time, export receipt, or 4CAT job
record linking these three exports to a precise acquisition clock time was
recovered. The maximum observed timestamp is therefore **not** silently treated
as the acquisition endpoint.

The raw-data record documents inclusive content dates. The audit converts each
last inclusive date to midnight UTC at the start of the next day:

| Subreddit | Documented coverage-boundary proxy | Basis | Uncertainty |
|---|---|---|---|
| conspiracy | 2022-10-31 00:00:00 UTC | coverage through 2022-10-30 | Not an acquisition clock time |
| crypto | 2022-10-31 00:00:00 UTC | coverage through 2022-10-30 | Not an acquisition clock time |
| politics | 2020-11-20 00:00:00 UTC | coverage through 2020-11-19 | Not an acquisition clock time |

The near-boundary raw maxima support continuous content coverage to the end of
the documented dates, but they do not remove the acquisition-time uncertainty.
The machine-readable decisions and caveats are in
[`collection_endpoints.csv`](collection_endpoints.csv).

## Final analysis lineage

For each subreddit, the audit traced the following frozen lineage:

1. Final cleaned roots: `Inputs/{subreddit}_threads.parquet`.
2. Final cleaned descendant comments: `Inputs/{subreddit}_comments.parquet`.
3. Root-level feature output:
   `Outputs/0_preprocessing/{subreddit}/{subreddit}_threads_extra_feats.parquet`.
4. The root rows are sorted by `timestamp`; the split index is
   `int(N * 0.8)`. Rows before it are training and rows from it onward are the
   later held-out population, as implemented by the frozen preprocessing code.
5. Root identifiers and order were reconciled to
   `tf-idf/{subreddit}_svd_enriched_test_data.parquet`; targets were reconciled
   to `{subreddit}_test_Y.parquet`.
6. Stage 1 uses the selected saved prediction parquet
   `Outputs/1_thread_start/{subreddit}/4_model/model_4/test_started_threads.parquet`.
7. Stage 2 uses the selected saved `test_preds.csv` and `model_data/y_test.parquet`
   under model 3 for conspiracy, model 2 for crypto, and model 3 for politics.
8. Saved hard decisions are linked to roots using the preserved zero-based
   held-out row index. No prediction was regenerated.

The selected Stage 1 configuration uses four features for every subreddit.
The selected Stage 2 configuration uses three features for conspiracy, two for
crypto, and three for politics. Exact feature names, weights, Stage 1 decision
thresholds, and Stage 2 bin edges are preserved in
[`outputs/model_configuration.csv`](outputs/model_configuration.csv).

## Identifier and join status

- Final cleaned root `thread_id` is non-null and unique for every subreddit.
- Stable comment `id` and root-link `thread_id` are present.
- The reconstructed chronological held-out root identifiers and order match
  the frozen enriched test artefacts exactly.
- Stage 1 and Stage 2 saved predictions join one-to-one by held-out row index.
- Stage 2 saved labels match both `y_test.parquet` and independently reconstructed
  classes from the frozen bin edges.
- Final cleaned comments have complete root-link coverage and reproduce the
  saved per-root observed `thread_size` counts.
- No blocking mismatch remains. All primary checks are in
  [`outputs/validation_checks.csv`](outputs/validation_checks.csv).

## Provenance statuses

- `frozen_final_artefact`: file in the extracted version-1.0 modelling deposit;
  used directly and hashed by the audit.
- `published_raw_archive_checksum_matched_local_mirror`: local raw ZIP whose
  MD5 equals the published record; streamed read-only for aggregate checks.
- `documented_coverage_boundary_proxy`: endpoint derived from published
  inclusive coverage dates; suitable for content-time opportunity calculations
  but not claimed as the acquisition completion time.
- `historical_locator_only`: earlier audit material used to find sources, not
  used as endpoint evidence or as a numerical input.

## Remaining provenance uncertainty

The principal unresolved provenance item is the acquisition/export completion
time. Recovering a 4CAT job receipt or equivalent acquisition log could refine
the endpoint from a date-boundary proxy to a true collection clock. It would
not, by itself, reveal replies never captured in the archived data.
