# Endpoint follow-up audit: metric and population definitions

## Populations

- **Final cleaned root population:** the rows in `Inputs/{subreddit}_threads.parquet` in the frozen extracted mirror of Zenodo record `10.5281/zenodo.17831100`.
- **Reconstructed modelling population:** the same roots after the frozen feature-construction step, sorted by `timestamp` exactly as in `0_Preprocessing/2_tf_idf_analysis.py`.
- **Held-out population:** rows from `int(N * 0.8)` to `N - 1` after that chronological sort. The audit must match this row order, timestamps, target values and row count to the frozen enriched test artefact before any endpoint or performance calculation is accepted.
- **Stage 1 observed class:** `Stalled` when `thread_size == 1`; `Started` when `thread_size > 1`, matching the frozen model code's `log_thread_size > log(1)` rule.
- **Stage 2 observed class:** the saved selected-model `true_class`, independently checked against the selected model's frozen log-scale bin edges and the model-data `y_test.parquet`. Class names are `Stalled`, `Small`, `Medium`, and `Large`.

`thread_size` is the original observed-size outcome. Endpoint exclusions do not redefine it and cannot recover comments that are absent from the archive.

## Endpoint and exposure

The acquisition-job clock time was not recovered. For each subreddit the audit therefore uses a **documented coverage-boundary proxy**: midnight UTC immediately after the last inclusive date in the raw-data Zenodo record. This is a content-time archive boundary, not a claim about when the export job ran.

For root `i`:

`potential_followup_hours_i = (coverage_boundary_UTC - root_timestamp_UTC) / 1 hour`

A root has limited follow-up at horizon `h` when `potential_followup_hours_i < h`. Horizons are 1, 6, 12, 24, 48, 72 and 168 hours. Counts and percentages use the complete held-out population as denominator, or the complete held-out observed class when results are stratified.

## Observed reply timing

- A descendant comment is linked through stable `thread_id`.
- Comment delay is `comment_timestamp - root_timestamp` in hours.
- Negative delays are invalid for timing calculations and are counted in validation outputs.
- **Horizon-specific cohort:** at horizon `h`, only roots with at least `h` hours of potential follow-up are eligible. The cumulative comment percentage is the share of all observed descendant comments attached to those eligible roots whose delay is at most `h`.
- **Common seven-day cohort:** for comparability across horizons, a second profile restricts every horizon to roots with at least 168 hours of potential follow-up.
- Time-to-first-comment summaries use roots with at least seven days of potential follow-up and at least one observed non-negative-delay descendant comment.

These are distributions of observed comments. They do not estimate unobserved comments after the coverage boundary.

## Frozen-prediction sensitivity

The saved selected-model hard predictions and labels are joined to reconstructed held-out roots by the preserved zero-based test-row index. No prediction is regenerated.

For minimum follow-up thresholds 24, 48, 72 and 168 hours, roots below the threshold are excluded and the following are recomputed:

- retained and excluded sample sizes and rates;
- observed class prevalence;
- Matthews correlation coefficient (multiclass form for both stages);
- balanced accuracy (unweighted mean of per-class recall);
- per-class recall; and
- confusion matrices as counts, within-true-class rates and whole-subset rates.

The full held-out result is included as the reference. A subset is marked `not_interpretable` if a class is absent, `caution_small` if `N < 500` or any observed class has fewer than 50 roots, and otherwise `adequate_descriptive`. These labels are transparent rules of thumb, not confidence intervals.

## Preliminary leads

The supplied approximate final-24-hour raw-root and within-24-hour observed-comment percentages are treated as hypotheses. Raw-archive diagnostics and final-cleaned-sample diagnostics are reported separately. Agreement is assessed on the unrounded value; no preliminary number controls an endpoint or model choice.
