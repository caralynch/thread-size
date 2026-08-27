# Study 1 endpoint follow-up and possible right-censoring audit

Audit date: 27 August 2026  
Repository state: `1051e881a2d0fe4a9ae7886d2d3b2985d0da610a`

## Bottom line

Endpoint exposure is **not negligible**. It is temporally localised to the end
of each observation window, but it is **potentially material for the completeness
of the observed thread-size target**, especially when “adequate follow-up” is
defined as 48 hours or longer. In the final held-out populations, 9.93%–16.50%
of roots had less than 24 hours of potential follow-up; 19.64%–33.70% had less
than 48 hours; and 28.85%–50.55% had less than 72 hours. Neither October
held-out set contains any root with seven complete days, and only 31.10% of the
politics held-out set does.

This exposure does **not** translate into a material change in the headline
held-out performance pattern when roots with limited opportunity are excluded.
Across usable 24-, 48-, and 72-hour subsets, MCC and balanced accuracy are close
to or slightly above the full held-out estimates. The qualitative ordering is
unchanged: Stage 1 remains strongest for politics, then crypto, then conspiracy;
Stage 2 remains appreciably weaker than Stage 1 in all three datasets. The
politics seven-day subset is highly selected and shows modestly lower metrics,
not a reversal. Seven-day sensitivity is unavailable for conspiracy and crypto.

These are complementary findings, not substitutes. Exposure is directly
measurable relative to a documented coverage-boundary proxy. Exclusion
sensitivity describes the observed saved labels and predictions after temporal
selection; it cannot recover or estimate comments that were never archived.

## Evidence status

The final populations and saved decisions were reproduced exactly from the
frozen version-1.0 modelling deposit
([Zenodo 10.5281/zenodo.17831100](https://doi.org/10.5281/zenodo.17831100)).
The published raw ZIPs in the local mirror match the MD5 checksums in the
version-1.0 raw record
([Zenodo 10.5281/zenodo.17079717](https://doi.org/10.5281/zenodo.17079717)).

No acquisition-job completion time was recovered. The endpoints are therefore
explicit **documented coverage-boundary proxies**, not maximum observed
timestamps and not claimed acquisition times:

- conspiracy and crypto: 2022-10-31 00:00:00 UTC;
- politics: 2020-11-20 00:00:00 UTC.

Raw `timestamp` and `unix_timestamp` fields agree under UTC. Raw content reaches
within 2–10 seconds of each proxy boundary, which supports the documented date
coverage but does not remove the acquisition-time uncertainty. See
[`provenance.md`](provenance.md) and
[`outputs/collection_endpoints_used.csv`](outputs/collection_endpoints_used.csv).

## Exact final-population reconciliation

| Subreddit | Final cleaned roots | First 80% | Later 20% held out | Held-out UTC range |
|---|---:|---:|---:|---|
| conspiracy | 11,395 | 9,116 | 2,279 | 2022-10-25 04:43:36 to 2022-10-30 23:55:25 |
| crypto | 14,818 | 11,854 | 2,964 | 2022-10-25 14:13:07 to 2022-10-30 23:52:09 |
| politics | 65,343 | 52,274 | 13,069 | 2020-11-10 23:19:02 to 2020-11-19 23:59:04 |

For every subreddit, the audit matched root identifiers, chronological order,
targets, and row counts to the frozen preprocessing/test artefacts. Root IDs are
unique; comment IDs are unique; all final cleaned comments join to a retained
root; per-root linked-comment counts reproduce `thread_size`; and the selected
Stage 1 and Stage 2 saved predictions join one-to-one by the preserved held-out
row index. All 84 primary reconciliation checks pass. The audit did not load a
model or regenerate a prediction.

## Direct finding 1: opportunity for follow-up

Counts below use the complete final held-out population as denominator.

| Subreddit | <1 h | <6 h | <12 h | <24 h | <48 h | <72 h | <7 d |
|---|---:|---:|---:|---:|---:|---:|---:|
| conspiracy (N=2,279) | 21 (0.92%) | 118 (5.18%) | 214 (9.39%) | 376 (16.50%) | 768 (33.70%) | 1,152 (50.55%) | 2,279 (100.00%) |
| crypto (N=2,964) | 15 (0.51%) | 121 (4.08%) | 268 (9.04%) | 426 (14.37%) | 907 (30.60%) | 1,439 (48.55%) | 2,964 (100.00%) |
| politics (N=13,069) | 102 (0.78%) | 526 (4.02%) | 957 (7.32%) | 1,298 (9.93%) | 2,567 (19.64%) | 3,771 (28.85%) | 9,005 (68.90%) |

Exposure is present in every observed target class, rather than being confined
to one outcome. At 24 hours, the most exposed Stage 2 class is Small in
conspiracy (17.28%), Large in crypto (17.02%), and Small in politics (12.26%).
At 72 hours those class-specific maxima are 52.97%, 53.46%, and 32.93%,
respectively. For Stage 1, Started roots are more exposed than Stalled roots in
conspiracy and politics; Stalled roots are slightly more exposed in crypto.
All counts and percentages for every class and horizon are in
[`outputs/endpoint_exposure.csv`](outputs/endpoint_exposure.csv).

## Direct finding 2: observed reply timing among eligible roots

The following percentages are comment-weighted. At each horizon, a root is
eligible only if the documented archive boundary covers that complete horizon.
The denominator therefore changes with the horizon; the accompanying CSV also
provides a common seven-day cohort for like-for-like comparison.

| Subreddit | ≤1 h | ≤6 h | ≤12 h | ≤24 h | ≤48 h | ≤72 h | ≤7 d |
|---|---:|---:|---:|---:|---:|---:|---:|
| conspiracy | 21.46% | 57.02% | 77.04% | 93.83% | 98.01% | 98.85% | 99.69% |
| crypto | 32.71% | 63.24% | 78.39% | 98.02% | 99.57% | 99.72% | 99.88% |
| politics | 20.15% | 70.24% | 91.29% | 98.59% | 99.56% | 99.71% | 99.88% |

For example, the 24-hour rows use 11,019 conspiracy roots and 356,131 observed
descendant comments; 14,392 crypto roots and 382,152 comments; and 64,045
politics roots and 3,519,693 comments. Thus, most *observed* comment activity is
early, but this cannot demonstrate that late-window roots had no additional
unobserved replies.

Among roots with at least seven days of coverage, the time-to-first-comment
profile is:

| Subreddit | Eligible roots | Roots with an observed comment | Median first comment among commented roots | 90th percentile |
|---|---:|---:|---:|---:|
| conspiracy | 8,634 | 7,307 (84.63%) | 2.38 minutes | 19.29 minutes |
| crypto | 10,970 | 5,167 (47.10%) | 2.05 minutes | 8.65 minutes |
| politics | 56,338 | 36,323 (64.47%) | 1.77 minutes | 7.40 minutes |

The full observed timing outputs are
[`outputs/reply_timing_comment_cumulative.csv`](outputs/reply_timing_comment_cumulative.csv)
and [`outputs/reply_timing_first_comment.csv`](outputs/reply_timing_first_comment.csv).

## Preliminary leads: reproduced or rejected

The raw and cleaned populations remain separate.

| Subreddit | Raw roots in final 24 h | Preliminary lead | Matched raw comments ≤24 h among 24 h-eligible roots | Preliminary lead | Status |
|---|---:|---:|---:|---:|---|
| conspiracy | 376 / 11,436 = 3.288% | ≈3.29% | 364,120 / 388,242 = 93.787% | ≈94.0% | Root lead reproduced; comment lead close but rejected at the declared 0.1-point tolerance |
| crypto | 428 / 14,842 = 2.884% | ≈2.88% | 406,914 / 416,247 = 97.758% | ≈97.8% | Both reproduced |
| politics | 1,494 / 67,920 = 2.200% | ≈2.20% | 5,362,891 / 5,429,318 = 98.777% | ≈98.8% | Both reproduced |

The raw diagnostic uses a two-pass `thread_id` join because each archive has
one timestamp-order reversal. Comments whose roots are absent from the same raw
archive are excluded from the matched-comment denominator and are reported
separately. See [`outputs/raw_archive_diagnostics.csv`](outputs/raw_archive_diagnostics.csv)
and [`outputs/raw_archive_validation.csv`](outputs/raw_archive_validation.csv).

## Sensitivity evidence: unchanged frozen decisions

Each cell reports retained N, MCC, and balanced accuracy. “≥24 h” means roots
with less than 24 hours of potential follow-up were excluded. No model or
decision rule changed.

| Subreddit / stage | Full held out | ≥24 h | ≥48 h | ≥72 h | ≥7 d |
|---|---|---|---|---|---|
| conspiracy Stage 1 | N=2,279; .309; .559 | N=1,903; .308; .558 | N=1,511; .312; .561 | N=1,127; .329; .569 | unavailable (N=0) |
| conspiracy Stage 2 | N=2,279; .177; .381 | N=1,903; .174; .380 | N=1,511; .180; .387 | N=1,127; .190; .396 | unavailable (N=0) |
| crypto Stage 1 | N=2,964; .539; .748 | N=2,538; .543; .750 | N=2,057; .547; .753 | N=1,525; .560; .756 | unavailable (N=0) |
| crypto Stage 2 | N=2,964; .330; .403 | N=2,538; .329; .400 | N=2,057; .330; .401 | N=1,525; .334; .401 | unavailable (N=0) |
| politics Stage 1 | N=13,069; .685; .825 | N=11,771; .693; .831 | N=10,502; .696; .833 | N=9,298; .697; .834 | N=4,064; .674; .823 |
| politics Stage 2 | N=13,069; .313; .468 | N=11,771; .317; .468 | N=10,502; .322; .470 | N=9,298; .327; .470 | N=4,064; .295; .456 |

At 24 hours, exclusion changes any observed class prevalence by at most 0.65
percentage points. At 72 hours, the largest changes are still modest: politics
Stalled rises by 2.55 points; conspiracy Stalled rises by 1.40 points and Small
falls by 1.52 points; crypto Small rises by 1.22 points and Large falls by 1.21
points. Per-class recalls likewise remain broadly stable. The largest 72-hour
recall change is a 7.13-point increase for conspiracy Stalled in Stage 2; other
changes do not alter the overall interpretation. Complete prevalence, recall,
and count tables are in
[`outputs/sensitivity_class_metrics.csv`](outputs/sensitivity_class_metrics.csv), and
all count and rate confusion matrices are in
[`outputs/sensitivity_confusion_matrices.csv`](outputs/sensitivity_confusion_matrices.csv).

Every non-empty subset passes the prespecified descriptive sample rule
(`N ≥ 500` and at least 50 observations in every observed class). Nevertheless,
the 72-hour October subsets and seven-day politics subset require caution
because they remove approximately half and 68.90% of their original held-out
populations, respectively. They are earlier-period temporal subsets, not
random samples. The empty October seven-day subsets are not interpretable.

## Interpretation

### What is directly supported

- Endpoint exposure exists and is substantial at 48–72 hours in the late
  held-out periods.
- Exposure is distributed across all outcomes, with modest class differences.
- Most observed comments attached to roots with complete coverage arrive within
  24 hours, especially in crypto and politics.
- Removing roots with limited opportunity does not materially weaken the
  reported held-out performance pattern under the unchanged saved decisions.

### What is not supported

- The audit cannot infer the number or class effect of comments that were never
  captured.
- Stable exclusion metrics do not prove that the original target is uncensored.
- The date-boundary proxy is not a recovered acquisition clock.
- A fixed-horizon outcome was not substituted for the original observed-size
  target and was not analysed here. It remains an optional, separately labelled
  further sensitivity.

## Assumptions and unresolved questions

1. Naive prepared timestamps are interpreted as UTC because their raw source
   strings agree with UNIX seconds under UTC. This is fully validated in the
   checksum-matched raw archives.
2. The documented inclusive coverage dates are converted to next-midnight UTC
   proxies. A recovered acquisition-job receipt could refine these endpoints.
3. The audit observes the final cleaned archive, not Reddit’s complete historical
   state. Deletions, removals, source API gaps, and post-export replies remain
   unobservable.
4. Reply-timing summaries are comment-weighted; a high-volume root contributes
   more observations. Root-level time-to-first-comment is reported separately.
5. Temporal exclusions may change event context, posting time, or covariates.
   They are robustness subsets, not causal adjustments.

## Required conclusion statements

- **Negligible, localised, or potentially material:** potentially material for
  target completeness, although temporally localised to the end of each archive
  window; not negligible at 24–72-hour horizons.
- **Most exposed outcomes and subreddit-periods:** the held-out tails of the two
  October 2022 datasets are most exposed at 48–72 hours and have no seven-day
  eligible observations. Small conspiracy and Large crypto are the most exposed
  Stage 2 classes at 24–72 hours; Small politics is most exposed at 24–72 hours.
  Class differences are modest compared with the shared calendar-tail effect.
- **Headline performance conclusions:** unchanged under all usable 24-, 48-,
  and 72-hour exclusions. Politics at seven days is modestly lower but remains
  qualitatively consistent; the October seven-day comparison is unavailable.

## Thesis-safe Methods wording

> We conducted a post hoc endpoint follow-up audit using the final cleaned root
> and comment data and the frozen held-out predictions. In the absence of a
> recoverable acquisition-job completion time, potential follow-up was measured
> from each root timestamp to midnight UTC immediately after the final inclusive
> date documented for the corresponding raw archive. We therefore treat this as
> a documented content-coverage boundary proxy rather than the acquisition
> clock time. The final chronological split was reconstructed by ordering roots
> by timestamp and assigning the first `int(N × 0.8)` observations to training
> and the remainder to the held-out set; identifiers, order, labels, and counts
> were reconciled to the frozen final artefacts. Using the saved hard predictions
> without refitting or retuning, held-out metrics were recomputed after excluding
> roots with less than 24, 48, 72, and 168 hours of potential follow-up.

## Thesis-safe Limitations wording

> Thread size was defined by the comments observed in the archived data rather
> than by a fixed post-publication horizon. Consequently, roots near the end of
> an archive window had less opportunity for later replies to be captured. In
> the final held-out sets, 9.93%–16.50% of roots had less than 24 hours of
> potential follow-up and 28.85%–50.55% had less than 72 hours, relative to the
> documented coverage-boundary proxy. Most comments observed for roots with
> complete horizon coverage arrived within 24 hours (93.83%–98.59%), and the
> qualitative held-out performance conclusions were unchanged when roots with
> limited 24-, 48-, or 72-hour follow-up were excluded. However, these checks
> cannot recover replies absent from the archive, and the endpoint is a
> date-derived content-coverage proxy because the acquisition-job completion
> time was not recovered. The seven-day comparison was unavailable for the two
> October datasets and highly selective for politics.

## Remaining viva questions

1. Why is a documented date-boundary proxy defensible, and what evidence would
   be needed to call it a true acquisition endpoint?
2. Why does a high proportion of observed comments arriving within 24 hours not
   prove that endpoint censoring is absent?
3. Why is excluding late roots a sensitivity analysis rather than a correction
   for missing outcomes?
4. How might temporal selection or event context explain small improvements in
   metrics after 48- or 72-hour exclusions?
5. Why was the original observed-size target retained instead of silently
   replacing it with a fixed-horizon target?
6. How were saved predictions linked to roots without rerunning the models, and
   what evidence establishes that the linkage is exact?
7. Would the substantive interpretation change if a future acquisition log
   established a later endpoint or if threads were recollected after a fixed
   follow-up period?
8. Why do raw and final cleaned archive denominators differ, and why must their
   diagnostics be reported separately?

## Validation and audit trail

- Primary reconciliation: 84/84 checks pass in
  [`outputs/validation_checks.csv`](outputs/validation_checks.csv).
- Independent package validation: 140/140 checks pass in
  [`outputs/package_validation.csv`](outputs/package_validation.csv).
- Raw validation: no failures; three documented file-order warnings in
  [`outputs/raw_archive_validation.csv`](outputs/raw_archive_validation.csv).
- Full metric definitions and stability rules:
  [`metric_definitions.md`](metric_definitions.md).
- Deterministic code:
  [`audit_endpoint_followup.py`](audit_endpoint_followup.py),
  [`raw_archive_diagnostics.py`](raw_archive_diagnostics.py), and
  [`validate_outputs.py`](validate_outputs.py).

No thesis file was edited.
