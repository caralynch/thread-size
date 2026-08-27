# Metric definitions

## Scope and source population

The calculations use only the three recovered historical 4CAT/Pushshift CSV
exports stored in ZIP archives. They are retrospective checks against the data
later used in the study; they are not assumed to be a pre-collection screening
population. No present-day Reddit data are used. No pickle is opened.

All calculations use `unix_timestamp` interpreted as seconds since the Unix
epoch in UTC. The human-readable `timestamp` column is retained only for an
offset-consistency check. A contribution is a row representing either a root
submission or a comment.

## Root submissions and monthly counts

A row is a root submission only when both conditions hold:

1. `parent` is empty; and
2. `id` equals `thread_id`.

Counts are grouped by UTC calendar month. A month is marked
`complete_inferred_from_timestamps` only when the observed dataset begins no
later than five minutes after the first instant of that month and ends no
earlier than five minutes before the next month. This is an inference from the
high-volume event timestamps, not a recovered collection instruction or log.
All other observed months are partial.

For each subreddit, the outputs report the count for every observed month and
the minimum, median, and arithmetic mean across:

- every observed month, including partial months; and
- inferred-complete months only.

The “over 1,000 monthly posts” statement is evaluated separately as:

- every observed month exceeds 1,000;
- the median observed month exceeds 1,000;
- the mean observed month exceeds 1,000; and
- every inferred-complete month exceeds 1,000.

If no complete calendar month is present, the final test is unresolved rather
than vacuously true.

## Submission format

Format is classified for root submissions only. Two deterministic rules are
reported.

### URL/domain-only rule

- `video`: a recognised video-only host or video file extension;
- `image`: a recognised image host or image file extension;
- `self_text`: a `self.<subreddit>` domain;
- `external_article_or_link`: a non-Reddit URL/domain not already classified
  as image or video; and
- `unknown`: an internal Reddit non-self URL or insufficient metadata.

### Metadata-inclusive rule

This starts with the URL/domain rule, additionally classifies a root as an
image when 4CAT's `image_file` or `image_md5` metadata are present, and
classifies a no-URL root with a non-empty body as self-text. Because 4CAT image
metadata may reflect downloaded image material rather than a fully recovered
Reddit post-type field, this is a sensitivity rule, not the privileged answer.

For each rule, the definite media lower bound is `image + video`. The media
upper bound treats every unknown as media. The claim “predominantly not image-
or video-based” is robust to missing classification only when
`self_text + external_article_or_link` exceeds 50% of roots. External links are
never equated with image/video posts merely because they are links.

## Repeat-account participation

The denominator is the number of unique observable, non-empty account labels
with at least one submission or comment in the full recovered export period.
The numerator is the number of those labels with at least two total
contributions in that same period. The reported percentage is
`100 × numerator / denominator`.

Four variants are reported:

1. `observable_nonblank_unfiltered`: includes every non-empty pseudonymous
   account label;
2. `literal_placeholders_excluded`: excludes literal deleted/removed labels if
   they survived pseudonymisation;
3. `documented_deleted_placeholder_excluded`: applies the dummy removed-
   account rule recorded in the 16 January 2023 Git version of
   `authors_to_remove.csv`; and
4. `documented_2023_cleaning_rules`: applies all applicable all-contribution or
   comment-only exclusions recorded in that file.

The historical exclusion file contains subreddit flags for `conspiracy` and
`crypto` but not `politics`; documented-rule variants for r/politics are
therefore unresolved. The rule file identifies deleted, bot, moderator, and
service/spam cases in prose. Account labels are used in memory only and are not
written to the audit package.

These are sensitivity analyses under newly defined rules unless a report row
explicitly says otherwise. They do not recover the unknown calculation that
produced the 2023 workbook percentages.

## Conclusion labels

Every substantive report conclusion uses one of four labels:

- **recovered original selection calculation** — an original, dated
  calculation and its inputs/rules were recovered;
- **retrospective validation using later study data** — the historical study
  exports were used after collection to test a claim;
- **sensitivity analysis under a newly defined rule** — the audit defines and
  varies a rule whose original handling is unavailable; or
- **unresolved** — required source evidence or an original rule is missing.
