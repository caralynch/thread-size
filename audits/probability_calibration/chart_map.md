# Chart map

Two task-specific appendix figures are generated from `outputs/reliability_bins.csv`.

## Thread initiation

- Files: `figures/thread_initiation_reliability.png` and `.svg`.
- Layout: one horizontal panel per subreddit.
- Series: saved calibrated probability of Started.
- Intended reading: compare each empirical reliability path with the dashed equality line.

## Four-class thread size

- Files: `figures/thread_size_reliability.png` and `.svg`.
- Layout: 4 x 3 small multiples: outcome classes in rows and subreddits in columns.
- Series: one saved calibrated one-vs-rest class probability per panel, directly labelled; no legend; a consistent blue treatment across panels.
- Intended reading: compare each class-specific empirical path with the dashed equality line, using the common axes.

Both figures use common 0-1 axes, larger print typography, no gridlines, and no figure-level title. The thread-initiation panels retain compact legends and panel subtitles; the four-class figure uses direct class labels and subreddit column headings with chronological-holdout denominators, without legends. Requested binning is ten probability quantiles per series; tied probabilities reduce each r/CryptoCurrency four-class series to seven realised non-empty bins. Exact counts and ranges are retained in the CSV.

These are audit reconstructions from chronological held-out rows. Existing Stage 1 PNGs are not reused because they contain the OOF curve that overwrote the test curve.
