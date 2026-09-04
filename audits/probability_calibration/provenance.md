# Provenance record

## Selection chain

- `Publication_Outputs/1_Thread_Start/s1_mods.txt` selects four predictors for each binary model.
- `Publication_Outputs/2_Thread_Size/s2_mods.txt` selects three, two and three predictors for r/Conspiracy, r/CryptoCurrency and r/politics.
- The published selected performance values match the unbalanced `Outputs/*/<subreddit>/4_model/evaluation.xlsx` workbooks rather than the `4_model_balanced` alternatives.
- In the output layout, `model_<n>` is the model using `<n>` candidate predictors; consequently the selected folders are Stage 1 `model_4` for all subreddits and Stage 2 `model_3`, `model_2`, and `model_3`.

## Calibration chain

The Stage 1 workbooks record `no_cal=0`; the selected script uses isotonic calibration. Outer-fold validation probabilities are produced after a calibration subset is removed from that fold's training rows. The final calibrated estimator is fitted using the training partition with five-fold stratified calibration, then applied to the chronological test partition.

The Stage 2 workbooks record `no_cal=False` and `cal=sigmoid`. Outer-fold validation uses a held-out calibration subset within the fold training data. The final five-fold stratified sigmoid calibrator is fitted on the training partition before chronological-test probabilities are generated.

No file named or documented as a matched uncalibrated prediction/probability output was found in the selected run directories or their executed run metadata. Serialized base estimators are not evidence of matched uncalibrated predictions, and they were not loaded or scored.

## Reliability-curve chain

Stage 1 calls `calibration_curve(true_y, probas, n_bins=10, strategy="quantile")` for `test` and then `oof`. It saves distinct coordinate dictionaries, but both plot iterations target `plots/calibration_curve.png`. The test dictionary modification time precedes the OOF dictionary, which immediately precedes the PNG, for all three selected models. The surviving PNG is therefore attributed to OOF predictions. The OOF observations are fold-held-out from fitting in their outer folds, but belong to the development/training population.

Stage 2 contains no `calibration_curve` call and no selected-run calibration-curve input or plot. The audit's Stage 2 reliability rows and panels are explicitly retrospective reconstructions from saved `y_proba.parquet` and `y_test.parquet`.

## Row identity and chronology

The prediction outputs store a zero-based ordinal `index` local to each split. The preprocessing Y and enriched row files have the same reset indices. Within each split, index, timestamp and `thread_size` order agree exactly, providing a deterministic mapping to retained `thread_id`. All partitions are timestamp-ordered and each training maximum precedes its test minimum. Exact dates and counts are in `outputs/chronological_partitions.csv`.

## Audit boundary

Only existing artefacts were read. Joblib was used solely for the small saved coordinate dictionaries, never model objects. Row-level predictions and identifiers are not duplicated into the package; only aggregate bins, metrics, checks, paths, timestamps, sizes and SHA-256 hashes are emitted.

