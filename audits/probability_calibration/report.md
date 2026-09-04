# Study 1 probability-calibration evidence audit

## Technical summary

The six compact final configurations reconcile to the published selections: four predictors for all three thread-initiation models, and three, two, and three predictors for the r/Conspiracy, r/CryptoCurrency, and r/politics four-class models. Their saved evaluation runs used the chronological final 20% partitions, with non-overlapping, ordered training and test timestamps.

Row-level calibrated probabilities and true labels survive for every final model and map from split-local ordinal indices to retained `thread_id` values. Matched uncalibrated probabilities do not survive for any model. The audit therefore describes the retained calibrated predictions but makes no before-versus-after claim about the effect of calibration.

## Key findings

1. **Thread initiation has both empirical and procedural evidence.** The executed code used isotonic `CalibratedClassifierCV`. Ten-bin quantile reliability coordinates survive for both chronological test and OOF predictions. Recalculation from the held-out row probabilities and labels reproduces all saved test coordinates to numerical tolerance.
2. **The surviving binary PNGs are not chronological-test plots.** The evaluation loop writes the test curve and then the OOF curve to the same `calibration_curve.png`. File modification times follow that order in all three selected directories, so the OOF curve overwrote the test curve. OOF observations were fold-held-out within the training population; they were not the final chronological test partition.
3. **Four-class rows survive, but no executed reliability curve does.** The code and run metadata show sigmoid calibration. `test_preds.csv`, `y_proba.parquet`, and `y_test.parquet` agree exactly, probabilities sum to one, encoded labels match the subreddit-specific class ranges, and predictions equal `argmax`. The pipeline did not call `calibration_curve` for Stage 2 and saved no curve coordinates or plot.
4. **The retained held-out probabilities permit a transparent retrospective diagnostic.** Binary Brier scores are 0.1045, 0.1627, and 0.1031; binary log losses are 0.3679, 0.5006, and 0.3503 for r/Conspiracy, r/CryptoCurrency, and r/politics. Ten-quantile-bin ECE values are 0.0266, 0.0250, and 0.0527. Four-class Brier scores (row-wise sum across classes) are 0.6922, 0.5067, and 0.6031, with log losses 1.2692, 0.9669, and 1.1304. These are probabilistic performance and bin-dependent descriptive diagnostics, not evidence that calibration improved the models.

The audit-reconstructed held-out reliability figures are in `figures/thread_initiation_reliability.png` and `figures/thread_size_reliability.png`; exact bin counts, denominators, ranges, means, observed fractions, and gaps are in `outputs/reliability_bins.csv`.

## Scope, populations, labels and identifiers

The final selected model directories are `model_4` for all binary tasks and `model_3`, `model_2`, and `model_3` for the three four-class tasks. The binary labels are Stalled (`thread_size=1`) and Started (`thread_size>1`). Four-class ranges are:

| Subreddit | Stalled | Small | Medium | Large |
|---|---:|---:|---:|---:|
| r/Conspiracy | 1 | 2–6 | 7–19 | ≥20 |
| r/CryptoCurrency | 1 | 2–10 | 11–32 | ≥33 |
| r/politics | 1 | 2–9 | 10–30 | ≥31 |

Saved prediction files use split-local indices beginning at zero, not `thread_id` directly. Exact equality of index, timestamp, and `thread_size` order links those rows to the enriched preprocessing files, where `thread_id` survives. The package deliberately does not copy row-level IDs or probabilities.

## Methods and validation

The audit parsed the published compact-selection files and the selected rows in each unbalanced `4_model/evaluation.xlsx`. It checked selected feature columns, labels, thresholds or class ranges, and directories. It then matched retained probability rows to preprocessing outcomes and identifiers, verified chronological separation, probability bounds/simplex constraints, threshold/argmax predictions, and duplicate CSV/Parquet representations.

For binary curves, the audit reproduced scikit-learn's `calibration_curve(..., n_bins=10, strategy="quantile")` allocation and compared coordinates with the saved joblib dictionaries. For four-class models it applied the same ten-quantile-bin rule separately to each one-vs-rest class. The original binary curves realise all ten requested bins. In the audit-reconstructed four-class curves, r/Conspiracy and r/politics realise ten bins per class, while tied probabilities collapse every r/CryptoCurrency class to seven non-empty bins. Denominators are 2,279/9,116, 2,964/11,854, and 13,069/52,274 for chronological-test/OOF binary curves.

All 45 automated checks pass. Source files are checksummed in `outputs/source_inventory.csv`; calculation versions and Git commit `1051e881a2d0fe4a9ae7886d2d3b2985d0da610a` are recorded in `outputs/run_record.json`.

## Limitations and robustness

- No matched uncalibrated probability matrix was found. Although raw estimator objects survive, scoring them would rerun inference and was outside scope. Calibration's effect is therefore not identifiable from the retained outputs.
- The reconstructed Stage 2 diagrams are new audit calculations from executed probability outputs, not plots executed by the original pipeline.
- ECE and MCE depend on the chosen binning rule. Brier score and log loss combine calibration with other aspects of probabilistic prediction.
- The audit verifies artefact consistency and provenance, not external generalisability or calibration on a new population.
- Final calibration was fitted within training data using five-fold stratified calibration in the executed code. OOF calibration used a held-out calibration subset inside each outer training fold. The final chronological test rows were not used to fit those calibrators.

## Chapter 4 recommendation

Add one compact appendix table and the two task-specific held-out reliability figures, with a short Chapter 4 cross-reference. The table should report model, calibration method, holdout N, Brier score, log loss, and the explicitly labelled ten-quantile-bin ECE; the four-class rows should state that ECE is reported one-vs-rest in the detailed appendix. This is useful because it makes the empirical evidence visible while avoiding an unsupported before/after claim. Do not reuse the existing binary PNGs as final-test evidence.

### Thesis-safe Methods wording

> For the final binary thread-initiation models, probabilities were calibrated with isotonic regression; for the four-class thread-size models, sigmoid calibration was applied in a one-vs-rest multiclass scheme. Calibration was fitted using training data only, while the final performance assessment used the later chronological holdout partition. Reliability was assessed descriptively on the retained held-out probabilities using ten quantile-based bins. Because matched uncalibrated held-out probabilities were not retained, these analyses do not estimate the change caused by calibration.

### Thesis-safe appendix wording

> Appendix Figure X shows reliability diagrams reconstructed from the saved calibrated probabilities and true labels for the chronological holdout sets. The binary pipeline also saved the corresponding curve coordinates, which were reproduced exactly. Its surviving `calibration_curve.png` files represent OOF training-population predictions because the OOF plot overwrote the test plot. No executed four-class reliability plot survived; those panels are audit reconstructions using one-vs-rest class probabilities. Brier score, log loss, ECE and MCE are reported as descriptive diagnostics under the stated definitions and binning rule.

### Thesis-safe viva wording

> I can show that calibration was part of the executed final pipeline and that calibrated probabilities and labels survive for every chronological holdout. I can also reproduce the binary held-out curve coordinates and calculate transparent held-out diagnostics for all six models. I cannot claim how much calibration improved performance, because matched uncalibrated probabilities were not retained and I did not regenerate them.

## Further questions

If a future reproducibility run is authorised, retain raw and calibrated probabilities from the same estimator and rows, save separate filenames for test and OOF plots, and pre-specify multiclass calibration summaries. That would permit a matched before/after analysis without ambiguity.

