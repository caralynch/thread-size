# Metric and curve definitions

- **Binary Brier score:** mean squared error between the probability of Started and the binary target, `mean((p-y)^2)`. Range 0–1; lower is better.
- **Multiclass Brier score:** mean, over rows, of the sum across four classes of `(p_k - I[y=k])^2`. Range 0–2; lower is better. The normalized value divides this result by four and is the mean per-class squared error.
- **Log loss:** mean negative natural log probability assigned to the true label/class. Lower is better.
- **AUC:** binary ROC area, included only to reconcile the saved final output. It measures ranking discrimination rather than calibration.
- **Reliability bin:** observations grouped by predicted-probability quantiles using the same allocation as scikit-learn `calibration_curve` with `n_bins=10` and `strategy="quantile"`. Duplicate quantile edges are collapsed and empty bins omitted: the binary series and the r/Conspiracy and r/politics four-class series realise ten bins, while each r/CryptoCurrency four-class series realises seven.
- **ECE:** count-weighted mean of the absolute difference between observed fraction and mean predicted probability across the ten quantile bins. This is a bin-dependent descriptive statistic.
- **MCE:** largest absolute observed-versus-predicted gap among the ten quantile bins. This is also bin-dependent.
- **Four-class reliability:** one-vs-rest for each class, with its own ten quantile bins.

Brier score and log loss evaluate probabilistic predictions but do not isolate calibration from discrimination or sharpness. None of these values constitutes a before/after estimate of calibration's effect.

