# MIT License
# Copyright (c) 2025 Cara Lynch
# See the LICENSE file for details.
"""
Stage 2.1 – baseline feature subsets for thread-size classification.

This script computes baseline performance curves for Stage 2
(thread-size / bin prediction) by:

  * Discretising the continuous log-thread-size target into
    a fixed number of ordinal bins (2, 3, or 4 classes).
  * Ranking features using LightGBM feature importance on
    each outer cross-validation fold.
  * Training a sequence of models with 1..N top-ranked features
    and evaluating classification metrics with bootstrap CIs.

High-level behaviour
--------------------
Inputs (CLI arguments)
~~~~~~~~~~~~~~~~~~~~~~
--subreddit
    Short subreddit label (e.g. "crypto", "politics", "conspiracy").
    Used only for labelling outputs.
--outdir
    Output directory for metrics, plots, and metadata.
--train_X
    Parquet file containing the Stage 2 training features.
--train_y
    Parquet file containing the target column (e.g. log_thread_size).
--y-col
    Name of the target column in train_y. Defaults to "log_thread_size".
--classes
    Number of classes for discretisation. Supported values: 2 (binary),
    3 or 4 (multiclass). Defaults to 4.
--debug
    If set, enables a lightweight configuration (fewer CV splits and
    bootstrap iterations).
--no-cal / -nc
    If set, disables probability calibration with isotonic regression.
--rs
    Random seed (int). Used for CV splitting and bootstrapping.
--splits
    Number of outer StratifiedKFold splits. Defaults to 5 (or 2 in debug).
--feats
    Maximum number of top-ranked features to explore. Defaults to 30
    (or 10 in debug).
--n-bs
    Number of bootstrap resamples per (fold, n_feats) configuration.
    Defaults to 1000 (or 20 in debug).
--cal
    Calibration method: "sigmoid" or "isotonic" (default "sigmoid").

Discretisation
~~~~~~~~~~~~~~
The script supports:
    * Binary mode (classes == 2): y_bins = 1{log_thread_size > log(1)}.
    * Multiclass mode (classes in {3, 4}):
        - Lower/upper cut points are constructed from:
            log(1), log(2) and quantiles of y > log(1) as defined in
            CLASS_BIN_EDGES.
        - A final upper bound of max(y) + 1e-3 is added.
        - pd.cut is used with these bin edges to obtain integer labels.

Modelling and evaluation
~~~~~~~~~~~~~~~~~~~~~~~~
For each outer CV fold:
    * Fit a LightGBM classifier to obtain feature importances (split + gain).
    * Rank features by a MinMax-scaled combination of split and gain.
    * For n_feats = 1..--feats:
        - Train a model on the top-n_feats subset.
        - Optionally calibrate probabilities on a held-out subset
          of the validation fold.
        - Evaluate macro F1, macro precision/recall, MCC, and
          balanced accuracy.
        - Bootstrap the metrics to obtain 95% percentile confidence intervals.

Outputs
~~~~~~~
Written to --outdir:

    summary_scores.csv
        Per-fold, per-n_feats metrics with bootstrap CIs.

    importance_dfs.jl
        Joblib list of per-fold feature-importance DataFrames.

    agg_dir.jl
        Joblib DataFrame of aggregated (across folds) metrics and CIs.

    baseline_scores.xlsx
        Excel workbook with:
            - model_info           (arguments, versions, binning info)
            - summary_scores       (per-fold, per-n_feats metrics)
            - aggregated_scores    (mean metrics vs n_feats)
            - feat_importance      (mean/std feature importances)

    baseline_info.jl
        Joblib dict of model_info (metadata and binning).

    plots/*.png
        Errorbar plots of each metric against the number of features.

Reproducibility
~~~~~~~~~~~~~~~
- All randomised components (CV splits and bootstrap sampling) are
  controlled via a top-level random seed (--rs).
- Command-line arguments and library versions (Python, pandas, numpy,
  LightGBM, scikit-learn) are recorded in model_info.
- Debug mode provides a reduced configuration for fast iteration
  without modifying the main experimental design.
"""

import sys
import argparse
import os

import datetime as dt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib

from functools import partial
import pandas as pd
import numpy as np
import lightgbm as lgb

import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL_LOOKUP = {
    "crypto": "r/CryptoCurrency",
    "politics": "r/politics",
    "conspiracy": "r/Conspiracy",
}

CLASS_BIN_EDGES = {
    3: [0.5],
    4: [1 / 3, 2 / 3],
}

SCORERS = {
    "MCC": matthews_corrcoef,
    "Report": partial(classification_report, output_dict=True, zero_division=0),
    "CM": confusion_matrix,
    "Balanced accuracy": balanced_accuracy_score,
    "Precision": partial(precision_score, zero_division=0, average="macro"),
    "Recall": partial(recall_score, zero_division=0, average="macro"),
}

EXCLUDE_SCORES = ["Report", "CM"]

CAL_METHODS = ["sigmoid", "isotonic"]


def ci(arr):
    """
    Compute a 95% bootstrap confidence interval for a 1D array-like.

    Parameters
    ----------
    arr : array-like
        Input sample of metric values (e.g. bootstrap scores).

    Returns
    -------
    np.ndarray, shape (2,)
        Lower and upper bounds of the 95%% percentile interval
        (2.5th and 97.5th percentiles).
    """
    return np.percentile(arr, [2.5, 97.5])


def main():
    """
    Main entry point for Stage 2 baseline feature evaluation.
    
    Evaluates multiclass thread-size classification performance across
    increasing feature set sizes with bootstrap confidence intervals.
    """
    print(f"{sys.argv[0]}")
    start = dt.datetime.now()
    print("[INFO] Feature baselines.")
    print(f"[INFO] STARTED AT {start}")

    # CL arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--subreddit", help="Subreddit")
    ap.add_argument(
        "--outdir", help="Output directory.",
    )
    ap.add_argument(
        "--train_X", help="Training X data filepath (parquet).",
    )

    ap.add_argument(
        "--train_y", help="Training y data filepath (parquet).",
    )

    ap.add_argument(
        "--y-col",
        default="log_thread_size",
        help="Target y column. Defaults to log_thread_size.",
    )

    ap.add_argument("--classes", default="4", help="Number of classes.")

    ap.add_argument(
        "--debug", action="store_true", help="Run the script in debug mode."
    )
    ap.add_argument(
        "-nc", "--no-cal", action="store_true", help="Deactivate model calibration."
    )

    ap.add_argument(
        "--cal",
        default="sigmoid",
        help="Calibration method (sigmoid or isotonic). Defaults to sigmoid.",
    )

    ap.add_argument("--rs", default="42", help="Random state, defaults to 42.")

    ap.add_argument(
        "--splits",
        default=None,
        help="Number of CV splits. Defaults to 5, or 2 in debug mode.",
    )
    ap.add_argument(
        "--feats",
        default=None,
        help="Max model features. Defaults to 30, or 10 in debug mode.",
    )
    ap.add_argument(
        "--n-bs",
        default=None,
        help="Number of bootstrap trials. Defaults to 1000, or 20 in debug mode.",
    )

    args = ap.parse_args()
    args.rs = int(args.rs)
    args.classes = int(args.classes)

    if args.classes == 2:
        mode = "binary"
    elif args.classes > 2:
        mode = "multiclass"
    else:
        raise ValueError(f"[ERROR] Invalid number of classes {args.classes}")
    if mode == "multiclass" and args.classes not in CLASS_BIN_EDGES:
        CLASS_BIN_EDGES[args.classes] = [
            i / (args.classes - 1) for i in range(1, args.classes - 1)
        ]

    if mode == "multiclass":
        SCORERS["F1"] = partial(f1_score, average="macro")
    else:
        SCORERS["F1"] = f1_score

    if str(args.subreddit).lower() not in LABEL_LOOKUP:
        print(
            f"[ERROR] Subreddit entered {args.subreddit} not in list: {LABEL_LOOKUP.keys()}. Exiting."
        )
        raise FileNotFoundError

    debug = False
    if args.debug:
        debug = True
        print("[INFO] DEBUG MODE ENGAGED")

    calibrate = True
    if args.no_cal:
        calibrate = False
        print("[INFO] Model calibration disengaged.")

    args.cal = str(args.cal).lower()
    if args.cal not in CAL_METHODS:
        raise ValueError(
            f"[ERROR] {args.cal} not a valid calibration method. Choose from {CAL_METHODS}"
        )

    if args.splits is None:
        args.splits = 5 if not debug else 2
    else:
        args.splits = int(args.splits)

    if args.n_bs is None:
        args.n_bs = 1000 if not debug else 20
    else:
        args.n_bs = int(args.n_bs)

    if args.feats is None:
        args.feats = 30 if not debug else 10
    else:
        args.feats = int(args.feats)

    print(f"[INFO] Args: {args}")

    os.makedirs(args.outdir, exist_ok=True)

    plot_outdir = f"{args.outdir}/plots"
    os.makedirs(plot_outdir, exist_ok=True)

    print(f"[INFO] Loading training data.")
    X = pd.read_parquet(args.train_X)
    y = pd.read_parquet(args.train_y)[args.y_col]

    # run data for outfile
    model_info = {
        "script": str(sys.argv[0]),
        "run_start": start,
    }
    model_info.update(vars(args))

    # Record library versions for reproducibility
    model_info["python_version"] = sys.version
    model_info["pandas_version"] = pd.__version__
    model_info["numpy_version"] = np.__version__
    model_info["lightgbm_version"] = lgb.__version__
    model_info["sklearn_version"] = sklearn.__version__

    if mode == "multiclass":
        bin_edges = [np.log(1) - 1e-3]
        bin_edges.append(np.log(2) - 1e-3)
        bin_edges.extend(
            [y[y > np.log(1)].quantile(x) for x in CLASS_BIN_EDGES[args.classes]]
        )
        bin_edges.append(y.max() + 1e-3)

        assert (
            len(bin_edges) == args.classes + 1
        ), f"bin_edges must have length args.classes+1 ({args.classes+1}), got {len(bin_edges)}."

        y_bins = pd.cut(y, bins=bin_edges, labels=False, include_lowest=True)

        unique_bins = np.unique(y_bins.dropna())
        assert len(unique_bins) == args.classes, (
            f"pd.cut produced {len(unique_bins)} classes, expected {args.classes}. "
            f"Unique bins found: {unique_bins}"
        )
        model_info["bins"] = bin_edges
        model_info["thread_size_bins"] = [round(np.exp(x)) for x in bin_edges]
    else:
        y_bins = (y > np.log(1)).astype(int)
    class_counts = y_bins.value_counts().sort_index()
    for class_num in class_counts.index:
        model_info[f"class_{class_num}_count"] = class_counts[class_num]

    importance_dfs = []
    all_fold_dfs = {}
    bootstrap_vals = {}

    print("[INFO] Discretizing target for cross-validation folds...")
    # Outer CV
    outer_cv = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.rs)

    print("[INFO] Training data on folds")
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y_bins)):
        print(f"[INFO] Fold {fold + 1}/{args.splits}")
        all_fold_dfs[fold + 1] = {}
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_bins.iloc[train_idx], y_bins.iloc[val_idx]
        if calibrate:
            X_calib, X_val, y_calib, y_val = train_test_split(
                X_val, y_val, test_size=0.5, stratify=y_val, random_state=args.rs,
            )

        print(f"[INFO] [Fold {fold+1}] Precomputing ranked features for candidate bins")
        # Get feature importances for ranking
        selector_params = {
            "objective": mode,
            "class_weight": "balanced",
            "random_state": args.rs,
            "verbose": -1,
        }
        if mode == "multiclass":
            selector_params["num_class"] = args.classes
        selector = lgb.LGBMClassifier(**selector_params)

        selector.fit(X_train, y_train)

        # Save feature importance from this fold
        # After fitting
        booster = selector.booster_
        split_importance = booster.feature_importance(importance_type="split")
        gain_importance = booster.feature_importance(importance_type="gain")
        feature_names = booster.feature_name()

        # scale the feature importances
        scaler = MinMaxScaler()
        split_scaled = scaler.fit_transform(split_importance.reshape(-1, 1)).flatten()
        gain_scaled = scaler.fit_transform(gain_importance.reshape(-1, 1)).flatten()
        combined_importance = (split_scaled + gain_scaled) / 2

        # Save feature importance from this fold
        fold_importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                f"split_fold_{fold+1}": split_importance,
                f"split_scaled_fold_{fold+1}": split_scaled,
                f"gain_fold_{fold+1}": gain_importance,
                f"gain_scaled_fold_{fold+1}": gain_scaled,
                f"importance_fold_{fold+1}": combined_importance,
            }
        )
        importance_dfs.append(fold_importance_df)
        ranked_features = pd.Series(combined_importance, index=feature_names)
        ranked_features = ranked_features.sort_values(ascending=False).index.tolist()

        for n_feats in range(1, args.feats + 1):
            top_feats = ranked_features[:n_feats]

            print(f"[INFO] [Fold {fold+1}] [{n_feats} feats] Training model")

            clf = lgb.LGBMClassifier(**selector_params)
            clf.fit(X_train[top_feats], y_train)
            if calibrate:
                calibrated_clf = CalibratedClassifierCV(
                    clf, method=args.cal, cv="prefit"
                )
                calibrated_clf.fit(X_calib[top_feats], y_calib)
                y_proba = calibrated_clf.predict_proba(X_val[top_feats])
                y_pred = calibrated_clf.predict(X_val[top_feats])
            else:
                y_proba = clf.predict_proba(X_val[top_feats])
                y_pred = clf.predict(X_val[top_feats])

            if mode == "binary":
                y_proba = y_proba[:, 1]

            classes_predicted = len(np.unique(y_pred))
            if not debug:
                if classes_predicted < args.classes:
                    print(
                        f"[INFO] [Fold {fold+1}] [{n_feats} feats] Only {classes_predicted} classes predicted."
                    )
                    continue  # filter out models that don't predict all three classes

            print(f"[INFO] [Fold {fold+1}] [{n_feats} feats] Getting report metrics")
            # Report metrics
            excluded_metrics = {}
            for k in EXCLUDE_SCORES:
                excluded_metrics[k] = SCORERS[k](y_val, y_pred)
            performance_metrics = {}
            for k, m_func in [
                (k, v) for (k, v) in SCORERS.items() if k not in EXCLUDE_SCORES
            ]:
                performance_metrics[k] = m_func(y_val, y_pred)

            print(f"[INFO] [Fold {fold+1}] [{n_feats} feats] Bootstrapping metrics")
            # Bootstrapping main metrics
            rng = np.random.RandomState(
                args.rs + fold * 1000 + n_feats
            )  # for reproducibility

            bootstrap_metrics = {}
            for k in performance_metrics:
                bootstrap_metrics[k] = []

            for i in range(args.n_bs):
                indices = rng.choice(len(y_val), size=len(y_val), replace=True)
                y_true_bs = np.array(y_val)[indices]
                y_pred_bs = np.array(y_pred)[indices]
                for k in bootstrap_metrics:
                    bootstrap_metrics[k].append(SCORERS[k](y_true_bs, y_pred_bs))

            metric_cis = {}
            for k, vals in bootstrap_metrics.items():
                metric_cis[k] = ci(vals).tolist()

            results_df = pd.DataFrame(
                {
                    "Metric": list(performance_metrics.keys()),
                    "Score": list(performance_metrics.values()),
                    "95%_CI": list(metric_cis.values()),
                }
            )
            if n_feats not in bootstrap_vals:
                bootstrap_vals[n_feats] = {}
                for k, vals in bootstrap_metrics.items():
                    bootstrap_vals[n_feats][k] = vals
            else:
                for k, vals in bootstrap_metrics.items():
                    bootstrap_vals[n_feats][k].extend(vals)
            top_feats = ranked_features[:n_feats]

            report_df = pd.DataFrame(excluded_metrics["Report"]).transpose()

            all_fold_dfs[fold + 1][n_feats] = {
                "n_feats": n_feats,
                "classes_predicted": classes_predicted,
                "features": top_feats,
                "results": results_df,
                "report": report_df,
            }

    ci_vals = {}
    for n_feats in bootstrap_vals:
        ci_vals[n_feats] = {}
        for key, values in bootstrap_vals[n_feats].items():
            if len(values) == 0:
                print(
                    f"Warning: No bootstrap samples for n_feats={n_feats}, metric={key}. Skipping CI computation."
                )
                continue
            conf_int = ci(values)
            ci_vals[n_feats][f"{key}_lower"] = conf_int[0]
            ci_vals[n_feats][f"{key}_upper"] = conf_int[1]

    ci_vals_df = (pd.DataFrame.from_dict(ci_vals, orient="index")).reset_index(
        names="n_feats"
    )
    summary_rows = []
    for fold, feats_dict in all_fold_dfs.items():
        for n_feats, data in feats_dict.items():
            metrics = [m for m in SCORERS if m not in EXCLUDE_SCORES]
            row = data["results"].set_index("Metric").loc[metrics]["Score"]
            summary_dict = {
                "fold": fold,
                "n_feats": n_feats,
                "classes_predicted": data["classes_predicted"],
            }
            for metric in metrics:
                summary_dict[metric] = row[metric]
                for i, ci_bound in enumerate(["lower", "upper"]):
                    summary_dict[f"{metric}_{ci_bound}"] = (
                        data["results"].set_index("Metric").loc[metric]["95%_CI"][i]
                    )
            summary_rows.append(summary_dict)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{args.outdir}/summary_scores.csv", index=False)
    agg_df = (
        summary_df.groupby("n_feats")
        .agg(dict(zip(metrics, ["mean"] * len(metrics))))
        .reset_index()
    )

    agg_df = agg_df.merge(ci_vals_df, on="n_feats", how="left")

    joblib.dump(importance_dfs, f"{args.outdir}/importance_dfs.jl")
    importance_df = importance_dfs[0]
    for df in importance_dfs[1:]:
        importance_df = importance_df.merge(df, on="feature", how="outer")
    importance_df.fillna(0, inplace=True)
    for imp_colname in ["importance", "split", "gain"]:
        imp_cols = [
            col
            for col in importance_df.columns
            if col.startswith(f"{imp_colname}_fold_")
        ]
        importance_df[f"mean_{imp_colname}"] = importance_df[imp_cols].mean(axis=1)
        importance_df[f"std_{imp_colname}"] = importance_df[imp_cols].std(axis=1)

    joblib.dump(agg_df, f"{args.outdir}/agg_dir.jl")
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        # Select rows where both CI bounds are present
        has_ci = agg_df[
            agg_df[f"{metric}_lower"].notna() & agg_df[f"{metric}_upper"].notna()
        ]
        if not has_ci.empty:
            lower = has_ci[f"{metric}_lower"]
            upper = has_ci[f"{metric}_upper"]
            err = [has_ci[metric] - lower, upper - has_ci[metric]]

            plt.errorbar(
                has_ci["n_feats"],
                has_ci[metric],
                yerr=err,
                fmt="-o",
                capsize=3,
                label=metric,
            )
        else:
            print(f"Warning: No CI data for {metric}. Plotting point means only.")
            plt.plot(
                agg_df["n_feats"], agg_df[metric], "-o", label=metric,
            )
        plt.title(f"{LABEL_LOOKUP[args.subreddit]}", loc="left", fontsize=12)
        plt.xlabel("Number of features", fontsize=11)
        plt.ylabel(metric, fontsize=11)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f"{plot_outdir}/{metric.lower()}_vs_n_feats_bootstrap.png", dpi=300)
        plt.close()

    with pd.ExcelWriter(f"{args.outdir}/baseline_scores.xlsx") as writer:
        pd.DataFrame.from_dict(model_info, orient="index").to_excel(
            writer, sheet_name="model_info", index=True
        )
        summary_df.to_excel(writer, sheet_name="summary_scores", index=False)
        agg_df.to_excel(writer, sheet_name="aggregated_scores", index=False)
        importance_df.to_excel(writer, sheet_name="feat_importance", index=False)

    joblib.dump(model_info, f"{args.outdir}/baseline_info.jl")
    end = dt.datetime.now()
    total_runtime = end - start
    print(f"[INFO] Finished at {end}. Runtime: {total_runtime}.")


if __name__ == "__main__":
    main()
