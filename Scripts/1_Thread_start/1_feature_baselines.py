"""
Stage 1 baseline feature scores for thread-start classification.

This script evaluates how predictive different-sized feature subsets are for
Stage 1 (thread-start vs stalled) using LightGBM classifiers and stratified
K-fold cross-validation.

High-level behaviour
--------------------
- Input:
    - `--train_X`: training feature matrix (parquet).
    - `--train_y`: training target data (parquet) containing `--y-col`.
    - `--subreddit`: key in LABEL_LOOKUP (e.g. 'conspiracy', 'crypto', 'politics').
- Target:
    - `--y-col` (default: 'log_thread_size') is thresholded at `--y_thresh`
      (default: log(1) == 0) to define a binary label:
          y = 1  ⇔  log_thread_size > y_thresh  (thread started)
          y = 0  ⇔  log_thread_size ≤ y_thresh  (thread stalled)
- Cross-validation:
    - StratifiedKFold with `--splits` folds (default 5, or 2 in debug mode),
      shuffling and fixed random seed.
- Feature ranking:
    - Fit a LightGBM classifier on each training fold.
    - Compute split- and gain-based feature importances from the booster.
    - Min–max scale both, then average them to obtain a combined importance
      score per feature.
- Baseline grid:
    - For each fold, sort features by combined importance.
    - For n_feats in {1, 2, ..., `--feats`} (default 30, or 10 in debug mode):
        - train a LightGBM model on the top-n features,
        - optionally calibrate probabilities with isotonic regression on a
          held-out calibration set (`--no-cal` disables calibration),
        - evaluate metrics on the validation set.
- Metrics and uncertainty:
    - Metrics include:
        - AUC (probability-based)
        - MCC, F1, F-beta (default beta=2)
        - Balanced accuracy
        - Class-wise precision/recall for Stalled/Started.
    - For each (fold, n_feats), metrics are bootstrapped `--n-bs` times
      (default 1000, or 20 in debug mode) with replacement to obtain 95%
      percentile confidence intervals.
- Outputs (in `--outdir`):
    - `summary_scores.csv`:
        per-fold, per-n_feats metrics and CIs.
    - `aggregated_scores.jl`:
        joblib of aggregated scores by n_feats.
    - `importance_dfs.jl`:
        joblib list of per-fold importance DataFrames.
    - `{subreddit}_1_feature_baselines.xlsx`:
        model_info, summary_scores, aggregated_scores, and feature importances.
    - `{subreddit}_baseline_info.jl`:
        joblib of run configuration and metadata (arguments, runtime, etc.).
    - `{metric}_vs_n_feats_bootstrap.png`:
        performance vs number of features with bootstrap CIs.

Reproducibility notes
---------------------
- All randomisation (CV splits and bootstrapping) is seeded by `--rs`.
- All command-line arguments are saved into `model_info`.
- Default behaviour is conservative; debug mode (`--debug`) reduces splits,
  features, and bootstrap trials for rapid iteration.
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef,
    f1_score,
    brier_score_loss,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    fbeta_score,
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

CLASS_NAMES = {
    0: "Stalled",
    1: "Started",
}

SCORERS = {
    "MCC": matthews_corrcoef,
    "F1": f1_score,
    "Report": partial(classification_report, output_dict=True, zero_division=0),
    "CM": confusion_matrix,
    "Balanced accuracy": balanced_accuracy_score,
}

EXCLUDE_SCORES = ["Report", "CM"]

for i, class_name in CLASS_NAMES.items():
    SCORERS.update(
        {
            f"Precision {class_name}": partial(
                precision_score, pos_label=i, zero_division=0
            ),
            f"Recall {class_name}": partial(recall_score, pos_label=i, zero_division=0),
        }
    )


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
    print(f"{sys.argv[0]}")
    start = dt.datetime.now()
    print("[INFO] Feature baselines.")
    print(f"[INFO] STARTED AT {start}")

    # CL arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--subreddit", help="Subreddit")
    ap.add_argument(
        "--outdir",
        help="Output directory.",
    )
    ap.add_argument(
        "--train_X",
        help="Training X data filepath (parquet).",
    )

    ap.add_argument(
        "--train_y",
        help="Training y data filepath (parquet).",
    )

    ap.add_argument(
        "--y-col",
        default="log_thread_size",
        help="Target y column. Defaults to log_thread_size.",
    )

    ap.add_argument(
        "--y_thresh", 
        default=None,
        help="Target y threshold to identify started threads. Defaults to log(1)."
    )

    ap.add_argument("--debug", action="store_true", help="Run the script in debug mode.")
    ap.add_argument("-nc", "--no-cal", action="store_true", help="Deactivate model calibration.")

    ap.add_argument("--rs", default="42", help="Random state, defaults to 42.")

    ap.add_argument(
        "--beta",
        default="2",
        help="Beta for f-beta score. Default 2.",
    )

    ap.add_argument("--splits", default=None, help="Number of CV splits. Defaults to 5, or 2 in debug mode.")
    ap.add_argument("--feats", default=None, help="Max model features. Defaults to 30, or 10 in debug mode.")
    ap.add_argument("--n-bs", default=None, help="Number of bootstrap trials. Defaults to 1000, or 20 in debug mode.")

    args = ap.parse_args()
    args.rs = int(args.rs)
    args.beta = float(args.beta)

    SCORERS["F-beta"] = partial(fbeta_score, beta=args.beta)

    if str(args.subreddit).lower() not in LABEL_LOOKUP:
        print(f"[ERROR] Subreddit entered {args.subreddit} not in list: {LABEL_LOOKUP.keys()}. Exiting.")
        raise FileNotFoundError
    
    debug = False
    if args.debug:
        debug = True
        print("[INFO] DEBUG MODE ENGAGED")

    calibrate = True
    if args.no_cal:
        calibrate = False
        print("[INFO] Model calibration disengaged.")

    if args.splits is None:
        args.splits = 5 if not debug else 2
    else:
        args.splits = int(args.splits)

    if args.n_bs is None:
        args.n_bs = 1000 if not debug else 20
    else:
        args.n_bs = int(args.n_bs)

    if args.y_thresh is None:
        args.y_thresh = np.log(1)
    else:
        args.y_thresh = float(args.y_thresh)
    
    print(f"[INFO] Args: {args}")
    
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[INFO] Loading training data.")
    X = pd.read_parquet(args.train_X)

    if args.feats is None: 
        args.feats = 30 if not debug else 10
    else:
        if args.feats == "max":
            args.feats = len(X.columns)
        else:
            args.feats = int(args.feats)

    y = pd.read_parquet(args.train_y)[args.y_col]
    # threshold to identify started threads
    y = (y > args.y_thresh).astype(int)

    print(f"[INFO] Setting up outer cross-validation.")
    outer_cv = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.rs)

    # run data for outfile
    model_info = {
        "script": str(sys.argv[0]),
        "run_start": start,
    }
    model_info.update(vars(args))

    importance_dfs = []
    all_fold_dfs = {}
    bootstrap_vals = {}
    print("[INFO] Training data on folds")
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
        print(f"[INFO] Fold {fold + 1}/{args.splits}")
        all_fold_dfs[fold + 1] = {}
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if calibrate:
            print(f"[INFO] [Fold {fold + 1}] Splitting x_val and y_val into calibration and eval sets")
            # Split X_val and y_val into calibration and evaluation sets
            X_calib, X_val, y_calib, y_val = train_test_split(
                X_val,
                y_val,
                test_size=0.5,  # 50% for calibration, 50% for evaluation
                stratify=y_val,  # preserve class balance
                random_state=args.rs,  # for reproducibility
            )
        print(f"[INFO] [Fold {fold + 1}] Getting feature importances for feature ranking")
        # Get feature importances for ranking
        selector = lgb.LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            random_state=args.rs,
            verbose=-1,
        )
        selector.fit(X_train, y_train)
        # Save feature importance from this fold
        # After fitting
        booster = selector.booster_
        split_importance = booster.feature_importance(importance_type="split")
        gain_importance = booster.feature_importance(importance_type="gain")
        feature_names = booster.feature_name()
        print(f"[INFO] [Fold {fold + 1}] Scaling feature importances")
        # scale the feature importances
        scaler = MinMaxScaler()
        split_scaled = scaler.fit_transform(split_importance.reshape(-1, 1)).flatten()
        gain_scaled = scaler.fit_transform(gain_importance.reshape(-1, 1)).flatten()
        combined_importance = (split_scaled + gain_scaled) / 2

        # Construct DataFrame
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
        print(f"[INFO] [Fold {fold + 1}] Getting ranked features")
        ranked_features = pd.Series(combined_importance, index=feature_names)
        ranked_features = ranked_features.sort_values(ascending=False).index.tolist()

        for n_feats in range(1, args.feats+1):
            top_feats = ranked_features[:n_feats]

            print(f"[INFO] [Fold {fold + 1}] [{n_feats} features] Training model")
            clf = lgb.LGBMClassifier(
                objective="binary",
                class_weight="balanced",
                random_state=args.rs,
                verbose=-1,
            )
            clf.fit(X_train[top_feats], y_train)
            if calibrate:
                print(f"[INFO] [Fold {fold + 1}] [{n_feats} features] Calibrating model")
                calibrated_clf = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
                calibrated_clf.fit(X_calib[top_feats], y_calib)
                y_proba = calibrated_clf.predict_proba(X_val[top_feats])[:, 1]
                y_pred = calibrated_clf.predict(X_val[top_feats])
            else:
                y_proba = clf.predict_proba(X_val[top_feats])[:, 1]
                y_pred = clf.predict(X_val[top_feats])
            if len(np.unique(y_pred)) < 2:
                print(
                    f"[WARNING] [Fold {fold + 1}] [{n_feats} features] Not all classes predicted (only {len(np.unique(y_pred))}) - excluding this {n_feats} feature model."
                )
                continue

            print(f"[INFO] [Fold {fold + 1}] [{n_feats} features] Getting report metrics")
            # Report metrics
            performance_metrics = {"AUC": roc_auc_score(y_val, y_proba)}
            report = SCORERS["Report"](y_val, y_pred)
            for key, scorer_func in [(k, v) for k, v in SCORERS.items() if k not in EXCLUDE_SCORES]:
                performance_metrics[key] = scorer_func(y_val, y_pred)

            # Store metric names once
            if "metric_names" not in locals():
                metric_names = list(performance_metrics.keys())
            
            print(f"[INFO] [Fold {fold + 1}] [{n_feats} features] Bootstrapping metrics")
            # Bootstrapping main metrics
            rng = np.random.RandomState(args.rs + fold * 1000 + n_feats) # for reproducibility

            bootstrap_metrics = {}
            for key in performance_metrics:
                bootstrap_metrics[key] = []

            for i in range(args.n_bs):
                indices = rng.choice(len(y_val), size=len(y_val), replace=True)
                y_true_bs = np.array(y_val)[indices]
                y_pred_bs = np.array(y_pred)[indices]
                y_proba_bs = np.array(y_proba)[indices]

                bootstrap_metrics["AUC"].append(roc_auc_score(y_true_bs, y_proba_bs))

                for key in performance_metrics:
                    if key != "AUC":
                        bootstrap_metrics[key].append(SCORERS[key](y_true_bs, y_pred_bs))
                    

            conf_intervals = {}
            for key, values in bootstrap_metrics.items():
                conf_intervals[key] = ci(values).tolist()

            results_df = pd.DataFrame(
                {
                    "Metric": list(performance_metrics.keys()),
                    "Score": list(performance_metrics.values()),
                    "95%_CI": list(conf_intervals.values()),
                }
            )

            if n_feats not in bootstrap_vals:
                bootstrap_vals[n_feats] = bootstrap_metrics
            else:
                for key, value in bootstrap_metrics.items():
                    bootstrap_vals[n_feats][key].extend(value)

            report_df = pd.DataFrame(report).transpose()

            all_fold_dfs[fold + 1][n_feats] = {
                "n_feats": n_feats,
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
                    f"[WARNING] No bootstrap samples for n_feats={n_feats}, metric={key}. Skipping CI computation."
                )
                continue
            conf_int = ci(values)
            ci_vals[n_feats][f"{key}_lower"] = conf_int[0]
            ci_vals[n_feats][f"{key}_upper"] = conf_int[1]

    ci_vals_df = (pd.DataFrame.from_dict(ci_vals, orient="index")).reset_index(
        names="n_feats"
    )
    joblib.dump(all_fold_dfs, f"{args.outdir}/all_fold_dfs.jl")
    summary_rows = []
    for fold, feats_dict in all_fold_dfs.items():
        for n_feats, data in feats_dict.items():
            metrics = metric_names
            row = data["results"].set_index("Metric").loc[metrics]["Score"]
            summary_dict = {
                "fold": fold,
                "n_feats": n_feats,
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
            col for col in importance_df.columns if col.startswith(f"{imp_colname}_fold_")
        ]
        importance_df[f"mean_{imp_colname}"] = importance_df[imp_cols].mean(axis=1)
        importance_df[f"std_{imp_colname}"] = importance_df[imp_cols].std(axis=1)

    importance_df.sort_values("mean_importance", ascending=False, inplace=True)

    joblib.dump(agg_df, f"{args.outdir}/aggregated_scores.jl")

    for metric in metrics:
        print(f"[INFO] Plotting {metric}")
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
            print(f"[Warning] No CI data for {metric}. Plotting point means only.")
            plt.plot(
                agg_df["n_feats"], agg_df[metric], "-o", label=metric,
            )
        plt.title(f"{LABEL_LOOKUP[args.subreddit]}", loc="left", fontsize=12)
        plt.xlabel("Number of features", fontsize=10)
        plt.ylabel(metric, fontsize=10)
        plt.xticks(fontsize=9,)
        plt.yticks(fontsize=9)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f"{args.outdir}/{metric.lower()}_vs_n_feats_bootstrap.png", dpi=300)
        plt.close()

    end = dt.datetime.now()
    total_runtime = end - start
    model_info["total_runtime"] = str(total_runtime)

    # Record library versions for reproducibility
    model_info["python_version"] = sys.version
    model_info["pandas_version"] = pd.__version__
    model_info["numpy_version"] = np.__version__
    model_info["lightgbm_version"] = lgb.__version__
    model_info["sklearn_version"] = sklearn.__version__

    print(f"[INFO] Finished at {end}. Runtime: {total_runtime}.")

    joblib.dump(model_info, f"{args.outdir}/{args.subreddit}_baseline_info.jl")

    with pd.ExcelWriter(f"{args.outdir}/{args.subreddit}_1_feature_baselines.xlsx") as writer:
        pd.DataFrame.from_dict(model_info, orient="index").to_excel(
            writer, sheet_name="model_info", index=True
        )
        summary_df.to_excel(writer, sheet_name="summary_scores", index=False)
        agg_df.to_excel(writer, sheet_name="aggregated_scores", index=False)
        importance_df.to_excel(writer, sheet_name="feat_importance", index=False)


if __name__ == "__main__":
    main()
