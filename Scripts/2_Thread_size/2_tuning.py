"""
Stage 2 – class-weight and threshold tuning for thread-size classification.

This script implements the second tuning stage for the thread-size (Stage 2)
classifier. Given:

    * a fixed discretisation of log(thread_size) into K ordinal bins, and
    * a ranked list of feature subsets of size n_feats,

it searches over:

    (i)  class-weighting schemes (balanced vs. custom per-class weights), and
    (ii) per-class decision thresholds applied to predicted class probabilities,

to maximise the Matthews correlation coefficient (MCC) under stratified
cross-validation.

High-level procedure
--------------------
1. Load Stage 2 training features (X) and continuous target y (log_thread_size).
2. Discretise y into K = --classes bins using fixed cut points based on
   log(1), log(2) and quantiles of y > log(1) (see CLASS_BIN_EDGES).
3. Run an outer StratifiedKFold CV with --splits folds.
   For each fold:
     a. Fit a balanced multiclass LightGBM model to obtain feature importances
        (split and gain) and rank features by a combined MinMax-scaled score.
     b. For each feature count n_feats in model_feats:
          - Use Optuna to search over:
              * cw_type  ∈ {"balanced", "custom"},
              * per-class weights when cw_type == "custom",
            while keeping tree hyperparameters fixed.
          - Evaluate MCC (and macro F1) on the validation subset.
4. Aggregate best parameters across folds for each n_feats by taking the mode
   of categorical / integer parameters and the mean of numeric parameters.
5. With aggregated class weights fixed, run a second CV loop to:
     a. Refit models on a calibration/validation split.
     b. Optionally apply sigmoid calibration to predicted probabilities.
     c. Optimise a vector of per-class thresholds by maximising MCC via
        L-BFGS-B over [0, 1]^K using the calibration subset.
     d. Evaluate MCC on the validation subset using the tuned thresholds.

Outputs
-------
Written to --outdir:

  * tuning_params.jl
      Final configuration per n_feats, including:
          - "features"            : top-n_feats features,
          - "bins"                : list of bin edges used for discretisation,
          - "final_class_weights" : aggregated per-class weights,
          - "final_threshold"     : per-class decision thresholds,
          - "mcc_before_thresh"   : mean MCC before threshold tuning,
          - "mcc_after_thresh"    : mean MCC after threshold tuning.

  * optuna_params.jl
      Summary of aggregated Optuna hyperparameters per n_feats and
      their mean best MCC across folds.

  * all_configs_df.(csv|jl)
      Long-format table of all non-pruned Optuna trials, including
      per-trial MCC, F1, cw_type and class-weight vectors.

  * feature_importances (in tuning_outputs.xlsx)
      Fold-wise and aggregated feature importances used to rank
      candidate feature subsets.

  * tuning_outputs.xlsx
      Excel workbook containing:
          - "model_info"
          - "params"               (final per-n_feats configs)
          - "all_configs"          (all trials)
          - "feature_importances"
          - "flattened_features"   (feature ranks + final weights/thresholds).

Reproducibility
---------------
- All randomised operations (CV splits, calibration splits, Optuna sampler)
  are controlled by --rs.
- Command-line arguments and library versions (Python, pandas, numpy,
  LightGBM, scikit-learn, Optuna) are stored in model_info and written to
  tuning_outputs.xlsx.
- The full Optuna trial history is written out, allowing complete auditing
  of the tuning process.
"""

import sys
import argparse
import os
import gc

import datetime as dt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib

from functools import partial, reduce
import pandas as pd
import numpy as np
import lightgbm as lgb

import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import compute_class_weight
from scipy.optimize import minimize

import optuna
import optuna.visualization as vis


LABEL_LOOKUP = {
    "crypto": "r/CryptoCurrency",
    "politics": "r/politics",
    "conspiracy": "r/Conspiracy",
}

CLASS_BIN_EDGES = {
    3: [0.5],
    4: [1 / 3, 2 / 3],
}

CLASS_NAMES = {
    3: ["Stalled", "Small", "Large"],
    4: ["Stalled", "Small", "Medium", "Large"],
}

SCORERS = {
    "MCC": matthews_corrcoef,
    "F1": partial(f1_score, average="macro"),
}

SCORER_MAP = {
    "mcc": "MCC",
    "f1": "F1",
    "f1-score": "F1",
}


def check_bin_balance(y_bins):
    """
    Compute class proportions for a 1D array of integer bin labels.

    Parameters
    ----------
    y_bins : array-like of shape (n_samples,)
        Integer class labels (e.g. output of pd.cut(..., labels=False)).

    Returns
    -------
    np.ndarray of shape (n_classes,)
        Proportion of samples in each class.
    """
    counts = np.bincount(y_bins)
    proportions = counts / counts.sum()
    return proportions


def get_preds(thresholds, y_probas):
    """
    Convert multiclass probabilities and per-class thresholds into hard labels.

    For each sample, probabilities below their class-specific thresholds are
    masked; the predicted class is then:
        - the argmax among classes that pass their threshold, or
        - the argmax over all classes if none pass.

    Parameters
    ----------
    thresholds : array-like of shape (n_classes,)
        Per-class probability thresholds in [0, 1].
    y_probas : array-like of shape (n_samples, n_classes)
        Predicted class probabilities.

    Returns
    -------
    list of int
        Predicted class indices for each sample.
    """
    preds = []
    for row in y_probas:
        passed = [
            row[i] if row[i] >= thresholds[i] else -1 for i in range(len(thresholds))
        ]
        preds.append(np.argmax(passed) if max(passed) != -1 else np.argmax(row))
    return preds


def neg_mcc(thresholds, y_probas, y_true):
    """
    Negative MCC objective for threshold optimisation.

    Parameters
    ----------
    thresholds : array-like of shape (n_classes,)
        Per-class probability thresholds.
    y_probas : array-like of shape (n_samples, n_classes)
        Predicted class probabilities.
    y_true : array-like of shape (n_samples,)
        True class labels.

    Returns
    -------
    float
        Negative Matthews correlation coefficient.
    """
    preds = get_preds(thresholds, y_probas)
    return -matthews_corrcoef(y_true, preds)


def main():
    print(f"{sys.argv[0]}")
    start = dt.datetime.now()
    print("[INFO] Class weight and threshold tuning.")
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

    ap.add_argument("--classes", default="4", help="Number of classes.")

    ap.add_argument(
        "--debug", action="store_true", help="Run the script in debug mode."
    )
    ap.add_argument(
        "-nc", "--no-cal", action="store_true", help="Deactivate model calibration."
    )

    ap.add_argument("--rs", default="42", help="Random state, defaults to 42.")
    ap.add_argument(
        "--scorer",
        default="MCC",
        help="Scorer to tune to (MCC or F1-score. Defaults to MCC)",
    )

    ap.add_argument(
        "--trials",
        default=None,
        help="Number of Optuna trials. Defaults to 300, or 10 in debug mode.",
    )
    ap.add_argument(
        "--splits",
        default=None,
        help="Number of CV splits. Defaults to 5, or 2 in debug mode.",
    )
    ap.add_argument(
        "--feats",
        default=None,
        help="Max model features. Defaults to 20, or 5 in debug mode, unless model features file specified.",
    )
    ap.add_argument(
        "--feats-file",
        default=None,
        help=(
            "Optional file containing comma-separated feature counts "
            "(e.g. model_features.txt). Overrides --feats "
            "if provided."
        ),
    )

    args = ap.parse_args()
    args.rs = int(args.rs)
    args.classes = int(args.classes)

    if args.classes not in CLASS_BIN_EDGES:
        raise ValueError(
            f"[ERROR] args.classes={args.classes} not supported. "
            f"Supported values: {list(CLASS_BIN_EDGES.keys())}."
        )

    scorer_key = str(args.scorer).lower()
    if scorer_key in SCORER_MAP:
        args.scorer = SCORER_MAP[scorer_key]

    if args.scorer not in SCORERS:
        raise ValueError(
            f"[ERROR] Invalid scorer '{args.scorer}'. "
            f"Must be one of {list(SCORERS.keys())} or their aliases."
        )

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

    if args.splits is None:
        args.splits = 5 if not debug else 2
    else:
        args.splits = int(args.splits)

    if args.trials is None:
        args.trials = 300 if not debug else 10
    else:
        args.trials = int(args.trials)

    if args.feats_file is not None:
        if os.path.exists(args.feats_file):
            print(f"[INFO] Getting feature counts from {args.feats_file}")
            with open(args.feats_file, "r") as f:
                model_feats = f.read()
            model_feats = [int(x) for x in model_feats.split(",")]
        else:
            raise FileNotFoundError(f"[ERROR] Couldn't find {args.feats_file}")
    else:
        if args.feats is None:
            args.feats = 20 if not debug else 5
        else:
            args.feats = int(args.feats)
        model_feats = list(range(1, args.feats + 1))

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
    model_info["cw_range"] = (1, 10) if not debug else (1, 1)
    model_info["threshold_range"] = (0.1, 0.9) if not debug else (0.4, 0.6)
    model_info["python_version"] = sys.version
    model_info["pandas_version"] = pd.__version__
    model_info["numpy_version"] = np.__version__
    model_info["lightgbm_version"] = lgb.__version__
    model_info["sklearn_version"] = sklearn.__version__
    model_info["optuna_version"] = optuna.__version__

    bounds = [(0.0, 1.0)] * args.classes
    initial_guess = [0.5] * args.classes

    print("[INFO] Defining bins")

    bin_edges = [np.log(1) - 1e-3]
    bin_edges.append(np.log(2) - 1e-3)
    bin_edges.extend(
        [y[y > np.log(1)].quantile(x) for x in CLASS_BIN_EDGES[args.classes]]
    )
    bin_edges.append(y.max() + 1e-3)

    assert (
        len(bin_edges) == args.classes + 1
    ), f"[WARNING] bin_edges must have length args.classes+1 ({args.classes+1}), got {len(bin_edges)}."

    model_info["fixed_bins"] = bin_edges
    model_info["bin_counts"] = dict(
        pd.cut(y, bins=bin_edges, labels=CLASS_NAMES[args.classes]).value_counts()
    )

    model_info["model_feats"] = model_feats
    print(f"[INFO] Feature counts: {model_feats}")

    print(f"[INFO] Setting up outer cross-validation.")
    y_bins_for_cv = pd.cut(y, bins=bin_edges, labels=False)
    unique_bins = np.unique(y_bins_for_cv.dropna())
    assert len(unique_bins) == args.classes, (
        f"[WARNING] pd.cut produced {len(unique_bins)} classes, expected {args.classes}. "
        f"[WARNING] Unique bins found: {unique_bins}"
    )
    outer_cv = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.rs)

    importance_dfs = {}
    foldwise_best_params = {}
    foldwise_best_mccs = {}
    all_configs = []
    foldwise_best_cws = {}

    print("[INFO] Starting fold-wise cross-validation")
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y_bins_for_cv)):
        print(f"[INFO] [{fold + 1}/{args.splits}] Started at {dt.datetime.now()}")
        # get training and validation data for this fold
        print(
            f"[INFO] [{fold + 1}/{args.splits}] Getting fold training and validation data"
        )
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]

        y_train, y_val = y_bins_for_cv.iloc[train_idx], y_bins_for_cv.iloc[val_idx]
        if calibrate:
            print(
                f"[INFO] [{fold + 1}/{args.splits}] Splitting x_val and y_val into calibration and eval sets"
            )
            X_calib, X_val, y_calib, y_val = train_test_split(
                X_val,
                y_val,
                test_size=0.5,
                stratify=y_val,
                random_state=args.rs,
            )
        print(
            f"[INFO] [{fold + 1}/{args.splits}] Precomputing ranked features for candidate bins"
        )
        selector = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=args.classes,
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

        # scale the feature importances
        scaler = MinMaxScaler()
        split_scaled = scaler.fit_transform(split_importance.reshape(-1, 1)).flatten()
        gain_scaled = scaler.fit_transform(gain_importance.reshape(-1, 1)).flatten()
        combined_importance = (split_scaled + gain_scaled) / 2

        importance_dfs[fold + 1] = pd.DataFrame(
            {
                "feature": feature_names,
                f"split_fold_{fold+1}": split_importance,
                f"split_scaled_fold_{fold+1}": split_scaled,
                f"gain_fold_{fold+1}": gain_importance,
                f"gain_scaled_fold_{fold+1}": gain_scaled,
                f"importance_fold_{fold+1}": combined_importance,
            }
        )

        ranked_features = (
            pd.Series(combined_importance, index=feature_names)
            .sort_values(ascending=False)
            .index.tolist()
        )

        for n_feats in model_feats:
            if n_feats not in foldwise_best_params:
                foldwise_best_params[n_feats] = []
                foldwise_best_mccs[n_feats] = []
                foldwise_best_cws[n_feats] = []
            print(f"[INFO][{fold + 1}/{args.splits}][{n_feats} feats]")

            def objective(trial, ranked_features):
                if not debug:
                    if (
                        y_train.nunique() < args.classes
                        or y_val.nunique() < args.classes
                    ):
                        raise optuna.exceptions.TrialPruned(
                            f"[WARNING][{fold + 1}/{args.splits}][{n_feats} feats] Not enough unique bins in target: train bins: {y_train.nunique()}, val bins: {y_val.nunique()}"
                        )
                    train_props = check_bin_balance(y_train)
                    val_props = check_bin_balance(y_val)
                    # Prune if any class is < 10%
                    if any(train_props < 0.10):
                        raise optuna.exceptions.TrialPruned(
                            f"[WARNING][{fold + 1}/{args.splits}][{n_feats} feats] Train bin config too imbalanced"
                        )
                    if any(val_props < 0.10):
                        raise optuna.exceptions.TrialPruned(
                            f"[WARNING][{fold + 1}/{args.splits}][{n_feats} feats] Val bin config too imbalanced"
                        )

                top_feats = ranked_features[:n_feats]

                # hyperparams
                cw_type = trial.suggest_categorical("cw_type", ["balanced", "custom"])
                if cw_type == "balanced":
                    suggest_cws = "balanced"
                else:
                    suggest_cws = {}
                    for i in range(args.classes):
                        suggest_cws[i] = trial.suggest_int(
                            f"cw_class_{i}",
                            model_info["cw_range"][0],
                            model_info["cw_range"][1],
                        )

                cw_values = compute_class_weight(
                    suggest_cws, classes=np.unique(y_train), y=y_train
                )

                class_weight = {i: w for i, w in enumerate(cw_values)}

                # train classifier
                clf = lgb.LGBMClassifier(
                    objective="multiclass",
                    num_class=args.classes,
                    class_weight=class_weight,
                    random_state=args.rs,
                    verbosity=-1,
                )
                clf.fit(X_train[top_feats], y_train)
                if calibrate:
                    calibrated_clf = CalibratedClassifierCV(
                        clf, method="sigmoid", cv="prefit"
                    )
                    calibrated_clf.fit(X_calib[top_feats], y_calib)

                    preds = calibrated_clf.predict(X_val[top_feats])
                else:

                    preds = clf.predict(X_val[top_feats])

                performance_metrics = {}
                for k, score_f in SCORERS.items():
                    performance_metrics[k] = score_f(y_val, preds)

                # track trial info
                trial.set_user_attr("n_feats", n_feats)
                trial.set_user_attr("features", top_feats)
                for k, v in performance_metrics.items():
                    trial.set_user_attr(k, v)
                trial.set_user_attr("cw_type", cw_type)
                trial.set_user_attr("cw", class_weight)
                trial.set_user_attr("cw_raw", cw_values)
                return performance_metrics[args.scorer]

            print(
                f"[INFO][{fold + 1}/{args.splits}][{n_feats} feats] Starting Optuna trials"
            )
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=args.rs),
            )
            study.optimize(
                lambda trial: objective(trial, ranked_features),
                n_trials=args.trials,
                n_jobs=-1,
            )

            foldwise_best_params[n_feats].append(study.best_params)
            foldwise_best_mccs[n_feats].append(study.best_value)
            foldwise_best_cws[n_feats].append(study.best_trial.user_attrs["cw"])

            try:
                fig = vis.plot_optimization_history(study)
                fig.write_image(f"{plot_outdir}/optuna_fold{fold+1}_convergence.png")
            except Exception as e:
                print(
                    f"[WARNING][{fold + 1}/{args.splits}] Could not save Optuna convergence plot: {e}"
                )

            # Collect all good trials
            fold_trials = []
            for t in study.trials:
                if t.value is not None:
                    to_append = {
                        "fold": fold + 1,
                        "trial_number": t.number,
                    }
                    to_append.update(t.user_attrs)
                    fold_trials.append(to_append)

            fold_results_df = pd.DataFrame(fold_trials)
            all_configs.append(fold_results_df)

        print(f"[OK][{fold + 1}/{args.splits}] Finished at {dt.datetime.now()}")

        del selector, fold_results_df, ranked_features
        gc.collect()

    good_enough_importance_dfs = [df for df in importance_dfs.values()]

    importance_merged = reduce(
        lambda left, right: pd.merge(left, right, on="feature", how="outer"),
        good_enough_importance_dfs,
    )
    # fill missing values with 0
    importance_merged.fillna(0, inplace=True)

    # Calculate mean and std of feature importance across folds
    for imp_colname in ["importance", "split", "gain"]:
        imp_cols = [
            col
            for col in importance_merged.columns
            if col.startswith(f"{imp_colname}_fold_")
        ]
        importance_merged[f"mean_{imp_colname}"] = importance_merged[imp_cols].mean(
            axis=1
        )
        importance_merged[f"std_{imp_colname}"] = importance_merged[imp_cols].std(
            axis=1
        )
    # sort by average importance
    importance_merged.sort_values(by="mean_importance", ascending=False, inplace=True)

    ranked_features = importance_merged["feature"].tolist()

    params = {}
    all_configs_df = pd.concat(all_configs, ignore_index=True)
    all_configs_df["cw_str"] = all_configs_df["cw"].astype(str)

    # Aggregate hyperparameters across folds
    def aggregate_params(param_list):
        df = pd.DataFrame(param_list)
        agg = {}
        for col in df.columns:
            # For categorical/int values (e.g., num_leaves, max_depth), use mode
            if df[col].dtype.kind in "iO":
                agg[col] = df[col].mode().iloc[0]
            else:
                agg[col] = df[col].mean()
        return agg

    optuna_params = {}
    class_weights = {}
    for n_feats in model_feats:
        best_params = aggregate_params(foldwise_best_params[n_feats])
        optuna_params[n_feats] = {
            "best_params": best_params,
            "best_MCC": np.mean(foldwise_best_mccs[n_feats]),
        }
        class_weights[n_feats] = aggregate_params(foldwise_best_cws[n_feats])

    joblib.dump(optuna_params, f"{args.outdir}/optuna_params.jl")
    joblib.dump(importance_dfs, f"{args.outdir}/foldwise_importances.jl")

    foldwise_thresholds = {}
    foldwise_mcc_thresholds = {}

    print("[INFO] Training data on folds")
    for fold, (cal_idx, val_idx) in enumerate(outer_cv.split(X, y_bins_for_cv)):
        print(f"[INFO][{fold + 1}/{args.splits}]")
        X_train, X_val = X.iloc[cal_idx], X.iloc[val_idx]
        y_train, y_val = y_bins_for_cv.iloc[cal_idx], y_bins_for_cv.iloc[val_idx]

        if calibrate:
            X_train, X_calib, y_train, y_calib = train_test_split(
                X_train, y_train, train_size=0.8, stratify=y_train
            )

        X_thresh_cal, X_val, y_thresh_cal, y_val = train_test_split(
            X_val, y_val, train_size=0.5, stratify=y_val
        )

        for n_feats in model_feats:
            print(f"[INFO][{fold + 1}/{args.splits}][{n_feats} feats]")
            if n_feats not in foldwise_thresholds:
                foldwise_thresholds[n_feats] = []
                foldwise_mcc_thresholds[n_feats] = []
            top_feats = ranked_features[:n_feats]
            best_params = optuna_params[n_feats]["best_params"]
            class_weight = class_weights[n_feats]

            clf = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=args.classes,
                class_weight=class_weight,
                random_state=args.rs,
                verbosity=-1,
            )
            clf.fit(X_train[top_feats], y_train)

            if calibrate:
                calibrated_clf = CalibratedClassifierCV(
                    clf, method="sigmoid", cv="prefit"
                )
                calibrated_clf.fit(X_calib[top_feats], y_calib)
                proba = calibrated_clf.predict_proba(X_thresh_cal[top_feats])

            else:
                proba = clf.predict_proba(X_thresh_cal[top_feats])

            result = minimize(
                neg_mcc,
                x0=initial_guess,
                bounds=bounds,
                args=(proba, y_thresh_cal),
                method="L-BFGS-B",
            )

            best_thresholds = result.x
            best_mcc = -result.fun
            foldwise_thresholds[n_feats].append(best_thresholds)

            proba = (
                calibrated_clf.predict_proba(X_val[top_feats])
                if calibrate
                else clf.predict_proba(X_val[top_feats])
            )
            preds = get_preds(best_thresholds, proba)
            foldwise_mcc_thresholds[n_feats].append(matthews_corrcoef(y_val, preds))

    for n_feats in model_feats:
        best_params = optuna_params[n_feats]["best_params"]
        class_weight = class_weights[n_feats]
        threshold_vals = []
        threshold_stds = []
        for i in range(args.classes):
            class_i_threshold = [t[i] for t in foldwise_thresholds[n_feats]]
            threshold_vals.append(np.mean(class_i_threshold))
            threshold_stds.append(np.std(class_i_threshold))

        params[n_feats] = {
            "n_feats": n_feats,
            "bins": bin_edges,
            "mcc_before_thresh": np.mean(foldwise_best_mccs[n_feats]),
            "final_threshold": threshold_vals,
            "threshold_stds": threshold_stds,
            "mcc_after_thresh": np.mean(foldwise_mcc_thresholds[n_feats]),
            "final_class_weights": class_weight,
            "features": ranked_features[:n_feats],
        }

    # Save joblib
    joblib.dump(params, f"{args.outdir}/tuning_params.jl")
    all_configs_df.to_csv(f"{args.outdir}/all_configs_df.csv", index=False)

    joblib.dump(all_configs_df, f"{args.outdir}/all_configs_df.jl")

    flat_params = []

    for n_feats in params:
        for i, feat in enumerate(params[n_feats]["features"]):
            to_append = {
                "n_feats": n_feats,
                "feature_rank": i + 1,
                "feature_name": feat,
            }
            to_append.update(
                {
                    "class_weights": params[n_feats]["final_class_weights"],
                    "threshold": params[n_feats]["final_threshold"],
                }
            )
            flat_params.append(to_append)

    with pd.ExcelWriter(f"{args.outdir}/tuning_outputs.xlsx") as writer:
        model_info_df = (
            pd.DataFrame.from_dict(model_info, orient="index")
            .reset_index(names="param")
            .rename(columns={0: "value"})
        )
        pd.DataFrame.from_dict(model_info_df, orient="index").to_excel(
            writer, sheet_name="model_info", index=False
        )
        pd.DataFrame.from_dict(params, orient="index").to_excel(
            writer, sheet_name="params", index=False
        )

        all_configs_df.to_excel(writer, sheet_name="all_configs", index=False)
        importance_merged.to_excel(
            writer, sheet_name="feature_importances", index=False
        )
        pd.DataFrame(flat_params).to_excel(
            writer, sheet_name="flattened_features", index=False
        )

    print(f"[INFO] Saved all outputs to: {args.outdir}")
    final_call = dt.datetime.now()
    print(f"[OK] Finished at {final_call}. Total time taken: {final_call - start}")


if __name__ == "__main__":
    main()
