# MIT License
# Copyright (c) 2025 Cara Lynch
# See the LICENSE file for details.
"""
Stage 1.2 - class-weight and threshold tuning for thread-start classification.

This script tunes LightGBM classifiers that predict whether a thread starts
(thread_size > 1) using cross-validated Optuna searches over class weights,
and then tunes a decision threshold per model to maximise a chosen metric.

Inputs
------
- --train_X : Training feature matrix (parquet).
- --train_y : Training target data (parquet) containing --y-col.
- --subreddit : Key in LABEL_LOOKUP (e.g. 'conspiracy', 'crypto', 'politics').

Target construction
-------------------
- --y-col (default: 'log_thread_size') is thresholded at --y_thresh
  (default: log(1) == 0) to define a binary label:
      y = 1  ⇔  log_thread_size > y_thresh  (thread started)
      y = 0  ⇔  log_thread_size ≤ y_thresh  (thread stalled).

Cross-validation & feature ranking
----------------------------------
- StratifiedKFold with --splits folds (default 5, or 2 in debug mode), using
  a fixed random seed (--rs).
- Within each outer fold, a LightGBM classifier is fitted with
  class_weight="balanced" to obtain split- and gain-based feature importances;
  these are min–max scaled and averaged to yield a combined importance score
  per feature.
- For each n_feats in the feature grid (from --feats or --feats-file), models
  are restricted to the top-n_feats ranked features.

Optuna tuning
-------------
- For each fold and each n_feats:
    - An Optuna study maximises the chosen scorer (--scorer: MCC, F-beta, F1,
      or balanced accuracy).
    - The search space includes:
        * class-weight type: "balanced" vs custom weighted,
        * positive-class weight ratio cw_ratio within cw_ratio_range.
    - If calibration is enabled (default), probabilities are calibrated via
      isotonic regression on a held-out calibration set (50% of the val fold).

Threshold tuning
----------------
- After aggregating best parameters and class weights across folds, models
  are re-trained.
- Validation data are split into:
    * threshold-calibration subset, and
    * evaluation subset.
- A scalar optimiser finds the threshold in [0, 1] that maximises the primary
  scorer (via a negative scoring wrapper).
- Metrics with a fixed 0.5 probability threshold and with the tuned threshold
  are recorded for the same models and validation splits, allowing a direct
  comparison of the impact of threshold optimisation.


Outputs
-------
Written to --outdir:

- tuned_params.jl:
    joblib dict with {"info": model_info, "params": per-n_feats configs}
    (used as --params input to Stage 1.3).
- optuna_params.jl:
    cross-fold aggregated best params & scores per n_feats.
- tuning_outputs.xlsx:
    * model_info
    * params (per n_feats: scores, thresholds, class weights, features)
    * feature_importances (cross-fold importances)
    * flattened_features (feature-by-feature)
    * all_configs (all Optuna trial configurations).
- optuna_fold*_convergence.png:
    convergence plots per fold and n_feats.

Reproducibility
---------------
- All randomised operations use the --rs seed (CV, Optuna sampler, splits
  for calibration/threshold tuning).
- Command-line arguments and library versions are stored in model_info.
- Debug mode (--debug) reduces folds, trials, feature counts, and (optionally)
  search ranges for quick iteration.
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
from scipy.optimize import minimize_scalar

import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    balanced_accuracy_score,
    fbeta_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import compute_class_weight

import optuna
import optuna.visualization as vis

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL_LOOKUP = {
    "crypto": "r/CryptoCurrency",
    "politics": "r/politics",
    "conspiracy": "r/Conspiracy",
}

SCORERS = {
    "MCC": matthews_corrcoef,
    "F1": f1_score,
    "Balanced accuracy": balanced_accuracy_score,
}

SCORER_MAP = {
    "mcc": "MCC",
    "f1": "F1",
    "f1-score": "F1",
    "f": "F-beta",
    "fb": "F-beta",
    "fbeta": "F-beta",
    "f-beta": "F-beta",
    "balanced": "Balanced accuracy",
    "balanced_accuracy": "Balanced accuracy",
}

CLASS_NAMES = {
    0: "Stalled",
    1: "Started",
}


def main():
    """
    Main entry point for Stage 1 class weight and threshold tuning.
    
    Uses Optuna to optimize class weights and decision thresholds for
    binary thread-start classification across multiple feature set sizes.
    """
    print(f"{sys.argv[0]}")
    start = dt.datetime.now()
    print("[INFO] Class weight and threshold tuning.")
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

    ap.add_argument(
        "--y_thresh",
        default=None,
        help="Target y threshold to identify started threads. Defaults to log(1).",
    )

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
        help="Scorer to tune to (F-beta, MCC or F1-score. Defaults to MCC)",
    )

    ap.add_argument(
        "--beta", default="2", help="Beta for f-beta score. Default 2.",
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
            "(e.g. model_features.txt from Stage 1.1). Overrides --feats "
            "if provided."
        ),
    )

    args = ap.parse_args()
    args.rs = int(args.rs)
    args.beta = float(args.beta)

    SCORERS["F-beta"] = partial(fbeta_score, beta=args.beta)

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

    if args.y_thresh is None:
        args.y_thresh = np.log(1)
    else:
        args.y_thresh = float(args.y_thresh)

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

    print(f"[INFO] Loading training data.")
    X = pd.read_parquet(args.train_X)
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
    model_info["cw_ratio_range"] = (0.1, 5.0) if not debug else (1.0, 1.0)
    model_info["python_version"] = sys.version
    model_info["pandas_version"] = pd.__version__
    model_info["numpy_version"] = np.__version__
    model_info["lightgbm_version"] = lgb.__version__
    model_info["sklearn_version"] = sklearn.__version__
    model_info["optuna_version"] = optuna.__version__

    importance_dfs = []
    foldwise_best_params = {}
    foldwise_best_scores = {}
    foldwise_best_cws = {}
    all_folds_df_dict = {}
    all_configs = []

    print("[INFO] Training data on folds")
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
        all_folds_df_dict[fold + 1] = {}
        print(f"[INFO] Outer fold {fold + 1}/{args.splits}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if calibrate:
            print(
                f"[INFO] [Fold {fold + 1}] Splitting x_val and y_val into calibration and eval sets"
            )
            # Split X_val and y_val into calibration and evaluation sets
            X_calib, X_val, y_calib, y_val = train_test_split(
                X_val,
                y_val,
                test_size=0.5,  # 50% for calibration, 50% for evaluation
                stratify=y_val,  # preserve class balance
                random_state=args.rs,  # for reproducibility
            )

        print(
            f"[INFO] [Fold {fold + 1}] Getting feature importances for feature ranking"
        )
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

        for n_feats in model_feats:
            if n_feats not in foldwise_best_params:
                foldwise_best_params[n_feats] = []
                foldwise_best_scores[n_feats] = []
                foldwise_best_cws[n_feats] = []
            print(f"[INFO] [Fold {fold + 1}] n_feats: {n_feats}")
            top_feats = ranked_features[:n_feats]

            def objective(trial):
                # Suggest hyperparameters
                cw_type = trial.suggest_categorical("cw_type", ["balanced", "weighted"])
                if cw_type == "balanced":
                    suggest_cws = "balanced"
                else:
                    cw_ratio = trial.suggest_float(
                        "cw_ratio",
                        model_info["cw_ratio_range"][0],
                        model_info["cw_ratio_range"][1],
                        log=True,
                    )
                    suggest_cws = {0: 1.0, 1: cw_ratio}

                cw_values = compute_class_weight(
                    suggest_cws, classes=np.unique(y_train), y=y_train
                )

                class_weight = {i: w for i, w in enumerate(cw_values)}

                clf = lgb.LGBMClassifier(
                    objective="binary",
                    class_weight=class_weight,
                    random_state=args.rs,
                    verbose=-1,
                )
                clf.fit(X_train[top_feats], y_train)

                if calibrate:
                    calibrated_clf = CalibratedClassifierCV(
                        clf, method="isotonic", cv="prefit"
                    )
                    calibrated_clf.fit(X_calib[top_feats], y_calib)
                    proba = calibrated_clf.predict_proba(X_val[top_feats])[:, 1]
                    preds = calibrated_clf.predict(X_val[top_feats])

                else:
                    proba = clf.predict_proba(X_val[top_feats])[:, 1]
                    preds = clf.predict(X_val[top_feats])

                scoring_metrics = {}
                for key, score_func in SCORERS.items():
                    scoring_metrics[key] = score_func(y_val, preds)
                    trial.set_user_attr(key, scoring_metrics[key])

                trial.set_user_attr("cw_type", cw_type)
                # trial.set_user_attr("cw_ratio", cw_ratio)
                trial.set_user_attr("cw", class_weight)
                trial.set_user_attr("n_feats", n_feats)
                trial.set_user_attr("features", top_feats)

                return scoring_metrics[
                    args.scorer
                ]  # adjust this to prioritize F1 or MCC if needed

            print(
                f"[INFO] [Fold {fold + 1}] [{n_feats} features] Launching Optuna study"
            )
            study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.rs)
            )
            study.optimize(objective, n_trials=args.trials)
            print(
                f"[OK] [Fold {fold + 1}] [{n_feats} features] Finished Optuna optimization"
            )

            foldwise_best_params[n_feats].append(study.best_params)
            foldwise_best_scores[n_feats].append(study.best_value)
            foldwise_best_cws[n_feats].append(study.best_trial.user_attrs["cw"])

            print(
                f"[INFO] [Fold {fold + 1}] [{n_feats} features] Plotting Optuna study"
            )
            try:
                fig = vis.plot_optimization_history(study)
                fig.write_image(f"{args.outdir}/optuna_fold{fold+1}_convergence.png")
            except Exception as e:
                print(
                    f"[WARNING][{fold + 1}/{args.splits}] Could not save Optuna convergence plot: {e}"
                )

            fold_trials = []
            for t in study.trials:
                if t.value is not None:
                    fold_trial_dict = {
                        "fold": fold + 1,
                        "n_feats": n_feats,
                    }
                    for key, val in t.user_attrs.items():
                        fold_trial_dict[key] = val
                    fold_trials.append(fold_trial_dict)

            fold_results_df = pd.DataFrame(fold_trials)
            all_folds_df_dict[fold + 1][n_feats] = fold_results_df
            all_configs.append(fold_results_df)

    # Aggregate hyperparameters across folds
    def aggregate_params(param_list):
        """
        Aggregate a list of per-fold best-parameter dictionaries.

        For each parameter key:
          - integer / object-like (categorical) parameters use the mode,
          - floating-point parameters use the mean.

        Parameters
        ----------
        param_list : list of dict
            One dictionary of best params per CV fold.

        Returns
        -------
        dict
            Aggregated parameters representative of cross-fold performance.
        """
        df = pd.DataFrame(param_list)
        agg = {}
        for col in df.columns:
            # For categorical/int values (e.g., num_leaves, max_depth), use mode
            if df[col].dtype.kind in "iO":
                agg[col] = df[col].mode().iloc[0]
            else:
                agg[col] = df[col].mean()
        return agg

    # Get cross-fold feature importance for output
    # Merge feature importances from all folds
    print(f"[INFO] Getting cross-fold feature importance")
    importance_merged = importance_dfs[0]
    for df in importance_dfs[1:]:
        importance_merged = importance_merged.merge(df, on="feature", how="outer")
    importance_merged.fillna(0, inplace=True)  # fill missing values with 0
    # Compute mean and std across folds
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
    # Sort by average importance
    importance_merged.sort_values("mean_importance", ascending=False, inplace=True)

    all_configs_df = pd.concat(all_configs, ignore_index=True)
    all_configs_df["cw_str"] = all_configs_df["cw"].astype(str)

    ranked_features = importance_merged["feature"].tolist()
    optuna_params = {}
    class_weights = {}
    for n_feats in model_feats:
        best_params = aggregate_params(foldwise_best_params[n_feats])
        optuna_params[n_feats] = {
            "best_params": best_params,
            f"best_{args.scorer}": np.mean(foldwise_best_scores[n_feats]),
        }
        class_weights[n_feats] = aggregate_params(foldwise_best_cws[n_feats])

    print(f"[INFO] Saving optuna parameters")
    joblib.dump(optuna_params, f"{args.outdir}/optuna_params.jl")

    def neg_scorer(threshold, y_proba, y_true):
        """
        Negative primary score as a function of decision threshold.

        Used with scipy.optimize.minimize_scalar to find the threshold in [0, 1]
        that maximises the selected scorer (MCC, F1, F-beta, etc.) on a
        calibration subset.

        Parameters
        ----------
        threshold : float
            Candidate probability threshold in [0, 1].
        y_proba : array-like, shape (n_samples,)
            Predicted positive-class probabilities.
        y_true : array-like, shape (n_samples,)
            True binary labels.

        Returns
        -------
        float
            Negative value of the chosen scorer at this threshold (so that
            minimisation corresponds to maximisation).
        """
        y_pred = (y_proba >= threshold).astype(int)
        return -SCORERS[args.scorer](y_true, y_pred)

    foldwise_thresholds = {}
    foldwise_score_thresholds = {}
    foldwise_score_baseline = {}
    print("[INFO] Training data with optimized model")
    for fold, (cal_idx, val_idx) in enumerate(outer_cv.split(X, y)):
        print(f"[INFO] [Fold {fold + 1}]")
        X_train, X_val = X.iloc[cal_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[cal_idx], y.iloc[val_idx]

        if calibrate:
            X_train, X_calib, y_train, y_calib = train_test_split(
                X_train, y_train, train_size=0.8, stratify=y_train, random_state=args.rs
            )

        X_thresh_cal, X_val, y_thresh_cal, y_val = train_test_split(
            X_val, y_val, train_size=0.5, stratify=y_val, random_state=args.rs
        )

        for n_feats in model_feats:
            if n_feats not in foldwise_thresholds:
                foldwise_thresholds[n_feats] = []
                foldwise_score_thresholds[n_feats] = []
            top_feats = ranked_features[:n_feats]
            best_params = optuna_params[n_feats]["best_params"]
            class_weight = class_weights[n_feats]

            clf = lgb.LGBMClassifier(
                objective="binary",
                class_weight=class_weight,
                random_state=args.rs,
                verbose=-1,
            )
            clf.fit(X_train[top_feats], y_train)

            if calibrate:
                calibrated_clf = CalibratedClassifierCV(
                    clf, method="isotonic", cv="prefit"
                )
                calibrated_clf.fit(X_calib[top_feats], y_calib)
                proba_cal = calibrated_clf.predict_proba(X_thresh_cal[top_feats])[:, 1]

            else:
                proba_cal = clf.predict_proba(X_thresh_cal[top_feats])[:, 1]

            result = minimize_scalar(
                neg_scorer, bounds=(0, 1), method="bounded", args=(proba_cal, y_thresh_cal)
            )
            best_thresh = result.x
            foldwise_thresholds[n_feats].append(best_thresh)

            proba_val = (
                calibrated_clf.predict_proba(X_val[top_feats])[:, 1]
                if calibrate
                else clf.predict_proba(X_val[top_feats])[:, 1]
            )
            # baseline: fixed 0.5 threshold
            baseline_preds = (proba_val >= 0.5).astype(int)
            baseline_score = SCORERS[args.scorer](y_val, baseline_preds)
            foldwise_score_baseline[n_feats].append(baseline_score)
            # after optimisation: tuned threshold
            tuned_preds = (proba_val >= best_thresh).astype(int)
            tuned_score = SCORERS[args.scorer](y_val, tuned_preds)
            foldwise_score_thresholds[n_feats].append(tuned_score)


    params = {}
    for n_feats in model_feats:
        best_params = optuna_params[n_feats]["best_params"]
        class_weight = class_weights[n_feats]
        params[n_feats] = {
            "n_feats": n_feats,
            # baseline: fixed 0.5 threshold on the same model/split
            f"{args.scorer}_before_thresh": np.mean(foldwise_score_baseline[n_feats]),
            # tuned scalar probability threshold
            "final_threshold": np.mean(foldwise_thresholds[n_feats]),
            # after: optimised threshold on the same model/split
            f"{args.scorer}_after_thresh": np.mean(
                foldwise_score_thresholds[n_feats]
            ),
            "final_class_weights": class_weight,
            "features": ranked_features[:n_feats],
        }


    end = dt.datetime.now()

    model_info["tuning_runtime"] = str(end - start)

    joblib.dump(params, f"{args.outdir}/tuned_params.jl")

    flat_params = []
    for n_feats, model_dict in params.items():
        for i, feat in enumerate(model_dict["features"]):
            flat_params.append(
                {
                    "n_feats": n_feats,
                    "feature_rank": i + 1,
                    "feature_name": feat,
                    "class_weights": model_dict["final_class_weights"],
                    "threshold": model_dict["final_threshold"],
                }
            )

    with pd.ExcelWriter(f"{args.outdir}/tuning_outputs.xlsx") as writer:
        model_info_df = (
            pd.DataFrame.from_dict(model_info, orient="index")
            .reset_index(names="param")
            .rename(columns={0: "value"})
        )
        model_info_df.to_excel(writer, sheet_name="model_info", index=False)
        pd.DataFrame.from_dict(params, orient="index").to_excel(
            writer, sheet_name="params", index=False
        )

        importance_merged.to_excel(
            writer, sheet_name="feature_importances", index=False
        )
        pd.DataFrame(flat_params).to_excel(
            writer, sheet_name="flattened_features", index=False
        )

        all_configs_df.to_excel(writer, sheet_name=f"all_configs", index=False)

    print(f"Saved all outputs to: {args.outdir}")
    print(f"Finished. Total runtime {end-start}.")


if __name__ == "__main__":
    main()
