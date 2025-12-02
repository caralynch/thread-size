# MIT License
# Copyright (c) 2025 Cara Lynch
# See the LICENSE file for details.
"""
Stage 1.3 - tree hyperparameter tuning for thread-start classification.

This script takes the tuned configurations from Stage 1.2 (feature subsets,
class weights, and decision thresholds for predicting whether a thread
starts) and refines the LightGBM tree hyperparameters using cross-validated
Optuna optimisation.

High-level behaviour
--------------------
Inputs:
    - --train_X : Training feature matrix (parquet).
    - --train_y : Training target data (parquet) containing --y-col.
    - --params  : Joblib file with 2_tuning outputs:
              {n_feats: {...}, ...}
      Each entry in "params" is expected to contain:
          * "features"           : list of feature names,
          * "final_class_weights": dict for class_weight,
          * "final_threshold"    : scalar decision threshold.

Target construction:
    - --y-col (default: "log_thread_size") is thresholded at --y_thresh
      (default log(1) == 0) to define a binary label:
          y = 1  ⇔  log_thread_size > y_thresh  (thread started)
          y = 0  ⇔  log_thread_size ≤ y_thresh  (thread stalled).

Cross-validation:
    - StratifiedKFold with --splits folds (default 5, or 2 in debug mode),
      shuffling and fixed random seed (--rs).
    - For each fold and each feature-count n_feats:
        * use the Stage 1.2 feature subset and class weights,
        * split the fold's validation data into calibration and evaluation
          subsets when calibration is enabled,
        * run an Optuna study to tune tree hyperparameters.

Tuning:
    - The Optuna objective maximises a primary scorer (--scorer) such as
      MCC, F1, or F-beta (with configurable --beta) evaluated on the
      evaluation subset.
    - The search space includes typical LightGBM tree and regularisation
      hyperparameters (num_leaves, max_depth, learning_rate, min_child_samples,
      subsample, colsample_bytree, reg_alpha, reg_lambda).
    - Trials that predict only a single class are pruned early.

Aggregation:
    - For each n_feats, best parameters across folds are aggregated:
        * integer / categorical parameters use the mode,
        * floating-point parameters use the mean.
    - The mean best score across folds for the chosen scorer is recorded.

Outputs
-------
Written to --outdir:

    - params_post_hyperparam_tuning.jl
        A joblib dict with:
            * "info"   : model_info (arguments, versions, runtime, feature_counts).
            * "params" : updated per-n_feats configs including:
                - "best_hyperparams"    : aggregated tree hyperparameters.
                - f"best_{scorer_name}" : mean cross-fold score.

    - {n_feats}_feats_best_hyperparams_fold_{k}.jl
        Best hyperparameters for each (n_feats, fold) combination.

    - {n_feats}_feats_best_hyperparams.jl
        Aggregated best hyperparameters per n_feats.

Reproducibility
---------------
- All randomised operations (CV splits, calibration splits, and Optuna sampler)
  are seeded via --rs.
- Command-line arguments and library versions (Python, pandas, numpy,
  LightGBM, scikit-learn) are stored in model_info.
- Debug mode (--debug) reduces CV splits and Optuna trials for rapid iteration.
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
    matthews_corrcoef,
    f1_score,
    balanced_accuracy_score,
    fbeta_score,
)

from sklearn.model_selection import StratifiedKFold, train_test_split

import optuna

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


def main():
    """
    Main entry point for Stage 1 tree hyperparameter tuning.
    
    Refines LightGBM tree hyperparameters for binary thread-start
    classification using cross-validated Optuna optimization.
    """
    print(f"{sys.argv[0]}")
    start = dt.datetime.now()
    print("[INFO] Hyperparameter tuning.")
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

    ap.add_argument("--params", help="Tuned model params file (jl).")

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

    if not os.path.isfile(args.params):
        raise FileNotFoundError(f"[ERROR] Model params file not found: {args.params}")

    print(f"[INFO] Args: {args}")

    os.makedirs(args.outdir, exist_ok=True)

    print(f"[INFO] Loading training data.")
    X = pd.read_parquet(args.train_X)
    y = pd.read_parquet(args.train_y)[args.y_col]
    # threshold to identify started threads
    y = (y > args.y_thresh).astype(int)

    print(f"[INFO] Loading tuning params")
    params = joblib.load(args.params)

    # run data for outfile
    model_info = {
        "script": str(sys.argv[0]),
        "run_start": start,
    }

    model_info.update(vars(args))
    model_info["python_version"] = sys.version
    model_info["pandas_version"] = pd.__version__
    model_info["numpy_version"] = np.__version__
    model_info["lightgbm_version"] = lgb.__version__
    model_info["sklearn_version"] = sklearn.__version__
    model_info["optuna_version"] = optuna.__version__

    print("[INFO] Getting feature counts from params...")
    feature_counts = list(params.keys())
    model_info["feature_counts"] = feature_counts
    print(f"[OK] Feature counts: {feature_counts}")

    print(f"[INFO] Setting up outer cross-validation.")
    outer_cv = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.rs)

    foldwise_best_params = {}
    foldwise_best_scores = {}

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
        print(f"[INFO] Outer CV fold {fold + 1}/{args.splits}")
        for n_feats in feature_counts:
            if n_feats not in foldwise_best_params:
                foldwise_best_params[n_feats] = []
                foldwise_best_scores[n_feats] = []
            config = params[n_feats]
            print(f"[INFO][{fold + 1}/{args.splits}][{n_feats} feats]")
            x_cols = config["features"]
            cw = config["final_class_weights"]
            thresh = config["final_threshold"]

            # Sanity check on threshold from Stage 1.2
            if not (0.0 <= float(thresh) <= 1.0):
                raise ValueError(
                    f"[ERROR] For n_feats={n_feats}, final_threshold={thresh} "
                    "is outside [0, 1]."
                )

            print(
                f"[INFO][{fold + 1}/{args.splits}][{n_feats} feats] Selected features: {x_cols}"
            )
            print(
                f"[INFO][{fold + 1}/{args.splits}][{n_feats} feats] "
                f"Class weights: {cw}, Threshold: {thresh}"
            )

            # Load training data
            X_tr, X_val = X[x_cols].iloc[train_idx], X[x_cols].iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if calibrate:
                # Split X_val and y_val into calibration and evaluation sets
                X_calib, X_eval, y_calib, y_eval = train_test_split(
                    X_val,
                    y_val,
                    test_size=0.5,  # 50% for calibration, 50% for evaluation
                    stratify=y_val,  # preserve class balance
                    random_state=args.rs,  # for reproducibility
                )
            else:
                X_eval, y_eval = X_val, y_val

            def objective(trial):
                param = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "boosting_type": "gbdt",
                    "random_state": args.rs,
                    "verbosity": -1,
                    "class_weight": cw,  # from STAGE_1_TUNING
                    "num_leaves": trial.suggest_int(
                        "num_leaves", 20, 25 if debug else 150
                    ),
                    "max_depth": trial.suggest_int("max_depth", 3, 6 if debug else 15),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.1 if debug else 1e-3, 0.2, log=True
                    ),
                    "min_child_samples": trial.suggest_int(
                        "min_child_samples", 5, 10 if debug else 100
                    ),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.5, 1.0
                    ),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
                }

                clf = lgb.LGBMClassifier(**param)
                clf.fit(X_tr, y_tr)
                if calibrate:
                    calibrated_clf = CalibratedClassifierCV(
                        clf, method="isotonic", cv="prefit"
                    )
                    calibrated_clf.fit(X_calib, y_calib)
                    proba = calibrated_clf.predict_proba(X_eval)[:, 1]
                else:
                    proba = clf.predict_proba(X_eval)[:, 1]

                preds = (proba >= thresh).astype(int)
                if not debug:
                    if len(np.unique(preds)) < 2:
                        raise optuna.TrialPruned(
                            f"[WARNING][{fold + 1}/{args.splits}][{n_feats} feats] Only one class predicted, skipping trial."
                        )

                score = SCORERS[args.scorer](y_eval, preds)

                return score

            print(
                f"[INFO][{fold + 1}/{args.splits}][{n_feats} feats] Starting hyperparameter tuning for {n_feats}"
            )

            sampler = optuna.samplers.TPESampler(seed=args.rs)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=args.trials)

            foldwise_best_params[n_feats].append(study.best_params)
            foldwise_best_scores[n_feats].append(study.best_value)

            print(
                f"[INFO] Fold {fold+1}: Best {args.scorer} for {n_feats} feats: {study.best_value}"
            )
            joblib.dump(
                study.best_params,
                f"{args.outdir}/{n_feats}_feats_best_hyperparams_fold_{fold+1}.jl",
            )
        # Save foldwise best scores for diagnostics / publication
    rows = []
    for n_feats in feature_counts:
        for fold_idx, score in enumerate(foldwise_best_scores[n_feats], start=1):
            rows.append(
                {"n_feats": n_feats, "fold": fold_idx, args.scorer: score,}
            )

    fold_scores_df = pd.DataFrame(rows)
    fold_scores_path = f"{args.outdir}/foldwise_best_scores.csv"
    fold_scores_df.to_csv(fold_scores_path, index=False)
    print(f"[INFO] Saved foldwise best {args.scorer} scores to {fold_scores_path}")

    for n_feats in feature_counts:
        config = params[n_feats]
        config["best_hyperparams"] = aggregate_params(foldwise_best_params[n_feats])
        config[f"best_{args.scorer}"] = np.mean(foldwise_best_scores[n_feats])
        joblib.dump(
            config["best_hyperparams"],
            f"{args.outdir}/{n_feats}_feats_best_hyperparams.jl",
        )
        print(
            f"[INFO][{n_feats}] Best hyperparameters: {config['best_hyperparams']}, "
            f"[INFO][{n_feats}] Mean {args.scorer}: {config[f'best_{args.scorer}']}"
        )

    end = dt.datetime.now()
    model_info["hyperparameter_tuning_runtime"] = str(end - start)
    pd.DataFrame.from_dict(model_info, orient="index").to_csv(
        f"{args.outdir}/tuning_log.csv"
    )

    joblib.dump(params, f"{args.outdir}/params_post_hyperparam_tuning.jl")
    print(f"[OK] Saved all outputs to: {args.outdir}")
    print(f"[OK] Finished. Total runtime {end-start}.")


if __name__ == "__main__":
    main()
