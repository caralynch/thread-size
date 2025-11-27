"""
3_hyperparameter_tuning.py

Multiclass hyperparameter tuning for thread-size classification.

This script refines LightGBM hyperparameters for the multiclass
thread-size model, conditional on:

    * a fixed discretisation of log(thread_size) into K ordinal bins, and
    * a fixed set of class weights and per-class decision thresholds, and
    * a set of feature subsets (indexed by n_feats).

Given a precomputed parameter dictionary (from the tuning script)
containing, for each n_feats:

    - `features`           : list of feature names to use,
    - `bins`               : bin edges for discretising log_thread_size,
    - `final_class_weights`: LightGBM class_weight mapping,
    - `final_threshold`    : per-class probability thresholds,

this script:

    1. Loads X (features) and y (continuous log_thread_size).
    2. Discretises y into ordinal class labels using the bin edges from the
       first configuration, and uses these labels for Stratified K-fold CV.
    3. For each outer CV fold and each n_feats:
         a. Restricts X to the selected feature subset.
         b. Optionally splits the validation fold into calibration and
            evaluation halves, and fits a CalibratedClassifierCV wrapper.
         c. Runs an Optuna/TPE search over LightGBM tree hyperparameters
            (num_leaves, max_depth, learning_rate, etc.), maximising a chosen
            scorer (MCC or macro F1) computed on the evaluation split, using
            fixed class weights and per-class thresholds.
    4. Aggregates per-fold best hyperparameters across folds (mode for integer
       / categorical params, mean for continuous params) to obtain a single
       configuration per n_feats.
    5. Writes out:
         - per-fold best hyperparameters (joblib),
         - aggregated best hyperparameters per n_feats (joblib),
         - a tuning log with runtime, arguments, and library versions (CSV).

This script does *not* report final model performance; it is intended as an
internal hyperparameter refinement step. Final evaluation and uncertainty
estimation are handled in the Stage 2 model-evaluation script.

Command-line interface
----------------------
--subreddit   : Subreddit key in {'crypto','politics','conspiracy'}.
--outdir      : Output directory for tuning artefacts.
--train_X     : Path to training feature matrix (parquet).
--train_y     : Path to training target DataFrame (parquet).
--y-col       : Name of the continuous target column (default 'log_thread_size').
--classes     : Number of ordinal classes (3 or 4 currently supported).
--scorer      : 'MCC' or 'F1'/'F1-score' (default 'MCC').
--rs          : Random seed (default 42).
--trials      : Number of Optuna trials (default 300, or 10 in debug mode).
--splits      : Number of StratifiedKFold splits (default 5, or 2 in debug).
--params      : Path to precomputed Stage 2 tuning params (joblib).
--debug       : If set, run in lightweight debug mode.
--no-cal / -nc: If set, disables probability calibration.
--cal         : "sigmoid" or "isotonic" (default "sigmoid").

Reproducibility
---------------
All stochastic components (Optuna sampler, CV splits, LightGBM) are seeded by
`--rs`. Library versions and all CLI arguments are logged in the tuning log.

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
)

from sklearn.model_selection import StratifiedKFold, train_test_split

import optuna


LABEL_LOOKUP = {
    "crypto": "r/CryptoCurrency",
    "politics": "r/politics",
    "conspiracy": "r/Conspiracy",
}


CLASS_NAMES = {
    3: ["Stalled", "Small", "Large"],
    4: ["Stalled", "Small", "Medium", "Large"],
}

CAL_METHODS = ["sigmoid", "isotonic"]

SCORERS = {
    "MCC": matthews_corrcoef,
    "F1": partial(f1_score, average="macro"),
}

SCORER_MAP = {
    "mcc": "MCC",
    "f1": "F1",
    "f1-score": "F1",
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

    ap.add_argument("--params", help="Tuned model params file (jl).")

    ap.add_argument(
        "--cal",
        default="sigmoid",
        help="Calibration method (sigmoid or isotonic). Defaults to sigmoid.",
    )

    args = ap.parse_args()
    args.rs = int(args.rs)
    args.classes = int(args.classes)

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

    args.cal = str(args.cal).lower()
    if args.cal not in CAL_METHODS:
        raise ValueError(
            f"[ERROR] {args.cal} not a valid calibration method. Choose from {CAL_METHODS}"
        )

    if args.splits is None:
        args.splits = 5 if not debug else 2
    else:
        args.splits = int(args.splits)

    if args.trials is None:
        args.trials = 300 if not debug else 10
    else:
        args.trials = int(args.trials)

    if not os.path.isfile(args.params):
        raise FileNotFoundError(f"[ERROR] Model params file not found: {args.params}")

    print(f"[INFO] Args: {args}")

    os.makedirs(args.outdir, exist_ok=True)

    plot_outdir = f"{args.outdir}/plots"
    os.makedirs(plot_outdir, exist_ok=True)

    print(f"[INFO] Loading training data.")
    X = pd.read_parquet(args.train_X)
    y = pd.read_parquet(args.train_y)[args.y_col]

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

    first_bins = params[feature_counts[0]]["bins"]
    y_bins = pd.cut(y, bins=first_bins, labels=False, include_lowest=True)
    foldwise_best_params = {}
    foldwise_best_scores = {}
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y_bins)):
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
            config["final_n_features"] = n_feats
            bins = config["bins"]

            # Sanity checks: bins and threshold length must match args.classes
            if len(bins) - 1 != args.classes:
                raise ValueError(
                    f"[ERROR] For n_feats={n_feats}, len(bins)-1={len(bins)-1} "
                    f"but args.classes={args.classes}"
                )
            if len(thresh) != args.classes:
                raise ValueError(
                    f"[ERROR] For n_feats={n_feats}, len(final_threshold)={len(thresh)} "
                    f"but args.classes={args.classes}"
                )

            print(
                f"[INFO][{fold + 1}/{args.splits}][{n_feats} feats] Selected features: {x_cols}"
            )
            print(
                f"[INFO][{fold + 1}/{args.splits}][{n_feats} feats] Class weights: {cw}, Threshold: {thresh}"
            )

            # Load training data
            X_tr, X_val = X[x_cols].iloc[train_idx], X[x_cols].iloc[val_idx]
            y_tr, y_val = y_bins.iloc[train_idx], y_bins.iloc[val_idx]

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
                    "objective": "multiclass",
                    "num_class": args.classes,
                    "metric": "multi_logloss",
                    "class_weight": cw,
                }

                param.update(
                    {
                        "boosting_type": "gbdt",
                        "random_state": args.rs,
                        "verbosity": -1,
                        "num_leaves": trial.suggest_int(
                            "num_leaves", 20, 25 if debug else 150
                        ),
                        "max_depth": trial.suggest_int(
                            "max_depth", 3, 6 if debug else 15
                        ),
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
                )
                clf = lgb.LGBMClassifier(**param)
                clf.fit(X_tr, y_tr)
                if calibrate:
                    calibrated_clf = CalibratedClassifierCV(
                        clf, method=args.cal, cv="prefit"
                    )
                    calibrated_clf.fit(X_calib, y_calib)
                    proba = calibrated_clf.predict_proba(X_eval)
                else:
                    proba = clf.predict_proba(X_eval)
                preds = []
                for row in proba:
                    passed = [
                        row[i] if row[i] >= thresh[i] else -1
                        for i in range(args.classes)
                    ]
                    preds.append(
                        np.argmax(passed) if max(passed) != -1 else np.argmax(row)
                    )

                score = SCORERS[args.scorer](y_eval, preds)
                return score

            print(
                f"[INFO][{fold + 1}/{args.splits}][{n_feats} feats] Starting hyperparameter tuning"
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

    rows = []
    for n_feats in feature_counts:
        for fold_idx, score in enumerate(foldwise_best_scores[n_feats], start=1):
            rows.append(
                {"n_feats": n_feats, "fold": fold_idx, args.scorer: score,}
            )

    fold_scores_df = pd.DataFrame(rows)
    fold_scores_df.to_csv(f"{args.outdir}/foldwise_best_scores.csv", index=False)

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
