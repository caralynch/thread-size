# MIT License
# Copyright (c) 2025 Cara Lynch
# See the LICENSE file for details.
"""
Stage 2.4 – final evaluation of tuned thread-size classifier.

This script evaluates multiclass LightGBM models for predicting discretised
thread size (e.g. stalled, small, medium, large) using tuned hyperparameters,
class weights and bin definitions from the Stage 2 tuning pipeline. For each
feature subset size (n_feats), it:

    * Loads training and test feature matrices and continuous log thread size
      targets, then discretises them into K ordinal classes using fixed bin
      edges supplied in the params file.

    * Runs an outer StratifiedKFold cross-validation loop on the training data
      to obtain out-of-fold (OOF) predicted probabilities. Within each fold:
        - Splits the fold training set into model-training and, optionally,
          a separate calibration subset for probability calibration.
        - Fits a multiclass LightGBM classifier with fixed class weights and
          tuned tree hyperparameters.
        - Optionally calibrates the model probabilities using sigmoid or
          isotonic regression (CalibratedClassifierCV).
        - Stores predicted class probabilities on the held-out fold.

    * Derives OOF hard class predictions by taking the argmax over the (optionally
      calibrated) OOF class probabilities, and similarly obtains hard predictions
      for the test set from a final model trained on the full training data
      (with optional recalibration).

    * Computes a rich set of performance metrics on both OOF and test predictions,
      including:
        - global metrics (MCC, macro F1, balanced accuracy, macro precision,
          macro recall),
        - per-class precision and recall,
        - bootstrap 95% confidence intervals for all metrics, and
        - bootstrap summary statistics for each cell of the confusion matrix.

    * Produces per-model artefacts:
        - joblib dumps of the fitted model and optional calibrated wrapper,
        - parquet files with X/y splits and predictions (test and OOF),
        - SHAP values and explainer objects, plus SHAP summary plots,
        - confusion-matrix plots and bootstrap-based summaries, and
        - Excel workbooks with run info, model parameters, hyperparameters,
          classification reports, performance metrics, class sizes, SHAP
          summaries, and (if applicable) words associated with SVD components.

    * Aggregates results across feature subset sizes into:
        - combined_scores.jl (metrics vs n_feats with CIs for test and OOF),
        - metric-vs-n_feats plots, and
        - an evaluation.xlsx with combined tables.

Inputs
------
- --train_X / --test_X : Stage 2 feature matrices (parquet).
- --train_y / --test_y : DataFrames containing the continuous target
                         (log_thread_size).
- --params             : params_post_hyperparam_tuning.jl from Stage 2.3,
                         containing, for each n_feats:
                             * "features"
                             * "bins"
                             * "final_class_weights"
                             * "best_hyperparams"
- --classes            : Number of ordinal classes (e.g. 3 or 4).
- --scorer             : Primary evaluation metric (e.g. MCC).
- --rs                 : Random seed for CV, bootstrapping, and LightGBM.

Outputs
-------
Per n_feats (written under {outdir}/model_{n_feats}/):

    - test_data_results.xlsx
        Per-model workbook with:
            * model_info
            * class sizes and binning
            * tuned hyperparameters and class weights
            * classification reports
            * OOF and test metrics
            * SHAP summaries and (if applicable) SVD word tables.

    - model_data/*.parquet and *.jl
        Parquet and joblib files containing:
            * X_train, X_test, y_train, y_test
            * OOF and test predictions and probabilities
            * fitted model and, if used, calibrated wrapper
            * SHAP explainer and SHAP values
            * confusion matrices and bootstrap summaries.

Run-level (across n_feats, written to --outdir):

    - combined_scores.jl
        Joblib dict with OOF/test metrics and their CIs vs n_feats.

    - evaluation.xlsx
        Excel workbook with combined summary tables for all n_feats:
            * global metrics
            * per-class metrics
            * model_info.

Reproducibility
---------------
All stochastic components (StratifiedKFold splits, LightGBM fitting,
bootstrap resampling) are controlled by the --rs random seed. Library
versions and all CLI arguments are recorded in the output metadata.
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
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    balanced_accuracy_score,
    precision_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import shap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

LABEL_LOOKUP = {
    "crypto": "r/CryptoCurrency",
    "politics": "r/politics",
    "conspiracy": "r/Conspiracy",
}

CLASS_NAMES = {
    3: ["Stalled", "Small", "Large"],
    4: ["Stalled", "Small", "Medium", "Large"],
}


SCORERS = {
    "MCC": matthews_corrcoef,
    "F1": partial(f1_score, average="macro"),
    "Balanced accuracy": balanced_accuracy_score,
    "Precision": partial(precision_score, zero_division=0, average="macro"),
    "Recall": partial(recall_score, zero_division=0, average="macro"),
}

SCORER_MAP = {
    "mcc": "MCC",
    "f1": "F1",
    "f1-score": "F1",
    "balanced": "Balanced accuracy",
    "balanced_accuracy": "Balanced accuracy",
}


REPORTS = {
    "Report": partial(classification_report, output_dict=True, zero_division=0),
    "CM": confusion_matrix,
}

CAL_METHODS = ["sigmoid", "isotonic"]


def class_precision(y_true, y_pred, class_idx):
    """
    Compute precision for a single target class in a multiclass setting.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    class_idx : int
        Index of the class for which precision is computed.

    Returns
    -------
    float
        Precision for the specified class. If the class is absent in the
        predictions, returns 0 (zero_division=0).
    """
    return precision_score(
        y_true, y_pred, zero_division=0, labels=[class_idx], average=None
    )[0]


def class_recall(y_true, y_pred, class_idx):
    """
    Compute recall for a single target class in a multiclass setting.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    class_idx : int
        Index of the class for which recall is computed.

    Returns
    -------
    float
        Recall for the specified class. If the class is absent in the
        true labels, returns 0 (zero_division=0).
    """
    return recall_score(
        y_true, y_pred, zero_division=0, labels=[class_idx], average=None
    )[0]


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
    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]  # exclude NaNs
    return np.percentile(arr, [2.5, 97.5])


def main():
    """
    Main entry point for Stage 2 final model evaluation.
    
    Trains final multiclass classifiers with tuned hyperparameters, generates
    confusion matrices, SHAP analyses, and publication-ready outputs.
    """
    print(f"{sys.argv[0]}")
    start = dt.datetime.now()
    print("[INFO] Running tuned model(s) and generating evaluation outputs.")
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

    ap.add_argument("--classes", default="4", help="Number of classes.")

    ap.add_argument(
        "--test_X", help="Test X data filepath (parquet).",
    )

    ap.add_argument(
        "--train_y", help="Training y data filepath (parquet).",
    )

    ap.add_argument(
        "--test_y", help="Test y data filepath (parquet).",
    )

    ap.add_argument("--params", help="Tuned model params file (jl).")

    ap.add_argument("--balanced", action="store_true", help="Run the models for balanced class weights.")

    ap.add_argument("--tfidf", help="TF-IDF model.")

    ap.add_argument("--svd", help="SVD model.")

    ap.add_argument(
        "--y-col",
        default="log_thread_size",
        help="Target y column. Defaults to log_thread_size.",
    )

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
        "--scorer",
        default="MCC",
        help="Scorer to tune to (F-beta, MCC or F1-score. Defaults to MCC)",
    )

    ap.add_argument(
        "--splits",
        default=None,
        help="Number of CV splits. Defaults to 5, or 2 in debug mode.",
    )
    ap.add_argument(
        "--n-bs",
        default=None,
        help="Number of bootstrap trials. Defaults to 1000, or 20 in debug mode.",
    )

    args = ap.parse_args()
    args.rs = int(args.rs)
    args.classes = int(args.classes)

    if args.classes not in CLASS_NAMES:
        raise ValueError(
            f"[ERROR] {args.classes} not a valid number of classes. Choose from {list(CLASS_NAMES.keys())}"
        )
    for i, class_name in enumerate(CLASS_NAMES[args.classes]):
        SCORERS[f"{class_name} precision"] = partial(class_precision, class_idx=i)
        SCORERS[f"{class_name} recall"] = partial(class_recall, class_idx=i)

    args.cal = str(args.cal).lower()
    if args.cal not in CAL_METHODS:
        raise ValueError(
            f"[ERROR] {args.cal} not a valid calibration method. Choose from {CAL_METHODS}"
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

    if args.n_bs is None:
        args.n_bs = 1000 if not debug else 20
    else:
        args.n_bs = int(args.n_bs)

    balanced = False
    if args.balanced:
        class_weights = "balanced"
        balanced = True
    if not os.path.isfile(args.params):
        raise FileNotFoundError(f"[ERROR] Model params file not found: {args.params}")

    print(f"[INFO] Args: {args}")

    os.makedirs(args.outdir, exist_ok=True)
    plot_outdir = f"{args.outdir}/plots"
    os.makedirs(plot_outdir, exist_ok=True)

    print(f"[INFO] Loading X train and test data.")
    X_train = pd.read_parquet(args.train_X)
    X_test = pd.read_parquet(args.test_X)
    print(f"[INFO] Loading y train and test data.")
    y_train_data = pd.read_parquet(args.train_y)[args.y_col]
    y_test_data = pd.read_parquet(args.test_y)[args.y_col]

    if not balanced:
        print(f"[INFO] Loading tuned model params from {args.params}.")
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
    model_info["shap_version"] = shap.__version__

    print("[INFO] Getting feature counts from params...")
    feature_counts = list(params.keys())
    model_info["feature_counts"] = feature_counts
    print(f"[OK] Feature counts: {feature_counts}")

    svd_cols_present = False
    for n_feats, config in params.items():
        x_cols = config["features"]
        svd_cols = [col for col in x_cols if col.startswith("svd")]
        if len(svd_cols) > 0:
            svd_cols_present = True
            break

    if svd_cols_present:
        print("[OK] SVD cols present")
        # load TF-IDF and svd model
        print(f"[INFO] Loading TF-IDF and SVD vectorizers")
        tfidf_vectorizer = joblib.load(f"{args.tfidf}")
        svd_model = joblib.load(f"{args.svd}")
        feature_names = tfidf_vectorizer.get_feature_names_out()
        svd_components = svd_model.components_

        def get_svd_words(svd_cols, top_n=20):
            svd_words = {}
            for svd_col in svd_cols:
                idx = int(svd_col.replace("svd_", ""))
                vector = svd_components[idx]
                top = np.argsort(vector)[-top_n:]
                bottom = np.argsort(vector)[:top_n]
                top_words = [(feature_names[i], vector[i]) for i in reversed(top)]
                bottom_words = [(feature_names[i], vector[i]) for i in bottom]
                svd_words[svd_col] = pd.DataFrame(
                    top_words + bottom_words, columns=["word", "value"]
                )
            return svd_words

    combined_summary = {
        "model_params": [],
        "hyperparams": [],
        "report": [],
        "class_sizes": [],
    }
    combined_scores = {
        "test": {},
        "oof": {},
    }
    for data_type, scores_dict in combined_scores.items():
        scores_dict["n_feats"] = []
        for k in SCORERS:
            scores_dict[k] = []
            scores_dict[f"{k} CI"] = []

    y_pred_dict = {}
    print(f"[INFO] Setting up outer cross-validation.")
    outer_cv = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.rs)

    print("[INFO] Starting loop through n_feats")
    for n_feats, config in params.items():
        model_outdir = f"{args.outdir}/model_{n_feats}"
        os.makedirs(model_outdir, exist_ok=True)
        model_plot_outdir = f"{model_outdir}/plots"
        os.makedirs(model_plot_outdir, exist_ok=True)
        model_data_outdir = f"{model_outdir}/model_data"
        os.makedirs(model_data_outdir, exist_ok=True)

        current_scores = {}

        x_cols = config["features"]
        cw = config["final_class_weights"] if not balanced else class_weights
        if y_test_data.max() > y_train_data.max():
            # have final bin edge be adjusted if test data has larger bin size
            config["bins"][-1] = y_test_data.max() + 1e-3
        config["final_n_features"] = n_feats
        bins = config["bins"]
        hyperparams = config["best_hyperparams"] if "best_hyperparams" in config else {}

        print(f"[INFO][{n_feats} feats] Selected features: {x_cols}")
        print(f"[INFO][{n_feats} feats] Class weights: {cw}")  # , Threshold: {thresh}")

        assert (
            len(bins) == args.classes + 1
        ), f"[ERROR][{n_feats} feats] bin_edges must have length args.classes+1 ({args.classes+1}), got {len(bins)}."
        print(f"[INFO][{n_feats} feats] Getting started train and test data")

        X_tr = X_train[x_cols]
        X_te = X_test[x_cols]

        y_tr = pd.cut(y_train_data, bins=bins, labels=False, include_lowest=True)
        y_te = pd.cut(y_test_data, bins=bins, labels=False, include_lowest=True)

        unique_bins = np.unique(y_tr.dropna())
        assert len(unique_bins) == args.classes, (
            f"[ERROR][{n_feats} feats] pd.cut produced {len(unique_bins)} classes, expected {args.classes}. "
            f"Unique bins found: {unique_bins}"
        )
        y_tr_bin_counts = y_tr.value_counts(sort=False).sort_index()
        y_tr_bin_counts.index.name = "Class"
        y_te_bin_counts = y_te.value_counts(sort=False).sort_index()
        y_te_bin_counts.index.name = "Class"
        thread_size_bins = [round(np.exp(x)) for x in bins]
        bin_ranges = [[thread_size_bins[0], thread_size_bins[0]]]
        for i in range(1, len(thread_size_bins) - 1):
            to_append = [bin_ranges[i - 1][1] + 1, thread_size_bins[i + 1]]
            bin_ranges.append(to_append)
        bin_count_df = pd.DataFrame(
            {"Range": bin_ranges, "Train": y_tr_bin_counts, "Test": y_te_bin_counts}
        )

        print(f"[INFO][{n_feats} feats] Getting OOF predictions")
        fold_idx = 1

        oof_probas = np.zeros((len(X_tr), args.classes))
        for train_idx, val_idx in outer_cv.split(X_tr, y_tr):
            print(f"[INFO][{n_feats} feats] Fold {fold_idx}/{args.splits}")
            X_fold_tr, X_fold_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            y_fold_tr, y_fold_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]

            # Optionally split further for probability calibration
            if calibrate:
                X_fold_tr, X_calib, y_fold_tr, y_calib = train_test_split(
                    X_fold_tr,
                    y_fold_tr,
                    train_size=0.9,
                    stratify=y_fold_tr,
                    random_state=args.rs,
                )

            print(f"[INFO][{n_feats} feats][{fold_idx}/{args.splits}] Training model.")

            model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=args.classes,
                class_weight=cw,
                random_state=args.rs,
                verbosity=-1,
                **hyperparams,
            )
            model.fit(X_fold_tr, y_fold_tr)

            if calibrate:
                calibrated_clf = CalibratedClassifierCV(
                    model, method=args.cal, cv="prefit",
                )
                calibrated_clf.fit(X_calib, y_calib)
                proba = calibrated_clf.predict_proba(X_fold_val)
            else:
                proba = model.predict_proba(X_fold_val)

            oof_probas[val_idx] = proba
            print(
                f"[INFO][{n_feats} feats][{fold_idx}/{args.splits}] "
                "Optimising per-class thresholds with L-BFGS-B"
            )
            fold_idx += 1


        oof_preds = np.argmax(oof_probas, axis=1)


        print(f"[INFO][{n_feats} feats] Creating and fitting model")
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=args.classes,
            class_weight=cw,
            random_state=args.rs,
            verbosity=-1,
            **hyperparams,
        )
        model.fit(X_tr, y_tr)

        if calibrate:
            print(f"[INFO][{n_feats} feats] Calibrating classifier")
            calibrated = CalibratedClassifierCV(
                model,
                method=args.cal,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=args.rs),
            )
            calibrated.fit(X_tr, y_tr)

            print(f"[INFO][{n_feats} feats] Getting test set predicted probabilities")
            y_proba = calibrated.predict_proba(X_te)
            y_pred = calibrated.predict(X_te)
        else:
            print(f"[INFO][{n_feats} feats] Getting test set predicted probabilities")
            y_proba = model.predict_proba(X_te)
            y_pred = model.predict(X_te)

        y_pred_dict[n_feats] = {"test": y_pred, "oof": oof_preds}

        # Save core model data as parquet
        X_tr.to_parquet(f"{model_data_outdir}/X_train.parquet")
        X_te.to_parquet(f"{model_data_outdir}/X_test.parquet")
        y_tr.to_frame(name="y_train").to_parquet(f"{model_data_outdir}/y_train.parquet")
        y_te.to_frame(name="y_test").to_parquet(f"{model_data_outdir}/y_test.parquet")

        # Predicted labels (test and OOF)
        pd.DataFrame({"y_pred": y_pred}, index=X_te.index).to_parquet(
            f"{model_data_outdir}/y_pred.parquet"
        )

        pd.DataFrame({"oof_preds": oof_preds}, index=X_tr.index).to_parquet(
            f"{model_data_outdir}/oof_pred.parquet"
        )

        # Predicted probabilities (test and OOF)
        y_proba_df = pd.DataFrame(
            y_proba,
            index=X_te.index,
            columns=[f"proba_class_{i}" for i in range(args.classes)],
        )
        y_proba_df.to_parquet(f"{model_data_outdir}/y_proba.parquet")

        oof_proba_df = pd.DataFrame(
            oof_probas,
            index=X_tr.index,
            columns=[f"proba_class_{i}" for i in range(args.classes)],
        )
        oof_proba_df.to_parquet(f"{model_data_outdir}/oof_proba.parquet")

        joblib.dump(model, f"{model_data_outdir}/model.jl")
        if calibrate:
            joblib.dump(calibrated, f"{model_data_outdir}/calibrated_model.jl")

        to_measure = {"test": (y_te, y_pred), "oof": (y_tr, oof_preds)}
        print(
            f"[INFO][{n_feats} feats] Calculating classification report and performance scoring metrics"
        )

        for key, (true_y, preds) in to_measure.items():

            print(
                f"[INFO][{n_feats} feats] Getting report metrics for {key} predictions"
            )
            # Report metrics
            performance_metrics = {}
            report_metrics = {}
            for k, score_f in SCORERS.items():
                performance_metrics[k] = score_f(true_y, preds)
            for k, report_f in REPORTS.items():
                report_metrics[k] = report_f(true_y, preds)

            print(
                f"[INFO][{n_feats} feats] Bootstrapping metrics for {key} predictions"
            )
            # Bootstrapping main metrics
            rng = np.random.RandomState(args.rs)  # for reproducibility

            bootstrap_metrics = {}
            for k in performance_metrics:
                bootstrap_metrics[k] = []

            bootstrap_cms = []

            for i in range(args.n_bs):
                indices = rng.choice(len(true_y), size=len(true_y), replace=True)
                y_true_bs = np.array(true_y)[indices]
                y_pred_bs = np.array(preds)[indices]
                for k, score_f in SCORERS.items():
                    bootstrap_metrics[k].append(score_f(y_true_bs, y_pred_bs))
                bootstrap_cms.append(
                    confusion_matrix(
                        y_true_bs, y_pred_bs, labels=list(range(0, args.classes))
                    )
                )

            metric_cis = {}
            for k, vals in bootstrap_metrics.items():
                metric_cis[k] = ci(vals).tolist()
            current_scores[key] = {
                "n_feats": n_feats,
            }

            cm_cis = {}
            cm_array = np.stack(bootstrap_cms)
            for lab in ["lower", "upper", "mean", "std"]:
                cm_cis[lab] = np.zeros(np.shape(bootstrap_cms[0]))

            for i in range(args.classes):
                for j in range(args.classes):
                    values = cm_array[:, i, j]
                    cm_cis["mean"][i, j] = np.mean(values)
                    cm_cis["lower"][i, j] = np.percentile(values, 2.5)
                    cm_cis["upper"][i, j] = np.percentile(values, 97.5)
                    cm_cis["std"][i, j] = np.std(values, ddof=1)

            combined_scores[key]["n_feats"].append(n_feats)
            for k, m in performance_metrics.items():
                combined_scores[key][k].append(m)
                combined_scores[key][f"{k} CI"].append(metric_cis[k])
                current_scores[key][k] = m
                current_scores[key][f"{k} CI"] = metric_cis[k]

            confusion_matrix_data = {
                "CM": report_metrics["CM"],
            }
            confusion_matrix_data.update(cm_cis)

            print(f"[INFO][{n_feats} feats] Creating plots for {key} predictions")
            joblib.dump(
                confusion_matrix_data,
                f"{model_data_outdir}/{key}_confusion_matrix_data.jl",
            )
            plt.figure()
            sns.heatmap(
                report_metrics["CM"],
                annot=True,
                fmt="d",
                cmap="YlGnBu",
                xticklabels=CLASS_NAMES[args.classes],
                yticklabels=CLASS_NAMES[args.classes],
            )
            plt.xlabel("Predicted Class")
            plt.ylabel("True Class")
            plt.title(
                f"{LABEL_LOOKUP[args.subreddit]} Thread Size {key} {n_feats} feats Confusion Matrix"
            )
            plt.tight_layout()
            plt.savefig(
                f"{plot_outdir}/{key}_{n_feats}_feats_confusion_matrix.png", dpi=300,
            )
            plt.close()

        # SHAP
        print(f"[INFO][{n_feats} feats] Getting SHAP values")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_te)
        joblib.dump(shap_values, f"{model_data_outdir}/shap_values.jl")
        joblib.dump(explainer, f"{model_data_outdir}/shap_explainer.jl")
        if isinstance(shap_values, np.ndarray):
            if shap_values.shape[2] == args.classes:  # (samples, features, classes)
                # Transpose to (classes, samples, features)
                shap_values = [
                    shap_values[:, :, i] for i in range(shap_values.shape[2])
                ]
        for class_idx, label in enumerate(CLASS_NAMES[args.classes]):

            plt.figure()
            shap.summary_plot(shap_values[class_idx], X_te, show=False)
            plt.title(f"{LABEL_LOOKUP[args.subreddit]} SHAP - {label}")
            plt.tight_layout()
            plt.savefig(f"{plot_outdir}/{n_feats}_feats_shap_{label}.png", dpi=300)
            plt.close()

        # Predictions
        pred_dict = {
            "test": {
                "index": X_te.index,
                "true_class": y_te,
                "predicted_class": y_pred,
            },
            "oof": {"index": X_tr.index, "oof_y_true": y_tr, "oof_y_pred": oof_preds},
        }
        for i in range(args.classes):
            pred_dict["test"][f"proba_class_{i}"] = y_proba[:, i]

        for key, df in pred_dict.items():
            pd.DataFrame(df).to_csv(f"{model_outdir}/{key}_preds.csv", index=False)

        # SVD words
        svd_cols = [col for col in x_cols if col.startswith("svd_")]
        svd_words = get_svd_words(svd_cols) if svd_cols else {}

        # Save to Excel
        report_df = (
            pd.DataFrame(report_metrics["Report"])
            .transpose()
            .reset_index(names=["Class"])
        )
        # Average absolute SHAP values across all classes
        # Stack into a 3D array: (args.classes, n_samples, n_features)
        if isinstance(shap_values, list):
            shap_array = np.stack([np.abs(sv) for sv in shap_values], axis=0)
            shap_mean = shap_array.mean(axis=(0, 1))
        else:
            shap_mean = np.abs(shap_values).mean(axis=0)

        shap_mean_df = pd.DataFrame({"feature": X_te.columns, "MeanAbsSHAP": shap_mean})
        shap_mean_df.sort_values("MeanAbsSHAP", ascending=False, inplace=True)
        current_scores_df = pd.DataFrame.from_dict(current_scores)
        excel_path = f"{model_outdir}/test_data_results.xlsx"
        print(f"[INFO][{n_feats} feats] Saving results to {excel_path}")
        params_df = pd.DataFrame(
            list(config.items()), columns=["Key", "Value"]
        ).reset_index(names=["Parameter"])
        hyperparams_df = pd.DataFrame.from_dict(
            hyperparams, orient="index", columns=["Value"]
        ).reset_index(names=["Hyperparameter"])
        hyperparams_df["n_feats"] = n_feats
        params_df["n_feats"] = n_feats
        report_df["n_feats"] = n_feats
        combined_summary["hyperparams"].append(hyperparams_df)
        combined_summary["model_params"].append(params_df)
        combined_summary["report"].append(report_df)
        combined_summary["class_sizes"].append(bin_count_df)
        with pd.ExcelWriter(excel_path) as writer:
            pd.DataFrame.from_dict(model_info, orient="index").to_excel(
                writer, sheet_name="info",
            )
            bin_count_df.to_excel(writer, sheet_name="class_sizes")
            params_df.to_excel(writer, sheet_name="model_params", index=False)
            hyperparams_df.to_excel(writer, sheet_name="hyperparams", index=False)
            report_df.to_excel(writer, sheet_name="classification_report", index=False)
            current_scores_df.to_excel(writer, sheet_name="performance")
            shap_mean_df.to_excel(writer, sheet_name="shap_summary", index=False)
            for svd_name, df in svd_words.items():
                df.to_excel(writer, sheet_name=f"{svd_name}_words", index=False)

    joblib.dump(combined_scores, f"{args.outdir}/combined_scores.jl")
    for metric in SCORERS:
        for key, score_dict in combined_scores.items():
            plt.figure(figsize=(8, 5))
            # Optional: plot 95% CI shaded area
            lower = [
                metric_value - score_dict[f"{metric} CI"][i][0]
                for i, metric_value in enumerate(score_dict[metric])
            ]
            upper = [
                score_dict[f"{metric} CI"][i][1] - metric_value
                for i, metric_value in enumerate(score_dict[metric])
            ]
            plt.errorbar(
                score_dict["n_feats"],
                score_dict[metric],
                yerr=[lower, upper],
                fmt="o",
                capsize=3,
                color="black",
                ecolor="gray",
            )

            plt.title(
                f"{LABEL_LOOKUP[args.subreddit]} - {metric.upper()} vs number of features"
            )
            plt.xlabel("Number of features")
            plt.ylabel(metric.upper())
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            plt.savefig(f"{plot_outdir}/{key}_{metric}_vs_n_feats.png", dpi=300)
            plt.close()

    output_dfs = {
        "info": pd.DataFrame.from_dict(model_info, orient="index").reset_index(
            names=["Parameter"]
        ),
    }
    for key, df_list in combined_summary.items():
        output_dfs[key] = pd.concat(df_list, ignore_index=True)

    excel_path = f"{args.outdir}/evaluation.xlsx"
    print(f"[INFO] Saving combined evaluation to {excel_path}")
    with pd.ExcelWriter(excel_path) as writer:
        for key, df in output_dfs.items():
            df.to_excel(writer, sheet_name=key, index=False)
        for key, out_dict in combined_scores.items():
            pd.DataFrame.from_dict(out_dict).to_excel(
                writer, sheet_name=f"{key}_metrics"
            )

    end = dt.datetime.now()
    print(f"[OK] Saved all outputs to: {args.outdir}")
    print(f"[OK] Finished. Total runtime {end-start}.")


if __name__ == "__main__":
    main()
