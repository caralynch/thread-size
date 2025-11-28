"""
4_run_tuned_model.py

Stage 1 – final evaluation of tuned thread-start classifier.

This script:

    * Loads tuned LightGBM hyperparameters and class weights per feature subset
      (n_feats) from the Stage 1 tuning pipeline.
    * Loads train/test feature matrices (X_train, X_test) and log thread size
      targets, then binarises them to "Stalled" vs "Started" based on a log-size
      threshold.
    * For each n_feats:
        - Trains a model under StratifiedKFold CV to obtain out-of-fold (OOF)
          predicted probabilities.
        - Within each fold, splits off a holdout subset to optimise a single
          decision threshold on the chosen scorer (MCC, F-beta, etc.).
        - Optionally calibrates probabilities via isotonic regression.
        - Trains a final classifier on the full training set and evaluates on
          the held-out test set.
        - Computes ROC and precision–recall curves, bootstrap confidence
          intervals for key metrics, confusion matrices, calibration curves,
          SVD component word lists (if present), and SHAP-based feature
          importance summaries.
    * Aggregates results across n_feats into combined score tables and plots
      (metrics vs number of features) and writes:
        - per-model Excel workbooks (config, metrics, hyperparams, SHAP),
        - a run-level Excel summary,
        - joblib artefacts with combined scores, OOF/test probabilities, and
          SHAP data.

This script is intended as the final evaluation and figure-artefact generator
for Stage 1. Higher-level “paper-ready” plots and tables that span multiple
subreddits are produced by separate make_outputs scripts.

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
import shap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
    precision_score,
    recall_score,
    balanced_accuracy_score,
    fbeta_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


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
    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]  # exclude NaNs
    return np.percentile(arr, [2.5, 97.5])


def main():
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

    ap.add_argument("--tfidf", help="TF-IDF model.")

    ap.add_argument("--svd", help="SVD model.")

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

    def neg_score(threshold, y_proba, y_true):
        y_pred = (y_proba >= threshold).astype(int)
        return -SCORERS[args.scorer](y_true, y_pred)

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

    if args.y_thresh is None:
        args.y_thresh = np.log(1)
    else:
        args.y_thresh = float(args.y_thresh)

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
    y_train = (y_train_data > args.y_thresh).astype(int)
    y_test = (y_test_data > args.y_thresh).astype(int)

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

    print("[INFO] Checking if SVD cols in chosen features...")

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

    combined_summary = {}
    y_probas = {}
    y_preds = {}
    oof_proba_list = {}
    oof_pred_list = {}

    combined_scores = {
        "test": {},
        "oof": {},
    }

    for data_type, scores_dict in combined_scores.items():
        scores_dict["n_feats"] = []
        scores_dict["AUC"] = []
        scores_dict["AUC CI"] = []
        for k in [i for i in SCORERS.keys() if i not in EXCLUDE_SCORES]:
            scores_dict[k] = []
            scores_dict[f"{k} CI"] = []
    print(f"[INFO] Setting up outer cross-validation.")
    outer_cv = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.rs)
    for n_feats, config in params.items():
        model_outdir = f"{args.outdir}/model_{n_feats}"
        os.makedirs(model_outdir, exist_ok=True)
        model_plot_outdir = f"{model_outdir}/plots"
        os.makedirs(model_plot_outdir, exist_ok=True)
        model_data_outdir = f"{model_outdir}/model_data"
        print(f"[INFO] [{n_feats} feats]")
        x_cols = config["features"]
        cw = config["final_class_weights"]
        # thresh = config["final_threshold"]
        best_hyperparams = (
            config.get("best_hyperparams", {}) if "best_hyperparams" in config else {}
        )

        print(f"[INFO] [{n_feats} feats] Selected features: {x_cols}")
        print(
            f"[INFO] [{n_feats} feats] Class weights: {cw}"
        )  # , Threshold: {thresh}")

        # Start with fixed parameters
        fixed_params = {
            "objective": "binary",
            "class_weight": cw,
            "random_state": args.rs,
            "verbose": -1,
        }

        # Add in best hyperparameters, but let fixed_params override if there’s conflict
        combined_params = {**best_hyperparams, **fixed_params}

        oof_probas = np.zeros(len(X_train))
        thresholds = []

        print(
            f"[INFO] [{n_feats} feats] Creating {args.splits} CV loops for OOF started train"
        )
        i = 1
        for train_idx, val_idx in outer_cv.split(X_train, y_train):
            print(f"[INFO] [{n_feats} feats] Loop {i}/{args.splits}")
            X_tr, X_val = X_train.iloc[train_idx][x_cols], X_train.iloc[val_idx][x_cols]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            X_tr, X_thresh_calib, y_tr, y_thresh_calib = train_test_split(
                X_tr,
                y_tr,
                train_size=0.8,
                stratify=y_tr,  # preserve class balance
                random_state=args.rs,  # for reproducibility
            )

            if calibrate:
                X_tr, X_calib, y_tr, y_calib = train_test_split(
                    X_tr,
                    y_tr,
                    train_size=0.9,  # p0% for train, 10% for calib
                    stratify=y_tr,  # preserve class balance
                    random_state=args.rs,  # for reproducibility
                )

            print(f"[INFO][{n_feats} feats][{i}/{args.splits}] Training model.")
            clf = lgb.LGBMClassifier(**combined_params)
            clf.fit(X_tr, y_tr)
            if calibrate:
                print(f"[INFO][{n_feats} feats][{i}/{args.splits}] Calibrating model")
                calibrated_clf = CalibratedClassifierCV(
                    clf, method="isotonic", cv="prefit"
                )
                calibrated_clf.fit(X_calib, y_calib)
                proba = calibrated_clf.predict_proba(X_val)[:, 1]
                calib_proba = calibrated_clf.predict_proba(X_thresh_calib)[:, 1]
            else:
                proba = clf.predict_proba(X_val)[:, 1]
                calib_proba = clf.predict_proba(X_thresh_calib)[:, 1]

            oof_probas[val_idx] = proba
            print(
                f"[INFO][{n_feats} feats][{i}/{args.splits}] Using minimize_scalar to get threshold"
            )
            result = minimize_scalar(
                neg_score,
                bounds=(0, 1),
                method="bounded",
                args=(calib_proba, y_thresh_calib),
            )
            thresholds.append(result.x)

            i += 1

        print(f"[INFO][{n_feats} feats] Averaging thresholds over folds")
        thresh = np.mean(thresholds)
        config["model_threshold"] = thresh
        oof_preds = (oof_probas >= thresh).astype(int)

        print(f"[INFO] [{n_feats} feats] Training final classifier")
        final_clf = lgb.LGBMClassifier(**combined_params)
        final_clf.fit(X_train[x_cols], y_train)

        if calibrate:
            print(f"[INFO][{n_feats} feats] Calibrating final classifier")
            calibrated_final = CalibratedClassifierCV(
                final_clf,
                method="isotonic",
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=args.rs),
            )
            calibrated_final.fit(X_train[x_cols], y_train)

            test_proba = calibrated_final.predict_proba(X_test[x_cols])[:, 1]
        else:
            test_proba = final_clf.predict_proba(X_test[x_cols])[:, 1]
        test_pred = (test_proba > thresh).astype(int)

        y_probas[n_feats] = test_proba
        y_preds[n_feats] = test_pred
        oof_proba_list[n_feats] = oof_probas
        oof_pred_list[n_feats] = oof_preds

        to_measure = {
            "test": (y_test, test_pred, test_proba),
            "oof": (y_train, oof_preds, oof_probas),
        }

        current_scores = {}
        reports = {}
        cms = {}
        for key, (true_y, preds, probas) in to_measure.items():

            print(
                f"[INFO] [{n_feats} feats] Getting report metrics for {key} predictions"
            )
            # Report metrics
            performance_metrics = {"AUC": roc_auc_score(true_y, probas)}
            cms[key] = SCORERS["CM"](true_y, preds)
            reports[key] = SCORERS["Report"](true_y, preds)
            for k, score_f in [
                (k, v) for (k, v) in SCORERS.items() if k not in EXCLUDE_SCORES
            ]:
                performance_metrics[k] = score_f(true_y, preds)

            print(f"[INFO] [{n_feats} feats] Plotting ROC and precision-recall curves")
            # ROC curve
            auc = performance_metrics["AUC"]
            fpr, tpr, roc_thresholds = roc_curve(true_y, probas)
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(
                f"{LABEL_LOOKUP[args.subreddit]} Stage 1 {n_feats} feats {key} ROC Curve"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{model_plot_outdir}/roc_curve_{key}.png", dpi=300)
            plt.close()

            # Precision-recall curve
            precision, recall, pr_thresholds = precision_recall_curve(true_y, probas)
            f1 = performance_metrics["F1"]
            plt.figure()
            plt.plot(recall, precision, label=f"PR curve (F1 = {f1:.2f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(
                f"{LABEL_LOOKUP[args.subreddit]} Stage 1 {n_feats} feats Precision-Recall Curve"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f"{model_plot_outdir}/precision_recall_curve_{key}.png", dpi=300,
            )
            plt.close()

            print(f"[INFO] [{n_feats} feats] Bootstrapping metrics")
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
                y_proba_bs = np.array(probas)[indices]
                bootstrap_cms.append(
                    confusion_matrix(y_true_bs, y_pred_bs, labels=list(range(0, 2)))
                )

                for k in bootstrap_metrics:
                    if k == "AUC":
                        bootstrap_metrics[k].append(
                            roc_auc_score(y_true_bs, y_proba_bs)
                        )
                    else:
                        bootstrap_metrics[k].append(SCORERS[k](y_true_bs, y_pred_bs))

            metric_cis = {}
            for k, v in bootstrap_metrics.items():
                metric_cis[k] = ci(v).tolist()
            current_scores[key] = {"n_feats": n_feats}
            combined_scores[key]["n_feats"].append(n_feats)
            for k, v in performance_metrics.items():
                combined_scores[key][k].append(v)
                combined_scores[key][f"{k} CI"].append(metric_cis[k])
                current_scores[key][k] = v
                for i, bound in enumerate(["lower", "upper"]):
                    current_scores[key][f"{k} CI {bound}"] = metric_cis[k][i]

            cm_cis = {}
            cm_array = np.stack(bootstrap_cms)
            for lab in ["lower", "upper", "mean", "std"]:
                cm_cis[lab] = np.zeros(np.shape(bootstrap_cms[0]))

            for i in range(0, 2):
                for j in range(0, 2):
                    values = cm_array[:, i, j]
                    cm_cis["mean"][i, j] = np.mean(values)
                    cm_cis["lower"][i, j] = np.percentile(values, 2.5)
                    cm_cis["upper"][i, j] = np.percentile(values, 97.5)
                    cm_cis["std"][i, j] = np.std(values, ddof=1)

            cm_cis.update(
                {"CM": cms[key],}
            )
            joblib.dump(
                cm_cis, f"{model_data_outdir}/{key}_confusion_matrix_data.jl",
            )
            prob_true, prob_pred = calibration_curve(
                true_y, probas, n_bins=10, strategy="quantile"
            )
            joblib.dump(
                {"prob_true": prob_true, "prob_pred": prob_pred},
                f"{model_data_outdir}/{key}_calibration_curve_inputs.jl",
            )

            plt.figure()
            plt.plot(prob_pred, prob_true, marker="o", label="Model")
            plt.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                color="gray",
                label="Perfectly calibrated",
            )
            plt.xlabel("Mean predicted probability")
            plt.ylabel("Fraction of positives")
            plt.title(
                f"{LABEL_LOOKUP[args.subreddit]} Stage 1 {n_feats} feats Calibration Curve"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f"{model_plot_outdir}/calibration_curve.png", dpi=300,
            )
            plt.close()

            plt.figure(figsize=(6, 5))
            plt.title(
                f"{LABEL_LOOKUP[args.subreddit]} Thread start {key} {n_feats} Confusion Matrix"
            )
            sns.heatmap(
                cms[key],
                annot=True,
                fmt="d",
                cmap="YlGnBu",
                xticklabels=["Stalled", "Started"],
                yticklabels=["Stalled", "Started"],
            )
            plt.xlabel("Predicted Class")
            plt.ylabel("True Class")
            plt.tight_layout()
            plt.savefig(f"{model_plot_outdir}/{key}_confusion_matrix.png", dpi=300)
            plt.close()

            joblib.dump(cms[key], f"{model_data_outdir}/{key}_confusion_matrix.jl")

        print(f"[INFO] [{n_feats} feats] Getting starting thread predictions")
        # Save predicted started threads
        started_dict = {
            "train": pd.DataFrame(
                {"index": X_train.index, "proba": oof_probas, "predicted": oof_preds,}
            ),
            "test": pd.DataFrame(
                {"index": X_test.index, "proba": test_proba, "predicted": test_pred,}
            ),
        }
        print(f"[INFO] [{n_feats} feats] Saving results to {model_outdir}")
        for key, df in started_dict.items():
            df.to_parquet(f"{model_outdir}/{key}_started_threads.parquet")
        models_dict = {"classifier": final_clf}
        if calibrate:
            models_dict["calibrated_classifier"] = calibrated_final
        models_dict["X_test"] = X_test[x_cols]
        models_dict["y_test"] = y_test
        models_dict["X_train"] = X_train[x_cols]
        models_dict["y_train"] = y_train
        joblib.dump(models_dict, f"{model_data_outdir}/model.jl")
        final_clf.booster_.save_model(f"{model_data_outdir}/final_model.txt")

        report_dfs = {}
        for key, report in reports.items():
            report_dfs[key] = pd.DataFrame(report).transpose()

        # SVD cols
        svd_cols = [col for col in x_cols if col.startswith("svd_")]
        svd_words = {}
        for svd_col in svd_cols:
            idx = int(svd_col.replace("svd_", ""))
            vector = svd_components[idx]
            top_idx = np.argsort(vector)[-15:]
            bottom_idx = np.argsort(vector)[:15]
            top_words = [(feature_names[i], vector[i]) for i in reversed(top_idx)]
            bottom_words = [(feature_names[i], vector[i]) for i in bottom_idx]
            svd_words[svd_col] = pd.DataFrame(
                top_words + bottom_words, columns=["word", "value"]
            )

        print(f"[INFO] [{n_feats} feats] Getting SHAP values")
        # SHAP visualizations
        explainer = shap.TreeExplainer(final_clf)
        shap_values = explainer.shap_values(X_test[x_cols])

        shap_explainer_output = {
            "explainer": explainer,
            "shap_values": shap_values,
            "X_test": X_test[x_cols],
        }
        joblib.dump(shap_explainer_output, f"{model_data_outdir}/shap_explainer.jl")

        # Handle binary classification (LightGBM returns list of 2 arrays)
        if isinstance(shap_values, list):
            if len(shap_values) == 1:
                shap_used = shap_values[0]
            elif len(shap_values) == 2:
                shap_used = shap_values[1]  # class 1
            else:
                raise ValueError(
                    f"[ERROR] [{n_feats} feats] Unexpected SHAP value list length."
                )
        elif isinstance(shap_values, np.ndarray):
            shap_used = shap_values
        else:
            raise TypeError(
                f"[ERROR] [{n_feats} feats] Unexpected SHAP output type: {type(shap_values)}"
            )

        shap_importance = np.abs(shap_used).mean(axis=0)
        shap_importance_df = pd.DataFrame(
            {"Feature": X_test[x_cols].columns, "MeanAbsoluteSHAP": shap_importance}
        ).sort_values(by="MeanAbsoluteSHAP", ascending=False)

        joblib.dump(
            shap_importance_df, f"{model_data_outdir}/shap_importance_df.jl",
        )
        joblib.dump(shap_used, f"{model_data_outdir}/shap_values.jl")

        for plot_type in ["bar", "dot"]:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_used, X_test[x_cols], plot_type=plot_type, show=False
            )
            plt.tight_layout()
            plt.savefig(f"{model_plot_outdir}/final_shap_{plot_type}.png")
            plt.clf()

        joblib.dump(
            {"shap_val": shap_used, "feat_name": X_test[x_cols]},
            f"{model_data_outdir}/shap_plot_data.jl",
        )

        if "best_hyperparams" in config:
            config.pop("best_hyperparams")
        config_df = pd.DataFrame([{"Key": k, "Value": v} for k, v in config.items()])

        current_scores_df = pd.DataFrame.from_dict(current_scores)

        hyperparams_df = pd.DataFrame.from_dict(
            best_hyperparams, orient="index", columns=["Value"]
        )
        with pd.ExcelWriter(f"{model_outdir}/test_data_results.xlsx") as writer:
            pd.DataFrame.from_dict(
                model_info, orient="index", columns=["Value"]
            ).to_excel(writer, sheet_name="model_info")
            config_df.to_excel(writer, sheet_name="model_params")
            current_scores_df.T.to_excel(writer, sheet_name="performance")
            hyperparams_df.to_excel(writer, sheet_name="hyperparams")

            for key, df in report_dfs.items():
                df.to_excel(writer, sheet_name=f"{key}_report")
            shap_importance_df.to_excel(
                writer, sheet_name="SHAP_Importance", index=False
            )

            for svd_col, df in svd_words.items():
                df.to_excel(writer, sheet_name=f"{svd_col}_words", index=False)

        combined_summary[n_feats] = {
            "params": config_df,
            "hyperparams": hyperparams_df,
            "shap": shap_importance_df,
        }
        for key, df in report_dfs.items():
            combined_summary[n_feats][f"{key}_report"] = df

        config["hyperparams"] = best_hyperparams

    joblib.dump(combined_scores, f"{args.outdir}/combined_scores.jl")

    inverted_summary = {}
    for model_name, sections in combined_summary.items():
        for section_name, df in sections.items():
            if section_name not in inverted_summary:
                inverted_summary[section_name] = {}
            inverted_summary[section_name][model_name] = df

    for key in inverted_summary:
        inverted_summary[key] = pd.concat(inverted_summary[key])

    print(f"[INFO] Plotting ROC and precision-recall curves for all")
    proba_dict = {
        "test": y_probas,
        "oof": oof_proba_list,
    }
    joblib.dump(
        {"y_test_binary": y_test, "y_probas": y_probas},
        f"{args.outdir}/ROC_PR_curve_test_data.jl",
    )
    joblib.dump(
        {"y_train": y_train, "oof_probas": oof_proba_list, "oof_preds": oof_pred_list},
        f"{args.outdir}/ROC_PR_curve_oof_data.jl",
    )
    for key, y_prob_dict in proba_dict.items():
        plt.figure()
        if key == "test":
            y_true = y_test
        else:
            y_true = y_train
        for n_feats, y_proba in y_prob_dict.items():
            # ROC curve
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
            plt.plot(fpr, tpr, label=str(n_feats))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{LABEL_LOOKUP[args.subreddit]} Stage 1 ROC Curve")
        plt.legend(title="Features")
        plt.tight_layout()
        plt.savefig(f"{plot_outdir}/{key}_roc_curve.png", dpi=300)
        plt.close()

        plt.figure()
        for n_feats, y_proba in y_prob_dict.items():
            # Precision-recall curves
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
            plt.plot(recall, precision, label=str(n_feats))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{LABEL_LOOKUP[args.subreddit]} Stage 1 Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_outdir}/{key}_precision_recall_curve.png", dpi=300)
        plt.close()

    metrics = [m for m in SCORERS.keys() if m not in EXCLUDE_SCORES]

    for key, results_dict in combined_scores.items():
        for metric in metrics:
            plt.figure(figsize=(8, 5))
            plt.plot(
                results_dict["n_feats"],
                results_dict[metric],
                marker="o",
                label=metric.upper(),
            )

            # Optional: plot 95% CI shaded area
            lower_err = []
            upper_err = []
            for metric_val, ci_bounds in zip(
                results_dict[metric], results_dict[f"{metric} CI"]
            ):
                lower_err.append(metric_val - ci_bounds[0])
                upper_err.append(ci_bounds[1] - metric_val)

            plt.errorbar(
                results_dict["n_feats"],
                results_dict[metric],
                yerr=[lower_err, upper_err],
                fmt="none",
                capsize=3,
                color="gray",
            )

            plt.title(
                f"{LABEL_LOOKUP[args.subreddit]} - {key} {metric} vs number of features"
            )
            plt.xlabel("Number of features")
            plt.ylabel(metric.upper())
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            plt.savefig(f"{plot_outdir}/{key}_{metric}_vs_n_feats.png", dpi=300)
            plt.close()

    end = dt.datetime.now()
    total_runtime = end - start
    model_info["total_runtime"] = str(total_runtime)
    with pd.ExcelWriter(f"{args.outdir}/stage1_model_summaries.xlsx") as writer:
        pd.DataFrame.from_dict(model_info, orient="index", columns=["Value"]).to_excel(
            writer, sheet_name="model_info"
        )
        for key, out_dict in combined_scores.items():
            pd.DataFrame.from_dict(out_dict).to_excel(
                writer, sheet_name=f"{key}_metrics"
            )
        for key, output_df in inverted_summary.items():
            output_df.to_excel(writer, sheet_name=f"all_{key}")

    model_params = {
        "params": params,
        "info": model_info,
    }
    joblib.dump(model_params, f"{args.outdir}/thread_start_final_model_params.jl")
    print(f"[OK] Saved all outputs to: {args.outdir}")
    print(f"[OK] Finished. Total runtime {end-start}.")


if __name__ == "__main__":
    main()
