import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import datetime as dt
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
from scipy.optimize import minimize
from functools import partial

LABEL_LOOKUP = {
    "crypto": "r/CryptoCurrency",
    "politics": "r/politics",
    "conspiracy": "r/Conspiracy",
}

usage_msg = (
    "Usage: python 47_STAGE_2_4_MODEL.py <subreddit> <run_number> <collection> (optional <n_feat mods>)\n"
    + "Test mode: python 47_STAGE_2_4_MODEL.py <subreddit> test"
)

CLASS_NAMES = {
    3: ["Stalled", "Small", "Large"],
    4: ["Stalled", "Small", "Medium", "Large"],
}
CLASS_BIN_EDGES = {
    3: [0.5],
    4: [1 / 3, 2 / 3],
}


def class_precision(y_true, y_pred, class_idx):
    return precision_score(
        y_true, y_pred, zero_division=0, labels=[class_idx], average=None
    )[0]


def class_recall(y_true, y_pred, class_idx):
    return recall_score(
        y_true, y_pred, zero_division=0, labels=[class_idx], average=None
    )[0]


scorers = {
    "MCC": matthews_corrcoef,
    "F1": partial(f1_score, average="macro"),
    "Balanced accuracy": balanced_accuracy_score,
    "Precision": partial(precision_score, zero_division=0, average="macro"),
    "Recall": partial(recall_score, zero_division=0, average="macro"),
}

reports = {
    "Report": partial(classification_report, output_dict=True, zero_division=0),
    "CM": confusion_matrix,
}


def ci(arr, interval=[2.5, 97.5]):
    return np.percentile(arr, [2.5, 97.5])


if len(sys.argv) < 4:
    print(usage_msg)
    sys.exit(1)

subreddit = sys.argv[1].lower()
if subreddit not in LABEL_LOOKUP:
    print(f"{subreddit} invalid, exiting.")
    exit()

tfidf_folder = f"REGDATA/outputs/tfidf_analysis/{subreddit}/Run_20250706"

collection = sys.argv[3]
run_number = sys.argv[2]

no_collection = False
if collection.lower() == "0":
    no_collection = True
    indir = f"REGDATA/outputs/{subreddit}"
    outdir = f"{indir}/Run_{run_number}"
    data_infile = f"{indir}/0_preprocessing/train_test_data_dict.jl"
else:
    indir = f"REGDATA/outputs/{subreddit}_14"
    outdir = f"{indir}/TwoStageModel/Run_{run_number}"
    data_infile = f"{indir}/train_test_data_dict.jl"

if "test" in run_number.lower():
    print("⚠ Running in TEST mode")
    test = True
    feature_counts = []
    n_splits = 2
    n_bootstrap = 100
else:
    test = False
    try:
        run_number = int(sys.argv[2])
    except:
        run_number = str(sys.argv[2])
    n_bootstrap = 1000
    n_splits = 5

print(f"Running stage 2 model evaluation for {subreddit}, run {run_number}")
start = dt.datetime.now()
print(f"Starting at {start}")
print("Loading data")

# Load data
print(f"Loading {data_infile}")
data_dict = joblib.load(data_infile)
X_full = data_dict["X_train"] if not test else data_dict["X_train"].head(200)
X_test = data_dict["X_test"] if not test else data_dict["X_test"].head(100)

y_test = data_dict["Y_test"] if not test else data_dict["Y_test"].head(100)
y_full = data_dict["Y_train"] if not test else data_dict["Y_train"].head(200)

if not no_collection:
    y_full = y_full["log_thread_size"]
    y_test = y_test["log_thread_size"]

# Load selected features and parameters
random_state = 42

model_info = {
    "subreddit": subreddit,
    "stage": 4,
    "stage_4_date": dt.date.today(),
    "run_number": run_number,
    "random_state": random_state,
    "ci_interval": [2.5, 97.5],
}

# Load text models for SVD explanation
print(f"Loading TF-IDF and SVD vectorizers from {tfidf_folder}")
tfidf_vectorizer = joblib.load(f"{tfidf_folder}/optuna_tfidf_vectorizer.jl")
svd_model = joblib.load(f"{tfidf_folder}/optuna_svd_model.jl")
feature_names = tfidf_vectorizer.get_feature_names_out()
svd_components = svd_model.components_


def get_preds(thresholds, y_probas):
    preds = []
    for row in y_probas:
        passed = [
            row[i] if row[i] >= thresholds[i] else -1 for i in range(len(thresholds))
        ]
        preds.append(np.argmax(passed) if max(passed) != -1 else np.argmax(row))
    return preds


def neg_mcc(thresholds, y_probas, y_true):
    preds = get_preds(thresholds, y_probas)
    return -matthews_corrcoef(y_true, preds)


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


model_start = dt.datetime.now()
results_outdir = f"{outdir}/4_model_evaluation"
os.makedirs(outdir, exist_ok=True)
os.makedirs(results_outdir, exist_ok=True)

tuning_params_infile = f"{outdir}/params_post_hyperparam_tuning.jl"
if not os.path.isfile(tuning_params_infile):
    print(
        "Hyperparameter tuning parameters not found - loading tuning parameters instead."
    )
    tuning_params_infile = f"{outdir}/tuning_param_dict.jl"
tuning_dict = joblib.load(tuning_params_infile)
tuning_params = tuning_dict["params"]
calibration = tuning_dict["info"]["calibration"]
n_classes = tuning_dict["info"]["n_classes"]
model_info["n_classes"] = n_classes
for i, class_name in enumerate(CLASS_NAMES[n_classes]):
    scorers[f"{class_name} precision"] = partial(class_precision, class_idx=i)
    scorers[f"{class_name} recall"] = partial(class_recall, class_idx=i)
feature_counts = list(tuning_params.keys())
print(f"Feature counts: {feature_counts}")
model_info["feature_counts"] = feature_counts

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
    for k in scorers:
        scores_dict[k] = []
        scores_dict[f"{k} CI"] = []
y_pred_dict = {}
print("Going through stage2 feature models")
for n_feats in feature_counts:
    model_outdir = f"{results_outdir}/model_{n_feats}"
    os.makedirs(model_outdir, exist_ok=True)
    current_scores = {}
    print(f"n_feats: {n_feats}")
    config = tuning_params[n_feats]
    x_cols = config["features"]
    cw = config["final_class_weights"]
    initial_guess = config["final_threshold"]
    if isinstance(initial_guess, dict):
        initial_guess = list(initial_guess.keys())
    bounds = [(0.0, 1.0)] * n_classes
    config["final_n_features"] = n_feats
    bins = config["bins"]
    hyperparams = config["best_hyperparams"] if "best_hyperparams" in config else {}

    assert (
        len(bins) == n_classes + 1
    ), f"bin_edges must have length n_classes+1 ({n_classes+1}), got {len(bins)}."
    print(f"Getting started train and test data")
    X_tr = X_full[x_cols]
    X_te = X_test[x_cols]
    y_tr = pd.cut(y_full, bins=bins, labels=False, include_lowest=True)
    y_te = pd.cut(y_test, bins=bins, labels=False, include_lowest=True)

    unique_bins = np.unique(y_tr.dropna())
    assert len(unique_bins) == n_classes, (
        f"pd.cut produced {len(unique_bins)} classes, expected {n_classes}. "
        f"Unique bins found: {unique_bins}"
    )
    y_tr_bin_counts = y_tr.value_counts(sort=False)
    y_tr_bin_counts.index.name = "Class"
    y_te_bin_counts = y_te.value_counts(sort=False)
    y_te_bin_counts.index.name = "Class"
    thread_size_bins = [round(np.exp(x)) for x in bins]
    bin_ranges = [
        [
            thread_size_bins[i] if i == 0 else thread_size_bins[i] + 1,
            thread_size_bins[i + 1],
        ]
        for i in range(0, len(thread_size_bins) - 1)
    ]
    bin_count_df = pd.DataFrame(
        {"Range": bin_ranges, "Train": y_tr_bin_counts, "Test": y_te_bin_counts}
    )

    print("Getting OOF predictions")
    print(f"Creating {n_splits} CV loops for OOF started train")
    i = 1

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_probas = np.zeros((len(X_tr), n_classes))
    thresholds = []
    for train_idx, val_idx in kf.split(X_tr, y_tr):
        print(f"Loop {i}/{n_splits}")
        X_fold_tr, X_fold_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_fold_tr, y_fold_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
        y_fold_cont_tr, y_fold_cont_val = (
            y_full.iloc[train_idx],
            y_full.iloc[val_idx],
        )

        if calibration:
            X_fold_tr, X_calib, y_fold_tr, y_calib = train_test_split(
                X_fold_tr,
                y_fold_tr,
                train_size=0.8,
                stratify=y_fold_tr,
                random_state=random_state,
            )

        X_fold_tr, X_thresh_calib, y_fold_tr, y_thresh_calib = train_test_split(
            X_fold_tr,
            y_fold_tr,
            train_size=0.8,
            stratify=y_fold_tr,
            random_state=random_state,
        )
        print("Training model.")

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            class_weight=cw,
            random_state=random_state,
            verbosity=-1,
            **hyperparams,
        )
        model.fit(X_fold_tr, y_fold_tr)
        if calibration:
            calibrated_clf = CalibratedClassifierCV(
                model,
                method="isotonic",
                cv="prefit",
            )
            calibrated_clf.fit(X_calib, y_calib)
            proba = calibrated_clf.predict_proba(X_fold_val)
            calib_proba = calibrated_clf.predict_proba(X_thresh_calib)
        else:
            proba = model.predict_proba(X_fold_val)
            calib_proba = model.predict_proba(X_thresh_calib)

        oof_probas[val_idx] = proba

        result = minimize(
            neg_mcc,
            x0=initial_guess,
            bounds=bounds,
            args=(calib_proba, y_thresh_calib),
            method="L-BFGS-B",
        )
        thresholds.append(result.x)
        i += 1

    thresholds_final = []
    for i in range(n_classes):
        thresholds_final.append(np.mean([t[i] for t in thresholds]))
    oof_preds = get_preds(thresholds_final, oof_probas)
    config["model_threshold"] = thresholds_final
    print(f"Creating and fitting model")
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        class_weight=cw,
        random_state=random_state,
        verbosity=-1,
        **hyperparams,
    )
    model.fit(X_tr, y_tr)

    if calibration:
        print(f"Calibrating classifier")
        calibrated = CalibratedClassifierCV(
            model,
            method="isotonic",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        )
        calibrated.fit(X_tr, y_tr)

        print("Getting test set predicted probabilities")
        y_proba = calibrated.predict_proba(X_te)
    else:
        y_proba = model.predict_proba(X_te)
    y_pred = get_preds(thresholds_final, y_proba)

    y_pred_dict[n_feats] = {"test": y_pred, "oof": oof_preds}
    model_data = {
        "subreddit": subreddit,
        "run_number": run_number,
        "n_feats": n_feats,
        "model": model,
        "X_train": X_tr,
        "y_train": y_tr,
        "X_test": X_te,
        "y_test": y_te,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    if calibration:
        model_data.update({"calibrated_model": calibrated})
    joblib.dump(model_data, f"{model_outdir}/{n_feats}_feats_model_data.jl")

    to_measure = {"test": (y_te, y_pred), "oof": (y_tr, oof_preds)}
    print("Calculating classification report and performance scoring metrics")

    for key, (true_y, preds) in to_measure.items():

        print(f"Getting report metrics for {key} predictions")
        # Report metrics
        performance_metrics = {}
        report_metrics = {}
        for k, score_f in scorers.items():
            performance_metrics[k] = score_f(true_y, preds)
        for k, report_f in reports.items():
            report_metrics[k] = report_f(true_y, preds)

        print("Bootstrapping metrics")
        # Bootstrapping main metrics
        rng = np.random.RandomState(random_state)  # for reproducibility

        bootstrap_metrics = {}
        for k in performance_metrics:
            bootstrap_metrics[k] = []

        bootstrap_cms = []

        for i in range(n_bootstrap):
            indices = rng.choice(len(true_y), size=len(true_y), replace=True)
            y_true_bs = np.array(true_y)[indices]
            y_pred_bs = np.array(preds)[indices]
            for k, score_f in scorers.items():
                bootstrap_metrics[k].append(score_f(y_true_bs, y_pred_bs))
            bootstrap_cms.append(
                confusion_matrix(y_true_bs, y_pred_bs, labels=list(range(0, n_classes)))
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

        for i in range(n_classes):
            for j in range(n_classes):
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

        print(f"Creating plots")
        joblib.dump(
            confusion_matrix_data,
            f"{model_outdir}/{key}_{n_feats}_confusion_matrix_data.jl",
        )
        plt.figure()
        sns.heatmap(
            report_metrics["CM"],
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=CLASS_NAMES[n_classes],
            yticklabels=CLASS_NAMES[n_classes],
        )
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.title(
            f"{LABEL_LOOKUP[subreddit]} Stage 2 {key} {n_feats} feats Confusion Matrix"
        )
        plt.tight_layout()
        plt.savefig(
            f"{model_outdir}/{key}_{n_feats}_feats_confusion_matrix.png",
            dpi=300,
        )
        plt.close()

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te)
    joblib.dump(shap_values, f"{model_outdir}/{n_feats}_feats_shap_values.jl")
    if isinstance(shap_values, np.ndarray):
        if shap_values.shape[2] == n_classes:  # (samples, features, classes)
            # Transpose to (classes, samples, features)
            shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    for class_idx, label in enumerate(CLASS_NAMES[n_classes]):

        plt.figure()
        shap.summary_plot(shap_values[class_idx], X_te, show=False)
        plt.title(f"{LABEL_LOOKUP[subreddit]} SHAP - {label}")
        plt.tight_layout()
        plt.savefig(f"{model_outdir}/{n_feats}_feats_shap_{label}.png", dpi=300)
        plt.close()

    joblib.dump(
        {"shap_values": shap_values, "X_test": X_te},
        f"{model_outdir}/{n_feats}_shap_plot_data.jl",
    )

    # Predictions
    pred_dict = {
        "test": {
            "index": X_te.index,
            "true_class": y_te,
            "predicted_class": y_pred,
        },
        "oof": {"index": X_tr.index, "oof_y_true": y_tr, "oof_y_pred": oof_preds},
    }
    for i in range(n_classes):
        pred_dict["test"][f"proba_class_{i}"] = y_proba[:, i]

    for key, df in pred_dict.items():
        pd.DataFrame(df).to_csv(
            f"{model_outdir}/{n_feats}_feats_{key}_preds.csv", index=False
        )

    # SVD words
    svd_cols = [col for col in x_cols if col.startswith("svd_")]
    svd_words = get_svd_words(svd_cols) if svd_cols else {}

    # Save to Excel
    report_df = (
        pd.DataFrame(report_metrics["Report"]).transpose().reset_index(names=["Class"])
    )
    # Average absolute SHAP values across all classes
    # Stack into a 3D array: (n_classes, n_samples, n_features)
    if isinstance(shap_values, list):
        shap_array = np.stack([np.abs(sv) for sv in shap_values], axis=0)
        shap_mean = shap_array.mean(axis=(0, 1))
    else:
        shap_mean = np.abs(shap_values).mean(axis=0)

    shap_mean_df = pd.DataFrame({"feature": X_te.columns, "MeanAbsSHAP": shap_mean})
    shap_mean_df.sort_values("MeanAbsSHAP", ascending=False, inplace=True)
    current_scores_df = pd.DataFrame.from_dict(current_scores)
    excel_path = f"{model_outdir}/{n_feats}_feats_test_data_results.xlsx"
    print(f"Saving results to {excel_path}")
    if "best_hyperparams" in config:
        config.pop("best_hyperparams")
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
            writer,
            sheet_name="info",
        )
        bin_count_df.to_excel(writer, sheet_name="class_sizes")
        params_df.to_excel(writer, sheet_name="model_params", index=False)
        hyperparams_df.to_excel(writer, sheet_name="hyperparams", index=False)
        report_df.to_excel(writer, sheet_name="classification_report", index=False)
        current_scores_df.to_excel(writer, sheet_name="performance")
        shap_mean_df.to_excel(writer, sheet_name="shap_summary", index=False)
        for svd_name, df in svd_words.items():
            df.to_excel(writer, sheet_name=f"{svd_name}_words", index=False)

joblib.dump(combined_scores, f"{results_outdir}/combined_scores.jl")
for metric in scorers:
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

        plt.title(f"{LABEL_LOOKUP[subreddit]} - {metric.upper()} vs number of features")
        plt.xlabel("Number of features")
        plt.ylabel(metric.upper())
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{results_outdir}/{key}_{metric}_vs_n_feats.png", dpi=300)
        plt.close()

output_dfs = {
    "info": pd.DataFrame.from_dict(model_info, orient="index").reset_index(
        names=["Parameter"]
    ),
}
for key, df_list in combined_summary.items():
    output_dfs[key] = pd.concat(df_list, ignore_index=True)

excel_path = f"{results_outdir}/evaluation.xlsx"
print(f"Saving results to {excel_path}")
with pd.ExcelWriter(excel_path) as writer:
    for key, df in output_dfs.items():
        df.to_excel(writer, sheet_name=key, index=False)
    for key, out_dict in combined_scores.items():
        pd.DataFrame.from_dict(out_dict).to_excel(writer, sheet_name=f"{key}_metrics")
