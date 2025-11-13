# Stage 1 model evaluation
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
from functools import partial
import matplotlib.pyplot as plt
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample
import sys
import seaborn as sns
import os
import shap
import datetime as dt
import gc
from scipy.optimize import minimize_scalar

LABEL_LOOKUP = {
    "crypto": "r/CryptoCurrency",
    "politics": "r/politics",
    "conspiracy": "r/Conspiracy",
}

THRESHOLD = np.log(1)

usage_msg = (
    "Usage: python 47_2_STAGE_1_MODEL.py <subreddit> <run_number> <collection>\n"
    + "Test mode: python 47_2_STAGE_1_MODEL.py <subreddit> test"
)
if len(sys.argv) < 3:
    print(usage_msg)
    sys.exit(1)

scorers = {
    "MCC": matthews_corrcoef,
    "F1": f1_score,
    "Report": partial(classification_report, output_dict=True, zero_division=0),
    "CM": confusion_matrix,
    "Balanced accuracy": balanced_accuracy_score,
}

EXCLUDE_SCORES = ["Report", "CM"]

CLASS_NAMES = {
    0: "Stalled",
    1: "Started",
}
for i, class_name in CLASS_NAMES.items():
    scorers.update(
        {
            f"Precision {class_name}": partial(
                precision_score, pos_label=i, zero_division=0
            ),
            f"Recall {class_name}": partial(recall_score, pos_label=i, zero_division=0),
        }
    )
subreddit = sys.argv[1].lower()
if subreddit not in LABEL_LOOKUP:
    print(f"{subreddit} invalid, exiting.")
    exit()

tfidf_folder = f"REGDATA/outputs/tfidf_analysis/{subreddit}/Run_20250706"

run_number = sys.argv[2]
if "test" in run_number.lower():
    test = True
    n_bootstrap = 20
    n_splits = 2
else:
    test = False
    try:
        run_number = int(sys.argv[2])
    except:
        run_number = str(sys.argv[2])

    n_bootstrap = 1000
    n_splits = 5


collection = sys.argv[3]
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

results_outdir = f"{outdir}/stage_1_4_model_evaluation"
os.makedirs(results_outdir, exist_ok=True)

print(f"Running final model evaluation for {subreddit}, run {run_number}")

start = dt.datetime.now()
print(f"Started at {start}.")

print(f"Loading data")
# Load data
data_dict = joblib.load(data_infile)
X_train = data_dict["X_train"] if not test else data_dict["X_train"].head(200)
X_test = data_dict["X_test"] if not test else data_dict["X_test"].head(100)
y_train_data = data_dict["Y_train"] if not test else data_dict["Y_train"].head(200)
y_test_data = data_dict["Y_test"] if not test else data_dict["Y_test"].head(100)
if not no_collection:
    y_train_data = y_train_data["log_thread_size"]
    y_test_data = y_test_data["log_thread_size"]
y_train = (y_train_data > np.log(1)).astype(int)
y_test_binary = (y_test_data > np.log(1)).astype(int)
del y_train_data, y_test_data
gc.collect()


# Load selected features and parameters
if os.path.isfile(f"{outdir}/params_post_hyperparam_tuning.jl"):
    print(
        f"Loading hyperparameter-tuned params from {outdir}/params_post_hyperparam_tuning.jl"
    )
    params_infile = f"{outdir}/params_post_hyperparam_tuning.jl"

else:
    print(f"Loading model params from {outdir}/stage1_model_param_dict.jl")
    params_infile = f"{outdir}/stage1_model_param_dict.jl"

model_params = joblib.load(params_infile)
params = model_params["params"]
model_info = model_params["info"]
model_info["model_evaluation_date"] = dt.date.today()
random_state = model_info["random_state"]
model_info["model_params_infile"] = params_infile
model_info["n_bootstrap"] = n_bootstrap
calibration = model_info["calibration"]
if "scorer" in model_info:
    scorer = model_info["scorer"]
else:
    scorer = "MCC"
if scorer == "F-beta":
    if "beta" in model_info:
        beta = model_info["beta"]
    else:
        beta = 2
    scorers["F-beta"] = partial(fbeta_score, beta=beta)

kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

svd_cols_present = False
for n_feats, config in params.items():
    x_cols = config["features"]
    svd_cols = [col for col in x_cols if col.startswith("svd")]
    if len(svd_cols) > 0:
        svd_cols_present = True
        break

if svd_cols_present:
    print("SVD cols present")
    # load TF-IDF and svd model
    print(f"Loading TF-IDF and SVD vectorizers from {tfidf_folder}")
    tfidf_vectorizer = joblib.load(f"{tfidf_folder}/optuna_tfidf_vectorizer.jl")
    svd_model = joblib.load(f"{tfidf_folder}/optuna_svd_model.jl")
    feature_names = tfidf_vectorizer.get_feature_names_out()
    svd_components = svd_model.components_


def neg_score(threshold, y_proba, y_true):
    y_pred = (y_proba >= threshold).astype(int)
    return -scorers[scorer](y_true, y_pred)


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
    for k in [i for i in scorers.keys() if i not in EXCLUDE_SCORES]:
        scores_dict[k] = []
        scores_dict[f"{k} CI"] = []

for n_feats, config in params.items():
    model_outdir = f"{results_outdir}/mod_{n_feats}_outputs"
    os.makedirs(model_outdir, exist_ok=True)
    model_start = dt.datetime.now()
    print(f"Model n_feats: {n_feats}")
    x_cols = config["features"]
    cw = config["final_class_weights"]
    # thresh = config["final_threshold"]
    best_hyperparams = (
        config.get("best_hyperparams", {}) if "best_hyperparams" in config else {}
    )

    print(f"Selected features: {x_cols}")
    print(f"Class weights: {cw}")  # , Threshold: {thresh}")

    # Start with fixed parameters
    fixed_params = {
        "objective": "binary",
        "class_weight": cw,
        "random_state": random_state,
        "verbose": -1,
    }

    # Add in best hyperparameters, but let fixed_params override if there’s conflict
    combined_params = {**best_hyperparams, **fixed_params}

    oof_probas = np.zeros(len(X_train))
    thresholds = []

    print(f"Creating {n_splits} CV loops for OOF started train")
    i = 1
    for train_idx, val_idx in kf.split(X_train, y_train):
        print(f"Loop {i}/{n_splits}")
        X_tr, X_val = X_train.iloc[train_idx][x_cols], X_train.iloc[val_idx][x_cols]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        X_tr, X_thresh_calib, y_tr, y_thresh_calib = train_test_split(
            X_tr,
            y_tr,
            train_size=0.8,
            stratify=y_tr,  # preserve class balance
            random_state=random_state,  # for reproducibility
        )

        if calibration:
            X_tr, X_calib, y_tr, y_calib = train_test_split(
                X_tr,
                y_tr,
                train_size=0.6,  # 60% for train, 40% for calib
                stratify=y_tr,  # preserve class balance
                random_state=random_state,  # for reproducibility
            )

        print("Training model.")
        clf = lgb.LGBMClassifier(**combined_params)
        clf.fit(X_tr, y_tr)
        if calibration:
            print("Calibrating model")
            calibrated_clf = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
            calibrated_clf.fit(X_calib, y_calib)
            proba = calibrated_clf.predict_proba(X_val)[:, 1]
            calib_proba = calibrated_clf.predict_proba(X_thresh_calib)[:, 1]
        else:
            proba = clf.predict_proba(X_val)[:, 1]
            calib_proba = clf.predict_proba(X_thresh_calib)[:, 1]

        oof_probas[val_idx] = proba
        print("Using minimize_scalar to get threshold")
        result = minimize_scalar(
            neg_score,
            bounds=(0, 1),
            method="bounded",
            args=(calib_proba, y_thresh_calib),
        )
        thresholds.append(result.x)

        i += 1

    print("Averaging thresholds over folds")
    thresh = np.mean(thresholds)
    config["model_threshold"] = thresh
    oof_preds = (oof_probas >= thresh).astype(int)

    # Compute confidence intervals
    def ci(arr):
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]  # exclude NaNs
        return np.percentile(arr, [2.5, 97.5])

    print("Training final classifier")
    final_clf = lgb.LGBMClassifier(**combined_params)
    final_clf.fit(X_train[x_cols], y_train)

    if calibration:
        print("Calibrating final classifier")
        calibrated_final = CalibratedClassifierCV(
            final_clf,
            method="isotonic",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
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
        "test": (y_test_binary, test_pred, test_proba),
        "oof": (y_train, oof_preds, oof_probas),
    }

    current_scores = {}
    for key, (true_y, preds, probas) in to_measure.items():

        print(f"Getting report metrics for {key} predictions")
        # Report metrics
        performance_metrics = {"AUC": roc_auc_score(true_y, probas)}
        cm = scorers["CM"](true_y, preds)
        report = scorers["Report"](true_y, preds)
        for k, score_f in [
            (k, v) for (k, v) in scorers.items() if k not in EXCLUDE_SCORES
        ]:
            performance_metrics[k] = score_f(true_y, preds)

        print("Plotting ROC and precision-recall curves")
        # ROC curve
        auc = performance_metrics["AUC"]
        fpr, tpr, roc_thresholds = roc_curve(true_y, probas)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{LABEL_LOOKUP[subreddit]} Stage 1 {n_feats} feats {key} ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{model_outdir}/roc_curve_{key}_{n_feats}_feats.png", dpi=300)
        plt.close()

        # Precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(true_y, probas)
        f1 = performance_metrics["F1"]
        plt.figure()
        plt.plot(recall, precision, label=f"PR curve (F1 = {f1:.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(
            f"{LABEL_LOOKUP[subreddit]} Stage 1 {n_feats} feats Precision-Recall Curve"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"{model_outdir}/stage1_{n_feats}_feats_precision_recall_curve_{key}.png",
            dpi=300,
        )
        plt.close()

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
            y_proba_bs = np.array(probas)[indices]
            bootstrap_cms.append(
                confusion_matrix(y_true_bs, y_pred_bs, labels=list(range(0, 2)))
            )

            for k in bootstrap_metrics:
                if k == "AUC":
                    bootstrap_metrics[k].append(roc_auc_score(y_true_bs, y_proba_bs))
                else:
                    bootstrap_metrics[k].append(scorers[k](y_true_bs, y_pred_bs))

        metric_cis = {}
        for k, v in bootstrap_metrics.items():
            metric_cis[k] = ci(v).tolist()
        current_scores[key] = {"n_feats": n_feats}
        combined_scores[key]["n_feats"].append(n_feats)
        for k, v in performance_metrics.items():
            combined_scores[key][k].append(v)
            combined_scores[key][f"{k} CI"].append(metric_cis[k])
            current_scores[key][k] = v
            for i, bound in enumerate(["upper", "lower"]):
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

        cm_cis.update({
            "CM": cm,
        })
        joblib.dump(
            cm_cis,
            f"{model_outdir}/{key}_{n_feats}_confusion_matrix_data.jl",
        )
        prob_true, prob_pred = calibration_curve(
            true_y, probas, n_bins=10, strategy="quantile"
        )
        joblib.dump(
            {"prob_true": prob_true, "prob_pred": prob_pred},
            f"{model_outdir}/{n_feats}_calibration_curve_inputs_{key}.jl",
        )

        plt.figure()
        plt.plot(prob_pred, prob_true, marker="o", label="Model")
        plt.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated"
        )
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title(
            f"{LABEL_LOOKUP[subreddit]} Stage 1 {n_feats} feats Calibration Curve"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"{model_outdir}/stage1_{n_feats}_{key}_feats_calibration_curve.png",
            dpi=300,
        )
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.title(f"{LABEL_LOOKUP[subreddit]} Stage 1 {key} {n_feats} Confusion Matrix")
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=["Stalled", "Started"],
            yticklabels=["Stalled", "Started"],
        )
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.tight_layout()
        plt.savefig(
            f"{model_outdir}/stage1_{n_feats}_{key}_confusion_matrix.png", dpi=300
        )
        plt.close()

        joblib.dump(cm, f"{model_outdir}/{n_feats}_{key}_confusion_matrix_plot_data.jl")

    print("Getting starting thread predictions")
    # Save predicted started threads
    started_dict = {
        "train": pd.DataFrame(
            {
                "index": X_train.index,
                "proba": oof_probas,
                "predicted": oof_preds,
            }
        ),
        "test": pd.DataFrame(
            {
                "index": X_test.index,
                "proba": test_proba,
                "predicted": test_pred,
            }
        ),
    }
    print(f"Saving results to {model_outdir}")
    models_dict = {"classifier": final_clf}
    if calibration:
        models_dict["calibrated_classifier"] = calibrated_final
    models_dict["X_test"] = X_test[x_cols]
    models_dict["y_test"] = true_y
    models_dict["X_train"] = X_tr
    models_dict["y_train"] = y_tr
    joblib.dump(started_dict, f"{model_outdir}/{n_feats}_feats_started_threads.jl")
    joblib.dump(models_dict, f"{model_outdir}/{n_feats}_feats_models.jl")
    final_clf.booster_.save_model(f"{outdir}/stage1_{n_feats}_feats_final_model.txt")

    report_df = pd.DataFrame(report).transpose()

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

    # SHAP visualizations
    explainer = shap.TreeExplainer(final_clf)
    shap_values = explainer.shap_values(X_test[x_cols])

    shap_explainer_output = {
        "explainer": explainer,
        "shap_values": shap_values,
        "X_test": X_test[x_cols]
    }
    joblib.dump(shap_explainer_output, f"{model_outdir}/shap_explainer.jl")

    # Handle binary classification (LightGBM returns list of 2 arrays)
    if isinstance(shap_values, list):
        if len(shap_values) == 1:
            shap_used = shap_values[0]
        elif len(shap_values) == 2:
            shap_used = shap_values[1]  # class 1
        else:
            raise ValueError("Unexpected SHAP value list length.")
    elif isinstance(shap_values, np.ndarray):
        shap_used = shap_values
    else:
        raise TypeError(f"Unexpected SHAP output type: {type(shap_values)}")

    shap_importance = np.abs(shap_used).mean(axis=0)
    shap_importance_df = pd.DataFrame(
        {"Feature": X_test[x_cols].columns, "MeanAbsoluteSHAP": shap_importance}
    ).sort_values(by="MeanAbsoluteSHAP", ascending=False)

    joblib.dump(
        shap_importance_df,
        f"{model_outdir}/stage1_{n_feats}_feats_shap_importance_df.jl",
    )
    joblib.dump(shap_used, f"{model_outdir}/stage1_{n_feats}_feats_shap_values.jl")

    for plot_type in ["bar", "dot"]:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_used, X_test[x_cols], plot_type=plot_type, show=False)
        plt.tight_layout()
        plt.savefig(f"{model_outdir}/stage1_{n_feats}_feats_final_shap_{plot_type}.png")
        plt.clf()

    joblib.dump(
        {"shap_val": shap_used, "feat_name": X_test[x_cols]},
        f"{model_outdir}/{n_feats}_shap_plot_data.jl",
    )

    model_end = dt.datetime.now()
    model_runtime = model_end - model_start
    model_info[f"{n_feats}_runtime"] = model_runtime
    if "best_hyperparams" in config:
        config.pop("best_hyperparams")
    config_df = pd.DataFrame([{"Key": k, "Value": v} for k, v in config.items()])

    current_scores_df = pd.DataFrame.from_dict(current_scores)

    hyperparams_df = pd.DataFrame.from_dict(
        best_hyperparams, orient="index", columns=["Value"]
    )
    with pd.ExcelWriter(
        f"{model_outdir}/stage1_{n_feats}_feats_test_data_results.xlsx"
    ) as writer:
        pd.DataFrame.from_dict(model_info, orient="index", columns=["Value"]).to_excel(
            writer, sheet_name="model_info"
        )
        config_df.to_excel(writer, sheet_name="model_params")
        current_scores_df.T.to_excel(writer, sheet_name="performance")
        hyperparams_df.to_excel(writer, sheet_name="hyperparams")

        report_df.to_excel(writer, sheet_name="classification_report")
        shap_importance_df.to_excel(writer, sheet_name="SHAP_Importance", index=False)

        for svd_col, df in svd_words.items():
            df.to_excel(writer, sheet_name=f"{svd_col}_words", index=False)

    combined_summary[n_feats] = {
        "report": report_df,
        "params": config_df,
        "hyperparams": hyperparams_df,
        "shap": shap_importance_df,
    }

    config["hyperparams"] = best_hyperparams

joblib.dump(combined_scores, f"{results_outdir}/combined_scores.jl")

inverted_summary = {}
for model_name, sections in combined_summary.items():
    for section_name, df in sections.items():
        if section_name not in inverted_summary:
            inverted_summary[section_name] = {}
        inverted_summary[section_name][model_name] = df

for key in inverted_summary:
    inverted_summary[key] = pd.concat(inverted_summary[key])


print("Plotting ROC and precision-recall curves for all")
proba_dict = {
    "test": y_probas,
    "oof": oof_proba_list,
}
joblib.dump(
    {"y_test_binary": y_test_binary, "y_probas": y_probas},
    f"{results_outdir}/ROC_PR_curve_test_data.jl",
)
joblib.dump(
    {"y_train": y_train, "oof_probas": oof_proba_list, "oof_preds": oof_pred_list},
    f"{results_outdir}/ROC_PR_curve_oof_data.jl",
)
for key, y_prob_dict in proba_dict.items():
    plt.figure()
    if key == "test":
        y_true = y_test_binary
    else:
        y_true = y_train
    for n_feats, y_proba in y_prob_dict.items():
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        plt.plot(fpr, tpr, label={n_feats})
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{LABEL_LOOKUP[subreddit]} Stage 1 ROC Curve")
    plt.legend(title="Features")
    plt.tight_layout()
    plt.savefig(f"{results_outdir}/roc_curve_{key}_all_feats.png", dpi=300)
    plt.close()

    plt.figure()
    for n_feats, y_proba in y_prob_dict.items():
        # Precision-recall curves
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
        plt.plot(recall, precision, label={n_feats})
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{LABEL_LOOKUP[subreddit]} Stage 1 Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_outdir}/all_feats_precision_recall_curve_{key}.png", dpi=300)
    plt.close()

metrics = [m for m in scorers.keys() if m not in EXCLUDE_SCORES]

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
        lower = [
            metric_val - ci[0]
            for metric_val, ci in zip(
                results_dict[metric], results_dict[f"{metric} CI"]
            )
        ]
        upper = [
            ci[1] - metric_val
            for metric_val, ci in zip(
                results_dict[metric], results_dict[f"{metric} CI"]
            )
        ]

        plt.errorbar(
            results_dict["n_feats"],
            results_dict[metric],
            yerr=[lower, upper],
            fmt="none",
            capsize=3,
            color="gray",
        )

        plt.title(f"{LABEL_LOOKUP[subreddit]} - {key} {metric} vs number of features")
        plt.xlabel("Number of features")
        plt.ylabel(metric.upper())
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{results_outdir}/{key}_{metric}_vs_n_feats.png", dpi=300)
        plt.close()


end = dt.datetime.now()
total_runtime = end - start
model_info["total_runtime"] = str(total_runtime)
print(f"Finished at {end}. Runtime: {total_runtime}.")
with pd.ExcelWriter(f"{results_outdir}/stage1_model_summaries.xlsx") as writer:
    pd.DataFrame.from_dict(model_info, orient="index", columns=["Value"]).to_excel(
        writer, sheet_name="model_info"
    )
    for key, out_dict in combined_scores.items():
        pd.DataFrame.from_dict(out_dict).to_excel(writer, sheet_name=f"{key}_metrics")
    for key, output_df in inverted_summary.items():
        output_df.to_excel(writer, sheet_name=f"all_{key}")

model_params = {
    "params": params,
    "info": model_info,
}
joblib.dump(model_params, f"{outdir}/stage_1_final_model_params.jl")


print(f"Final performance saved to {results_outdir}")
