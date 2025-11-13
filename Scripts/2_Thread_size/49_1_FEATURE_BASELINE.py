# Stage 2.1 - baseline feature scores
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
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
)
from sklearn.utils import resample
import sys
import seaborn as sns
import os
import json
import shap
import datetime as dt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from functools import partial

RANDOM_STATE = 42
PERFORMANCE_TOLERANCE = 0.95

LABEL_LOOKUP = {
    "crypto": "r/CryptoCurrency",
    "politics": "r/politics",
    "conspiracy": "r/Conspiracy",
}

THRESHOLD = np.log(1)

usage_msg = (
    "Usage: python 47_STAGE_2_1_FEATURE_BASELINE.py <subreddit> <run_number> <collection> <calibration> <n_classes> <max_feats>\n"
    + "Test mode: python 47_STAGE_2_1_FEATURE_BASELINE.py <subreddit> test <collection> "
)
if len(sys.argv) < 3:
    print(usage_msg)
    sys.exit(1)


subreddit = sys.argv[1].lower()
if subreddit not in LABEL_LOOKUP:
    print(f"{subreddit} invalid, exiting.")
    exit()


collection = sys.argv[3]
run_number = sys.argv[2]


calibration = sys.argv[4]
CALIBRATION_LOOKUP = ["cal", "c", "calibration", "calibrate"]
if calibration.lower() in CALIBRATION_LOOKUP:
    CALIBRATION = True
else:
    CALIBRATION = False
print(f"Calibration: {CALIBRATION}")

n_classes = int(sys.argv[5])
if n_classes == 2:
    mode = "binary"
elif n_classes > 2:
    mode = "multiclass"
else:
    print(f"Invalid number of classes {n_classes}")
    exit()
CLASS_NAMES = {
    3: ["Stalled", "Small", "Large"],
    4: ["Stalled", "Small", "Medium", "Large"],
}
CLASS_BIN_EDGES = {
    3: [0.5],
    4: [1 / 3, 2 / 3],
}

scorers = {
    "MCC": matthews_corrcoef,
    "Report": partial(classification_report, output_dict=True, zero_division=0),
    "CM": confusion_matrix,
    "Balanced accuracy": balanced_accuracy_score,
    "Precision": partial(precision_score, zero_division=0, average="macro"),
    "Recall": partial(recall_score, zero_division=0, average="macro"),
}

if mode == "multiclass":
    scorers["F1"] = partial(f1_score, average="macro")
else:
    scorers["F1"] = f1_score


EXCLUDE_SCORES = ["Report", "CM"]

no_collection = False
if collection.lower() == "0":
    no_collection = True
    indir = f"REGDATA/outputs/{subreddit}"
    data_infile = f"{indir}/0_preprocessing/train_test_data_dict.jl"

if "test" in run_number.lower():
    test = True
    n_bootstrap = 20
    max_feats = 4
    cv_splits = 2
else:
    test = False
    try:
        run_number = int(sys.argv[2])
    except:
        run_number = str(sys.argv[2])
    max_feats = sys.argv[6]
    if max_feats != "max":
        max_feats = int(max_feats)
    n_bootstrap = 1000
    # n_bootstrap = 10
    cv_splits = 5
    # cv_splits = 2
    # stage1_files_dict = {stage1_model_n_feats[0]: stage1_started_files[0]}

print(f"Running feature model baselines for {subreddit}, run {run_number}")

start = dt.datetime.now()
print(f"Started at {start}.")

print(f"Loading data")
# Load data
data_dict = joblib.load(data_infile)
X = data_dict["X_train"] if not test else data_dict["X_train"].head(200)
X_test = data_dict["X_test"] if not test else data_dict["X_test"].head(100)
y_test = data_dict["Y_test"] if not test else data_dict["Y_test"].head(100)
y = data_dict["Y_train"] if not test else data_dict["Y_train"].head(200)

if not no_collection:
    y = y["log_thread_size"]
    y_test = y_test["log_thread_size"]

if max_feats == "max":
    max_feats = len(X.columns)

if max_feats < 51:
    feature_counts = list(range(1, min(max_feats + 1, 51)))
else:
    feature_counts = list(range(1, min(max_feats + 1, 26)))
    if max_feats > 27:
        feature_counts += list(range(28, min(max_feats + 1, 51), 2))
        if max_feats > 54:
            feature_counts += list(range(55, min(max_feats + 1, 101), 5))
        if max_feats > 109:
            feature_counts += list(range(110, min(max_feats + 1, 150), 10))
        if max_feats > 149:
            feature_counts += list(range(150, max_feats + 1), 25)

# Load selected features and parameters
random_state = 42




model_info = {
    "stage": 1,
    "stage_2_1_date": dt.datetime.today(),
    "2_1_cv_splits": cv_splits,
    "subreddit": subreddit,
    "n_classes": n_classes,
    "feature_grid": feature_counts,
    "random_state": random_state,
    "run_number": run_number,
    "max_feats": max_feats,
    "performance_tolerance": PERFORMANCE_TOLERANCE,
    "calibration": CALIBRATION,
}

outdir = f"{indir}/Run_{run_number}"
os.makedirs(outdir, exist_ok=True)

model_start = dt.datetime.now()

results_outdir = f"{outdir}/1_feature_baselines"
os.makedirs(results_outdir, exist_ok=True)

if mode = "multiclass":
    bin_edges = [np.log(1) - 1e-3]
    bin_edges.append(np.log(2) - 1e-3)
    bin_edges.extend([y[y > np.log(1)].quantile(x) for x in CLASS_BIN_EDGES[n_classes]])
    bin_edges.append(y.max() + 1e-3)

    assert (
        len(bin_edges) == n_classes + 1
    ), f"bin_edges must have length n_classes+1 ({n_classes+1}), got {len(bin_edges)}."

    y_bins = pd.cut(y, bins=bin_edges, labels=False)


    unique_bins = np.unique(y_bins.dropna())
    assert len(unique_bins) == n_classes, (
        f"pd.cut produced {len(unique_bins)} classes, expected {n_classes}. "
        f"Unique bins found: {unique_bins}"
    )
    model_info["bins"] = bin_edges
    model_info["thread_size_bins"] = [round(np.exp(x)) for x in bin_edges]
else:
    y_bins = (y > np.log(1)).astype(int)
class_counts = y_bins.value_counts(sort=False)
for class_num in class_counts.index:
    model_info[f"class_{class_num}_count"] = class_counts[class_num]

importance_dfs = []
all_fold_dfs = {}
bootstrap_vals = {}

print("Discretizing target for cross-validation folds...")
# Outer CV
outer_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
print("Training data on folds")
for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y_bins)):
    fold_start_time = dt.datetime.now()
    print(f"Fold {fold + 1}/{cv_splits}")
    all_fold_dfs[fold + 1] = {}
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_bins.iloc[train_idx], y_bins.iloc[val_idx]
    if CALIBRATION:
        X_calib, X_val, y_calib, y_val = train_test_split(
            X_val,
            y_val,
            test_size=0.5,
            stratify=y_val,
            random_state=RANDOM_STATE,
        )

    print(f"Fold {fold+1}: Precomputing ranked features for candidate bins")
    ranked_features_dict = {}
    # Get feature importances for ranking
    selector_params = {
        'objective': mode,
        'class_weight': "balanced",
        'random_state': random_state,
        'verbose':-1,
    }
    if mode == "multiclass":
        selector_params['num_class'] = n_classes
    selector = lgb.LGBMClassifier(
        **selector_params
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
    ranked_features = pd.Series(combined_importance, index=X_train.columns)
    ranked_features = ranked_features.sort_values(ascending=False).index.tolist()

    for n_feats in feature_counts:
        top_feats = ranked_features[:n_feats]

        print(f"Fold {fold+1}, {n_feats} features: Training model")

        clf = lgb.LGBMClassifier(
            **selector_params
        )
        clf.fit(X_train[top_feats], y_train)
        if CALIBRATION:
            calibrated_clf = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
            calibrated_clf.fit(X_calib[top_feats], y_calib)
            y_proba = calibrated_clf.predict_proba(X_val[top_feats])
            y_pred = calibrated_clf.predict(X_val[top_feats])
        else:
            y_proba = clf.predict_proba(X_val[top_feats])
            y_pred = clf.predict(X_val[top_feats])
        
        if mode == "binary":
            y_proba = y_proba[:,1]

        classes_predicted = len(np.unique(y_pred))
        if not test:
            if classes_predicted < n_classes:
                print(
                    f"Fold {fold+1}, {n_feats} features: Only {classes_predicted} classes predicted."
                )
                insufficient_classes_flag = True
                continue  # filter out models that don't predict all three classes
            else:
                insufficient_classes_flag = False

        print(f"Fold {fold+1}, {n_feats} features: Getting report metrics")
        # Report metrics
        excluded_metrics = {}
        for k in EXCLUDE_SCORES:
            excluded_metrics[k] = scorers[k](y_val, y_pred)
        performance_metrics = {}
        for k, m_func in [
            (k, v) for (k, v) in scorers.items() if k not in EXCLUDE_SCORES
        ]:
            performance_metrics[k] = m_func(y_val, y_pred)

        print(f"Fold {fold+1}, {n_feats} features: Bootstrapping metrics")
        # Bootstrapping main metrics
        rng = np.random.RandomState(RANDOM_STATE)  # for reproducibility

        bootstrap_metrics = {}
        for k in performance_metrics:
            bootstrap_metrics[k] = []

        for i in range(n_bootstrap):
            indices = rng.choice(len(y_val), size=len(y_val), replace=True)
            y_true_bs = np.array(y_val)[indices]
            y_pred_bs = np.array(y_pred)[indices]
            for k in bootstrap_metrics:
                bootstrap_metrics[k].append(scorers[k](y_true_bs, y_pred_bs))

        def ci(arr):
            return np.percentile(arr, [2.5, 97.5])

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
        metrics = [m for m in scorers if m not in EXCLUDE_SCORES]
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
summary_df.to_csv(f"{results_outdir}/summary_scores.csv", index=False)
agg_df = (
    summary_df.groupby("n_feats")
    .agg(dict(zip(metrics, ["mean"] * len(metrics))))
    .reset_index()
)

agg_df = agg_df.merge(ci_vals_df, on="n_feats", how="left")

joblib.dump(importance_dfs, f"{results_outdir}/importance_dfs.jl")
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

joblib.dump(agg_df, f"{results_outdir}/agg_dir.jl")
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
            agg_df["n_feats"],
            agg_df[metric],
            "-o",
            label=metric,
        )
    plt.title(f"{LABEL_LOOKUP[subreddit]}", loc="left", fontsize=12)
    plt.xlabel("Number of features", fontsize=11)
    plt.ylabel(metric, fontsize=11)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{results_outdir}/{metric.lower()}_vs_n_feats_bootstrap.png", dpi=300)
    plt.close()


models_to_run = [x for x in range(1,11)]
with open(f"{outdir}/model_features.txt", "w") as f:
    to_write = ",".join([str(x) for x in models_to_run])
    f.writelines(to_write)

model_end = dt.datetime.now()
print(f"Runtime {model_end - model_start}")
model_info["1_runtime"] = model_end - model_start

with pd.ExcelWriter(f"{results_outdir}/baseline_scores.xlsx") as writer:
    pd.DataFrame.from_dict(model_info, orient="index").to_excel(
        writer, sheet_name="model_info", index=True
    )
    summary_df.to_excel(writer, sheet_name="summary_scores", index=False)
    agg_df.to_excel(writer, sheet_name="aggregated_scores", index=False)
    importance_df.to_excel(writer, sheet_name="feat_importance", index=False)

joblib.dump(model_info, f"{results_outdir}/baseline_info.jl")
end = dt.datetime.now()
total_runtime = end - start
print(f"Finished at {end}. Runtime: {total_runtime}.")
