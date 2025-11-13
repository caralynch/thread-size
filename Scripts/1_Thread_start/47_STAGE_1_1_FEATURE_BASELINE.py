# Stage 1.1 - baseline feature scores
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
    fbeta_score,
)
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import sys
import seaborn as sns
import os
import json
import shap
import datetime as dt
from sklearn.model_selection import StratifiedKFold, train_test_split
import gc

RANDOM_STATE = 42

LABEL_LOOKUP = {
    "crypto": "r/CryptoCurrency",
    "politics": "r/politics",
    "conspiracy": "r/Conspiracy",
}

THRESHOLD = np.log(1)
PERFORMANCE_TOLERANCE = 0.95
BETA = 2
scorers = {
    "MCC": matthews_corrcoef,
    "F-beta": partial(fbeta_score, beta=BETA),
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

SCORER_LOOKUP = {"MCC": ["mcc", "m"], "F-beta": ["f", "fbeta", "f-beta", "fb"]}

usage_msg = (
    "Usage: python 47_1_STAGE_1_FEATURE_BASELINE.py <subreddit> <run_number> <collection> <calibration> <max_feats>\n"
    + "Test mode: python 47_1_STAGE_1_FEATURE_BASELINE.py <subreddit> test"
)
if len(sys.argv) < 3:
    print(usage_msg)
    sys.exit(1)


subreddit = sys.argv[1].lower()
if subreddit not in LABEL_LOOKUP:
    print(f"{subreddit} invalid, exiting.")
    exit()

run_number = sys.argv[2]
if "test" in run_number.lower():
    test = True
    n_bootstrap = 20
    max_feats = 5
    cv_splits = 2
    print("### TEST MODE ACTIVATED ###")
else:
    test = False
    try:
        run_number = int(sys.argv[2])
    except:
        run_number = str(sys.argv[2])
    max_feats = sys.argv[5]
    if max_feats != "max":
        max_feats = int(max_feats)
    n_bootstrap = 1000
    cv_splits = 5

calibration = sys.argv[4]
CALIBRATION_LOOKUP = ["cal", "c", "calibration", "calibrate"]
if calibration.lower() in CALIBRATION_LOOKUP:
    CALIBRATION = True
else:
    CALIBRATION = False
print(f"Calibration: {CALIBRATION}")
collection = sys.argv[3]
no_collection = False
if collection.lower() == "0":
    no_collection = True
    print("No-collection models!")
    indir = f"REGDATA/outputs/{subreddit}"
    outdir = f"{indir}/Run_{run_number}"
    data_infile = f"{indir}/0_preprocessing/train_test_data_dict.jl"
else:
    indir = f"REGDATA/outputs/{subreddit}_14"
    outdir = f"{indir}/TwoStageModel/Run_{run_number}"
    data_infile = f"{indir}/train_test_data_dict.jl"

results_outdir = f"{outdir}/stage_1_1_feature_baselines"
os.makedirs(outdir, exist_ok=True)
os.makedirs(results_outdir, exist_ok=True)

print(f"Running feature model baselines for {subreddit}, run {run_number}")

start = dt.datetime.now()
print(f"Started at {start}.")

print(f"Loading data")
# Load data
data_dict = joblib.load(data_infile)
X = data_dict["X_train"] if not test else data_dict["X_train"].head(200)
y_train_data = data_dict["Y_train"] if not test else data_dict["Y_train"].head(200)
if not no_collection:
    y_train_data = y_train_data["log_thread_size"]
y = (y_train_data > np.log(1)).astype(int)
del y_train_data
gc.collect()


# Outer CV
outer_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

if max_feats == "max":
    max_feats = len(X.columns)

if max_feats <= 50:
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

model_info = {
    "stage": 1.1,
    "stage_1_1_date": dt.datetime.today(),
    "1_1_cv_splits": cv_splits,
    "subreddit": subreddit,
    "feature_grid": feature_counts,
    "random_state": RANDOM_STATE,
    "run_number": run_number,
    "max_feats": max_feats,
    "performance_tolerance": PERFORMANCE_TOLERANCE,
    "calibration": CALIBRATION,
}

importance_dfs = []
all_fold_dfs = {}
bootstrap_vals = {}
print("Training data on folds")
for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
    print(f"Fold {fold + 1}/{cv_splits}")
    all_fold_dfs[fold + 1] = {}
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    
    if CALIBRATION:
        print("Splitting x_val and y_val into calibration and eval sets")
        # Split X_val and y_val into calibration and evaluation sets
        X_calib, X_val, y_calib, y_val = train_test_split(
            X_val,
            y_val,
            test_size=0.5,  # 50% for calibration, 50% for evaluation
            stratify=y_val,  # preserve class balance
            random_state=RANDOM_STATE,  # for reproducibility
        )
    print("Getting feature importances for feature ranking")
    # Get feature importances for ranking
    selector = lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    selector.fit(X_train, y_train)
    # Save feature importance from this fold
    # After fitting
    booster = selector.booster_
    split_importance = booster.feature_importance(importance_type="split")
    gain_importance = booster.feature_importance(importance_type="gain")
    feature_names = booster.feature_name()
    print("Scaling feature importances")
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
    print("Getting ranked features")
    ranked_features = pd.Series(combined_importance, index=feature_names)
    ranked_features = ranked_features.sort_values(ascending=False).index.tolist()

    for n_feats in feature_counts:
        top_feats = ranked_features[:n_feats]

        print(f"Fold {fold+1}, {n_feats} features: Training model")
        clf = lgb.LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        clf.fit(X_train[top_feats], y_train)
        if CALIBRATION:
            print(f"Fold {fold+1}, {n_feats} features: Calibrating model")
            calibrated_clf = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
            calibrated_clf.fit(X_calib[top_feats], y_calib)
            y_proba = calibrated_clf.predict_proba(X_val[top_feats])[:, 1]
            y_pred = calibrated_clf.predict(X_val[top_feats])
        else:
            y_proba = clf.predict_proba(X_val[top_feats])[:, 1]
            y_pred = clf.predict(X_val[top_feats])
        if len(np.unique(y_pred)) < 2:
            print(
                f"Not all classes predicted (only {len(np.unique(y_pred))}) - excluding this {n_feats} feature model."
            )
            continue

        print(f"Fold {fold+1}, {n_feats} features: Getting report metrics")
        # Report metrics
        performance_metrics = {"AUC": roc_auc_score(y_val, y_proba)}
        report = scorers["Report"](y_val, y_pred)
        for key, scorer_func in [(k,v) for k,v in scorers.items() if k not in EXCLUDE_SCORES]:
            performance_metrics[key] = scorer_func(y_val, y_pred)

        print(f"Fold {fold+1}, {n_feats} features: Bootstrapping metrics")
        # Bootstrapping main metrics
        rng = np.random.RandomState(RANDOM_STATE)  # for reproducibility

        bootstrap_metrics = {}
        for key in performance_metrics:
            bootstrap_metrics[key] = []

        for i in range(n_bootstrap):
            indices = rng.choice(len(y_val), size=len(y_val), replace=True)
            y_true_bs = np.array(y_val)[indices]
            y_pred_bs = np.array(y_pred)[indices]
            y_proba_bs = np.array(y_proba)[indices]

            bootstrap_metrics["AUC"].append(roc_auc_score(y_true_bs, y_proba_bs))

            for key in performance_metrics:
                if key != "AUC":
                    bootstrap_metrics[key].append(scorers[key](y_true_bs, y_pred_bs))
                

        def ci(arr):
            return np.percentile(arr, [2.5, 97.5])

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
        metrics = list(performance_metrics.keys())
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

importance_df.sort_values("mean_importance", ascending=False, inplace=True)

joblib.dump(agg_df, f"{results_outdir}/aggregated_scores.jl")

for metric in metrics:
    print(f"Plotting {metric}")
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
            agg_df["n_feats"], agg_df[metric], "-o", label=metric,
        )
    plt.title(f"{LABEL_LOOKUP[subreddit]}", loc="left", fontsize=12)
    plt.xlabel("Number of features", fontsize=10)
    plt.ylabel(metric, fontsize=10)
    plt.xticks(fontsize=9,)
    plt.yticks(fontsize=9)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{results_outdir}/{metric.lower()}_vs_n_feats_bootstrap.png", dpi=300)
    plt.close()


def ci_overlap_and_close(row, highest_row, metric):
    if pd.isna(row[f"{metric}_lower"]) or pd.isna(row[f"{metric}_upper"]):
        return False
    return (
        not (
            row[f"{metric}_upper"] < highest_row[f"{metric}_lower"]
            or row[f"{metric}_lower"] > highest_row[f"{metric}_upper"]
        )
        and row[metric] >= highest_row[metric] * PERFORMANCE_TOLERANCE
    )


def detect_elbow_model(agg_df_sorted, metric="MCC"):
    """
    Elbow detection based on 1st-order deltas that:
    - Scans all models,
    - Picks the model with the highest efficiency (gain per feature added),
    - Ensures the performance is close to the best model (≥95% of best MCC)
    """
    # Compute deltas
    # 1st-order deltas
    agg_df_sorted = agg_df.sort_values("n_feats").reset_index(drop=True)
    agg_df_sorted["delta"] = agg_df_sorted[metric].diff()
    agg_df_sorted["delta_next"] = agg_df_sorted[metric].diff().shift(-1)

    # Find best performance
    best_score = agg_df_sorted[metric].max()
    best_row = agg_df_sorted.loc[agg_df_sorted[metric].idxmax()]
    if (
        best_row["n_feats"] == 1
    ):  # if best model has lowest number of features, it is most efficient
        return best_row["n_feats"]

    # Compute efficiency: (performance gain) / (feature count increase)
    agg_df_sorted["gain_per_feature"] = (
        agg_df_sorted["delta"] / agg_df_sorted["n_feats"].diff()
    )

    # Remove NaN/inf rows (first row has NaN delta)
    efficiency_df = agg_df_sorted.dropna(subset=["gain_per_feature"]).copy()

    # remove models where the performance lowers compared to previous model
    efficiency_df = efficiency_df[efficiency_df["delta"] > 0]
    # Only consider models that are at least 95% of the best MCC
    efficiency_df = efficiency_df[
        efficiency_df[metric] >= best_score * PERFORMANCE_TOLERANCE
    ]

    if len(efficiency_df) == 0:
        # fallback: best model
        print("No model meets tolerance threshold. Defaulting to best model.")
        return best_row["n_feats"]

    # Choose model with highest gain per feature added
    elbow_model = efficiency_df.loc[efficiency_df["gain_per_feature"].idxmax()]

    print(
        f"Elbow model: {elbow_model['n_feats']} features with gain per feature {elbow_model['gain_per_feature']:.4f}"
    )

    return elbow_model["n_feats"]

print("Selecting models")
for metric in metrics:
    # get best model
    best_idx = agg_df[metric].idxmax()
    best_score = agg_df.loc[best_idx, metric]
    best_n_feats = agg_df.loc[best_idx, "n_feats"]

    # get CI bounds
    best_row = agg_df.loc[best_idx]

    agg_df[f"{metric}_CI_width"] = agg_df[f"{metric}_upper"] - agg_df[f"{metric}_lower"]
    agg_df[f"{metric}_ci_overlap_best"] = agg_df.apply(
        lambda row: ci_overlap_and_close(row, best_row, metric), axis=1
    )
    overlapping_models = agg_df[agg_df[f"{metric}_ci_overlap_best"]].copy()

    # select model with fewest features
    simple_n_feats = overlapping_models.sort_values("n_feats").iloc[0]["n_feats"]

    # select most reliable model over folds
    robust_n_feats = overlapping_models.sort_values(f"{metric}_CI_width").iloc[0][
        "n_feats"
    ]

    efficient_n_feats = detect_elbow_model(agg_df, metric=metric)
    print(f"{metric} Best model: {best_n_feats} features")
    print(
        f"{metric} Good-enough model: {efficient_n_feats} features (≥95% of best {metric})"
    )
    print(f"{metric} Robust model: {robust_n_feats} features")
    print(f"{metric} Efficient model: {efficient_n_feats} features")

    model_info[f"{metric}_efficient_model"] = efficient_n_feats
    model_info[f"{metric}_best_model"] = best_n_feats
    model_info[f"{metric}_simple_model"] = simple_n_feats
    model_info[f"{metric}_robust_model"] = robust_n_feats


end = dt.datetime.now()
total_runtime = end - start
model_info["total_runtime"] = str(total_runtime)
print(f"Finished at {end}. Runtime: {total_runtime}.")

models_to_run = list(
    set([int(x) for key, x in model_info.items() if key.endswith("_model")])
)
with open(f"{outdir}/model_features.txt", "w") as f:
    to_write = ",".join([str(x) for x in models_to_run])
    f.writelines(to_write)

joblib.dump(model_info, f"{results_outdir}/baseline_info.jl")

with pd.ExcelWriter(f"{results_outdir}/baseline_scores.xlsx") as writer:
    pd.DataFrame.from_dict(model_info, orient="index").to_excel(
        writer, sheet_name="model_info", index=True
    )
    summary_df.to_excel(writer, sheet_name="summary_scores", index=False)
    agg_df.to_excel(writer, sheet_name="aggregated_scores", index=False)
    importance_df.to_excel(writer, sheet_name="feat_importance", index=False)
