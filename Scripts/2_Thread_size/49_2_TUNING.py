import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
from sklearn.utils import compute_class_weight
import sys
import os
import json
import datetime as dt
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gc
from functools import reduce
from collections import Counter
import optuna
import optuna.visualization as vis
from scipy.optimize import minimize
from functools import partial

RANDOM_STATE = 42

LABEL_LOOKUP = {
    "crypto": "r/CryptoCurrency",
    "politics": "r/politics",
    "conspiracy": "r/Conspiracy",
}


usage_msg = (
    "Usage: python 47_STAGE_2_2_TUNING.py <subreddit> <run_number> <collection> <calibration> <n_classes>n"
    + "Test mode: python 47_STAGE_2_2_TUNING.py <subreddit> test <collection>"
)


if len(sys.argv) < 3:
    print(usage_msg)
    sys.exit(1)

subreddit = sys.argv[1].lower()
if subreddit not in LABEL_LOOKUP:
    print(f"{subreddit} invalid subreddit, exiting.")
    exit()

print(f"Input entered: {sys.argv}")

run_number = sys.argv[2]
collection = sys.argv[3]
n_classes = int(sys.argv[5])
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
    "F1": partial(f1_score, average="macro"),
}

calibration = sys.argv[4]
CALIBRATION_LOOKUP = ["cal", "c", "calibration", "calibrate"]
if calibration.lower() in CALIBRATION_LOOKUP:
    CALIBRATION = True
else:
    CALIBRATION = False
print(f"Calibration: {CALIBRATION}")
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
    print("⚠ Running in TEST mode — limited grid, CV and trials for fast debugging.")
    test = True
    cv_splits = 2  # fewer splits for faster testing
    n_trials = 10
    feature_counts = []
else:
    try:
        run_number = int(sys.argv[2])
    except:
        run_number = str(sys.argv[2])
    feature_counts = []
    test = False
    cv_splits = 5
    n_trials = 300


def check_bin_balance(y_bins):
    counts = np.bincount(y_bins)
    proportions = counts / counts.sum()
    return proportions


start = dt.datetime.now()
print(f"Starting at {start}")
print("Loading data")

# Load data
print(f"Loading {data_infile}")
data_dict = joblib.load(data_infile)
X = data_dict["X_train"] if not test else data_dict["X_train"].head(200)

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
    "stage": 2,
    "n_classes": n_classes,
    "stage_2_date": dt.date.today(),
    "run_number": run_number,
    "tuning_type": "optuna",
    "tuning_trials": n_trials,
    "tuning_cv_splits": cv_splits,
    "cw_range": (1, 10) if not test else (1, 1),
    "threshold_range": (0.1, 0.9) if not test else (0.4, 0.6),
    "random_state": random_state,
    "calibration": CALIBRATION,
}


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


bounds = [(0.0, 1.0)] * n_classes
initial_guess = [0.5] * n_classes


model_start = dt.datetime.now()
results_outdir = f"{outdir}/2_tuning"
os.makedirs(outdir, exist_ok=True)
os.makedirs(results_outdir, exist_ok=True)


print("Defining bins")
min_y = min(y_full.min(), y_test.min())
max_y = max(y_full.max(), y_test.max())

bin_edges = [np.log(1) - 1e-3]
bin_edges.append(np.log(2) - 1e-3)
bin_edges.extend(
    [y_full[y_full > np.log(1)].quantile(x) for x in CLASS_BIN_EDGES[n_classes]]
)

bin_edges.append(y_full.max() + 1e-3)
assert (
    len(bin_edges) == n_classes + 1
), f"bin_edges must have length n_classes+1 ({n_classes+1}), got {len(bin_edges)}."

model_info["fixed_bins"] = bin_edges
model_info["bin_counts"] = dict(
    pd.cut(y_full, bins=bin_edges, labels=CLASS_NAMES[n_classes]).value_counts()
)

model_feats_file = f"{outdir}/model_features.txt"
if os.path.exists(model_feats_file) and len(feature_counts) < 1:
    print(f"Getting feature counts from baseline file info")
    with open(model_feats_file, "r") as f:
        model_feats = f.read()
    model_feats = [int(x) for x in model_feats.split(",")]
    if len(feature_counts) < 1:
        feature_counts = model_feats
else:
    feature_counts = list(range(1,21))

if test:
    feature_counts = [feature_counts[0]]
model_info["feature_counts"] = feature_counts
print(f"Feature counts: {feature_counts}")

# K-fold validation setup
print("Discretizing target for cross-validation folds...")
y_bins_for_cv = pd.cut(y_full, bins=bin_edges, labels=False)
unique_bins = np.unique(y_bins_for_cv.dropna())
assert len(unique_bins) == n_classes, (
    f"pd.cut produced {len(unique_bins)} classes, expected {n_classes}. "
    f"Unique bins found: {unique_bins}"
)
outer_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

all_outer_results = []
importance_dfs = {}
all_folds_df_dict = {}
foldwise_best_params = {}
foldwise_best_mccs = {}
all_configs = []
foldwise_best_cws = {}

print("Starting fold cross-validation")
for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y_bins_for_cv)):
    fold_start_time = dt.datetime.now()
    all_folds_df_dict[fold + 1] = {}
    print(f"Outer fold {fold + 1}/{cv_splits} starting at {dt.datetime.now()}")
    # get training and validation data for this fold
    print(f"Fold {fold+1}: Getting fold training and validation data")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]

    y_train, y_val = y_bins_for_cv.iloc[train_idx], y_bins_for_cv.iloc[val_idx]
    if CALIBRATION:
        X_calib, X_val, y_calib, y_val = train_test_split(
            X_val,
            y_val,
            test_size=0.5,
            stratify=y_val,
            random_state=RANDOM_STATE,
        )
    print(f"Fold {fold+1}: Precomputing ranked features for candidate bins")
    selector = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
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

    for n_feats in feature_counts:
        if n_feats not in foldwise_best_params:
            foldwise_best_params[n_feats] = []
            foldwise_best_mccs[n_feats] = []
            foldwise_best_cws[n_feats] = []
        print(f"Features: {n_feats}")

        def objective(trial, ranked_features):
            bins = bin_edges

            # bin y values
            y_train_bins = y_train
            y_val_bins = y_val
            if not test:
                if (
                    y_train_bins.nunique() < n_classes
                    or y_val_bins.nunique() < n_classes
                ):
                    raise optuna.exceptions.TrialPruned(
                        f"Not enough unique bins in target: train bins: {y_train_bins.nunique()}, val bins: {y_val_bins.nunique()}"
                    )
                train_props = check_bin_balance(y_train_bins)
                val_props = check_bin_balance(y_val_bins)
                # Prune if any class is < 10%
                if any(train_props < 0.10):
                    raise optuna.exceptions.TrialPruned(
                        "Train bin config too imbalanced"
                    )
                if any(val_props < 0.10):
                    raise optuna.exceptions.TrialPruned("Val bin config too imbalanced")

            top_feats = ranked_features[:n_feats]

            # hyperparams
            cw_type = trial.suggest_categorical("cw_type", ["balanced", "custom"])
            if cw_type == "balanced":
                suggest_cws = "balanced"
            else:
                suggest_cws = {}
                for i in range(n_classes):
                    suggest_cws[i] = trial.suggest_int(
                        f"cw_class_{i}",
                        model_info["cw_range"][0],
                        model_info["cw_range"][1],
                    )
                
            
            cw_values = compute_class_weight(suggest_cws, classes=np.unique(y_train_bins), y=y_train_bins)
            
            class_weight = {i: w for i, w in enumerate(cw_values)}

            # train classifier
            clf = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=n_classes,
                class_weight=class_weight,
                random_state=RANDOM_STATE,
                verbosity=-1,
            )
            clf.fit(X_train[top_feats], y_train_bins)
            if CALIBRATION:
                calibrated_clf = CalibratedClassifierCV(
                    clf, method="isotonic", cv="prefit"
                )
                calibrated_clf.fit(X_calib[top_feats], y_calib)
                proba = calibrated_clf.predict_proba(X_val[top_feats])
                preds = calibrated_clf.predict(X_val[top_feats])
            else:
                proba = clf.predict_proba(X_val[top_feats])
                preds = clf.predict(X_val[top_feats])
            # if not test:
            #     if len(set(preds)) < n_classes:
            #         raise optuna.exceptions.TrialPruned(
            #             "Not enough unique predictions in validation set"
            #         )

            performance_metrics = {}
            for k, score_f in scorers.items():
                performance_metrics[k] = score_f(y_val_bins, preds)

            # track trial info
            trial.set_user_attr("n_feats", n_feats)
            trial.set_user_attr("features", top_feats)
            for k, v in performance_metrics.items():
                trial.set_user_attr(k, v)
            trial.set_user_attr("cw_type", cw_type)
            trial.set_user_attr("cw", class_weight)
            trial.set_user_attr("cw_raw", cw_values)
            return performance_metrics["MCC"]

        print(f"Fold {fold+1}, {n_feats} feats: Starting Optuna trials")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(
            lambda trial: objective(trial, ranked_features),
            n_trials=n_trials,
            n_jobs=-1,
        )

        foldwise_best_params[n_feats].append(study.best_params)
        foldwise_best_mccs[n_feats].append(study.best_value)
        foldwise_best_cws[n_feats].append(study.best_trial.user_attrs["cw"])
        

        fig = vis.plot_optimization_history(study)
        fig.write_image(f"{results_outdir}/optuna_fold{fold+1}_convergence.png")

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
        all_outer_results.append(fold_trials)
        all_folds_df_dict[fold + 1][n_feats] = fold_results_df
        all_configs.append(fold_results_df)

    fold_runtime = dt.datetime.now() - fold_start_time
    print(f"Fold {fold+1} runtime: {fold_runtime}")
    print(f"Estimated runtime left: {fold_runtime*(cv_splits-fold+1)}")

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
    importance_merged[f"mean_{imp_colname}"] = importance_merged[imp_cols].mean(axis=1)
    importance_merged[f"std_{imp_colname}"] = importance_merged[imp_cols].std(axis=1)
# sort by average importance
importance_merged.sort_values(by="mean_importance", ascending=False, inplace=True)

ranked_features = importance_merged["feature"].tolist()

string_cols = {"threshold": "final_threshold"}

model_configs_dfs = {}
params = {}
feature_freq_dfs = {}
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
for n_feats in feature_counts:
    best_params = aggregate_params(foldwise_best_params[n_feats])
    optuna_params[n_feats] = {
        "best_params": best_params,
        "best_MCC": np.mean(foldwise_best_mccs[n_feats]),
    }
    class_weights[n_feats] = aggregate_params(foldwise_best_cws[n_feats])


joblib.dump(optuna_params, f"{results_outdir}/optuna_params.jl")

foldwise_thresholds = {}
foldwise_mcc_thresholds = {}

print("Training data on folds")
for fold, (cal_idx, val_idx) in enumerate(outer_cv.split(X, y_bins_for_cv)):
    print(f"Outer fold {fold + 1}/{cv_splits}")
    X_train, X_val = X.iloc[cal_idx], X.iloc[val_idx]
    y_train, y_val = y_bins_for_cv.iloc[cal_idx], y_bins_for_cv.iloc[val_idx]

    if CALIBRATION:
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train, y_train, train_size=0.8, stratify=y_train
        )

    X_thresh_cal, X_val, y_thresh_cal, y_val = train_test_split(
        X_val, y_val, train_size=0.5, stratify=y_val
    )

    for n_feats in feature_counts:
        if n_feats not in foldwise_thresholds:
            foldwise_thresholds[n_feats] = []
            foldwise_mcc_thresholds[n_feats] = []
        top_feats = ranked_features[:n_feats]
        best_params = optuna_params[n_feats]["best_params"]
        class_weight = class_weights[n_feats]

        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
        clf.fit(X_train[top_feats], y_train)

        if CALIBRATION:
            calibrated_clf = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
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
            if CALIBRATION
            else clf.predict_proba(X_val[top_feats])
        )
        preds = get_preds(best_thresholds, proba)
        foldwise_mcc_thresholds[n_feats].append(matthews_corrcoef(y_val, preds))

for n_feats in feature_counts:
    best_params = optuna_params[n_feats]["best_params"]
    class_weight = class_weights[n_feats]
    threshold_vals = []
    threshold_stds = []
    for i in range(n_classes):
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

for n_feats in params:
    for key, value in params[n_feats].items():
        print(f"{n_feats} {key}: {value}")

model_end = dt.datetime.now()
runtime_minutes = round((model_end - model_start).total_seconds() / 60, 2)
print(f"model runtime minutes: {runtime_minutes}")

model_info["stage_2_runtime"] = str(model_end - model_start)

model_params = {"info": model_info, "params": params}

# Save joblib
joblib.dump(model_params, f"{outdir}/tuning_param_dict.jl")
all_configs_df.to_csv(f"{results_outdir}/tuning_all_cv_results.csv", index=False)

joblib.dump(all_configs_df, f"{results_outdir}/all_configs_df.jl")

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

with pd.ExcelWriter(f"{results_outdir}/tuning_outputs.xlsx") as writer:
    pd.DataFrame.from_dict(model_info, orient="index").to_excel(
        writer, sheet_name="model_info", index=True
    )
    pd.DataFrame.from_dict(params, orient="index").to_excel(
        writer, sheet_name="params", index=True
    )

    all_configs_df.to_excel(writer, sheet_name="all_configs")
    importance_merged.to_excel(writer, sheet_name="feature_importances", index=False)
    pd.DataFrame(flat_params).to_excel(
        writer, sheet_name="flattened_features", index=False
    )

print(f"Saved all outputs to: {results_outdir}")
final_call = dt.datetime.now()
print(f"Finished at {final_call}")
print(f"Total time taken: {final_call - start}")
