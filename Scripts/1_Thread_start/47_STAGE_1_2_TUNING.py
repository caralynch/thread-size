import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, matthews_corrcoef, fbeta_score
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import json
import datetime as dt
import seaborn as sns
import matplotlib
from functools import partial

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import optuna
import optuna.visualization as vis
import joblib
import gc
from scipy.optimize import minimize_scalar
from sklearn.utils import compute_class_weight

RANDOM_STATE = 42

LABEL_LOOKUP = {
    "crypto": "r/CryptoCurrency",
    "politics": "r/politics",
    "conspiracy": "r/Conspiracy",
}


usage_msg = (
    "Usage: python 47_STAGE_1_2_TUNING.py <subreddit> <run_number> <collection> <calibration> <scoring_metric> <features>\n"
    + "Test mode: python 47_STAGE_1_2_TUNING.py <subreddit> test\n"
)

BETA = 2
scorers = {
    "MCC": matthews_corrcoef,
    "F-beta": partial(fbeta_score, beta=BETA),
    "F1": f1_score,
}
CLASS_NAMES = {
    0: "Stalled",
    1: "Started",
}

SCORER_LOOKUP = ["f", "fb", "fbeta", "f-beta"]

if len(sys.argv) < 2:
    print(usage_msg)
    sys.exit(1)

subreddit = sys.argv[1].lower()
run_number = sys.argv[2]
if "test" in run_number.lower():
    test = True
    feature_counts = []
    print("Test mode engaged!")
    cv_splits = 2
    n_trials = 10
else:
    if len(sys.argv) < 3:
        print(usage_msg)
        sys.exit(1)
    test = False
    cv_splits = 5
    try:
        run_number = int(sys.argv[2])
    except:
        run_number = str(sys.argv[2])
    if len(sys.argv) > 6:
        feature_counts = [int(x) for x in sys.argv[5:]]
    else:
        feature_counts = []
    n_trials = 300

calibration = sys.argv[4]
CALIBRATION_LOOKUP = ["cal", "c", "calibration", "calibrate"]
if calibration.lower() in CALIBRATION_LOOKUP:
    CALIBRATION = True
else:
    CALIBRATION = False
print(f"Calibration: {CALIBRATION}")

if len(sys.argv) > 5:
    scorer = sys.argv[5].lower()
    if scorer in SCORER_LOOKUP:
        scorer = "F-beta"
    else:
        scorer = "MCC"
else:
    scorer = "MCC"

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
results_outdir = f"{outdir}/stage_1_2_tuning"
os.makedirs(outdir, exist_ok=True)
os.makedirs(results_outdir, exist_ok=True)

model_feats_file = f"{outdir}/model_features.txt"
if os.path.exists(model_feats_file) and len(feature_counts) < 1:
    print(f"Getting feature counts from baseline file info")
    with open(model_feats_file, "r") as f:
        model_feats = f.read()
    model_feats = [int(x) for x in model_feats.split(",")]
    if len(feature_counts) < 1:
        feature_counts = model_feats
        if test:
            feature_counts = feature_counts[0]
if len(feature_counts) < 1:
    feature_counts = list(range(1,21))

if len(feature_counts) < 1:
    print("No features specified. Try again.")
    sys.exit()

print(f"Tuning models with {feature_counts} features...")
start = dt.datetime.now()
print(f"Starting at {start}")
print(f"Subreddit: {subreddit}")
print(f"Features: {feature_counts}")
print("Loading data")
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

model_info = {
    "stage": 1.2,
    "stage_1_2_date": dt.datetime.today(),
    "tuning_method": "optuna",
    "tuning_cv_splits": cv_splits,
    "tuning_n_trials": n_trials,
    "subreddit": subreddit,
    "cw_ratio_range": (0.2, 5.0) if not test else (1.0, 1.0),
    "threshold_range": (0.2, 0.9) if not test else (0.4, 0.6),
    "feature_grid": feature_counts,
    "random_state": RANDOM_STATE,
    "run_number": run_number,
    "calibration": CALIBRATION,
    "scorer": scorer,
}
if scorer == "F-beta":
    model_info["beta"] = BETA

importance_dfs = []
foldwise_best_params = {}
foldwise_best_scores = {}
foldwise_best_cws = {}
all_folds_df_dict = {}
all_configs = []

print("Training data on folds")
for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
    all_folds_df_dict[fold + 1] = {}
    print(f"Outer fold {fold + 1}/{cv_splits}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    if CALIBRATION:
        # Split X_val and y_val into calibration and evaluation sets
        X_calib, X_val, y_calib, y_val = train_test_split(
            X_val,
            y_val,
            test_size=0.5,  # 50% for calibration, 50% for evaluation
            stratify=y_val,  # preserve class balance
            random_state=RANDOM_STATE,  # for reproducibility
        )

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
    ranked_features = pd.Series(combined_importance, index=feature_names)
    ranked_features = ranked_features.sort_values(ascending=False).index.tolist()

    for n_feats in feature_counts:
        if n_feats not in foldwise_best_params:
            foldwise_best_params[n_feats] = []
            foldwise_best_scores[n_feats] = []
            foldwise_best_cws[n_feats] = []
        print(f"n_feats: {n_feats}")
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
            
            cw_values = compute_class_weight(suggest_cws, classes=np.unique(y_train), y=y_train)
            
            class_weight = {i: w for i, w in enumerate(cw_values)}

            clf = lgb.LGBMClassifier(
                objective="binary",
                class_weight=class_weight,
                random_state=RANDOM_STATE,
                verbose=-1,
            )
            clf.fit(X_train[top_feats], y_train)

            if CALIBRATION:
                calibrated_clf = CalibratedClassifierCV(
                    clf, method="isotonic", cv="prefit"
                )
                calibrated_clf.fit(X_calib[top_feats], y_calib)
                proba = calibrated_clf.predict_proba(X_val[top_feats])[:, 1]
                preds = calibrated_clf.predict(X_val[top_feats])

            else:
                proba = clf.predict_proba(X_val[top_feats])[:, 1]
                preds = clf.predict(X_val[top_feats])

            #if len(np.unique(preds)) < 2:
                #raise optuna.TrialPruned()  # avoids evaluating degenerate models

            scoring_metrics = {}
            for key, score_func in scorers.items():
                scoring_metrics[key] = score_func(y_val, preds)
                trial.set_user_attr(key, scoring_metrics[key])

            trial.set_user_attr("cw_type", cw_type)
            #trial.set_user_attr("cw_ratio", cw_ratio)
            trial.set_user_attr("cw", class_weight)
            trial.set_user_attr("n_feats", n_feats)
            trial.set_user_attr("features", top_feats)

            return scoring_metrics[
                scorer
            ]  # adjust this to prioritize F1 or MCC if needed

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )
        study.optimize(objective, n_trials=n_trials)

        foldwise_best_params[n_feats].append(study.best_params)
        foldwise_best_scores[n_feats].append(study.best_value)
        foldwise_best_cws[n_feats].append(study.best_trial.user_attrs["cw"])

        fig = vis.plot_optimization_history(study)
        fig.write_image(
            f"{results_outdir}/optuna_fold{fold+1}_{n_feats}_feats_convergence.png"
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
    importance_merged[f"mean_{imp_colname}"] = importance_merged[imp_cols].mean(axis=1)
    importance_merged[f"std_{imp_colname}"] = importance_merged[imp_cols].std(axis=1)
# Sort by average importance
importance_merged.sort_values("mean_importance", ascending=False, inplace=True)


all_configs_df = pd.concat(all_configs, ignore_index=True)
all_configs_df["cw_str"] = all_configs_df["cw"].astype(str)


ranked_features = importance_merged["feature"].tolist()
optuna_params = {}
class_weights = {}
for n_feats in feature_counts:
    best_params = aggregate_params(foldwise_best_params[n_feats])
    optuna_params[n_feats] = {
        "best_params": best_params,
        f"best_{scorer}": np.mean(foldwise_best_scores[n_feats]),
    }
    class_weights[n_feats] = aggregate_params(foldwise_best_cws[n_feats])

joblib.dump(optuna_params, f"{results_outdir}/optuna_params.jl")


def neg_scorer(threshold, y_proba, y_true):
    y_pred = (y_proba >= threshold).astype(int)
    return -scorers[scorer](y_true, y_pred)


foldwise_thresholds = {}
foldwise_score_thresholds = {}
print("Training data on folds")
for fold, (cal_idx, val_idx) in enumerate(outer_cv.split(X, y)):
    print(f"Outer fold {fold + 1}/{cv_splits}")
    X_train, X_val = X.iloc[cal_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[cal_idx], y.iloc[val_idx]

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
            foldwise_score_thresholds[n_feats] = []
        top_feats = ranked_features[:n_feats]
        best_params = optuna_params[n_feats]["best_params"]
        class_weight = class_weights[n_feats]

        clf = lgb.LGBMClassifier(
            objective="binary",
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        clf.fit(X_train[top_feats], y_train)

        if CALIBRATION:
            calibrated_clf = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
            calibrated_clf.fit(X_calib[top_feats], y_calib)
            proba = calibrated_clf.predict_proba(X_thresh_cal[top_feats])[:, 1]

        else:
            proba = clf.predict_proba(X_thresh_cal[top_feats])[:, 1]

        result = minimize_scalar(
            neg_scorer, bounds=(0, 1), method="bounded", args=(proba, y_thresh_cal)
        )
        best_thresh = result.x
        foldwise_thresholds[n_feats].append(best_thresh)

        proba = (
            calibrated_clf.predict_proba(X_val[top_feats])[:, 1]
            if CALIBRATION
            else clf.predict_proba(X_val[top_feats])[:, 1]
        )
        preds = (proba >= best_thresh).astype(int)
        foldwise_score_thresholds[n_feats].append(scorers[scorer](y_val, preds))

params = {}
for n_feats in feature_counts:
    best_params = optuna_params[n_feats]["best_params"]
    class_weight = class_weights[n_feats]
    params[n_feats] = {
        "n_feats": n_feats,
        f"{scorer}_before_thresh": np.mean(foldwise_best_scores[n_feats]),
        "final_threshold": np.mean(foldwise_thresholds[n_feats]),
        f"{scorer}_after_thresh": np.mean(foldwise_score_thresholds[n_feats]),
        "final_class_weights": class_weight,
        "features": ranked_features[:n_feats],
    }


for n_feats in params:
    for key, value in params[n_feats].items():
        print(f"{n_feats} {key}: {value}")


end = dt.datetime.now()

model_info["tuning_runtime"] = str(end - start)

model_params = {
    "info": model_info,
    "params": params,
}

joblib.dump(model_params, f"{outdir}/stage1_model_param_dict.jl")

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

with pd.ExcelWriter(f"{results_outdir}/stage1_tuning_outputs.xlsx") as writer:
    pd.DataFrame.from_dict(model_info, orient="index").to_excel(
        writer, sheet_name="model_info", index=True
    )
    pd.DataFrame.from_dict(params, orient="index").to_excel(
        writer, sheet_name="params", index=True
    )

    importance_merged.to_excel(writer, sheet_name="feature_importances", index=False)
    pd.DataFrame(flat_params).to_excel(
        writer, sheet_name="flattened_features", index=False
    )

    all_configs_df.to_excel(writer, sheet_name=f"all_configs", index=False)

print(f"Saved all outputs to: {results_outdir}")
print(f"Finished. Total runtime {end-start}.")
