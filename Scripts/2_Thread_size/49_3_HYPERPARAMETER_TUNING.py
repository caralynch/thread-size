import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import matthews_corrcoef, f1_score
import joblib
import numpy as np
import datetime as dt
import json
import os
import sys
import pandas as pd
from functools import partial

LABEL_LOOKUP = {
    "crypto": "r/CryptoCurrency",
    "politics": "r/politics",
    "conspiracy": "r/Conspiracy",
}


usage_msg = (
    "Usage: python 47_STAGE_2_3_HYPERPARAMETER_TUNING.py <subreddit> <run_number> <collection> \n"
    + "Test mode: python 47_STAGE_2_3_HYPERPARAMETER_TUNING.py <subreddit> test"
)

CLASS_NAMES = {
    3: ["Stalled", "Small", "Large"],
    4: ["Stalled", "Small", "Medium", "Large"],
}
CLASS_BIN_EDGES = {
    3: [0.5],
    4: [1 / 3, 2 / 3],
}


if len(sys.argv) < 4:
    print(usage_msg)
    sys.exit(1)

subreddit = sys.argv[1].lower()
if subreddit not in LABEL_LOOKUP:
    raise ValueError(f"{subreddit} invalid, exiting.")

run_number = sys.argv[2]
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

if "test" in run_number.lower():
    print("⚠ Running in TEST mode")
    test = True
    feature_counts = []
    cv_splits = 2  # fewer splits for faster testing
    n_trials = 5
else:
    test = False
    try:
        run_number = int(sys.argv[2])
    except:
        run_number = str(sys.argv[2])
    test = False
    cv_splits = 5
    n_trials = 300


print(f"Running hyperparameter tuning for {subreddit}, run {run_number}")
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
    "stage": 3,
    "stage_3_date": dt.date.today(),
    "run_number": run_number,
    "hyperparameter_tuning_type": "optuna",
    "hyperparameter_tuning_trials": n_trials,
    "hyperparameter_tuning_cv_splits": cv_splits,
    "random_state": random_state,
}


def get_mcc_fixed(thresh):
    def scorer(estimator, X, y):
        proba = estimator.predict_proba(X)
        preds = []
        for row in proba:
            passed = [row[i] if row[i] >= thresh[i] else -1 for i in range(3)]
            preds.append(np.argmax(passed) if max(passed) != -1 else np.argmax(row))
        return matthews_corrcoef(y, preds)

    return scorer


print("Iterating through the stage 1 models")

model_start = dt.datetime.now()
results_outdir = f"{outdir}/3_hyperparameter_tuning"
os.makedirs(outdir, exist_ok=True)
os.makedirs(results_outdir, exist_ok=True)

tuning_params_infile = joblib.load(f"{outdir}/tuning_param_dict.jl")
n_classes = tuning_params_infile["info"]["n_classes"]
calibration = tuning_params_infile["info"]["calibration"]
tuning_params = tuning_params_infile["params"]
feature_counts = list(tuning_params.keys())
print(f"Feature counts: {feature_counts}")
model_info["feature_counts"] = feature_counts
model_info["calibration"] = calibration
model_info["n_classes"] = n_classes

min_y = min(y_full)
max_y = max(y_full)

outer_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

first_bins = tuning_params[feature_counts[0]]["bins"]
y_bins = pd.cut(y_full, bins=first_bins, labels=False, include_lowest=True)
foldwise_best_params = {}
foldwise_best_mccs = {}
for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_full, y_bins)):
    print(f"Outer CV fold {fold + 1}/{cv_splits}")
    print("Going through stage2 feature models")
    for n_feats in feature_counts:
        if n_feats not in foldwise_best_params:
            foldwise_best_params[n_feats] = []
            foldwise_best_mccs[n_feats] = []
        print(f"n_feats: {n_feats}")
        config = tuning_params[n_feats]
        x_cols = config["features"]

        cw = config["final_class_weights"]
        thresh = config["final_threshold"]
        config["final_n_features"] = n_feats
        bins = config["bins"]
        tuned_params = {}
        print(f"Number of features selected: {n_feats}")
        print(f"Selected features: {x_cols}")
        print(f"Class weights: {cw}, Threshold: {thresh}")
        X, X_val = X_full[x_cols].iloc[train_idx], X_full[x_cols].iloc[val_idx]
        y_bins = pd.cut(y_full, bins=bins, labels=False, include_lowest=True)

        y, y_val = y_bins.iloc[train_idx], y_bins.iloc[val_idx]
        if calibration:
            X_calib, X_val, y_calib, y_val = train_test_split(
                X_val,
                y_val,
                test_size=0.5,
                stratify=y_val,
                random_state=random_state,
            )
        y_val_bins = y_val

        def objective(trial):
            param = {
                "objective": "multiclass",
                "num_class": n_classes,
                "metric": "multi_logloss",
                "class_weight": cw,  # from STAGE_2_TUNING
            }

            param.update(
                {
                    "boosting_type": "gbdt",
                    "random_state": random_state,
                    "verbosity": -1,
                    "num_leaves": trial.suggest_int(
                        "num_leaves", 20, 25 if test else 150
                    ),
                    "max_depth": trial.suggest_int("max_depth", 3, 6 if test else 15),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.1 if test else 1e-3, 0.2, log=True
                    ),
                    "min_child_samples": trial.suggest_int(
                        "min_child_samples", 5, 10 if test else 100
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
            clf.fit(X, y)
            if calibration:
                calibrated_clf = CalibratedClassifierCV(
                    clf, method="isotonic", cv="prefit"
                )
                calibrated_clf.fit(X_calib, y_calib)
                proba = calibrated_clf.predict_proba(X_val)
            else:
                proba = clf.predict_proba(X_val)
            preds = []
            for row in proba:
                passed = [
                    row[i] if row[i] >= thresh[i] else -1 for i in range(n_classes)
                ]
                preds.append(np.argmax(passed) if max(passed) != -1 else np.argmax(row))

            mcc = matthews_corrcoef(y_val_bins, preds)
            return mcc

        print(f"Starting hyperparameter tuning for {n_feats}")

        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        foldwise_best_params[n_feats].append(study.best_params)
        foldwise_best_mccs[n_feats].append(study.best_value)

        print(f"Best MCC for {n_feats} feats: {study.best_value}")
        # joblib.dump(study.best_params, f"{outdir}/{n_feats}_feats_best_hyperparams.jl")

        model_end = dt.datetime.now()
        model_info["hyperparameter_tuning_runtime"] = str(model_end - model_start)


# Aggregate hyperparameters across folds
def aggregate_params(param_list):
    df = pd.DataFrame(param_list)
    agg = {}
    for col in df.columns:
        if df[col].dtype.kind in "iO":
            agg[col] = df[col].mode().iloc[0]
        else:
            agg[col] = df[col].mean()
    return agg


for n_feats in feature_counts:
    config = tuning_params[n_feats]
    config["best_hyperparams"] = aggregate_params(foldwise_best_params[n_feats])
    config["best_MCC"] = np.mean(foldwise_best_mccs[n_feats])
    joblib.dump(
        config["best_hyperparams"], f"{outdir}/{n_feats}_feats_best_hyperparams.jl"
    )
    print(
        f"Best hyperparameters for {n_feats} features: {config['best_hyperparams']}, "
        f"Mean MCC: {config['best_MCC']}"
    )

new_model_params = {"info": model_info, "params": tuning_params}

joblib.dump(new_model_params, f"{outdir}/params_post_hyperparam_tuning.jl")

with pd.ExcelWriter(f"{results_outdir}/outputs.xlsx") as writer:
    pd.DataFrame.from_dict(model_info, orient="index").to_excel(
        writer, sheet_name="model_info", index=True
    )
    for n_feats, params_dict in tuning_params.items():
        pd.DataFrame.from_dict(
            params_dict["best_hyperparams"], orient="index", columns=["Value"]
        ).to_excel(writer, sheet_name=f"mod_{n_feats}_hyperparams", index=True)
        params_copy = params_dict.copy()
        params_copy.pop("best_hyperparams")
        params_df = pd.DataFrame(list(params_copy.items()), columns=["Key", "Value"])

        params_df.to_excel(writer, sheet_name=f"mod_{n_feats}_params", index=False)

end = dt.datetime.now()
print(f"Finished at {end}.")
print(f"Runtime {end-start}.")
