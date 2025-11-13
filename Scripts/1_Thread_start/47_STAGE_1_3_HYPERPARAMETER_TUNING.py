import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import matthews_corrcoef, fbeta_score
import joblib
import numpy as np
import datetime as dt
import json
import os
import sys
import pandas as pd
import gc
from functools import partial

LABEL_LOOKUP = {
    "crypto": "r/CryptoCurrency",
    "politics": "r/politics",
    "conspiracy": "r/Conspiracy",
}

THRESHOLD = np.log(1)

usage_msg = (
    "Usage: python 47_3_STAGE_1_HYPERPARAMETER_TUNING.py <subreddit> <run_number> <collection>(optional <n_feat mods>)\n"
    + "Test mode: python 47_3_STAGE_1_HYPERPARAMETER_TUNING.py <subreddit> test"
)

if len(sys.argv) < 3:
    print(usage_msg)
    sys.exit(1)

subreddit = sys.argv[1].lower()
if subreddit not in LABEL_LOOKUP:
    raise ValueError(f"{subreddit} invalid, exiting.")

run_number = sys.argv[2]
if "test" in run_number.lower():
    test = True
    feature_counts = []
else:
    test = False
    try:
        run_number = int(sys.argv[2])
    except:
        run_number = str(sys.argv[2])
    if len(sys.argv) > 4:
        feature_counts = [int(x) for x in sys.argv[4:]]
    else:
        feature_counts = []
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

os.makedirs(outdir, exist_ok=True)


def get_mcc_fixed(thresh):
    def scorer(estimator, X, y):
        proba = estimator.predict_proba(X)[:, 1]
        preds = (proba >= thresh).astype(int)
        return matthews_corrcoef(y, preds)

    return scorer


print(f"Running hyperparameter tuning for {subreddit}, run {run_number}")

start = dt.datetime.now()
print(f"Started at {start}.")

print(f"Loading data")
# Load data
data_dict = joblib.load(data_infile)
X_train = data_dict["X_train"] if not test else data_dict["X_train"].head(200)
X_test = data_dict["X_test"] if not test else data_dict["X_test"].head(100)

y_train_data = data_dict["Y_train"] if not test else data_dict["Y_train"].head(200)
if not no_collection:
    y_train_data = y_train_data["log_thread_size"]
y_train_binary = (y_train_data > np.log(1)).astype(int)

del y_train_data
gc.collect()

# Load selected features and parameters
if os.path.isfile(f"{outdir}/selected_model_params.json"):
    print(f"Loading model params from {outdir}/selected_model_params.json")
    model_params = json.load(f"{outdir}/selected_model_params.json")
    model_info = model_params["info"]
    params = model_params["params"]
else:
    print(f"Loading model params from {outdir}/stage1_model_param_dict.p")
    model_params = joblib.load(f"{outdir}/stage1_model_param_dict.jl")
    model_info = model_params["info"]
    params = model_params["params"]

model_info["hyperparameter_tuning_date"] = dt.date.today()
random_state = model_info["random_state"]
calibration = model_info["calibration"]
scorer = model_info["scorer"]
if "beta" in model_info:
    beta = model_info["beta"]
else:
    beta = 2

scorers = {
    "MCC": matthews_corrcoef,
    "F-beta": partial(fbeta_score, beta=beta),
}
CLASS_NAMES = {
    0: "Stalled",
    1: "Started",
}
cv_splits = 2 if test else 5
n_trials = 10 if test else 100

model_info["hyperparam_cv_splits"] = cv_splits
model_info["hyperparam_n_trials"] = n_trials

if len(feature_counts) < 1:
    feature_counts = list(params.keys())
    print("Getting feature counts from params...")
    if test:
        feature_counts = [feature_counts[0]]
model_info["feature_counts"] = feature_counts
print(f"Feature counts: {feature_counts}")

foldwise_best_params = {}
foldwise_best_scores = {}
outer_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train_binary)):
    print(f"Outer CV fold {fold + 1}/{cv_splits}")
    for n_feats in feature_counts:
        if n_feats not in foldwise_best_params:
            foldwise_best_params[n_feats] = []
            foldwise_best_scores[n_feats] = []
        config = params[n_feats]
        print(f"Model features: {n_feats}")
        x_cols = config["features"]
        cw = config["final_class_weights"]
        thresh = config["final_threshold"]
        tuned_params = {}

        print(f"Number of features selected: {n_feats}")
        print(f"Selected features: {x_cols}")
        print(f"Class weights: {cw}, Threshold: {thresh}")

        # Load training data
        X, X_val = X_train[x_cols].iloc[train_idx], X_train[x_cols].iloc[val_idx]
        y, y_val = y_train_binary.loc[train_idx], y_train_binary.loc[val_idx]

        if calibration:
            # Split X_val and y_val into calibration and evaluation sets
            X_calib, X_eval, y_calib, y_eval = train_test_split(
                X_val,
                y_val,
                test_size=0.5,  # 50% for calibration, 50% for evaluation
                stratify=y_val,  # preserve class balance
                random_state=random_state,  # for reproducibility
            )
        else:
            X_eval, y_eval = X_val, y_val

        def objective(trial):
            param = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "random_state": random_state,
                "verbosity": -1,
                "class_weight": cw,  # from STAGE_1_TUNING
                "num_leaves": trial.suggest_int("num_leaves", 20, 25 if test else 150),
                "max_depth": trial.suggest_int("max_depth", 3, 6 if test else 15),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.1 if test else 1e-3, 0.2, log=True
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 5, 10 if test else 100
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            }

            clf = lgb.LGBMClassifier(**param)
            clf.fit(X, y)
            if calibration:
                calibrated_clf = CalibratedClassifierCV(
                    clf, method="isotonic", cv="prefit"
                )
                calibrated_clf.fit(X_calib, y_calib)
                proba = calibrated_clf.predict_proba(X_eval)[:, 1]
            else:
                proba = clf.predict_proba(X_val)[:, 1]

            preds = (proba >= thresh).astype(int)
            if not test:
                if len(np.unique(preds)) < 2:
                    raise optuna.TrialPruned("Only one class predicted, skipping trial.")

            score = scorers[scorer](y_eval, preds)

            return score

    print(f"Starting hyperparameter tuning for {n_feats}")

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    foldwise_best_params[n_feats].append(study.best_params)
    foldwise_best_scores[n_feats].append(study.best_value)

    print(f"Fold {fold+1}: Best {scorer} for {n_feats} feats: {study.best_value}")
    joblib.dump(
        study.best_params, f"{outdir}/{n_feats}_feats_best_hyperparams_fold_{fold+1}.jl"
    )


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


for n_feats in feature_counts:
    config = params[n_feats]
    config["best_hyperparams"] = aggregate_params(foldwise_best_params[n_feats])
    config[f"best_{scorer}"] = np.mean(foldwise_best_scores[n_feats])
    joblib.dump(
        config["best_hyperparams"], f"{outdir}/{n_feats}_feats_best_hyperparams.jl"
    )
    print(
        f"Best hyperparameters for {n_feats} features: {config['best_hyperparams']}, "
        f"Mean {scorer}: {config[f'best_{scorer}']}"
    )


end = dt.datetime.now()
model_info["hyperparameter_tuning_runtime"] = str(end - start)
new_model_params = {"info": model_info, "params": params}

joblib.dump(new_model_params, f"{outdir}/params_post_hyperparam_tuning.jl")

print(f"Finished at {end}.")
print(f"Runtime {end-start}.")
