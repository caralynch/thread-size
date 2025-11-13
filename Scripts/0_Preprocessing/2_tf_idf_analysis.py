"""
TF-IDF + SVD model selection for text features using Optuna.

This script tunes a TF-IDF vectorizer and TruncatedSVD projection using
Stratified K-fold cross-validation and a LightGBM probe model that optimizes Matthews Correlation Coefficient (MCC) on a binary target derived from `y_col`. It enforces a minimum explained variance by SVD and persists the selected text models and SVD features for downstream stages.

Inputs (CLI):
    subreddit:        Subreddit identifier (used only in paths/metadata).
    outdir:           Output directory for artifacts.
    comments:         Path to comments DataFrame (joblib .jl or pickle).
    threads:          Path to thread-level DataFrame (joblib .jl or pickle).
    --text-col:       Name of text column in thread DataFrame (default: "clean_text").
    --y-col:          Name of thread target column (default: "thread_size").
    --train-split:    Fraction for chronological split into train/test (default: 0.8).
    --rs:             Random seed (default: 42).
    --test:           If set, runs with fewer CV splits/trials.
    --splits:         CV splits (default: 5, or 2 in test).
    --trials:         Optuna trials (default: 100, or 10 in test).

Procedure:
    1) Chronologically split threads into train/test using `train_split`.
    2) Build a binary label y = 1[thread_size > 1] on TRAIN ONLY.
    3) Optuna searches over TF-IDF hyperparameters (max_features, min_df,
       ngram_range, text source: posts vs posts+comments). SVD n_components
       is chosen per-trial with a constraint on explained variance.
    4) For each trial: fit TF-IDF on the chosen training text, transform
       training posts, fit SVD, then evaluate a LightGBM classifier with
       Stratified K-fold CV to obtain mean MCC/F1.
    5) Persist the best TF-IDF and SVD models, and save SVD feature matrices
       for train and test, plus merged train/test DataFrames with SVD columns.

Outputs:
    {outdir}/{subreddit}_tf_idf_model_params.csv
    {outdir}/{subreddit}_tf_idf_optuna_study.jl
    {outdir}/{subreddit}_best_trial.csv
    {outdir}/{subreddit}_optuna_tfidf_vectorizer.jl
    {outdir}/{subreddit}_optuna_svd_model.jl
    {outdir}/{subreddit}_svd_train_df.parquet
    {outdir}/{subreddit}_svd_test_df.parquet
    {outdir}/{subreddit}_svd_enriched_train_data.parquet
    {outdir}/{subreddit}_svd_enriched_test_data.parquet

Notes:
    - Reproducibility: seed all randomized steps with `--rs`.
    - No leakage: TF-IDF is fit on training text only; test is transformed
      with the persisted vectorizer/SVD.
"""

import os
import sys
import argparse

import datetime as dt
import gc
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import joblib

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, f1_score
import lightgbm as lgb
import optuna


# optuna features to tune
INT_FEATURES = {
    "max_features": (50, 1000),
    "min_df": (5, 20),
    "n_components": (80, 400)
}
CATEGORICAL_FEATURES = {
    "ngram_range": [(1, 1), (1, 2)],
    "text_data": ["posts", "all"]
}

def main():
    print(f"{sys.argv[0]}")
    start = dt.datetime.now()
    print(f"[INFO] STARTED AT {start}")

    # CL arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("--subreddit", help="Subreddit")
    ap.add_argument(
        "--outdir",
        help="Output directory.",
    )
    ap.add_argument(
        "--comments",
        help="Comments filepath.",
    )
    ap.add_argument(
        "--threads",
        default=None,
        help="Threads filepath.",
    )
    ap.add_argument(
        "--text-col",
        default="clean_text",
        help="Text column in dfs, defaults to clean_text",
    )
    ap.add_argument("--y-col", default="thread_size", help="Target y column for tuning the MCC and downstream modelling. Defaults to thread_size.")

    ap.add_argument("--train-split", default="0.8", help="Train-test split. Defaults to 0.8.")

    ap.add_argument("--rs", default="42", help="Random state, defaults to 42.")

    ap.add_argument("-t", "--test", action='store_true', help="Run the script in test mode")

    ap.add_argument("--splits", default=None, help="Number of CV splits. Defaults to 5, or 2 in test mode.")
    ap.add_argument("--trials", default=None, help="Number of Optuna trials. Defaults to 100 or 10 in test mode.")

    args = ap.parse_args()

    args.rs = int(args.rs)
    args.train_split = float(args.train_split)

    subreddit = args.subreddit
    
    print(f"[INFO] {subreddit}")

    test = False
    if args.test:
        test = True # test mode engaged
        print("[INFO] TEST MODE ENGAGED")
    
    if args.splits is None:
        args.splits = 5 if not test else 2
    if args.trials is None:
        args.trials = 100 if not test else 10
    
    print(f"[INFO] Args: {args}")

    print(f"[INFO] Reading threads file: {args.threads}")
    df_threads = pd.read_parquet(args.threads)
    df_comments = pd.read_parquet(args.comments)
    os.makedirs(args.outdir, exist_ok=True)

    # in test mode only look at small subset of data
    if test:
        df_threads = df_threads.head(1000)
        df_comments = df_comments[df_comments.thread_id.isin(df_threads.thread_id)]

    # make sure thread data sorted by date
    df_threads = df_threads.sort_values(by="timestamp").reset_index(drop=True)
    # need to separate into train and test data
    print("[INFO] Separating into train and test data")
    split_index = int(len(df_threads) * args.train_split)
    threads = {
        "train": df_threads.iloc[:split_index].copy(),
        "test": df_threads.iloc[split_index:].copy(),
    }
    data_info = {}
    for k, df in threads.items():
        print(f"[INFO] {k} threads: {len(df)}")
        data_info[f"{k}_threads"] = len(df)
    
    comments_train = df_comments[df_comments.thread_id.isin(threads['train'].thread_id)]
    all_training_data = list(threads['train'].clean_text) + list(comments_train.clean_text)
    posts_training_data = list(threads['train'].clean_text)
    y = (threads['train'][args.y_col] > 1).astype(
        int
    )  # binary target based on thread size

    # for output
    model_params = {
        "script": str(sys.argv[0]),
        "date": start,
        **vars(args),
        **INT_FEATURES,
        **CATEGORICAL_FEATURES,
    }

    pd.DataFrame.from_dict(model_params, orient="index").to_csv(f"{args.outdir}/{args.subreddit}_tf_idf_model_params.csv")

    # objective function for Optuna
    def objective(trial):
        params = {}
        for key, values in INT_FEATURES.items():
            if key != "n_components":
                params[key] = trial.suggest_int(key, values[0], values[1])
        for key, values in CATEGORICAL_FEATURES.items():
            params[key] = trial.suggest_categorical(key, values)
        try:
            tfidf = TfidfVectorizer(
                max_features=params['max_features'],
                ngram_range=params['ngram_range'],
                min_df=params['min_df'],
                stop_words="english",
            )
            if params['text_data'] == "posts":
                text_training_data = posts_training_data
            else:
                text_training_data = all_training_data
            tfidf.fit(text_training_data)

            X_tfidf = tfidf.transform(threads['train'][args.text_col])
        except ValueError:
            raise optuna.TrialPruned("[WARNING] After pruning, no terms remain - need a lower min_df or higher max_df")
        key = "n_components"
        if X_tfidf.shape[1] <= INT_FEATURES[key][0]:
            raise optuna.TrialPruned("[WARNING] Not enough TF-IDF features for SVD")
        n_comp = trial.suggest_int(key, INT_FEATURES[key][0], min(INT_FEATURES[key][1], X_tfidf.shape[1]))
        if X_tfidf.shape[1] < n_comp:
            raise optuna.TrialPruned("[WARNING] Not enough TF-IDF features for SVD")
        svd = TruncatedSVD(n_components=n_comp, random_state=args.rs)
        X_svd = svd.fit_transform(X_tfidf)

        explained_var = svd.explained_variance_ratio_.sum()
        if explained_var < 0.6:
            raise optuna.TrialPruned("[WARNING] Explained variance too low")

        mccs, f1s = [], []
        skf = StratifiedKFold(
            n_splits=args.splits, shuffle=True, random_state=args.rs
        )
        for train_idx, val_idx in skf.split(X_svd, y):
            X_tr, X_val = X_svd[train_idx], X_svd[val_idx]
            y_tr, y_val = np.array(y.iloc[train_idx]), np.array(y.iloc[val_idx])

            model = lgb.LGBMClassifier(
                objective="binary", random_state=args.rs, verbose=-1
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            mccs.append(matthews_corrcoef(y_val, preds))
            f1s.append(f1_score(y_val, preds))

        trial.set_user_attr("explained_variance", explained_var)
        trial.set_user_attr("mean_mcc", np.mean(mccs))
        trial.set_user_attr("mean_f1", np.mean(f1s))
        del X_tfidf, X_svd, svd, tfidf
        gc.collect()
        return np.mean(mccs)

    # Run Optuna optimization
    print("[INFO] Starting Optuna optimization")
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.rs)
    )
    study.optimize(objective, n_trials=args.trials)

    print("[INFO] Optuna optimization complete")

    study_out = f"{args.outdir}/{subreddit}_tf_idf_optuna_study.jl"
    print(f"[INFO] Saving study to {study_out}.")
    joblib.dump(study, study_out)

    # Save results
    print("[INFO] Saving best trial and study results")
    best_trial = study.best_trial
    results_df = pd.DataFrame(
        [
            {
                **t.params,
                "value": t.value,
                "explained_variance": t.user_attrs.get("explained_variance", np.nan),
                "mean_f1": t.user_attrs.get("mean_f1", np.nan),
                "mean_mcc": t.user_attrs.get("mean_mcc", np.nan),
            }
            for t in study.trials
            if t.value is not None
        ]
    )

    results_df.to_csv(f"{args.outdir}/{args.subreddit}_optuna_results.csv")

    print("Best trial:")
    print(best_trial.params)
    print("MCC:", best_trial.value)
    print("Explained variance:", best_trial.user_attrs.get("explained_variance"))

    best_trial_df = pd.DataFrame.from_dict(
        {
            **best_trial.params,
            "value": best_trial.value,
            "explained_variance": best_trial.user_attrs.get(
                "explained_variance", np.nan
            ),
            "mean_f1": best_trial.user_attrs.get("mean_f1", np.nan),
            "mean_mcc": best_trial.user_attrs.get("mean_mcc", np.nan),
            "train_data": len(threads['train']),
            "test_data": len(threads['test']),
        },
        orient="index",
    )
    best_trial_df.to_csv(f"{args.outdir}/{args.subreddit}_best_trial.csv")
    print("[INFO] Saving best trial model and data")

    n_comp = best_trial.params["n_components"]
    tfidf = TfidfVectorizer(
        max_features=best_trial.params["max_features"],
        ngram_range=best_trial.params["ngram_range"],
        min_df=best_trial.params["min_df"],
        stop_words="english",
    )
    if best_trial.params["text_data"] == "posts":
        text_training_data = posts_training_data
    else:
        text_training_data = all_training_data
    tfidf.fit(text_training_data)

    tfidf_dfs = {}
    for k, df in threads.items():
        tfidf_dfs[k] = tfidf.transform(df[args.text_col])

    print("[INFO] Saving vectorizer and svd model")
    # output vectorizer and svd model
    joblib.dump(tfidf, f"{args.outdir}/{args.subreddit}_optuna_tfidf_vectorizer.jl")

    svd = TruncatedSVD(n_components=n_comp, random_state=args.rs)
    svd_matrices = {"train": svd.fit_transform(tfidf_dfs['train'])}
    svd_matrices['test'] = svd.transform(tfidf_dfs['test'])
    joblib.dump(svd, f"{args.outdir}/{args.subreddit}_optuna_svd_model.jl")
    
    # Convert SVD matrices to df
    print("[INFO] Converting SVD matrices to dfs")
    svd_dfs = {}
    for k, m in svd_matrices.items():
        svd_dfs[k] = pd.DataFrame(m, columns=[f"svd_{i}" for i in range(n_comp)])

    print("[INFO] Adding SVD data to train and test dfs")
    # Concat with original dfs
    model_data = {}
    for k, df in threads.items():
        model_data[k] = pd.concat([df.reset_index(drop=True), svd_dfs[k]], axis=1)

    print("[INFO] Saving train and test dfs")
    # dump train and test data
    for k, df in model_data.items():
        df.to_parquet(f"{args.outdir}/{args.subreddit}_svd_enriched_{k}_data.parquet")
        svd_dfs[k].to_parquet(f"{args.outdir}/{args.subreddit}_svd_{k}_df.parquet")

    print("Finished")



if __name__ == "__main__":
    main()