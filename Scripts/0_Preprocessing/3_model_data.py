"""
Builds model-ready train/test matrices: merges metadata/features, applies
author/domain encodings, filters collinearity, and persists X/y splits.

Inputs (CLI):
    --train:       Path to train DataFrame (parquet) (svd_enriched_train_data).
    --test:        Path to test DataFrame (parquet) (svd_enriched_test_data).
    --y-col:       Name of the target column (defaults to "thread_size").
    --rs:          Random seed for any randomized steps.
    --corr:        Absolute Pearson correlation threshold for pruning
                   highly correlated feature pairs (defaults to 0.5).
    --outdir:      Output directory.
    --subreddit:   Used for naming outputs.

Processing:
    1) Optionally add log-transformed target `log_{y_col}` when strictly
       positive (retained as the modeling target if created).
    2) Fit label encoders for `author` and `domain` **on the TRAIN split only**,
       persist the mapping, and apply to both splits as `encoded_author`,
       `encoded_domain`. Unseen categories map to a reserved "unknown" code.
    3) Build X (feature matrix) by excluding target columns and any listed in
       `COLS_TO_REMOVE`. Keep encoded columns if requested by config.
    4) Identify highly correlated pairs in TRAIN (|r| >= `--corr`), and drop
       features by a deterministic priority rule (PRIORITY_ORDER).
    5) Persist X/y for train and test, plus a run log with configuration,
       feature inventory, and the list of removed correlated features.

Outputs:
    {outdir}/label_encoder_maps.jl
    {outdir}/{subreddit}_train_test_data.jl
    {outdir}/3_model_data_log.xlsx  # sheets: params, threads, features, correlated_feats, removed_feats

Notes:
    - Reproducibility: all randomness is seeded with `--rs`.
    - No leakage: encoding and correlation filtering decisions are derived
      from TRAIN only and then applied to TEST.
"""

import sys
import argparse
import os

import datetime as dt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import joblib

import pandas as pd
import numpy as np

COLS_TO_REMOVE = [
    "body",
    "subject",
    "success",
    "thread_id",
    "author",
    "domain",
    "clean_text",
    "domain_type",
]
TARGET_Y_COLS = [
    "timestamp",
    "thread_size",
    "authors",
    "score",
    "thread_depth",
    "direct_reply_count",
    "direct_reply_sentiment_std",
    "mean_direct_reply_length",
]

# Order in which to handle correlated pairs. Lower indices indicate higher priority for keeping column.
PRIORITY_ORDER = [
    "author_freq",
    "domain_freq",
    "encoded_author",
    "encoded_domain",
    "domain_pagerank",
    "avg_word_length",
    "subject_sentiment_score",
    "subject_length",
    "hour",
    "dayofweek",
    "stopword_ratio",
    "verb_ratio",
    "noun_ratio",
    "unique_word_ratio",
    "exclamation_ratio",
    "caps_ratio",
    "question_ratio",
    "is_external_domain",
    "external_domain",
    "includes_image",
    "includes_video",
    "reddit_domain",
    "includes_body",
    "author_unseen",
    "domain_unseen",
]


def log_vals(y: pd.Series):
    if y.min() < 0:
        print(f"[INFO] Values are negative, not taking log.")
        return None
    elif y.min() < 1:
        print(f"[INFO] Min value < 1, taking log(y+1)")
        return np.log(y + 1)
    else:
        print(f"[INFO] Taking log(y)")
        return np.log(y)


def get_high_corr_pairs(df: pd.DataFrame, corr_threshold=0.5):
    corr_matrix = df.corr()
    high_corr_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1) == 1)
        .stack()
        .reset_index()
    )
    high_corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
    high_corr_pairs = high_corr_pairs[
        high_corr_pairs["Correlation"].abs() > corr_threshold
    ]
    return high_corr_pairs


def get_feats_to_remove(row):
    # return lower-priority feature. Doesn't include SVD feats.
    feat_1 = row["Feature 1"]
    feat_2 = row["Feature 2"]
    if PRIORITY_ORDER.index(feat_1) > PRIORITY_ORDER.index(feat_2):
        return feat_1
    else:
        return feat_2


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
        "--train",
        help="Training data filepath (parquet).",
    )

    ap.add_argument(
        "--test",
        help="Test data filepath (parquet).",
    )

    ap.add_argument(
        "--y-col",
        default="thread_size",
        help="Target y column for tuning the MCC and downstream modelling. Defaults to thread_size.",
    )

    ap.add_argument("--rs", default="42", help="Random state, defaults to 42.")

    ap.add_argument(
        "--corr",
        default="0.5",
        help="Correlation threshold above which to remove highly correlated pairs. Default 0.5.",
    )

    args = ap.parse_args()

    args.rs = int(args.rs)
    args.corr = float(args.corr)
    if (args.corr > 1.0) or (args.corr < 0.0):
        raise ValueError(
            f"{args.corr} not a valid correlation threshold. Must be between 0 and 1."
        )

    print(f"[INFO] Args: {args}")

    print(f"[INFO] Loading training and test data")
    data = {"train": pd.read_parquet(args.train), "test": pd.read_parquet(args.test)}

    encoder_maps = {}
    for col in ["author", "domain"]:
        print(f"[INFO] Creating {col} ID label encoder map (train-only)")
        # classes_ is sorted; enumerate gives 0..K-1 -> add +1 so 0 is reserved for unknown
        classes = pd.Index(data["train"][col].astype(str).unique())
        encoder_maps[col] = {
            "label_to_code": {label: i + 1 for i, label in enumerate(classes)},
            "unknown_code": 0,
        }
        print(f"[INFO] {col} frequency encoding")
        freq = data["train"][col].astype(str).value_counts(normalize=True)
        for k, df in data.items():
            df[f"{col}_freq"] = (
                df[col].astype(str).map(freq).fillna(0.0).astype("float32")
            )

    print(f"[INFO] Encoding domain names and author IDs and taking log(y)")
    for k, df in data.items():
        log_y_col = log_vals(df[args.y_col])
        if log_y_col is not None:
            df[f"log_{args.y_col}"] = log_y_col
        for col, colmap in encoder_maps.items():
            m = colmap["label_to_code"]
            unk = colmap["unknown_code"]
            df[f"encoded_{col}"] = df[col].astype(str).map(m).fillna(unk)
            df[f"{col}_unseen"] = df[f"encoded_{col}"] == unk
            df[f"encoded_{col}"] = df[f"encoded_{col}"].astype("category")

    if f"log_{args.y_col}" in data["train"]:
        TARGET_Y_COLS.append(f"log_{args.y_col}")
        y_col = f"log_{args.y_col}"
    else:
        y_col = args.y_col

    # outputs should exclude target cols and encoded cols
    cols_to_include = [
        x
        for x in data["train"].columns
        if ((x not in COLS_TO_REMOVE) & (x not in TARGET_Y_COLS))
    ]
    print(f"[INFO] Cols to include in output: {cols_to_include}")

    x_dfs = {}
    y_dfs = {}
    for i, df in data.items():
        x_dfs[i] = df[cols_to_include]
        y_dfs[i] = df[TARGET_Y_COLS]
        print(f"[INFO] {i} threads: {len(x_dfs[i])}")

    print(f"[INFO] Getting highly correlated pairs")
    high_corr_pairs = get_high_corr_pairs(x_dfs["train"], corr_threshold=args.corr)
    print(f"[INFO] Highly correlated pairs: {high_corr_pairs}")
    # keep features according to priority levels in PRIORITY_ORDER
    to_remove = set(high_corr_pairs.apply(get_feats_to_remove, axis=1).tolist())
    print(f"[INFO] Removing: {to_remove}")
    new_cols = [x for x in x_dfs["train"].columns if x not in to_remove]
    for k, x_df in x_dfs.items():
        x_dfs[k] = x_df[new_cols]

    # Save encoder map
    encoders_outfile = f"{args.outdir}/label_encoder_maps.jl"
    print(f"[INFO] Saving encoders to {encoders_outfile}")
    joblib.dump(encoder_maps, encoders_outfile)

    # also export encoders as flat table for review
    encoder_rows = []
    for col, mapping in encoder_maps.items():
        for label, code in mapping["label_to_code"].items():
            encoder_rows.append({"column": col, "label": label, "code": code})
        # explicitly record the reserved unknown code
        encoder_rows.append(
            {"column": col, "label": "__UNKNOWN__", "code": mapping["unknown_code"]}
        )
    encoders_df = pd.DataFrame(encoder_rows)
    encoders_parquet = os.path.join(args.outdir, "label_encoder_maps.parquet")
    encoders_df.to_parquet(encoders_parquet, index=False)
    print(f"[INFO] Encoder maps saved to Parquet at {encoders_parquet}")

    # save train and test dfs
    for k, x_df in x_dfs.items():
        x_out = f"{args.outdir}/{args.subreddit}_{k}_X.parquet"
        y_out = f"{args.outdir}/{args.subreddit}_{k}_Y.parquet"
        print(f"[INFO] Saving {k} dfs to\n{x_out}\n{y_out}")
        x_df.to_parquet(x_out)
        y_dfs[k].to_parquet(y_out)

    # get train and test data info
    model_info = {
        "non_svd_cols": len([x for x in cols_to_include if "svd" not in x]),
        "svd_cols": len([x for x in cols_to_include if "svd" in x]),
    }
    for k, x in x_dfs.items():
        model_info[f"{k}_threads"] = len(x)

    # for output
    params = {
        "script": str(sys.argv[0]),
        "run_start": start,
        "subreddit": args.subreddit,
        "train_data": args.train,
        "test_data": args.test,
        "y_col": args.y_col,
        "random_state": args.rs,
        "corr_thresh": args.corr,
        "encoders_out": encoders_outfile,
        "outdir": args.outdir,
    }

    # output run data
    run_info = {
        "params": pd.DataFrame.from_dict(params, orient="index", columns=["value"]),
        "threads": pd.DataFrame.from_dict(
            model_info, orient="index", columns=["value"]
        ),
        "features": pd.DataFrame(
            [x for x in new_cols if "svd" not in x], columns=["feature"]
        ),
        "correlated_feats": high_corr_pairs,
        "removed_feats": pd.DataFrame(to_remove, columns=["feature"]),
    }

    logfile = f"{args.outdir}/3_model_data_log.xlsx"
    print(f"[INFO] Saving run info to {logfile}")
    with pd.ExcelWriter(logfile) as writer:
        for key, df in run_info.items():
            df.to_excel(writer, sheet_name=key)

    print(f"[INFO] Finished at {dt.datetime.now()}")


if __name__ == "__main__":
    main()
