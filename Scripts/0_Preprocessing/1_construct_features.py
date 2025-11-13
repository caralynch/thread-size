"""
1_construct_features.py

Preprocessing step 1: construct comment- and thread-level feature tables.

This script loads raw Reddit threads and comments for a single subreddit,
extracts text and structural features, and writes enriched DataFrames for
downstream modeling.

Command-line interface
----------------------
subreddit   : Subreddit key in {'conspiracy','crypto','politics'}.
outdir      : Output directory.
comments    : Path to comments dataframe.
threads     : Path to threads dataframe..

Inputs (expected columns)
-------------------------
Comments dataframe:
    id (str)                      Unique comment id.
    parent (str)                  Id of parent (comment or root).
    thread_id (str)               Thread identifier.
    body (str)                    Comment text.
    body_sentiment_score (float)  Sentiment score (precomputed).
    level (int)                   Depth relative to thread root.

Threads dataframe:
    thread_id (str)               Thread identifier.
    subject (str)                 Thread title text.
    body (str or NaN)             OP body text (if any).
    domain (str)                  Linked domain (if any).
    timestamp (datetime-like)     UTC timestamp.
    thread_size (int)             Total number of comments in the thread.

Engineered features (examples)
------------------------------
Per-comment:
    body_length, direct_reply_count, direct_reply_sentiment_std,
    mean_direct_reply_length, avg_word_length, unique_word_ratio,
    stopword_ratio, noun_ratio, verb_ratio, exclamation_ratio,
    caps_ratio, question_ratio, clean_text

Per-thread:
    includes_body, subject_length, hour, dayofweek, includes_image,
    includes_video, reddit_domain, avg_word_length, unique_word_ratio,
    stopword_ratio, noun_ratio, verb_ratio, exclamation_ratio,
    caps_ratio, question_ratio, reply_count, reply_sentiment_std,
    reply_length_mean, thread_depth, clean_text, log_thread_size

Outputs
-------
Writes two enriched DataFrames to --outdir:
    {subreddit}_comments_extra_feats.parquet
    {subreddit}_threads_extra_feats.parquet

Notes
-----
- Requires NLTK tokenizers and stopwords. If not installed, download
  'punkt', 'averaged_perceptron_tagger' (or 'averaged_perceptron_tagger_eng'),
  and 'stopwords' before running. If you run into LookupError(resource_not_found)
  due to missing NLTK data, run:
  python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('stopwords')"
  Note that NLTK's downloader is silent when running in a python -c, so it may
  look like nothing is happening even when it's working.

- For reproducibility across environments, pin scikit-learn version if
  using OneHotEncoder(sparse_output=False).
"""

import argparse
import sys
import os

import datetime as dt

import numpy as np
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords


# Ensure required NLTK data are available
def ensure_nltk_resources(download: bool = True) -> None:
    """
    Ensure that required NLTK resources are available in the environment.

    Parameters
    ----------
    download : bool, default True
        If True, attempt to download missing resources using nltk.download().
        If False, raise a RuntimeError with a helpful message instead.
    """
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab/english",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
        "stopwords": "corpora/stopwords",
    }

    missing = []
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            missing.append(name)

    if not missing:
        return

    if not download:
        raise RuntimeError(
            "Missing NLTK resources: "
            + ", ".join(missing)
            + ". Please install them, e.g.:\n"
            "    >>> import nltk\n"
            "    >>> nltk.download('punkt'); nltk.download('punkt_tab');\n"
            "    >>> nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords')"
        )

    # Attempt download
    for name in missing:
        nltk.download(name)


ensure_nltk_resources(download=True)

# list of expected subreddits
POSSIBLE_SUBREDDITS = ["conspiracy", "crypto", "politics"]

# text functions and lexicons
STOP_WORDS = set(stopwords.words("english"))

# expected reddit domain strs
DOMAIN_STRS = {
    "i.redd.it": "includes_image",
    "v.redd.it": "includes_video",
    "reddit.com": "reddit_domain",
}


def normalize_domain(d):
    if d.endswith(".reddit.com"):
        return "reddit.com"
    if d == "redd.it":
        return "reddit.com"
    if d.startswith("i.redd.it"):
        return "i.redd.it"  # image host
    if d.startswith("v.redd.it"):
        return "v.redd.it"  # video host
    return d


def remove_links(text: str):
    text = str(text)
    if len(text) == 0:
        return ""
    else:
        # Regex pattern to match URLs
        url_pattern = r"https?://\S+|www\.\S+"
        return re.sub(url_pattern, "", text)


def get_tokens(text: str):
    text = str(text)
    text = text.lower()
    text = remove_links(text)
    tokens = nltk.pos_tag(nltk.word_tokenize(text))
    return tokens


def get_words(text: str):
    tokens = get_tokens(text)
    words = [token[0] for token in tokens if token[0].isalpha()]
    return words


def get_non_stopwords(text: str):
    words = get_words(text)
    non_stopwords = [x for x in words if x not in STOP_WORDS]
    return non_stopwords


def get_word_tokens(text: str):
    tokens = get_tokens(text)
    word_tokens = [token for token in tokens if token[0].isalpha()]
    return word_tokens


def get_nouns(tokens: list):
    nouns = [token[0] for token in tokens if token[1].startswith("NN")]
    return nouns


def get_verbs(tokens: list):
    verbs = [token[0] for token in tokens if token[1].startswith("VB")]
    return verbs


def non_stopword_strings(text: str):
    tokens = get_non_stopwords(text)
    outstr = ""
    for word in tokens:
        outstr += f" {word}"
    return outstr.strip()


def compute_text_features(text: str):
    words = get_words(text)
    if len(words) == 0:
        return pd.Series([0, 0, 0, 0, 0, 0, 0, 0])  # Avoid division by zero

    unique_words = set(words)
    stopword_count = sum(1 for word in words if word.lower() in STOP_WORDS)

    avg_word_length = np.mean([len(word) for word in words]) if words else 0

    unique_word_ratio = len(unique_words) / len(words)

    stopword_ratio = stopword_count / len(words)

    word_tokens = get_word_tokens(text)
    nouns = get_nouns(word_tokens)
    verbs = get_verbs(word_tokens)

    noun_ratio = len(nouns) / len(words)
    verb_ratio = len(verbs) / len(words)

    exclamation_ratio = text.count("!") / len(words)
    question_ratio = text.count("?") / len(words)

    caps_ratio = sum(c.isupper() for c in text) / sum(c.isalpha() for c in text)

    return pd.Series(
        [
            avg_word_length,
            unique_word_ratio,
            stopword_ratio,
            noun_ratio,
            verb_ratio,
            exclamation_ratio,
            question_ratio,
            caps_ratio,
        ]
    )


def classify_domains(domain_str):
    domain_str = normalize_domain(domain_str.lower())
    domain_list = []
    for red_str in DOMAIN_STRS.keys():
        if red_str in domain_str:
            domain_list.append(1)
        else:
            domain_list.append(0)
    return domain_list


def main():
    print(f"{sys.argv[0]}")
    print(f"[INFO] STARTED AT {dt.datetime.now()}")

    # CL arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--subreddit", help="Subreddit")
    ap.add_argument(
        "--outdir",
        help="Output directory.",
    )
    ap.add_argument(
        "--comments",
        help="Comments file.",
    )
    ap.add_argument(
        "--threads",
        help="Threads file.",
    )

    args = ap.parse_args()
    print(f"[INFO] Args: {args}")

    if args.subreddit not in POSSIBLE_SUBREDDITS:
        print(
            f"Subreddit given {args.subreddit} not in possible subreddits: {POSSIBLE_SUBREDDITS}"
        )
        exit()

    # load thread and comment data
    print("[INFO] Loading thread and comment data")

    df_comments = pd.read_parquet(args.comments)
    df_threads = pd.read_parquet(args.threads)

    print("[INFO] Domains")
    # get domain types
    df_threads[list(DOMAIN_STRS.values())] = df_threads.domain.apply(classify_domains)

    print("[INFO] Getting basic text features")
    # basic features
    df_comments["body_length"] = df_comments.body.apply(lambda x: len(str(x)))
    df_threads["includes_body"] = df_threads.body.notna()
    df_threads["subject_length"] = df_threads.subject.apply(lambda x: len(str(x)))

    print("[INFO] Getting more text features")
    # text features
    print("[INFO][Text features] Comments")
    df_comments[
        [
            "avg_word_length",
            "unique_word_ratio",
            "stopword_ratio",
            "noun_ratio",
            "verb_ratio",
            "exclamation_ratio",
            "caps_ratio",
            "question_ratio",
        ]
    ] = df_comments.body.apply(compute_text_features)
    print("[INFO][Text features] Threads")
    df_threads[
        [
            "avg_word_length",
            "unique_word_ratio",
            "stopword_ratio",
            "noun_ratio",
            "verb_ratio",
            "reading_ease",
            "exclamation_ratio",
            "caps_ratio",
            "question_ratio",
        ]
    ] = df_threads.subject.apply(compute_text_features)

    print("[INFO] Getting reply data")
    # replies
    reply_counts = (
        df_comments.groupby("parent")["id"].count().rename("direct_reply_count")
    )
    df_comments = df_comments.merge(
        reply_counts, left_on="id", right_index=True, how="left"
    ).fillna(0)
    reply_sentiments = (
        df_comments.groupby("parent")["body_sentiment_score"]
        .std()
        .rename("direct_reply_sentiment_std")
    )
    df_comments = df_comments.merge(
        reply_sentiments, left_on="id", right_index=True, how="left"
    ).fillna(0)
    reply_lengths = (
        df_comments.groupby("parent")["body_length"]
        .mean()
        .rename("mean_direct_reply_length")
    )
    df_comments = df_comments.merge(
        reply_lengths, left_on="id", right_index=True, how="left"
    ).fillna(0)

    print("[INFO] Merging with thread data")
    # merge with thread data
    df_threads = df_threads.merge(
        reply_counts, left_on="thread_id", right_on="id", how="left"
    ).fillna(0)
    df_threads = df_threads.merge(
        reply_sentiments,
        left_on="thread_id",
        right_on="id",
        right_index=True,
        how="left",
    ).fillna(0)
    df_threads = df_threads.merge(
        reply_lengths, left_on="thread_id", right_on="id", how="left"
    ).fillna(0)

    print("[INFO] Getting thread depth")
    # thread depth
    thread_depth = (
        df_comments.groupby("thread_id")["level"].max().rename("thread_depth")
    )
    df_threads = df_threads.merge(
        thread_depth, left_on="thread_id", right_index=True, how="left"
    ).fillna(0)

    print("[INFO] Time-related features")
    # extract hour and day of week
    df_threads["hour"] = df_threads["timestamp"].dt.hour
    df_threads["dayofweek"] = df_threads["timestamp"].dt.dayofweek

    print("[INFO] Cleaning text")
    # clean subject and body text for TF-IDF vectorizer
    df_threads["clean_text"] = df_threads.subject.apply(non_stopword_strings)
    df_comments["clean_text"] = df_comments.body.apply(non_stopword_strings)

    print("[INFO] Log thread size")
    # take log of thread size
    df_threads["log_thread_size"] = np.log(df_threads["thread_size"])

    print(f"[INFO] Dumping enriched dfs to {args.outdir}")
    os.makedirs(args.outdir, exist_ok=True)

    df_comments.to_parquet(
        f"{args.outdir}/{args.subreddit}_comments_extra_feats.parquet"
    )
    df_threads.to_parquet(f"{args.outdir}/{args.subreddit}_threads_extra_feats.parquet")

    print("[INFO] Finished.")


if __name__ == "__main__":
    main()
