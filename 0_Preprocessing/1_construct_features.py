# MIT License
# Copyright (c) 2025 Cara Lynch
# See the LICENSE file for details.
"""
Preprocessing step 1: construct comment- and thread-level feature tables.

This script loads raw Reddit threads and comments for a single subreddit,
extracts text and structural features, and writes enriched DataFrames for
downstream modeling.

Command-line interface
----------------------
--subreddit : Subreddit key in {'conspiracy','crypto','politics'}.
--outdir    : Output directory.
--comments  : Path to comments dataframe.
--threads   : Path to threads dataframe.

Inputs
------
Comments dataframe:
    id, parent, thread_id, body, body_sentiment_score, level

Threads dataframe:
    thread_id, subject, body, domain, timestamp, thread_size

Outputs
-------
Writes two enriched DataFrames to --outdir:
    {subreddit}_comments_extra_feats.parquet
    {subreddit}_threads_extra_feats.parquet
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

TEXT_COLS = ["subject", "body", "domain", "author", "id", "thread_id", "parent"]


def normalize_domain(d):
    """
    Normalize Reddit domain strings to canonical forms.
    
    Parameters
    ----------
    d : str or None
        Raw domain string from Reddit post.
    
    Returns
    -------
    str
        Normalized domain string.
    """
    if pd.isna(d):
        return ""
    d = str(d).lower()
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
    """
    Remove URLs from text using regex pattern matching.
    
    Parameters
    ----------
    text : str
        Input text potentially containing URLs.
    
    Returns
    -------
    str
        Text with all URLs removed.
    """
    text = str(text)
    if len(text) == 0:
        return ""
    else:
        # Regex pattern to match URLs
        url_pattern = r"https?://\S+|www\.\S+"
        return re.sub(url_pattern, "", text)


def get_tokens(text: str):
    """
    Tokenize and POS-tag text using NLTK.
    
    Parameters
    ----------
    text : str
        Input text to tokenize.
    
    Returns
    -------
    list of tuple
        List of (token, POS_tag) tuples.
    """
    text = str(text)
    text = text.lower()
    text = remove_links(text)
    tokens = nltk.pos_tag(nltk.word_tokenize(text))
    return tokens


def get_words(text: str):
    """
    Extract alphabetic word tokens from text.
    
    Parameters
    ----------
    text : str
        Input text.
    
    Returns
    -------
    list of str
        Alphabetic tokens only.
    """
    tokens = get_tokens(text)
    words = [token[0] for token in tokens if token[0].isalpha()]
    return words


def get_non_stopwords(text: str):
    """
    Extract non-stopword tokens from text.
    
    Parameters
    ----------
    text : str
        Input text.
    
    Returns
    -------
    list of str
        Alphabetic tokens excluding English stopwords.
    """
    words = get_words(text)
    non_stopwords = [x for x in words if x not in STOP_WORDS]
    return non_stopwords


def get_word_tokens(text: str):
    """
    Get POS-tagged word tokens (alphabetic only).
    
    Parameters
    ----------
    text : str
        Input text.
    
    Returns
    -------
    list of tuple
        List of (word, POS_tag) tuples for alphabetic tokens.
    """
    tokens = get_tokens(text)
    word_tokens = [token for token in tokens if token[0].isalpha()]
    return word_tokens


def get_nouns(tokens: list):
    """
    Extract noun tokens from POS-tagged tokens.
    
    Parameters
    ----------
    tokens : list of tuple
        POS-tagged tokens from nltk.pos_tag.
    
    Returns
    -------
    list of str
        Tokens with noun POS tags (NN*).
    """
    nouns = [token[0] for token in tokens if token[1].startswith("NN")]
    return nouns


def get_verbs(tokens: list):
    """
    Extract verb tokens from POS-tagged tokens.
    
    Parameters
    ----------
    tokens : list of tuple
        POS-tagged tokens from nltk.pos_tag.
    
    Returns
    -------
    list of str
        Tokens with verb POS tags (VB*).
    """
    verbs = [token[0] for token in tokens if token[1].startswith("VB")]
    return verbs


def non_stopword_strings(text: str):
    """
    Extract non-stopwords as a single space-separated string.
    
    Parameters
    ----------
    text : str
        Input text.
    
    Returns
    -------
    str
        Space-separated non-stopword tokens.
    """
    tokens = get_non_stopwords(text)
    outstr = ""
    for word in tokens:
        outstr += f" {word}"
    return outstr.strip()


def compute_text_features(text: str):
    """
    Compute multiple text-based features from input text.
    
    Parameters
    ----------
    text : str
        Input text to analyze.
    
    Returns
    -------
    pd.Series
        Series containing: avg_word_length, unique_word_ratio, stopword_ratio,
        noun_ratio, verb_ratio, exclamation_ratio, question_ratio, caps_ratio.
    """
    if pd.isna(text):
        return pd.Series([0, 0, 0, 0, 0, 0, 0, 0])
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

    letter_count = sum(c.isalpha() for c in text)
    if letter_count == 0:
        caps_ratio = 0
    else:
        caps_ratio = sum(c.isupper() for c in text) / sum(c.isalpha() for c in text)

    non_ws_chars = len(
        [c for c in text if not c.isspace()]
    )  # non whitespace characters
    exclamation_ratio = 0 if non_ws_chars == 0 else text.count("!") / non_ws_chars
    question_ratio = 0 if non_ws_chars == 0 else text.count("?") / non_ws_chars

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
    """
    Classify domain string into Reddit content type flags.
    
    Parameters
    ----------
    domain_str : str
        Domain string to classify.
    
    Returns
    -------
    list of int
        Binary flags: [includes_image, includes_video, reddit_domain, is_external_domain].
    """
    domain_norm = normalize_domain(domain_str)

    # flags for image/video/reddit-hosted
    flags = []
    for red_str in DOMAIN_STRS.keys():
        flags.append(1 if red_str in domain_norm else 0)

    # new flag: external domain
    is_external = 1 if all(flag == 0 for flag in flags) and domain_norm != "" else 0

    flags.append(is_external)
    return flags


def main():
    """
    Main entry point for feature construction pipeline.
    
    Loads raw Reddit data, extracts text and structural features,
    and writes enriched DataFrames for downstream modeling.
    """
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

    # set types correctly
    for df in [df_comments, df_threads]:
        for col in set(TEXT_COLS) & set(df.columns):
            df[col] = df[col].astype("string")

    print("[INFO] Domains")
    # get domain types
    domain_flags = df_threads["domain"].apply(classify_domains).tolist()
    df_threads[list(DOMAIN_STRS.values()) + ["is_external_domain"]] = domain_flags

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
    )
    df_comments["direct_reply_count"] = df_comments["direct_reply_count"].fillna(0)

    reply_sentiments = (
        df_comments.groupby("parent")["body_sentiment_score"]
        .std()
        .rename("direct_reply_sentiment_std")
    )
    df_comments = df_comments.merge(
        reply_sentiments, left_on="id", right_index=True, how="left"
    )
    df_comments["direct_reply_sentiment_std"] = df_comments[
        "direct_reply_sentiment_std"
    ].fillna(0)

    reply_lengths = (
        df_comments.groupby("parent")["body_length"]
        .mean()
        .rename("mean_direct_reply_length")
    )
    df_comments = df_comments.merge(
        reply_lengths, left_on="id", right_index=True, how="left"
    )
    df_comments["mean_direct_reply_length"] = df_comments[
        "mean_direct_reply_length"
    ].fillna(0)

    print("[INFO] Merging with thread data")
    # merge with thread data
    df_threads = df_threads.merge(
        reply_counts, left_on="thread_id", right_index=True, how="left"
    )
    df_threads["direct_reply_count"] = df_threads["direct_reply_count"].fillna(0)

    df_threads = df_threads.merge(
        reply_sentiments, left_on="thread_id", right_index=True, how="left"
    )
    df_threads["direct_reply_sentiment_std"] = df_threads[
        "direct_reply_sentiment_std"
    ].fillna(0)

    df_threads = df_threads.merge(
        reply_lengths, left_on="thread_id", right_index=True, how="left"
    )
    df_threads["mean_direct_reply_length"] = df_threads[
        "mean_direct_reply_length"
    ].fillna(0)

    print("[INFO] Getting thread depth")
    thread_depth = (
        df_comments.groupby("thread_id")["level"].max().rename("thread_depth")
    )
    df_threads = df_threads.merge(
        thread_depth, left_on="thread_id", right_index=True, how="left"
    )
    df_threads["thread_depth"] = df_threads["thread_depth"].fillna(0)

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
