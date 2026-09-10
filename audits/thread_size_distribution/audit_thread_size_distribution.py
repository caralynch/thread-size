#!/usr/bin/env python3
"""Provenance-controlled audit of final Study 1 observed thread sizes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import openpyxl
import pandas as pd

from render_ccdf import render


AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[2]
OUT = AUDIT / "outputs"
FIG = AUDIT / "figures"
SUBS = ("conspiracy", "crypto", "politics")
NAMES = {
    "conspiracy": "r/Conspiracy",
    "crypto": "r/CryptoCurrency",
    "politics": "r/politics",
}
SELECTED_MODEL = {"conspiracy": 3, "crypto": 2, "politics": 3}
CUTS_T = {
    "conspiracy": (1, 6, 19),
    "crypto": (1, 10, 32),
    "politics": (1, 9, 30),
}
CLASS_NAMES = ("Stalled", "Small", "Medium", "Large")
PARTITIONS = ("full", "training", "held_out")
QUANTILES = {
    "q1": 0.25,
    "median": 0.50,
    "q3": 0.75,
    "p90": 0.90,
    "p95": 0.95,
    "p99": 0.99,
}
KEY_OUTPUTS = (
    "summary_statistics.csv",
    "ccdf.csv",
    "population_class_counts.csv",
    "self_comment_checks.csv",
    "evidence_matrix.csv",
    "validation_checks.csv",
    "source_inventory.csv",
    "run_record.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Replace outputs in this audit package only.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def check(rows: list[dict], subreddit: str, category: str, name: str, passed: bool, detail: str) -> None:
    rows.append(
        {
            "subreddit": NAMES.get(subreddit, subreddit),
            "category": category,
            "check": name,
            "status": "PASS" if bool(passed) else "FAIL",
            "detail": detail,
        }
    )


def stats(values: pd.Series, prefix: str) -> dict:
    values = pd.Series(values, dtype="float64")
    quantiles = values.quantile(list(QUANTILES.values()), interpolation="linear")
    result = {
        f"{prefix}_min": float(values.min()),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_max": float(values.max()),
    }
    for (label, _), value in zip(QUANTILES.items(), quantiles):
        result[f"{prefix}_{label}"] = float(value)
    return result


def class_labels(t: pd.Series, cuts: tuple[int, int, int]) -> np.ndarray:
    values = t.to_numpy()
    return np.select([values <= cuts[0], values <= cuts[1], values <= cuts[2]], [0, 1, 2], default=3).astype(int)


def workbook_counts(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="class_sizes")
    return frame[["Class", "Range", "Train", "Test"]].copy()


def log_counts(path: Path) -> tuple[int, int]:
    frame = pd.read_excel(path, sheet_name="threads", index_col=0)
    return int(frame.loc["train_threads", "value"]), int(frame.loc["test_threads", "value"])


def publication_true_ratios(path: Path, subreddit: str, partition: str) -> np.ndarray:
    sheet_prefix = "oof" if partition == "training" else "test"
    frame = pd.read_excel(path, sheet_name=f"{sheet_prefix}_{subreddit}")
    class_column = next((column for column in frame.columns if str(column).startswith("Unnamed:")), None)
    if class_column is None:
        raise ValueError(f"No class-index column found in {path}::{sheet_prefix}_{subreddit}")
    return frame.sort_values(class_column)["true"].to_numpy(float)


def git_commit() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT / "Scripts"), "rev-parse", "HEAD"], text=True).strip()


def fmt_num(value: float, decimals: int = 2) -> str:
    if math.isclose(value, round(value), rel_tol=0, abs_tol=1e-12):
        return f"{int(round(value)):,}"
    return f"{value:,.{decimals}f}"


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(c) for c in frame.columns]
    rows = [[str(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def compact_stat_table(summary: pd.DataFrame, basis: str, prefix: str) -> pd.DataFrame:
    frame = summary[summary.population_basis == basis].copy()
    frame["Population"] = frame["subreddit"] + " — " + frame["partition"].replace({"held_out": "held-out"})
    if basis == "started_roots":
        lead = ["Population", "eligible_roots", "stalled_n", "stalled_pct", "started_n"]
    else:
        lead = ["Population", "eligible_roots", "stalled_n", "stalled_pct", "started_n"]
    cols = [f"{prefix}_min", f"{prefix}_q1", f"{prefix}_median", f"{prefix}_q3", f"{prefix}_p90", f"{prefix}_p95", f"{prefix}_p99", f"{prefix}_mean", f"{prefix}_max"]
    out = frame[lead + cols].copy()
    out.columns = ["Population", "N", "Stalled n", "Stalled %", "Started n", "Min", "Q1", "Median", "Q3", "P90", "P95", "P99", "Mean", "Max"]
    out["N"] = out["N"].map(lambda x: f"{int(x):,}")
    out["Stalled n"] = out["Stalled n"].map(lambda x: f"{int(x):,}")
    out["Started n"] = out["Started n"].map(lambda x: f"{int(x):,}")
    out["Stalled %"] = out["Stalled %"].map(lambda x: f"{x:.2f}")
    for col in ["Min", "Q1", "Median", "Q3", "P90", "P95", "P99", "Mean", "Max"]:
        out[col] = out[col].map(fmt_num)
    return out


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    existing = [OUT / name for name in KEY_OUTPUTS if (OUT / name).exists()]
    if existing and not args.force:
        raise SystemExit("Audit outputs already exist; rerun with --force to replace this package only.")

    validation: list[dict] = []
    evidence: list[dict] = []
    inventory_paths: dict[Path, tuple[str, str]] = {}
    summary_rows: list[dict] = []
    ccdf_rows: list[dict] = []
    class_rows: list[dict] = []
    self_rows: list[dict] = []
    population_frames: dict[str, pd.DataFrame] = {}

    code_sources = [
        ROOT / "Scripts/0_Preprocessing/1_construct_features.py",
        ROOT / "Scripts/0_Preprocessing/2_tf_idf_analysis.py",
        ROOT / "Scripts/0_Preprocessing/3_model_data.py",
        ROOT / "Scripts/2_Thread_size/2_tuning.py",
        ROOT / "Scripts/2_Thread_size/3_hyperparameter_tuning.py",
        ROOT / "Scripts/2_Thread_size/4_run_tuned_model.py",
        ROOT / "Scripts/audits/probability_calibration/outputs/chronological_partitions.csv",
        ROOT / "Scripts/audits/probability_calibration/outputs/validation_checks.csv",
        Path(__file__).resolve(),
        AUDIT / "render_ccdf.py",
    ]
    for path in code_sources:
        inventory_paths[path] = ("code or prior validation evidence", "Scripts Git commit")

    publication_ratios_path = ROOT / "Publication_Outputs/2_Thread_Size/cms/predicted_class_ratios.xlsx"
    inventory_paths[publication_ratios_path] = ("thesis-facing true class-distribution ratios", "frozen modelling output")

    for sub in SUBS:
        name = NAMES[sub]
        raw_threads_path = ROOT / f"Inputs/{sub}_threads.parquet"
        raw_comments_path = ROOT / f"Inputs/{sub}_comments.parquet"
        enriched_threads_path = ROOT / f"Outputs/0_preprocessing/{sub}/{sub}_threads_extra_feats.parquet"
        train_y_path = ROOT / f"Outputs/0_preprocessing/{sub}/{sub}_train_Y.parquet"
        test_y_path = ROOT / f"Outputs/0_preprocessing/{sub}/{sub}_test_Y.parquet"
        train_map_path = ROOT / f"Outputs/0_preprocessing/{sub}/tf-idf/{sub}_svd_enriched_train_data.parquet"
        test_map_path = ROOT / f"Outputs/0_preprocessing/{sub}/tf-idf/{sub}_svd_enriched_test_data.parquet"
        model_log_path = ROOT / f"Outputs/0_preprocessing/{sub}/3_model_data_log.xlsx"
        selected_book_path = ROOT / f"Outputs/2_thread_size/{sub}/4_model/model_{SELECTED_MODEL[sub]}/test_data_results.xlsx"
        params_path = ROOT / f"Outputs/2_thread_size/{sub}/3_h_tuning/params_post_hyperparam_tuning.jl"
        run_log_paths = [
            ROOT / f"Outputs/0_preprocessing/{sub}/logs/{sub}_0_1_construct_features.out",
            ROOT / f"Outputs/0_preprocessing/{sub}/logs/{sub}_0_2_tfidf_analysis.out",
            ROOT / f"Outputs/0_preprocessing/{sub}/logs/{sub}_0_3_model_data.out",
        ]
        for path, role in [
            (raw_threads_path, "raw eligible root records"),
            (raw_comments_path, "raw retained descendant-comment records"),
            (enriched_threads_path, "feature-construction root output"),
            (train_y_path, "final training outcomes"),
            (test_y_path, "final chronological held-out outcomes"),
            (train_map_path, "training row-to-thread mapping"),
            (test_map_path, "held-out row-to-thread mapping"),
            (model_log_path, "final preprocessing population table"),
            (selected_book_path, "selected-model class-distribution table"),
            (params_path, "frozen training-derived class boundaries"),
            *[(path, "executed preprocessing log") for path in run_log_paths],
        ]:
            inventory_paths[path] = (role, "frozen input/output; not tracked by Scripts Git")

        raw_threads = pd.read_parquet(raw_threads_path, columns=["thread_id", "thread_size", "timestamp", "author", "success"])
        raw_comments = pd.read_parquet(raw_comments_path, columns=["thread_id", "id", "author"])
        enriched = pd.read_parquet(enriched_threads_path, columns=["thread_id", "thread_size", "timestamp"])
        train_y = pd.read_parquet(train_y_path, columns=["timestamp", "thread_size", "log_thread_size"])
        test_y = pd.read_parquet(test_y_path, columns=["timestamp", "thread_size", "log_thread_size"])
        train_map = pd.read_parquet(train_map_path, columns=["thread_id", "timestamp", "thread_size"])
        test_map = pd.read_parquet(test_map_path, columns=["thread_id", "timestamp", "thread_size"])

        train = train_y.assign(thread_id=train_map.thread_id.to_numpy(), partition="training")
        held = test_y.assign(thread_id=test_map.thread_id.to_numpy(), partition="held_out")
        full = pd.concat([train, held], ignore_index=True)
        population_frames[sub] = full

        expected_train_n, expected_test_n = log_counts(model_log_path)
        check(validation, sub, "population", "training count matches final preprocessing table", len(train) == expected_train_n, f"observed={len(train)}; table={expected_train_n}")
        check(validation, sub, "population", "held-out count matches final preprocessing table", len(held) == expected_test_n, f"observed={len(held)}; table={expected_test_n}")
        check(validation, sub, "population", "full count is exact train plus held-out", len(full) == len(train) + len(held) == len(raw_threads), f"full={len(full)}; train={len(train)}; held-out={len(held)}")
        mapping_ok = (
            train_y.index.equals(train_map.index)
            and test_y.index.equals(test_map.index)
            and train_y.timestamp.equals(train_map.timestamp)
            and test_y.timestamp.equals(test_map.timestamp)
            and train_y.thread_size.equals(train_map.thread_size)
            and test_y.thread_size.equals(test_map.thread_size)
        )
        check(validation, sub, "population", "split-local outcomes map exactly to retained thread IDs", mapping_ok, "index, timestamp and thread_size agree in both splits")
        check(validation, sub, "population", "chronological partitions are internally ordered and non-overlapping", train.timestamp.is_monotonic_increasing and held.timestamp.is_monotonic_increasing and train.timestamp.max() < held.timestamp.min(), f"train_end={train.timestamp.max()}; held_out_start={held.timestamp.min()}")
        raw_compare = raw_threads[["thread_id", "thread_size", "timestamp"]].sort_values("thread_id").reset_index(drop=True)
        full_compare = full[["thread_id", "thread_size", "timestamp"]].sort_values("thread_id").reset_index(drop=True)
        enriched_compare = enriched.sort_values("thread_id").reset_index(drop=True)
        check(validation, sub, "population", "final population equals raw eligible roots", raw_compare.equals(full_compare), "thread_id, timestamp and thread_size match exactly")
        enriched_ok = (
            np.array_equal(raw_compare.thread_id.astype(str).to_numpy(), enriched_compare.thread_id.astype(str).to_numpy())
            and np.array_equal(raw_compare.thread_size.to_numpy(), enriched_compare.thread_size.to_numpy())
            and np.array_equal(raw_compare.timestamp.to_numpy(), enriched_compare.timestamp.to_numpy())
        )
        check(validation, sub, "population", "feature-construction output preserves raw eligible roots", enriched_ok, "thread_id, timestamp and thread_size values match exactly; storage dtype may differ")
        check(validation, sub, "data quality", "eligible root IDs are unique", raw_threads.thread_id.is_unique, f"unique={raw_threads.thread_id.nunique()}; rows={len(raw_threads)}")
        check(validation, sub, "data quality", "retained comment IDs are unique", raw_comments.id.is_unique, f"unique={raw_comments.id.nunique()}; rows={len(raw_comments)}")
        orphan_n = int((~raw_comments.thread_id.isin(raw_threads.thread_id)).sum())
        check(validation, sub, "data quality", "no retained comments point outside eligible roots", orphan_n == 0, f"orphan_comments={orphan_n}")

        comment_counts = raw_comments.groupby("thread_id").size().rename("comment_rows")
        linked = raw_threads.merge(comment_counts, left_on="thread_id", right_index=True, how="left")
        linked["comment_rows"] = linked.comment_rows.fillna(0).astype(int)
        size_ok = linked.thread_size.eq(1 + linked.comment_rows)
        check(validation, sub, "definition", "root-inclusive T equals one plus retained descendant count", bool(size_ok.all()), f"mismatched_roots={int((~size_ok).sum())}; comments={len(raw_comments)}")
        success_ok = raw_threads.success.astype(int).eq((raw_threads.thread_size > 1).astype(int))
        check(validation, sub, "definition", "started/stalled indicator matches T", bool(success_ok.all()), f"mismatched_roots={int((~success_ok).sum())}")

        comment_author = raw_comments.merge(
            raw_threads[["thread_id", "author"]].rename(columns={"author": "root_author"}),
            on="thread_id",
            how="left",
            validate="many_to_one",
        )
        invalid_tokens = {"[deleted]", "[removed]", "", "nan", "none"}
        valid_author = (
            comment_author.author.notna()
            & comment_author.root_author.notna()
            & ~comment_author.author.astype(str).str.lower().isin(invalid_tokens)
            & ~comment_author.root_author.astype(str).str.lower().isin(invalid_tokens)
        )
        comment_author["self_comment"] = valid_author & comment_author.author.astype(str).eq(comment_author.root_author.astype(str))
        comment_author["author_comparable"] = valid_author
        self_by_root = comment_author.groupby("thread_id").self_comment.sum().rename("self_comments")
        linked_self = full[["thread_id", "partition", "thread_size"]].merge(self_by_root, left_on="thread_id", right_index=True, how="left")
        linked_self["self_comments"] = linked_self.self_comments.fillna(0).astype(int)
        for partition, data in [("full", linked_self), ("training", linked_self[linked_self.partition == "training"]), ("held_out", linked_self[linked_self.partition == "held_out"])]:
            eligible_comment_rows = comment_author[comment_author.thread_id.isin(data.thread_id)]
            self_rows.append(
                {
                    "subreddit": name,
                    "partition": partition,
                    "eligible_roots": len(data),
                    "retained_comments": int(data.thread_size.sum() - len(data)),
                    "author_comparable_comments": int(eligible_comment_rows.author_comparable.sum()),
                    "root_author_self_comments": int(data.self_comments.sum()),
                    "roots_with_self_comments": int((data.self_comments > 0).sum()),
                    "root_share_with_self_comment_pct": 100 * float((data.self_comments > 0).mean()),
                }
            )
        self_included_ok = bool(
            (linked_self.loc[linked_self.self_comments > 0, "thread_size"] > 1).all()
            and size_ok.all()
            and int(linked_self.self_comments.sum()) == int(comment_author.self_comment.sum())
        )
        check(validation, sub, "self-comments", "retained root-author self-comments contribute to C", self_included_ok, f"self_comments={int(linked_self.self_comments.sum())}; affected_roots={int((linked_self.self_comments > 0).sum())}")

        split_data = {"full": full, "training": train, "held_out": held}
        for partition in PARTITIONS:
            data = split_data[partition]
            t_all = data.thread_size.astype(int)
            c_all = t_all - 1
            started_mask = c_all >= 1
            stalled_n = int((c_all == 0).sum())
            started_n = int(started_mask.sum())
            for basis, mask in [("all_roots", pd.Series(True, index=data.index)), ("started_roots", started_mask)]:
                c = c_all[mask]
                t = t_all[mask]
                summary_rows.append(
                    {
                        "subreddit": name,
                        "partition": partition,
                        "population_basis": basis,
                        "eligible_roots": len(data),
                        "stalled_n": stalled_n,
                        "stalled_pct": 100 * stalled_n / len(data),
                        "started_n": started_n,
                        "analyzed_n": len(c),
                        "quantile_method": "Hyndman-Fan type 7 / pandas linear interpolation",
                        **stats(c, "C"),
                        **stats(t, "T"),
                    }
                )

            labels = class_labels(t_all, CUTS_T[sub])
            counts = pd.Series(labels).value_counts().reindex(range(4), fill_value=0).sort_index()
            for class_id, count_value in counts.items():
                lower = 1 if class_id == 0 else CUTS_T[sub][class_id - 1] + 1
                upper = CUTS_T[sub][class_id] if class_id < 3 else np.inf
                class_rows.append(
                    {
                        "subreddit": name,
                        "partition": partition,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "T_lower_inclusive": lower,
                        "T_upper_inclusive": "infinity" if np.isinf(upper) else int(upper),
                        "count": int(count_value),
                        "percent": 100 * int(count_value) / len(t_all),
                    }
                )

        for partition, data in [("training", train), ("held_out", held)]:
            counts = data.thread_size.astype(int).value_counts().sort_index()
            survivor = counts.sort_index(ascending=False).cumsum().sort_index()
            denominator = len(data)
            for t_value in counts.index:
                ccdf_rows.append(
                    {
                        "subreddit": name,
                        "partition": partition,
                        "T": int(t_value),
                        "frequency_at_T": int(counts.loc[t_value]),
                        "survivor_count_ge": int(survivor.loc[t_value]),
                        "denominator": denominator,
                        "survival_probability_ge": int(survivor.loc[t_value]) / denominator,
                    }
                )

        expected = workbook_counts(selected_book_path)
        for partition, workbook_col in [("training", "Train"), ("held_out", "Test")]:
            recomputed = pd.Series(class_labels(split_data[partition].thread_size, CUTS_T[sub])).value_counts().reindex(range(4), fill_value=0).sort_index()
            workbook = expected.sort_values("Class")[workbook_col].astype(int).to_numpy()
            check(validation, sub, "class reconciliation", f"{partition} class counts match selected-model class table", np.array_equal(recomputed.to_numpy(), workbook), f"recomputed={recomputed.tolist()}; workbook={workbook.tolist()}")
            published = publication_true_ratios(publication_ratios_path, sub, partition)
            ratios = recomputed.to_numpy() / recomputed.sum()
            check(validation, sub, "class reconciliation", f"{partition} class shares match thesis-facing publication table", np.allclose(ratios, published, rtol=0, atol=1e-15), f"max_abs_difference={np.max(np.abs(ratios-published)):.3g}")

        params = joblib.load(params_path)
        selected = params[SELECTED_MODEL[sub]]
        finite_bins = np.asarray(selected["bins"], dtype=float)
        infinite_bins = finite_bins.copy()
        infinite_bins[-1] = np.inf
        for partition in ("training", "held_out"):
            y = split_data[partition].log_thread_size
            finite_labels = pd.cut(y, bins=finite_bins, labels=False, include_lowest=True)
            infinite_labels = pd.cut(y, bins=infinite_bins, labels=False, include_lowest=True)
            unchanged = finite_labels.equals(infinite_labels) and finite_labels.notna().all()
            check(validation, sub, "class boundaries", f"positive-infinity top edge leaves {partition} labels unchanged", unchanged, f"finite_top={finite_bins[-1]:.15g}; max_log_T={y.max():.15g}; changed_rows={int((finite_labels != infinite_labels).sum())}")

        train_started_logs = train.loc[train.thread_size > 1, "log_thread_size"]
        recomputed_lower = np.asarray([np.log(2) - 1e-3, train_started_logs.quantile(1 / 3), train_started_logs.quantile(2 / 3)], dtype=float)
        stored_lower = finite_bins[1:4]
        lower_ok = np.allclose(recomputed_lower, stored_lower, rtol=0, atol=1e-15)
        check(validation, sub, "leakage", "stored lower boundaries reproduce from training labels only", lower_ok, f"stored={stored_lower.tolist()}; training_recomputed={recomputed_lower.tolist()}")
        check(validation, sub, "leakage", "held-out labels are not an input to lower-bound construction", lower_ok, "2_tuning.py loads args.train_y only (lines 309-335); downstream tuning reuses stored bins; final evaluation can adjust only bins[-1]")

    summary = pd.DataFrame(summary_rows)
    ccdf = pd.DataFrame(ccdf_rows)
    classes = pd.DataFrame(class_rows)
    self_comments = pd.DataFrame(self_rows)
    validations = pd.DataFrame(validation)
    evidence.extend(
        [
            {
                "claim_id": "E01",
                "requirement": "Exact final eligible full/training/held-out populations",
                "evidence_type": "direct final-data reconciliation",
                "source_paths": "Outputs/0_preprocessing/*/*_{train,test}_Y.parquet | Outputs/0_preprocessing/*/tf-idf/*_svd_enriched_{train,test}_data.parquet | Inputs/*_threads.parquet | 3_model_data_log.xlsx",
                "result": "Exact row, order, timestamp, thread-size, and count reconciliation for all three subreddits",
                "status": "PASS",
                "limitation": "Observed within the archived snapshot; not eventual lifetime size",
            },
            {
                "claim_id": "E02",
                "requirement": "C counts retained descendants and T=1+C",
                "evidence_type": "raw root-comment aggregation",
                "source_paths": "Inputs/*_threads.parquet | Inputs/*_comments.parquet",
                "result": "Exact for every eligible root; stalled roots have C=0 and T=1",
                "status": "PASS",
                "limitation": "Only retained comments in the archive are observable",
            },
            {
                "claim_id": "E03",
                "requirement": "Started and all-root distribution summaries",
                "evidence_type": "independent descriptive calculation",
                "source_paths": "Outputs/0_preprocessing/*/*_{train,test}_Y.parquet",
                "result": "Full/training/held-out C and T statistics frozen in outputs/summary_statistics.csv",
                "status": "PASS",
                "limitation": "Quantiles use Hyndman-Fan type 7 linear interpolation",
            },
            {
                "claim_id": "E04",
                "requirement": "Exact empirical training versus held-out CCDF",
                "evidence_type": "frequency and survivor-count enumeration",
                "source_paths": "outputs/ccdf.csv | render_ccdf.py",
                "result": "P(T>=t) retained for every observed unique T with numerator and denominator",
                "status": "PASS",
                "limitation": "Descriptive empirical survival only; no tail model fitted",
            },
            {
                "claim_id": "E05",
                "requirement": "Population and class counts match thesis-facing tables",
                "evidence_type": "workbook and publication-output reconciliation",
                "source_paths": "3_model_data_log.xlsx | selected test_data_results.xlsx | Publication_Outputs/2_Thread_Size/cms/predicted_class_ratios.xlsx",
                "result": "Training and held-out counts/shares match exactly for every class and subreddit",
                "status": "PASS",
                "limitation": "The thesis manuscript itself is not present in this checkout",
            },
            {
                "claim_id": "E06",
                "requirement": "Retained root-author self-comments count in C",
                "evidence_type": "code inspection plus author-equality check",
                "source_paths": "1_construct_features.py | Inputs/*_threads.parquet | Inputs/*_comments.parquet",
                "result": "No author exclusion in code; all root-comment count identities include matched self-comments",
                "status": "PASS",
                "limitation": "Identity is assessed from the retained author field; external account ownership is not re-identified",
            },
            {
                "claim_id": "E07",
                "requirement": "Infinite top edge preserves final labels",
                "evidence_type": "row-level counterfactual classification",
                "source_paths": "params_post_hyperparam_tuning.jl | final train/test Y parquet files",
                "result": "Zero changed assignments in all six final partitions",
                "status": "PASS",
                "limitation": "Applies to the frozen final populations",
            },
            {
                "claim_id": "E08",
                "requirement": "Held-out labels do not set lower class boundaries",
                "evidence_type": "data-flow code inspection plus numerical reproduction",
                "source_paths": "2_tuning.py | 3_hyperparameter_tuning.py | 4_run_tuned_model.py | params_post_hyperparameter_tuning.jl",
                "result": "Lower edges reproduce from training outcomes only; held-out access is confined to evaluation and a possible top-edge extension",
                "status": "PASS",
                "limitation": "The executed-code claim is tied to the frozen logs and artifacts in this checkout",
            },
        ]
    )

    summary.to_csv(OUT / "summary_statistics.csv", index=False, float_format="%.15g", lineterminator="\n")
    ccdf.to_csv(OUT / "ccdf.csv", index=False, float_format="%.17g", lineterminator="\n")
    classes.to_csv(OUT / "population_class_counts.csv", index=False, float_format="%.15g", lineterminator="\n")
    self_comments.to_csv(OUT / "self_comment_checks.csv", index=False, float_format="%.15g", lineterminator="\n")
    pd.DataFrame(evidence).to_csv(OUT / "evidence_matrix.csv", index=False, lineterminator="\n")
    validations.to_csv(OUT / "validation_checks.csv", index=False, lineterminator="\n")

    if not validations.status.eq("PASS").all():
        failed = validations[validations.status != "PASS"]
        raise AssertionError(f"{len(failed)} validation checks failed; inspect outputs/validation_checks.csv")

    render()

    scripts_commit = git_commit()
    inventory = []
    for path, (role, version_basis) in sorted(inventory_paths.items(), key=lambda item: rel(item[0])):
        stat = path.stat()
        inventory.append(
            {
                "path": rel(path),
                "role": role,
                "bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": sha256(path),
                "version_basis": scripts_commit if "Scripts Git commit" in version_basis else version_basis,
            }
        )
    pd.DataFrame(inventory).to_csv(OUT / "source_inventory.csv", index=False, lineterminator="\n")

    run_record = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scripts_git_commit": scripts_commit,
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "openpyxl": openpyxl.__version__,
        "model_loaded_or_scored": False,
        "writes_outside_audit_directory": False,
        "quantile_method": "Hyndman-Fan type 7 / pandas linear interpolation",
        "ccdf_definition": "survival_probability_ge = count(T >= t) / partition N for every observed unique integer t",
        "validation_checks": len(validations),
        "validation_failures": int((validations.status != "PASS").sum()),
    }
    (OUT / "run_record.json").write_text(json.dumps(run_record, indent=2) + "\n", encoding="utf-8")

    full_rows = summary[(summary.partition == "full") & (summary.population_basis == "all_roots")].set_index("subreddit")
    started_full = summary[(summary.partition == "full") & (summary.population_basis == "started_roots")].set_index("subreddit")
    report = f"""# Study 1 observed thread-size distribution audit

## Audit conclusion

**Ready within reviewed scope.** The exact final eligible populations are {int(full_rows.loc['r/Conspiracy','eligible_roots']):,} r/Conspiracy roots, {int(full_rows.loc['r/CryptoCurrency','eligible_roots']):,} r/CryptoCurrency roots, and {int(full_rows.loc['r/politics','eligible_roots']):,} r/politics roots. Their training/held-out counts, stalled/started counts, and four-class distributions reconcile exactly with the final preprocessing logs, selected-model class tables, and thesis-facing publication ratios. All {len(validations)} automated checks pass.

This is an audit of **observed size within finite archived coverage**, not eventual thread size. `C` is the number of retained descendant-comment rows and `T = 1 + C` includes the root. A stalled root has `C = 0` and `T = 1`; a started root has `C >= 1` and `T >= 2`.

The earlier validated probability-calibration audit supplies reliable partition and class-range evidence but not these raw distribution summaries or this CCDF. No older plot was reused.

## Population and stalled prevalence

{markdown_table(summary[(summary.population_basis == 'all_roots')][['subreddit','partition','eligible_roots','stalled_n','stalled_pct','started_n']].assign(partition=lambda d:d.partition.str.replace('_','-',regex=False), stalled_pct=lambda d:d.stalled_pct.map(lambda x:f'{x:.2f}')).rename(columns={'subreddit':'Subreddit','partition':'Partition','eligible_roots':'Eligible roots','stalled_n':'Stalled n','stalled_pct':'Stalled %','started_n':'Started n'}))}

## Started-root statistics

Percentiles use the Hyndman-Fan type 7 definition (pandas linear interpolation). Thus an interpolated percentile need not itself be an observed integer. The frozen CSV retains 15 significant digits. Because `T = C + 1`, every started-root `T` statistic is exactly one larger than its `C` counterpart.

### Descendant comments, C, among started roots

{markdown_table(compact_stat_table(summary, 'started_roots', 'C'))}

### Root-inclusive size, T, among started roots

{markdown_table(compact_stat_table(summary, 'started_roots', 'T'))}

## Corresponding all-root statistics

All-root summaries are interpretable as the observed mixture including stalled roots; they therefore have minima `C=0` and `T=1`. They should not be substituted for the started-root conditional summaries.

### Descendant comments, C, across all roots

{markdown_table(compact_stat_table(summary, 'all_roots', 'C'))}

### Root-inclusive size, T, across all roots

{markdown_table(compact_stat_table(summary, 'all_roots', 'T'))}

## Proposed empirical CCDF

![Three-panel empirical CCDF](figures/thread_size_ccdf.png)

**Figure caption.** Empirical complementary cumulative distributions of root-inclusive observed thread size, `P(T >= t)`, for the chronological training and held-out populations in each subreddit. Both axes are logarithmic. Every plotted step comes from an observed integer `T` and the exact survivor numerator and partition denominator in `outputs/ccdf.csv`; no distributional model or power-law fit is shown. Stalled roots occur at `T=1` and comprise {full_rows.loc['r/Conspiracy','stalled_pct']:.2f}%, {full_rows.loc['r/CryptoCurrency','stalled_pct']:.2f}%, and {full_rows.loc['r/politics','stalled_pct']:.2f}% of the full r/Conspiracy, r/CryptoCurrency, and r/politics populations, respectively; split-specific prevalence is printed in each panel and tabulated above.

The long upper tails are descriptive features of these finite archived observations. This figure alone does not justify calling any distribution a power law.

## Required verification results

1. **Population and class reconciliation — pass.** Full populations equal training plus held-out populations exactly. Selected-model training and held-out class counts are reproduced row for row: r/Conspiracy `1404/2625/2532/2555` and `316/706/631/626`; r/CryptoCurrency `6356/1835/1853/1810` and `1692/510/386/376`; r/politics `18814/11717/10816/10927` and `3876/2921/3121/3151` for Stalled/Small/Medium/Large. The corresponding true-class shares equal the thesis-facing publication workbook to machine precision.
2. **Root-author self-comments — included.** Feature construction reads the comment parquet without an author-based exclusion and carries the pre-existing `thread_size` target forward. Direct raw-data aggregation shows `T = 1 +` all retained comment rows for every eligible root. Exact retained author equality identifies {int(self_comments[(self_comments.subreddit=='r/Conspiracy')&(self_comments.partition=='full')].root_author_self_comments.iloc[0]):,}, {int(self_comments[(self_comments.subreddit=='r/CryptoCurrency')&(self_comments.partition=='full')].root_author_self_comments.iloc[0]):,}, and {int(self_comments[(self_comments.subreddit=='r/politics')&(self_comments.partition=='full')].root_author_self_comments.iloc[0]):,} root-author comments, affecting {int(self_comments[(self_comments.subreddit=='r/Conspiracy')&(self_comments.partition=='full')].roots_with_self_comments.iloc[0]):,}, {int(self_comments[(self_comments.subreddit=='r/CryptoCurrency')&(self_comments.partition=='full')].roots_with_self_comments.iloc[0]):,}, and {int(self_comments[(self_comments.subreddit=='r/politics')&(self_comments.partition=='full')].roots_with_self_comments.iloc[0]):,} roots. Excluding them would contradict the final `C` values for those roots.
3. **Infinite final edge — pass.** Replacing each finite upper bin edge by positive infinity changes zero assignments in all six final training/held-out partitions. The finite edge already exceeds the largest frozen target in its subreddit.
4. **No held-out determination of lower boundaries — pass.** `2_tuning.py` loads only the training outcome file before constructing the fixed lower edges from started-training quantiles. Hyperparameter tuning reuses those bins. Final evaluation loads held-out labels only after the boundaries exist; its sole possible boundary mutation is an extension of the final upper edge. Stored lower edges reproduce numerically from training labels alone for all three subreddits.

## Thesis-safe interpretation and limitations

- The archive observes retained Reddit records over finite collection coverage. It does not establish how large each thread eventually became after the archive stopped observing it.
- `C` counts retained descendant comment rows, including retained comments by the root author. It is not a count of unique commenters.
- Train/held-out differences are descriptive distribution shifts across later chronological records, not causal effects and not evidence of statistical significance.
- Linear-interpolated quantiles are conventional summaries; maxima and tail means are sensitive to a small number of very large observed threads.
- The repository checkout contains the final pipeline inputs, outputs, and prior audit records but not the thesis manuscript itself. Reconciliation therefore uses the final thesis-facing preprocessing and publication tables stored here.
- No formal tail-family comparison, goodness-of-fit test, or power-law analysis was performed. No such claim is supported by this package.

## Exact Chapter 4 and Appendix A recommendation

### Chapter 4 paragraph justified by this audit

> Thread size was measured over the finite archived coverage as `T = 1 + C`, where `C` is the number of retained descendant comments and the added unit is the root post. The final eligible populations contained 11,395 r/Conspiracy, 14,818 r/CryptoCurrency and 65,343 r/politics roots, divided chronologically into 9,116/2,279, 11,854/2,964 and 52,274/13,069 training/held-out observations. In the full populations, 15.09%, 54.31% and 34.72% of roots were stalled (`C=0`), respectively. Among started roots, median root-inclusive observed sizes were {fmt_num(started_full.loc['r/Conspiracy','T_median'])}, {fmt_num(started_full.loc['r/CryptoCurrency','T_median'])} and {fmt_num(started_full.loc['r/politics','T_median'])}; the corresponding means were {started_full.loc['r/Conspiracy','T_mean']:.2f}, {started_full.loc['r/CryptoCurrency','T_mean']:.2f} and {started_full.loc['r/politics','T_mean']:.2f}, reflecting long upper tails. These quantities describe archived observations rather than eventual thread size, and the empirical tail plots do not by themselves imply a power-law distribution.

### Chapter 4 table justified by this audit

Use one compact table with one row per subreddit × partition (full, training, held-out) and these columns: eligible roots; stalled `n (%)`; started roots; and, among started roots, `T` median `[Q1, Q3]`, P90, P95, P99, mean and maximum. This keeps the root-inclusive outcome used in the thesis visible while retaining the stalled mass. The exact source is `outputs/summary_statistics.csv`.

### Chapter 4 figure justified by this audit

Use `figures/thread_size_ccdf.svg` (or the 300-dpi PNG) with the caption above: three panels, training versus held-out empirical `P(T >= t)`, log-scaled axes, stalled prevalence disclosed, and no fitted tail law.

### Appendix A

Place the full paired `C` and `T` started-root and all-root tables, the quantile convention, exact class reconciliation, self-comment evidence, infinite-edge equivalence, held-out-leakage check, validation matrix, and provenance/hashes in Appendix A. These are necessary audit details but would overburden the Chapter 4 results narrative. `C` remains important in the appendix because it makes the stalled definition and descendant-only count explicit even though every `T` statistic is its one-unit translation.

## Files

- `outputs/summary_statistics.csv`: frozen full/training/held-out descriptive statistics.
- `outputs/ccdf.csv`: exact empirical CCDF points with frequencies, survivor numerators and denominators.
- `outputs/population_class_counts.csv`: recomputed four-class populations and shares.
- `outputs/self_comment_checks.csv`: aggregate author-comparability and self-comment evidence.
- `outputs/evidence_matrix.csv` and `outputs/validation_checks.csv`: claim-to-source matrix and automated results.
- `outputs/source_inventory.csv`, `outputs/run_record.json`, and `provenance.md`: hashes, versions and scope.
- `render_ccdf.py`: deterministic figure rendering from the two frozen aggregate CSV inputs.
"""
    (AUDIT / "report.md").write_text(report, encoding="utf-8")

    provenance_rows = pd.read_csv(OUT / "source_inventory.csv")
    provenance_table = provenance_rows[["path", "role", "sha256"]].copy()
    provenance = f"""# Provenance

## Version anchors

- Scripts Git repository: `{(ROOT / 'Scripts').resolve()}`
- Scripts commit: `{scripts_commit}`
- Scripts working-tree note at audit time: prior audit directories and this new audit package are untracked; no modelling or thesis output was modified.
- Data archive metadata: `Zenodo_upload/README.md` identifies archive version 1.0 and DOI `10.5281/zenodo.17831100`.
- Raw `Inputs/*_threads.parquet` files were separately checked as byte-identical to the corresponding `Zenodo_upload/data_raw/*_threads.parquet` files.
- Generated UTC: `{run_record['generated_utc']}`

## Authority chain

The executed preprocessing logs identify `Inputs/{{subreddit}}_threads.parquet` and `Inputs/{{subreddit}}_comments.parquet` as feature-construction inputs. The feature output is chronologically split 80/20 by `2_tf_idf_analysis.py`; the split rows are retained in `*_svd_enriched_{{train,test}}_data.parquet`; and `3_model_data.py` carries their outcomes into the final `*_{{train,test}}_Y.parquet` files. Those final outcome files control the distribution audit. Raw comments are used only to verify the `T=1+C` identity and self-comment treatment. Selected-model workbooks and the publication ratio workbook are reconciliation targets, not sources used to define the distributions.

## SHA-256 source inventory

{markdown_table(provenance_table.rename(columns={'path':'Path','role':'Role','sha256':'SHA-256'}))}

The machine-readable inventory also records byte sizes, modification times and version bases in `outputs/source_inventory.csv`. Generated aggregate outputs are reproducible from the named sources; the report and figures are not inputs to any modelling pipeline.
"""
    (AUDIT / "provenance.md").write_text(provenance, encoding="utf-8")

    generated_paths = [
        AUDIT / "README.md",
        AUDIT / "audit_thread_size_distribution.py",
        AUDIT / "render_ccdf.py",
        AUDIT / "report.md",
        AUDIT / "provenance.md",
        FIG / "thread_size_ccdf.svg",
        FIG / "thread_size_ccdf.png",
        *[OUT / name for name in KEY_OUTPUTS],
    ]
    package_rows = []
    for path in generated_paths:
        package_rows.append({"path": path.relative_to(AUDIT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(package_rows).to_csv(AUDIT / "package_manifest.csv", index=False, lineterminator="\n")
    print(f"PASS: {len(validations)} validation checks; outputs written to {AUDIT}")


if __name__ == "__main__":
    main()
