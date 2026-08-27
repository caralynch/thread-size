"""Provenance-first endpoint follow-up audit for Study 1.

This script is deliberately read-only with respect to the frozen data/model
mirror. It writes aggregate, identifier-free outputs only to the requested
audit directory. It never loads a trained model, fits a model, ranks features,
or regenerates predictions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SUBREDDITS = ("conspiracy", "crypto", "politics")
S1_MODELS = {"conspiracy": 4, "crypto": 4, "politics": 4}
S2_MODELS = {"conspiracy": 3, "crypto": 2, "politics": 3}
S1_CLASS_NAMES = {0: "Stalled", 1: "Started"}
S2_CLASS_NAMES = {0: "Stalled", 1: "Small", 2: "Medium", 3: "Large"}
EXPOSURE_HOURS = (1, 6, 12, 24, 48, 72, 168)
SENSITIVITY_HOURS = (24, 48, 72, 168)
LEAD_RAW_ROOT_24H = {"conspiracy": 3.29, "crypto": 2.88, "politics": 2.20}
LEAD_COMMENT_24H = {"conspiracy": 94.0, "crypto": 97.8, "politics": 98.8}


class AuditMismatch(RuntimeError):
    """Raised when the frozen final modelling population cannot be reproduced."""


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"L:\Documents\reddit_analyses\thread-size"),
        help="Read-only extracted mirror of Zenodo 10.5281/zenodo.17831100.",
    )
    parser.add_argument(
        "--endpoint-config",
        type=Path,
        default=here / "collection_endpoints.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=here / "outputs")
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Skip SHA-256 calculation during development runs.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def utc_series(values: pd.Series) -> pd.Series:
    # Frozen parquet timestamps are timezone-naive; raw UNIX agreement supports UTC.
    return pd.to_datetime(values, utc=True, errors="raise")


def iso_utc(value: pd.Timestamp | datetime) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def parquet_schema(path: Path) -> tuple[int, str, str]:
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    compact = " | ".join(f"{field.name}:{field.type}" for field in schema)
    columns = "|".join(field.name for field in schema)
    return pf.metadata.num_rows, columns, compact


def csv_schema(path: Path) -> tuple[int, str, str]:
    frame = pd.read_csv(path)
    compact = " | ".join(f"{col}:{frame[col].dtype}" for col in frame.columns)
    return len(frame), "|".join(frame.columns), compact


def excel_schema(path: Path) -> tuple[int | None, str, str]:
    sheets = pd.ExcelFile(path).sheet_names
    return None, "|".join(sheets), "Excel workbook; sheets=" + "|".join(sheets)


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true.astype(int), y_pred.astype(int)), 1)
    return cm


def frame_values_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Compare stored values while tolerating equivalent pandas extension dtypes."""
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        return False
    for column in left.columns:
        lhs = left[column]
        rhs = right[column]
        if pd.api.types.is_datetime64_any_dtype(lhs.dtype) and pd.api.types.is_datetime64_any_dtype(rhs.dtype):
            equal = np.array_equal(lhs.astype("int64").to_numpy(), rhs.astype("int64").to_numpy())
        elif pd.api.types.is_numeric_dtype(lhs.dtype) and pd.api.types.is_numeric_dtype(rhs.dtype):
            equal = np.array_equal(lhs.to_numpy(), rhs.to_numpy(), equal_nan=True)
        else:
            equal = np.array_equal(lhs.astype(str).to_numpy(), rhs.astype(str).to_numpy())
        if not equal:
            return False
    return True


def mcc_from_cm(cm: np.ndarray) -> float:
    true_sum = cm.sum(axis=1, dtype=float)
    pred_sum = cm.sum(axis=0, dtype=float)
    n = float(cm.sum())
    numerator = float(np.trace(cm)) * n - float(np.dot(true_sum, pred_sum))
    denominator_sq = (n * n - float(np.dot(pred_sum, pred_sum))) * (
        n * n - float(np.dot(true_sum, true_sum))
    )
    if denominator_sq <= 0:
        return 0.0
    return numerator / math.sqrt(denominator_sq)


def metrics_from_cm(cm: np.ndarray) -> dict[str, object]:
    row_sum = cm.sum(axis=1)
    recalls = np.divide(
        np.diag(cm),
        row_sum,
        out=np.full(len(row_sum), np.nan, dtype=float),
        where=row_sum != 0,
    )
    return {
        "mcc": mcc_from_cm(cm) if cm.sum() else np.nan,
        "balanced_accuracy": float(np.nanmean(recalls)) if np.any(row_sum) else np.nan,
        "recalls": recalls,
        "true_counts": row_sum,
    }


def parse_np_float_list(value: object) -> list[float]:
    text = str(value)
    wrapped = re.findall(r"np\.float64\(([-+0-9.eE]+)\)", text)
    if wrapped:
        return [float(item) for item in wrapped]
    return [float(item) for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]


def stability_label(n: int, true_counts: np.ndarray) -> str:
    if np.any(true_counts == 0):
        return "not_interpretable"
    if n < 500 or int(true_counts.min()) < 50:
        return "caution_small"
    return "adequate_descriptive"


def add_check(
    checks: list[dict[str, object]],
    subreddit: str,
    check: str,
    passed: bool,
    observed: object,
    expected: object,
    severity: str = "high",
    blocking: bool = False,
) -> None:
    checks.append(
        {
            "subreddit": subreddit,
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "expected": expected,
            "severity": severity,
            "blocking_for_population_reproduction": blocking,
        }
    )


def source_paths(root: Path) -> list[tuple[str, Path, str, str]]:
    sources: list[tuple[str, Path, str, str]] = []
    for sub in SUBREDDITS:
        base0 = root / "Outputs" / "0_preprocessing" / sub
        s1 = root / "Outputs" / "1_thread_start" / sub / "4_model" / f"model_{S1_MODELS[sub]}"
        s2 = root / "Outputs" / "2_thread_size" / sub / "4_model" / f"model_{S2_MODELS[sub]}"
        sources.extend(
            [
                (f"{sub}_clean_threads", root / "Inputs" / f"{sub}_threads.parquet", "final cleaned roots", "parquet"),
                (f"{sub}_clean_comments", root / "Inputs" / f"{sub}_comments.parquet", "final cleaned comments", "parquet"),
                (f"{sub}_feature_threads", base0 / f"{sub}_threads_extra_feats.parquet", "feature-construction root output", "parquet"),
                (f"{sub}_enriched_test", base0 / "tf-idf" / f"{sub}_svd_enriched_test_data.parquet", "chronological held-out artefact with stable root ID", "parquet"),
                (f"{sub}_test_X", base0 / f"{sub}_test_X.parquet", "final held-out feature matrix", "parquet"),
                (f"{sub}_test_Y", base0 / f"{sub}_test_Y.parquet", "final held-out target artefact", "parquet"),
                (f"{sub}_s1_predictions", s1 / "test_started_threads.parquet", "selected Stage 1 frozen held-out predictions", "parquet"),
                (f"{sub}_s1_metadata", s1 / "test_data_results.xlsx", "selected Stage 1 model metadata and metrics", "xlsx"),
                (f"{sub}_s2_predictions", s2 / "test_preds.csv", "selected Stage 2 frozen held-out predictions", "csv"),
                (f"{sub}_s2_labels", s2 / "model_data" / "y_test.parquet", "selected Stage 2 saved held-out labels", "parquet"),
                (f"{sub}_s2_metadata", s2 / "test_data_results.xlsx", "selected Stage 2 model metadata and metrics", "xlsx"),
            ]
        )
    sources.extend(
        [
            ("stage1_model_selection", root / "Publication_Outputs" / "1_Thread_Start" / "s1_mods.txt", "publication model selection", "text"),
            ("stage2_model_selection", root / "Publication_Outputs" / "2_Thread_Size" / "s2_mods.txt", "publication model selection", "text"),
            ("zenodo_frozen_readme", root / "Zenodo_upload" / "README.md", "frozen deposit documentation", "text"),
        ]
    )
    return sources


def build_source_inventory(root: Path, skip_hashes: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_id, path, role, kind in source_paths(root):
        if not path.exists():
            rows.append(
                {
                    "source_id": source_id,
                    "path_or_url": str(path),
                    "role": role,
                    "kind": kind,
                    "exists": False,
                    "provenance_status": "missing_required_local_source",
                }
            )
            continue
        if kind == "parquet":
            n_rows, columns, schema = parquet_schema(path)
        elif kind == "csv":
            n_rows, columns, schema = csv_schema(path)
        elif kind == "xlsx":
            n_rows, columns, schema = excel_schema(path)
        else:
            n_rows, columns, schema = None, None, "plain text"
        stat = path.stat()
        rows.append(
            {
                "source_id": source_id,
                "path_or_url": str(path),
                "role": role,
                "kind": kind,
                "exists": True,
                "version": "Zenodo 10.5281/zenodo.17831100 v1.0 extracted mirror",
                "sha256": "not_computed" if skip_hashes else sha256_file(path),
                "file_size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "row_count": n_rows,
                "columns_or_sheets": columns,
                "schema": schema,
                "timestamp_fields": "timestamp" if schema and "timestamp" in schema else "",
                "timezone_in_file": "naive" if schema and "timestamp[ns]" in schema else "not_applicable_or_unspecified",
                "interpreted_timezone": "UTC" if schema and "timestamp[ns]" in schema else "not_applicable",
                "provenance_status": "frozen_final_artefact",
            }
        )
    rows.extend(
        [
            {
                "source_id": "zenodo_model_record",
                "path_or_url": "https://doi.org/10.5281/zenodo.17831100",
                "role": "authoritative deposit metadata for frozen modelling artefacts",
                "kind": "Zenodo record",
                "exists": True,
                "version": "1.0; published 2025-12-05; record 17831100",
                "provenance_status": "authoritative_external_metadata",
            },
            {
                "source_id": "zenodo_raw_record",
                "path_or_url": "https://doi.org/10.5281/zenodo.17079717",
                "role": "authoritative raw-data coverage dates and raw archive files",
                "kind": "Zenodo record",
                "exists": True,
                "version": "1.0; published 2025-09-08; record 17079717",
                "provenance_status": "authoritative_external_metadata; coverage boundary only",
            },
        ]
    )
    return pd.DataFrame(rows)


def selected_model_metadata(root: Path, sub: str) -> tuple[dict[str, object], list[float]]:
    s1_path = root / "Outputs" / "1_thread_start" / sub / "4_model" / f"model_{S1_MODELS[sub]}" / "test_data_results.xlsx"
    s2_path = root / "Outputs" / "2_thread_size" / sub / "4_model" / f"model_{S2_MODELS[sub]}" / "test_data_results.xlsx"
    s1_params_df = pd.read_excel(s1_path, sheet_name="model_params")
    s1_params = dict(zip(s1_params_df["Key"], s1_params_df["Value"]))
    s2_params_df = pd.read_excel(s2_path, sheet_name="model_params")
    s2_params = dict(zip(s2_params_df["Key"], s2_params_df["Value"]))
    bins = parse_np_float_list(s2_params["bins"])
    metadata = {
        "subreddit": sub,
        "stage1_selected_n_features": S1_MODELS[sub],
        "stage1_features": s1_params.get("features"),
        "stage1_class_weights": s1_params.get("final_class_weights"),
        "stage1_frozen_decision_threshold": s1_params.get("model_threshold"),
        "stage2_selected_n_features": S2_MODELS[sub],
        "stage2_features": s2_params.get("features"),
        "stage2_class_weights": s2_params.get("final_class_weights"),
        "stage2_log_bin_edges": json.dumps(bins),
        "stage2_approx_exp_bin_edges": json.dumps([math.exp(x) for x in bins]),
        "prediction_policy": "saved hard predictions; no regeneration",
    }
    return metadata, bins


def workbook_reference_metrics(path: Path, stage: str) -> dict[str, float]:
    if stage == "stage1":
        perf = pd.read_excel(path, sheet_name="performance", index_col=0)
        row = perf.loc["test"]
        return {
            "mcc": float(row["MCC"]),
            "balanced_accuracy": float(row["Balanced accuracy"]),
            "recall_0": float(row["Recall Stalled"]),
            "recall_1": float(row["Recall Started"]),
        }
    perf = pd.read_excel(path, sheet_name="performance", index_col=0)
    return {
        "mcc": float(perf.loc["MCC", "test"]),
        "balanced_accuracy": float(perf.loc["Balanced accuracy", "test"]),
        "recall_0": float(perf.loc["Stalled recall", "test"]),
        "recall_1": float(perf.loc["Small recall", "test"]),
        "recall_2": float(perf.loc["Medium recall", "test"]),
        "recall_3": float(perf.loc["Large recall", "test"]),
    }


def add_exposure_rows(
    rows: list[dict[str, object]],
    sub: str,
    frame: pd.DataFrame,
    stage: str,
    class_column: str | None,
    class_names: dict[int, str] | None,
) -> None:
    groups: list[tuple[str, pd.DataFrame]] = [("All held-out roots", frame)]
    if class_column and class_names:
        groups.extend(
            (class_names[class_value], frame[frame[class_column] == class_value])
            for class_value in sorted(class_names)
        )
    for class_label, group in groups:
        for horizon in EXPOSURE_HOURS:
            limited = group["potential_followup_hours"] < horizon
            rows.append(
                {
                    "subreddit": sub,
                    "classification_stage": stage,
                    "observed_class": class_label,
                    "horizon_hours": horizon,
                    "heldout_roots": len(group),
                    "limited_followup_roots": int(limited.sum()),
                    "limited_followup_pct": float(limited.mean() * 100) if len(group) else np.nan,
                    "endpoint_kind": group["endpoint_kind"].iloc[0] if len(group) else "",
                }
            )


def add_sensitivity_rows(
    perf_rows: list[dict[str, object]],
    class_rows: list[dict[str, object]],
    cm_rows: list[dict[str, object]],
    sub: str,
    stage: str,
    frame: pd.DataFrame,
    true_col: str,
    pred_col: str,
    class_names: dict[int, str],
) -> None:
    specs: list[tuple[str, int | None]] = [("full_heldout", None)] + [
        (f"at_least_{hours}h", hours) for hours in SENSITIVITY_HOURS
    ]
    n_classes = len(class_names)
    original_n = len(frame)
    for label, hours in specs:
        subset = frame if hours is None else frame[frame["potential_followup_hours"] >= hours]
        y_true = subset[true_col].to_numpy(dtype=int)
        y_pred = subset[pred_col].to_numpy(dtype=int)
        cm = confusion(y_true, y_pred, n_classes)
        metric = metrics_from_cm(cm)
        excluded_n = original_n - len(subset)
        perf_rows.append(
            {
                "subreddit": sub,
                "stage": stage,
                "exclusion": label,
                "minimum_followup_hours": hours,
                "retained_n": len(subset),
                "excluded_n": excluded_n,
                "excluded_pct": excluded_n / original_n * 100,
                "mcc": metric["mcc"],
                "balanced_accuracy": metric["balanced_accuracy"],
                "minimum_observed_class_n": int(np.min(metric["true_counts"])),
                "stability_flag": stability_label(len(subset), metric["true_counts"]),
            }
        )
        for class_idx, class_name in class_names.items():
            true_count = int(metric["true_counts"][class_idx])
            class_rows.append(
                {
                    "subreddit": sub,
                    "stage": stage,
                    "exclusion": label,
                    "minimum_followup_hours": hours,
                    "class_index": class_idx,
                    "observed_class": class_name,
                    "true_count": true_count,
                    "prevalence_pct": true_count / len(subset) * 100 if len(subset) else np.nan,
                    "recall": metric["recalls"][class_idx],
                }
            )
            for pred_idx, pred_name in class_names.items():
                count = int(cm[class_idx, pred_idx])
                cm_rows.append(
                    {
                        "subreddit": sub,
                        "stage": stage,
                        "exclusion": label,
                        "minimum_followup_hours": hours,
                        "true_class_index": class_idx,
                        "true_class": class_name,
                        "predicted_class_index": pred_idx,
                        "predicted_class": pred_name,
                        "count": count,
                        "within_true_class_pct": count / true_count * 100 if true_count else np.nan,
                        "whole_subset_pct": count / len(subset) * 100 if len(subset) else np.nan,
                    }
                )


def scan_comments(
    sub: str,
    roots: pd.DataFrame,
    comments_path: Path,
    endpoint: pd.Timestamp,
    checks: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Stream comments to keep the 3.6M-row politics scan bounded in memory."""
    parquet = pq.ParquetFile(comments_path)
    id_values = pq.read_table(comments_path, columns=["id"])["id"]
    comment_rows = parquet.metadata.num_rows
    id_nulls = int(id_values.null_count)
    distinct_ids = int(pc.count_distinct(id_values).as_py())
    add_check(checks, sub, "comment id non-null", id_nulls == 0, id_nulls, 0, "high")
    add_check(checks, sub, "comment id unique", distinct_ids == comment_rows, comment_rows - distinct_ids, 0, "high")
    del id_values
    gc.collect()

    root_ids = roots["thread_id"].astype(str).to_numpy()
    root_position = {thread_id: i for i, thread_id in enumerate(root_ids)}
    root_timestamp_ns = roots["timestamp_utc"].astype("int64").to_numpy()
    root_followup = roots["potential_followup_hours"].to_numpy(dtype=float)
    expected_comments = roots["thread_size"].to_numpy(dtype=np.int64) - 1
    actual_comments = np.zeros(len(roots), dtype=np.int64)
    first_delay = np.full(len(roots), np.inf, dtype=float)
    endpoint_ns = int(endpoint.value)

    horizon_den = {h: 0 for h in EXPOSURE_HOURS}
    horizon_num = {h: 0 for h in EXPOSURE_HOURS}
    common_den = 0
    common_num = {h: 0 for h in EXPOSURE_HOURS}
    orphan_comments = 0
    negative_delays = 0
    comments_after_boundary = 0
    min_comment_ns: int | None = None
    max_comment_ns: int | None = None

    for batch in parquet.iter_batches(columns=["thread_id", "timestamp"], batch_size=250_000):
        frame = batch.to_pandas()
        positions = frame["thread_id"].astype(str).map(root_position).fillna(-1).to_numpy(dtype=np.int64)
        comment_ns = pd.to_datetime(frame["timestamp"], utc=True, errors="raise").astype("int64").to_numpy()
        if len(comment_ns):
            batch_min = int(comment_ns.min())
            batch_max = int(comment_ns.max())
            min_comment_ns = batch_min if min_comment_ns is None else min(min_comment_ns, batch_min)
            max_comment_ns = batch_max if max_comment_ns is None else max(max_comment_ns, batch_max)
        comments_after_boundary += int((comment_ns >= endpoint_ns).sum())
        found = positions >= 0
        orphan_comments += int((~found).sum())
        if not np.any(found):
            continue
        pos = positions[found]
        c_ns = comment_ns[found]
        actual_comments += np.bincount(pos, minlength=len(roots))
        delays = (c_ns - root_timestamp_ns[pos]) / 3_600_000_000_000
        nonnegative = delays >= 0
        negative_delays += int((~nonnegative).sum())
        if np.any(nonnegative):
            pos_valid = pos[nonnegative]
            delay_valid = delays[nonnegative]
            np.minimum.at(first_delay, pos_valid, delay_valid)
            followup_valid = root_followup[pos_valid]
            for horizon in EXPOSURE_HOURS:
                eligible = followup_valid >= horizon
                horizon_den[horizon] += int(eligible.sum())
                horizon_num[horizon] += int((eligible & (delay_valid <= horizon)).sum())
            common = followup_valid >= 168
            common_den += int(common.sum())
            for horizon in EXPOSURE_HOURS:
                common_num[horizon] += int((common & (delay_valid <= horizon)).sum())

    add_check(checks, sub, "comment-to-root join coverage", orphan_comments == 0, orphan_comments, 0, "critical")
    size_mismatch = int((actual_comments != expected_comments).sum())
    add_check(checks, sub, "thread_size equals root plus linked descendant comments", size_mismatch == 0, size_mismatch, 0, "high")
    started_without_comment = int(((roots["thread_size"].to_numpy() > 1) & (actual_comments == 0)).sum())
    add_check(checks, sub, "all observed-started roots have linked comments", started_without_comment == 0, started_without_comment, 0, "high")
    add_check(checks, sub, "comment delays are non-negative", negative_delays == 0, negative_delays, 0, "high")
    add_check(checks, sub, "cleaned comments precede coverage boundary", comments_after_boundary == 0, comments_after_boundary, 0, "high")

    timing_rows: list[dict[str, object]] = []
    common_root_count = int((root_followup >= 168).sum())
    for horizon in EXPOSURE_HOURS:
        eligible_root_count = int((root_followup >= horizon).sum())
        for cohort_design, root_count, denominator, numerator in (
            ("horizon_specific", eligible_root_count, horizon_den[horizon], horizon_num[horizon]),
            ("common_7d", common_root_count, common_den, common_num[horizon]),
        ):
            timing_rows.append(
                {
                    "subreddit": sub,
                    "population": "final_cleaned_all_roots",
                    "cohort_design": cohort_design,
                    "horizon_hours": horizon,
                    "eligible_roots": root_count,
                    "observed_descendant_comments": denominator,
                    "comments_within_horizon": numerator,
                    "cumulative_comment_pct": numerator / denominator * 100 if denominator else np.nan,
                }
            )

    common_first = first_delay[(root_followup >= 168) & np.isfinite(first_delay)]
    first_rows = [
        {
            "subreddit": sub,
            "population": "final_cleaned_roots_with_at_least_7d_followup",
            "eligible_roots": common_root_count,
            "roots_with_observed_comment": len(common_first),
            "roots_with_observed_comment_pct": len(common_first) / common_root_count * 100 if common_root_count else np.nan,
            "minimum_hours": float(np.min(common_first)) if len(common_first) else np.nan,
            "p25_hours": float(np.quantile(common_first, 0.25)) if len(common_first) else np.nan,
            "median_hours": float(np.median(common_first)) if len(common_first) else np.nan,
            "mean_hours": float(np.mean(common_first)) if len(common_first) else np.nan,
            "p75_hours": float(np.quantile(common_first, 0.75)) if len(common_first) else np.nan,
            "p90_hours": float(np.quantile(common_first, 0.90)) if len(common_first) else np.nan,
            "p95_hours": float(np.quantile(common_first, 0.95)) if len(common_first) else np.nan,
            "maximum_hours": float(np.max(common_first)) if len(common_first) else np.nan,
        }
    ]
    stats = {
        "comment_rows": comment_rows,
        "comment_min_utc": iso_utc(pd.Timestamp(min_comment_ns, tz="UTC")) if min_comment_ns is not None else "",
        "comment_max_utc": iso_utc(pd.Timestamp(max_comment_ns, tz="UTC")) if max_comment_ns is not None else "",
        "started_roots_without_comment": started_without_comment,
    }
    return timing_rows, first_rows, stats


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    endpoints = pd.read_csv(args.endpoint_config)
    endpoints["endpoint_utc"] = pd.to_datetime(endpoints["endpoint_utc"], utc=True)
    endpoint_lookup = endpoints.set_index("subreddit")

    inventory = build_source_inventory(args.data_root, args.skip_hashes)
    inventory.to_csv(args.output_dir / "source_inventory.csv", index=False)
    missing = inventory[inventory["provenance_status"] == "missing_required_local_source"]
    if len(missing):
        raise FileNotFoundError("Required frozen sources are missing; see source_inventory.csv")

    checks: list[dict[str, object]] = []
    population_rows: list[dict[str, object]] = []
    exposure_rows: list[dict[str, object]] = []
    perf_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    cm_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    first_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    class_definition_rows: list[dict[str, object]] = []
    lead_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    blocking_failures: list[str] = []

    for sub in SUBREDDITS:
        sub_blocking_start = len(blocking_failures)
        endpoint = endpoint_lookup.loc[sub, "endpoint_utc"]
        endpoint_kind = endpoint_lookup.loc[sub, "endpoint_kind"]
        base0 = args.data_root / "Outputs" / "0_preprocessing" / sub
        threads_path = args.data_root / "Inputs" / f"{sub}_threads.parquet"
        comments_path = args.data_root / "Inputs" / f"{sub}_comments.parquet"
        extra_path = base0 / f"{sub}_threads_extra_feats.parquet"
        enriched_test_path = base0 / "tf-idf" / f"{sub}_svd_enriched_test_data.parquet"
        test_x_path = base0 / f"{sub}_test_X.parquet"
        test_y_path = base0 / f"{sub}_test_Y.parquet"
        s1_dir = args.data_root / "Outputs" / "1_thread_start" / sub / "4_model" / f"model_{S1_MODELS[sub]}"
        s2_dir = args.data_root / "Outputs" / "2_thread_size" / sub / "4_model" / f"model_{S2_MODELS[sub]}"

        threads = pd.read_parquet(threads_path, columns=["thread_id", "timestamp", "thread_size", "success"])
        extra = pd.read_parquet(extra_path, columns=["thread_id", "timestamp", "thread_size", "success", "log_thread_size"])
        add_check(checks, sub, "root thread_id non-null", threads["thread_id"].notna().all(), int(threads["thread_id"].isna().sum()), 0, "critical")
        add_check(checks, sub, "root thread_id unique", threads["thread_id"].is_unique, int(threads["thread_id"].duplicated().sum()), 0, "critical")

        input_key = threads[["thread_id", "timestamp", "thread_size", "success"]].reset_index(drop=True)
        extra_key = extra[["thread_id", "timestamp", "thread_size", "success"]].reset_index(drop=True)
        feature_match = frame_values_equal(input_key, extra_key)
        add_check(checks, sub, "feature output preserves final cleaned roots exactly", feature_match, feature_match, True, "critical", True)
        if not feature_match:
            blocking_failures.append(f"{sub}: cleaned roots do not match feature output")

        sorted_extra = extra.sort_values(by="timestamp").reset_index(drop=True)
        split_index = int(len(sorted_extra) * 0.8)
        train = sorted_extra.iloc[:split_index].copy()
        test = sorted_extra.iloc[split_index:].copy().reset_index(drop=True)
        enriched_test = pd.read_parquet(enriched_test_path, columns=["thread_id", "timestamp", "thread_size", "log_thread_size"])
        test_y = pd.read_parquet(test_y_path, columns=["timestamp", "thread_size", "log_thread_size"])
        test_x_rows = pq.ParquetFile(test_x_path).metadata.num_rows

        ordered_match = frame_values_equal(test[["thread_id", "timestamp", "thread_size", "log_thread_size"]],
            enriched_test[["thread_id", "timestamp", "thread_size", "log_thread_size"]].reset_index(drop=True)
        )
        add_check(checks, sub, "reconstructed chronological test IDs and order match frozen enriched test", ordered_match, ordered_match, True, "critical", True)
        if not ordered_match:
            blocking_failures.append(f"{sub}: chronological held-out ID/order mismatch")
        target_match = frame_values_equal(test[["timestamp", "thread_size", "log_thread_size"]], test_y.reset_index(drop=True))
        add_check(checks, sub, "reconstructed test targets match frozen test_Y", target_match, target_match, True, "critical", True)
        if not target_match:
            blocking_failures.append(f"{sub}: held-out target mismatch")
        count_match = len(test) == len(enriched_test) == len(test_y) == test_x_rows
        add_check(checks, sub, "held-out row counts reconcile across frozen artefacts", count_match, f"{len(test)}|{len(enriched_test)}|{len(test_y)}|{test_x_rows}", "all equal", "critical", True)
        if not count_match:
            blocking_failures.append(f"{sub}: held-out row-count mismatch")

        test["timestamp_utc"] = utc_series(test["timestamp"])
        test["potential_followup_hours"] = (endpoint - test["timestamp_utc"]).dt.total_seconds() / 3600
        test["endpoint_kind"] = endpoint_kind
        extra["timestamp_utc"] = utc_series(extra["timestamp"])
        extra["potential_followup_hours"] = (endpoint - extra["timestamp_utc"]).dt.total_seconds() / 3600
        extra["endpoint_kind"] = endpoint_kind
        add_check(checks, sub, "all cleaned root timestamps precede coverage boundary", bool((extra["potential_followup_hours"] >= 0).all()), int((extra["potential_followup_hours"] < 0).sum()), 0, "critical")

        model_metadata, bins = selected_model_metadata(args.data_root, sub)
        model_rows.append(model_metadata)

        s1_pred = pd.read_parquet(s1_dir / "test_started_threads.parquet")
        expected_index = np.arange(len(test), dtype=np.int64)
        s1_index_match = len(s1_pred) == len(test) and np.array_equal(s1_pred["index"].to_numpy(), expected_index)
        add_check(checks, sub, "Stage 1 saved predictions join one-to-one by held-out row index", s1_index_match, s1_index_match, True, "critical", True)
        if not s1_index_match:
            blocking_failures.append(f"{sub}: Stage 1 prediction linkage mismatch")
        test["s1_true"] = (test["thread_size"] > 1).astype(int)
        test["s1_pred"] = s1_pred["predicted"].to_numpy(dtype=int)

        s2_pred = pd.read_csv(s2_dir / "test_preds.csv")
        s2_saved_y = pd.read_parquet(s2_dir / "model_data" / "y_test.parquet")["y_test"].to_numpy(dtype=int)
        s2_index_match = len(s2_pred) == len(test) and np.array_equal(s2_pred["index"].to_numpy(dtype=int), expected_index)
        add_check(checks, sub, "Stage 2 saved predictions join one-to-one by held-out row index", s2_index_match, s2_index_match, True, "critical", True)
        if not s2_index_match:
            blocking_failures.append(f"{sub}: Stage 2 prediction linkage mismatch")
        saved_label_match = len(s2_saved_y) == len(s2_pred) and np.array_equal(s2_saved_y, s2_pred["true_class"].to_numpy(dtype=int))
        add_check(checks, sub, "Stage 2 CSV labels match saved y_test", saved_label_match, saved_label_match, True, "critical", True)
        if not saved_label_match:
            blocking_failures.append(f"{sub}: Stage 2 label mismatch")
        reconstructed_s2 = pd.cut(np.log(test["thread_size"]), bins=bins, labels=False, include_lowest=True).to_numpy(dtype=int)
        bins_match = np.array_equal(reconstructed_s2, s2_pred["true_class"].to_numpy(dtype=int))
        add_check(checks, sub, "Stage 2 labels reproduce from frozen bin edges", bins_match, bins_match, True, "critical", True)
        if not bins_match:
            blocking_failures.append(f"{sub}: Stage 2 bins do not reproduce labels")
        test["s2_true"] = s2_pred["true_class"].to_numpy(dtype=int)
        test["s2_pred"] = s2_pred["predicted_class"].to_numpy(dtype=int)

        if len(blocking_failures) > sub_blocking_start:
            continue

        for class_idx, class_name in S2_CLASS_NAMES.items():
            observed_sizes = test.loc[test["s2_true"] == class_idx, "thread_size"]
            class_definition_rows.append(
                {
                    "subreddit": sub,
                    "stage": "stage2",
                    "class_index": class_idx,
                    "observed_class": class_name,
                    "lower_log_edge": bins[class_idx],
                    "upper_log_edge": bins[class_idx + 1],
                    "right_closed": True,
                    "observed_heldout_min_thread_size": int(observed_sizes.min()),
                    "observed_heldout_max_thread_size": int(observed_sizes.max()),
                }
            )

        population_rows.append(
            {
                "subreddit": sub,
                "final_cleaned_roots": len(extra),
                "reconstructed_train_roots": len(train),
                "reconstructed_heldout_roots": len(test),
                "split_index": split_index,
                "train_fraction_realized": len(train) / len(extra),
                "heldout_start_utc": iso_utc(test["timestamp_utc"].min()),
                "heldout_end_utc": iso_utc(test["timestamp_utc"].max()),
                "coverage_boundary_utc": iso_utc(endpoint),
                "endpoint_kind": endpoint_kind,
                "stage1_stalled_n": int((test["s1_true"] == 0).sum()),
                "stage1_started_n": int((test["s1_true"] == 1).sum()),
                "stage2_stalled_n": int((test["s2_true"] == 0).sum()),
                "stage2_small_n": int((test["s2_true"] == 1).sum()),
                "stage2_medium_n": int((test["s2_true"] == 2).sum()),
                "stage2_large_n": int((test["s2_true"] == 3).sum()),
            }
        )

        add_exposure_rows(exposure_rows, sub, test, "root_population", None, None)
        add_exposure_rows(exposure_rows, sub, test, "stage1", "s1_true", S1_CLASS_NAMES)
        add_exposure_rows(exposure_rows, sub, test, "stage2", "s2_true", S2_CLASS_NAMES)
        add_sensitivity_rows(perf_rows, class_rows, cm_rows, sub, "stage1", test, "s1_true", "s1_pred", S1_CLASS_NAMES)
        add_sensitivity_rows(perf_rows, class_rows, cm_rows, sub, "stage2", test, "s2_true", "s2_pred", S2_CLASS_NAMES)

        for stage, names, true_col, pred_col, workbook in (
            ("stage1", S1_CLASS_NAMES, "s1_true", "s1_pred", s1_dir / "test_data_results.xlsx"),
            ("stage2", S2_CLASS_NAMES, "s2_true", "s2_pred", s2_dir / "test_data_results.xlsx"),
        ):
            cm = confusion(test[true_col].to_numpy(), test[pred_col].to_numpy(), len(names))
            computed = metrics_from_cm(cm)
            reference = workbook_reference_metrics(workbook, stage)
            for metric_name in ("mcc", "balanced_accuracy"):
                delta = abs(float(computed[metric_name]) - reference[metric_name])
                add_check(checks, sub, f"{stage} full held-out {metric_name} matches frozen workbook", delta < 1e-12, delta, "<1e-12", "critical")
            for class_idx in names:
                delta = abs(float(computed["recalls"][class_idx]) - reference[f"recall_{class_idx}"])
                add_check(checks, sub, f"{stage} full held-out recall class {class_idx} matches frozen workbook", delta < 1e-12, delta, "<1e-12", "critical")

        sub_timing, sub_first, comment_stats = scan_comments(sub, extra, comments_path, endpoint, checks)
        timing_rows.extend(sub_timing)
        first_rows.extend(sub_first)
        coverage_rows.append(
            {
                "subreddit": sub,
                "cleaned_root_rows": len(extra),
                "cleaned_root_min_utc": iso_utc(extra["timestamp_utc"].min()),
                "cleaned_root_max_utc": iso_utc(extra["timestamp_utc"].max()),
                "cleaned_comment_rows": comment_stats["comment_rows"],
                "cleaned_comment_min_utc": comment_stats["comment_min_utc"],
                "cleaned_comment_max_utc": comment_stats["comment_max_utc"],
                "coverage_boundary_utc": iso_utc(endpoint),
                "endpoint_kind": endpoint_kind,
            }
        )

        final_24_pct = float((extra["potential_followup_hours"] < 24).mean() * 100)
        matched_24 = next(
            row["cumulative_comment_pct"]
            for row in sub_timing
            if row["cohort_design"] == "horizon_specific" and row["horizon_hours"] == 24
        )
        lead_rows.extend(
            [
                {
                    "subreddit": sub,
                    "lead_metric": "raw roots in final 24 hours",
                    "preliminary_pct": LEAD_RAW_ROOT_24H[sub],
                    "reproduced_pct": np.nan,
                    "population_used": "not computed here; raw archive diagnostic is separate",
                    "status": "requires_raw_archive_output",
                },
                {
                    "subreddit": sub,
                    "lead_metric": "final cleaned roots in final 24 hours",
                    "preliminary_pct": LEAD_RAW_ROOT_24H[sub],
                    "reproduced_pct": final_24_pct,
                    "population_used": "final_cleaned_all_roots",
                    "status": "not_directly_comparable_to_raw_lead",
                },
                {
                    "subreddit": sub,
                    "lead_metric": "matched observed comments within 24 hours",
                    "preliminary_pct": LEAD_COMMENT_24H[sub],
                    "reproduced_pct": matched_24,
                    "population_used": "final_cleaned_horizon_specific_24h_eligible_roots",
                    "status": "reproduced_if_absolute_difference_le_0.1pp" if abs(matched_24 - LEAD_COMMENT_24H[sub]) <= 0.1 else "rejected_at_0.1pp_tolerance",
                },
            ]
        )

    pd.DataFrame(checks).to_csv(args.output_dir / "validation_checks.csv", index=False)
    if blocking_failures:
        raise AuditMismatch("Final population/prediction linkage mismatch: " + "; ".join(blocking_failures))

    outputs = {
        "population_reconciliation.csv": population_rows,
        "model_configuration.csv": model_rows,
        "target_class_definitions.csv": class_definition_rows,
        "endpoint_exposure.csv": exposure_rows,
        "sensitivity_performance.csv": perf_rows,
        "sensitivity_class_metrics.csv": class_rows,
        "sensitivity_confusion_matrices.csv": cm_rows,
        "reply_timing_comment_cumulative.csv": timing_rows,
        "reply_timing_first_comment.csv": first_rows,
        "preliminary_lead_comparison.csv": lead_rows,
        "data_coverage.csv": coverage_rows,
        "collection_endpoints_used.csv": endpoints.assign(endpoint_utc=endpoints["endpoint_utc"].map(iso_utc)).to_dict("records"),
    }
    for filename, records in outputs.items():
        pd.DataFrame(records).to_csv(args.output_dir / filename, index=False)

    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except Exception:
        git_head = "unavailable"
    run_record = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "data_root": str(args.data_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "repository_git_head": git_head,
        "python_version": sys.version,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "pyarrow_version": pa.__version__,
        "hashes_skipped": args.skip_hashes,
        "models_loaded_or_fitted": False,
        "predictions_regenerated": False,
    }
    (args.output_dir / "run_record.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(f"Audit outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
