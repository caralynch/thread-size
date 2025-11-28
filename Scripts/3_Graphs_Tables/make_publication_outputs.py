"""
make_publication_outputs.py

Unified Stage 1 & Stage 2 – collate evaluation artefacts for publication.

This script post-processes the outputs of the tuned model evaluation scripts

    * Stage_1_4_run_tuned_model.py  (thread start: stalled vs started)
    * Stage_2_4_run_tuned_model.py  (thread size: ordinal classes)

for a given stage. It assumes a directory
structure of the form:

    outputs_root/
        conspiracy/
            combined_scores.jl
            model_<n_feats>/
                ...
        crypto/
            combined_scores.jl
            model_<n_feats>/
                ...
        politics/
            combined_scores.jl
            model_<n_feats>/
                ...

where each subreddit subdirectory is exactly the --outdir that was passed to
Stage 1 4_run_tuned_model.py or Stage 2 4_run_tuned_model.py.

For a given stage, this script:

  * loads combined metric summaries (combined_scores.jl) for each subreddit;
  * generates publication-ready metric-vs-feature-count plots;
  * writes wide and long-format metric tables (including bootstrap CIs);
  * aggregates the chosen "best" models across subreddits into a CSV; and
  * plots confusion matrices for OOF and test predictions, exporting the
    underlying matrices and bootstrap summaries.

All figures are written as PNG and EPS. All numerical data behind the plots –
including means and bootstrap confidence intervals – are exported as CSV or
Excel files to satisfy journal reproducibility requirements.

Run once per stage, with different --outputs-root / --outdir values, e.g.:

    # Stage 1 (thread start)
    python make_publication_outputs.py \\
        --stage 1 \\
        --outputs-root Outputs/1_Thread_Start \\
        --selected-models publication_stage1/stage1_selected_models.csv \\
        --outdir publication_stage1

    # Stage 2 (thread size)
    python make_publication_outputs.py \\
        --stage 2 \\
        --outputs-root Outputs/2_Thread_Size \\
        --selected-models publication_stage2/stage2_selected_models.csv \\
        --n-classes 4 \\
        --outdir publication_stage2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.rcParams["font.family"] = "Arial"

COLORS = sns.color_palette("colorblind", n_colors=4)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pretty labels for subreddits – fall back to raw name if missing
SUBREDDIT_LABELS = {
    "conspiracy": "r/Conspiracy",
    "crypto": "r/CryptoCurrency",
    "politics": "r/politics",
}

DATA_LABELS = {
    "oof": "OOF",
    "test": "Test",
}

CLASS_NAMES_STAGE1 = ["Stalled", "Started"]

CLASS_NAMES_STAGE2 = {
    3: ["Stalled", "Small", "Large"],
    4: ["Stalled", "Small", "Medium", "Large"],
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ci_to_str(value: Iterable[float]) -> str:
    """Format a bootstrap CI [lower, upper] as a string."""
    lower, upper = float(value[0]), float(value[1])
    return f"[{lower:.4f}, {upper:.4f}]"


def load_selected_models(path: Path) -> Dict[str, int]:
    """
    Load mapping subreddit -> n_feats.

    Supports two formats:

      1) CSV with header: columns 'subreddit' and 'n_feats'.
      2) Plain text file with one line per subreddit:
             subreddit,n_feats[,<ignored>]
    """
    # First, try CSV with headers
    try:
        df = pd.read_csv(path)
        if {"subreddit", "n_feats"}.issubset(df.columns):
            return {
                str(row["subreddit"]): int(row["n_feats"]) for _, row in df.iterrows()
            }
    except Exception:
        pass

    # Fallback: simple line-parsing
    selected: Dict[str, int] = {}
    with path.open() as f:
        for line in f:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) < 2:
                continue
            subreddit, n_feats = parts[0], parts[1]
            selected[subreddit] = int(n_feats)

    if not selected:
        raise ValueError(
            f"Could not parse selected-models file {path}. "
            "Expected either a CSV with 'subreddit' and 'n_feats' columns or "
            "lines of the form 'subreddit,n_feats'."
        )
    return selected


def load_combined_summaries(
    selected_models: Mapping[str, int],
    outputs_root: Path,
) -> Tuple[
    Dict[str, Dict[str, pd.DataFrame]],  # summary_dfs[subreddit][data_type]
    List[str],  # metrics
]:
    """
    Load combined_scores.jl for each subreddit and return:

      - summary_dfs: subreddit -> {data_type: DataFrame}
      - metrics: common list of metric names (intersection across subreddits)
    """
    summary_dfs: Dict[str, Dict[str, pd.DataFrame]] = {}
    metrics: List[str] | None = None

    for subreddit in selected_models:
        stage_dir = outputs_root / subreddit
        combined_scores_path = stage_dir / "4_model/combined_scores.jl"
        if not combined_scores_path.is_file():
            raise FileNotFoundError(
                f"combined_scores.jl not found for {subreddit} at {combined_scores_path}"
            )

        combined_scores = joblib.load(combined_scores_path)
        # combined_scores is a dict keyed by "test"/"oof" -> dict of lists
        summary_dfs[subreddit] = {
            key: pd.DataFrame(results) for key, results in combined_scores.items()
        }

        # Derive metric names from keys (exclude CIs and n_feats)
        keys = combined_scores["oof"].keys()
        candidate_metrics = [k for k in keys if k != "n_feats" and not k.endswith("CI")]
        if metrics is None:
            metrics = candidate_metrics
        else:
            metrics = [m for m in metrics if m in candidate_metrics]

    if metrics is None:
        metrics = []

    # Stable ordering: put common "headline" metrics first if present
    priority = [
        "AUC",
        "MCC",
        "F1",
        "Balanced accuracy",
        "Precision",
        "Recall",
    ]
    metrics = sorted(
        metrics,
        key=lambda x: (priority.index(x) if x in priority else len(priority), x),
    )
    return summary_dfs, metrics


# ---------------------------------------------------------------------------
# Metrics – enrichment, tables, plots
# ---------------------------------------------------------------------------


def enrich_metric_tables(
    summary_dfs: Mapping[str, Mapping[str, pd.DataFrame]],
    metrics: List[str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    For each subreddit and data_type, compute:

      - ratio_max_<metric>
      - CI_width_<metric>
      - <metric>_%improvement (relative change vs previous n_feats)
    """
    for subreddit in summary_dfs:
        for key, df in summary_dfs[subreddit].items():
            df = df.copy()
            for metric in metrics:
                max_val = df[metric].max()
                df[f"ratio_max_{metric}"] = df[metric] / max_val

                # CI width: upper - lower
                df[f"CI_width_{metric}"] = df[f"{metric} CI"].apply(
                    lambda x: (
                        x[1] - x[0]
                        if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 2
                        else np.nan
                    )
                )

                # % improvement vs previous n_feats
                df[f"{metric}_%improvement"] = df[metric].diff() / df[metric] * 100

            summary_dfs[subreddit][key] = df
    return summary_dfs


def format_metric_tables_for_excel(
    summary_dfs: Mapping[str, Mapping[str, pd.DataFrame]],
    metrics: List[str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Create nicely formatted tables (rounded, CI as strings) for Excel.

    Returns a nested dict:

        formatted[subreddit][data_type] -> DataFrame indexed by n_feats.
    """
    formatted: Dict[str, Dict[str, pd.DataFrame]] = {}

    for subreddit in summary_dfs:
        formatted[subreddit] = {}
        for data_type, df in summary_dfs[subreddit].items():
            tmp = df.copy()

            # Round numeric values
            tmp = tmp.applymap(
                lambda x: (
                    f"{x:.4f}"
                    if isinstance(x, (float, np.floating)) and np.isfinite(x)
                    else x
                )
            )

            # CIs as strings
            ci_cols = [c for c in df.columns if c.endswith("CI")]
            tmp[ci_cols] = df[ci_cols].map(ci_to_str)

            # Ensure n_feats is int and index
            tmp["n_feats"] = df["n_feats"].astype(int)
            tmp.set_index("n_feats", inplace=True)

            # Column ordering: group by metric
            ordered_cols: List[str] = []
            for metric in metrics:
                ordered_cols.extend(
                    [
                        metric,
                        f"{metric} CI",
                        f"ratio_max_{metric}",
                        f"CI_width_{metric}",
                        f"{metric}_%improvement",
                    ]
                )
            tmp = tmp[ordered_cols]
            formatted[subreddit][data_type] = tmp

    return formatted


def save_metric_tables_to_excel(
    formatted_summary_dfs: Mapping[str, Mapping[str, pd.DataFrame]],
    outdir: Path,
    stage: int,
) -> None:
    """Write formatted metric tables to a single Excel workbook."""
    out_path = outdir / f"metrics_stage{stage}.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        for subreddit, data_dict in formatted_summary_dfs.items():
            for data_type, df in data_dict.items():
                sheet = f"{subreddit}_{data_type}"
                df.to_excel(writer, sheet_name=sheet)


def save_metric_long_csv(
    summary_dfs: Mapping[str, Mapping[str, pd.DataFrame]],
    metrics: List[str],
    outdir: Path,
    stage: int,
) -> None:
    """
    Export a long-format CSV with all metric values and CIs across:

      stage, subreddit, data_type, n_feats, metric, value, ci_lower, ci_upper.

    This file contains the data behind the metric-vs-n_feats plots.
    """
    records: List[Dict[str, object]] = []
    for subreddit, data_dict in summary_dfs.items():
        for data_type, df in data_dict.items():
            for _, row in df.iterrows():
                n_feats = int(row["n_feats"])
                for metric in metrics:
                    ci_val = row[f"{metric} CI"]
                    if (
                        isinstance(ci_val, (list, tuple, np.ndarray))
                        and len(ci_val) == 2
                    ):
                        lower, upper = ci_val
                    else:
                        lower, upper = (np.nan, np.nan)
                    records.append(
                        {
                            "stage": stage,
                            "subreddit": subreddit,
                            "data_type": data_type,
                            "n_feats": n_feats,
                            "metric": metric,
                            "value": row[metric],
                            "ci_lower": lower,
                            "ci_upper": upper,
                        }
                    )

    df_long = pd.DataFrame.from_records(records)
    df_long.to_csv(outdir / f"metrics_stage{stage}_long.csv", index=False)


def save_selected_models_table(
    selected_models: Mapping[str, int],
    formatted_summary_dfs: Mapping[str, Mapping[str, pd.DataFrame]],
    metrics: List[str],
    outdir: Path,
    stage: int,
) -> None:
    """
    Write a compact table with the chosen n_feats and metrics per subreddit.

    Uses the test split for metric values and formatted CIs.
    """
    rows: List[pd.DataFrame] = []

    for subreddit, n_feats in selected_models.items():
        df_test = formatted_summary_dfs[subreddit]["test"]
        row = df_test.loc[[n_feats]].copy()
        row["n_feats"] = n_feats
        row["Subreddit"] = SUBREDDIT_LABELS.get(subreddit, subreddit)

        cols_to_keep = ["Subreddit", "n_feats"]
        for metric in metrics:
            cols_to_keep.extend([metric, f"{metric} CI"])
        row = row[cols_to_keep]
        rows.append(row)

    out_df = pd.concat(rows, ignore_index=True)
    out_df.to_csv(outdir / f"selected_models_stage{stage}.csv", index=False)


def plot_metrics(
    summary_dfs: Mapping[str, Mapping[str, pd.DataFrame]],
    metrics: List[str],
    selected_models: Mapping[str, int],
    outdir: Path,
    stage: int,
) -> None:
    """Plot metric vs n_feats curves, saving PNG/EPS per metric."""
    subreddits = list(selected_models.keys())
    n_subs = len(subreddits)

    # Use a 2x2 grid for up to 4 subreddits; extend if needed later
    n_rows, n_cols = 2, 2

    for metric in metrics:
        png_path = outdir / f"metric_stage{stage}_{metric.replace(' ', '_')}.png"
        eps_path = outdir / f"metric_stage{stage}_{metric.replace(' ', '_')}.eps"

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 9))
        axes = axes.flatten()

        handles_dict = {}
        for idx, subreddit in enumerate(subreddits):
            ax = axes[idx]
            data_dict = summary_dfs[subreddit]

            # Plot OOF and Test for this subreddit
            for i, (data_type, label) in enumerate(DATA_LABELS.items()):
                df = pd.DataFrame(data_dict[data_type])

                lower = [
                    metric_val - ci[0]
                    for metric_val, ci in zip(df[metric], df[f"{metric} CI"])
                ]
                upper = [
                    ci[1] - metric_val
                    for metric_val, ci in zip(df[metric], df[f"{metric} CI"])
                ]

                line = ax.errorbar(
                    df["n_feats"],
                    df[metric],
                    yerr=[lower, upper],
                    fmt="o-",
                    capsize=3,
                    color=COLORS[i],
                    ecolor=COLORS[i],
                    label=label,
                )
                handles_dict[label] = line

            ax.set_title(
                SUBREDDIT_LABELS.get(subreddit, subreddit),
                loc="left",
                fontsize=12,
                weight="bold",
            )
            ax.set_xlabel("Number of features", fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)
            ax.set_xticks(df["n_feats"], labels=df["n_feats"].astype(int))

        # Remove unused axes
        for j in range(len(subreddits), len(axes)):
            fig.delaxes(axes[j])

        # Shared legend
        handles = [handles_dict[key] for key in DATA_LABELS.values()]
        fig.legend(
            handles=handles,
            labels=list(DATA_LABELS.values()),
            loc="center right",
            fontsize=10,
            title="Data split",
            title_fontsize=11,
            frameon=True,
            fancybox=True,
            bbox_to_anchor=(0.95, 0.5),
        )

        plt.tight_layout()
        plt.savefig(eps_path, dpi=350, format="eps", bbox_inches="tight")
        plt.savefig(png_path, dpi=400, format="png", bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------


def load_confusion_matrix_data(
    stage_dir: Path,
    n_feats: int,
    data_type: str,
) -> object:
    """
    Load confusion-matrix bootstrap data for a given subreddit / n_feats / split.

    To be resilient to historical runs and future harmonisation, this tries
    several filename patterns in order and returns the first one that exists:

      1) {stage_dir}/model_{n_feats}/model_data/{data_type}_confusion_matrix_data.jl
         (current Stage 2 pattern)

      2) {stage_dir}/model_{n_feats}/{data_type}_{n_feats}_confusion_matrix_data.jl
         (current Stage 1 pattern)

      3) {stage_dir}/model_{n_feats}/{data_type}_confusion_matrix_data.jl
         (proposed harmonised pattern; safe to adopt in both stages later)
    """
    candidates = [
        stage_dir / "4_model"
        / f"model_{n_feats}"
        / "model_data"
        / f"{data_type}_confusion_matrix_data.jl",
        stage_dir / "4_model"
        / f"model_{n_feats}"
        / f"{data_type}_{n_feats}_confusion_matrix_data.jl",
        stage_dir / "4_model" / f"model_{n_feats}" / f"{data_type}_confusion_matrix_data.jl",
    ]

    for path in candidates:
        if path.is_file():
            return joblib.load(path)

    raise FileNotFoundError(
        f"No confusion-matrix data found for n_feats={n_feats}, "
        f"data_type='{data_type}' in {stage_dir}"
    )


def save_confusion_matrix_csvs(
    cm_data_obj,
    subreddit: str,
    data_type: str,
    class_names: List[str],
    outdir: Path,
    stage: int,
) -> None:
    """
    Save confusion-matrix point estimate and, if available, bootstrap summaries
    (mean / lower / upper / std) to CSV.

    Filenames are:

        cm_stage{stage}_{data_type}_{subreddit}.csv        # point estimate
        cm_stage{stage}_{data_type}_{subreddit}_long.csv   # long/tidy format
        cm_stage{stage}_{data_type}_{subreddit}_mean.csv   # bootstrap mean
        cm_stage{stage}_{data_type}_{subreddit}_lower.csv  # lower CI bound
        cm_stage{stage}_{data_type}_{subreddit}_upper.csv  # upper CI bound
        cm_stage{stage}_{data_type}_{subreddit}_std.csv    # bootstrap SD
    """
    if isinstance(cm_data_obj, dict):
        cm = cm_data_obj.get("CM")
        cm_dict = cm_data_obj
    else:
        cm = cm_data_obj
        cm_dict = {"CM": cm}

    base_name = f"cm_stage{stage}_{data_type}_{subreddit}"

    # Wide format (point estimate matrix)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.index.name = "true_class"
    df_cm.to_csv(outdir / f"{base_name}.csv")

    # Long format
    df_long = df_cm.reset_index().melt(
        id_vars="true_class", var_name="pred_class", value_name="count"
    )
    df_long.to_csv(outdir / f"{base_name}_long.csv", index=False)

    # If bootstrap summaries exist, save each as its own matrix CSV
    for key in ["mean", "lower", "upper", "std"]:
        if key in cm_dict:
            df_tmp = pd.DataFrame(cm_dict[key], index=class_names, columns=class_names)
            df_tmp.index.name = "true_class"
            df_tmp.to_csv(outdir / f"{base_name}_{key}.csv")


def plot_confusion_matrices(
    selected_models: Mapping[str, int],
    outputs_root: Path,
    class_names: List[str],
    outdir: Path,
    stage: int,
) -> None:
    """
    Plot confusion matrices and export underlying data.

    One figure per data_type (OOF/Test), with up to 3 subplots (one per subreddit).
    """
    data_types = list(DATA_LABELS.keys())
    subreddits = list(selected_models.keys())
    n_rows, n_cols = 2, 2  # 3 subplots + one empty

    for data_type in data_types:
        png_path = outdir / f"cm_stage{stage}_{data_type}.png"
        eps_path = outdir / f"cm_stage{stage}_{data_type}.eps"

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 9))
        axes = axes.flatten()

        for i, subreddit in enumerate(subreddits):
            ax = axes[i]
            stage_dir = outputs_root / subreddit
            n_feats = selected_models[subreddit]

            cm_data = load_confusion_matrix_data(stage_dir, n_feats, data_type)

            if isinstance(cm_data, dict):
                cm = cm_data.get("CM")
            else:
                cm = cm_data

            # Save CSVs (point estimate + bootstrap summaries)
            save_confusion_matrix_csvs(
                cm_data,
                subreddit=subreddit,
                data_type=data_type,
                class_names=class_names,
                outdir=outdir,
                stage=stage,
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
                annot_kws={"fontsize": 10},
                cbar_kws={"shrink": 0.9},
            )
            ax.set_title(
                SUBREDDIT_LABELS.get(subreddit, subreddit),
                loc="left",
                fontsize=12,
                weight="bold",
            )
            ax.set_xlabel("Predicted class", fontsize=11)
            ax.set_ylabel("True class", fontsize=11)

        # Remove unused axes
        for j in range(len(subreddits), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(eps_path, dpi=350, format="eps", bbox_inches="tight")
        plt.savefig(png_path, dpi=400, format="png", bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collate Stage 1 or Stage 2 evaluation outputs into "
            "publication-ready figures and tables."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=[1, 2],
        help="Model stage: 1 = thread start, 2 = thread size.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        required=True,
        help=(
            "Root directory for this stage, containing per-subreddit folders. "
            "Each subreddit folder must be the --outdir used when running the "
            "corresponding Stage_1_4_run_tuned_model.py or Stage_2_4_run_tuned_model.py."
        ),
    )
    parser.add_argument(
        "--selected-models",
        type=Path,
        required=True,
        help=(
            "CSV or text file specifying the chosen model per subreddit "
            "(columns: subreddit,n_feats)."
        ),
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=None,
        help=(
            "Number of Stage 2 classes (3 or 4). Required if --stage 2. "
            "Ignored for --stage 1."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory where publication-ready figures and tables will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, regenerate outputs even if summary files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage == 2 and args.n_classes is None:
        raise ValueError(
            "You must specify --n-classes (3 or 4) when --stage 2 is selected."
        )

    ensure_dir(args.outdir)

    print(f"[INFO] Publication outputs for Stage {args.stage}")
    print(f"[INFO] outputs_root: {args.outputs_root}")
    print(f"[INFO] outdir: {args.outdir}")

    # Light short-circuit: if core outputs exist and overwrite is False, skip work
    core_files = [
        f"metrics_stage{args.stage}.xlsx",
        f"metrics_stage{args.stage}_long.csv",
        f"selected_models_stage{args.stage}.csv",
    ]
    if not args.overwrite and all(
        (args.outdir / fname).is_file() for fname in core_files
    ):
        print(
            "[INFO] Core publication tables already exist; use --overwrite to regenerate."
        )
        return

    # 1) Selected models & combined scores
    selected_models = load_selected_models(args.selected_models)
    print(f"[INFO] Selected models (subreddit -> n_feats): {selected_models}")

    summary_dfs, metrics = load_combined_summaries(
        selected_models,
        outputs_root=args.outputs_root,
    )
    print(f"[INFO] Metrics found in combined_scores: {metrics}")

    # 2) Metric enrichment + tables + plots
    summary_dfs = enrich_metric_tables(summary_dfs, metrics)
    formatted_summary_dfs = format_metric_tables_for_excel(summary_dfs, metrics)

    save_metric_tables_to_excel(formatted_summary_dfs, args.outdir, stage=args.stage)
    save_metric_long_csv(summary_dfs, metrics, args.outdir, stage=args.stage)
    save_selected_models_table(
        selected_models, formatted_summary_dfs, metrics, args.outdir, stage=args.stage
    )
    plot_metrics(summary_dfs, metrics, selected_models, args.outdir, stage=args.stage)

    # 3) Confusion matrices (plots + CSVs including bootstrap summaries)
    if args.stage == 1:
        class_names = CLASS_NAMES_STAGE1
    else:
        class_names = CLASS_NAMES_STAGE2[args.n_classes]

    plot_confusion_matrices(
        selected_models,
        outputs_root=args.outputs_root,
        class_names=class_names,
        outdir=args.outdir,
        stage=args.stage,
    )

    print("[INFO] Finished publication outputs for Stage", args.stage)


if __name__ == "__main__":
    main()
