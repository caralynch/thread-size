import argparse
from pathlib import Path
import os
from typing import Dict, Iterable, List, Mapping, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from PIL import Image

# Matplotlib style
plt.rcParams["font.family"] = "Arial"

COLORS = sns.color_palette("colorblind", n_colors=4)

# Constants
SUBREDDIT_LABELS = {
    "conspiracy": "r/Conspiracy",
    "crypto": "r/CryptoCurrency",
    "politics": "r/politics",
}


LETTER_LOOKUP = {
    0: "(a)",
    1: "(b)",
    2: "(c)",
    3: "(d)",
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

S1_METRICS = ["MCC", "Balanced accuracy", "AUC", "F1", "F-beta", "Precision", "Recall"]

S2_METRICS = ["MCC", "Balanced accuracy", "F1"]

INDEX_COL = "n_feats"

def format_label(text, max_word_length=20):
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    
    # Capitalize first letter of each word
    words = text.split()
    formatted_words = []
    
    for i,word in enumerate(words):
        if word.lower() == "avg":
            word += "."
        if word.lower() == "pagerank":
            formatted_words.append("PageRank")
        elif (word.lower() == "reddit") or (i == 0):
            formatted_words.append(word.capitalize())
        # Keep known acronyms uppercase, otherwise title case
        elif word.upper() in ['PR', 'URL', 'ID']:
            formatted_words.append(word.upper())
        else:
            formatted_words.append(word.lower())
    
    # Greedily pack words into lines
    if len(formatted_words) <= 1:
        return ' '.join(formatted_words)
    
    lines = []
    current_line = [formatted_words[0]]
    
    for word in formatted_words[1:]:
        # Check if adding this word would exceed max length
        test_line = ' '.join(current_line + [word])
        if len(test_line) <= max_word_length:
            current_line.append(word)
        else:
            # Start a new line
            lines.append(' '.join(current_line))
            current_line = [word]
    
    # Add the last line
    lines.append(' '.join(current_line))
    
    return '\n'.join(lines)

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


def get_cm_data(selected_model_dirs, data_type="test"):
    cms = {}
    for subreddit, indir in selected_model_dirs.items():
        cms[subreddit] = joblib.load(f"{indir}/{data_type}_confusion_matrix_data.jl")
    return cms


def plot_cm(cm, class_names, ax):
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"fontsize": 10},  # annotation text size
        cbar_kws={"shrink": 0.9},
    )

def make_cm_fig(cm_dicts, outfile, class_names):
    i = 0
    fig, axes = plt.subplots(2,2, figsize=(12,10))
    axes = axes.flatten()
    for subreddit, cm_dict in cm_dicts.items():
        plot_cm(cm_dict["CM"], class_names, axes[i])
        axes[i].set_title(
            f"{SUBREDDIT_LABELS[subreddit]}",
            loc="left",
            fontsize=12,
            weight="bold",
        )
        axes[i].set_xlabel("Predicted Class", fontsize=11)
        axes[i].set_ylabel("True Class", fontsize=11)
        i+=1
    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(
        f"{outfile}.eps",
        dpi=350,
        format="eps",
        bbox_inches="tight",
    )
    plt.savefig(
        f"{outfile}.png",
        dpi=400,
        format="png",
        bbox_inches="tight",
    )
    plt.close()

def confusion_matrixes(selected_model_dirs, outdir, plot_outdir, class_names):
    cms = {}
    print(f"[INFO] Loading test and OOF set confusion matrix data")
    for k in DATA_LABELS:
        cms[k] = get_cm_data(selected_model_dirs, data_type=k)
        print(f"[INFO] Saving {k} confusion matrix to {outdir}/{k}_CM")
        make_cm_fig(cms[k], f"{args.outdir}/{k}_CM", class_names)
        print(f"[INFO] Saving {k} confusion matrix data to {plot_outdir}.")
        for sub in cms[k]:
            with pd.ExcelWriter(f"{plot_outdir}/{sub}_{k}_cm_data.xlsx") as writer:
                for j, cm in cms[k][sub].items():
                    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
                    df_cm.index.name="true_class"
                    df_cm.to_excel(writer, sheet_name=j)


def performance_metrics(mod_dirs, outdir, metrics, plot_outdir):
    print(f"[INFO] Loading performance metrics.")
    combined_scores = dict(zip(DATA_LABELS.keys(), [{}, {}]))
    for sub, mod_dir in mod_dirs.items():
        sub_scores = joblib.load(f"{mod_dir}/combined_scores.jl")
        for k in combined_scores:
            combined_scores[k][sub] = pd.DataFrame(sub_scores[sub][k])

        # don't want to try to graph metrics that aren't in the data
        metrics = list(set(metrics) & set(sub_scores["test"].keys()))
    
    print(f"[INFO] Plotting performance metrics.")
    for metric in metrics:
        plot_metric(metric, combined_scores, f"{outdir}/{metric}")

    perf_xlsx = f"{plot_outdir}/performance_metrics.xlsx"
    print(f"[INFO] Saving performance metrics to {perf_xlsx}.")
    with pd.ExcelWriter(perf_xlsx) as writer:
        for k, sub_dict in combined_scores.items():
            for sub, df in sub_dict.items():
                df.to_excel(writer, sheet_name=f"{sub}_{k}")

    output_tabs_xlsx = f"{outdir}/table_metrics.xlsx"
    print(f"[INFO] Saving summary dfs to {output_tabs_xlsx}")
    with pd.ExcelWriter(output_tabs_xlsx) as writer:
        for k, sub_dict in combined_scores.items():
            for sub, df in sub_dict.items():
                formatted_df = format_df_for_pretty_output(df)
                formatted_df.to_excel(writer, sheet_name=f"{sub}_{k}")
    
    output_ratios_xlsx = f"{outdir}/metric_ratios.xlsx"
    print(f"[INFO] Saving metric ratios to {output_ratios_xlsx}")
    with pd.ExcelWriter(output_tabs_xlsx) as writer:
        for k, sub_dict in combined_scores.items():
            for sub, df in sub_dict.items():
                output_df = get_ratio_max_metric(df, metrics)
                output_df.to_excel(writer, sheet_name=f"{sub}_{k}")
    return metrics

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
        "--root",
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


def get_errorbars(df, metric):
    lower = [
        metric_val - ci[0]
        for metric_val, ci in zip(
            df[metric], df[f"{metric} CI"]
        )
    ]
    upper = [
        ci[1] - metric_val
        for metric_val, ci in zip(
            df[metric], df[f"{metric} CI"]
        )
    ]
    return lower, upper


def plot_metric(metric, combined_scores, outfile):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    handles_dict = {}
    j=0
    for data_type, sub_dicts in combined_scores.items():
        i = 0
        for sub, df in sub_dicts.items():
            ax = axes[i]

            lower, upper = get_errorbars(df, metric)

            line = ax.errorbar(
                df["n_feats"],
                df[f"{metric}"],
                yerr=[lower, upper],
                fmt="o-",
                capsize=3,
                color=COLORS[i],
                ecolor=COLORS[i],
            )
            i+=1
            handles_dict[DATA_LABELS[data_type]] = line
            if j == 0:
                ax.set_title(
                    f"{SUBREDDIT_LABELS[sub]}", loc="left", fontsize=12, weight="bold"
                )
                if i > 0:
                    ax.set_xlabel("Number of features", fontsize=11)
                if i % 2 == 0:
                    ax.set_ylabel(metric, fontsize=11)
                ax.tick_params(axis="x", labelsize=10)
                ax.tick_params(axis="y", labelsize=10)
                ax.set_xticks(df["n_feats"], labels=df["n_feats"].astype(int))
        j+=1
    handles = [handles_dict[d] for d in handles_dict]
    fig.legend(
        handles=handles,
        labels=handles_dict.keys(),
        loc="center right",
        fontsize=10,
        title="Metric",
        title_fontsize=11,
        frameon=True,  # ✅ Enables the box
        edgecolor="black",  # Optional: border color
        fancybox=True,
        ncol=1,
        bbox_to_anchor=(0.65, 0.4),
    )
    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(f"{outfile}.eps", dpi=350, format="eps")
    plt.savefig(f"{outfile}.png", dpi=400, format="png")
    plt.close()

def get_ci_str_from_list(value):
    return f"[{float(value[0]):.4f}, {float(value[1]):.4f}]"

def make_n_index(df):
    df[INDEX_COL] = df[INDEX_COL].astype(int)
    df.set_index(INDEX_COL, inplace=True)
    return df
        
def format_df_for_pretty_output(df):
    formatted_df = df.copy()
    
    # n_feats should be only int col and usable as index
    if INDEX_COL in formatted_df.columns:
        formatted_df = make_n_index(formatted_df)

    # have all float cols have 4 digits after 0
    formatted_df = formatted_df.applymap(
        lambda x: f"{x:.4f}" if isinstance(x, (float)) and pd.notnull(x) else x
    )

    # if CI cols in df, want to have them as ranges [lower, upper]
    formatted_df[
        [x for x in formatted_df.columns if x.endswith("CI")]
    ] = formatted_df[
        [x for x in formatted_df.columns if x.endswith("CI")]
    ].applymap(
        get_ci_str_from_list
    )

    return formatted_df

def get_ratio_max_metric(df, metrics):
    if INDEX_COL in df.columns:
        df = make_n_index(df)
    new_df = df[metrics]
    for metric in metrics:
        max = new_df[metric].max()
        new_df[f"ratio_max_{metric}"] = new_df[metric] / max
    return new_df

def get_sub_shap_plot(sub_shap_dict, outfile):
    plt.figure(figsize=(10,6))
    shap.summary_plot(
        sub_shap_dict["shap_val"],
        sub_shap_dict["feat_name"],
        plot_type="dot",
        show=False,
        feature_names = [format_label(x) for x in sub_shap_dict["feat_name"].columns]
    )
    fig = plt.gcf()
    axes = fig.get_children()
    main_ax = [
        a
        for a in axes
        if isinstance(a, plt.Axes) and "SHAP value" in a.get_xlabel()
    ][0]
    colorbar_ax = [
        a
        for a in axes
        if isinstance(a, plt.Axes) and "Feature value" in a.get_ylabel()
    ][0]

    colorbar_ax.set_ylabel("Feature value", fontsize=11)
    colorbar_ax.tick_params(labelsize=10)  # ticks along the colorbar

    main_ax.set_title(
        f"{SUBREDDIT_LABELS[sub]}", loc="left", fontsize=12, weight="bold"
    )
    main_ax.set_xlabel(
        "SHAP value", fontsize=11,
    )
    main_ax.set_ylabel("Feature", fontsize=11)
    main_ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(f"{outfile}.eps", dpi=350, format="eps")
    plt.savefig(f"{outfile}.png", dpi=400, format="png")
    plt.close()

def combine_plots_vertical(png_filenames, outfile):
    # Load saved plots
    imgs = [Image.open(f) for f in png_filenames]

    # Determine width and total height
    max_width = np.max([img.width for img in imgs])
    total_height = np.sum([img.height for img in imgs])

    # Create new blank image (white background)
    combined = Image.new("RGB", (max_width, total_height), (255, 255, 255))

    # Paste images one below the other
    y_offset = 0
    for img in imgs:
        combined.paste(img, (0, y_offset))
        y_offset += img.height
    for ext in ["png", "eps"]:
        combined.save(f"{outfile}.{ext}", dpi=(400, 400))

def plot_s1_shap_vals(selected_model_dirs, outdir):
    shap_vals = {}
    shap_outfiles = []

    print("[INFO] Making SHAP plots for each subreddit")
    for sub, mod_dir in selected_model_dirs.items():
        shap_vals[sub] = joblib.load(f"{mod_dir}/shap_plot_data.jl")
        outfile_name = f"{outdir}/{sub}_shap"
        shap_outfiles.append(f"{outfile_name}.png")
        get_sub_shap_plot(shap_vals[sub], outfile_name)
    
    print("[INFO] Combining SHAP plots")
    combine_plots_vertical(shap_outfiles, f"{outdir}/combined_shap")
        


def output_s1_feature_importances(selected_model_dirs, outfile):
    # get shap importance dfs and model classifiers
    shap_importance_dfs = {}
    mod_classifiers = {}
    print("[INFO] Reading in SHAP importance and model data")
    for sub, mod_dir in selected_model_dirs.items():
        shap_importance_dfs[sub] = joblib.load(f"{mod_dir}/shap_importance_df.jl")
        mod_classifiers[sub] = joblib.load(f"{mod_dir}/model.jl")["classifier"]
    
    print("[INFO] Getting tree importance")
    feat_importance_dfs = {}
    for sub, shap_imp_df in shap_importance_dfs.items():
        tree_importance = get_tree_importance(mod_classifiers[sub])
        feat_importance_df = get_feat_importance_df(shap_imp_df, tree_importance)
        feat_importance_dfs[sub] = format_df_for_pretty_output(feat_importance_df)
    
    print(f"[INFO] Saving feature importances to {outfile}")
    with pd.ExcelWriter(outfile) as writer:
        for sub, df in feat_importance_dfs.items():
            df.to_excel(writer, sheet_name=sub, index=True)

def get_feat_importance_df(shap_importance_df, tree_importance_df):
    feat_importance_df = pd.merge(
        shap_importance_df, tree_importance_df, on="Feature"
    ).sort_values(
        by="MeanAbsoluteSHAP", ascending=False
    )
    feat_importance_df["Feature"] = feat_importance_df["Feature"].apply(format_label)
    return feat_importance_df

def get_tree_importance(lgbm_classifier):
    tree_importance_df = pd.DataFrame(
        [lgbm_classifier.booster_.feature_name(),
        lgbm_classifier.booster_.feature_importance(),
        lgbm_classifier.booster_.feature_importance(importance_type="gain")],
        index=["Feature", "Split importance", "Gain importance"],
    ).T
    return tree_importance_df



def main() -> None:
    print(f"{sys.argv[0]}")
    start = dt.datetime.now()
    print("[INFO] Generating publication outputs.")
    print(f"[INFO] STARTED AT {start}")
    args = parse_args()

    if args.stage == 2 and args.n_classes is None:
        raise ValueError(
            "You must specify --n-classes (3 or 4) when --stage 2 is selected."
        )
    
    if args.stage == 2:
        class_names = CLASS_NAMES_STAGE2[args.n_classes]
        metrics = S2_METRICS
    else:
        class_names = CLASS_NAMES_STAGE1
        metrics = S1_METRICS
    
    for class_name in class_names:
        metrics.append(f"Precision {class_name}")

    # add to S2 metrics
    
    print(f"[INFO] Args: {args}")
    
    os.makedirs(args.outdir, exist_ok=True)
    plot_outdir = f"{args.outdir}/plot_data"
    os.makedirs(plot_outdir, exist_ok=True)
    table_outdir = f"{args.outdir}/table_data"
    os.makedirs(table_outdir, exist_ok=True)

    selected_models = load_selected_models(args.selected_models)
    print(f"[INFO] Selected models: {selected_models}")

    model_dirs = {}
    selected_model_dirs = {}
    for sub, i in selected_models.items():
        model_dirs[sub] = f"{args.root}/{sub}/4_model"
        selected_model_dirs[sub] = f"{model_dirs}/model_{i}/model_data"
    
    # handle confusion matrices
    confusion_matrixes(selected_model_dirs, args.outdir, plot_outdir, class_names)

    # handle performance metric outputs
    metrics = performance_metrics(model_dirs, args.outdir, metrics, plot_outdir)

    # output feat importance
    output_s1_feature_importances(selected_model_dirs, f"{args.outdir}/feat_importances.xlsx")

    # output shap vals
    plot_s1_shap_vals(selected_model_dirs, f"{args.outdir}")

if __name__ == "__main__":
    main()