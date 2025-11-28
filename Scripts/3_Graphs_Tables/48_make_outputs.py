import sys
import argparse
import os

import datetime as dt

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib

from functools import partial
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

pd.options.mode.chained_assignment = None


plt.rcParams["font.family"] = "Arial"

SUBREDDITS = {
    "conspiracy": "(a) R/Conspiracy",
    "crypto": "(b) R/CryptoCurrency",
    "politics": "(c) R/politics",
}

S1_METRICS = ["MCC", "Balanced accuracy", "AUC", "F1", "F-beta", "Precision", "Recall"]
for binary_class in ["Stalled", "Started"]:
    S1_METRICS.append(f"Precision {binary_class}")
    S1_METRICS.append(f"Recall {binary_class}")
S2_METRICS = ["MCC", "Balanced accuracy", "F1", "Precision", "Recall"]
for class_name in ["Stalled", "Small", "Medium", "Large"]:
    for metric in ["precision", "recall"]:
        S2_METRICS.append(f"{class_name} {metric}")
DATA_TYPES = ["OOF", "Test"]
DATA_LABELS = {
    "test": "Test",
    "oof": "OOF",
}

S2_CLASS_NAMES = {
    3: ["Stalled", "Small", "Large"],
    4: ["Stalled", "Small", "Medium", "Large"],
}

LETTER_LOOKUP = {
    0: "(a)",
    1: "(b)",
    2: "(c)",
    3: "(d)",
}

FEATURE_NAMES = {
    "encoded_domain": "Domain",
    "subject_length": "Subject\nlength",
    "encoded_author": "Author",
    "caps_ratio": "Caps ratio",
    "domain_pagerank": "Domain\nPageRank",
    "avg_word_length": "Avg. word\nlength",
    "stopword_ratio": "Stopword\nratio",
    "domain_type_Reddit": "Reddit\ndomain",
    "stage1_proba": "Stage 1\nprobability",
    "hour": "Hour",
    "dayofweek": "Day of\nweek",
    "verb_ratio": "Verb ratio",
}

TABLE_FEATURE_NAMES = {
    "encoded_domain": "Domain",
    "subject_length": "Subject length",
    "encoded_author": "Author",
    "caps_ratio": "Caps ratio",
    "domain_pagerank": "Domain PageRank",
    "avg_word_length": "Avg. word length",
    "unique_word_ratio": "Unique word ratio",
    "stopword_ratio": "Stopword ratio",
    "domain_type_Reddit": "Reddit domain",
    "stage1_proba": "Stage 1 probability",
    "hour": "Hour",
    "dayofweek": "Day of week",
    "verb_ratio": "Verb ratio",
    "noun_ratio": "Noun ratio",
    "body_length": "Body length",
    "reading_ease": "Reading ease",
    "exclamation_ratio": "Exclamation ratio",
    "question_ratio": "Question ratio",
}
svd_cols = []

colors = sns.color_palette("colorblind", n_colors=4)


def get_ci_str_from_list(value):
    return f"[{float(value[0]):.4f}, {float(value[1]):.4f}]"


run_number = sys.argv[1]
print(f"Running for Run {run_number}")
OUTDIR = f"REGDATA/outputs/{run_number}_results"
os.makedirs(OUTDIR, exist_ok=True)

# s1_modsfile = sys.argv[2]
# stage1_selected_models = {}
# stage2_selected_models = {}
# with open(s1_modsfile, "r") as f:
#     modlist = [x.strip().split(",") for x in f.readlines()]
#     for mod in modlist:
#         stage1_selected_models[mod[0]] = int(mod[1])
#         if len(mod) > 2:
#             stage2_selected_models[mod[0]] = int(mod[2])

selected_models = {}
modsfile = sys.argv[2]
with open(modsfile, "r") as f:
    modlist = [x.strip().split(",") for x in f.readlines()]
    for mod in modlist:
        selected_models[mod[0]] = int(mod[1])

# print(f"Selected stage 1 models: {stage1_selected_models}")
# print(f"Selected stage 2 models: {stage2_selected_models}")
n_classes = int(sys.argv[3])

run_stage1 = True
run_stage2 = True
run_stage3 = True

if len(sys.argv) > 4:
    stages_to_run = sys.argv[4]
    if "1" not in stages_to_run:
        run_stage1 = False
    else:
        stage1_outdir = f"{OUTDIR}/stage1"
        os.makedirs(stage1_outdir, exist_ok=True)
        stage1_files = [
            "selected_mods.csv",
            "feature_importances.xlsx",
            "combined_shap.png",
            "cm_oof.png",
            "mcc.png",
            "recall.png",
            "metrics.xlsx",
        ]
        stage1_files_exist = [
            os.path.isfile(f"{stage1_outdir}/{f}") for f in stage1_files
        ]
        if all(stage1_files_exist):
            run_stage1 = False
    if "2" not in stages_to_run:
        run_stage2 = False
    else:
        stage2_outdir = f"{OUTDIR}/stage2"
        os.makedirs(stage2_outdir, exist_ok=True)
        stage2_files = [
            "mcc.png",
            "Large recall.png",
            "metrics.xlsx",
            "selected_mods.csv",
            "cm_oof.png",
            "feature_importances.xlsx",
            "combined_crypto_shap.png",
        ]
        stage2_files_exist = [
            os.path.isfile(f"{stage2_outdir}/{f}") for f in stage2_files
        ]
        if all(stage2_files_exist):
            run_stage2 = False
    if "3" not in stages_to_run:
        run_stage3 = False
    else:
        stage3_outdir = f"{OUTDIR}/stage3"
        os.makedirs(stage3_outdir, exist_ok=True)
        stage3_files = ["mcc.png", "Large recall.png", "metrics.xlsx", "selected_mods.csv", "cm_OOF.png"]
        stage3_files_exist = [
            os.path.isfile(f"{stage3_outdir}/{f}") for f in stage3_files
        ]
        if all(stage3_files_exist):
            run_stage3 = False
    if not run_stage1 and not run_stage2 and not run_stage3:
        print("No stages to run. Exiting.")
        sys.exit(0)

run_dirs = {}
for subreddit in selected_models.keys():
    run_dirs[subreddit] = f"REGDATA/outputs/{subreddit}/Run_{run_number}"

# RUNNING STAGE 1
if run_stage1:
    print("Running stage 1")

    s1_indirs = {}
    for subreddit, mod in selected_models.items():
        s1_indirs[
            subreddit
        ] = f"{run_dirs[subreddit]}/stage_1_4_model_evaluation/mod_{mod}_outputs"

    print("    Plotting confusion matrices")

    for data_type in [x.lower() for x in DATA_TYPES]:
        if not os.path.isfile(f"{stage1_outdir}/cm_{data_type}.png"):
            i = 0
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            for subreddit, indir in s1_indirs.items():
                mod_n = selected_models[subreddit]
                cm = joblib.load(
                    f"{indir}/{mod_n}_{data_type}_confusion_matrix_plot_data.jl"
                )
                if isinstance(cm, dict):
                    cm = cm["CM"]
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["Stalled", "Started"],
                    yticklabels=["Stalled", "Started"],
                    ax=axes[i],
                    annot_kws={"fontsize": 10},  # annotation text size
                    cbar_kws={"shrink": 0.9},
                )
                axes[i].set_title(
                    f"{SUBREDDITS[subreddit]}", loc="left", fontsize=12, weight="bold"
                )
                axes[i].set_xlabel("Predicted Class", fontsize=11)
                axes[i].set_ylabel("True Class", fontsize=11)
                i += 1
            fig.delaxes(axes[-1])
            plt.tight_layout()
            plt.savefig(
                f"{stage1_outdir}/cm_{data_type}.eps",
                dpi=350,
                format="eps",
                bbox_inches="tight",
            )
            plt.savefig(
                f"{stage1_outdir}/cm_{data_type}.png",
                dpi=400,
                format="png",
                bbox_inches="tight",
            )
            plt.close()

    print("    Reading in performance metrics")
    combined_summaries = {}
    metrics = []
    for subreddit, indir in s1_indirs.items():
        combined_summaries[subreddit] = joblib.load(f"{indir}/../combined_scores.jl")
        new_metrics = [
            m for m in combined_summaries[subreddit]["oof"].keys() if m in S1_METRICS
        ]
        if len(metrics) < 1 or len(new_metrics) < len(metrics):
            metrics = new_metrics
    for metric in metrics:
        if not os.path.isfile(f"{stage1_outdir}/{metric}.png"):
            print(f"    Plotting {metric}")
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes = axes.flatten()
            handles_dict = {}
            for idx, subreddit in enumerate(SUBREDDITS):
                i = 0
                for key, label in data_labels.items():
                    results = pd.DataFrame(combined_summaries[subreddit][key])
                    if subreddit != "politics":
                        results = results.iloc[:-1,:]
                    ax = axes[idx]
                    lower = [
                        metric_val - ci[0]
                        for metric_val, ci in zip(
                            results[metric], results[f"{metric} CI"]
                        )
                    ]
                    upper = [
                        ci[1] - metric_val
                        for metric_val, ci in zip(
                            results[metric], results[f"{metric} CI"]
                        )
                    ]
                    line = ax.errorbar(
                        results["n_feats"],
                        results[f"{metric}"],
                        yerr=[lower, upper],
                        fmt="o-",
                        capsize=3,
                        color=colors[i],
                        ecolor=colors[i],
                    )
                    i += 1
                    handles_dict[label] = line
                ax.set_title(
                    f"{SUBREDDITS[subreddit]}", loc="left", fontsize=12, weight="bold"
                )
                if idx > 0:
                    ax.set_xlabel("Number of features", fontsize=11)
                if idx % 2 == 0:
                    ax.set_ylabel(metric, fontsize=11)
                ax.tick_params(axis="x", labelsize=10)
                ax.tick_params(axis="y", labelsize=10)
                ax.set_xticks(results["n_feats"], labels=results["n_feats"].astype(int))
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
            plt.savefig(f"{stage1_outdir}/{metric}.eps", dpi=350, format="eps")
            plt.savefig(f"{stage1_outdir}/{metric}.png", dpi=400, format="png")
            plt.close()

    summary_dfs = {}
    for subreddit in combined_summaries:
        summary_dfs[subreddit] = {}
        for key, results in combined_summaries[subreddit].items():
            summary_dfs[subreddit][key] = pd.DataFrame(results)

    for subreddit in summary_dfs:
        for key, df in summary_dfs[subreddit].items():
            for metric in metrics:
                max = df[metric].max()
                df[f"ratio_max_{metric}"] = df[metric] / max
                df[f"CI_width_{metric}"] = df[f"{metric} CI"].apply(
                    lambda x: x[1] - x[0]
                )
                df[f"{metric}_%improvement"] = df[metric].diff() / df[metric] * 100

    print("   Formatting summary dataframes for output")
    formatted_dfs = {}
    for subreddit in summary_dfs:
        formatted_dfs[subreddit] = {}
        for key, df in summary_dfs[subreddit].items():
            formatted_df = df.copy()
            formatted_df["n_feats"] = formatted_df["n_feats"].astype(int)
            formatted_df.set_index("n_feats", inplace=True)
            formatted_df = formatted_df.applymap(
                lambda x: f"{x:.4f}" if isinstance(x, (float)) and pd.notnull(x) else x
            )
            formatted_df[
                [x for x in formatted_df.columns if x.endswith("CI")]
            ] = formatted_df[
                [x for x in formatted_df.columns if x.endswith("CI")]
            ].applymap(
                get_ci_str_from_list
            )
            formatted_cols = []
            for metric in metrics:
                formatted_cols.extend(
                    [
                        metric,
                        f"{metric} CI",
                        f"ratio_max_{metric}",
                        f"CI_width_{metric}",
                        f"{metric}_%improvement",
                    ]
                )
            formatted_df = formatted_df[formatted_cols]
            formatted_dfs[subreddit][key] = formatted_df

    if not os.path.isfile(f"{stage1_outdir}/metrics.xlsx"):
        with pd.ExcelWriter(f"{stage1_outdir}/metrics.xlsx") as writer:
            for subreddit in formatted_dfs:
                for key, df in formatted_dfs[subreddit].items():
                    df.to_excel(writer, sheet_name=f"{subreddit}_{key}", index=True)

    if not os.path.isfile(f"{stage1_outdir}/selected_mods.csv"):
        print("    Making a table of selected models.")
        selected_mods = []
        for subreddit in formatted_dfs:
            mod = selected_models[subreddit]
            df = formatted_dfs[subreddit]["test"]
            row = df[df.index == mod]
            row["Subreddit"] = SUBREDDITS[subreddit][4:]
            row["Features"] = mod
            cols_to_keep = ["Subreddit", "Features"]
            for metric in metrics:
                cols_to_keep.extend([metric, f"{metric} CI"])
            row = row[cols_to_keep]
            selected_mods.append(row)
        pd.concat(selected_mods, ignore_index=True).to_csv(
            f"{stage1_outdir}/selected_mods.csv", index=True
        )

    print("    Getting SHAP, split, and gain importances.")
    stage1_final_mods = {}
    stage1_shapvals = {}
    for subreddit in SUBREDDITS:
        infile = [
            x for x in os.listdir(run_dirs[subreddit]) if x.endswith("stage1_models.jl")
        ]
        stage1_final_mods[subreddit] = joblib.load(
            os.path.join(run_dirs[subreddit], infile[0])
        )
        shap_infile = [
            x
            for x in os.listdir(s1_indirs[subreddit])
            if x.endswith("_shap_plot_data.jl")
        ]
        stage1_shapvals[subreddit] = joblib.load(
            os.path.join(s1_indirs[subreddit], shap_infile[0])
        )
    
    for subreddit, df in stage1_shapvals.items():
        colnames = df["feat_name"].columns.tolist()
        svd_cols.extend([col for col in colnames if col.startswith("svd_")])
    for col in svd_cols:
        feature_names[col] = col.replace("svd_", "SVD ")
        table_feature_names[col] = col.replace("svd_", "SVD ")

    if not os.path.isfile(f"{stage1_outdir}/feature_importances.xlsx"):
        feat_importance_dfs = {}
        for subreddit in SUBREDDITS:
            shap_importance = np.abs(stage1_shapvals[subreddit]["shap_val"]).mean(
                axis=0
            )
            shap_importance_df = pd.DataFrame(
                {
                    "Feature": stage1_shapvals[subreddit]["feat_name"].columns,
                    "MeanAbsoluteSHAP": shap_importance,
                }
            )
            tree_importance_df = pd.DataFrame(
                [
                    stage1_final_mods[subreddit]["classifier"].booster_.feature_name(),
                    stage1_final_mods[subreddit][
                        "classifier"
                    ].booster_.feature_importance(),
                    stage1_final_mods[subreddit][
                        "classifier"
                    ].booster_.feature_importance(importance_type="gain"),
                ],
                index=["Feature", "Split importance", "Gain importance"],
            ).T
            feat_importance_df = pd.merge(
                shap_importance_df, tree_importance_df, on="Feature"
            )
            feat_importance_dfs[subreddit] = feat_importance_df.sort_values(
                by="MeanAbsoluteSHAP", ascending=False
            )

        importance_df_out = {}
        for subreddit, df in feat_importance_dfs.items():
            new_df = df.copy()
            new_df["Gain importance"] = (
                new_df["Gain importance"].apply(np.round).astype(int)
            )
            new_df = new_df.map(
                lambda x: f"{x:.4f}" if isinstance(x, (float)) and pd.notnull(x) else x,
            )
            new_df["Feature"] = new_df["Feature"].map(table_feature_names)
            importance_df_out[subreddit] = new_df

        with pd.ExcelWriter(f"{stage1_outdir}/feature_importances.xlsx") as writer:
            for subreddit, df in importance_df_out.items():
                print(f"    Writing {subreddit} data to Excel.")
                df.to_excel(writer, sheet_name=subreddit, index=True)

    print("   Plotting SHAP values")
    plot_type = "dot"
    for i, subreddit in enumerate(list(SUBREDDITS.keys())):
        if not os.path.isfile(f"{stage1_outdir}/{subreddit}_shap.png"):
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                stage1_shapvals[subreddit]["shap_val"],
                stage1_shapvals[subreddit]["feat_name"],
                plot_type=plot_type,
                show=False,
                feature_names=[
                    feature_names[x]
                    for x in stage1_shapvals[subreddit]["feat_name"].columns
                ],
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
                f"{SUBREDDITS[subreddit]}", loc="left", fontsize=12, weight="bold"
            )
            main_ax.set_xlabel(
                "SHAP value", fontsize=11,
            )
            main_ax.set_ylabel("Feature", fontsize=11)
            main_ax.tick_params(labelsize=10)

            plt.tight_layout()
            plt.savefig(f"{stage1_outdir}/{subreddit}_shap.eps", dpi=350, format="eps")
            plt.savefig(f"{stage1_outdir}/{subreddit}_shap.png", dpi=400, format="png")
            plt.close()

    print("    Combining SHAP plots into one figure")
    if not os.path.isfile(f"{stage1_outdir}/combined_shap.png"):
        # Load saved SHAP plots
        imgs = [
            Image.open(f"{stage1_outdir}/{subreddit}_shap.png")
            for subreddit in ["conspiracy", "crypto", "politics"]
        ]

        # Determine the width and total height
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
            combined.save(f"{stage1_outdir}/combined_shap.{ext}", dpi=(400, 400))

        print("Stage 1 completed.")

# RUNNING STAGE 2
if run_stage2:
    print("Running stage 2")
    s2_indirs = {}
    print("    Reading in metrics")
    combined_summaries = {}
    summary_dfs = {}
    metrics = []
    #for subreddit, mod in stage1_selected_models.items():
    for subreddit in SUBREDDITS:
        s2_indirs[
            subreddit
        ] = f"{run_dirs[subreddit]}/4_model_evaluation"
        #f"{run_dirs[subreddit]}/stage2/stage1_{mod}feats/2_4_model_evaluation"
        combined_summaries[subreddit] = joblib.load(
            f"{s2_indirs[subreddit]}/combined_scores.jl"
        )
        new_metrics = [
            m for m in combined_summaries[subreddit]["oof"].keys() if m in S2_METRICS
        ]
        if len(metrics) < 1 or len(new_metrics) < len(metrics):
            metrics = new_metrics
        summary_dfs[subreddit] = {}
        for key, results in combined_summaries[subreddit].items():
            summary_dfs[subreddit][key] = pd.DataFrame(results)
    print(f"    Metrics: {metrics}")
    print("    Plotting performance metrics")
    for metric in metrics:
        if not os.path.isfile(f"{stage2_outdir}/{metric}.png"):
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes = axes.flatten()
            handles_dict = {}
            for idx, subreddit in enumerate(SUBREDDITS):
                ax = axes[idx]
                i = 0
                for key, label in data_labels.items():
                    results = pd.DataFrame(combined_summaries[subreddit][key])
                    if subreddit != "politics":
                        results = results.iloc[:-1, :]

                    lower = [
                        metric_val - ci[0]
                        for metric_val, ci in zip(
                            results[metric], results[f"{metric} CI"]
                        )
                    ]
                    upper = [
                        ci[1] - metric_val
                        for metric_val, ci in zip(
                            results[metric], results[f"{metric} CI"]
                        )
                    ]
                    line = ax.errorbar(
                        results["n_feats"],
                        results[f"{metric}"],
                        yerr=[lower, upper],
                        fmt="o-",
                        capsize=3,
                        color=colors[i],
                        ecolor=colors[i],
                    )
                    i += 1
                    handles_dict[label] = line
                ax.set_title(
                    f"{SUBREDDITS[subreddit]}", loc="left", fontsize=12, weight="bold"
                )
                if idx > 0:
                    ax.set_xlabel("Number of features", fontsize=11)
                if idx % 2 == 0:
                    ax.set_ylabel(metric, fontsize=11)
                ax.tick_params(axis="x", labelsize=10)
                ax.tick_params(axis="y", labelsize=10)
                ax.set_xticks(results["n_feats"], labels=results["n_feats"].astype(int))

            handles = [handles_dict[key] for key in data_labels.values()]
            fig.legend(
                handles=handles,
                labels=data_labels.values(),
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
            plt.savefig(f"{stage2_outdir}/{metric}.eps", dpi=350, format="eps")
            plt.savefig(f"{stage2_outdir}/{metric}.png", dpi=400, format="png")
            plt.close()

    print("    Formatting summary dataframes for output")
    for subreddit in summary_dfs:
        for key in summary_dfs[subreddit]:
            for metric in metrics:
                max = summary_dfs[subreddit][key][metric].max()
                summary_dfs[subreddit][key][f"ratio_max_{metric}"] = (
                    summary_dfs[subreddit][key][metric] / max
                )
                summary_dfs[subreddit][key][f"CI_width_{metric}"] = summary_dfs[
                    subreddit
                ][key][f"{metric} CI"].apply(
                    lambda x: x[1] - x[0]
                    if pd.notna(x[0]) and pd.notna(x[1])
                    else np.nan
                )
                summary_dfs[subreddit][key][f"{metric}_%improvement"] = (
                    summary_dfs[subreddit][key][metric].diff()
                    / summary_dfs[subreddit][key][metric]
                    * 100
                )

    formatted_summary_dfs = {}
    for subreddit in summary_dfs:
        formatted_summary_dfs[subreddit] = {}
        for data_type, df in summary_dfs[subreddit].items():
            formatted_df = df.applymap(
                lambda x: f"{x:.4f}" if isinstance(x, (float)) and pd.notnull(x) else x
            )
            formatted_df[[x for x in df.columns if x.endswith("CI")]] = df[
                [x for x in df.columns if x.endswith("CI")]
            ].map(get_ci_str_from_list)
            formatted_df["n_feats"] = formatted_df["n_feats"].astype(int)
            formatted_df.set_index("n_feats", inplace=True)
            df_formatted_columns = []
            for metric in metrics:
                df_formatted_columns.extend(
                    [
                        metric,
                        f"{metric} CI",
                        f"ratio_max_{metric}",
                        f"CI_width_{metric}",
                        f"{metric}_%improvement",
                    ]
                )
            formatted_df = formatted_df[df_formatted_columns]
            formatted_summary_dfs[subreddit][data_type] = formatted_df
    if not os.path.isfile(f"{stage2_outdir}/metrics.xlsx"):
        with pd.ExcelWriter(f"{stage2_outdir}/metrics.xlsx") as writer:
            for subreddit in formatted_summary_dfs:
                for key, df in formatted_summary_dfs[subreddit].items():
                    print(f"    Writing {subreddit} data to Excel.")
                    df.to_excel(writer, sheet_name=f"{subreddit}_{key}", index=True)
    if len(selected_models.keys()) > 0:
        if not os.path.isfile(f"{stage2_outdir}/selected_mods.csv"):
            print("    Outputting table of selected s2 models")
            selected_mods = []
            for subreddit in formatted_summary_dfs:
                selected_s2_mod = selected_models[subreddit]
                df = formatted_summary_dfs[subreddit]["test"]
                row = df[df.index == selected_s2_mod]
                row["n_feats"] = selected_s2_mod
                row["Subreddit"] = SUBREDDITS[subreddit][4:]
                cols_to_keep = ["Subreddit", "n_feats"]
                for metric in metrics:
                    cols_to_keep.extend([metric, f"{metric} CI"])
                row = row[cols_to_keep]
                selected_mods.append(row)

            pd.concat(selected_mods, ignore_index=True).to_csv(
                f"{stage2_outdir}/selected_mods.csv", index=True
            )

        print("    Plotting confusion matrices")

        s2_mod_indirs = {}
        for subreddit, mod in selected_models.items():
            s2_mod_indirs[subreddit] = f"{s2_indirs[subreddit]}/model_{mod}"
        for data_type in [x.lower() for x in DATA_TYPES]:
            if not os.path.isfile(f"{stage2_outdir}/cm_{data_type}.png"):
                i = 0
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()
                for subreddit, mod in selected_models.items():
                    cm = joblib.load(
                        f"{s2_mod_indirs[subreddit]}/{data_type}_{mod}_confusion_matrix_data.jl"
                    )
                    if isinstance(cm, dict):
                        cm = cm["CM"]
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=S2_CLASS_NAMES[n_classes],
                        yticklabels=S2_CLASS_NAMES[n_classes],
                        ax=axes[i],
                        annot_kws={"fontsize": 10},  # annotation text size
                        cbar_kws={"shrink": 0.9},
                    )
                    axes[i].set_title(
                        f"{SUBREDDITS[subreddit]}",
                        loc="left",
                        fontsize=12,
                        weight="bold",
                    )
                    axes[i].set_xlabel("Predicted Class", fontsize=11)
                    axes[i].set_ylabel("True Class", fontsize=11)

                    i += 1
                fig.delaxes(axes[-1])
                plt.tight_layout()
                plt.savefig(
                    f"{stage2_outdir}/cm_{data_type}.eps",
                    dpi=350,
                    format="eps",
                    bbox_inches="tight",
                )
                plt.savefig(
                    f"{stage2_outdir}/cm_{data_type}.png",
                    dpi=400,
                    format="png",
                    bbox_inches="tight",
                )

                plt.close()

        print("    SHAP, split and gain importances")

        stage2_final_mods = {}
        stage2_shapvals = {}
        stage2_xtest = {}
        feat_importance_dfs = {}

        for subreddit, mod in selected_models.items():
            indir = s2_mod_indirs[subreddit]
            model_infile = f"{indir}/{mod}_feats_model_data.jl"
            shap_infile = f"{indir}/{mod}_feats_shap_values.jl"
            model_dict = joblib.load(model_infile)
            shap_array = joblib.load(shap_infile)
            classifier = model_dict["model"]
            stage2_final_mods[subreddit] = classifier
            stage2_shapvals[subreddit] = shap_array
            stage2_xtest[subreddit] = model_dict["X_test"]
            shap_importance = np.abs(shap_array).mean(axis=0)
            shap_dict = {
                "Feature": model_dict["X_test"].columns,
                "MeanAbsoluteSHAP": np.abs(shap_importance).mean(axis=1),
            }
            for i in range(shap_importance.shape[1]):
                shap_dict[f"MeanSHAP_{i}"] = shap_importance[:,i]

            shap_importance_df = pd.DataFrame(
                shap_dict
            )
            tree_importance_df = pd.DataFrame(
                [
                    classifier.booster_.feature_name(),
                    classifier.booster_.feature_importance(),
                    classifier.booster_.feature_importance(importance_type="gain"),
                ],
                index=["Feature", "Split importance", "Gain importance"],
            ).T
            feat_importance_df = pd.merge(
                shap_importance_df, tree_importance_df, on="Feature"
            )
            feat_importance_dfs[subreddit] = feat_importance_df.sort_values(
                by="MeanAbsoluteSHAP", ascending=False
            )
        for subreddit, df in stage2_xtest.items():
            colnames = df.columns.tolist()
            svd_cols.extend([col for col in colnames if col.startswith("svd_")])
        for col in svd_cols:
            feature_names[col] = col.replace("svd_", "SVD ")
            table_feature_names[col] = col.replace("svd_", "SVD ")
        if not os.path.isfile(f"{stage2_outdir}/feature_importances.xlsx"):
            importance_df_out = {}
            for subreddit, df in feat_importance_dfs.items():
                new_df = df.copy()
                new_df["Gain importance"] = (
                    new_df["Gain importance"].apply(np.round).astype(int)
                )
                new_df = new_df.map(
                    lambda x: f"{x:.4f}"
                    if isinstance(x, (float)) and pd.notnull(x)
                    else x,
                )
                new_df['Feature'] = new_df['Feature'].map(table_feature_names)
                importance_df_out[subreddit] = new_df

            with pd.ExcelWriter(f"{stage2_outdir}/feature_importances.xlsx") as writer:
                for subreddit, df in importance_df_out.items():
                    print(f"    Writing {subreddit} data to Excel.")
                    df.to_excel(writer, sheet_name=subreddit, index=True)


        print("    Plotting SHAP vals")
        plot_type = "dot"
        for i, subreddit in enumerate(list(SUBREDDITS.keys())):
            for class_idx, label in enumerate(S2_CLASS_NAMES[n_classes]):
                if not os.path.isfile(f"{stage2_outdir}/{subreddit}_{label}_shap.png"):
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(
                        stage2_shapvals[subreddit][:, :, class_idx],
                        stage2_xtest[subreddit],
                        plot_type=plot_type,
                        show=False,
                        feature_names=[
                            feature_names[x] for x in stage2_xtest[subreddit].columns
                        ],
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
                        f"{letter_lookup[class_idx]} {label}",
                        loc="left",
                        fontsize=12,
                        weight="bold",
                    )
                    main_ax.set_xlabel(
                        "SHAP value", fontsize=11,
                    )
                    main_ax.set_ylabel("Feature", fontsize=11)
                    main_ax.tick_params(labelsize=10)

                    plt.tight_layout()
                    plt.savefig(
                        f"{stage2_outdir}/{subreddit}_{label}_shap.eps",
                        dpi=350,
                        format="eps",
                    )
                    plt.savefig(
                        f"{stage2_outdir}/{subreddit}_{label}_shap.png",
                        dpi=400,
                        format="png",
                    )
                    plt.close()

        print("    Combining SHAP plots")
        for subreddit in SUBREDDITS:
            if not os.path.isfile(f"{stage2_outdir}/combined_{subreddit}_shap.png"):
                # load saved SHAP plots
                imgs = [
                    Image.open(f"{stage2_outdir}/{subreddit}_{label}_shap.png")
                    for label in S2_CLASS_NAMES[n_classes]
                ]

                # Determine the width and total height
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
                    combined.save(
                        f"{stage2_outdir}/combined_{subreddit}_shap.{ext}",
                        dpi=(400, 400),
                    )

    print("    Stage 2 finished")

"""
if run_stage3:
    print("Running stage 3")
    print("    Reading in stage 3 files")
    stage3_indirs = {}
    stage3_cms = {}
    stage3_results = {}
    for subreddit, mod in stage1_selected_models.items():
        stage3_indirs[
            subreddit
        ] = f"{run_dirs[subreddit]}/stage2/stage1_{mod}feats/stage_3_evaluation"
        cm = joblib.load(
            f"{stage3_indirs[subreddit]}/{stage2_selected_models[subreddit]}_confusion_matrix_data.jl"
        )
        if subreddit not in stage3_cms:
            stage3_cms[subreddit] = {}
        if isinstance(cm, dict):
            for data_type in cm:
                stage3_cms[subreddit][data_type] = cm[data_type]["CM"]
        else:
            stage3_cms[subreddit]["Test"] = cm
        stage3_results[subreddit] = joblib.load(
            f"{stage3_indirs[subreddit]}/combined_results.jl"
        )

    print("    Getting feature counts from stages 1 and 2 combined")
    combined_features = {}
    for subreddit, mod in stage1_selected_models.items():
        stage_1_infile = f"{run_dirs[subreddit]}/{mod}_feats_stage1_models.jl"
        stage_1_features = joblib.load(stage_1_infile)[
            "classifier"
        ].feature_names_in_.tolist()
        stage_2_features = {}
        combined_features[subreddit] = {}
        stage2_indir = (
            f"{run_dirs[subreddit]}/stage2/stage1_{mod}feats/2_4_model_evaluation"
        )
        model_dirs = [
            f"{stage2_indir}/{x}"
            for x in os.listdir(stage2_indir)
            if x.startswith("model_")
        ]
        for model_data_dir in model_dirs:
            model_data_files = [
                x
                for x in os.listdir(model_data_dir)
                if x.endswith("_feats_model_data.jl")
            ]
            feats = joblib.load(f"{model_data_dir}/{model_data_files[0]}")[
                "X_test"
            ].columns.tolist()
            stage_2_features[len(feats)] = feats
            combined_features[subreddit][len(feats)] = list(
                set(stage_1_features) | set(stage_2_features[len(feats)])
            )

    print("    Making summary dfs of performance metrics across all subreddits")
    summary_dfs = {}
    metrics = []
    for subreddit in stage3_results:
        summary_dfs[subreddit] = {}
        for d, results_dict in stage3_results[subreddit].items():
            summary_dfs[subreddit][d] = pd.DataFrame(results_dict)
            new_metrics = [x for x in results_dict.keys() if x in S2_METRICS]
            if len(metrics) < 1 or len(new_metrics) < len(metrics):
                metrics = new_metrics

    print("    Formatting summary dfs for output")
    formatted_dfs = {}
    for subreddit in summary_dfs:
        formatted_dfs[subreddit] = {}
        for key, df in summary_dfs[subreddit].items():
            formatted_df = df.copy()
            formatted_df["n_feats"] = formatted_df["n_feats"].astype(int)
            formatted_df.set_index("n_feats", inplace=True)
            formatted_df = formatted_df.applymap(
                lambda x: f"{x:.4f}" if isinstance(x, (float)) and pd.notnull(x) else x
            )
            formatted_df[
                [x for x in formatted_df.columns if x.endswith("CI")]
            ] = formatted_df[
                [x for x in formatted_df.columns if x.endswith("CI")]
            ].applymap(
                get_ci_str_from_list
            )
            formatted_cols = []
            for metric in metrics:
                formatted_cols.extend(
                    [metric, f"{metric} CI",]
                )
            formatted_df = formatted_df[formatted_cols]
            formatted_dfs[subreddit][key] = formatted_df

    if not os.path.isfile(f"{stage3_outdir}/metrics.xlsx"):
        with pd.ExcelWriter(f"{stage3_outdir}/metrics.xlsx") as writer:
            for subreddit in formatted_dfs:
                for key, df in formatted_dfs[subreddit].items():
                    df.to_excel(writer, sheet_name=f"{subreddit}_{key}", index=True)

    print("    Plotting metrics")
    for metric in metrics:
        if not os.path.isfile(f"{stage3_outdir}/{metric}.png"):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            handles_dict = {}
            for idx, subreddit in enumerate(SUBREDDITS):
                j = 0
                for d, results in summary_dfs[subreddit].items():
                    ax = axes[idx]
                    i = 0
                    if subreddit != "politics":
                        results = pd.DataFrame(results).iloc[:-1, :]
                    lower = [
                        metric_val - ci[0]
                        for metric_val, ci in zip(
                            results[metric], results[f"{metric} CI"]
                        )
                    ]
                    upper = [
                        ci[1] - metric_val
                        for metric_val, ci in zip(
                            results[metric], results[f"{metric} CI"]
                        )
                    ]
                    line = ax.errorbar(
                        results["n_feats"],
                        results[f"{metric}"],
                        yerr=[lower, upper],
                        fmt="o-",
                        capsize=3,
                        color=colors[j],
                        ecolor=colors[j],
                    )
                    j += 1
                    handles_dict[d] = line
                i += 1
                ax.set_title(
                    f"{SUBREDDITS[subreddit]}", loc="left", fontsize=12, weight="bold"
                )
                if idx > 0:
                    ax.set_xlabel("Total number of features", fontsize=11)
                if idx % 2 == 0:
                    ax.set_ylabel(metric, fontsize=11)
                ax.set_xticks(
                    results["n_feats"],
                    labels=[
                        len(combined_features[subreddit][n]) for n in results["n_feats"]
                    ],
                    rotation=0,
                )
                ax.tick_params(axis="x", labelsize=10)
                ax.tick_params(axis="y", labelsize=10)

            # Sort to maintain order of new_metrics
            handles = [handles_dict[d] for d in stage3_results[subreddit]]

            fig.legend(
                handles=handles,
                labels=stage3_results[subreddit].keys(),
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

            fig.delaxes(
                axes[-1]
            )  # Remove the last empty subplot if there are only 3 subreddits
            plt.subplots_adjust(hspace=0.3)
            plt.tight_layout()
            plt.savefig(f"{stage3_outdir}/{metric}.eps", dpi=350, format="eps")
            plt.savefig(f"{stage3_outdir}/{metric}.png", dpi=400, format="png")
            plt.close()

    if len(stage2_selected_models.keys()) > 0:
        if not os.path.isfile(f"{stage3_outdir}/selected_mods.csv"):
            print("    Making a table of selected models.")
            selected_mods = []
            for subreddit in formatted_dfs:
                mod = stage2_selected_models[subreddit]
                df = formatted_dfs[subreddit]["Test"]
                row = df[df.index == mod]
                row["n_feats"] = mod
                row["Subreddit"] = SUBREDDITS[subreddit][4:]
                row["Total_n"] = len(combined_features[subreddit][int(row["n_feats"])])
                cols_to_keep = ["Subreddit", "n_feats", "Total_n"]
                for metric in metrics:
                    cols_to_keep.extend([metric, f"{metric} CI"])
                selected_mods.append(row)
            pd.concat(selected_mods, ignore_index=True).to_csv(
                f"{stage3_outdir}/selected_mods.csv", index=True
            )

        print("    Plotting confusion matrices")
        for d in DATA_TYPES:
            if not os.path.isfile(f"{stage3_outdir}/cm_{d}.png"):
                i = 0
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()
                for subreddit in stage3_cms:
                    cm = stage3_cms[subreddit][d]
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=S3_CLASS_NAMES[n_classes],
                        yticklabels=S3_CLASS_NAMES[n_classes],
                        ax=axes[i],
                        annot_kws={"fontsize": 10},  # annotation text size
                        cbar_kws={"shrink": 0.9},
                    )
                    axes[i].set_title(
                        f"{SUBREDDITS[subreddit]}",
                        loc="left",
                        fontsize=12,
                        weight="bold",
                    )
                    axes[i].set_xlabel("Predicted Class", fontsize=11)
                    axes[i].set_ylabel("True Class", fontsize=11)
                    i += 1

                fig.delaxes(axes[-1])
                plt.tight_layout()
                plt.savefig(
                    f"{stage3_outdir}/cm_{d}.eps",
                    dpi=350,
                    format="eps",
                    bbox_inches="tight",
                )
                plt.savefig(
                    f"{stage3_outdir}/cm_{d}.png",
                    dpi=400,
                    format="png",
                    bbox_inches="tight",
                )
                plt.close()
    print("Stage 3 completed")

print("All done!")
"""