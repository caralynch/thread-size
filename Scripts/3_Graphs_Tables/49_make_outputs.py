import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import lightgbm as lgb
import shap
from PIL import Image
import sys

pd.options.mode.chained_assignment = None
# Usage: python 49_make_outputs.py <run_number> <model_file> <n_classes>


plt.rcParams["font.family"] = "Arial"

SUBREDDITS = {
    "conspiracy": "(a) R/Conspiracy",
    "crypto": "(b) R/CryptoCurrency",
    "politics": "(c) R/politics",
}

METRICS = ["MCC", "Balanced accuracy", "F1", "Precision", "Recall"]
for class_name in ["Stalled", "Small", "Medium", "Large"]:
    for metric in ["precision", "recall"]:
        METRICS.append(f"{class_name} {metric}")
DATA_TYPES = ["OOF", "Test"]
data_labels = {
    "test": "Test",
    "oof": "OOF",
}

CLASS_NAMES = {
    3: ["Stalled", "Small", "Large"],
    4: ["Stalled", "Small", "Medium", "Large"],
}

letter_lookup = {
    0: "(a)",
    1: "(b)",
    2: "(c)",
    3: "(d)",
}

feature_names = {
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
    "subject_sentiment_score": "Subject sentiment\nscore",
    "question_ratio": "Question\nratio",
}

table_feature_names = {
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
    "subject_sentiment_score": "Subject sentiment score",
}
svd_cols = []

colors = sns.color_palette("colorblind", n_colors=4)


def get_ci_str_from_list(value):
    return f"[{float(value[0]):.4f}, {float(value[1]):.4f}]"


run_number = sys.argv[1]
print(f"Running for Run {run_number}")
OUTDIR = f"REGDATA/outputs/{run_number}_results"
os.makedirs(OUTDIR, exist_ok=True)

modsfile = sys.argv[2]
selected_models = {}
with open(modsfile, "r") as f:
    modlist = [x.strip().split(",") for x in f.readlines()]
    for mod in modlist:
        selected_models[mod[0]] = int(mod[1])

print(f"Selected models: {selected_models}")
n_classes = int(sys.argv[3])

stage_files = [
    "selected_mods.csv",
    "feature_importances.xlsx",
    "combined_shap.png",
    "cm_oof.png",
    "mcc.png",
    "recall.png",
    "metrics.xlsx",
]
stage_files_exist = [
    os.path.isfile(f"{OUTDIR}/{f}") for f in stage_files
]
if all(stage_files_exist):
    run_stage = False
    print("No stage to run. Exiting.")
    sys.exit(0)

run_dirs = {}
for subreddit in selected_models.keys():
    run_dirs[subreddit] = f"REGDATA/outputs/{subreddit}/Run_{run_number}"

indirs = {}
print("    Reading in metrics")
combined_summaries = {}
summary_dfs = {}
metrics = []
for subreddit, mod in selected_models.items():
    indirs[
        subreddit
    ] = f"{run_dirs[subreddit]}/4_model_evaluation"
    combined_summaries[subreddit] = joblib.load(
        f"{indirs[subreddit]}/combined_scores.jl"
    )
    new_metrics = [
        m for m in combined_summaries[subreddit]["oof"].keys() if m in METRICS
    ]
    if len(metrics) < 1 or len(new_metrics) < len(metrics):
        metrics = new_metrics
    summary_dfs[subreddit] = {}
    for key, results in combined_summaries[subreddit].items():
        summary_dfs[subreddit][key] = pd.DataFrame(results)
print(f"    Metrics: {metrics}")
print("    Plotting performance metrics")
for metric in metrics:
    if not os.path.isfile(f"{OUTDIR}/{metric}.png"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        handles_dict = {}
        for idx, subreddit in enumerate(SUBREDDITS):
            ax = axes[idx]
            i = 0
            for key, label in data_labels.items():
                results = pd.DataFrame(combined_summaries[subreddit][key])

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
        plt.savefig(f"{OUTDIR}/{metric}.eps", dpi=350, format="eps")
        plt.savefig(f"{OUTDIR}/{metric}.png", dpi=400, format="png")
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
if not os.path.isfile(f"{OUTDIR}/metrics.xlsx"):
    with pd.ExcelWriter(f"{OUTDIR}/metrics.xlsx") as writer:
        for subreddit in formatted_summary_dfs:
            for key, df in formatted_summary_dfs[subreddit].items():
                print(f"    Writing {subreddit} data to Excel.")
                df.to_excel(writer, sheet_name=f"{subreddit}_{key}", index=True)
if len(selected_models.keys()) > 0:
    if not os.path.isfile(f"{OUTDIR}/selected_mods.csv"):
        print("    Outputting table of selected s2 models")
        selected_mods = []
        for subreddit in formatted_summary_dfs:
            selected_mod = selected_models[subreddit]
            df = formatted_summary_dfs[subreddit]["test"]
            row = df[df.index == selected_mod]
            row["n_feats"] = selected_mod
            row["Subreddit"] = SUBREDDITS[subreddit][4:]
            cols_to_keep = ["Subreddit", "n_feats"]
            for metric in metrics:
                cols_to_keep.extend([metric, f"{metric} CI"])
            row = row[cols_to_keep]
            selected_mods.append(row)

        pd.concat(selected_mods, ignore_index=True).to_csv(
            f"{OUTDIR}/selected_mods.csv", index=True
        )

    print("    Plotting confusion matrices")

    mod_indirs = {}
    for subreddit, mod in selected_models.items():
        mod_indirs[subreddit] = f"{indirs[subreddit]}/model_{mod}"
    for data_type in [x.lower() for x in DATA_TYPES]:
        if not os.path.isfile(f"{OUTDIR}/cm_{data_type}.png"):
            i = 0
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            for subreddit, mod in selected_models.items():
                cm = joblib.load(
                    f"{mod_indirs[subreddit]}/{data_type}_{mod}_confusion_matrix_data.jl"
                )
                if isinstance(cm, dict):
                    cm = cm["CM"]
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=CLASS_NAMES[n_classes],
                    yticklabels=CLASS_NAMES[n_classes],
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
                f"{OUTDIR}/cm_{data_type}.eps",
                dpi=350,
                format="eps",
                bbox_inches="tight",
            )
            plt.savefig(
                f"{OUTDIR}/cm_{data_type}.png",
                dpi=400,
                format="png",
                bbox_inches="tight",
            )

            plt.close()

    print("    SHAP, split and gain importances")

    final_mods = {}
    shapvals = {}
    xtest = {}
    feat_importance_dfs = {}

    for subreddit, mod in selected_models.items():
        indir = mod_indirs[subreddit]
        model_infile = f"{indir}/stage2_{mod}_feats_model_data.jl"
        shap_infile = f"{indir}/stage2_{mod}_feats_shap_values.jl"
        model_dict = joblib.load(model_infile)
        shap_array = joblib.load(shap_infile)
        classifier = model_dict["model"]
        final_mods[subreddit] = classifier
        shapvals[subreddit] = shap_array
        xtest[subreddit] = model_dict["X_test"]
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
    for subreddit, df in xtest.items():
        colnames = df.columns.tolist()
        svd_cols.extend([col for col in colnames if col.startswith("svd_")])
    for col in svd_cols:
        feature_names[col] = col.replace("svd_", "SVD ")
        table_feature_names[col] = col.replace("svd_", "SVD ")
    if not os.path.isfile(f"{OUTDIR}/feature_importances.xlsx"):
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

        with pd.ExcelWriter(f"{OUTDIR}/feature_importances.xlsx") as writer:
            for subreddit, df in importance_df_out.items():
                print(f"    Writing {subreddit} data to Excel.")
                df.to_excel(writer, sheet_name=subreddit, index=True)


    print("    Plotting SHAP vals")
    plot_type = "dot"
    for i, subreddit in enumerate(list(SUBREDDITS.keys())):
        for class_idx, label in enumerate(CLASS_NAMES[n_classes]):
            if not os.path.isfile(f"{OUTDIR}/{subreddit}_{label}_shap.png"):
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shapvals[subreddit][:, :, class_idx],
                    xtest[subreddit],
                    plot_type=plot_type,
                    show=False,
                    feature_names=[
                        feature_names[x] for x in xtest[subreddit].columns
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
                    f"{OUTDIR}/{subreddit}_{label}_shap.eps",
                    dpi=350,
                    format="eps",
                )
                plt.savefig(
                    f"{OUTDIR}/{subreddit}_{label}_shap.png",
                    dpi=400,
                    format="png",
                )
                plt.close()

    print("    Combining SHAP plots")
    for subreddit in SUBREDDITS:
        if not os.path.isfile(f"{OUTDIR}/combined_{subreddit}_shap.png"):
            # load saved SHAP plots
            imgs = [
                Image.open(f"{OUTDIR}/{subreddit}_{label}_shap.png")
                for label in CLASS_NAMES[n_classes]
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
                    f"{OUTDIR}/combined_{subreddit}_shap.{ext}",
                    dpi=(400, 400),
                )

print("    Finished")