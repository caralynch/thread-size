#!/usr/bin/env python3
"""Render the three-panel empirical CCDF from frozen aggregate CSV inputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


AUDIT = Path(__file__).resolve().parent
OUT = AUDIT / "outputs"
FIG = AUDIT / "figures"
SUBREDDITS = ["r/Conspiracy", "r/CryptoCurrency", "r/politics"]
STYLES = {
    "training": {"color": "#1769AA", "linestyle": "-", "linewidth": 1.8},
    "held_out": {"color": "#D97904", "linestyle": "--", "linewidth": 1.8},
}


def render() -> None:
    ccdf = pd.read_csv(OUT / "ccdf.csv")
    summary = pd.read_csv(OUT / "summary_statistics.csv")
    FIG.mkdir(exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "svg.fonttype": "none",
            "svg.hashsalt": "study1-thread-size-ccdf",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.05), sharex=True, sharey=True)

    for ax, subreddit in zip(axes, SUBREDDITS):
        panel = ccdf[ccdf["subreddit"] == subreddit]
        for partition in ("training", "held_out"):
            rows = panel[panel["partition"] == partition].sort_values("T")
            ax.step(
                rows["T"],
                rows["survival_probability_ge"],
                where="post",
                label="Training" if partition == "training" else "Held-out",
                **STYLES[partition],
            )

        stalled = summary[
            (summary["subreddit"] == subreddit)
            & (summary["population_basis"] == "all_roots")
            & (summary["partition"].isin(["training", "held_out"]))
        ].set_index("partition")["stalled_pct"]
        ax.text(
            0.97,
            0.96,
            f"Stalled\ntrain {stalled['training']:.1f}%\nheld-out {stalled['held_out']:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "#C8CDD2", "boxstyle": "round,pad=0.3", "alpha": 0.92},
        )
        ax.set_title(subreddit)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 2e4)
        ax.set_ylim(1e-5, 1.05)
        ax.grid(True, which="major", color="#D9DEE3", linewidth=0.7)
        ax.grid(True, which="minor", color="#EEF0F2", linewidth=0.45)
        ax.set_xlabel("Root-inclusive observed size, T")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel(r"Empirical survival, $P(T \geq t)$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Observed thread-size distributions in the final Study 1 populations", y=0.985, fontsize=12)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.17, top=0.80, wspace=0.13)
    fig.savefig(FIG / "thread_size_ccdf.svg", format="svg", metadata={"Date": None})
    fig.savefig(FIG / "thread_size_ccdf.png", format="png", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    render()
