


"""Unified paper-figure plotting helpers for relationship and proof results."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .plotting import apply_paper_style, save_figure_bundle
from .utils import pretty_group_name

matplotlib.use("Agg")

VISUAL_COLOR = "#2C6BA0"
AUDIO_COLOR = "#C17C10"
FUSION_COLOR = "#111827"
NEUTRAL_COLOR = "#9CA3AF"
GRID_COLOR = "#E5E7EB"
TEXT_MUTED = "#6B7280"

CLAIM_STYLE = {
    "supported_confirmatory": {"edgecolor": "#111827", "linestyle": "-", "linewidth": 1.2},
    "directionally_consistent_but_not_significant": {"edgecolor": "#4B5563", "linestyle": "-", "linewidth": 1.1},
    "unstable_under_sensitivity": {"edgecolor": "#B45309", "linestyle": "-.", "linewidth": 1.0},
    "not_supported": {"edgecolor": "#94A3B8", "linestyle": "--", "linewidth": 1.0},
    "not_testable": {"edgecolor": "#CBD5E1", "linestyle": ":", "linewidth": 0.9},
}

CLAIM_ORDER = [
    "fusion_superior_to_both",
    "fusion_complementary_but_not_superior",
    "no_evidence_fusion_better",
    "single_modality_stronger",
]

CLAIM_ABBR = {
    "fusion_superior_to_both": "Sup",
    "fusion_complementary_but_not_superior": "Comp",
    "no_evidence_fusion_better": "NoEv",
    "single_modality_stronger": "Single",
}

CLAIM_DISPLAY = {
    "fusion_superior_to_both": "Fusion superior",
    "fusion_complementary_but_not_superior": "Fusion complementary",
    "no_evidence_fusion_better": "No evidence fusion better",
    "single_modality_stronger": "Single modality stronger",
}

ASSOCIATION_CMAP = LinearSegmentedColormap.from_list(
    "paper_audio_visual_diverging",
    ["#2B6EA6", "#F8F8F8", "#C26D12"],
)


def _nice_target_label(name: str) -> str:
    text = str(name).replace("_", " ").strip()
    if not text:
        return str(name)
    return text[0].upper() + text[1:]


def _group_boundaries(data: pd.DataFrame, group_col: str) -> List[Tuple[int, int, str]]:
    boundaries: List[Tuple[int, int, str]] = []
    if data.empty:
        return boundaries
    start = 0
    current = str(data.iloc[0][group_col])
    for idx in range(1, len(data)):
        item = str(data.iloc[idx][group_col])
        if item != current:
            boundaries.append((start, idx - 1, current))
            start = idx
            current = item
    boundaries.append((start, len(data) - 1, current))
    return boundaries


def plot_fig_a1_pls_lv1_coupling(
    score_df: pd.DataFrame,
    *,
    corr_value: float,
    permutation_p: float,
    n_full: int,
    n_thin: int,
    out_base: Path,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    ax.scatter(
        score_df["x_score"],
        score_df["y_score"],
        s=24,
        facecolor=VISUAL_COLOR,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.88,
        zorder=3,
    )
    if len(score_df) >= 2 and score_df["x_score"].nunique() > 1:
        slope, intercept = np.polyfit(score_df["x_score"], score_df["y_score"], deg=1)
        x_line = np.linspace(float(score_df["x_score"].min()), float(score_df["x_score"].max()), 200)
        ax.plot(x_line, intercept + slope * x_line, color=FUSION_COLOR, linewidth=1.15, zorder=4)
    ax.set_title("Dominant latent audio–visual coupling")
    ax.set_xlabel("Visual LV1 score")
    ax.set_ylabel("Audio LV1 score")
    ax.grid(color=GRID_COLOR, linewidth=0.55, axis="both", alpha=0.75)
    note = f"r = {corr_value:.3f}\nperm. p = {permutation_p:.3f}\nn = {n_full}, thin = {n_thin}"
    ax.text(
        0.03,
        0.97,
        note,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color=FUSION_COLOR,
        bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "linewidth": 0.6, "boxstyle": "round,pad=0.22"},
    )
    save_figure_bundle(fig, out_base, dpi=600)


def plot_fig_a2_group_association_matrix(
    combined_df: pd.DataFrame,
    *,
    visual_groups: Sequence[str],
    audio_groups: Sequence[str],
    out_base: Path,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(
        figsize=(max(5.6, 1.15 + 0.72 * len(audio_groups)), max(4.4, 1.15 + 0.52 * len(visual_groups)))
    )
    matrix_df = combined_df[combined_df["testable_flag"]].pivot(
        index="visual_group",
        columns="audio_group",
        values="spearman_rho_full",
    )
    matrix_df = matrix_df.reindex(index=list(visual_groups), columns=list(audio_groups))
    values = matrix_df.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    vmax = max(0.20, min(1.0, float(np.ceil(vmax * 10.0) / 10.0)))
    im = ax.imshow(values, cmap=ASSOCIATION_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_title("Group-level audio–visual association matrix")
    ax.set_xticks(np.arange(len(audio_groups)))
    ax.set_xticklabels([pretty_group_name(x) for x in audio_groups], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(visual_groups)))
    ax.set_yticklabels([pretty_group_name(x) for x in visual_groups])
    ax.set_xlabel("Audio groups")
    ax.set_ylabel("Visual groups")

    lookup = {
        (str(row["visual_group"]), str(row["audio_group"])): row
        for _, row in combined_df.iterrows()
        if bool(row.get("testable_flag", False))
    }
    for i, visual_group in enumerate(visual_groups):
        for j, audio_group in enumerate(audio_groups):
            row = lookup.get((str(visual_group), str(audio_group)))
            if row is None or not np.isfinite(values[i, j]):
                continue
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="#FFFFFF",
                    linewidth=0.55,
                    alpha=0.9,
                )
            )
            if str(row.get("family", "")) == "confirmatory":
                style = CLAIM_STYLE.get(str(row.get("confirmatory_claim", "")), CLAIM_STYLE["not_supported"])
                ax.add_patch(
                    Rectangle(
                        (j - 0.48, i - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor=style["edgecolor"],
                        linewidth=float(style["linewidth"]),
                        linestyle=str(style["linestyle"]),
                    )
                )
            if float(row.get("spearman_q_full", np.nan)) < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=10.5, color=FUSION_COLOR, zorder=5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Spearman rho")
    cbar.outline.set_linewidth(0.6)
    save_figure_bundle(fig, out_base, dpi=600)


def plot_fig_b1_targetwise_model_dumbbell(plot_df: pd.DataFrame, *, out_base: Path) -> None:
    apply_paper_style()
    data = plot_df.copy().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.6, max(4.5, 0.58 * len(data) + 1.8)))
    y = np.arange(len(data))[::-1]

    values = np.concatenate(
        [
            data["visual_value"].to_numpy(dtype=float),
            data["audio_value"].to_numpy(dtype=float),
            data["fusion_value"].to_numpy(dtype=float),
        ]
    )
    finite = values[np.isfinite(values)]
    xmin = float(np.nanmin(finite)) if finite.size else 0.0
    xmax = float(np.nanmax(finite)) if finite.size else 1.0
    pad = max(0.025, 0.08 * (xmax - xmin if xmax > xmin else 1.0))
    ax.set_xlim(xmin - pad, xmax + pad * 1.9)

    for idx, row in data.iterrows():
        y0 = y[idx]
        ax.plot(
            [float(row["visual_value"]), float(row["fusion_value"])],
            [y0, y0],
            color=VISUAL_COLOR,
            linewidth=0.95,
            alpha=0.75,
            solid_capstyle="round",
            zorder=1,
        )
        ax.plot(
            [float(row["audio_value"]), float(row["fusion_value"])],
            [y0, y0],
            color=AUDIO_COLOR,
            linewidth=0.95,
            alpha=0.75,
            linestyle="--",
            solid_capstyle="round",
            zorder=1,
        )

    ax.scatter(
        data["visual_value"],
        y,
        s=30,
        facecolors="white",
        edgecolors=VISUAL_COLOR,
        linewidths=1.0,
        marker="o",
        zorder=3,
    )
    ax.scatter(
        data["audio_value"],
        y,
        s=32,
        facecolors="white",
        edgecolors=AUDIO_COLOR,
        linewidths=1.0,
        marker="s",
        zorder=3,
    )
    ax.scatter(
        data["fusion_value"],
        y,
        s=52,
        facecolors=FUSION_COLOR,
        edgecolors="white",
        linewidths=0.5,
        marker="D",
        zorder=4,
    )

    right_x = ax.get_xlim()[1] - pad * 0.05
    for idx, row in data.iterrows():
        ax.text(
            right_x,
            y[idx],
            CLAIM_ABBR.get(str(row["claim_label"]), str(row["claim_label"])),
            ha="right",
            va="center",
            fontsize=8,
            color=TEXT_MUTED,
        )

    ax.set_title("Target-wise performance of visual, audio, and fusion models", pad=14)
    ax.set_xlabel("Primary metric (lower is better)")
    ax.set_yticks(y)
    ax.set_yticklabels([_nice_target_label(x) for x in data["target_name"]])
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.55, alpha=0.8)

    for start, end, claim in _group_boundaries(data, "claim_label"):
        if end < len(data) - 1:
            ax.axhline(y[end] - 0.5, color="#E5E7EB", linewidth=0.8)
        group_mid = (y[start] + y[end]) / 2.0
        ax.text(
            0.01,
            group_mid,
            CLAIM_DISPLAY.get(str(claim), str(claim).replace("_", " ")),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8,
            color=TEXT_MUTED,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.18, "alpha": 0.92},
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=VISUAL_COLOR, markersize=5.8, label="Visual-only"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor=AUDIO_COLOR, markersize=5.8, label="Audio-only"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=FUSION_COLOR, markeredgecolor="white", markersize=6.6, label="Fusion"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    save_figure_bundle(fig, out_base, dpi=600)


def plot_fig_b2_fusion_incremental_forest(plot_df: pd.DataFrame, *, out_base: Path) -> None:
    apply_paper_style()
    data = plot_df.copy().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.4, max(4.5, 0.39 * len(data) + 1.8)))

    lower = data["delta_mean"].to_numpy(dtype=float) - data["ci_lower"].to_numpy(dtype=float)
    upper = data["ci_upper"].to_numpy(dtype=float) - data["delta_mean"].to_numpy(dtype=float)
    values = np.concatenate(
        [
            data["ci_lower"].to_numpy(dtype=float),
            data["ci_upper"].to_numpy(dtype=float),
            np.array([0.0], dtype=float),
        ]
    )
    finite = values[np.isfinite(values)]
    xmin = float(np.nanmin(finite)) if finite.size else -1.0
    xmax = float(np.nanmax(finite)) if finite.size else 1.0
    pad = max(0.025, 0.09 * (xmax - xmin if xmax > xmin else 1.0))
    ax.set_xlim(xmin - pad * 1.6, xmax + pad * 1.8)
    ax.axvline(0.0, color="#4B5563", linewidth=1.0, linestyle="--", zorder=0)

    y = np.arange(len(data))[::-1]
    comparison_style = {
        "fusion_vs_visual_only": {"color": VISUAL_COLOR, "marker": "o"},
        "fusion_vs_audio_only": {"color": AUDIO_COLOR, "marker": "s"},
    }

    for idx, row in data.iterrows():
        style = comparison_style.get(str(row["comparison"]), {"color": NEUTRAL_COLOR, "marker": "o"})
        ax.errorbar(
            [float(row["delta_mean"])],
            [y[idx]],
            xerr=np.array([[lower[idx]], [upper[idx]]], dtype=float),
            fmt="none",
            ecolor=style["color"],
            elinewidth=1.0,
            capsize=2.5,
            zorder=1,
        )
        ax.scatter(
            [float(row["delta_mean"])],
            [y[idx]],
            s=34,
            facecolors=style["color"] if bool(row.get("significant", False)) else "white",
            edgecolors=style["color"],
            linewidths=1.0,
            marker=style["marker"],
            zorder=3,
        )

    ax.set_title("Incremental value of fusion across targets")
    ax.set_xlabel("Delta in primary loss (negative favors fusion)")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.55, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["row_label"].tolist())

    for start, end, claim in _group_boundaries(data, "claim_label"):
        if end < len(data) - 1:
            ax.axhline(y[end] - 0.5, color="#E5E7EB", linewidth=0.8)
        ax.text(
            0.0,
            y[start] + 0.42,
            str(claim).replace("_", " "),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=8,
            color=TEXT_MUTED,
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=VISUAL_COLOR, markersize=5.8, label="Fusion vs visual-only"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor=AUDIO_COLOR, markersize=5.8, label="Fusion vs audio-only"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FUSION_COLOR, markeredgecolor=FUSION_COLOR, markersize=5.5, label="q < 0.05"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        ncol=3,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    save_figure_bundle(fig, out_base, dpi=600)
