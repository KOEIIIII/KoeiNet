


"""Paper-style plotting helpers for research modules."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import pretty_model_name, short_feature_label

matplotlib.use("Agg")

VISUAL_COLOR = "#1b3a4b"
AUDIO_COLOR = "#6c7a89"
FUSION_COLOR = "#0f172a"
BASELINE_COLOR = "#94a3b8"
GRID_COLOR = "#d9dee5"


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#4b5563",
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.6,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure_bundle(fig: plt.Figure, out_base: Path, dpi: int = 300) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=int(dpi), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def plot_cross_modal_heatmap(group_df: pd.DataFrame, out_base: Path) -> None:
    apply_paper_style()
    if group_df.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.text(0.5, 0.5, "No group summary available", ha="center", va="center")
        ax.set_axis_off()
        save_figure_bundle(fig, out_base)
        return
    pivot = group_df.pivot(index="visual_group", columns="audio_group", values="group_effect")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    data = pivot.to_numpy(dtype=float)
    vmax = float(np.nanmax(data)) if np.isfinite(data).any() else 1.0
    im = ax.imshow(data, cmap="bone_r", vmin=0.0, vmax=max(vmax, 1e-6), aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(x).replace("_", " ") for x in pivot.columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(x).replace("_", " ") for x in pivot.index])
    ax.set_title("Cross-modal coupling")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean combined effect")
    save_figure_bundle(fig, out_base)


def plot_top_links_lollipop(links_df: pd.DataFrame, out_base: Path, top_n: int = 15) -> None:
    apply_paper_style()
    top = links_df.sort_values(["effect_rank", "combined_effect"]).head(int(top_n)).copy()
    if top.empty:
        top = pd.DataFrame(
            {
                "pair_label": ["No significant links"],
                "spearman_rho": [0.0],
                "direction_consistent": [False],
            }
        )
    else:
        top["pair_label"] = top["visual_feature"].map(short_feature_label) + " | " + top["audio_feature"].map(short_feature_label)
    top = top.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(8.2, max(4.0, len(top) * 0.34)))
    ax.axvline(0.0, color="#9ca3af", linewidth=0.9)
    line_colors = np.where(top["direction_consistent"].to_numpy(dtype=bool), VISUAL_COLOR, AUDIO_COLOR)
    ax.hlines(y=y, xmin=0.0, xmax=top["spearman_rho"], color=line_colors, linewidth=1.1)
    ax.scatter(top["spearman_rho"], y, s=38, color=line_colors, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(top["pair_label"])
    ax.set_xlabel("Spearman rho")
    ax.set_title("Top audio-visual links")
    ax.grid(axis="x", alpha=0.45)
    save_figure_bundle(fig, out_base)


def plot_pls_scores(scores_df: pd.DataFrame, out_base: Path) -> None:
    apply_paper_style()
    components = sorted(scores_df["component"].dropna().unique().tolist())
    n_cols = max(1, len(components))
    fig, axes = plt.subplots(1, n_cols, figsize=(4.8 * n_cols, 4.1), squeeze=False)
    for i, component in enumerate(components):
        ax = axes[0, i]
        sub = scores_df[scores_df["component"] == component].copy()
        ax.scatter(sub["x_score"], sub["y_score"], s=28, color=VISUAL_COLOR, alpha=0.85)
        if len(sub) >= 2 and sub["x_score"].nunique() > 1:
            slope, intercept = np.polyfit(sub["x_score"], sub["y_score"], deg=1)
            x_line = np.linspace(float(sub["x_score"].min()), float(sub["x_score"].max()), 100)
            ax.plot(x_line, intercept + slope * x_line, color="#111827", linewidth=1.0)
        ax.set_xlabel("Visual score")
        ax.set_ylabel("Audio score")
        corr = float(sub["score_correlation"].iloc[0]) if "score_correlation" in sub.columns else np.nan
        p_value = float(sub["permutation_p"].iloc[0]) if "permutation_p" in sub.columns else np.nan
        ax.set_title(f"LV{int(component)} (r={corr:.2f}, p={p_value:.3f})")
        ax.grid(alpha=0.35)
    save_figure_bundle(fig, out_base)


def _plot_loading_panel(ax: plt.Axes, df: pd.DataFrame, color: str, title: str, top_n: int) -> None:
    sub = df.copy()
    sub["abs_loading"] = sub["loading"].abs()
    sub = sub.sort_values("abs_loading", ascending=False).head(int(top_n)).iloc[::-1]
    ax.barh(np.arange(len(sub)), sub["loading"], color=color, alpha=0.92)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["feature_name"].map(short_feature_label))
    ax.axvline(0.0, color="#9ca3af", linewidth=0.8)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.35)


def plot_pls_loadings(loadings_df: pd.DataFrame, out_base: Path, top_n: int = 8) -> None:
    apply_paper_style()
    components = sorted(loadings_df["component"].dropna().unique().tolist())
    n_rows = max(1, len(components))
    fig, axes = plt.subplots(n_rows, 2, figsize=(10.4, 3.2 * n_rows), squeeze=False)
    for row_idx, component in enumerate(components):
        for col_idx, modality in enumerate(["visual", "audio"]):
            ax = axes[row_idx, col_idx]
            sub = loadings_df[(loadings_df["component"] == component) & (loadings_df["modality"] == modality)].copy()
            color = VISUAL_COLOR if modality == "visual" else AUDIO_COLOR
            _plot_loading_panel(ax, sub, color=color, title=f"LV{int(component)} {modality}", top_n=top_n)
    save_figure_bundle(fig, out_base)


def plot_fusion_delta_forest(forest_df: pd.DataFrame, out_base: Path) -> None:
    apply_paper_style()
    if forest_df.empty:
        fig, ax = plt.subplots(figsize=(8.0, 4.0))
        ax.text(0.5, 0.5, "No paired delta summary available", ha="center", va="center")
        ax.set_axis_off()
        save_figure_bundle(fig, out_base)
        return
    data = forest_df.sort_values(["target_tier", "target_name", "comparison"]).copy().reset_index(drop=True)
    data["display_label"] = data["target_name"] + " | " + data["comparison_label"]
    y = np.arange(len(data))[::-1]
    colors = data["comparison"].map(
        {
            "fusion_vs_visual_only": VISUAL_COLOR,
            "fusion_vs_audio_only": AUDIO_COLOR,
        }
    ).fillna(FUSION_COLOR).tolist()
    fig, ax = plt.subplots(figsize=(8.8, max(4.0, len(data) * 0.38)))
    ax.axvline(0.0, color="#6b7280", linewidth=0.9, linestyle="--")
    lower = (data["delta_mean"] - data["ci_lower"]).to_numpy(dtype=float)
    upper = (data["ci_upper"] - data["delta_mean"]).to_numpy(dtype=float)
    for idx, (_, row) in enumerate(data.iterrows()):
        ax.errorbar(
            [float(row["delta_mean"])],
            [y[idx]],
            xerr=np.array([[lower[idx]], [upper[idx]]], dtype=float),
            fmt="none",
            ecolor=colors[idx],
            elinewidth=1.1,
            capsize=2.8,
        )
    ax.scatter(data["delta_mean"], y, s=34, color=colors, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(data["display_label"])
    ax.set_xlabel("Delta in primary loss (negative favors fusion)")
    ax.set_title("Fusion delta with 95% CI")
    ax.grid(axis="x", alpha=0.35)
    save_figure_bundle(fig, out_base)


def plot_oof_performance_by_target(model_perf_df: pd.DataFrame, out_base: Path) -> None:
    apply_paper_style()
    if model_perf_df.empty:
        fig, ax = plt.subplots(figsize=(8.0, 4.0))
        ax.text(0.5, 0.5, "No OOF performance summary available", ha="center", va="center")
        ax.set_axis_off()
        save_figure_bundle(fig, out_base)
        return
    targets = model_perf_df["target_name"].dropna().unique().tolist()
    fig, axes = plt.subplots(len(targets), 1, figsize=(8.4, max(5.2, 2.15 * len(targets))), squeeze=False)
    for ax, target_name in zip(axes[:, 0], targets):
        sub = model_perf_df[model_perf_df["target_name"] == target_name].copy()
        sub = sub.sort_values("primary_value", ascending=True).reset_index(drop=True)
        y = np.arange(len(sub))
        ax.plot(sub["primary_value"], y, color=BASELINE_COLOR, linewidth=0.9)
        colors = [FUSION_COLOR if x == "early_fusion_screened" else BASELINE_COLOR for x in sub["model_group"]]
        ax.scatter(sub["primary_value"], y, s=36, color=colors, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([pretty_model_name(x) for x in sub["model_group"]])
        metric_label = str(sub["primary_metric_label"].iloc[0]) if "primary_metric_label" in sub.columns and not sub.empty else "Primary metric"
        ax.set_xlabel(metric_label)
        ax.set_title(str(target_name))
        ax.grid(axis="x", alpha=0.35)
    save_figure_bundle(fig, out_base)
