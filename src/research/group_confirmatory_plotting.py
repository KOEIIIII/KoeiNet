


"""Paper-style plotting for group-level confirmatory relationship analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .plotting import apply_paper_style, save_figure_bundle
from .utils import pretty_group_name

matplotlib.use("Agg")

CLAIM_COLORS = {
    "supported_confirmatory": "#0f172a",
    "directionally_consistent_but_not_significant": "#5b6b7a",
    "unstable_under_sensitivity": "#b45309",
    "not_supported": "#94a3b8",
    "not_testable": "#cbd5e1",
}

VISUAL_COLOR = "#1b3a4b"
AUDIO_COLOR = "#6c7a89"


def plot_group_confirmatory_heatmap(
    combined_df: pd.DataFrame,
    *,
    visual_groups: Sequence[str],
    audio_groups: Sequence[str],
    out_base: Path,
) -> None:
    apply_paper_style()
    if combined_df.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.text(0.5, 0.5, "No group-pair tests available", ha="center", va="center")
        ax.set_axis_off()
        save_figure_bundle(fig, out_base)
        return

    sub = combined_df[combined_df["testable_flag"]].copy()
    pivot = sub.pivot(index="visual_group", columns="audio_group", values="spearman_rho_full")
    pivot = pivot.reindex(index=list(visual_groups), columns=list(audio_groups))
    fig, ax = plt.subplots(figsize=(1.0 + 1.2 * len(audio_groups), 1.4 + 0.8 * len(visual_groups)))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(audio_groups)))
    ax.set_xticklabels([pretty_group_name(x) for x in audio_groups], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(visual_groups)))
    ax.set_yticklabels([pretty_group_name(x) for x in visual_groups])
    ax.set_title("Group-level cross-modal rho")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Spearman rho")

    lookup = {
        (str(row["visual_group"]), str(row["audio_group"])): row
        for _, row in sub.iterrows()
    }
    for i, visual_group in enumerate(visual_groups):
        for j, audio_group in enumerate(audio_groups):
            row = lookup.get((visual_group, audio_group))
            if row is None or not np.isfinite(data[i, j]):
                continue
            text = ""
            if bool(row.get("family") == "confirmatory"):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False, edgecolor="#111827", linewidth=1.2))
            if float(row.get("spearman_q_full", np.nan)) < 0.05:
                text = "*"
            if text:
                ax.text(j, i, text, ha="center", va="center", color="#111827", fontsize=11)
    save_figure_bundle(fig, out_base)


def plot_confirmatory_forest(confirm_df: pd.DataFrame, out_base: Path) -> None:
    apply_paper_style()
    if confirm_df.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.text(0.5, 0.5, "No confirmatory hypotheses available", ha="center", va="center")
        ax.set_axis_off()
        save_figure_bundle(fig, out_base)
        return

    claim_col = "confirmatory_claim" if "confirmatory_claim" in confirm_df.columns else "claim_label"
    data = confirm_df.sort_values([claim_col, "hypothesis_id"]).copy().reset_index(drop=True)
    data["label"] = data["hypothesis_id"] + " | " + data["visual_group"].map(pretty_group_name) + " vs " + data["audio_group"].map(pretty_group_name)
    y = np.arange(len(data))[::-1]
    colors = [CLAIM_COLORS.get(str(x), "#94a3b8") for x in data[claim_col]]
    fig, ax = plt.subplots(figsize=(9.0, max(4.0, len(data) * 0.55)))
    ax.axvline(0.0, color="#6b7280", linewidth=0.8, linestyle="--")
    for idx, (_, row) in enumerate(data.iterrows()):
        lower = float(row["spearman_rho_full"] - row["spearman_ci_low"])
        upper = float(row["spearman_ci_high"] - row["spearman_rho_full"])
        ax.errorbar(
            [float(row["spearman_rho_full"])],
            [y[idx]],
            xerr=np.array([[lower], [upper]], dtype=float),
            fmt="none",
            ecolor=colors[idx],
            elinewidth=1.1,
            capsize=2.8,
        )
    ax.scatter(data["spearman_rho_full"], y, s=38, color=colors, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"])
    ax.set_xlabel("Spearman rho (full sample)")
    ax.set_title("Confirmatory group-pair estimates")
    ax.grid(axis="x", alpha=0.35)
    save_figure_bundle(fig, out_base)


def plot_group_composite_profile(
    composite_df: pd.DataFrame,
    *,
    group_modalities: Mapping[str, str],
    out_base: Path,
) -> None:
    apply_paper_style()
    group_cols = [c for c in composite_df.columns if c != "segment_id" and c != "center_time_sec"]
    if not group_cols:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.text(0.5, 0.5, "No group composites available", ha="center", va="center")
        ax.set_axis_off()
        save_figure_bundle(fig, out_base)
        return

    order = sorted(group_cols, key=lambda g: (group_modalities.get(g, "z"), g))
    values = [pd.to_numeric(composite_df[g], errors="coerce").dropna().to_numpy(dtype=float) for g in order]
    colors = [VISUAL_COLOR if group_modalities.get(g) == "visual" else AUDIO_COLOR for g in order]
    fig, ax = plt.subplots(figsize=(8.5, max(4.2, len(order) * 0.5)))
    box = ax.boxplot(values, vert=False, patch_artist=True, widths=0.58, labels=[pretty_group_name(g) for g in order])
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.0)
    for median in box["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.1)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Composite score")
    ax.set_title("Group composite distributions")
    ax.grid(axis="x", alpha=0.35)
    save_figure_bundle(fig, out_base)
