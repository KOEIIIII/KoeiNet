


"""Render episode cards and export a PDF contact sheet."""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

from .export_utils import pick_cjk_font

matplotlib.use("Agg")

logger = logging.getLogger("deliverable.card_renderer")


def _wrap(text: str, width: int) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "N/A"
    return textwrap.fill(raw, width=max(10, int(width)))


def _section_box(
    ax: plt.Axes,
    *,
    title: str,
    body: str,
    font_prop: Optional[FontProperties],
    title_color: str = "#111827",
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        Rectangle((0, 0), 1, 1, facecolor="#F8FAFC", edgecolor="#D1D5DB", linewidth=0.8)
    )
    ax.text(
        0.03,
        0.93,
        title,
        va="top",
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=title_color,
        fontproperties=font_prop,
    )
    ax.text(
        0.03,
        0.82,
        body,
        va="top",
        ha="left",
        fontsize=9.1,
        color="#1F2937",
        fontproperties=font_prop,
    )


def render_problem_episode_cards(
    merged_df: pd.DataFrame,
    *,
    out_dir: Path,
) -> Tuple[List[Dict[str, str]], str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    font_prop, font_name = pick_cjk_font()
    plt.rcParams["axes.unicode_minus"] = False
    card_records: List[Dict[str, str]] = []

    for _, row in merged_df.iterrows():
        episode_id = str(row["episode_id"])
        out_path = out_dir / f"{episode_id}.png"
        fig = plt.figure(figsize=(12.2, 7.2), facecolor="white")
        gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1.0], height_ratios=[0.72, 0.64, 0.84], hspace=0.18, wspace=0.15)

        ax_img = fig.add_subplot(gs[:, 0])
        ax_head = fig.add_subplot(gs[0, 1])
        ax_sound = fig.add_subplot(gs[1, 1])
        ax_bottom = fig.add_subplot(gs[2, 1])

        hero_path = Path(str(row.get("hero_frame_path", "")))
        if hero_path.exists():
            try:
                img = mpimg.imread(hero_path.as_posix())
                ax_img.imshow(img)
                ax_img.set_title("Representative frame", fontsize=10.5, fontproperties=font_prop)
            except Exception as exc:
                logger.warning("card image load failed for %s: %s", hero_path.as_posix(), exc)
                ax_img.text(0.5, 0.5, "Frame unavailable", ha="center", va="center", fontproperties=font_prop)
        else:
            ax_img.text(0.5, 0.5, "Frame unavailable", ha="center", va="center", fontproperties=font_prop)
        ax_img.axis("off")

        title_text = "\n".join(
            [
                _wrap(str(row.get("episode_title", "")), 24),
                f"Time: {float(row.get('start_time_sec', 0.0)):.1f}s - {float(row.get('end_time_sec', 0.0)):.1f}s",
                f"Rep. segment: {int(row.get('representative_segment_id', -1))}",
                f"Hero frame: {row.get('hero_frame_index', 'N/A')}",
            ]
        )
        _section_box(
            ax_head,
            title="Episode overview",
            body=title_text,
            font_prop=font_prop,
            title_color="#0F172A",
        )

        sound_body = "\n\n".join(
            [
                "Soundscape problem:\n" + _wrap(str(row.get("soundscape_problem", "")), 28),
                "Visual problem:\n" + _wrap(str(row.get("visual_problem", "")), 28),
            ]
        )
        _section_box(
            ax_sound,
            title="Problem statement",
            body=sound_body,
            font_prop=font_prop,
            title_color="#1D4ED8",
        )

        prompt_excerpt = str(row.get("edit_prompt", ""))[:430]
        bottom_body = "\n\n".join(
            [
                "Fused judgement:\n" + _wrap(str(row.get("fused_problem", "")), 30),
                "Why problematic:\n" + _wrap(str(row.get("why_it_is_problematic", "")), 30),
                "Prompt excerpt:\n" + _wrap(prompt_excerpt, 42),
                f"Priority: {row.get('priority_level', '')} | Confidence: {row.get('confidence_level', '')} | Theme: {row.get('suggested_intervention_theme', '')}",
            ]
        )
        _section_box(
            ax_bottom,
            title="Design direction",
            body=bottom_body,
            font_prop=font_prop,
            title_color="#B45309",
        )

        fig.suptitle(
            str(row.get("short_caption", row.get("episode_title", episode_id))),
            x=0.52,
            y=0.985,
            fontsize=12,
            fontweight="bold",
            color="#111827",
            fontproperties=font_prop,
        )
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        card_records.append({"episode_id": episode_id, "card_png": out_path.as_posix()})

    logger.info("deliverable cards rendered | n=%s font=%s", len(card_records), font_name)
    return card_records, font_name


def export_contact_sheet_pdf(card_records: Sequence[Mapping[str, str]], *, out_path: Path) -> Optional[str]:
    if not card_records:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cards = [Path(str(item.get("card_png", ""))) for item in card_records if Path(str(item.get("card_png", ""))).exists()]
    if not cards:
        return None
    with PdfPages(out_path) as pdf:
        for page_start in range(0, len(cards), 4):
            page_cards = cards[page_start : page_start + 4]
            fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
            for ax in axes.ravel():
                ax.axis("off")
            for ax, card_path in zip(axes.ravel(), page_cards):
                try:
                    img = mpimg.imread(card_path.as_posix())
                    ax.imshow(img)
                    ax.set_title(card_path.stem, fontsize=9)
                except Exception as exc:
                    logger.warning("contact sheet image load failed for %s: %s", card_path.as_posix(), exc)
                    ax.text(0.5, 0.5, card_path.stem, ha="center", va="center")
            fig.tight_layout()
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
            plt.close(fig)
    logger.info("deliverable contact sheet exported | %s", out_path.as_posix())
    return out_path.as_posix()
