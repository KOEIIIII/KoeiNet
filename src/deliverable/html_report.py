


"""HTML and markdown report generation for deliverable layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

from .export_utils import safe_relative_path


def write_problem_episode_html(
    merged_df: pd.DataFrame,
    *,
    deliverable_dir: Path,
    card_dir: Path,
    out_path: Path,
) -> str:
    cards_html = []
    for _, row in merged_df.iterrows():
        card_path = card_dir / f"{row['episode_id']}.png"
        card_rel = safe_relative_path(card_path, deliverable_dir)
        hero_rel = safe_relative_path(Path(str(row.get("hero_frame_path", ""))), deliverable_dir)
        cards_html.append(
            f"""
<section class="card">
  <div class="thumb">
    <a href="{card_rel}"><img src="{card_rel}" alt="{row['episode_id']}"></a>
  </div>
  <div class="meta">
    <h2>{row.get('episode_title','')}</h2>
    <p class="small"><strong>Episode:</strong> {row['episode_id']} | <strong>Time:</strong> {row['start_time_sec']:.1f}s - {row['end_time_sec']:.1f}s | <strong>Rep segment:</strong> {row['representative_segment_id']}</p>
    <p><strong>Summary:</strong> {row.get('one_sentence_summary','')}</p>
    <p><strong>Soundscape:</strong> {row.get('soundscape_problem','')}</p>
    <p><strong>Visual:</strong> {row.get('visual_problem','')}</p>
    <p><strong>Fused:</strong> {row.get('fused_problem','')}</p>
    <p><strong>Intervention theme:</strong> {row.get('suggested_intervention_theme','')}</p>
    <p><strong>Prompt:</strong> {row.get('edit_prompt','')}</p>
    <p><strong>Negative prompt:</strong> {row.get('negative_prompt','')}</p>
    <p class="small"><strong>Hero frame:</strong> <a href="{hero_rel}">{hero_rel}</a></p>
  </div>
</section>
"""
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Problem Episode Cards</title>
  <style>
    body {{ font-family: "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; margin: 24px; background: #ffffff; color: #111827; }}
    h1 {{ font-size: 24px; margin-bottom: 6px; }}
    .lead {{ color: #4b5563; margin-bottom: 24px; }}
    .card {{ display: grid; grid-template-columns: 360px 1fr; gap: 20px; margin-bottom: 28px; padding: 18px; border: 1px solid #d1d5db; border-radius: 12px; }}
    .thumb img {{ width: 100%; border-radius: 8px; border: 1px solid #e5e7eb; }}
    .meta h2 {{ margin: 0 0 10px 0; font-size: 20px; }}
    .meta p {{ margin: 7px 0; line-height: 1.45; }}
    .small {{ color: #6b7280; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Problem Episode Overview</h1>
  <p class="lead">Deliverable layer overview page generated from Step 8 and related evidence outputs.</p>
  {''.join(cards_html)}
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path.as_posix()


def build_deliverable_summary_markdown(
    *,
    resolved_paths: Mapping[str, str],
    mapping_notes: Sequence[str],
    episodes_df: pd.DataFrame,
    final_df: pd.DataFrame,
    prompt_mode: str,
    selection_mode: str,
    font_name: str,
    rendered_outputs: Mapping[str, str],
) -> str:
    lines = [
        "# Deliverable Layer Summary",
        "",
        "## 1) 输入文件清单",
    ]
    for key, value in sorted(resolved_paths.items()):
        lines.append(f"- `{key}` -> `{value}`")
    if mapping_notes:
        lines.append("")
        lines.append("### 文件名自动匹配 / 适配说明")
        lines.extend([f"- {item}" for item in mapping_notes])
    lines.extend(
        [
            "",
            "## 2) 实际识别出的 problem episodes",
            f"- 共识别出 `{len(episodes_df)}` 个 episodes。",
            f"- segment 筛选策略：`{selection_mode}`。",
            "- episode 合并规则：按时间排序后，若相邻高优先级 segment 的时间间隔不超过 `max_gap_seconds` 或重叠比例超过阈值，则并入同一 episode。",
            "",
            "## 3) 每个 episode 的时间范围与代表帧策略",
        ]
    )
    for _, row in episodes_df.iterrows():
        lines.append(
            f"- `{row['episode_id']}`: {row['start_time_sec']:.1f}s–{row['end_time_sec']:.1f}s, "
            f"代表 segment={row['representative_segment_id']}, hero frame={row.get('hero_frame_index','')}"
        )
    lines.extend(
        [
            "",
            "## 4) 声景问题如何提取",
            "- 以 `audio_events__group_ratio_*`、`audio_signal__loudness_proxy_db`、validation 中的 pleasantness/eventfulness、以及 Step 8 `soundscape_state` 为主要依据。",
            "- 通过确定性阈值与全视频分位数，将 episode 标记为 `traffic_mechanical_dominant`、`high_loudness`、`low_natural_sound`、`human_voice_dominant` 等可读标签。",
            "",
            "## 5) 视觉问题如何提取",
            "- 以 `people__total_people__mean`、`green_view__greenviewindex__mean`、`visual_major__construction__mean`、`visual_major__vehicle__mean`、`visual_semantic__road/sidewalk`、以及 emotion 相关指标为主。",
            "- 通过确定性规则生成 `crowding`、`low_green_view`、`high_hardscape`、`vehicle_dominance`、`poor_walkability_cues`、`low_aesthetic_quality` 等标签。",
            "",
            "## 6) 融合问题如何形成",
            "- 融合问题不是 LLM 自由判断，而是基于视觉标签、声景标签和 `multimodal_consistency_flag` 的规则组合生成。",
            "- 代表性组合包括：`crowded_and_noise_dominant`、`low_greenery_with_high_mechanical_noise`、`active_but_acoustically_harsh` 等。",
            "",
            "## 7) prompt 生成模式",
            f"- 当前 prompt 生成模式：`{prompt_mode}`。",
            "- 若 GLM 可用且显式启用，则只做受控润色；否则始终使用 deterministic template prompt。",
            "",
            "## 8) 输出的卡片与报告文件",
        ]
    )
    for key, value in rendered_outputs.items():
        if value:
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## 9) 主流程与 deliverable layer 的衔接",
            "- Step 7.5 仍是最终模型证据层。",
            "- Step 8 仍是设计映射主入口。",
            "- deliverable layer 仅在 Step 8 之后把 segment-level 结果合并为 episode，并封装为可汇报、可浏览、可导出的最终交付件。",
            "",
            f"卡片渲染字体回退结果：`{font_name}`",
        ]
    )
    return "\n".join(lines)


def build_deliverable_onepage_markdown(
    final_df: pd.DataFrame,
    *,
    best_view_file: Optional[str],
) -> str:
    lines = [
        "# Deliverable One-Page Report",
        "",
        "## 前若干问题路段",
    ]
    top_df = final_df.sort_values(["priority_rank", "priority_score"]).head(6)
    for _, row in top_df.iterrows():
        lines.append(
            f"- `{row['episode_id']}` {row['start_time_sec']:.1f}s–{row['end_time_sec']:.1f}s: "
            f"{row.get('problem_summary', '')} | intervention={row.get('intervention_theme','')} | prompt={str(row.get('short_caption',''))}"
        )
    lines.extend(
        [
            "",
            "## 最适合直接查看最终成果的文件",
            f"- `{best_view_file or ''}`",
        ]
    )
    return "\n".join(lines)


def build_shadow_eval_notes() -> str:
    return "\n".join(
        [
            "# Shadow Evaluation Notes",
            "",
            "当前 deliverable layer 不会重跑 Step 7.5，只提供将来的评估接口说明。",
            "",
            "## Future-ready split configuration",
            "```yaml",
            "split_strategy: time_aware_or_grouped",
            "group_key: video_id_or_route_id",
            "time_key: center_time_sec",
            "gap_segments: 1",
            "block_size: 2",
            "notes:",
            "  - For multi-video studies, prefer GroupKFold-style separation by route/video/source.",
            "  - For single long-route studies, prefer blocked or time-aware split with an explicit temporal gap.",
            "  - Never evaluate adjacent overlapping windows across train/test without a guard gap.",
            "```",
            "",
            "## Integration intent",
            "- Future Step 7.5 reruns can plug into this schema without changing the current deliverable layer outputs.",
            "- The current implementation keeps this as documentation only and does not alter the stable refined pipeline.",
        ]
    )
