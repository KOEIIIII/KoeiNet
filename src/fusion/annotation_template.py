


"""Shared templates and constants for two-rater blind validation workflow."""

from __future__ import annotations

from typing import Dict, List

CONTROLLED_PRIMARY_PROBLEM_LABELS: List[str] = [
    "traffic_noise",
    "pedestrian_discomfort",
    "low_greenery",
    "visual_clutter",
    "weak_stay_quality",
    "low_vitality",
    "safety_risk",
    "mixed_or_unclear",
    "no_major_problem",
]

ANNOTATOR_PACK_COLUMNS: List[str] = [
    "displayed_item_id",
    "segment_id",
    "start_time_sec",
    "end_time_sec",
    "preview_path",
    "primary_preview_path",
    "context_strip_path",
    "audio_clip_path",
    "safety_score",
    "comfort_score",
    "vitality_score",
    "overall_problem_severity",
    "soundscape_pleasantness",
    "soundscape_eventfulness",
    "primary_problem_label",
    "confidence_score",
    "notes",
]

SCALAR_SCORE_FIELDS: List[str] = [
    "safety_score",
    "comfort_score",
    "vitality_score",
    "overall_problem_severity",
    "soundscape_pleasantness",
    "soundscape_eventfulness",
    "confidence_score",
]

CATEGORICAL_FIELDS: List[str] = ["primary_problem_label"]

PRIMARY_PROBLEM_LABEL_ZH: Dict[str, str] = {
    "traffic_noise": "交通噪声",
    "pedestrian_discomfort": "步行不适",
    "low_greenery": "绿化不足",
    "visual_clutter": "视觉杂乱",
    "weak_stay_quality": "停留品质弱",
    "low_vitality": "活力不足",
    "safety_risk": "安全风险",
    "mixed_or_unclear": "混合或不明确",
    "no_major_problem": "无明显问题",
}

LIKERT_GUIDE: Dict[str, str] = {
    "1": "非常低 / 非常差 / 强烈负向",
    "2": "较低",
    "3": "略低",
    "4": "中性 / 混合",
    "5": "略高",
    "6": "较高",
    "7": "非常高 / 非常好 / 强烈正向",
}


def build_annotation_instructions_markdown() -> str:
    """返回双评审盲评协议的中文标注说明。"""
    lines = [
        "# 双评审盲评标注说明",
        "",
        "本轮为盲评标注。两位评审需独立完成，不得在双方提交前讨论评分结果。",
        "",
        "## 通用规则",
        "- 每一行都要独立判断，只依据提供的主评分图与可选上下文图。",
        "- 请先查看主评分图，再试听音频片段，最后再填写“声景愉悦度”和“声景事件性”。",
        "- 安全性、舒适度、活力度、整体问题严重度可以综合图像与音频理解，但声景相关评分必须参考音频。",
        "- 所有量表字段使用 1-7 的整数评分。",
        "- 不要推测隐藏元数据。即使画面看起来重复，也请按新条目独立评分。",
        "- 除非该条目确实无法观察，否则不要留空评分项。",
        "",
        "## 评分维度说明",
        "- `safety_score`（安全性评分）：行人安全感与风险感知。",
        "- `comfort_score`（舒适度评分）：步行/停留的舒适程度。",
        "- `vitality_score`（活力度评分）：空间活力、活动性与街道生命感。",
        "- `overall_problem_severity`（整体问题严重度）：环境与空间问题的总体严重程度。",
        "- `soundscape_pleasantness`（声景愉悦度）：声音环境是否令人愉悦。",
        "- `soundscape_eventfulness`（声景事件性）：声音事件是否丰富、繁忙。",
        "- `confidence_score`（评分信心）：你对本条评分结论的把握程度。",
        "",
        "## 1-7 量表锚点",
    ]
    for k, v in LIKERT_GUIDE.items():
        lines.append(f"- `{k}`: {v}")

    lines.extend(
        [
            "",
            "## 主要问题标签（受控词表）",
            "从以下编码中选择一个最主要问题（系统内部保存英文编码）：",
            "- `traffic_noise`：交通噪声",
            "- `pedestrian_discomfort`：步行不适",
            "- `low_greenery`：绿化不足",
            "- `visual_clutter`：视觉杂乱",
            "- `weak_stay_quality`：停留品质弱",
            "- `low_vitality`：活力不足",
            "- `safety_risk`：安全风险",
            "- `mixed_or_unclear`：混合或不明确",
            "- `no_major_problem`：无明显问题",
            "",
            "## 备注",
            "- 用简洁文本记录关键证据、疑点或不确定性。",
            "",
            "## 独立性要求",
            "- 两位评审必须独立完成整个会话，提交前不得互相讨论。",
            "- 请随时保存，可中断后恢复继续填写。",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_anchor_examples_markdown() -> str:
    """返回中文锚点示例说明。"""
    lines = [
        "# 评分锚点示例",
        "",
        "以下示例用于统一不同评审之间的判断尺度。",
        "",
        "请先查看主评分图，再试听音频片段，最后再填写“声景愉悦度”和“声景事件性”。",
        "",
        "## 安全性 / 舒适度 / 活力度区分",
        "- 安全高、舒适低：防护较好但噪声大、拥挤明显。",
        "- 安全低、舒适中：过街风险较高，但通行条件尚可。",
        "- 活力高、舒适低：人流活动旺盛，但交通/人群压力明显。",
        "",
        "## 声景评分参考",
        "- 愉悦度高（6-7）：自然/柔和声音占主导，刺耳噪声少。",
        "- 愉悦度低（1-2）：交通或机械噪声刺耳且持续。",
        "- 事件性高（6-7）：声音事件频繁且类型多样。",
        "- 事件性低（1-2）：声音场景较静态、变化少。",
        "",
        "## 主要问题标签选择建议",
        "- `traffic_noise`：噪声负担最突出时选择。",
        "- `pedestrian_discomfort`：步行/停留不适最突出时选择。",
        "- `low_greenery`：绿量不足是主要问题时选择。",
        "- `visual_clutter`：画面杂乱、元素破碎感最突出时选择。",
        "- `weak_stay_quality`：空间停留吸引力和可停留性不足时选择。",
        "- `low_vitality`：街道活动弱、活力不足时选择。",
        "- `safety_risk`：安全隐患最突出时选择。",
        "- `mixed_or_unclear`：多个问题并存且主次不清时选择。",
        "- `no_major_problem`：未观察到明显主要问题时选择。",
        "",
        "## 评分信心参考",
        "- 1-2：证据不足或非常不确定。",
        "- 3-5：中等把握。",
        "- 6-7：证据明确、把握较高。",
    ]
    return "\n".join(lines).strip() + "\n"
