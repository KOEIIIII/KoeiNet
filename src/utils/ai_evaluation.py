


"""
AI 六项活动评估模块

主模式:
- 基于智谱 AI（多模态）+ Prompt 的逐帧评估

兜底模式:
- 当未配置智谱 API Key 或请求失败时，使用本地统计指标规则融合打分
"""

import os
import re
import json
import time
import base64
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


ACTIVITY_SPECS = [
    ("坐下休息", "Sitting", "坐下休息_score", "坐下休息_reason"),
    ("站着停留", "Standing", "站着停留_score", "站着停留_reason"),
    ("散步", "Walking", "散步_score", "散步_reason"),
    ("跑步", "Running", "跑步_score", "跑步_reason"),
    ("健身锻炼", "Fitness", "健身锻炼_score", "健身锻炼_reason"),
    ("买菜购物", "Shopping", "买菜购物_score", "买菜购物_reason"),
]

PROMPT_TEMPLATE = """你是城市街道空间活动适宜性评估专家。请仅基于给定街景图片进行判断。
请对以下六项活动分别打分（1-5分，1最低，5最高），并给出每项不超过30字的中文理由：
1. 坐下休息
2. 站着停留
3. 散步
4. 跑步
5. 健身锻炼
6. 买菜购物

必须严格输出 JSON，不要输出任何额外文字、解释或 Markdown：
{
  "坐下休息": {"score": 1, "reason": "原因"},
  "站着停留": {"score": 1, "reason": "原因"},
  "散步": {"score": 1, "reason": "原因"},
  "跑步": {"score": 1, "reason": "原因"},
  "健身锻炼": {"score": 1, "reason": "原因"},
  "买菜购物": {"score": 1, "reason": "原因"}
}
"""


def _extract_frame_num(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and not np.isnan(value):
        return int(value)
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else None


def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("读取 CSV 失败 %s: %s", path, exc)
        return None


def _load_local_env_file() -> Dict[str, str]:
    """
    从项目根目录读取 apikey.env（如果存在）。
    不覆盖系统环境变量，且只解析 KEY=VALUE 的简单格式。
    """
    env_map: Dict[str, str] = {}
    root_dir = Path(__file__).resolve().parents[2]
    env_path = root_dir / "apikey.env"
    if not env_path.exists():
        return env_map

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                env_map[key] = value
    except Exception as exc:
        logger.warning("读取 apikey.env 失败: %s", exc)

    return env_map


def _prepare_frame_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    out = df.copy()
    if "FrameNum" in out.columns:
        out["FrameNum"] = pd.to_numeric(out["FrameNum"], errors="coerce")
    elif "frame_num" in out.columns:
        out["FrameNum"] = pd.to_numeric(out["frame_num"], errors="coerce")
    elif "Frame" in out.columns:
        out["FrameNum"] = out["Frame"].apply(_extract_frame_num)
    elif "frame_name" in out.columns:
        out["FrameNum"] = out["frame_name"].apply(_extract_frame_num)
    else:
        return None

    out = out.dropna(subset=["FrameNum"]).copy()
    out["FrameNum"] = out["FrameNum"].astype(int)
    return out


def _get_zhipu_client():
    local_env = _load_local_env_file()


    api_key = (
        os.getenv("ZHIPUAI_API_KEY")
        or os.getenv("ZHIPU_API_KEY")
        or local_env.get("ZHIPUAI_API_KEY")
        or local_env.get("ZHIPU_API_KEY")
    )
    if not api_key:
        return None, "未找到 ZHIPUAI_API_KEY/ZHIPU_API_KEY"
    try:
        from zhipuai import ZhipuAI
        return ZhipuAI(api_key=api_key), None
    except Exception as exc:
        return None, f"导入/初始化 zhipuai 失败: {exc}"


def _encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def _extract_json_text(text: str) -> Optional[str]:
    raw = (text or "").strip()
    if not raw:
        return None


    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)


    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        return m.group(0)
    return None


def _normalize_activity_json(obj: Dict) -> Dict[str, Dict[str, str]]:
    aliases = {
        "sitting": "坐下休息",
        "standing": "站着停留",
        "walking": "散步",
        "running": "跑步",
        "fitness": "健身锻炼",
        "exercise": "健身锻炼",
        "shopping": "买菜购物",
    }

    normalized: Dict[str, Dict[str, str]] = {}

    obj2: Dict = {}
    for k, v in obj.items():
        key = aliases.get(str(k).strip().lower(), str(k).strip())
        obj2[key] = v

    for cn, _, _, _ in ACTIVITY_SPECS:
        item = obj2.get(cn, {})
        score = 3
        reason = "模型未返回有效理由"

        if isinstance(item, dict):
            raw_score = item.get("score", 3)
            raw_reason = item.get("reason", reason)
        else:
            raw_score = item
            raw_reason = reason

        try:
            score = int(round(float(raw_score)))
        except Exception:
            score = 3
        score = max(1, min(5, score))

        if isinstance(raw_reason, str) and raw_reason.strip():
            reason = raw_reason.strip().replace("\n", " ")

        normalized[cn] = {"score": score, "reason": reason}

    return normalized


def _parse_model_result(content: str) -> Optional[Dict[str, Dict[str, str]]]:
    json_text = _extract_json_text(content)
    if not json_text:
        return None
    try:
        data = json.loads(json_text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return _normalize_activity_json(data)


def _call_zhipu_once(client, model: str, image_b64: str, prompt: str):
    data_url = f"data:image/jpeg;base64,{image_b64}"


    payload_variants = [
        [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
        [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": data_url},
        ],
    ]

    last_exc = None
    for content in payload_variants:
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[{"role": "user", "content": content}],
            )

            choice = resp.choices[0].message.content
            if isinstance(choice, list):
                text_parts = []
                for part in choice:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    else:
                        text_parts.append(str(part))
                return "\n".join(text_parts).strip()

            return str(choice).strip()
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(f"智谱请求失败: {last_exc}")


def _select_frame_paths(video_dir: str) -> List[str]:
    frame_dir = os.path.join(video_dir, "frames")
    if not os.path.isdir(frame_dir):
        return []

    frame_paths = [
        os.path.join(frame_dir, name)
        for name in os.listdir(frame_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png")) and name.startswith("frame_")
    ]
    frame_paths = sorted(frame_paths)
    if not frame_paths:
        return frame_paths

    local_env = _load_local_env_file()
    stride = max(1, int(os.getenv("AI_ACTIVITY_FRAME_STRIDE") or local_env.get("AI_ACTIVITY_FRAME_STRIDE") or "1"))
    frame_paths = frame_paths[::stride]

    max_frames = max(1, int(os.getenv("AI_ACTIVITY_MAX_FRAMES") or local_env.get("AI_ACTIVITY_MAX_FRAMES") or "120"))
    if len(frame_paths) > max_frames:
        idx = np.linspace(0, len(frame_paths) - 1, max_frames).astype(int)
        frame_paths = [frame_paths[i] for i in idx]

    return frame_paths


def _evaluate_with_zhipu(video_dir: str, output_dir: str) -> Optional[pd.DataFrame]:
    client, err = _get_zhipu_client()
    if client is None:
        logger.warning("AI 活动评估切换到兜底模式: %s", err)
        return None

    frame_paths = _select_frame_paths(video_dir)
    if not frame_paths:
        logger.warning("AI 活动评估切换到兜底模式: 未找到可评估帧")
        return None

    local_env = _load_local_env_file()
    model = os.getenv("AI_ACTIVITY_MODEL") or local_env.get("AI_ACTIVITY_MODEL") or "glm-4v-plus"
    retry = max(0, int(os.getenv("AI_ACTIVITY_RETRY") or local_env.get("AI_ACTIVITY_RETRY") or "2"))
    sleep_s = float(os.getenv("AI_ACTIVITY_SLEEP") or local_env.get("AI_ACTIVITY_SLEEP") or "0.2")

    rows = []
    for i, frame_path in enumerate(frame_paths, 1):
        frame_file = os.path.basename(frame_path)
        frame_num = _extract_frame_num(frame_file)
        frame_name = os.path.splitext(frame_file)[0]
        if frame_num is None:
            continue

        raw_text = ""
        parsed = None

        for attempt in range(retry + 1):
            try:
                b64 = _encode_image_base64(frame_path)
                raw_text = _call_zhipu_once(client, model, b64, PROMPT_TEMPLATE)
                parsed = _parse_model_result(raw_text)
                if parsed:
                    break
            except Exception as exc:
                raw_text = f"请求失败: {exc}"
                time.sleep(0.8 * (attempt + 1))


        if not parsed:
            parsed = {
                cn: {"score": 3, "reason": "模型返回异常，使用中性分"}
                for cn, _, _, _ in ACTIVITY_SPECS
            }

        row = {"frame_name": frame_name, "frame_num": int(frame_num)}
        response_lines = []
        for cn, _, score_col, reason_col in ACTIVITY_SPECS:
            score = int(parsed[cn]["score"])
            reason = parsed[cn]["reason"]
            row[score_col] = score
            row[reason_col] = reason
            response_lines.append(f"{cn}：{score}分 - {reason}")
        row["full_response"] = raw_text if raw_text else "\n".join(response_lines)
        rows.append(row)

        if i % 10 == 0 or i == len(frame_paths):
            logger.info("AI 活动评估进度: %d/%d", i, len(frame_paths))

        if sleep_s > 0:
            time.sleep(sleep_s)

    if not rows:
        return None

    result = pd.DataFrame(rows).sort_values("frame_num").reset_index(drop=True)
    return result


def _build_metrics_base(video_dir: str) -> pd.DataFrame:
    stats_dir = os.path.join(video_dir, "stats")
    major_csv = os.path.join(stats_dir, "visual_elements", "major_categories_proportion.csv")
    green_csv = os.path.join(stats_dir, "green_view", "green_view_index.csv")
    emotion_csv = os.path.join(stats_dir, "emotion", "emotion_scores.csv")
    people_csv = os.path.join(stats_dir, "people_count", "people_count.csv")

    major_df = _prepare_frame_df(_safe_read_csv(major_csv))
    if major_df is None or major_df.empty:
        raise FileNotFoundError(f"缺少活动评估所需文件: {major_csv}")

    keep_major = ["Frame", "FrameNum", "Flat", "Construction", "Object", "Nature", "Sky", "Human", "Vehicle"]
    for col in keep_major:
        if col not in major_df.columns:
            major_df[col] = np.nan
    merged = major_df[keep_major].copy()

    green_df = _prepare_frame_df(_safe_read_csv(green_csv))
    if green_df is not None and "GreenViewIndex" in green_df.columns:
        merged = merged.merge(green_df[["FrameNum", "GreenViewIndex"]], on="FrameNum", how="left")

    emotion_df = _prepare_frame_df(_safe_read_csv(emotion_csv))
    if emotion_df is not None:
        cols = [c for c in ["beautiful", "lively", "safety"] if c in emotion_df.columns]
        if cols:
            merged = merged.merge(emotion_df[["FrameNum"] + cols], on="FrameNum", how="left")

    people_df = _prepare_frame_df(_safe_read_csv(people_csv))
    if people_df is not None and "total_people" in people_df.columns:
        merged = merged.merge(people_df[["FrameNum", "total_people"]], on="FrameNum", how="left")

    return merged.sort_values("FrameNum").reset_index(drop=True)


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).clip(0, 1)


def _evaluate_with_fallback_metrics(video_dir: str) -> pd.DataFrame:
    df = _build_metrics_base(video_dir)

    green = _clip01(df.get("GreenViewIndex", pd.Series(0, index=df.index)))
    flat = _clip01(df.get("Flat", pd.Series(0, index=df.index)))
    construction = _clip01(df.get("Construction", pd.Series(0, index=df.index)))
    obj = _clip01(df.get("Object", pd.Series(0, index=df.index)))
    nature = _clip01(df.get("Nature", pd.Series(0, index=df.index)))
    sky = _clip01(df.get("Sky", pd.Series(0, index=df.index)))
    human = _clip01(df.get("Human", pd.Series(0, index=df.index)))
    vehicle = _clip01(df.get("Vehicle", pd.Series(0, index=df.index)))
    beautiful = _clip01(df.get("beautiful", pd.Series(0.5, index=df.index)))
    lively = _clip01(df.get("lively", pd.Series(0.5, index=df.index)))
    safety = _clip01(df.get("safety", pd.Series(0.5, index=df.index)))
    people = pd.to_numeric(df.get("total_people", pd.Series(0, index=df.index)), errors="coerce").fillna(0).clip(lower=0)

    people_norm = (people / 12.0).clip(0, 1)
    quiet = 1 - people_norm
    people_mid = (1 - (people_norm - 0.35).abs() / 0.35).clip(0, 1)
    openness = (0.6 * flat + 0.2 * sky + 0.2 * (1 - construction)).clip(0, 1)
    commercial = (0.45 * construction + 0.25 * human + 0.15 * obj + 0.15 * vehicle).clip(0, 1)

    def to_score(raw: pd.Series) -> pd.Series:
        return (1 + 4 * raw.clip(0, 1)).clip(1, 5).round(3)

    scores = {
        "坐下休息_score": to_score(0.28 * safety + 0.22 * green + 0.20 * quiet + 0.15 * flat + 0.15 * beautiful),
        "站着停留_score": to_score(0.28 * safety + 0.20 * human + 0.18 * construction + 0.17 * people_mid + 0.17 * (1 - vehicle)),
        "散步_score": to_score(0.26 * flat + 0.22 * safety + 0.18 * nature + 0.16 * green + 0.10 * people_mid + 0.08 * (1 - vehicle)),
        "跑步_score": to_score(0.30 * flat + 0.24 * safety + 0.18 * openness + 0.15 * (1 - vehicle) + 0.13 * quiet),
        "健身锻炼_score": to_score(0.27 * flat + 0.22 * safety + 0.20 * nature + 0.16 * lively + 0.15 * quiet),
        "买菜购物_score": to_score(0.33 * commercial + 0.22 * safety + 0.20 * people_mid + 0.15 * construction + 0.10 * vehicle),
    }

    out = pd.DataFrame(scores)
    out["frame_num"] = df["FrameNum"].astype(int)
    out["frame_name"] = out["frame_num"].map(lambda x: f"frame_{int(x):06d}")

    for cn, _, score_col, reason_col in ACTIVITY_SPECS:
        out[reason_col] = "智谱评估不可用，使用本地规则融合评分"

    lines = []
    for _, row in out.iterrows():
        items = [f"{cn}：{row[score_col]:.2f}分 - {row[reason_col]}" for cn, _, score_col, reason_col in ACTIVITY_SPECS]
        lines.append("\n".join(items))
    out["full_response"] = lines
    return out


def _write_scores_csv(df: pd.DataFrame, output_dir: str) -> str:
    ordered_cols = [
        "坐下休息_score", "站着停留_score", "散步_score", "跑步_score", "健身锻炼_score", "买菜购物_score",
        "坐下休息_reason", "站着停留_reason", "散步_reason", "跑步_reason", "健身锻炼_reason", "买菜购物_reason",
        "frame_name", "frame_num", "full_response",
    ]
    result = df[ordered_cols].copy()
    path = os.path.join(output_dir, "activity_scores.csv")
    result.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _write_summary_csv(df: pd.DataFrame, output_dir: str) -> str:
    rows = []
    for cn, en, score_col, _ in ACTIVITY_SPECS:
        mean_score = float(pd.to_numeric(df[score_col], errors="coerce").mean())
        max_score = float(pd.to_numeric(df[score_col], errors="coerce").max())
        min_score = float(pd.to_numeric(df[score_col], errors="coerce").min())
        suitable = mean_score >= 3.0
        rows.append({
            "Activity": cn,
            "Activity_EN": en,
            "Mean_Score": round(mean_score, 4),
            "Max_Score": round(max_score, 4),
            "Min_Score": round(min_score, 4),
            "Is_Suitable": bool(suitable),
            "Suitable_Reasons": "综合评分达到可接受阈值" if suitable else "",
            "Unsuitable_Reasons": "" if suitable else "综合评分偏低，建议优化空间安全与可达性",
        })
    summary_path = os.path.join(output_dir, "activity_suitable_summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path


def _save_plot(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_activity_charts(activity_df: pd.DataFrame, output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    chart_files: List[str] = []

    score_cols = [s[2] for s in ACTIVITY_SPECS]
    en_labels = [s[1] for s in ACTIVITY_SPECS]
    frame_nums = pd.to_numeric(activity_df["frame_num"], errors="coerce").fillna(0).astype(int)


    plt.figure(figsize=(14, 7))
    for col, label in zip(score_cols, en_labels):
        plt.plot(frame_nums, pd.to_numeric(activity_df[col], errors="coerce"), linewidth=1.5, label=label)
    plt.title("AI Activity Scores Trend")
    plt.xlabel("Frame")
    plt.ylabel("Score")
    plt.ylim(0, 5.2)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(ncol=3, frameon=True)
    trend_png = os.path.join(output_dir, "activity_scores_trend.png")
    _save_plot(trend_png)
    chart_files.append(trend_png)


    heat = activity_df[score_cols].apply(pd.to_numeric, errors="coerce")
    heat.columns = en_labels
    if len(heat) > 400:
        idx = np.linspace(0, len(heat) - 1, 400).astype(int)
        heat = heat.iloc[idx]
    plt.figure(figsize=(14, 5.8))
    sns.heatmap(heat.T, cmap="YlGnBu", vmin=0, vmax=5, cbar_kws={"label": "Score"})
    plt.title("AI Activity Scores Heatmap")
    plt.xlabel("Sampled Frame Index")
    plt.ylabel("Activity")
    heat_png = os.path.join(output_dir, "activity_scores_heatmap.png")
    _save_plot(heat_png)
    chart_files.append(heat_png)


    avg_vals = [float(pd.to_numeric(activity_df[col], errors="coerce").mean()) for col in score_cols]
    angles = np.linspace(0, 2 * np.pi, len(en_labels), endpoint=False).tolist()
    vals = avg_vals + [avg_vals[0]]
    theta = angles + [angles[0]]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(theta, vals, linewidth=2)
    ax.fill(theta, vals, alpha=0.2)
    ax.set_thetagrids(np.degrees(angles), en_labels)
    ax.set_ylim(0, 5)
    ax.set_title("AI Activity Radar (Average)")
    radar_png = os.path.join(output_dir, "activity_scores_radar.png")
    plt.tight_layout()
    plt.savefig(radar_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    chart_files.append(radar_png)


    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(en_labels, avg_vals, color="#3a86ff")
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=10)
    plt.title("AI Activity Average Scores")
    plt.ylabel("Score")
    plt.ylim(0, 5.2)
    plt.grid(axis="y", alpha=0.2, linestyle="--")
    avg_png = os.path.join(output_dir, "activity_avg_scores.png")
    _save_plot(avg_png)
    chart_files.append(avg_png)


    mean_vals = np.array(avg_vals, dtype=float)
    std = float(mean_vals.std())
    z_vals = (mean_vals - mean_vals.mean()) / std if std > 1e-9 else np.zeros_like(mean_vals)
    plt.figure(figsize=(10, 5.5))
    colors = ["#4caf50" if x >= 0 else "#ef5350" for x in z_vals]
    bars = plt.bar(en_labels, z_vals, color=colors)
    for bar in bars:
        y = bar.get_height()
        offset = 0.03 if y >= 0 else -0.06
        plt.text(bar.get_x() + bar.get_width() / 2, y + offset, f"{y:.2f}", ha="center", va="bottom", fontsize=10)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("AI Activity Z-Score (Average)")
    plt.ylabel("Z-Score")
    plt.grid(axis="y", alpha=0.2, linestyle="--")
    z_png = os.path.join(output_dir, "activity_avg_scores_z.png")
    _save_plot(z_png)
    chart_files.append(z_png)


    try:
        import plotly.graph_objects as go

        trend_html = os.path.join(output_dir, "activity_scores_trend_interactive.html")
        fig1 = go.Figure()
        for col, label in zip(score_cols, en_labels):
            fig1.add_trace(go.Scatter(x=frame_nums, y=activity_df[col], mode="lines", name=label))
        fig1.update_layout(
            title="AI Activity Scores Trend",
            xaxis_title="Frame",
            yaxis_title="Score",
            yaxis=dict(range=[0, 5.2]),
            template="plotly_white",
            height=560,
            width=980,
        )
        fig1.write_html(trend_html, include_plotlyjs="cdn")
        chart_files.append(trend_html)

        heat_html = os.path.join(output_dir, "activity_scores_heatmap.html")
        fig2 = go.Figure(data=go.Heatmap(
            z=activity_df[score_cols].T.values,
            x=frame_nums.values,
            y=en_labels,
            colorscale="YlGnBu",
            zmin=0,
            zmax=5,
            colorbar=dict(title="Score"),
        ))
        fig2.update_layout(
            title="AI Activity Scores Heatmap",
            xaxis_title="Frame",
            yaxis_title="Activity",
            template="plotly_white",
            height=560,
            width=980,
        )
        fig2.write_html(heat_html, include_plotlyjs="cdn")
        chart_files.append(heat_html)

        radar_html = os.path.join(output_dir, "activity_scores_radar.html")
        fig3 = go.Figure(data=go.Scatterpolar(
            r=avg_vals + [avg_vals[0]],
            theta=en_labels + [en_labels[0]],
            fill="toself",
            name="Average",
        ))
        fig3.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            title="AI Activity Radar (Average)",
            template="plotly_white",
            height=560,
            width=760,
        )
        fig3.write_html(radar_html, include_plotlyjs="cdn")
        chart_files.append(radar_html)
    except Exception as exc:
        logger.warning("生成活动评估交互图失败: %s", exc)

    return chart_files


def generate_ai_activity_outputs(video_dir: str, output_dir: Optional[str] = None) -> List[str]:
    """
    生成 AI 六项活动评估输出（CSV + 图表）

    优先使用智谱 AI Prompt 评估；失败时自动降级到本地规则评分。
    """
    output_dir = output_dir or os.path.join(video_dir, "ai_evaluation")
    os.makedirs(output_dir, exist_ok=True)

    try:
        activity_df = _evaluate_with_zhipu(video_dir, output_dir)
        if activity_df is None or activity_df.empty:
            activity_df = _evaluate_with_fallback_metrics(video_dir)

        scores_csv = _write_scores_csv(activity_df, output_dir)
        summary_csv = _write_summary_csv(activity_df, output_dir)
        chart_files = generate_activity_charts(activity_df, output_dir)
        return [scores_csv, summary_csv] + chart_files
    except FileNotFoundError as exc:
        logger.warning("跳过 AI 活动评估: %s", exc)
        return []
    except Exception as exc:
        logger.error("AI 活动评估失败: %s", exc, exc_info=True)
        return []
