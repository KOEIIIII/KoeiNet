


"""本地 Step-5 双评审盲评标注应用（Streamlit）。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fusion.annotation_template import (
    CONTROLLED_PRIMARY_PROBLEM_LABELS,
    PRIMARY_PROBLEM_LABEL_ZH,
)

logger = logging.getLogger("validation.web")

SCORE_FIELDS: List[str] = [
    "safety_score",
    "comfort_score",
    "vitality_score",
    "overall_problem_severity",
    "soundscape_pleasantness",
    "soundscape_eventfulness",
    "confidence_score",
]

REQUIRED_COMPLETE_FIELDS: List[str] = SCORE_FIELDS + ["primary_problem_label"]

ANNOTATION_FIELDS: List[str] = [
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

PATH_COLUMNS: List[str] = ["preview_path", "primary_preview_path", "context_strip_path", "audio_clip_path"]
BASE_COLUMNS: List[str] = ["displayed_item_id", "segment_id", "start_time_sec", "end_time_sec"] + PATH_COLUMNS

SCORE_OPTIONS: List[Any] = ["", 1, 2, 3, 4, 5, 6, 7]

FIELD_LABELS_ZH: Dict[str, str] = {
    "safety_score": "安全性评分",
    "comfort_score": "舒适度评分",
    "vitality_score": "活力度评分",
    "overall_problem_severity": "整体问题严重度",
    "soundscape_pleasantness": "声景愉悦度",
    "soundscape_eventfulness": "声景事件性",
    "confidence_score": "评分信心",
    "primary_problem_label": "主要问题标签",
    "notes": "备注",
}

FIELD_HELP_ZH: Dict[str, str] = {
    "safety_score": "1=非常不安全，7=非常安全。",
    "comfort_score": "1=非常不舒适，7=非常舒适。",
    "vitality_score": "1=活力很低，7=活力很高。",
    "overall_problem_severity": "1=问题很轻，7=问题很重。",
    "soundscape_pleasantness": "1=非常不愉悦，7=非常愉悦。",
    "soundscape_eventfulness": "1=非常平静，7=事件性很强。",
    "confidence_score": "1=非常不确定，7=非常确定。",
    "primary_problem_label": "请选择一个最主要问题；CSV 中会保存英文编码。",
    "notes": "记录关键依据、疑点或补充说明。",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--video_dir", default="", help="视频输出目录（output/<video>）")
    parser.add_argument("--rater", default="A", help="评审者（A 或 B）")
    parser.add_argument("--csv_path", default="", help="标注 CSV 的显式路径")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def _format_primary_problem_option(code: Any) -> str:
    text = _normalize_text(code)
    if not text:
        return "请选择"
    return PRIMARY_PROBLEM_LABEL_ZH.get(text, text)


def _normalize_score(v: Any) -> Any:
    if v is None:
        return ""
    text = str(v).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        iv = int(float(text))
    except Exception:
        return ""
    if 1 <= iv <= 7:
        return iv
    return ""


def _normalize_text(v: Any) -> str:
    if v is None:
        return ""
    text = str(v).strip()
    if text.lower() == "nan":
        return ""
    return text


def _normalize_row_value(field: str, value: Any) -> Any:
    if field in SCORE_FIELDS:
        return _normalize_score(value)
    if field == "primary_problem_label":
        return _normalize_text(value)
    if field == "notes":
        return _normalize_text(value)
    return value


def _default_pack_path(video_dir: str, rater: str) -> str:
    rid = str(rater).strip().upper()
    return os.path.join(video_dir, "validation", f"rater_{rid}_annotation_pack.csv")


def _resolve_media_path(raw: Any, csv_path: str) -> Optional[str]:
    txt = _normalize_text(raw)
    if not txt:
        return None
    p = Path(txt)
    if p.is_file():
        return p.as_posix()
    if not p.is_absolute():
        candidate = (Path(csv_path).parent / p).resolve()
        if candidate.is_file():
            return candidate.as_posix()
        candidate2 = (REPO_ROOT / p).resolve()
        if candidate2.is_file():
            return candidate2.as_posix()
    return None


def _load_annotation_pack(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"未找到标注包 CSV：{p.as_posix()}")
    df = pd.read_csv(p)
    for col in BASE_COLUMNS + ANNOTATION_FIELDS:
        if col not in df.columns:
            df[col] = ""
    df["displayed_item_id"] = df["displayed_item_id"].map(lambda x: str(x).strip())
    for col in SCORE_FIELDS:
        df[col] = df[col].map(_normalize_score)
    df["primary_problem_label"] = df["primary_problem_label"].map(_normalize_text)
    df["notes"] = df["notes"].map(_normalize_text)
    return df


def _safe_write_csv(df: pd.DataFrame, target_csv: str) -> None:
    target = Path(target_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f"{target.stem}_", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    try:
        df.to_csv(temp_path, index=False, encoding="utf-8")
        os.replace(temp_path, target)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _is_row_completed(row: pd.Series) -> bool:
    for field in REQUIRED_COMPLETE_FIELDS:
        v = _normalize_row_value(field, row.get(field))
        if field in SCORE_FIELDS:
            if v == "":
                return False
        else:
            if not str(v).strip():
                return False
    return True


def _collect_widget_values() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for field in SCORE_FIELDS:
        out[field] = _normalize_score(st.session_state.get(f"w_{field}", ""))
    out["primary_problem_label"] = _normalize_text(st.session_state.get("w_primary_problem_label", ""))
    out["notes"] = st.session_state.get("w_notes", "")
    return out


def _row_saved_values(row: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for field in SCORE_FIELDS:
        out[field] = _normalize_score(row.get(field))
    out["primary_problem_label"] = _normalize_text(row.get("primary_problem_label"))
    out["notes"] = _normalize_text(row.get("notes"))
    return out


def _sync_widgets_from_row(row: pd.Series) -> None:
    row_key = str(row.get("displayed_item_id", ""))
    current_key = st.session_state.get("w_loaded_row_id")
    if current_key == row_key:
        return
    for field in SCORE_FIELDS:
        st.session_state[f"w_{field}"] = _normalize_score(row.get(field))
    st.session_state["w_primary_problem_label"] = _normalize_text(row.get("primary_problem_label"))
    st.session_state["w_notes"] = _normalize_text(row.get("notes"))
    st.session_state["w_loaded_row_id"] = row_key


def _save_current_row(df: pd.DataFrame, csv_path: str, row_idx: int) -> bool:
    vals = _collect_widget_values()
    for field, value in vals.items():
        df.at[row_idx, field] = value
    _safe_write_csv(df, csv_path)
    st.session_state["annotation_df"] = df
    st.session_state["last_save_message"] = (
        f"已保存 {df.at[row_idx, 'displayed_item_id']} -> {Path(csv_path).as_posix()}"
    )
    return True


def _move_to_neighbor(delta: int, visible_indices: List[int], current_row_idx: int) -> int:
    if not visible_indices:
        return current_row_idx
    if current_row_idx not in visible_indices:
        return visible_indices[0]
    pos = visible_indices.index(current_row_idx)
    next_pos = max(0, min(len(visible_indices) - 1, pos + delta))
    return visible_indices[next_pos]


def _inject_beforeunload_warning(enabled: bool) -> None:
    if enabled:
        components.html(
            """
            <script>
            window.onbeforeunload = function () {
              return "当前有未保存的标注修改。";
            };
            </script>
            """,
            height=0,
        )
    else:
        components.html(
            """
            <script>
            window.onbeforeunload = null;
            </script>
            """,
            height=0,
        )


def main() -> None:
    args = _parse_args()
    rater = str(args.rater or "A").strip().upper()
    if rater not in {"A", "B"}:
        rater = "A"

    st.set_page_config(page_title=f"第5步盲评标注 - 评审 {rater}", layout="wide")
    st.title(f"第5步双评审盲评标注 - 评审 {rater}")

    initial_csv_path = args.csv_path
    if not initial_csv_path and args.video_dir:
        initial_csv_path = _default_pack_path(args.video_dir, rater=rater)
    if not initial_csv_path:
        initial_csv_path = "output/<video>/validation/rater_A_annotation_pack.csv"

    st.sidebar.header("标注会话")
    csv_path = st.sidebar.text_input("标注 CSV 路径", value=st.session_state.get("csv_path", initial_csv_path))
    st.session_state["csv_path"] = csv_path
    show_incomplete_only = st.sidebar.toggle(
        "仅显示未完成条目",
        value=bool(st.session_state.get("show_incomplete_only", False)),
        key="show_incomplete_only",
    )

    reload_clicked = st.sidebar.button("重新加载 CSV")
    if reload_clicked or st.session_state.get("loaded_csv_path") != csv_path:
        try:
            df = _load_annotation_pack(csv_path)
            st.session_state["annotation_df"] = df
            st.session_state["loaded_csv_path"] = csv_path
            st.session_state["current_row_idx"] = int(df.index[0]) if len(df) else 0
            st.session_state["w_loaded_row_id"] = ""
            st.session_state["last_save_message"] = ""
        except Exception as exc:
            st.error(f"加载标注包失败：{exc}")
            return

    if "annotation_df" not in st.session_state:
        st.warning("请先加载有效的评审标注 CSV。")
        return

    df: pd.DataFrame = st.session_state["annotation_df"]
    if df.empty:
        st.warning("标注 CSV 为空。")
        return

    completed_count = int(sum(_is_row_completed(df.iloc[i]) for i in range(len(df))))
    total_count = int(len(df))
    incomplete_count = total_count - completed_count

    st.sidebar.markdown("### 进度汇总")
    st.sidebar.write(f"总条目数：{total_count}")
    st.sidebar.write(f"已完成：{completed_count}")
    st.sidebar.write(f"未完成：{incomplete_count}")

    visible_indices: List[int]
    if show_incomplete_only:
        visible_indices = [int(i) for i in df.index if not _is_row_completed(df.loc[i])]
    else:
        visible_indices = [int(i) for i in df.index]

    if not visible_indices:
        st.success("当前筛选范围内已全部完成。")
        return

    displayed_ids = [str(df.at[i, "displayed_item_id"]) for i in df.index]
    jump_id = st.sidebar.selectbox("跳转到条目编号", options=[""] + displayed_ids, index=0)
    if st.sidebar.button("执行跳转"):
        if jump_id:
            idx_match = df.index[df["displayed_item_id"].astype(str) == str(jump_id)]
            if len(idx_match):
                target_idx = int(idx_match[0])
                st.session_state["current_row_idx"] = target_idx
                if show_incomplete_only and target_idx not in visible_indices:
                    st.session_state["show_incomplete_only"] = False
                st.session_state["w_loaded_row_id"] = ""
                st.rerun()

    row_idx = int(st.session_state.get("current_row_idx", visible_indices[0]))
    if row_idx not in visible_indices:
        row_idx = visible_indices[0]
        st.session_state["current_row_idx"] = row_idx
    row = df.loc[row_idx]
    _sync_widgets_from_row(row)

    saved_values = _row_saved_values(row)
    current_values = _collect_widget_values()
    dirty = current_values != saved_values
    _inject_beforeunload_warning(enabled=dirty)

    head_cols = st.columns([2, 2, 2])
    with head_cols[0]:
        st.markdown(f"**条目编号**：`{row.get('displayed_item_id', '')}`")
    with head_cols[1]:
        st.markdown(f"**时间范围**：`{row.get('start_time_sec', '')} 秒` - `{row.get('end_time_sec', '')} 秒`")
    with head_cols[2]:
        pos = visible_indices.index(row_idx) + 1
        st.markdown(f"**当前位置**：`{pos}/{len(visible_indices)}`")

    st.info(
        "盲评提醒：请独立评分，不讨论结果；不要推测隐藏重复项。"
        "可随时保存并中断，后续可继续从同一 CSV 恢复。"
    )
    st.info(
        "请先查看主评分图，再试听音频片段，最后再填写“声景愉悦度”和“声景事件性”。"
        "安全性、舒适度、活力度、整体问题严重度可以综合图像与音频理解，"
        "但声景相关评分必须参考音频。"
    )
    with st.expander("评分指南（中文）", expanded=False):
        st.markdown(
            "- 量表统一使用 **1-7 分整数**：1=非常低/非常差，4=中性，7=非常高/非常好。  \n"
            "- **安全性评分**：关注行人风险与安全感。  \n"
            "- **舒适度评分**：关注步行与停留舒适感。  \n"
            "- **活力度评分**：关注街道活跃度与生命感。  \n"
            "- **整体问题严重度**：综合判断主要问题强弱。  \n"
            "- **声景愉悦度/事件性**：分别评价声音环境好坏与事件丰富度。  \n"
            "- **评分信心**：反映你对本条判断把握程度。  \n"
            "- 请先看主评分图，再试听音频；辅助时间上下文图仅做补充参考。"
        )

    if st.session_state.get("last_save_message"):
        st.success(st.session_state["last_save_message"])
        st.session_state["last_save_message"] = ""

    if dirty:
        st.warning("检测到未保存修改。离开前请先保存，或使用“上一条/下一条（自动保存）”。")

    primary_path = _resolve_media_path(row.get("primary_preview_path") or row.get("preview_path"), csv_path=csv_path)
    context_path = _resolve_media_path(row.get("context_strip_path"), csv_path=csv_path)
    audio_path = _resolve_media_path(row.get("audio_clip_path"), csv_path=csv_path)

    if primary_path:
        st.image(primary_path, caption="主评分图（主要评分依据）", use_container_width=True)
    else:
        st.warning("未找到该条目的主评分图。")

    if context_path:
        st.markdown("**辅助时间上下文（仅辅助，不作为主评分目标）**")
        st.image(context_path, caption="辅助时间上下文图", use_container_width=True)

    st.markdown("### 音频片段（请先试听后再评价声景）")
    audio_available = bool(audio_path)
    if audio_available:
        st.audio(audio_path, format="audio/wav")
        st.checkbox(
            "我已试听当前音频",
            key=f"w_audio_listened_{row.get('displayed_item_id', '')}",
            value=bool(st.session_state.get(f"w_audio_listened_{row.get('displayed_item_id', '')}", False)),
        )
    else:
        st.warning("当前条目缺少音频，暂不建议填写声景评分。")

    st.markdown("### 评分表单（1-7 分）")
    form_cols_a = st.columns(3)
    with form_cols_a[0]:
        st.selectbox(
            FIELD_LABELS_ZH["safety_score"],
            options=SCORE_OPTIONS,
            key="w_safety_score",
            help=FIELD_HELP_ZH["safety_score"],
        )
        st.selectbox(
            FIELD_LABELS_ZH["comfort_score"],
            options=SCORE_OPTIONS,
            key="w_comfort_score",
            help=FIELD_HELP_ZH["comfort_score"],
        )
        st.selectbox(
            FIELD_LABELS_ZH["vitality_score"],
            options=SCORE_OPTIONS,
            key="w_vitality_score",
            help=FIELD_HELP_ZH["vitality_score"],
        )
    with form_cols_a[1]:
        st.selectbox(
            FIELD_LABELS_ZH["overall_problem_severity"],
            options=SCORE_OPTIONS,
            key="w_overall_problem_severity",
            help=FIELD_HELP_ZH["overall_problem_severity"],
        )
        st.selectbox(
            FIELD_LABELS_ZH["soundscape_pleasantness"],
            options=SCORE_OPTIONS,
            key="w_soundscape_pleasantness",
            help=FIELD_HELP_ZH["soundscape_pleasantness"],
        )
        st.selectbox(
            FIELD_LABELS_ZH["soundscape_eventfulness"],
            options=SCORE_OPTIONS,
            key="w_soundscape_eventfulness",
            help=FIELD_HELP_ZH["soundscape_eventfulness"],
        )
    with form_cols_a[2]:
        st.selectbox(
            FIELD_LABELS_ZH["primary_problem_label"],
            options=[""] + CONTROLLED_PRIMARY_PROBLEM_LABELS,
            key="w_primary_problem_label",
            format_func=_format_primary_problem_option,
            help=FIELD_HELP_ZH["primary_problem_label"],
        )
        st.selectbox(
            FIELD_LABELS_ZH["confidence_score"],
            options=SCORE_OPTIONS,
            key="w_confidence_score",
            help=FIELD_HELP_ZH["confidence_score"],
        )

    st.text_area(FIELD_LABELS_ZH["notes"], key="w_notes", height=140, help=FIELD_HELP_ZH["notes"])

    soundscape_filled = (
        _normalize_score(st.session_state.get("w_soundscape_pleasantness", "")) != ""
        or _normalize_score(st.session_state.get("w_soundscape_eventfulness", "")) != ""
    )
    if (not audio_available) and soundscape_filled:
        st.warning("当前无音频但已填写声景评分，请谨慎核对。")

    nav_cols = st.columns([1, 1, 1, 1])
    with nav_cols[0]:
        if st.button("保存/更新", use_container_width=True):
            try:
                _save_current_row(df, csv_path=csv_path, row_idx=row_idx)
                st.session_state["w_loaded_row_id"] = ""
                st.rerun()
            except Exception as exc:
                st.error(f"保存失败：{exc}")
    with nav_cols[1]:
        if st.button("上一条（自动保存）", use_container_width=True):
            try:
                _save_current_row(df, csv_path=csv_path, row_idx=row_idx)
                st.session_state["current_row_idx"] = _move_to_neighbor(-1, visible_indices, row_idx)
                st.session_state["w_loaded_row_id"] = ""
                st.rerun()
            except Exception as exc:
                st.error(f"自动保存失败：{exc}")
    with nav_cols[2]:
        if st.button("下一条（自动保存）", use_container_width=True):
            try:
                _save_current_row(df, csv_path=csv_path, row_idx=row_idx)
                st.session_state["current_row_idx"] = _move_to_neighbor(1, visible_indices, row_idx)
                st.session_state["w_loaded_row_id"] = ""
                st.rerun()
            except Exception as exc:
                st.error(f"自动保存失败：{exc}")
    with nav_cols[3]:
        if st.button("保存并停留（自动保存）", use_container_width=True):
            try:
                _save_current_row(df, csv_path=csv_path, row_idx=row_idx)
                st.session_state["w_loaded_row_id"] = ""
                st.rerun()
            except Exception as exc:
                st.error(f"自动保存失败：{exc}")

    st.caption(
        "盲评协议：本页面不会显示重复标记、管理员元数据或模型结果。"
        "请先试听音频再填写声景评分；可使用 Tab + Enter 提升键盘操作效率。"
    )


if __name__ == "__main__":
    main()
