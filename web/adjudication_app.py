


"""Step-5.5 disagreement adjudication web app (Streamlit)."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fusion.annotation_template import CONTROLLED_PRIMARY_PROBLEM_LABELS, PRIMARY_PROBLEM_LABEL_ZH

SCALAR_FIELDS: List[str] = [
    "safety_score",
    "comfort_score",
    "vitality_score",
    "overall_problem_severity",
    "soundscape_pleasantness",
    "soundscape_eventfulness",
]

ADJ_COLS: List[str] = [
    "adjudicated_safety_score",
    "adjudicated_comfort_score",
    "adjudicated_vitality_score",
    "adjudicated_overall_problem_severity",
    "adjudicated_soundscape_pleasantness",
    "adjudicated_soundscape_eventfulness",
    "adjudicated_primary_problem_label",
    "adjudication_notes",
]

REQUIRED_COMPLETE: List[str] = ADJ_COLS[:-1]

FIELD_LABEL_ZH: Dict[str, str] = {
    "safety_score": "安全性评分",
    "comfort_score": "舒适度评分",
    "vitality_score": "活力度评分",
    "overall_problem_severity": "整体问题严重度",
    "soundscape_pleasantness": "声景愉悦度",
    "soundscape_eventfulness": "声景事件性",
}

SCORE_OPTIONS: List[Any] = ["", 1, 2, 3, 4, 5, 6, 7]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--video_dir", default="", help="视频输出目录（output/<video>）")
    parser.add_argument("--csv_path", default="", help="裁决包 CSV 路径")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


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


def _format_problem_label(code: Any) -> str:
    c = _normalize_text(code)
    if not c:
        return "请选择"
    return PRIMARY_PROBLEM_LABEL_ZH.get(c, c)


def _resolve_media_path(raw: Any, csv_path: str) -> Optional[str]:
    txt = _normalize_text(raw)
    if not txt:
        return None
    p = Path(txt)
    if p.is_file():
        return p.as_posix()
    if not p.is_absolute():
        c1 = (Path(csv_path).parent / p).resolve()
        if c1.is_file():
            return c1.as_posix()
        c2 = (REPO_ROOT / p).resolve()
        if c2.is_file():
            return c2.as_posix()
    return None


def _load_pack(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"未找到裁决包 CSV：{p.as_posix()}")
    df = pd.read_csv(p)
    needed = [
        "segment_id",
        "start_time_sec",
        "end_time_sec",
        "primary_preview_path",
        "context_strip_path",
        "audio_clip_path",
        "selection_reasons",
    ]
    for rid in ("A", "B"):
        for f in SCALAR_FIELDS:
            needed.append(f"rater_{rid}_{f}")
        needed.extend([f"rater_{rid}_primary_problem_label", f"rater_{rid}_confidence_score"])
    for f in SCALAR_FIELDS:
        needed.extend([f"abs_diff_{f}", f"flag_disagree_{f}"])
    needed.extend(ADJ_COLS)
    for c in needed:
        if c not in df.columns:
            df[c] = ""
    df["segment_id"] = pd.to_numeric(df["segment_id"], errors="coerce")
    df = df.dropna(subset=["segment_id"]).copy()
    df["segment_id"] = df["segment_id"].astype(int)
    for c in ADJ_COLS[:-2]:
        df[c] = df[c].map(_normalize_score)
    df["adjudicated_primary_problem_label"] = df["adjudicated_primary_problem_label"].map(_normalize_text)
    df["adjudication_notes"] = df["adjudication_notes"].map(_normalize_text)
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


def _is_completed(row: pd.Series) -> bool:
    for c in REQUIRED_COMPLETE:
        if c == "adjudicated_primary_problem_label":
            if not _normalize_text(row.get(c)):
                return False
        else:
            if _normalize_score(row.get(c)) == "":
                return False
    return True


def _sync_widgets_from_row(row: pd.Series) -> None:
    row_key = str(row.get("segment_id"))
    if st.session_state.get("adj_loaded_row_key") == row_key:
        return
    st.session_state["adj_safety"] = _normalize_score(row.get("adjudicated_safety_score"))
    st.session_state["adj_comfort"] = _normalize_score(row.get("adjudicated_comfort_score"))
    st.session_state["adj_vitality"] = _normalize_score(row.get("adjudicated_vitality_score"))
    st.session_state["adj_overall"] = _normalize_score(row.get("adjudicated_overall_problem_severity"))
    st.session_state["adj_soundscape_pleasant"] = _normalize_score(row.get("adjudicated_soundscape_pleasantness"))
    st.session_state["adj_soundscape_eventful"] = _normalize_score(row.get("adjudicated_soundscape_eventfulness"))
    st.session_state["adj_label"] = _normalize_text(row.get("adjudicated_primary_problem_label"))
    st.session_state["adj_notes"] = _normalize_text(row.get("adjudication_notes"))
    st.session_state["adj_loaded_row_key"] = row_key


def _collect_widget_values() -> Dict[str, Any]:
    return {
        "adjudicated_safety_score": _normalize_score(st.session_state.get("adj_safety")),
        "adjudicated_comfort_score": _normalize_score(st.session_state.get("adj_comfort")),
        "adjudicated_vitality_score": _normalize_score(st.session_state.get("adj_vitality")),
        "adjudicated_overall_problem_severity": _normalize_score(st.session_state.get("adj_overall")),
        "adjudicated_soundscape_pleasantness": _normalize_score(st.session_state.get("adj_soundscape_pleasant")),
        "adjudicated_soundscape_eventfulness": _normalize_score(st.session_state.get("adj_soundscape_eventful")),
        "adjudicated_primary_problem_label": _normalize_text(st.session_state.get("adj_label")),
        "adjudication_notes": _normalize_text(st.session_state.get("adj_notes")),
    }


def _save_current_row(df: pd.DataFrame, csv_path: str, row_idx: int) -> None:
    vals = _collect_widget_values()
    for k, v in vals.items():
        df.at[row_idx, k] = v
    _safe_write_csv(df, csv_path)
    st.session_state["adj_df"] = df
    st.session_state["adj_last_save"] = f"已保存 segment_id={df.at[row_idx, 'segment_id']}"


def _move_to_neighbor(delta: int, visible: List[int], current: int) -> int:
    if not visible:
        return current
    if current not in visible:
        return visible[0]
    pos = visible.index(current)
    nxt = max(0, min(len(visible) - 1, pos + delta))
    return visible[nxt]


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title="第5.5步争议裁决", layout="wide")
    st.title("第5.5步 争议条目裁决")

    init_csv = args.csv_path
    if not init_csv and args.video_dir:
        init_csv = os.path.join(args.video_dir, "validation", "adjudication_pack.csv")
    if not init_csv:
        init_csv = "output/<video>/validation/adjudication_pack.csv"

    st.sidebar.header("裁决会话")
    csv_path = st.sidebar.text_input("裁决包 CSV 路径", value=st.session_state.get("adj_csv_path", init_csv))
    st.session_state["adj_csv_path"] = csv_path
    show_incomplete_only = st.sidebar.toggle(
        "仅显示未完成条目",
        value=bool(st.session_state.get("adj_show_incomplete_only", False)),
        key="adj_show_incomplete_only",
    )
    reload_clicked = st.sidebar.button("重新加载 CSV")

    if reload_clicked or st.session_state.get("adj_loaded_csv_path") != csv_path:
        try:
            df = _load_pack(csv_path)
            st.session_state["adj_df"] = df
            st.session_state["adj_loaded_csv_path"] = csv_path
            st.session_state["adj_row_idx"] = int(df.index[0]) if len(df) else 0
            st.session_state["adj_loaded_row_key"] = ""
            st.session_state["adj_last_save"] = ""
        except Exception as exc:
            st.error(f"加载裁决包失败：{exc}")
            return

    if "adj_df" not in st.session_state:
        st.warning("请先加载裁决包 CSV。")
        return

    df: pd.DataFrame = st.session_state["adj_df"]
    if df.empty:
        st.success("当前无需裁决的争议条目。")
        return

    completed = int(sum(_is_completed(df.iloc[i]) for i in range(len(df))))
    total = int(len(df))
    incomplete = total - completed
    st.sidebar.markdown("### 进度")
    st.sidebar.write(f"总条目：{total}")
    st.sidebar.write(f"已裁决：{completed}")
    st.sidebar.write(f"待裁决：{incomplete}")

    visible: List[int]
    if show_incomplete_only:
        visible = [int(i) for i in df.index if not _is_completed(df.loc[i])]
    else:
        visible = [int(i) for i in df.index]
    if not visible:
        st.success("当前筛选范围内已全部完成。")
        return

    seg_ids = [int(df.at[i, "segment_id"]) for i in df.index]
    jump_sid = st.sidebar.selectbox("跳转到 segment_id", options=[""] + seg_ids, index=0)
    if st.sidebar.button("执行跳转") and jump_sid != "":
        idx_match = df.index[df["segment_id"].astype(int) == int(jump_sid)]
        if len(idx_match):
            target = int(idx_match[0])
            st.session_state["adj_row_idx"] = target
            if show_incomplete_only and target not in visible:
                st.session_state["adj_show_incomplete_only"] = False
            st.session_state["adj_loaded_row_key"] = ""
            st.rerun()

    row_idx = int(st.session_state.get("adj_row_idx", visible[0]))
    if row_idx not in visible:
        row_idx = visible[0]
        st.session_state["adj_row_idx"] = row_idx
    row = df.loc[row_idx]
    _sync_widgets_from_row(row)

    head = st.columns([2, 2, 2])
    with head[0]:
        st.markdown(f"**segment_id**：`{int(row.get('segment_id'))}`")
    with head[1]:
        st.markdown(f"**时间范围**：`{row.get('start_time_sec', '')} 秒` - `{row.get('end_time_sec', '')} 秒`")
    with head[2]:
        pos = visible.index(row_idx) + 1
        st.markdown(f"**当前位置**：`{pos}/{len(visible)}`")

    st.info(
        "裁决规则：先看主图，再听音频。声景相关字段（愉悦度/事件性）必须以音频为依据。"
    )
    reasons = _normalize_text(row.get("selection_reasons"))
    if reasons:
        st.markdown(f"**进入裁决原因**：`{reasons}`")

    if st.session_state.get("adj_last_save"):
        st.success(st.session_state["adj_last_save"])
        st.session_state["adj_last_save"] = ""

    primary_path = _resolve_media_path(row.get("primary_preview_path"), csv_path)
    context_path = _resolve_media_path(row.get("context_strip_path"), csv_path)
    audio_path = _resolve_media_path(row.get("audio_clip_path"), csv_path)

    if primary_path:
        st.image(primary_path, caption="主评分图", use_container_width=True)
    else:
        st.warning("未找到主评分图。")
    if context_path:
        st.image(context_path, caption="辅助时间上下文", use_container_width=True)

    st.markdown("### 音频片段（裁决声景字段前必须试听）")
    if audio_path:
        st.audio(audio_path, format="audio/wav")
        st.checkbox(
            "我已试听当前音频",
            key=f"adj_audio_listened_{int(row.get('segment_id'))}",
            value=bool(st.session_state.get(f"adj_audio_listened_{int(row.get('segment_id'))}", False)),
        )
    else:
        st.warning("当前条目缺少音频，暂不建议填写声景评分。")

    st.markdown("### 两位评审原始分数对照")
    compare_rows: List[Dict[str, Any]] = []
    for f in SCALAR_FIELDS:
        compare_rows.append(
            {
                "字段": FIELD_LABEL_ZH[f],
                "评审A": row.get(f"rater_A_{f}"),
                "评审B": row.get(f"rater_B_{f}"),
                "绝对差": row.get(f"abs_diff_{f}"),
            }
        )
    compare_rows.append(
        {
            "字段": "主要问题标签",
            "评审A": _format_problem_label(row.get("rater_A_primary_problem_label")),
            "评审B": _format_problem_label(row.get("rater_B_primary_problem_label")),
            "绝对差": "-",
        }
    )
    compare_rows.append(
        {
            "字段": "评分信心",
            "评审A": row.get("rater_A_confidence_score"),
            "评审B": row.get("rater_B_confidence_score"),
            "绝对差": "-",
        }
    )
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

    st.markdown("### 裁决填写")
    cols = st.columns(3)
    with cols[0]:
        st.selectbox("裁决-安全性评分", options=SCORE_OPTIONS, key="adj_safety")
        st.selectbox("裁决-舒适度评分", options=SCORE_OPTIONS, key="adj_comfort")
        st.selectbox("裁决-活力度评分", options=SCORE_OPTIONS, key="adj_vitality")
    with cols[1]:
        st.selectbox("裁决-整体问题严重度", options=SCORE_OPTIONS, key="adj_overall")
        st.selectbox("裁决-声景愉悦度", options=SCORE_OPTIONS, key="adj_soundscape_pleasant")
        st.selectbox("裁决-声景事件性", options=SCORE_OPTIONS, key="adj_soundscape_eventful")
    with cols[2]:
        st.selectbox(
            "裁决-主要问题标签",
            options=[""] + CONTROLLED_PRIMARY_PROBLEM_LABELS,
            key="adj_label",
            format_func=_format_problem_label,
        )
    st.text_area("裁决备注", key="adj_notes", height=120)

    soundscape_filled = (
        _normalize_score(st.session_state.get("adj_soundscape_pleasant", "")) != ""
        or _normalize_score(st.session_state.get("adj_soundscape_eventful", "")) != ""
    )
    if (not audio_path) and soundscape_filled:
        st.warning("当前无音频但已填写声景裁决，请谨慎核对。")

    nav = st.columns([1, 1, 1, 1])
    with nav[0]:
        if st.button("保存/更新", use_container_width=True):
            try:
                _save_current_row(df, csv_path, row_idx)
                st.session_state["adj_loaded_row_key"] = ""
                st.rerun()
            except Exception as exc:
                st.error(f"保存失败：{exc}")
    with nav[1]:
        if st.button("上一条（自动保存）", use_container_width=True):
            try:
                _save_current_row(df, csv_path, row_idx)
                st.session_state["adj_row_idx"] = _move_to_neighbor(-1, visible, row_idx)
                st.session_state["adj_loaded_row_key"] = ""
                st.rerun()
            except Exception as exc:
                st.error(f"自动保存失败：{exc}")
    with nav[2]:
        if st.button("下一条（自动保存）", use_container_width=True):
            try:
                _save_current_row(df, csv_path, row_idx)
                st.session_state["adj_row_idx"] = _move_to_neighbor(1, visible, row_idx)
                st.session_state["adj_loaded_row_key"] = ""
                st.rerun()
            except Exception as exc:
                st.error(f"自动保存失败：{exc}")
    with nav[3]:
        if st.button("保存并停留（自动保存）", use_container_width=True):
            try:
                _save_current_row(df, csv_path, row_idx)
                st.session_state["adj_loaded_row_key"] = ""
                st.rerun()
            except Exception as exc:
                st.error(f"自动保存失败：{exc}")

    st.caption("本页面仅用于争议条目裁决；不会改动原始 Step 5 可靠性报告文件。")


if __name__ == "__main__":
    main()

