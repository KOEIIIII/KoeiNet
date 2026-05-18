


"""Build a targeted Step-5.5 adjudication subset from two-rater disagreements."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .validation_finalize import CORE_OUTPUT_COLUMNS, collapse_rater_hidden_duplicates

logger = logging.getLogger("fusion.adjudication_pack")


SCALAR_DISAGREE_FIELDS: List[str] = [
    "safety_score",
    "overall_problem_severity",
    "soundscape_pleasantness",
    "soundscape_eventfulness",
    "comfort_score",
    "vitality_score",
]


@dataclass(frozen=True)
class AdjudicationPackConfig:
    """Config for disagreement subset selection."""

    scalar_diff_threshold: float = 2.0
    low_confidence_threshold: float = 3.0
    intra_duplicate_mae_threshold: float = 1.0


def _to_num(v: Any) -> float:
    s = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    return float(s) if pd.notna(s) else float("nan")


def _clean_text(v: Any) -> str:
    if v is None:
        return ""
    text = str(v).strip()
    if text.lower() == "nan":
        return ""
    return text


def _read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path.as_posix()}")
    return pd.read_csv(path)


def _read_json_required(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path.as_posix()}")
    return json.loads(path.read_text(encoding="utf-8"))


def _first_non_empty(series: pd.Series) -> Any:
    for v in series.tolist():
        text = _clean_text(v)
        if text:
            return v
    return ""


def _build_segment_meta(
    rater_df: pd.DataFrame,
    admin_df: pd.DataFrame,
    rater_id: str,
) -> pd.DataFrame:
    a = admin_df[admin_df["assigned_rater"] == rater_id][["canonical_segment_id", "displayed_item_id"]].copy()
    if a.empty:
        return pd.DataFrame(columns=["segment_id", "start_time_sec", "end_time_sec", "primary_preview_path", "context_strip_path", "audio_clip_path"])

    work = rater_df.copy()
    work["displayed_item_id"] = work["displayed_item_id"].astype(str)
    merged = a.merge(work, on="displayed_item_id", how="left")
    merged["canonical_segment_id"] = pd.to_numeric(merged["canonical_segment_id"], errors="coerce")
    merged = merged.dropna(subset=["canonical_segment_id"]).copy()
    merged["segment_id"] = merged["canonical_segment_id"].astype(int)

    out_rows: List[Dict[str, Any]] = []
    for sid, g in merged.groupby("segment_id", sort=True):
        out_rows.append(
            {
                "segment_id": int(sid),
                "start_time_sec": _to_num(_first_non_empty(g.get("start_time_sec", pd.Series(dtype=object)))),
                "end_time_sec": _to_num(_first_non_empty(g.get("end_time_sec", pd.Series(dtype=object)))),
                "primary_preview_path": _clean_text(
                    _first_non_empty(g.get("primary_preview_path", pd.Series(dtype=object)))
                )
                or _clean_text(_first_non_empty(g.get("preview_path", pd.Series(dtype=object)))),
                "context_strip_path": _clean_text(
                    _first_non_empty(g.get("context_strip_path", pd.Series(dtype=object)))
                ),
                "audio_clip_path": _clean_text(_first_non_empty(g.get("audio_clip_path", pd.Series(dtype=object)))),
            }
        )
    return pd.DataFrame(out_rows)


def _build_duplicate_instability(
    rater_df: pd.DataFrame,
    admin_df: pd.DataFrame,
    rater_id: str,
) -> Dict[int, Dict[str, Any]]:
    """
    Build per-segment duplicate instability summary for one rater.

    Returns dict:
    {
      segment_id: {
        "has_duplicate_pair": bool,
        "label_conflict": bool,
        "field_abs_diff": {field: diff},
      }
    }
    """
    a = admin_df[admin_df["assigned_rater"] == rater_id].copy()
    a = a[a["duplicate_group_id"].astype(str).str.strip() != ""].copy()
    if a.empty:
        return {}

    w = rater_df.copy()
    w["displayed_item_id"] = w["displayed_item_id"].astype(str)
    m = a.merge(w, on="displayed_item_id", how="left")
    m["order_idx"] = m["displayed_item_id"].str.extract(r"_(\d+)$").astype(float)

    out: Dict[int, Dict[str, Any]] = {}
    for _, g in m.groupby("duplicate_group_id"):
        g = g.sort_values("order_idx")
        if len(g) < 2:
            continue
        r1 = g.iloc[0]
        r2 = g.iloc[1]
        sid = int(pd.to_numeric(r1.get("canonical_segment_id"), errors="coerce"))
        rec = out.setdefault(
            sid,
            {
                "has_duplicate_pair": True,
                "label_conflict": False,
                "field_abs_diff": {f: float("nan") for f in SCALAR_DISAGREE_FIELDS},
            },
        )
        for f in SCALAR_DISAGREE_FIELDS:
            v1 = _to_num(r1.get(f))
            v2 = _to_num(r2.get(f))
            if np.isnan(v1) or np.isnan(v2):
                continue
            diff = abs(v1 - v2)
            cur = rec["field_abs_diff"].get(f, float("nan"))
            if np.isnan(cur) or diff > cur:
                rec["field_abs_diff"][f] = float(diff)

        l1 = _clean_text(r1.get("primary_problem_label"))
        l2 = _clean_text(r2.get("primary_problem_label"))
        if l1 != l2 and (l1 or l2):
            rec["label_conflict"] = True
    return out


def _build_adjudication_instructions_markdown() -> str:
    lines = [
        "# Step 5.5 裁决说明",
        "",
        "本表仅包含分歧较大的条目，用于小规模复核，不替代原始 Step 5 全量双评审。",
        "",
        "## 裁决流程",
        "- 先看主评分图，再听音频片段；声景相关评分必须基于音频。",
        "- 对照两位评审原始分数，选择证据最充分的最终值。",
        "- 当前界面默认 1-7 整数评分；如无法确定主问题标签，请使用 `mixed_or_unclear`。",
        "- 裁决备注请记录关键依据和取舍理由。",
        "",
        "## 声景评分要求（强制）",
        "- 请先查看主评分图，再试听音频片段，最后再填写“声景愉悦度”和“声景事件性”。",
        "- 安全性、舒适度、活力度、整体问题严重度可以综合图像与音频理解，但声景相关评分必须参考音频。",
        "",
        "## 输出约定",
        "- 未进入裁决子集的 segment 会保留原 `final_annotation_labels.csv` 结果。",
        "- 裁决只会覆盖已填写的 `adjudicated_*` 字段。",
    ]
    return "\n".join(lines).strip() + "\n"


def _pack_columns() -> List[str]:
    cols: List[str] = [
        "segment_id",
        "start_time_sec",
        "end_time_sec",
        "primary_preview_path",
        "context_strip_path",
        "audio_clip_path",
    ]
    for rid in ("A", "B"):
        for f in SCALAR_DISAGREE_FIELDS:
            cols.append(f"rater_{rid}_{f}")
        cols.extend(
            [
                f"rater_{rid}_primary_problem_label",
                f"rater_{rid}_confidence_score",
                f"rater_{rid}_notes",
            ]
        )
    for f in SCALAR_DISAGREE_FIELDS:
        cols.append(f"abs_diff_{f}")
        cols.append(f"flag_disagree_{f}")
    cols.extend(
        [
            "flag_label_disagreement",
            "flag_selected_by_scalar",
            "flag_selected_by_label",
            "flag_selected_by_intra",
            "flag_intra_rater_scalar_instability",
            "flag_intra_rater_label_conflict",
            "selection_reasons",
            "adjudicated_safety_score",
            "adjudicated_comfort_score",
            "adjudicated_vitality_score",
            "adjudicated_overall_problem_severity",
            "adjudicated_soundscape_pleasantness",
            "adjudicated_soundscape_eventfulness",
            "adjudicated_primary_problem_label",
            "adjudication_notes",
        ]
    )
    return cols


def build_adjudication_pack(
    video_dir: str,
    config: Optional[AdjudicationPackConfig] = None,
    rater_a_csv: Optional[str] = None,
    rater_b_csv: Optional[str] = None,
    admin_manifest_csv: Optional[str] = None,
    reliability_report_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build disagreement-focused adjudication subset and export pack/admin files.

    Outputs:
    - validation/adjudication_pack.csv
    - validation/adjudication_manifest_admin.csv
    - validation/adjudication_instructions.md
    """
    cfg = config or AdjudicationPackConfig()
    vdir = Path(video_dir)
    val_dir = vdir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    a_path = Path(rater_a_csv) if rater_a_csv else val_dir / "rater_A_annotation_pack.csv"
    b_path = Path(rater_b_csv) if rater_b_csv else val_dir / "rater_B_annotation_pack.csv"
    admin_path = Path(admin_manifest_csv) if admin_manifest_csv else val_dir / "sample_manifest_admin.csv"
    rel_path = Path(reliability_report_json) if reliability_report_json else val_dir / "reliability_report.json"

    a_df = _read_csv_required(a_path, "rater A csv")
    b_df = _read_csv_required(b_path, "rater B csv")
    admin_df = _read_csv_required(admin_path, "sample manifest admin csv")
    rel_report = _read_json_required(rel_path, "reliability report json")

    canon_a, _ = collapse_rater_hidden_duplicates(a_df, admin_df, rater_id="A")
    canon_b, _ = collapse_rater_hidden_duplicates(b_df, admin_df, rater_id="B")
    merged = canon_a.merge(canon_b, on="segment_id", how="inner", suffixes=("_A", "_B"))

    meta_a = _build_segment_meta(a_df, admin_df, rater_id="A")
    meta_b = _build_segment_meta(b_df, admin_df, rater_id="B")
    meta = pd.concat([meta_a, meta_b], axis=0, ignore_index=True)
    meta = (
        meta.groupby("segment_id", as_index=False)
        .agg(
            {
                "start_time_sec": "first",
                "end_time_sec": "first",
                "primary_preview_path": _first_non_empty,
                "context_strip_path": _first_non_empty,
                "audio_clip_path": _first_non_empty,
            }
        )
    )

    inst_a = _build_duplicate_instability(a_df, admin_df, rater_id="A")
    inst_b = _build_duplicate_instability(b_df, admin_df, rater_id="B")

    rows: List[Dict[str, Any]] = []
    admin_rows: List[Dict[str, Any]] = []
    scalar_disagreement_segments = 0
    label_disagreement_segments = 0
    intra_instability_segments = 0

    for _, r in merged.iterrows():
        sid = int(r["segment_id"])
        conf_a = _to_num(r.get("confidence_score_A"))
        conf_b = _to_num(r.get("confidence_score_B"))
        scalar_flags: Dict[str, bool] = {}
        abs_diffs: Dict[str, float] = {}
        scalar_reason_fields: List[str] = []

        for f in SCALAR_DISAGREE_FIELDS:
            va = _to_num(r.get(f"{f}_A"))
            vb = _to_num(r.get(f"{f}_B"))
            diff = abs(va - vb) if (not np.isnan(va) and not np.isnan(vb)) else float("nan")
            abs_diffs[f] = float(diff) if not np.isnan(diff) else float("nan")
            ratings_differ = (not np.isnan(va)) and (not np.isnan(vb)) and (abs(va - vb) > 0)
            low_conf_disagree = ratings_differ and (
                ((not np.isnan(conf_a)) and conf_a <= float(cfg.low_confidence_threshold))
                or ((not np.isnan(conf_b)) and conf_b <= float(cfg.low_confidence_threshold))
            )
            flag = ((not np.isnan(diff)) and diff >= float(cfg.scalar_diff_threshold)) or low_conf_disagree
            scalar_flags[f] = bool(flag)
            if flag:
                scalar_reason_fields.append(f)

        label_a = _clean_text(r.get("primary_problem_label_A"))
        label_b = _clean_text(r.get("primary_problem_label_B"))
        label_flag = bool((label_a or label_b) and (label_a != label_b))

        ia = inst_a.get(sid, {})
        ib = inst_b.get(sid, {})
        intra_field_flags: List[str] = []
        for f in SCALAR_DISAGREE_FIELDS:
            da = ia.get("field_abs_diff", {}).get(f, float("nan"))
            db = ib.get("field_abs_diff", {}).get(f, float("nan"))
            flag_f = (
                ((not np.isnan(da)) and da >= float(cfg.intra_duplicate_mae_threshold))
                or ((not np.isnan(db)) and db >= float(cfg.intra_duplicate_mae_threshold))
            )
            if flag_f:
                intra_field_flags.append(f)
        intra_label_conflict = bool(ia.get("label_conflict", False) or ib.get("label_conflict", False))
        has_dup_pair = bool(ia.get("has_duplicate_pair", False) or ib.get("has_duplicate_pair", False))
        intra_flag = bool(has_dup_pair and (len(intra_field_flags) > 0 or intra_label_conflict))

        scalar_any = any(scalar_flags.values())
        selected = bool(scalar_any or label_flag or intra_flag)
        if not selected:
            continue

        if scalar_any:
            scalar_disagreement_segments += 1
        if label_flag:
            label_disagreement_segments += 1
        if intra_flag:
            intra_instability_segments += 1

        mrow = meta[meta["segment_id"] == sid]
        if not mrow.empty:
            m = mrow.iloc[0]
            start_time_sec = _to_num(m.get("start_time_sec"))
            end_time_sec = _to_num(m.get("end_time_sec"))
            primary_preview_path = _clean_text(m.get("primary_preview_path"))
            context_strip_path = _clean_text(m.get("context_strip_path"))
            audio_clip_path = _clean_text(m.get("audio_clip_path"))
        else:
            start_time_sec = float("nan")
            end_time_sec = float("nan")
            primary_preview_path = ""
            context_strip_path = ""
            audio_clip_path = ""

        reason_tokens: List[str] = []
        if scalar_any:
            reason_tokens.append("scalar_disagreement")
        if label_flag:
            reason_tokens.append("label_disagreement")
        if intra_flag:
            reason_tokens.append("intra_rater_instability")

        row: Dict[str, Any] = {
            "segment_id": sid,
            "start_time_sec": start_time_sec,
            "end_time_sec": end_time_sec,
            "primary_preview_path": primary_preview_path,
            "context_strip_path": context_strip_path,
            "audio_clip_path": audio_clip_path,
        }
        for f in SCALAR_DISAGREE_FIELDS:
            row[f"rater_A_{f}"] = _to_num(r.get(f"{f}_A"))
            row[f"rater_B_{f}"] = _to_num(r.get(f"{f}_B"))
        row["rater_A_primary_problem_label"] = label_a
        row["rater_B_primary_problem_label"] = label_b
        row["rater_A_confidence_score"] = conf_a
        row["rater_B_confidence_score"] = conf_b
        row["rater_A_notes"] = _clean_text(r.get("notes_A"))
        row["rater_B_notes"] = _clean_text(r.get("notes_B"))

        for f in SCALAR_DISAGREE_FIELDS:
            row[f"abs_diff_{f}"] = abs_diffs[f]
            row[f"flag_disagree_{f}"] = bool(scalar_flags[f])
        row["flag_label_disagreement"] = bool(label_flag)
        row["flag_selected_by_scalar"] = bool(scalar_any)
        row["flag_selected_by_label"] = bool(label_flag)
        row["flag_selected_by_intra"] = bool(intra_flag)
        row["flag_intra_rater_scalar_instability"] = bool(len(intra_field_flags) > 0)
        row["flag_intra_rater_label_conflict"] = bool(intra_label_conflict)
        row["selection_reasons"] = ",".join(reason_tokens)

        row["adjudicated_safety_score"] = ""
        row["adjudicated_comfort_score"] = ""
        row["adjudicated_vitality_score"] = ""
        row["adjudicated_overall_problem_severity"] = ""
        row["adjudicated_soundscape_pleasantness"] = ""
        row["adjudicated_soundscape_eventfulness"] = ""
        row["adjudicated_primary_problem_label"] = ""
        row["adjudication_notes"] = ""
        rows.append(row)

        admin_rows.append(
            {
                "segment_id": sid,
                "selection_reasons": ",".join(reason_tokens),
                "scalar_disagreement_fields": ",".join(scalar_reason_fields),
                "label_disagreement": bool(label_flag),
                "intra_scalar_unstable_fields": ",".join(intra_field_flags),
                "intra_label_conflict": bool(intra_label_conflict),
                "start_time_sec": start_time_sec,
                "end_time_sec": end_time_sec,
                "primary_preview_path": primary_preview_path,
                "context_strip_path": context_strip_path,
                "audio_clip_path": audio_clip_path,
            }
        )

    pack_path = val_dir / "adjudication_pack.csv"
    admin_out_path = val_dir / "adjudication_manifest_admin.csv"
    inst_path = val_dir / "adjudication_instructions.md"

    pack_df = pd.DataFrame(rows, columns=_pack_columns())
    admin_df_out = pd.DataFrame(admin_rows)

    pack_df.to_csv(pack_path, index=False, encoding="utf-8")
    admin_df_out.to_csv(admin_out_path, index=False, encoding="utf-8")
    inst_path.write_text(_build_adjudication_instructions_markdown(), encoding="utf-8")

    out = {
        "adjudication_pack_csv": pack_path.as_posix(),
        "adjudication_manifest_admin_csv": admin_out_path.as_posix(),
        "adjudication_instructions_md": inst_path.as_posix(),
        "flagged_segments": int(len(pack_df)),
        "segments_with_scalar_disagreement": int(scalar_disagreement_segments),
        "segments_with_label_disagreement": int(label_disagreement_segments),
        "segments_with_intra_rater_instability": int(intra_instability_segments),
        "source_reliability_report_json": rel_path.as_posix(),
        "pre_adjudication_inter_unique_segments": int(
            rel_report.get("protocol", {}).get("inter_rater_unique_segments", len(merged))
        ),
        "downstream_schema_compatible_step7": bool(all(c in CORE_OUTPUT_COLUMNS for c in CORE_OUTPUT_COLUMNS)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(
        "adjudication pack built | flagged=%d scalar=%d label=%d intra=%d",
        out["flagged_segments"],
        out["segments_with_scalar_disagreement"],
        out["segments_with_label_disagreement"],
        out["segments_with_intra_rater_instability"],
    )
    return out
