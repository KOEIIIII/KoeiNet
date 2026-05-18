


"""Finalize Step-5.5 adjudication outputs into a Step-7-compatible label table."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .validation_finalize import CORE_OUTPUT_COLUMNS, collapse_rater_hidden_duplicates

logger = logging.getLogger("fusion.adjudication_finalize")


ADJUDICATED_TO_CORE: Dict[str, str] = {
    "adjudicated_safety_score": "safety_score",
    "adjudicated_comfort_score": "comfort_score",
    "adjudicated_vitality_score": "vitality_score",
    "adjudicated_overall_problem_severity": "overall_problem_severity",
    "adjudicated_soundscape_pleasantness": "soundscape_pleasantness",
    "adjudicated_soundscape_eventfulness": "soundscape_eventfulness",
    "adjudicated_primary_problem_label": "primary_problem_label",
}


def _clean_text(v: Any) -> str:
    if v is None:
        return ""
    text = str(v).strip()
    if text.lower() == "nan":
        return ""
    return text


def _to_num(v: Any) -> float:
    s = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    return float(s) if pd.notna(s) else float("nan")


def _read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path.as_posix()}")
    return pd.read_csv(path)


def _raw_agreement(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    if len(labels_a) == 0:
        return float("nan")
    return float(np.mean([str(x) == str(y) for x, y in zip(labels_a, labels_b)]))


def _pabak(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    po = _raw_agreement(labels_a, labels_b)
    if np.isnan(po):
        return float("nan")
    return float(2.0 * po - 1.0)


def _gwet_ac1(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """
    Approximate Gwet's AC1 for nominal labels with two raters.
    """
    if len(labels_a) == 0:
        return float("nan")
    a = [str(x) for x in labels_a]
    b = [str(x) for x in labels_b]
    cats = sorted(set(a) | set(b))
    k = len(cats)
    if k <= 1:
        return float("nan")
    n = len(a)
    po = _raw_agreement(a, b)
    p: Dict[str, float] = {}
    for c in cats:
        p[c] = float((sum(1 for x in a if x == c) + sum(1 for y in b if y == c)) / (2.0 * n))
    pe = float(sum(pi * (1.0 - pi) for pi in p.values()) / (k - 1))
    if pe >= 1.0:
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def _supplementary_label_agreement(
    label_a: pd.Series,
    label_b: pd.Series,
) -> Dict[str, Any]:
    a = label_a.fillna("").astype(str).str.strip()
    b = label_b.fillna("").astype(str).str.strip()
    valid = (a != "") & (b != "")
    aa = a[valid].tolist()
    bb = b[valid].tolist()
    return {
        "n": int(len(aa)),
        "raw_agreement": _raw_agreement(aa, bb),
        "pabak": _pabak(aa, bb),
        "gwet_ac1": _gwet_ac1(aa, bb),
    }


def _to_bool_series(series: pd.Series) -> pd.Series:
    txt = series.fillna("").astype(str).str.strip().str.lower()
    return txt.isin({"1", "true", "yes", "y", "t"})


def finalize_adjudicated_labels(
    video_dir: str,
    adjudication_pack_csv: Optional[str] = None,
    baseline_final_csv: Optional[str] = None,
    rater_a_csv: Optional[str] = None,
    rater_b_csv: Optional[str] = None,
    admin_manifest_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Apply Step-5.5 adjudicated values onto Step-5 final labels.

    Outputs:
    - validation/final_annotation_labels_adjudicated.csv
    - validation/adjudication_report.json
    """
    vdir = Path(video_dir)
    val_dir = vdir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    pack_path = Path(adjudication_pack_csv) if adjudication_pack_csv else val_dir / "adjudication_pack.csv"
    base_path = Path(baseline_final_csv) if baseline_final_csv else val_dir / "final_annotation_labels.csv"
    a_path = Path(rater_a_csv) if rater_a_csv else val_dir / "rater_A_annotation_pack.csv"
    b_path = Path(rater_b_csv) if rater_b_csv else val_dir / "rater_B_annotation_pack.csv"
    admin_path = Path(admin_manifest_csv) if admin_manifest_csv else val_dir / "sample_manifest_admin.csv"

    pack_df = _read_csv_required(pack_path, "adjudication pack csv")
    base_df = _read_csv_required(base_path, "final annotation labels csv")
    for col in CORE_OUTPUT_COLUMNS:
        if col not in base_df.columns:
            raise ValueError(f"baseline final labels missing required column: {col}")

    out_df = base_df.copy()
    out_df["segment_id"] = pd.to_numeric(out_df["segment_id"], errors="coerce")
    out_df = out_df.dropna(subset=["segment_id"]).copy()
    out_df["segment_id"] = out_df["segment_id"].astype(int)
    out_df = out_df.sort_values("segment_id").reset_index(drop=True)
    out_df_idx = out_df.set_index("segment_id", drop=False)

    pack_work = pack_df.copy()
    pack_work["segment_id"] = pd.to_numeric(pack_work["segment_id"], errors="coerce")
    pack_work = pack_work.dropna(subset=["segment_id"]).copy()
    pack_work["segment_id"] = pack_work["segment_id"].astype(int)

    overwrites_by_field: Dict[str, int] = {core: 0 for core in ADJUDICATED_TO_CORE.values()}
    overwrites_by_field["notes"] = 0
    overwritten_segments = 0
    missing_baseline_segments = 0
    fully_empty_adjudication_rows = 0

    for _, row in pack_work.iterrows():
        sid = int(row["segment_id"])
        if sid not in out_df_idx.index:
            missing_baseline_segments += 1
            continue
        changed = False
        for adj_col, core_col in ADJUDICATED_TO_CORE.items():
            if adj_col not in row.index:
                continue
            if core_col == "primary_problem_label":
                label = _clean_text(row.get(adj_col))
                if not label:
                    continue
                cur = _clean_text(out_df_idx.at[sid, core_col])
                if label != cur:
                    out_df_idx.at[sid, core_col] = label
                    overwrites_by_field[core_col] += 1
                    changed = True
            else:
                v = _to_num(row.get(adj_col))
                if np.isnan(v):
                    continue
                cur = _to_num(out_df_idx.at[sid, core_col])
                if np.isnan(cur) or abs(cur - v) > 1e-12:
                    out_df_idx.at[sid, core_col] = float(v)
                    overwrites_by_field[core_col] += 1
                    changed = True

        notes = _clean_text(row.get("adjudication_notes"))
        if notes:
            cur_notes = _clean_text(out_df_idx.at[sid, "notes"])
            new_notes = f"{cur_notes} || adjudication:{notes}" if cur_notes else f"adjudication:{notes}"
            if new_notes != cur_notes:
                out_df_idx.at[sid, "notes"] = new_notes
                overwrites_by_field["notes"] += 1
                changed = True

        if changed:
            overwritten_segments += 1
        else:
            fully_empty_adjudication_rows += 1

    out_df_final = out_df_idx.reset_index(drop=True)
    out_df_final = out_df_final[base_df.columns.tolist()]

    out_path = val_dir / "final_annotation_labels_adjudicated.csv"
    report_path = val_dir / "adjudication_report.json"
    out_df_final.to_csv(out_path, index=False, encoding="utf-8")

    flagged_scalar = int(_to_bool_series(pack_df.get("flag_selected_by_scalar", pd.Series(dtype=object))).sum()) if not pack_df.empty else 0
    flagged_label = int(_to_bool_series(pack_df.get("flag_selected_by_label", pd.Series(dtype=object))).sum()) if not pack_df.empty else 0

    pre_all = {}
    pre_flagged = {}
    try:
        a_df = _read_csv_required(a_path, "rater A csv")
        b_df = _read_csv_required(b_path, "rater B csv")
        admin_df = _read_csv_required(admin_path, "sample manifest admin csv")
        canon_a, _ = collapse_rater_hidden_duplicates(a_df, admin_df, rater_id="A")
        canon_b, _ = collapse_rater_hidden_duplicates(b_df, admin_df, rater_id="B")
        merged = canon_a.merge(canon_b, on="segment_id", how="inner", suffixes=("_A", "_B"))
        pre_all = _supplementary_label_agreement(
            merged.get("primary_problem_label_A", pd.Series(dtype=object)),
            merged.get("primary_problem_label_B", pd.Series(dtype=object)),
        )

        if not pack_work.empty:
            flagged_ids = set(pack_work["segment_id"].astype(int).tolist())
            m2 = merged[merged["segment_id"].astype(int).isin(flagged_ids)].copy()
            pre_flagged = _supplementary_label_agreement(
                m2.get("primary_problem_label_A", pd.Series(dtype=object)),
                m2.get("primary_problem_label_B", pd.Series(dtype=object)),
            )
    except Exception as exc:
        logger.warning("supplementary agreement metrics failed: %s", exc)
        pre_all = {"error": str(exc)}
        pre_flagged = {"error": str(exc)}

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_dir": vdir.as_posix(),
        "inputs": {
            "adjudication_pack_csv": pack_path.as_posix(),
            "baseline_final_annotation_labels_csv": base_path.as_posix(),
            "rater_a_csv": a_path.as_posix(),
            "rater_b_csv": b_path.as_posix(),
            "sample_manifest_admin_csv": admin_path.as_posix(),
        },
        "summary": {
            "flagged_segments": int(len(pack_work)),
            "segments_with_scalar_disagreement": int(flagged_scalar),
            "segments_with_label_disagreement": int(flagged_label),
            "segments_overwritten": int(overwritten_segments),
            "rows_without_adjudicated_values": int(fully_empty_adjudication_rows),
            "missing_baseline_segments": int(missing_baseline_segments),
        },
        "overwrites_by_field": overwrites_by_field,
        "supplementary_categorical_agreement": {
            "pre_adjudication_all_segments": pre_all,
            "pre_adjudication_flagged_subset": pre_flagged,
        },
        "output_file": out_path.as_posix(),
        "step7_core_schema_compatible": bool(all(c in out_df_final.columns for c in CORE_OUTPUT_COLUMNS)),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "adjudication finalize done | flagged=%d overwritten=%d output=%s",
        len(pack_work),
        overwritten_segments,
        out_path.as_posix(),
    )
    return {
        "final_annotation_labels_adjudicated_csv": out_path.as_posix(),
        "adjudication_report_json": report_path.as_posix(),
        "flagged_segments": int(len(pack_work)),
        "segments_overwritten": int(overwritten_segments),
        "step7_core_schema_compatible": bool(all(c in out_df_final.columns for c in CORE_OUTPUT_COLUMNS)),
    }
