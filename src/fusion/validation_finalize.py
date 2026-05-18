


"""Finalize two-rater validation labels into one canonical segment-level table."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .annotation_template import CATEGORICAL_FIELDS, SCALAR_SCORE_FIELDS

logger = logging.getLogger("fusion.validation_finalize")

CORE_OUTPUT_COLUMNS: List[str] = [
    "segment_id",
    "safety_score",
    "comfort_score",
    "vitality_score",
    "overall_problem_severity",
    "soundscape_pleasantness",
    "soundscape_eventfulness",
    "primary_problem_label",
    "notes",
]


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path.as_posix()}")
    return pd.read_csv(path)


def _resolve_category_with_confidence(
    labels: Sequence[str],
    confidences: Sequence[float],
) -> Tuple[str, bool, str]:
    """
    Resolve categorical disagreement with confidence rule.

    Returns:
    - resolved_label
    - had_disagreement
    - resolution_rule
    """
    cleaned = [str(x).strip() for x in labels if str(x).strip()]
    if not cleaned:
        return "", False, "empty"
    uniq = list(dict.fromkeys(cleaned))
    if len(uniq) == 1:
        return uniq[0], False, "all_same"

    conf = [float(c) if c is not None and not np.isnan(c) else np.nan for c in confidences]
    if len(conf) >= 2:
        c0, c1 = conf[0], conf[1]
        if not np.isnan(c0) and not np.isnan(c1) and abs(c0 - c1) >= 2:
            idx = 0 if c0 > c1 else 1
            return cleaned[idx], True, "higher_confidence_selected"
    return cleaned[0], True, "first_kept_due_to_close_confidence"


def collapse_rater_hidden_duplicates(
    rater_df: pd.DataFrame,
    admin_df: pd.DataFrame,
    rater_id: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Collapse hidden duplicates into one canonical row per segment for one rater.
    """
    a = admin_df[admin_df["assigned_rater"] == rater_id].copy()
    if a.empty:
        raise ValueError(f"no admin rows for rater {rater_id}")
    required_admin = {"canonical_segment_id", "displayed_item_id", "duplicate_group_id", "is_hidden_duplicate"}
    missing = required_admin - set(a.columns)
    if missing:
        raise ValueError(f"admin manifest missing columns for collapse: {sorted(missing)}")

    required_rater = {"displayed_item_id", "segment_id", *SCALAR_SCORE_FIELDS, *CATEGORICAL_FIELDS, "notes"}
    missing_rater = required_rater - set(rater_df.columns)
    if missing_rater:
        raise ValueError(f"rater csv missing columns: {sorted(missing_rater)}")

    work = rater_df.copy()
    work["displayed_item_id"] = work["displayed_item_id"].astype(str)
    for c in SCALAR_SCORE_FIELDS:
        work[c] = _to_numeric_series(work[c])
    for c in CATEGORICAL_FIELDS + ["notes"]:
        work[c] = work[c].map(_clean_text)

    merged = a.merge(work, on="displayed_item_id", how="left", suffixes=("_admin", ""))
    merged["canonical_segment_id"] = pd.to_numeric(merged["canonical_segment_id"], errors="coerce")
    merged = merged.dropna(subset=["canonical_segment_id"]).copy()
    merged["canonical_segment_id"] = merged["canonical_segment_id"].astype(int)

    rows: List[Dict[str, Any]] = []
    duplicate_conflicts = 0
    segments_with_duplicate = 0

    for seg_id, g in merged.groupby("canonical_segment_id", sort=True):
        g = g.copy()
        if len(g) >= 2:
            segments_with_duplicate += 1

        row: Dict[str, Any] = {
            "segment_id": int(seg_id),
            "rater_id": rater_id,
            "confidence_score": float(pd.to_numeric(g["confidence_score"], errors="coerce").mean())
            if g["confidence_score"].notna().sum() > 0
            else np.nan,
            "notes": " || ".join(
                [f"{rid}:{txt}" for rid, txt in zip(g["displayed_item_id"], g["notes"]) if _clean_text(txt)]
            ),
        }
        for c in SCALAR_SCORE_FIELDS:
            s = pd.to_numeric(g[c], errors="coerce")
            row[c] = float(s.mean()) if s.notna().sum() > 0 else np.nan

        labels = [x for x in g["primary_problem_label"].tolist() if _clean_text(x)]
        conf = pd.to_numeric(g["confidence_score"], errors="coerce").tolist()
        label, had_disagreement, rule = _resolve_category_with_confidence(labels, conf)
        row["primary_problem_label"] = label
        row["duplicate_label_disagreement"] = bool(had_disagreement)
        row["duplicate_label_resolution_rule"] = rule
        if had_disagreement:
            duplicate_conflicts += 1
        rows.append(row)

    collapsed = pd.DataFrame(rows).sort_values("segment_id").reset_index(drop=True)
    summary = {
        "rater_id": rater_id,
        "input_rows": int(len(rater_df)),
        "canonical_rows": int(len(collapsed)),
        "segments_with_duplicate": int(segments_with_duplicate),
        "duplicate_label_conflicts": int(duplicate_conflicts),
    }
    return collapsed, summary


def finalize_two_rater_labels(
    video_dir: str,
    rater_a_csv: Optional[str] = None,
    rater_b_csv: Optional[str] = None,
    admin_manifest_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Finalize labels into one canonical segment-level table for downstream modeling.

    Outputs:
    - validation/final_annotation_labels.csv
    - validation/finalization_report.json
    """
    vdir = Path(video_dir)
    val_dir = vdir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    a_path = Path(rater_a_csv) if rater_a_csv else val_dir / "rater_A_annotation_pack.csv"
    b_path = Path(rater_b_csv) if rater_b_csv else val_dir / "rater_B_annotation_pack.csv"
    admin_path = Path(admin_manifest_csv) if admin_manifest_csv else val_dir / "sample_manifest_admin.csv"

    a_df = _read_csv_required(a_path, "rater A csv")
    b_df = _read_csv_required(b_path, "rater B csv")
    admin_df = _read_csv_required(admin_path, "sample manifest admin csv")

    collapsed_a, summary_a = collapse_rater_hidden_duplicates(a_df, admin_df, rater_id="A")
    collapsed_b, summary_b = collapse_rater_hidden_duplicates(b_df, admin_df, rater_id="B")

    merged = collapsed_a.merge(
        collapsed_b,
        on="segment_id",
        how="outer",
        suffixes=("_A", "_B"),
    )
    merged = merged.sort_values("segment_id").reset_index(drop=True)

    final_rows: List[Dict[str, Any]] = []
    categorical_disagreements = 0
    for _, row in merged.iterrows():
        out: Dict[str, Any] = {
            "segment_id": int(row["segment_id"]),
            "notes_rater_a": _clean_text(row.get("notes_A")),
            "notes_rater_b": _clean_text(row.get("notes_B")),
        }

        for c in SCALAR_SCORE_FIELDS:
            vals = [row.get(f"{c}_A"), row.get(f"{c}_B")]
            arr = pd.to_numeric(pd.Series(vals), errors="coerce")
            out[c] = float(arr.mean()) if arr.notna().sum() > 0 else np.nan

        label_a = _clean_text(row.get("primary_problem_label_A"))
        label_b = _clean_text(row.get("primary_problem_label_B"))
        conf_a = pd.to_numeric(pd.Series([row.get("confidence_score_A")]), errors="coerce").iloc[0]
        conf_b = pd.to_numeric(pd.Series([row.get("confidence_score_B")]), errors="coerce").iloc[0]

        if label_a and label_b and label_a == label_b:
            out["primary_problem_label"] = label_a
            out["label_disagreement"] = False
            out["label_resolution_rule"] = "same_label"
        elif label_a and label_b and label_a != label_b:
            if not np.isnan(conf_a) and not np.isnan(conf_b) and abs(conf_a - conf_b) >= 2:
                out["primary_problem_label"] = label_a if conf_a > conf_b else label_b
                out["label_disagreement"] = True
                out["label_resolution_rule"] = "higher_confidence_rater_selected"
            else:
                out["primary_problem_label"] = "mixed_or_unclear"
                out["label_disagreement"] = True
                out["label_resolution_rule"] = "confidence_close_set_mixed_or_unclear"
            categorical_disagreements += 1
        else:
            out["primary_problem_label"] = label_a or label_b or "mixed_or_unclear"
            out["label_disagreement"] = bool(label_a != label_b)
            out["label_resolution_rule"] = "single_rater_available_or_empty"

        out["rater_a_confidence"] = float(conf_a) if not np.isnan(conf_a) else np.nan
        out["rater_b_confidence"] = float(conf_b) if not np.isnan(conf_b) else np.nan
        out["notes"] = " || ".join([x for x in [out["notes_rater_a"], out["notes_rater_b"]] if x])
        final_rows.append(out)

    final_df = pd.DataFrame(final_rows).sort_values("segment_id").reset_index(drop=True)


    for c in CORE_OUTPUT_COLUMNS:
        if c not in final_df.columns:
            final_df[c] = np.nan if c != "primary_problem_label" and c != "notes" else ""

    out_cols = CORE_OUTPUT_COLUMNS + [
        c
        for c in final_df.columns
        if c not in CORE_OUTPUT_COLUMNS
    ]
    final_df = final_df[out_cols]

    final_path = val_dir / "final_annotation_labels.csv"
    report_path = val_dir / "finalization_report.json"
    final_df.to_csv(final_path, index=False, encoding="utf-8")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_dir": vdir.as_posix(),
        "inputs": {
            "rater_a_csv": a_path.as_posix(),
            "rater_b_csv": b_path.as_posix(),
            "sample_manifest_admin_csv": admin_path.as_posix(),
        },
        "collapse_summary": {
            "rater_A": summary_a,
            "rater_B": summary_b,
        },
        "merge_summary": {
            "final_rows": int(len(final_df)),
            "categorical_disagreements": int(categorical_disagreements),
            "downstream_schema_compatible_step7": bool(all(c in final_df.columns for c in CORE_OUTPUT_COLUMNS)),
        },
        "rules": {
            "duplicate_collapse_scalar": "mean_of_duplicate_pair",
            "duplicate_collapse_categorical": (
                "if same keep; if different and confidence_diff>=2 use higher confidence; else keep first and flag"
            ),
            "two_rater_scalar_merge": "mean_across_raters",
            "two_rater_categorical_merge": (
                "if same keep; if different and confidence_diff>=2 use higher confidence rater; else mixed_or_unclear"
            ),
        },
        "output_file": final_path.as_posix(),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "validation finalize done | final_rows=%d output=%s",
        len(final_df),
        final_path.as_posix(),
    )
    return {
        "final_annotation_labels_csv": final_path.as_posix(),
        "finalization_report_json": report_path.as_posix(),
        "final_rows": int(len(final_df)),
        "downstream_schema_compatible_step7": bool(all(c in final_df.columns for c in CORE_OUTPUT_COLUMNS)),
    }

