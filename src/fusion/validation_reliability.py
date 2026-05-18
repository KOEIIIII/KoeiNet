


"""Compute inter-rater and intra-rater reliability for two-rater validation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .annotation_template import CATEGORICAL_FIELDS, SCALAR_SCORE_FIELDS
from .validation_finalize import collapse_rater_hidden_duplicates

logger = logging.getLogger("fusion.validation_reliability")


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return float("nan")
    return float(x.corr(y, method="spearman"))


def _mae(x: pd.Series, y: pd.Series) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.mean(np.abs(x.to_numpy(dtype=float) - y.to_numpy(dtype=float))))


def _icc_2_1_from_two_raters(x: pd.Series, y: pd.Series) -> float:
    """
    ICC(2,1) absolute agreement for matrix n x 2.
    """
    data = np.column_stack([x.to_numpy(dtype=float), y.to_numpy(dtype=float)])
    n, k = data.shape
    if n < 2 or k != 2:
        return float("nan")

    mean_targets = np.mean(data, axis=1, keepdims=True)
    mean_raters = np.mean(data, axis=0, keepdims=True)
    grand = np.mean(data)

    ss_targets = k * np.sum((mean_targets - grand) ** 2)
    ss_raters = n * np.sum((mean_raters - grand) ** 2)
    ss_error = np.sum((data - mean_targets - mean_raters + grand) ** 2)

    ms_targets = ss_targets / (n - 1)
    ms_raters = ss_raters / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom = ms_targets + (k - 1) * ms_error + (k * (ms_raters - ms_error) / n)
    if denom == 0:
        return float("nan")
    return float((ms_targets - ms_error) / denom)


def _cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    if len(labels_a) == 0:
        return float("nan")
    a = [str(x) for x in labels_a]
    b = [str(x) for x in labels_b]
    cats = sorted(set(a) | set(b))
    if not cats:
        return float("nan")
    p0 = float(np.mean([x == y for x, y in zip(a, b)]))
    pa = {c: float(np.mean([x == c for x in a])) for c in cats}
    pb = {c: float(np.mean([y == c for y in b])) for c in cats}
    pe = float(sum(pa[c] * pb[c] for c in cats))
    if pe >= 1.0:
        return float("nan")
    return float((p0 - pe) / (1.0 - pe))


def _raw_agreement(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    if len(labels_a) == 0:
        return float("nan")
    return float(np.mean([str(x) == str(y) for x, y in zip(labels_a, labels_b)]))


def _duplicate_pairs_for_rater(
    rater_df: pd.DataFrame,
    admin_df: pd.DataFrame,
    rater_id: str,
) -> pd.DataFrame:
    a = admin_df[admin_df["assigned_rater"] == rater_id].copy()
    a = a[a["duplicate_group_id"].astype(str).str.strip() != ""].copy()
    if a.empty:
        return pd.DataFrame()

    w = rater_df.copy()
    w["displayed_item_id"] = w["displayed_item_id"].astype(str)
    m = a.merge(w, on="displayed_item_id", how="left")
    m["order_idx"] = m["displayed_item_id"].str.extract(r"_(\d+)$").astype(float)

    rows: List[Dict[str, Any]] = []
    for gid, g in m.groupby("duplicate_group_id"):
        g = g.sort_values("order_idx")
        if len(g) < 2:
            continue
        r1 = g.iloc[0]
        r2 = g.iloc[1]
        row = {
            "duplicate_group_id": str(gid),
            "segment_id": int(pd.to_numeric(r1.get("canonical_segment_id"), errors="coerce")),
        }
        for f in SCALAR_SCORE_FIELDS + CATEGORICAL_FIELDS:
            row[f"{f}_1"] = r1.get(f)
            row[f"{f}_2"] = r2.get(f)
        rows.append(row)
    return pd.DataFrame(rows)


def _scalar_metric_block(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    xx = _to_numeric(x)
    yy = _to_numeric(y)
    valid = xx.notna() & yy.notna()
    xx = xx[valid]
    yy = yy[valid]
    n = int(len(xx))
    if n == 0:
        return {"n": 0, "icc_2_1": float("nan"), "spearman": float("nan"), "mae": float("nan")}
    return {
        "n": n,
        "icc_2_1": _icc_2_1_from_two_raters(xx, yy),
        "spearman": _spearman(xx, yy),
        "mae": _mae(xx, yy),
    }


def _categorical_metric_block(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    xx = x.fillna("").astype(str).str.strip()
    yy = y.fillna("").astype(str).str.strip()
    valid = (xx != "") & (yy != "")
    xx = xx[valid]
    yy = yy[valid]
    n = int(len(xx))
    if n == 0:
        return {"n": 0, "cohen_kappa": float("nan"), "raw_agreement": float("nan")}
    return {
        "n": n,
        "cohen_kappa": _cohen_kappa(xx.tolist(), yy.tolist()),
        "raw_agreement": _raw_agreement(xx.tolist(), yy.tolist()),
    }


def compute_validation_reliability(
    video_dir: str,
    rater_a_csv: Optional[str] = None,
    rater_b_csv: Optional[str] = None,
    admin_manifest_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute reliability metrics and export JSON + Markdown summary.

    Outputs:
    - validation/reliability_report.json
    - validation/reliability_summary.md
    """
    vdir = Path(video_dir)
    val_dir = vdir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    a_path = Path(rater_a_csv) if rater_a_csv else val_dir / "rater_A_annotation_pack.csv"
    b_path = Path(rater_b_csv) if rater_b_csv else val_dir / "rater_B_annotation_pack.csv"
    admin_path = Path(admin_manifest_csv) if admin_manifest_csv else val_dir / "sample_manifest_admin.csv"

    if not a_path.is_file() or not b_path.is_file() or not admin_path.is_file():
        raise FileNotFoundError("reliability inputs missing: require rater A/B csv and sample manifest admin")

    a_df = pd.read_csv(a_path)
    b_df = pd.read_csv(b_path)
    admin_df = pd.read_csv(admin_path)

    canon_a, summary_a = collapse_rater_hidden_duplicates(a_df, admin_df, rater_id="A")
    canon_b, summary_b = collapse_rater_hidden_duplicates(b_df, admin_df, rater_id="B")

    inter_df = canon_a.merge(canon_b, on="segment_id", how="inner", suffixes=("_A", "_B"))

    inter_scalar: Dict[str, Any] = {}
    for f in SCALAR_SCORE_FIELDS:
        inter_scalar[f] = _scalar_metric_block(inter_df.get(f"{f}_A", pd.Series(dtype=float)), inter_df.get(f"{f}_B", pd.Series(dtype=float)))

    inter_cat: Dict[str, Any] = {}
    for f in CATEGORICAL_FIELDS:
        inter_cat[f] = _categorical_metric_block(
            inter_df.get(f"{f}_A", pd.Series(dtype=object)),
            inter_df.get(f"{f}_B", pd.Series(dtype=object)),
        )

    intra: Dict[str, Any] = {}
    for rid, rdf in (("A", a_df), ("B", b_df)):
        pairs = _duplicate_pairs_for_rater(rdf, admin_df, rater_id=rid)
        scalar_block: Dict[str, Any] = {}
        for f in SCALAR_SCORE_FIELDS:
            scalar_block[f] = _scalar_metric_block(
                pairs.get(f"{f}_1", pd.Series(dtype=float)),
                pairs.get(f"{f}_2", pd.Series(dtype=float)),
            )
        cat_block: Dict[str, Any] = {}
        for f in CATEGORICAL_FIELDS:
            cat_block[f] = _categorical_metric_block(
                pairs.get(f"{f}_1", pd.Series(dtype=object)),
                pairs.get(f"{f}_2", pd.Series(dtype=object)),
            )
        intra[rid] = {
            "duplicate_pair_count": int(len(pairs)),
            "scalar_fields": scalar_block,
            "categorical_fields": cat_block,
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_dir": vdir.as_posix(),
        "inputs": {
            "rater_a_csv": a_path.as_posix(),
            "rater_b_csv": b_path.as_posix(),
            "sample_manifest_admin_csv": admin_path.as_posix(),
        },
        "protocol": {
            "inter_rater_unique_segments": int(len(inter_df)),
            "scalar_fields": SCALAR_SCORE_FIELDS,
            "categorical_fields": CATEGORICAL_FIELDS,
        },
        "collapse_summary": {"rater_A": summary_a, "rater_B": summary_b},
        "inter_rater": {
            "scalar_fields": inter_scalar,
            "categorical_fields": inter_cat,
        },
        "intra_rater": intra,
    }

    json_path = val_dir / "reliability_report.json"
    md_path = val_dir / "reliability_summary.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = [
        "# Reliability Summary",
        "",
        f"- unique segments for inter-rater: {len(inter_df)}",
        f"- rater A duplicate pairs: {intra['A']['duplicate_pair_count']}",
        f"- rater B duplicate pairs: {intra['B']['duplicate_pair_count']}",
        "",
        "## Inter-rater (scalar)",
    ]
    for f in SCALAR_SCORE_FIELDS:
        m = inter_scalar[f]
        lines.append(
            f"- {f}: n={m['n']}, ICC(2,1)={m['icc_2_1']:.4f}, Spearman={m['spearman']:.4f}, MAE={m['mae']:.4f}"
        )
    lines.append("")
    lines.append("## Inter-rater (categorical)")
    for f in CATEGORICAL_FIELDS:
        m = inter_cat[f]
        lines.append(
            f"- {f}: n={m['n']}, Cohen_kappa={m['cohen_kappa']:.4f}, raw_agreement={m['raw_agreement']:.4f}"
        )
    lines.append("")
    lines.append("## Intra-rater QC")
    for rid in ("A", "B"):
        lines.append(f"### Rater {rid}")
        lines.append(f"- duplicate_pair_count: {intra[rid]['duplicate_pair_count']}")
        for f in SCALAR_SCORE_FIELDS:
            m = intra[rid]["scalar_fields"][f]
            lines.append(
                f"- {f}: n={m['n']}, ICC(2,1)={m['icc_2_1']:.4f}, Spearman={m['spearman']:.4f}, MAE={m['mae']:.4f}"
            )
        for f in CATEGORICAL_FIELDS:
            m = intra[rid]["categorical_fields"][f]
            lines.append(
                f"- {f}: n={m['n']}, Cohen_kappa={m['cohen_kappa']:.4f}, raw_agreement={m['raw_agreement']:.4f}"
            )
        lines.append("")
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    logger.info(
        "validation reliability done | inter_n=%d report=%s",
        len(inter_df),
        json_path.as_posix(),
    )
    return {
        "reliability_report_json": json_path.as_posix(),
        "reliability_summary_md": md_path.as_posix(),
        "inter_rater_unique_segments": int(len(inter_df)),
    }

