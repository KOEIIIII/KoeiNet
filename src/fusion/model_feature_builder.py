


"""Build modeling-safe segment feature table from raw fusion warehouse."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("fusion.model_feature_builder")

EVENT_DIST_COL = "audio_events__event_class_distribution_json"
TOPK_COL = "audio_events__top_k_events"


def _sanitize_name(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "token"


def _to_numeric_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if np.isnan(out):
        return float("nan")
    return out


def _load_csv_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing required raw fusion table: {path.as_posix()}")
    return pd.read_csv(path)


def _load_group_map(feature_dict_path: Path) -> Tuple[Dict[str, str], Optional[str]]:
    if not feature_dict_path.is_file():
        return {}, f"missing feature dictionary: {feature_dict_path.as_posix()}"
    try:
        payload = json.loads(feature_dict_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"feature dictionary parse error: {feature_dict_path.as_posix()} ({exc})"

    out: Dict[str, str] = {}
    groups = payload.get("feature_groups", {})
    if isinstance(groups, dict):
        for group_name, info in groups.items():
            cols = []
            if isinstance(info, dict):
                cols = info.get("columns", [])
            elif isinstance(info, list):
                cols = info
            if isinstance(cols, list):
                for col in cols:
                    out[str(col)] = str(group_name)
    return out, None


def _source_group_for_column(col: str, group_map: Mapping[str, str]) -> str:
    if col in group_map:
        return str(group_map[col])
    if "__" in col:
        return str(col.split("__", 1)[0])
    return "base"


def _parse_event_distribution(value: Any) -> Tuple[Dict[str, float], Optional[str]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}, "missing"
    if isinstance(value, dict):
        raw = value
    else:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return {}, "missing"
        try:
            raw = json.loads(text)
        except Exception as exc:
            return {}, f"invalid_json:{exc}"
    if not isinstance(raw, dict):
        return {}, "not_json_object"

    parsed: Dict[str, float] = {}
    for k, v in raw.items():
        key = str(k).strip()
        if not key:
            continue
        score = _to_numeric_or_nan(v)
        if np.isnan(score) or score <= 0:
            continue
        parsed[key] = float(parsed.get(key, 0.0) + score)

    total = float(sum(parsed.values()))
    if total <= 0:
        return {}, None
    norm = {k: float(v / total) for k, v in parsed.items()}
    return norm, None


def _parse_topk_events(value: Any) -> Tuple[List[str], Optional[str]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return [], "missing"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return [], "missing"

    tokens: List[str] = []
    for part in text.split(";"):
        raw = part.strip()
        if not raw:
            continue
        if ":" in raw:
            name = raw.split(":", 1)[0].strip()
        else:
            name = raw
        if name:
            tokens.append(name)
    return tokens, None


def _build_vocab(
    score_map: Mapping[str, float],
    top_n: int,
) -> List[str]:
    ordered = sorted(score_map.items(), key=lambda kv: (-float(kv[1]), str(kv[0]).lower()))
    return [k for k, _ in ordered[: max(0, int(top_n))]]


def _unique_sanitized_mapping(
    original_tokens: Sequence[str],
    prefix: str,
) -> Dict[str, str]:
    used: set[str] = set()
    mapping: Dict[str, str] = {}
    for token in original_tokens:
        base = _sanitize_name(token)
        candidate = f"{prefix}{base}"
        i = 2
        while candidate in used:
            candidate = f"{prefix}{base}_{i}"
            i += 1
        mapping[str(token)] = candidate
        used.add(candidate)
    return mapping


def _json_dtype(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, np.integer)):
        return "int64"
    if isinstance(value, (float, np.floating)):
        return "float64"
    return type(value).__name__


def _is_numeric_or_bool(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_bool_dtype(series.dtype)
        or pd.api.types.is_integer_dtype(series.dtype)
        or pd.api.types.is_float_dtype(series.dtype)
    )


def _missing_ratio(series: pd.Series) -> float:
    if len(series) == 0:
        return 1.0
    return float(series.isna().mean())


def build_model_feature_table(
    video_dir: str,
    event_vocab_top_n: int = 30,
    topk_vocab_top_n: int = 20,
    drop_high_missing: bool = True,
    high_missing_threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Build modeling-safe table from raw fusion segment warehouse.

    Input files:
    - `<video_dir>/fusion/segment_feature_table.csv`
    - `<video_dir>/fusion/feature_dictionary.json`

    Output files:
    - `<video_dir>/fusion/model_feature_table.csv`
    - `<video_dir>/fusion/model_feature_dictionary.json`
    - `<video_dir>/fusion/model_feature_report.json`
    """
    vdir = Path(video_dir)
    fusion_dir = vdir / "fusion"
    raw_table_path = fusion_dir / "segment_feature_table.csv"
    raw_dict_path = fusion_dir / "feature_dictionary.json"

    model_table_path = fusion_dir / "model_feature_table.csv"
    model_dict_path = fusion_dir / "model_feature_dictionary.json"
    model_report_path = fusion_dir / "model_feature_report.json"

    raw_df = _load_csv_required(raw_table_path)
    if "segment_id" not in raw_df.columns:
        raise ValueError("raw fusion table missing required column: segment_id")
    raw_rows = int(len(raw_df))
    raw_cols = int(len(raw_df.columns))

    group_map, group_map_warning = _load_group_map(raw_dict_path)

    model_df = pd.DataFrame({"segment_id": pd.to_numeric(raw_df["segment_id"], errors="coerce")})
    if model_df["segment_id"].isna().any():
        raise ValueError("segment_id contains invalid values in raw fusion table")
    model_df["segment_id"] = model_df["segment_id"].astype(int)

    feature_meta: Dict[str, Dict[str, Any]] = {
        "segment_id": {
            "source_group": "base",
            "source_column": "segment_id",
            "role": "numeric_direct",
            "dtype": "int64",
            "description": "Segment identifier key.",
        }
    }
    excluded_columns: List[Dict[str, Any]] = []
    transformed_columns: List[str] = []
    warnings: List[str] = []
    if group_map_warning:
        warnings.append(group_map_warning)
        logger.warning("[fusion.model] %s", group_map_warning)

    role_counts: Dict[str, int] = {
        "numeric_direct": 1,
        "parsed_json": 0,
        "multi_hot": 0,
        "summary": 0,
        "excluded": 0,
    }


    event_dist_mapping: Dict[str, str] = {}
    if EVENT_DIST_COL in raw_df.columns:
        parsed_dists: List[Dict[str, float]] = []
        global_dist_score: Dict[str, float] = {}
        invalid_json_rows = 0
        for v in raw_df[EVENT_DIST_COL].tolist():
            dist, err = _parse_event_distribution(v)
            if err and err != "missing":
                invalid_json_rows += 1
            parsed_dists.append(dist)
            for k, score in dist.items():
                global_dist_score[k] = float(global_dist_score.get(k, 0.0) + score)

        vocab = _build_vocab(global_dist_score, top_n=event_vocab_top_n)
        event_dist_mapping = _unique_sanitized_mapping(vocab, prefix="audio_events_dist__")
        for event_name in vocab:
            col = event_dist_mapping[event_name]
            model_df[col] = [float(d.get(event_name, 0.0)) for d in parsed_dists]
            transformed_columns.append(col)
            role_counts["parsed_json"] += 1
            feature_meta[col] = {
                "source_group": "audio_events",
                "source_column": EVENT_DIST_COL,
                "role": "parsed_json",
                "dtype": "float64",
                "description": f"Normalized event distribution for class `{event_name}`.",
            }

        other_col = "audio_events_dist__other"
        model_df[other_col] = [
            float(max(0.0, 1.0 - sum(float(d.get(ev, 0.0)) for ev in vocab))) if d else 0.0
            for d in parsed_dists
        ]
        transformed_columns.append(other_col)
        role_counts["parsed_json"] += 1
        feature_meta[other_col] = {
            "source_group": "audio_events",
            "source_column": EVENT_DIST_COL,
            "role": "parsed_json",
            "dtype": "float64",
            "description": "Overflow probability mass outside selected event distribution vocabulary.",
        }

        if invalid_json_rows > 0:
            warnings.append(f"{EVENT_DIST_COL}: invalid_json_rows={invalid_json_rows}")
            logger.warning("[fusion.model] %s invalid_json_rows=%d", EVENT_DIST_COL, invalid_json_rows)

        excluded_columns.append(
            {
                "column": EVENT_DIST_COL,
                "reason": "raw_text_not_model_safe",
                "source_group": "audio_events",
                "note": "Expanded into numeric distribution vocabulary columns.",
            }
        )

    topk_event_mapping: Dict[str, str] = {}
    if TOPK_COL in raw_df.columns:
        parsed_tokens: List[List[str]] = []
        token_counts: Dict[str, float] = {}
        invalid_rows = 0
        for v in raw_df[TOPK_COL].tolist():
            tokens, err = _parse_topk_events(v)
            if err and err != "missing":
                invalid_rows += 1
            parsed_tokens.append(tokens)
            for token in set(tokens):
                token_counts[token] = float(token_counts.get(token, 0.0) + 1.0)

        vocab = _build_vocab(token_counts, top_n=topk_vocab_top_n)
        topk_event_mapping = _unique_sanitized_mapping(vocab, prefix="audio_events_topk__")
        vocab_set = set(vocab)

        for token in vocab:
            col = topk_event_mapping[token]
            model_df[col] = [1 if token in set(tokens) else 0 for tokens in parsed_tokens]
            transformed_columns.append(col)
            role_counts["multi_hot"] += 1
            feature_meta[col] = {
                "source_group": "audio_events",
                "source_column": TOPK_COL,
                "role": "multi_hot",
                "dtype": "int64",
                "description": f"Multi-hot presence of top-k event token `{token}` in segment.",
            }

        c_count = "audio_events_topk_count"
        c_known = "audio_events_topk_known_count"
        model_df[c_count] = [int(len(tokens)) for tokens in parsed_tokens]
        model_df[c_known] = [int(sum(1 for t in tokens if t in vocab_set)) for tokens in parsed_tokens]
        transformed_columns.extend([c_count, c_known])
        role_counts["summary"] += 2
        feature_meta[c_count] = {
            "source_group": "audio_events",
            "source_column": TOPK_COL,
            "role": "summary",
            "dtype": "int64",
            "description": "Count of parsed top-k event tokens in segment.",
        }
        feature_meta[c_known] = {
            "source_group": "audio_events",
            "source_column": TOPK_COL,
            "role": "summary",
            "dtype": "int64",
            "description": "Count of parsed top-k event tokens that match selected vocabulary.",
        }
        if invalid_rows > 0:
            warnings.append(f"{TOPK_COL}: invalid_rows={invalid_rows}")
            logger.warning("[fusion.model] %s invalid_rows=%d", TOPK_COL, invalid_rows)

        excluded_columns.append(
            {
                "column": TOPK_COL,
                "reason": "raw_text_not_model_safe",
                "source_group": "audio_events",
                "note": "Expanded into multi-hot token columns and summary counts.",
            }
        )

    transformed_source_cols = {EVENT_DIST_COL, TOPK_COL}


    for col in raw_df.columns:
        if col == "segment_id":
            continue
        if col in transformed_source_cols:
            continue

        series = raw_df[col]
        if _is_numeric_or_bool(series):
            model_df[col] = series
            role_counts["numeric_direct"] += 1
            feature_meta[col] = {
                "source_group": _source_group_for_column(col, group_map),
                "source_column": col,
                "role": "numeric_direct",
                "dtype": str(series.dtype),
                "description": "Direct numeric/bool feature from raw warehouse.",
            }
            continue

        reason = "unsupported_dtype"
        if pd.api.types.is_object_dtype(series.dtype):
            if str(col).endswith("_json"):
                reason = "raw_json_not_expanded"
            else:
                reason = "raw_text_not_model_safe"
        excluded_columns.append(
            {
                "column": col,
                "reason": reason,
                "source_group": _source_group_for_column(col, group_map),
                "dtype": str(series.dtype),
            }
        )


    dropped_high_missing: List[Dict[str, Any]] = []
    if bool(drop_high_missing):
        drop_cols: List[str] = []
        for col in model_df.columns:
            if col == "segment_id":
                continue
            miss = _missing_ratio(model_df[col])
            if miss > float(high_missing_threshold):
                drop_cols.append(col)
                dropped_high_missing.append(
                    {
                        "column": col,
                        "missing_ratio": float(miss),
                        "reason": "all_missing" if miss >= 1.0 else "too_sparse",
                    }
                )
        if drop_cols:
            model_df = model_df.drop(columns=drop_cols)
            for item in dropped_high_missing:
                col = item["column"]
                src_meta = feature_meta.get(col, {})
                excluded_columns.append(
                    {
                        "column": col,
                        "reason": item["reason"],
                        "source_group": src_meta.get("source_group", "unknown"),
                        "source_column": src_meta.get("source_column", col),
                        "missing_ratio": item["missing_ratio"],
                    }
                )
                feature_meta.pop(col, None)

    role_counts["excluded"] = int(len(excluded_columns))


    object_cols = [
        c
        for c in model_df.columns
        if c != "segment_id" and pd.api.types.is_object_dtype(model_df[c].dtype)
    ]
    if object_cols:
        for col in object_cols:
            excluded_columns.append(
                {
                    "column": col,
                    "reason": "unsupported_dtype",
                    "source_group": _source_group_for_column(col, group_map),
                    "dtype": str(model_df[col].dtype),
                }
            )
            feature_meta.pop(col, None)
        model_df = model_df.drop(columns=object_cols)
        role_counts["excluded"] = int(len(excluded_columns))
        warnings.append(f"removed_object_columns={len(object_cols)}")


    if int(len(model_df)) != raw_rows:
        raise RuntimeError(
            f"model table row mismatch: raw_rows={raw_rows}, model_rows={len(model_df)}"
        )

    model_df.to_csv(model_table_path, index=False, encoding="utf-8")

    final_role_counts: Dict[str, int] = {
        "numeric_direct": 0,
        "parsed_json": 0,
        "multi_hot": 0,
        "summary": 0,
    }
    for col, meta in feature_meta.items():
        role = str(meta.get("role", "numeric_direct"))
        if role in final_role_counts:
            final_role_counts[role] += 1


    for col in list(feature_meta.keys()):
        if col not in model_df.columns:
            feature_meta.pop(col, None)
            continue
        feature_meta[col]["dtype"] = _json_dtype(model_df[col].dtype.name if hasattr(model_df[col].dtype, "name") else model_df[col].dtype)

        feature_meta[col]["dtype"] = str(model_df[col].dtype)

    dictionary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_dir": vdir.as_posix(),
        "source_files": {
            "raw_feature_table_csv": raw_table_path.as_posix(),
            "raw_feature_dictionary_json": raw_dict_path.as_posix(),
        },
        "config": {
            "model_event_vocab_top_n": int(event_vocab_top_n),
            "model_topk_event_vocab_top_n": int(topk_vocab_top_n),
            "model_drop_high_missing": bool(drop_high_missing),
            "model_high_missing_threshold": float(high_missing_threshold),
        },
        "sanitization_mapping": {
            "event_distribution_class_to_column": event_dist_mapping,
            "topk_event_token_to_column": topk_event_mapping,
        },
        "feature_role_counts": final_role_counts,
        "feature_metadata": feature_meta,
        "excluded_columns_metadata": excluded_columns,
        "table_schema": list(model_df.columns),
    }
    model_dict_path.write_text(
        json.dumps(dictionary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dtype_summary = {
        str(dtype): int(count)
        for dtype, count in model_df.dtypes.astype(str).value_counts().to_dict().items()
    }
    report_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_dir": vdir.as_posix(),
        "raw_table": {
            "rows": raw_rows,
            "columns": raw_cols,
        },
        "model_table": {
            "rows": int(len(model_df)),
            "columns": int(len(model_df.columns)),
            "row_count_matches_raw": bool(int(len(model_df)) == raw_rows),
            "dtype_summary": dtype_summary,
        },
        "feature_role_counts": final_role_counts,
        "excluded_columns": excluded_columns,
        "dropped_high_missing_columns": dropped_high_missing,
        "newly_expanded_columns_count": int(len(transformed_columns)),
        "transformed_columns": transformed_columns,
        "remaining_object_columns": object_cols,
        "quality_checks": {
            "no_raw_object_text_columns_except_segment_id": bool(len(object_cols) == 0),
            "row_count_matches_raw": bool(int(len(model_df)) == raw_rows),
        },
        "warnings": warnings,
    }
    model_report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "model feature table built | rows=%d cols=%d expanded=%d excluded=%d",
        len(model_df),
        len(model_df.columns),
        len(transformed_columns),
        len(excluded_columns),
    )

    return {
        "model_feature_table_csv": model_table_path.as_posix(),
        "model_feature_dictionary_json": model_dict_path.as_posix(),
        "model_feature_report_json": model_report_path.as_posix(),
        "rows": int(len(model_df)),
        "columns": int(len(model_df.columns)),
        "feature_role_counts": final_role_counts,
        "excluded_columns": excluded_columns,
        "transformed_columns": transformed_columns,
        "schema": list(model_df.columns),
        "preview_rows": model_df.head(3).to_dict("records"),
    }

