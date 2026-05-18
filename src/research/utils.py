


"""Shared utilities for research analysis modules."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

VISUAL_SOURCE_GROUPS = {
    "visual_semantic",
    "visual_major",
    "green_view",
    "emotion",
    "people",
    "color",
    "ai_activity",
    "ai_activity_summary",
}

AUDIO_SOURCE_GROUPS = {
    "audio_events",
    "audio_signal",
    "audio_embedding",
}

MODEL_NAME_MAP = {
    "early_fusion_screened": "Fusion",
    "visual_only_screened": "Visual-only",
    "audio_only_screened": "Audio-only",
    "late_fusion_legacy": "Late-fusion legacy",
}

GROUP_NAME_MAP = {
    "visual_semantic": "Visual semantic",
    "visual_major": "Visual major",
    "green_view": "Green view",
    "emotion": "Emotion",
    "people": "People",
    "color": "Color",
    "ai_activity": "AI activity",
    "ai_activity_summary": "AI activity summary",
    "audio_events": "Audio events",
    "audio_signal": "Audio signal",
    "audio_embedding": "Audio embedding",
    "people_presence": "People presence",
    "green_nature": "Green / nature",
    "traffic_road_hardscape": "Traffic / road / hardscape",
    "visual_emotion_aesthetic": "Visual emotion / aesthetic",
    "ai_activity": "AI activity",
    "visual_color": "Visual color",
    "visual_semantic_general": "General visual semantics",
    "audio_signal_level": "Audio signal level",
    "audio_event_human": "Human audio events",
    "audio_event_traffic_mechanical": "Traffic / mechanical audio",
    "audio_event_natural": "Natural audio events",
    "audio_event_general": "General audio events",
    "audio_embedding_general": "General audio embedding",
}

ACTIVITY_TOKEN_MAP = {
    "坐下休息": "sitting",
    "站着停留": "standing",
    "散步": "walking",
    "跑步": "running",
    "健身锻炼": "fitness",
    "买菜购物": "shopping",
}


def normalize_segment_id(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "segment_id" not in df.columns:
        raise ValueError(f"{name} missing required column: segment_id")
    out = df.copy()
    out["segment_id"] = pd.to_numeric(out["segment_id"], errors="coerce")
    out = out.dropna(subset=["segment_id"]).copy()
    out["segment_id"] = out["segment_id"].astype(int)
    return out


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_json_value(dict(payload)), ensure_ascii=False, indent=2), encoding="utf-8")


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(p_values), dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return out
    pv = arr[mask]
    order = np.argsort(pv)
    ranked = pv[order]
    n = ranked.size
    adjusted = ranked * n / np.arange(1, n + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    out[mask] = restored
    return out


def infer_feature_group(feature_name: str, feature_meta: Mapping[str, Any]) -> Tuple[str, str]:
    meta = feature_meta.get(feature_name, {})
    source_group = ""
    if isinstance(meta, dict):
        source_group = str(meta.get("source_group") or "").strip()

    if not source_group:
        if feature_name.startswith(("visual_semantic__",)):
            source_group = "visual_semantic"
        elif feature_name.startswith(("visual_major__",)):
            source_group = "visual_major"
        elif feature_name.startswith(("green_view__",)):
            source_group = "green_view"
        elif feature_name.startswith(("emotion__",)):
            source_group = "emotion"
        elif feature_name.startswith(("people__",)):
            source_group = "people"
        elif feature_name.startswith(("color__",)):
            source_group = "color"
        elif feature_name.startswith(("ai_activity__",)):
            source_group = "ai_activity"
        elif feature_name.startswith(("audio_events__", "audio_events_dist__", "audio_events_topk__")):
            source_group = "audio_events"
        elif feature_name.startswith(("audio_signal__",)):
            source_group = "audio_signal"
        elif feature_name.startswith(("audio_embedding__",)):
            source_group = "audio_embedding"
        elif "__" in feature_name:
            source_group = feature_name.split("__", 1)[0]
        else:
            source_group = "unknown"

    if source_group in VISUAL_SOURCE_GROUPS:
        return "visual", source_group
    if source_group in AUDIO_SOURCE_GROUPS:
        return "audio", source_group
    return "other", source_group


def obvious_non_feature_reason(feature_name: str) -> str:
    name = feature_name.lower()
    if feature_name == "segment_id":
        return "segment_key"
    bad_tokens = (
        "path",
        "file",
        "filename",
        "folder",
        "url",
        "json",
        "text",
        "note",
        "label",
        "target",
    )
    if any(tok in name for tok in bad_tokens):
        return "id_or_text_like_column"
    if name in {
        "start_time_sec",
        "end_time_sec",
        "center_time_sec",
        "included_frame_count",
        "linked_audio_start_time_sec",
        "linked_audio_end_time_sec",
        "audio_event_row_count",
        "audio_events__audio_event_row_count",
        "audio_signal__linked_audio_start_time_sec",
        "audio_signal__linked_audio_end_time_sec",
        "audio_signal__sample_rate",
        "audio_embedding__panns_embedding_dim",
    }:
        return "timing_or_bookkeeping_column"
    return ""


def _is_near_zero_variance(series: pd.Series, tolerance: float = 1e-8) -> bool:
    clean = series.dropna()
    if clean.empty:
        return True
    unique_count = int(clean.nunique(dropna=True))
    if unique_count <= 1:
        return True
    if float(clean.std(ddof=0)) <= tolerance:
        return True
    top_frequency = float(clean.value_counts(normalize=True, dropna=False).iloc[0])
    if unique_count <= 5 and top_frequency >= 0.98:
        return True
    return False


def prepare_cross_modal_feature_table(
    model_df: pd.DataFrame,
    feature_dict: Mapping[str, Any],
    *,
    missing_threshold: float = 0.30,
) -> Dict[str, Any]:
    feature_meta = feature_dict.get("feature_metadata", {})
    if not isinstance(feature_meta, dict):
        feature_meta = {}

    candidate_records: List[Dict[str, Any]] = []
    kept_features: Dict[str, pd.Series] = {}
    seen_hashes: Dict[Tuple[float, ...], str] = {}

    for feature_name in [c for c in model_df.columns if c != "segment_id"]:
        modality, source_group = infer_feature_group(feature_name, feature_meta)
        base_reason = obvious_non_feature_reason(feature_name)
        series = pd.to_numeric(model_df[feature_name], errors="coerce")
        missing_ratio = float(series.isna().mean())
        record: Dict[str, Any] = {
            "feature_name": feature_name,
            "modality": modality,
            "source_group": source_group,
            "missing_ratio": missing_ratio,
            "n_unique_non_na": int(series.dropna().nunique()),
            "kept_or_dropped": "dropped",
            "reason": "",
        }

        if modality not in {"visual", "audio"}:
            record["reason"] = "unsupported_modality"
            candidate_records.append(record)
            continue
        if base_reason:
            record["reason"] = base_reason
            candidate_records.append(record)
            continue
        if missing_ratio > float(missing_threshold):
            record["reason"] = f"missing_ratio_gt_{missing_threshold:.2f}"
            candidate_records.append(record)
            continue
        if _is_near_zero_variance(series):
            record["reason"] = "near_zero_variance"
            candidate_records.append(record)
            continue

        filled = series.fillna(series.median())
        filled = filled.fillna(0.0).astype(float)
        hashed = tuple(np.round(filled.to_numpy(), 12).tolist())
        if hashed in seen_hashes:
            record["reason"] = f"duplicate_of:{seen_hashes[hashed]}"
            candidate_records.append(record)
            continue

        seen_hashes[hashed] = feature_name
        record["kept_or_dropped"] = "kept"
        record["reason"] = "kept"
        kept_features[feature_name] = filled
        candidate_records.append(record)

    if not kept_features:
        raise RuntimeError("No usable cross-modal features remained after conservative screening.")

    kept_df = pd.DataFrame(kept_features)
    registry_df = pd.DataFrame(candidate_records).sort_values(["modality", "source_group", "feature_name"]).reset_index(drop=True)
    visual_features = registry_df.loc[
        (registry_df["kept_or_dropped"] == "kept") & (registry_df["modality"] == "visual"),
        "feature_name",
    ].tolist()
    audio_features = registry_df.loc[
        (registry_df["kept_or_dropped"] == "kept") & (registry_df["modality"] == "audio"),
        "feature_name",
    ].tolist()
    if not visual_features or not audio_features:
        raise RuntimeError(
            f"Cross-modal analysis requires both visual and audio features after screening; "
            f"got visual={len(visual_features)} audio={len(audio_features)}."
        )
    return {
        "feature_df": kept_df,
        "feature_registry": registry_df,
        "feature_meta": feature_meta,
        "visual_features": visual_features,
        "audio_features": audio_features,
    }


def order_segments(manifest_df: pd.DataFrame, segment_ids: Sequence[int]) -> pd.DataFrame:
    ordered = normalize_segment_id(manifest_df, "segment_manifest").copy()
    ordered = ordered[ordered["segment_id"].isin([int(x) for x in segment_ids])].copy()
    sort_cols = [c for c in ["center_time_sec", "start_time_sec", "segment_id"] if c in ordered.columns]
    if not sort_cols:
        sort_cols = ["segment_id"]
    ordered = ordered.sort_values(sort_cols).reset_index(drop=True)
    ordered["analysis_order"] = np.arange(len(ordered), dtype=int)
    return ordered


def thin_mask(n_rows: int, offset: int = 0) -> np.ndarray:
    mask = np.zeros(int(n_rows), dtype=bool)
    mask[offset::2] = True
    return mask


def contiguous_blocks(n_rows: int, block_size: int) -> List[np.ndarray]:
    n_rows = int(n_rows)
    block_size = max(1, int(block_size))
    return [np.arange(i, min(i + block_size, n_rows), dtype=int) for i in range(0, n_rows, block_size)]


def circular_block_permutations(n_rows: int, block_size: int, include_identity: bool = False) -> List[np.ndarray]:
    blocks = contiguous_blocks(n_rows=n_rows, block_size=block_size)
    if len(blocks) <= 1:
        return [np.arange(n_rows, dtype=int)] if include_identity else []
    perms: List[np.ndarray] = []
    start = 0 if include_identity else 1
    for shift in range(start, len(blocks)):
        reordered = blocks[shift:] + blocks[:shift]
        perms.append(np.concatenate(reordered).astype(int))
    return perms


def block_bootstrap_ci(
    values: Sequence[float],
    *,
    block_size: int,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.95,
) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "ok": False,
            "n": 0,
            "mean": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }
    rng = np.random.default_rng(int(seed))
    blocks = contiguous_blocks(len(arr), block_size=block_size)
    draws = np.empty(int(max(1, n_bootstrap)), dtype=float)
    for i in range(draws.size):
        sample_idx: List[int] = []
        while len(sample_idx) < len(arr):
            block = blocks[int(rng.integers(0, len(blocks)))]
            sample_idx.extend(block.tolist())
        sample = arr[np.asarray(sample_idx[: len(arr)], dtype=int)]
        draws[i] = float(np.mean(sample))
    lo = float((1.0 - alpha) / 2.0)
    hi = float(1.0 - lo)
    return {
        "ok": True,
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "ci_lower": float(np.quantile(draws, lo)),
        "ci_upper": float(np.quantile(draws, hi)),
    }


def block_sign_flip_test(
    values: Sequence[float],
    *,
    block_size: int,
    n_permutations: int,
    seed: int,
) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "ok": False,
            "observed_mean": np.nan,
            "p_value": np.nan,
            "n": 0,
            "n_blocks": 0,
            "method": "block_sign_flip",
        }
    blocks = contiguous_blocks(len(arr), block_size=block_size)
    block_means = np.asarray([float(np.mean(arr[idx])) for idx in blocks], dtype=float)
    block_weights = np.asarray([len(idx) for idx in blocks], dtype=float)
    observed = float(np.average(block_means, weights=block_weights))
    n_blocks = len(blocks)
    rng = np.random.default_rng(int(seed))

    if n_blocks <= 15:
        signs = np.asarray(np.meshgrid(*([[-1.0, 1.0]] * n_blocks))).T.reshape(-1, n_blocks)
        null = np.average(signs * block_means[None, :], weights=block_weights, axis=1)
    else:
        draws = max(1000, int(n_permutations))
        signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, n_blocks), replace=True)
        null = np.average(signs * block_means[None, :], weights=block_weights, axis=1)

    p_value = float((np.sum(np.abs(null) >= abs(observed)) + 1.0) / (len(null) + 1.0))
    return {
        "ok": True,
        "observed_mean": observed,
        "p_value": p_value,
        "n": int(arr.size),
        "n_blocks": int(n_blocks),
        "method": "block_sign_flip",
    }


def standardize_rank_frame(df: pd.DataFrame) -> np.ndarray:
    ranks = df.rank(method="average", na_option="keep")
    arr = ranks.to_numpy(dtype=float)
    arr = arr - np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, ddof=1, keepdims=True)
    std[~np.isfinite(std) | (std <= 0)] = 1.0
    return arr / std


def centered_distance_flatten(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    mats: List[np.ndarray] = []
    dvar2: List[float] = []
    for col in df.columns:
        values = df[col].to_numpy(dtype=float)
        dist = squareform(pdist(values[:, None], metric="euclidean"))
        row_mean = dist.mean(axis=1, keepdims=True)
        col_mean = dist.mean(axis=0, keepdims=True)
        centered = dist - row_mean - col_mean + dist.mean()
        mats.append(centered.reshape(-1))
        dvar2.append(float(np.mean(centered * centered)))
    return np.vstack(mats).astype(float), np.asarray(dvar2, dtype=float)


def two_sided_spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    try:
        corr = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    except Exception:
        corr = np.nan
    return float(corr) if corr is not None else np.nan


def pretty_model_name(model_group: str) -> str:
    return MODEL_NAME_MAP.get(str(model_group), str(model_group))


def pretty_group_name(group_name: str) -> str:
    return GROUP_NAME_MAP.get(str(group_name), str(group_name).replace("_", " ").title())


def short_feature_label(feature_name: str) -> str:
    label = str(feature_name)
    for src, dst in ACTIVITY_TOKEN_MAP.items():
        label = label.replace(src, dst)
    replacements = {
        "visual_semantic__": "vs:",
        "visual_major__": "vm:",
        "green_view__": "gv:",
        "emotion__": "em:",
        "people__": "pp:",
        "color__": "cl:",
        "ai_activity__": "ai:",
        "audio_events_dist__": "aed:",
        "audio_events_topk__": "aet:",
        "audio_events__": "ae:",
        "audio_signal__": "as:",
        "audio_embedding__": "ab:",
        "__mean": ":mean",
        "__std": ":std",
        "__var": ":var",
    }
    for src, dst in replacements.items():
        label = label.replace(src, dst)
    label = label.replace("__", ":").replace("_", " ")
    return label[:64]
