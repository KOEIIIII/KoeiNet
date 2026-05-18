


"""Leakage-safe feature screening utilities for Step-7.5 refined evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

logger = logging.getLogger("fusion.feature_screening")


@dataclass(frozen=True)
class FeatureScreeningConfig:
    """Deterministic screening config for small-sample/high-dimensional settings."""

    missing_ratio_threshold: float = 0.95
    variance_threshold: float = 1e-12
    correlation_threshold: float = 0.95
    top_k_visual: int = 20
    top_k_audio: int = 20
    top_k_early: int = 30
    min_modality_each_early: int = 8
    random_seed: int = 20260311


def _to_numeric_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series.dtype):
            out[col] = series.astype("int64")
        else:
            out[col] = pd.to_numeric(series, errors="coerce")
    return out


def _missing_ratio(series: pd.Series) -> float:
    if len(series) == 0:
        return 1.0
    return float(series.isna().mean())


def _drop_high_missing(
    x_df: pd.DataFrame,
    threshold: float,
) -> Tuple[List[str], List[str]]:
    kept: List[str] = []
    dropped: List[str] = []
    for col in x_df.columns:
        if _missing_ratio(x_df[col]) > float(threshold):
            dropped.append(str(col))
        else:
            kept.append(str(col))
    return kept, dropped


def _drop_low_variance(
    x_df: pd.DataFrame,
    cols: Sequence[str],
    threshold: float,
) -> Tuple[List[str], List[str], Dict[str, float]]:
    kept: List[str] = []
    dropped: List[str] = []
    var_map: Dict[str, float] = {}
    for col in cols:
        if col not in x_df.columns:
            continue
        s = x_df[col]
        var = float(pd.to_numeric(s, errors="coerce").var(skipna=True))
        if not np.isfinite(var):
            var = 0.0
        var_map[str(col)] = var
        if var <= float(threshold):
            dropped.append(str(col))
        else:
            kept.append(str(col))
    return kept, dropped, var_map


def _prune_correlated(
    x_df: pd.DataFrame,
    cols: Sequence[str],
    threshold: float,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    if len(cols) <= 1:
        return list(cols), []

    work = x_df[list(cols)].copy()

    for col in work.columns:
        med = pd.to_numeric(work[col], errors="coerce").median(skipna=True)
        if not np.isfinite(float(med)) if med is not None else True:
            med = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(float(med))

    corr = work.corr().abs()
    ordered = sorted([str(c) for c in cols])
    kept: List[str] = []
    dropped: List[Dict[str, Any]] = []

    for feat in ordered:
        if not kept:
            kept.append(feat)
            continue
        vec = corr.loc[feat, kept]
        if isinstance(vec, pd.Series):
            max_corr = float(vec.max()) if not vec.empty else 0.0
            if np.isfinite(max_corr) and max_corr > float(threshold):
                ref = str(vec.idxmax())
                dropped.append(
                    {
                        "feature": feat,
                        "correlated_with": ref,
                        "abs_corr": max_corr,
                    }
                )
                continue
        kept.append(feat)

    return kept, dropped


def _rank_regression_abs_spearman(
    x_df: pd.DataFrame,
    cols: Sequence[str],
    y_train: np.ndarray,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    y = np.asarray(y_train, dtype=float)
    for col in cols:
        try:
            x = pd.to_numeric(x_df[col], errors="coerce").to_numpy(dtype=float)
            corr = spearmanr(x, y, nan_policy="omit").correlation
            val = float(abs(corr)) if corr is not None and np.isfinite(corr) else 0.0
        except Exception:
            val = 0.0
        scores[str(col)] = val
    return scores


def _rank_classification_mi(
    x_df: pd.DataFrame,
    cols: Sequence[str],
    y_train: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    scores: Dict[str, float] = {str(c): 0.0 for c in cols}
    if len(cols) == 0:
        return scores
    x = x_df[list(cols)].copy()
    for col in x.columns:
        med = pd.to_numeric(x[col], errors="coerce").median(skipna=True)
        if not np.isfinite(float(med)) if med is not None else True:
            med = 0.0
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(float(med))
    try:
        mi = mutual_info_classif(
            x.to_numpy(dtype=float),
            np.asarray(y_train),
            discrete_features=False,
            random_state=int(seed),
        )
        for i, col in enumerate(cols):
            val = float(mi[i]) if i < len(mi) and np.isfinite(mi[i]) else 0.0
            scores[str(col)] = val
    except Exception as exc:
        logger.warning("classification MI ranking failed: %s", exc)
    return scores


def _sorted_ranked_features(score_map: Mapping[str, float]) -> List[Tuple[str, float]]:
    rows = [(str(k), float(v)) for k, v in score_map.items()]
    rows.sort(key=lambda kv: (-float(kv[1]), str(kv[0]).lower()))
    return rows


def _select_top_k(
    ranked: Sequence[Tuple[str, float]],
    top_k: int,
) -> List[str]:
    return [feat for feat, _ in ranked[: max(0, int(top_k))]]


def _select_top_k_balanced_early(
    ranked: Sequence[Tuple[str, float]],
    visual_set: set[str],
    audio_set: set[str],
    top_k: int,
    min_each: int,
) -> Tuple[List[str], Dict[str, Any]]:
    ranked_features = [f for f, _ in ranked]
    vis_ranked = [f for f in ranked_features if f in visual_set]
    aud_ranked = [f for f in ranked_features if f in audio_set]

    selected: List[str] = []
    meta: Dict[str, Any] = {
        "balance_enforced": False,
        "balance_note": "",
        "selected_visual_count": 0,
        "selected_audio_count": 0,
    }

    k = int(max(0, top_k))
    m = int(max(0, min_each))
    can_balance = len(vis_ranked) >= m and len(aud_ranked) >= m and k >= (2 * m)
    if can_balance:
        selected.extend(vis_ranked[:m])
        selected.extend(aud_ranked[:m])
        meta["balance_enforced"] = True
        meta["balance_note"] = "enforced_minimum_modality_quota"
    else:
        if len(vis_ranked) < m or len(aud_ranked) < m:
            meta["balance_note"] = (
                f"insufficient_features_for_balance visual={len(vis_ranked)} audio={len(aud_ranked)} min_each={m}"
            )
        elif k < (2 * m):
            meta["balance_note"] = f"top_k_too_small_for_balance top_k={k} min_each={m}"
        else:
            meta["balance_note"] = "balance_not_enforced"

    selected_set = set(selected)
    for feat in ranked_features:
        if len(selected) >= k:
            break
        if feat in selected_set:
            continue
        selected.append(feat)
        selected_set.add(feat)

    selected = selected[:k]
    meta["selected_visual_count"] = int(sum(1 for f in selected if f in visual_set))
    meta["selected_audio_count"] = int(sum(1 for f in selected if f in audio_set))
    return selected, meta


def _screen_one_group(
    *,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    task_type: str,
    candidate_features: Sequence[str],
    missing_ratio_threshold: float,
    variance_threshold: float,
    corr_threshold: float,
    top_k: int,
    ranking_seed: int,
    early_balance: bool = False,
    visual_set: set[str] | None = None,
    audio_set: set[str] | None = None,
    min_each: int = 8,
) -> Tuple[List[str], Dict[str, Any]]:
    x_num = _to_numeric_frame(x_train, candidate_features)
    initial_cols = [str(c) for c in x_num.columns]

    keep_after_missing, dropped_missing = _drop_high_missing(
        x_num,
        threshold=float(missing_ratio_threshold),
    )
    keep_after_var, dropped_low_var, var_map = _drop_low_variance(
        x_num,
        cols=keep_after_missing,
        threshold=float(variance_threshold),
    )
    keep_after_corr, dropped_corr = _prune_correlated(
        x_num,
        cols=keep_after_var,
        threshold=float(corr_threshold),
    )

    if str(task_type) == "classification":
        score_map = _rank_classification_mi(
            x_num,
            cols=keep_after_corr,
            y_train=y_train,
            seed=int(ranking_seed),
        )
        ranking_method = "mutual_info_classif"
    else:
        score_map = _rank_regression_abs_spearman(
            x_num,
            cols=keep_after_corr,
            y_train=y_train,
        )
        ranking_method = "abs_spearman"

    ranked = _sorted_ranked_features(score_map)
    balance_meta: Dict[str, Any] = {}
    if early_balance:
        selected, balance_meta = _select_top_k_balanced_early(
            ranked=ranked,
            visual_set=visual_set or set(),
            audio_set=audio_set or set(),
            top_k=int(top_k),
            min_each=int(min_each),
        )
    else:
        selected = _select_top_k(ranked=ranked, top_k=int(top_k))

    info: Dict[str, Any] = {
        "initial_feature_count": int(len(initial_cols)),
        "initial_features": initial_cols,
        "dropped_high_missing_features": sorted([str(x) for x in dropped_missing]),
        "dropped_low_variance_features": sorted([str(x) for x in dropped_low_var]),
        "low_variance_values": {k: float(v) for k, v in var_map.items()},
        "dropped_correlated_features": dropped_corr,
        "post_prune_feature_count": int(len(keep_after_corr)),
        "ranking_method": ranking_method,
        "top_ranked_features": [
            {"feature": str(f), "score": float(s)}
            for f, s in ranked[: min(200, len(ranked))]
        ],
        "selected_features": [str(x) for x in selected],
        "selected_feature_count": int(len(selected)),
    }
    info.update(balance_meta)
    return selected, info


def screen_features_for_fold(
    *,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    task_type: str,
    base_feature_groups: Mapping[str, Sequence[str]],
    config: FeatureScreeningConfig,
    ranking_seed: int,
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """
    Screen features inside one outer training fold only (no leakage).

    Returns:
    - selected feature map for refined model groups
    - detailed screening registry payload for this target/fold
    """
    visual_candidates = list(dict.fromkeys([str(c) for c in base_feature_groups.get("visual_only", [])]))
    audio_candidates = list(dict.fromkeys([str(c) for c in base_feature_groups.get("audio_only", [])]))
    early_candidates = list(
        dict.fromkeys([str(c) for c in base_feature_groups.get("early_fusion", [*visual_candidates, *audio_candidates])])
    )

    visual_set = set(visual_candidates)
    audio_set = set(audio_candidates)

    visual_selected, visual_info = _screen_one_group(
        x_train=x_train,
        y_train=y_train,
        task_type=task_type,
        candidate_features=visual_candidates,
        missing_ratio_threshold=float(config.missing_ratio_threshold),
        variance_threshold=float(config.variance_threshold),
        corr_threshold=float(config.correlation_threshold),
        top_k=int(config.top_k_visual),
        ranking_seed=int(ranking_seed + 11),
    )
    audio_selected, audio_info = _screen_one_group(
        x_train=x_train,
        y_train=y_train,
        task_type=task_type,
        candidate_features=audio_candidates,
        missing_ratio_threshold=float(config.missing_ratio_threshold),
        variance_threshold=float(config.variance_threshold),
        corr_threshold=float(config.correlation_threshold),
        top_k=int(config.top_k_audio),
        ranking_seed=int(ranking_seed + 17),
    )
    early_selected, early_info = _screen_one_group(
        x_train=x_train,
        y_train=y_train,
        task_type=task_type,
        candidate_features=early_candidates,
        missing_ratio_threshold=float(config.missing_ratio_threshold),
        variance_threshold=float(config.variance_threshold),
        corr_threshold=float(config.correlation_threshold),
        top_k=int(config.top_k_early),
        ranking_seed=int(ranking_seed + 23),
        early_balance=True,
        visual_set=visual_set,
        audio_set=audio_set,
        min_each=int(config.min_modality_each_early),
    )

    selected_map = {
        "visual_only_screened": visual_selected,
        "audio_only_screened": audio_selected,
        "early_fusion_screened": early_selected,
    }
    registry = {
        "config": {
            "missing_ratio_threshold": float(config.missing_ratio_threshold),
            "variance_threshold": float(config.variance_threshold),
            "correlation_threshold": float(config.correlation_threshold),
            "top_k_visual": int(config.top_k_visual),
            "top_k_audio": int(config.top_k_audio),
            "top_k_early": int(config.top_k_early),
            "min_modality_each_early": int(config.min_modality_each_early),
            "ranking_seed": int(ranking_seed),
            "task_type": str(task_type),
        },
        "groups": {
            "visual_only_screened": visual_info,
            "audio_only_screened": audio_info,
            "early_fusion_screened": early_info,
        },
    }
    return selected_map, registry
