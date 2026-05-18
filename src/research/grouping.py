


"""Deterministic theory-driven group construction for confirmatory relationship analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import prepare_cross_modal_feature_table


@dataclass(frozen=True)
class GroupRow:
    group_name: str
    modality: str
    constituent_feature: str
    source_group: str
    sign_direction: int
    keep_flag: bool
    reason: str


VISUAL_GROUP_ORDER = [
    "people_presence",
    "green_nature",
    "traffic_road_hardscape",
    "visual_emotion_aesthetic",
    "ai_activity",
    "visual_color",
    "visual_semantic_general",
]

AUDIO_GROUP_ORDER = [
    "audio_signal_level",
    "audio_event_human",
    "audio_event_traffic_mechanical",
    "audio_event_natural",
    "audio_event_general",
    "audio_embedding_general",
]


def _safe_numeric_series(df: pd.DataFrame, feature_name: str) -> Optional[pd.Series]:
    if feature_name not in df.columns:
        return None
    series = pd.to_numeric(df[feature_name], errors="coerce")
    return series if series.notna().any() else None


def _normalized_entropy(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    if not columns:
        return pd.Series(np.nan, index=df.index)
    mat = df.loc[:, list(columns)].to_numpy(dtype=float)
    mat = np.clip(mat, a_min=0.0, a_max=None)
    row_sum = mat.sum(axis=1, keepdims=True)
    probs = np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum > 0)
    logk = np.log(max(2, probs.shape[1]))
    entropy = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=1)
    return pd.Series(entropy / logk, index=df.index, dtype=float)


def _winsorize_series(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    if clean.notna().sum() < 5:
        return clean
    lower = float(clean.quantile(lower_q))
    upper = float(clean.quantile(upper_q))
    return clean.clip(lower=lower, upper=upper)


def _standardize_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        return pd.Series(np.zeros(len(clean), dtype=float), index=clean.index)
    return (clean - mean) / std


def _cronbach_alpha(df: pd.DataFrame) -> float:
    if df.shape[1] < 2:
        return np.nan
    values = df.to_numpy(dtype=float)
    item_vars = np.nanvar(values, axis=0, ddof=1)
    total = np.nansum(values, axis=1)
    total_var = float(np.nanvar(total, ddof=1))
    if not np.isfinite(total_var) or total_var <= 0:
        return np.nan
    n_items = df.shape[1]
    return float((n_items / (n_items - 1)) * (1.0 - np.nansum(item_vars) / total_var))


def _avg_within_group_corr(df: pd.DataFrame) -> float:
    if df.shape[1] < 2:
        return np.nan
    corr = df.corr(method="spearman").to_numpy(dtype=float)
    tri = corr[np.triu_indices_from(corr, k=1)]
    tri = tri[np.isfinite(tri)]
    return float(np.mean(tri)) if tri.size else np.nan


def _lofo_stability(composite: pd.Series, aligned_df: pd.DataFrame) -> Tuple[float, float, int]:
    if aligned_df.shape[1] < 2:
        return 1.0, 1.0, 0
    cors: List[float] = []
    for feature_name in aligned_df.columns:
        sub = aligned_df.drop(columns=[feature_name])
        lofo = sub.mean(axis=1)
        cors.append(float(composite.corr(lofo, method="pearson")))
    if not cors:
        return 1.0, 1.0, 0
    return float(np.nanmean(cors)), float(np.nanmin(cors)), int(len(cors))


def _relation_feature_registry(
    model_df: pd.DataFrame,
    feature_dict: Mapping[str, Any],
    existing_feature_registry: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if existing_feature_registry is not None and not existing_feature_registry.empty:
        cols = ["feature_name", "modality", "source_group", "missing_ratio", "n_unique_non_na", "kept_or_dropped", "reason"]
        registry = existing_feature_registry.copy()
        for col in cols:
            if col not in registry.columns:
                raise ValueError(f"relationship feature_registry missing required column: {col}")
        return registry[cols].copy()
    prepared = prepare_cross_modal_feature_table(model_df=model_df, feature_dict=feature_dict)
    return prepared["feature_registry"].copy()


def _add_row(rows: List[Dict[str, Any]], **kwargs: Any) -> None:
    rows.append(dict(kwargs))


def _mean_cols(feature_names: Sequence[str]) -> List[str]:
    return [x for x in feature_names if x.endswith("__mean")]


def _feature_pool_map(registry_df: pd.DataFrame) -> Dict[str, List[str]]:
    kept = registry_df[registry_df["kept_or_dropped"] == "kept"].copy()
    out: Dict[str, List[str]] = {}
    for modality in ["visual", "audio"]:
        out[modality] = kept[kept["modality"] == modality]["feature_name"].tolist()
    return out


def build_group_artifacts(
    *,
    model_df: pd.DataFrame,
    feature_dict: Mapping[str, Any],
    existing_feature_registry: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    registry_df = _relation_feature_registry(
        model_df=model_df,
        feature_dict=feature_dict,
        existing_feature_registry=existing_feature_registry,
    )
    kept_registry = registry_df[registry_df["kept_or_dropped"] == "kept"].copy()
    kept_features = kept_registry["feature_name"].tolist()
    pool_map = _feature_pool_map(registry_df)

    derived_series: Dict[str, pd.Series] = {}
    semantic_mean_cols = [c for c in kept_features if c.startswith("visual_semantic__") and c.endswith("__mean")]
    major_mean_cols = [c for c in kept_features if c.startswith("visual_major__") and c.endswith("__mean")]
    audio_event_dist_cols = [c for c in kept_features if c.startswith("audio_events_dist__")]
    if semantic_mean_cols:
        derived_series["derived::visual_semantic_entropy"] = _normalized_entropy(model_df, semantic_mean_cols)
    if major_mean_cols:
        derived_series["derived::visual_major_entropy"] = _normalized_entropy(model_df, major_mean_cols)
    if audio_event_dist_cols:
        derived_series["derived::audio_event_entropy"] = _normalized_entropy(model_df, audio_event_dist_cols)

    group_candidate_rows: List[Dict[str, Any]] = []
    group_feature_map: Dict[str, List[Tuple[str, str, int]]] = {}

    def register(group_name: str, modality: str, feature_name: str, source_group: str, sign: int, reason: str = "candidate") -> None:
        keep_flag = bool(feature_name)
        _add_row(
            group_candidate_rows,
            group_name=group_name,
            modality=modality,
            constituent_feature=feature_name,
            source_group=source_group,
            sign_direction=sign if keep_flag else "",
            keep_flag=keep_flag,
            reason=reason,
        )
        if keep_flag:
            group_feature_map.setdefault(group_name, []).append((feature_name, source_group, int(sign)))

    visual_pool = pool_map["visual"]
    audio_pool = pool_map["audio"]

    for feature_name in visual_pool:
        if feature_name in {
            "visual_semantic__person__mean",
            "visual_semantic__rider__mean",
            "visual_major__human__mean",
        } or (
            feature_name.startswith("people__")
            and (feature_name.endswith("__mean") or feature_name.endswith("__max"))
        ):
            register("people_presence", "visual", feature_name, "people_or_human", +1)

        if feature_name in {
            "visual_major__nature__mean",
            "visual_semantic__vegetation__mean",
        }:
            register("green_nature", "visual", feature_name, "green_or_nature", +1)

        if feature_name in {
            "visual_semantic__road__mean",
            "visual_semantic__sidewalk__mean",
            "visual_semantic__traffic_light__mean",
            "visual_semantic__traffic_sign__mean",
            "visual_semantic__car__mean",
            "visual_semantic__bus__mean",
            "visual_semantic__truck__mean",
            "visual_semantic__train__mean",
            "visual_semantic__motorcycle__mean",
            "visual_semantic__bicycle__mean",
            "visual_major__construction__mean",
            "visual_major__vehicle__mean",
        }:
            register("traffic_road_hardscape", "visual", feature_name, "traffic_or_hardscape", +1)

        emotion_sign_map = {
            "emotion__beautiful__mean": +1,
            "emotion__lively__mean": +1,
            "emotion__boring__mean": -1,
            "emotion__depressing__mean": -1,
        }
        if feature_name in emotion_sign_map:
            register("visual_emotion_aesthetic", "visual", feature_name, "emotion", emotion_sign_map[feature_name])

        if feature_name.startswith("ai_activity__") and (
            feature_name.endswith("_score__mean")
            or feature_name.endswith("_score__max")
            or feature_name.endswith("_mean_score")
            or feature_name.endswith("_max_score")
            or feature_name.endswith("_is_suitable")
        ):
            register("ai_activity", "visual", feature_name, "ai_activity", +1)

        color_sign_map = {
            "color__accent__mean": +1,
            "color__brick_red__mean": +1,
            "color__natural__mean": +1,
            "color__gray__mean": -1,
            "color__neutral__mean": -1,
        }
        if feature_name in color_sign_map:
            register("visual_color", "visual", feature_name, "color", color_sign_map[feature_name])

    if "derived::visual_semantic_entropy" in derived_series:
        register("visual_semantic_general", "visual", "derived::visual_semantic_entropy", "derived", +1, reason="derived_entropy")
    if "derived::visual_major_entropy" in derived_series:
        register("visual_semantic_general", "visual", "derived::visual_major_entropy", "derived", +1, reason="derived_entropy")

    for feature_name in audio_pool:
        if feature_name in {
            "audio_signal__loudness_proxy_db",
            "audio_signal__rms_energy",
            "audio_signal__spectral_flux",
        }:
            register("audio_signal_level", "audio", feature_name, "audio_signal", +1)

        human_event_features = {
            "audio_events__group_ratio_human",
            "audio_events_dist__speech",
            "audio_events_dist__conversation",
            "audio_events_dist__chatter",
            "audio_events_dist__crowd",
            "audio_events_dist__laughter",
            "audio_events_dist__hubbub_speech_noise_speech_babble",
            "audio_events_dist__child_speech_kid_speaking",
            "audio_events_dist__singing",
            "audio_events_dist__run",
        }
        if feature_name in human_event_features:
            register("audio_event_human", "audio", feature_name, "audio_events", +1)

        traffic_mech_features = {
            "audio_events__group_ratio_traffic",
            "audio_events__group_ratio_mechanical",
            "audio_events_dist__vehicle",
            "audio_events_dist__bus",
            "audio_events_dist__motor_vehicle_road",
            "audio_events_dist__noise",
            "audio_events_dist__outside_urban_or_manmade",
        }
        if feature_name in traffic_mech_features:
            register("audio_event_traffic_mechanical", "audio", feature_name, "audio_events", +1)

        natural_features = {
            "audio_events__group_ratio_nature",
            "audio_events_dist__animal",
            "audio_events_dist__cat",
            "audio_events_dist__horse",
            "audio_events_dist__domestic_animals_pets",
            "audio_events_dist__outside_rural_or_natural",
            "audio_events_dist__rain",
            "audio_events_dist__rain_on_surface",
        }
        if feature_name in natural_features:
            register("audio_event_natural", "audio", feature_name, "audio_events", +1)

        if feature_name in {
            "audio_events_topk_known_count",
            "audio_events__group_ratio_other",
        }:
            register(
                "audio_event_general",
                "audio",
                feature_name,
                "audio_events",
                -1 if feature_name == "audio_events__group_ratio_other" else +1,
            )

        if feature_name in {
            "audio_embedding__panns_emb_l2",
            "audio_embedding__panns_emb_std",
        }:
            register("audio_embedding_general", "audio", feature_name, "audio_embedding", +1)

    if "derived::audio_event_entropy" in derived_series:
        register("audio_event_general", "audio", "derived::audio_event_entropy", "derived", +1, reason="derived_entropy")

    group_definition_rows: List[GroupRow] = []
    diagnostics_rows: List[Dict[str, Any]] = []
    composite_df = model_df[["segment_id"]].copy()
    retained_groups: Dict[str, Dict[str, Any]] = {}

    def _value_for_feature(feature_name: str) -> Optional[pd.Series]:
        if feature_name.startswith("derived::"):
            return derived_series.get(feature_name)
        return _safe_numeric_series(model_df, feature_name)

    all_groups = VISUAL_GROUP_ORDER + AUDIO_GROUP_ORDER
    group_modality_map = {**{g: "visual" for g in VISUAL_GROUP_ORDER}, **{g: "audio" for g in AUDIO_GROUP_ORDER}}

    for group_name in all_groups:
        modality = group_modality_map[group_name]
        members = group_feature_map.get(group_name, [])
        if not members:
            group_definition_rows.append(
                GroupRow(
                    group_name=group_name,
                    modality=modality,
                    constituent_feature="",
                    source_group="",
                    sign_direction=0,
                    keep_flag=False,
                    reason="no_theory_consistent_features_identified",
                )
            )
            continue

        aligned_features: Dict[str, pd.Series] = {}
        reasons: Dict[str, str] = {}
        for feature_name, source_group, sign in members:
            series = _value_for_feature(feature_name)
            if series is None:
                reasons[feature_name] = "missing_feature_values"
                continue
            missing_ratio = float(series.isna().mean())
            if missing_ratio > 0.30:
                reasons[feature_name] = "missing_ratio_gt_0.30"
                continue
            if series.dropna().nunique() <= 1 or float(series.dropna().std(ddof=0) if series.dropna().size else 0.0) <= 1e-8:
                reasons[feature_name] = "near_zero_variance"
                continue
            processed = _winsorize_series(series)
            standardized = _standardize_series(processed) * float(sign)
            aligned_features[feature_name] = standardized
            reasons[feature_name] = "kept"

        deduped: Dict[Tuple[float, ...], str] = {}
        final_features: Dict[str, pd.Series] = {}
        for feature_name, aligned in aligned_features.items():
            key = tuple(np.round(aligned.fillna(0.0).to_numpy(dtype=float), 10).tolist())
            if key in deduped:
                reasons[feature_name] = f"duplicate_of:{deduped[key]}"
                continue
            deduped[key] = feature_name
            final_features[feature_name] = aligned

        for feature_name, source_group, sign in members:
            group_definition_rows.append(
                GroupRow(
                    group_name=group_name,
                    modality=modality,
                    constituent_feature=feature_name,
                    source_group=source_group,
                    sign_direction=sign,
                    keep_flag=bool(reasons.get(feature_name) == "kept"),
                    reason=reasons.get(feature_name, "not_evaluated"),
                )
            )

        if not final_features:
            group_definition_rows.append(
                GroupRow(
                    group_name=group_name,
                    modality=modality,
                    constituent_feature="",
                    source_group="",
                    sign_direction=0,
                    keep_flag=False,
                    reason="all_candidate_features_removed_after_screening",
                )
            )
            continue

        aligned_df = pd.DataFrame(final_features)
        composite = aligned_df.mean(axis=1)
        composite_df[group_name] = composite.to_numpy(dtype=float)

        loo_mean_corr, loo_min_corr, loo_variants = _lofo_stability(composite=composite, aligned_df=aligned_df)
        diagnostics_rows.append(
            {
                "group_name": group_name,
                "modality": modality,
                "n_features": int(aligned_df.shape[1]),
                "single_feature_group": bool(aligned_df.shape[1] == 1),
                "constituent_features": "|".join(aligned_df.columns.tolist()),
                "missingness_mean": float(np.mean([float(_value_for_feature(f).isna().mean()) for f in aligned_df.columns])),
                "avg_within_group_correlation": _avg_within_group_corr(aligned_df),
                "cronbach_alpha": _cronbach_alpha(aligned_df),
                "lofo_mean_corr": loo_mean_corr,
                "lofo_min_corr": loo_min_corr,
                "lofo_variants": int(loo_variants),
                "composite_sd": float(np.nanstd(composite.to_numpy(dtype=float), ddof=1)),
            }
        )
        retained_groups[group_name] = {
            "modality": modality,
            "aligned_df": aligned_df,
            "constituents": [
                {"feature_name": col, "sign_direction": int(next(sign for fn, _, sign in members if fn == col))}
                for col in aligned_df.columns
            ],
        }

    group_definition_df = pd.DataFrame([row.__dict__ for row in group_definition_rows])
    diagnostics_df = pd.DataFrame(diagnostics_rows).sort_values(["modality", "group_name"]).reset_index(drop=True)

    visual_groups = [g for g in VISUAL_GROUP_ORDER if g in retained_groups]
    audio_groups = [g for g in AUDIO_GROUP_ORDER if g in retained_groups]

    return {
        "group_definition_registry": group_definition_df,
        "group_composites": composite_df,
        "group_composite_diagnostics": diagnostics_df,
        "retained_groups": retained_groups,
        "visual_groups": visual_groups,
        "audio_groups": audio_groups,
        "feature_registry_used": registry_df,
    }
