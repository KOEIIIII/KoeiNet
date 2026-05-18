


"""Aggregate segment-level evidence into episode-level structured records."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .export_utils import coalesce_texts, mode_or_first, split_semicolon_values

logger = logging.getLogger("deliverable.episode_evidence")


def _mean_numeric(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return np.nan
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def _safe_quantile(df: pd.DataFrame, column: str, q: float, default: float) -> float:
    if column not in df.columns or df.empty:
        return float(default)
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return float(default)
    return float(series.quantile(float(q)))


def _band_from_confidence(score: float) -> str:
    if not np.isfinite(score):
        return "medium"
    if score >= 0.88 or score >= 5.6:
        return "high"
    if score >= 0.72 or score >= 4.2:
        return "medium"
    return "low"


def _strength_from_count(n_items: int, band: str) -> str:
    if n_items >= 3 and band == "high":
        return "high"
    if n_items >= 2:
        return "medium"
    return "low"


class EpisodeEvidenceAssembler:
    """Deterministic evidence assembler for problem episodes."""

    def __init__(
        self,
        *,
        ranking_df: pd.DataFrame,
        model_df: pd.DataFrame,
        audio_df: pd.DataFrame,
        validation_df: Optional[pd.DataFrame],
        design_plan_map: Mapping[int, Mapping[str, Any]],
        edit_prompt_map: Mapping[int, Mapping[str, Any]],
        intervention_df: pd.DataFrame,
        profile_map: Mapping[int, Mapping[str, Any]],
        diagnosis_map: Mapping[int, Mapping[str, Any]],
        critic_map: Mapping[int, Mapping[str, Any]],
        step8_evidence_registry: Optional[Mapping[str, Any]] = None,
        proof_claim_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.ranking_df = ranking_df.copy()
        self.ranking_df["segment_id"] = pd.to_numeric(self.ranking_df["segment_id"], errors="coerce").astype(int)
        self.rank_map = self.ranking_df.set_index("segment_id").to_dict(orient="index")

        self.model_df = model_df.copy()
        self.model_df["segment_id"] = pd.to_numeric(self.model_df["segment_id"], errors="coerce").astype(int)
        self.model_df = self.model_df.set_index("segment_id", drop=False)

        self.audio_df = audio_df.copy()
        self.audio_df["segment_id"] = pd.to_numeric(self.audio_df["segment_id"], errors="coerce").astype(int)
        self.audio_df = self.audio_df.set_index("segment_id", drop=False)

        self.validation_df = None
        if validation_df is not None and not validation_df.empty:
            tmp = validation_df.copy()
            tmp["segment_id"] = pd.to_numeric(tmp["segment_id"], errors="coerce")
            tmp = tmp.dropna(subset=["segment_id"]).copy()
            tmp["segment_id"] = tmp["segment_id"].astype(int)
            self.validation_df = tmp.set_index("segment_id", drop=False)

        self.design_plan_map = {int(k): dict(v) for k, v in design_plan_map.items()}
        self.edit_prompt_map = {int(k): dict(v) for k, v in edit_prompt_map.items()}
        self.intervention_df = intervention_df.copy()
        if not self.intervention_df.empty:
            self.intervention_df["segment_id"] = pd.to_numeric(self.intervention_df["segment_id"], errors="coerce").astype(int)
            self.intervention_df = self.intervention_df.set_index("segment_id", drop=False)
        self.profile_map = {int(k): dict(v) for k, v in profile_map.items()}
        self.diagnosis_map = {int(k): dict(v) for k, v in diagnosis_map.items()}
        self.critic_map = {int(k): dict(v) for k, v in critic_map.items()}
        self.step8_evidence_registry = dict(step8_evidence_registry or {})
        self.proof_claim_df = proof_claim_df.copy() if proof_claim_df is not None else pd.DataFrame()
        self.thresholds = self._compute_thresholds()

    @staticmethod
    def _hardscape_series(df: pd.DataFrame) -> pd.Series:
        cols = [
            "visual_major__construction__mean",
            "visual_semantic__road__mean",
            "visual_semantic__sidewalk__mean",
            "visual_semantic__wall__mean",
            "visual_semantic__fence__mean",
            "visual_semantic__pole__mean",
            "visual_semantic__traffic_light__mean",
            "visual_semantic__traffic_sign__mean",
        ]
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.Series(dtype=float)
        return df[present].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    @staticmethod
    def _vehicle_series(df: pd.DataFrame) -> pd.Series:
        cols = [
            "visual_major__vehicle__mean",
            "visual_semantic__car__mean",
            "visual_semantic__truck__mean",
            "visual_semantic__bus__mean",
            "visual_semantic__motorcycle__mean",
        ]
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.Series(dtype=float)
        return df[present].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    @staticmethod
    def _traffic_mechanical_series(df: pd.DataFrame) -> pd.Series:
        traffic = pd.to_numeric(df.get("audio_events__group_ratio_traffic"), errors="coerce")
        mechanical = pd.to_numeric(df.get("audio_events__group_ratio_mechanical"), errors="coerce")
        if traffic is None:
            traffic = pd.Series(dtype=float)
        if mechanical is None:
            mechanical = pd.Series(dtype=float)
        if traffic.empty and mechanical.empty:
            return pd.Series(dtype=float)
        if traffic.empty:
            return mechanical.fillna(0.0)
        if mechanical.empty:
            return traffic.fillna(0.0)
        return traffic.fillna(0.0) + mechanical.fillna(0.0)

    def _compute_thresholds(self) -> Dict[str, float]:
        temp = self.model_df.reset_index(drop=True)
        hardscape = self._hardscape_series(temp)
        vehicle = self._vehicle_series(temp)
        traffic = self._traffic_mechanical_series(temp)
        nature = pd.to_numeric(temp.get("audio_events__group_ratio_nature"), errors="coerce")
        loudness = pd.to_numeric(temp.get("audio_signal__loudness_proxy_db"), errors="coerce")
        return {
            "people_high": max(18.0, _safe_quantile(temp, "people__total_people__mean", 0.70, 18.0)),
            "green_low": min(0.28, _safe_quantile(temp, "green_view__greenviewindex__mean", 0.40, 0.28)),
            "hardscape_high": float(hardscape.quantile(0.70)) if not hardscape.dropna().empty else 0.35,
            "vehicle_high": float(vehicle.quantile(0.70)) if not vehicle.dropna().empty else 0.10,
            "beautiful_low": _safe_quantile(temp, "emotion__beautiful__mean", 0.30, 0.55),
            "boring_high": _safe_quantile(temp, "emotion__boring__mean", 0.70, 0.55),
            "depressing_high": _safe_quantile(temp, "emotion__depressing__mean", 0.70, 0.50),
            "human_high": _safe_quantile(temp, "audio_events__group_ratio_human", 0.70, 0.65),
            "traffic_high": float(traffic.quantile(0.70)) if not traffic.dropna().empty else 0.08,
            "nature_low": float(nature.quantile(0.35)) if nature is not None and not nature.dropna().empty else 0.01,
            "loudness_high": float(loudness.quantile(0.70)) if loudness is not None and not loudness.dropna().empty else -24.0,
        }

    def _validation_summary(self, segment_ids: Sequence[int]) -> Tuple[str, Dict[str, Any]]:
        if self.validation_df is None:
            return "No adjudicated validation labels available for this episode.", {}
        sub = self.validation_df[self.validation_df["segment_id"].isin(list(segment_ids))].copy()
        if sub.empty:
            return "No adjudicated validation labels available for the selected segments.", {}
        top_labels = sub["primary_problem_label"].fillna("").astype(str).value_counts().head(3).index.tolist()
        metrics = {
            "comfort_score": _mean_numeric(sub, "comfort_score"),
            "vitality_score": _mean_numeric(sub, "vitality_score"),
            "soundscape_pleasantness": _mean_numeric(sub, "soundscape_pleasantness"),
            "soundscape_eventfulness": _mean_numeric(sub, "soundscape_eventfulness"),
            "overall_problem_severity": _mean_numeric(sub, "overall_problem_severity"),
            "confidence_score": _mean_numeric(sub, "confidence_score"),
        }
        parts = [
            f"comfort={metrics['comfort_score']:.2f}" if np.isfinite(metrics["comfort_score"]) else None,
            f"vitality={metrics['vitality_score']:.2f}" if np.isfinite(metrics["vitality_score"]) else None,
            f"pleasantness={metrics['soundscape_pleasantness']:.2f}" if np.isfinite(metrics["soundscape_pleasantness"]) else None,
            f"eventfulness={metrics['soundscape_eventfulness']:.2f}" if np.isfinite(metrics["soundscape_eventfulness"]) else None,
            f"severity={metrics['overall_problem_severity']:.2f}" if np.isfinite(metrics["overall_problem_severity"]) else None,
            f"primary_label={'/'.join(top_labels[:2])}" if top_labels else None,
        ]
        return "; ".join([p for p in parts if p]), metrics

    def _confidence_band(self, segment_ids: Sequence[int], validation_metrics: Mapping[str, Any]) -> str:
        critic_scores: List[float] = []
        for segment_id in segment_ids:
            critic_json = self.critic_map.get(int(segment_id), {}).get("critic_json", {})
            if isinstance(critic_json, Mapping):
                score = pd.to_numeric(critic_json.get("confidence_score"), errors="coerce")
                if np.isfinite(score):
                    critic_scores.append(float(score))
        if critic_scores:
            return _band_from_confidence(float(np.mean(critic_scores)))
        val_conf = pd.to_numeric(validation_metrics.get("confidence_score"), errors="coerce")
        return _band_from_confidence(float(val_conf) if np.isfinite(val_conf) else np.nan)

    def _profile_summary(self, rep_segment_id: int, segment_ids: Sequence[int]) -> str:
        rep_profile = self.profile_map.get(int(rep_segment_id), {}).get("profile_json", {})
        if isinstance(rep_profile, Mapping):
            concise = str(rep_profile.get("concise_summary", "")).strip()
            if concise:
                return concise
        facts: List[str] = []
        for segment_id in segment_ids:
            profile_json = self.profile_map.get(int(segment_id), {}).get("profile_json", {})
            if isinstance(profile_json, Mapping):
                facts.extend([str(x).strip() for x in profile_json.get("visual_facts", [])[:2] if str(x).strip()])
                facts.extend([str(x).strip() for x in profile_json.get("audio_facts", [])[:1] if str(x).strip()])
        return " / ".join(facts[:4])

    def _diagnosis_summary(self, rep_segment_id: int, segment_ids: Sequence[int]) -> Tuple[str, List[str]]:
        labels: List[str] = []
        reasons: List[str] = []
        for segment_id in segment_ids:
            diagnosis_json = self.diagnosis_map.get(int(segment_id), {}).get("diagnosis_json", {})
            if not isinstance(diagnosis_json, Mapping):
                continue
            labels.extend([str(x).strip() for x in diagnosis_json.get("problem_labels", []) if str(x).strip()])
            reason = str(diagnosis_json.get("cross_modal_reason", "")).strip()
            if reason:
                reasons.append(reason)
        top_labels = pd.Series(labels).value_counts().head(4).index.tolist() if labels else []
        rep_reason = ""
        rep_diagnosis = self.diagnosis_map.get(int(rep_segment_id), {}).get("diagnosis_json", {})
        if isinstance(rep_diagnosis, Mapping):
            rep_reason = str(rep_diagnosis.get("cross_modal_reason", "")).strip()
        summary = "; ".join(top_labels[:3]) if top_labels else "No diagnosis labels available."
        if rep_reason:
            summary = f"{summary}. {rep_reason}"
        elif reasons:
            summary = f"{summary}. {reasons[0]}"
        return summary, top_labels

    def _soundscape_state_mode(self, segment_ids: Sequence[int]) -> Dict[str, Any]:
        states: List[Mapping[str, Any]] = []
        for segment_id in segment_ids:
            state = self.design_plan_map.get(int(segment_id), {}).get("soundscape_state", {})
            if isinstance(state, Mapping):
                states.append(state)
        return {
            "eventfulness_state": mode_or_first([s.get("eventfulness_state") for s in states], default="balanced"),
            "pleasantness_support_level": mode_or_first([s.get("pleasantness_support_level") for s in states], default="unknown"),
            "dominant_sources": coalesce_texts([s.get("dominant_sources", []) for s in states], limit=4),
        }

    def _visual_tags_and_summary(self, metrics: Mapping[str, float], diagnosis_labels: Sequence[str]) -> Tuple[List[str], str, List[str]]:
        tags: List[str] = []
        evidence: List[str] = []
        people_mean = float(metrics.get("people__total_people__mean", np.nan))
        green = float(metrics.get("green_view__greenviewindex__mean", np.nan))
        hardscape = float(metrics.get("hardscape_index", np.nan))
        vehicle = float(metrics.get("vehicle_index", np.nan))
        road = float(metrics.get("visual_semantic__road__mean", np.nan))
        sidewalk = float(metrics.get("visual_semantic__sidewalk__mean", np.nan))
        beautiful = float(metrics.get("emotion__beautiful__mean", np.nan))
        boring = float(metrics.get("emotion__boring__mean", np.nan))
        depressing = float(metrics.get("emotion__depressing__mean", np.nan))

        if np.isfinite(people_mean) and people_mean >= self.thresholds["people_high"]:
            tags.append("crowding")
            evidence.append(f"mean_people={people_mean:.1f}")
        if np.isfinite(green) and green <= self.thresholds["green_low"]:
            tags.append("low_green_view")
            evidence.append(f"green_view={green:.3f}")
        if np.isfinite(hardscape) and hardscape >= self.thresholds["hardscape_high"]:
            tags.append("high_hardscape")
            evidence.append(f"hardscape_index={hardscape:.3f}")
        if np.isfinite(vehicle) and vehicle >= self.thresholds["vehicle_high"]:
            tags.append("vehicle_dominance")
            evidence.append(f"vehicle_index={vehicle:.3f}")
        if np.isfinite(road) and np.isfinite(sidewalk) and road > sidewalk + 0.04:
            tags.append("poor_walkability_cues")
            evidence.append(f"road_vs_sidewalk={road:.3f}/{sidewalk:.3f}")
        if (
            (np.isfinite(beautiful) and beautiful <= self.thresholds["beautiful_low"])
            or (np.isfinite(boring) and boring >= self.thresholds["boring_high"])
            or (np.isfinite(depressing) and depressing >= self.thresholds["depressing_high"])
        ):
            tags.append("low_aesthetic_quality")
            evidence.append(f"beautiful={beautiful:.3f}; boring={boring:.3f}; depressing={depressing:.3f}")

        diagnosis_text = " ".join([str(x).lower() for x in diagnosis_labels])
        if "green" in diagnosis_text and "low_green_view" not in tags:
            tags.append("low_green_view")
        if "crowd" in diagnosis_text and "crowding" not in tags:
            tags.append("crowding")
        if "visual clutter" in diagnosis_text and "low_aesthetic_quality" not in tags:
            tags.append("low_aesthetic_quality")
        tags = tags[:5]

        if "low_green_view" in tags and "high_hardscape" in tags:
            summary = "视觉上绿量偏低、硬质界面偏强，街道缓冲与舒适性不足。"
        elif "crowding" in tags and "vehicle_dominance" in tags:
            summary = "视觉上人流密集且车辆存在感较强，步行空间秩序与安全边界不够清晰。"
        elif "crowding" in tags:
            summary = "视觉上人流较密，街道使用压力偏高。"
        elif "low_aesthetic_quality" in tags:
            summary = "视觉上审美质量与空间秩序感偏弱。"
        elif tags:
            summary = "视觉层面存在可见的环境缓冲、秩序或步行友好性问题。"
        else:
            summary = "视觉问题不算极端，但街道环境的缓冲与品质支撑仍然有限。"
        return tags, summary, evidence

    def _sound_tags_and_summary(
        self,
        metrics: Mapping[str, float],
        validation_metrics: Mapping[str, Any],
        soundscape_state: Mapping[str, Any],
    ) -> Tuple[List[str], str, List[str]]:
        tags: List[str] = []
        evidence: List[str] = []
        human = float(metrics.get("audio_events__group_ratio_human", np.nan))
        traffic = float(metrics.get("audio_events__group_ratio_traffic", np.nan))
        mechanical = float(metrics.get("audio_events__group_ratio_mechanical", np.nan))
        nature = float(metrics.get("audio_events__group_ratio_nature", np.nan))
        loudness = float(metrics.get("audio_signal__loudness_proxy_db", np.nan))
        pleasantness = float(validation_metrics.get("soundscape_pleasantness", np.nan))
        eventfulness = float(validation_metrics.get("soundscape_eventfulness", np.nan))
        traffic_mechanical = sum([x for x in [traffic, mechanical] if np.isfinite(x)])

        if np.isfinite(traffic_mechanical) and traffic_mechanical >= self.thresholds["traffic_high"]:
            tags.append("traffic_mechanical_dominant")
            evidence.append(f"traffic_mechanical_ratio={traffic_mechanical:.3f}")
        if np.isfinite(loudness) and loudness >= self.thresholds["loudness_high"]:
            tags.append("high_loudness")
            evidence.append(f"loudness_proxy_db={loudness:.2f}")
        if np.isfinite(nature) and nature <= self.thresholds["nature_low"]:
            tags.append("low_natural_sound")
            evidence.append(f"nature_ratio={nature:.3f}")
        if np.isfinite(human) and human >= self.thresholds["human_high"]:
            tags.append("human_voice_dominant")
            evidence.append(f"human_ratio={human:.3f}")
        if (
            np.isfinite(pleasantness)
            and pleasantness <= 3.5
            and ("high_loudness" in tags or "traffic_mechanical_dominant" in tags)
        ):
            if "noisy_but_low_pleasantness" not in tags:
                tags.insert(0, "noisy_but_low_pleasantness")
        if np.isfinite(eventfulness) and eventfulness >= 6.0:
            tags.append("high_eventfulness")
        if str(soundscape_state.get("eventfulness_state", "")).strip() == "understimulated":
            tags.append("low_eventfulness")
        tags = tags[:5]

        if "noisy_but_low_pleasantness" in tags:
            summary = "声景偏响且愉悦度偏低，存在明确的听觉压力。"
        elif "traffic_mechanical_dominant" in tags and "low_natural_sound" in tags:
            summary = "声景缺少自然声缓冲，交通或机械声暴露偏高。"
        elif "human_voice_dominant" in tags and "high_loudness" in tags:
            summary = "声景以人声活动为主，但整体响度偏高，容易形成持续听觉负担。"
        elif "human_voice_dominant" in tags and "low_natural_sound" in tags:
            summary = "声景主要由人声构成，缺少自然声层次与环境缓冲。"
        elif "low_eventfulness" in tags:
            summary = "声景并不压迫，但活力层次偏弱，缺少更细腻的公共生活声线索。"
        elif tags:
            summary = "声景问题主要体现为缓冲不足、层次单一或响度偏高。"
        else:
            summary = "声景没有极端负面暴露，但自然声与柔性缓冲仍较有限。"
        return tags, summary, evidence

    @staticmethod
    def _fusion_tags_and_summary(
        visual_tags: Sequence[str],
        sound_tags: Sequence[str],
        consistency_flag: str,
    ) -> Tuple[List[str], str]:
        tags: List[str] = []
        if "crowding" in visual_tags and ("traffic_mechanical_dominant" in sound_tags or "high_loudness" in sound_tags):
            tags.append("crowded_and_noise_dominant")
        if "low_green_view" in visual_tags and ("low_natural_sound" in sound_tags or "traffic_mechanical_dominant" in sound_tags):
            tags.append("low_greenery_with_high_mechanical_noise")
        if "crowding" in visual_tags and "human_voice_dominant" in sound_tags and "high_loudness" in sound_tags:
            tags.append("active_but_acoustically_harsh")
        if not tags and sound_tags and consistency_flag in {"mixed", "consistent"}:
            tags.append("visually_tolerable_but_acoustically_stressful")
        tags = tags[:3]

        if "low_greenery_with_high_mechanical_noise" in tags:
            summary = "视觉上的低绿量与声景中的缓冲不足相互叠加，使整体环境偏硬、偏干。"
        elif "crowded_and_noise_dominant" in tags:
            summary = "视觉上的高使用压力与听觉上的持续刺激共同推高了步行负担。"
        elif "active_but_acoustically_harsh" in tags:
            summary = "空间活跃度较高，但听觉体验偏粗糙，缺少舒缓与分层。"
        elif tags:
            summary = "跨模态证据表明，问题不只来自单一视觉或单一听觉，而是两者共同构成了体验压力。"
        else:
            summary = "跨模态层面主要表现为环境缓冲不足与公共活动组织不够均衡。"
        return tags, summary

    def _intervention_payload(self, segment_ids: Sequence[int], rep_segment_id: int) -> Dict[str, Any]:
        theme_values = [self.rank_map.get(int(sid), {}).get("recommended_intervention_theme", "") for sid in segment_ids]
        theme = mode_or_first(
            theme_values,
            default=self.rank_map.get(int(rep_segment_id), {}).get("recommended_intervention_theme", "mixed_rebalancing"),
        )
        actions: List[str] = []
        forbidden: List[str] = []
        for segment_id in segment_ids:
            plan = self.design_plan_map.get(int(segment_id), {})
            actions.extend([str(x).strip() for x in plan.get("allowed_interventions", []) if str(x).strip()])
            forbidden.extend([str(x).strip() for x in plan.get("forbidden_changes", []) if str(x).strip()])
        dedup_actions: List[str] = []
        for action in actions:
            if action not in dedup_actions:
                dedup_actions.append(action)
        dedup_forbidden: List[str] = []
        for item in forbidden:
            if item not in dedup_forbidden:
                dedup_forbidden.append(item)

        keep_map = {
            "preserve_positive_human_activity": "保留已有的正向人类活动与街道生活线索",
            "protect_existing_greenery": "保留现有绿化与树木缓冲",
            "maintain_accessibility": "保持行人与慢行通达性",
            "support_natural_sound_buffer": "保留或补强有助于自然声缓冲的要素",
            "support_cultural_liveliness": "保留场地内已有的积极文化/社交活力",
        }
        avoid_map = {
            "avoid_overactivation": "避免通过夸张增人、增店招或噪声源制造过度刺激",
            "prioritize_calmness": "避免增加新的噪声暴露或混乱边界",
        }
        keep_items: List[str] = []
        avoid_items: List[str] = []
        if not self.intervention_df.empty:
            subset = self.intervention_df[self.intervention_df["segment_id"].isin(list(segment_ids))].copy()
            for column, label in keep_map.items():
                if column in subset.columns and subset[column].fillna(0).astype(float).mean() > 0:
                    keep_items.append(label)
            for column, label in avoid_map.items():
                if column in subset.columns and subset[column].fillna(0).astype(float).mean() > 0:
                    avoid_items.append(label)
        for item in dedup_forbidden:
            if item not in avoid_items:
                avoid_items.append(item)
        return {
            "intervention_theme": theme,
            "intervention_actions": dedup_actions[:8],
            "must_keep_elements": keep_items[:5],
            "must_avoid_elements": avoid_items[:8],
        }

    def _proof_snapshot(self, segment_ids: Sequence[int]) -> str:
        if self.proof_claim_df.empty:
            return ""
        focus_targets: List[str] = []
        for segment_id in segment_ids:
            focus_targets.extend(
                [str(x).strip() for x in self.design_plan_map.get(int(segment_id), {}).get("confirmatory_target_focus", []) if str(x).strip()]
            )
        focus_targets = [x for x in dict.fromkeys(focus_targets)]
        if not focus_targets:
            return ""
        subset = self.proof_claim_df[self.proof_claim_df["target_name"].isin(focus_targets)].copy()
        if subset.empty:
            return ""
        return "; ".join([f"{row['target_name']}={row['claim_label']}" for _, row in subset.iterrows()])

    def build(self, episodes_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        rows: List[Dict[str, Any]] = []
        jsonl_rows: List[Dict[str, Any]] = []
        for _, episode in episodes_df.iterrows():
            segment_ids = [int(x) for x in episode["segment_ids"]]
            rep_segment_id = int(episode["representative_segment_id"])
            ranking_subset = self.ranking_df[self.ranking_df["segment_id"].isin(segment_ids)].copy()
            model_subset = self.model_df[self.model_df["segment_id"].isin(segment_ids)].copy()
            validation_text, validation_metrics = self._validation_summary(segment_ids)
            confidence_band = self._confidence_band(segment_ids, validation_metrics)
            diagnosis_summary, diagnosis_labels = self._diagnosis_summary(rep_segment_id, segment_ids)
            soundscape_state = self._soundscape_state_mode(segment_ids)

            metrics = {
                "people__total_people__mean": _mean_numeric(model_subset, "people__total_people__mean"),
                "green_view__greenviewindex__mean": _mean_numeric(model_subset, "green_view__greenviewindex__mean"),
                "visual_semantic__road__mean": _mean_numeric(model_subset, "visual_semantic__road__mean"),
                "visual_semantic__sidewalk__mean": _mean_numeric(model_subset, "visual_semantic__sidewalk__mean"),
                "emotion__beautiful__mean": _mean_numeric(model_subset, "emotion__beautiful__mean"),
                "emotion__boring__mean": _mean_numeric(model_subset, "emotion__boring__mean"),
                "emotion__depressing__mean": _mean_numeric(model_subset, "emotion__depressing__mean"),
                "audio_events__group_ratio_human": _mean_numeric(model_subset, "audio_events__group_ratio_human"),
                "audio_events__group_ratio_traffic": _mean_numeric(model_subset, "audio_events__group_ratio_traffic"),
                "audio_events__group_ratio_mechanical": _mean_numeric(model_subset, "audio_events__group_ratio_mechanical"),
                "audio_events__group_ratio_nature": _mean_numeric(model_subset, "audio_events__group_ratio_nature"),
                "audio_signal__loudness_proxy_db": _mean_numeric(model_subset, "audio_signal__loudness_proxy_db"),
                "hardscape_index": float(self._hardscape_series(model_subset.reset_index(drop=True)).mean()),
                "vehicle_index": float(self._vehicle_series(model_subset.reset_index(drop=True)).mean()),
            }

            visual_tags, visual_summary, visual_evidence = self._visual_tags_and_summary(metrics, diagnosis_labels)
            sound_tags, sound_summary, sound_evidence = self._sound_tags_and_summary(metrics, validation_metrics, soundscape_state)
            consistency_flag = mode_or_first(ranking_subset["multimodal_consistency_flag"].tolist(), default="mixed")
            fusion_tags, fusion_summary = self._fusion_tags_and_summary(visual_tags, sound_tags, consistency_flag)
            intervention_payload = self._intervention_payload(segment_ids, rep_segment_id)

            row = {
                "episode_id": str(episode["episode_id"]),
                "start_time_sec": float(episode["start_time_sec"]),
                "end_time_sec": float(episode["end_time_sec"]),
                "duration_sec": float(episode["duration_sec"]),
                "segment_ids": [int(x) for x in segment_ids],
                "n_segments": int(episode["n_segments"]),
                "representative_segment_id": rep_segment_id,
                "hero_frame_index": episode.get("hero_frame_index"),
                "hero_frame_path": str(episode.get("hero_frame_path", "")),
                "representative_frame_indices": episode.get("representative_frame_indices", []),
                "representative_frame_paths": episode.get("representative_frame_paths", []),
                "priority_score": float(pd.to_numeric(ranking_subset["priority_score"], errors="coerce").max()),
                "priority_rank": int(pd.to_numeric(ranking_subset["priority_rank"], errors="coerce").min()),
                "priority_level": mode_or_first(ranking_subset["priority_level"].tolist(), default="medium"),
                "street_type": mode_or_first(ranking_subset["street_type"].tolist(), default=""),
                "visual_problem_tags": visual_tags,
                "visual_problem_summary": visual_summary,
                "visual_evidence_features": visual_evidence,
                "visual_evidence_strength": _strength_from_count(len(visual_tags), confidence_band),
                "soundscape_problem_tags": sound_tags,
                "soundscape_problem_summary": sound_summary,
                "soundscape_evidence_features": sound_evidence,
                "soundscape_evidence_strength": _strength_from_count(len(sound_tags), confidence_band),
                "fusion_problem_tags": fusion_tags,
                "fusion_problem_summary": fusion_summary,
                "fusion_evidence_strength": "high" if consistency_flag == "consistent" and fusion_tags else ("medium" if fusion_tags else "low"),
                "cross_modal_consistency_flag": consistency_flag,
                "validation_label_summary": validation_text,
                "agent_profile_summary": self._profile_summary(rep_segment_id, segment_ids),
                "agent_diagnosis_summary": diagnosis_summary,
                "diagnosis_confidence_band": confidence_band,
                "intervention_theme": intervention_payload["intervention_theme"],
                "intervention_actions": intervention_payload["intervention_actions"],
                "must_keep_elements": intervention_payload["must_keep_elements"],
                "must_avoid_elements": intervention_payload["must_avoid_elements"],
                "design_target_focus": coalesce_texts(
                    [self.design_plan_map.get(int(sid), {}).get("confirmatory_target_focus", []) for sid in segment_ids],
                    limit=6,
                ),
                "proof_claim_snapshot": self._proof_snapshot(segment_ids),
                "soundscape_state_snapshot": soundscape_state,
                "structured_prompt_input_json": {},
                "edit_prompt": "",
                "negative_prompt": "",
                "short_caption": "",
                "prompt_mode": "",
            }
            rows.append(row)
            jsonl_rows.append(dict(row))

        df = pd.DataFrame(rows)
        logger.info("deliverable evidence built | episodes=%s", len(df))
        return df, jsonl_rows
